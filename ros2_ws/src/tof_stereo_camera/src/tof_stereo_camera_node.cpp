/**
 * @file tof_stereo_camera_node.cpp
 * @brief 打包 stereo_camera SDK 的 ROS 2 图像与 IMU 发布节点。
 * @author 待确认
 * @date 创建：待确认
 * @date 修改：2026-08-20
 */
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"

#include "stereo_camera/stereo_camera.h"
#include "tof_stereo_camera/frame_utils.hpp"
#include "tof_stereo_camera/stream_profile.hpp"
#include "tof_stereo_camera/xu_config.hpp"

namespace tof_stereo_camera {
namespace {

/** @brief 获取当前稳态时钟纳秒计数。 */
std::int64_t SteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
/** @brief 捕获系统时钟夹在两个稳态样本之间的中点配对锚点。 */
void CaptureClockAnchor(std::int64_t *steady_ns, std::int64_t *system_ns) {
  const std::int64_t before_ns = SteadyNowNs();
  const std::int64_t unix_ns =
      rclcpp::Clock(RCL_SYSTEM_TIME).now().nanoseconds();
  const std::int64_t after_ns = SteadyNowNs();
  *steady_ns = before_ns + (after_ns - before_ns) / 2;
  *system_ns = unix_ns;
}

/**
 * @brief 创建仅允许节点启动时赋值的 ROS 参数描述符。
 * @param[in] description 参数用途说明。
 * @return 设置了只读标志的参数描述符。
 */
rcl_interfaces::msg::ParameterDescriptor
ReadOnlyParameter(const std::string &description) {
  rcl_interfaces::msg::ParameterDescriptor descriptor; ///< 待返回的只读描述符。
  descriptor.description = description;
  descriptor.read_only = true;
  return descriptor;
}

} // namespace

/**
 * @class StereoCameraNode
 * @brief 管理 SDK 相机流并发布既有 ROS 图像和 IMU 话题。
 * @details 节点独占相机句柄和采集线程；SDK 帧 payload 只在下一次解析前有效。
 */
class StereoCameraNode final : public rclcpp::Node {
public:
  /** @brief 声明运行参数、建立时间同步器并启动采集线程。 */
  StereoCameraNode() : Node("tof_stereo_camera_node") {
    const bool enable_rgb = this->declare_parameter<bool>(
        "enable_rgb", true,
        ReadOnlyParameter("同时启用设备端 RGB 流和 ROS RGB topic"));
    const bool enable_itof_depth = this->declare_parameter<bool>(
        "enable_itof_depth", true,
        ReadOnlyParameter("同时启用设备端 iTOF 深度流和 ROS 深度 topic"));
    const bool enable_itof_gray = this->declare_parameter<bool>(
        "enable_itof_gray", true,
        ReadOnlyParameter("同时启用设备端 iTOF 灰度流和 ROS 灰度 topic"));
    const bool enable_imu = this->declare_parameter<bool>("enable_imu", true);
    const int imu_accel_hz = this->declare_parameter<int>(
        "imu_accel_hz", 100,
        ReadOnlyParameter("设备加速度计采样频率，单位 Hz"));
    const int imu_gyro_hz = this->declare_parameter<int>(
        "imu_gyro_hz", 100, ReadOnlyParameter("设备陀螺仪采样频率，单位 Hz"));
    const std::string stream_profile_name =
        this->declare_parameter<std::string>(
            "stream_profile", "main",
            ReadOnlyParameter("码流档位，可选 main 或 sub"));
    const std::string pixel_format =
        this->declare_parameter<std::string>("pixel_format", "YUYV");
    const std::string rgb_output_encoding =
        this->declare_parameter<std::string>(
            "rgb_output_encoding", "yuv422_yuy2",
            ReadOnlyParameter("RGB 图像输出编码，可选 yuv422_yuy2 或 bgr8"));
    const std::string device_path =
        this->declare_parameter<std::string>("device_path", "");
    // 是否启用 SDK 面向真机排障的内部文件日志。
    const bool enable_sdk_log =
        this->declare_parameter<bool>("enable_sdk_log", false);
    // SDK 日志文件路径；空值交由 SDK 选择默认路径。
    const int timestamp_calibration_frames =
        this->declare_parameter<int>("timestamp_calibration_frames", 30);
    const int timestamp_window_frames =
        this->declare_parameter<int>("timestamp_window_frames", 120);
    const double timestamp_max_slew_ppm =
        this->declare_parameter<double>("timestamp_max_slew_ppm", 200.0);
    const std::string sdk_log_path =
        this->declare_parameter<std::string>("sdk_log_path", "");
    rgb_frame_id_ = this->declare_parameter<std::string>(
        "rgb_frame_id", "tof_stereo_camera_rgb_optical_frame");
    itof_frame_id_ = this->declare_parameter<std::string>(
        "itof_frame_id", "tof_stereo_camera_itof_optical_frame");
    imu_frame_id_ = this->declare_parameter<std::string>(
        "imu_frame_id", "tof_stereo_camera_imu_frame");
    StreamProfile stream_profile; ///< 码流档位对应的固定复合帧格式。
    std::string stream_profile_error; ///< 接收非法码流档位诊断。
    if (!ResolveStreamProfile(stream_profile_name, &stream_profile,
                              &stream_profile_error)) {
      throw std::invalid_argument(stream_profile_error);
    }
    if (pixel_format != "YUYV" && pixel_format != "NV12") {
      throw std::invalid_argument("pixel_format must be YUYV or NV12");
    }
    std::string rgb_output_encoding_error; ///< 接收 RGB 输出编码参数诊断。
    if (!ParseRgbOutputEncoding(rgb_output_encoding, &rgb_output_encoding_,
                                &rgb_output_encoding_error)) {
      throw std::invalid_argument(rgb_output_encoding_error);
    }
    if (enable_rgb &&
        rgb_output_encoding_ == RgbOutputEncoding::kYuv422Yuy2 &&
        pixel_format != "YUYV") {
      throw std::invalid_argument(
          "rgb_output_encoding=yuv422_yuy2 requires pixel_format=YUYV");
    }
    if (timestamp_calibration_frames < 2 ||
        timestamp_window_frames < timestamp_calibration_frames ||
        !std::isfinite(timestamp_max_slew_ppm) ||
        timestamp_max_slew_ppm <= 0.0 ||
        timestamp_max_slew_ppm >= 1'000'000.0) {
      throw std::invalid_argument(
          "invalid timestamp synchronization parameters");
    }
    XuConfiguration xu_configuration; ///< 经过校验的 SDK XU 启动配置。
    std::string xu_error; ///< 接收 XU 参数校验或设备命令失败原因。
    if (!BuildXuConfiguration(enable_rgb, enable_itof_depth, enable_itof_gray,
                              enable_imu, imu_accel_hz, imu_gyro_hz,
                              &xu_configuration, &xu_error)) {
      throw std::invalid_argument(xu_error);
    }
    timestamp_calibration_frames_ = timestamp_calibration_frames;
    timestamp_window_frames_ = timestamp_window_frames;
    timestamp_max_slew_ppm_ = timestamp_max_slew_ppm;
    std::int64_t anchor_steady_ns = 0;
    std::int64_t anchor_system_ns = 0;
    CaptureClockAnchor(&anchor_steady_ns, &anchor_system_ns);
    const auto make_mapper = [&]() {
      return std::make_unique<TimestampMapper>(
          anchor_steady_ns, anchor_system_ns,
          static_cast<std::size_t>(timestamp_calibration_frames),
          static_cast<std::size_t>(timestamp_window_frames),
          timestamp_max_slew_ppm);
    };
    rgb_timestamp_mapper_ = make_mapper();
    itof_depth_timestamp_mapper_ = make_mapper();
    itof_gray_timestamp_mapper_ = make_mapper();
    imu_timestamp_mapper_ = make_mapper();
    const auto sensor_qos = rclcpp::SensorDataQoS();
    if (enable_rgb) {
      rgb_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
          "rgb/image_raw", sensor_qos);
    }
    if (enable_itof_depth) {
      itof_depth_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
          "itof/depth/image_raw", sensor_qos);
    }
    if (enable_itof_gray) {
      itof_gray_publisher_ = this->create_publisher<sensor_msgs::msg::Image>(
          "itof/gray/image_raw", sensor_qos);
    }
    if (enable_imu) {
      imu_publisher_ = this->create_publisher<sensor_msgs::msg::Imu>(
          "imu/data_raw", sensor_qos);
    }

    const char *requested_device =
        device_path.empty() ? nullptr : device_path.c_str();
    camera_ = stereo_camera_open(requested_device);
    if (camera_ == nullptr) {
      throw std::runtime_error("unable to open a stereo camera");
    }
    if (stereo_camera_set_format(camera_, stream_profile.composite_width,
                                 stream_profile.composite_height,
                                 pixel_format.c_str()) != 0) {
      stereo_camera_close(camera_);
      camera_ = nullptr;
      throw std::runtime_error(
          "failed to negotiate the requested camera format");
    }

    int actual_width = 0;
    int actual_height = 0;
    char actual_format[5] = {};
    if (stereo_camera_get_format(camera_, &actual_width, &actual_height,
                                 actual_format) != 0) {
      stereo_camera_close(camera_);
      camera_ = nullptr;
      throw std::runtime_error("failed to read the negotiated camera format");
    }

    // 在采集启动前配置 SDK 的全局内部日志开关。
    const char *sdk_log_path_value =
        sdk_log_path.empty() ? nullptr : sdk_log_path.c_str();
    stereo_camera_set_log(enable_sdk_log ? 1 : 0, sdk_log_path_value);
    sdk_log_configured_ = true;

    if (!ApplyXuConfiguration(camera_, xu_configuration, &xu_error)) {
      stereo_camera_close(camera_);
      camera_ = nullptr;
      stereo_camera_set_log(0, nullptr);
      sdk_log_configured_ = false;
      throw std::runtime_error(xu_error);
    }
    RCLCPP_INFO(this->get_logger(),
                "SDK XU configuration applied: accel_hz=%u gyro_hz=%u "
                "stream_mask=0x%08x ack=0",
                xu_configuration.accel_hz, xu_configuration.gyro_hz,
                xu_configuration.stream_mask);

    if (stereo_camera_start_stream(camera_) != 0) {
      stereo_camera_close(camera_);
      camera_ = nullptr;
      stereo_camera_set_log(0, nullptr);
      sdk_log_configured_ = false;
      throw std::runtime_error("failed to start the camera stream");
    }
    stream_started_ = true;
    running_.store(true);
    capture_thread_ = std::thread(&StereoCameraNode::CaptureLoop, this);
    RCLCPP_INFO(this->get_logger(),
                "stereo camera stream started: device=%s, profile=%s "
                "rgb=%dx%d, format=%s %dx%d, rgb_output_encoding=%s",
                device_path.empty() ? "auto" : device_path.c_str(),
                stream_profile_name.c_str(), stream_profile.rgb_width,
                stream_profile.rgb_height, actual_format, actual_width,
                actual_height, rgb_output_encoding.c_str());
  }

  /** @brief 析构时停止采集并释放相机资源。 */
  ~StereoCameraNode() override { StopCamera(); }

private:
  /**
   * @brief 停止阻塞读取、关闭 SDK 日志并释放相机资源。
   */
  void StopCamera() {
    running_.store(false);
    if (camera_ != nullptr) {
      stereo_camera_cancel_read(camera_);
    }
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    if (sdk_log_configured_) {
      stereo_camera_set_log(0, nullptr);
      sdk_log_configured_ = false;
    }
    if (camera_ != nullptr) {
      if (stream_started_) {
        stereo_camera_stop_stream(camera_);
      }
      stereo_camera_close(camera_);
      camera_ = nullptr;
      stream_started_ = false;
    }
  }

  /** @brief 按类别键独立输出节流告警。 */
  void WarnThrottled(const std::string &key, const std::string &message) {
    const auto now = std::chrono::steady_clock::now();
    const auto found = last_warnings_.find(key);
    if (found == last_warnings_.end() ||
        now - found->second >= std::chrono::seconds(5)) {
      RCLCPP_WARN(this->get_logger(), "%s", message.c_str());
      last_warnings_[key] = now;
    }
  }

  /** @brief 将同步结果构造为系统时钟 ROS 时间。 */
  rclcpp::Time FrameTime(const FrameTimestampResult &time) const {
    return rclcpp::Time(time.system_ns, RCL_SYSTEM_TIME);
  }
  /** @brief 转换并发布一帧 RGB 图像。 */
  void PublishRgb(const stereo_camera_frame_t &frame,
                  const FrameTimestampResult &time) {
    sensor_msgs::msg::Image message;
    std::string error;
    if (!ConvertRgbFrame(frame, rgb_output_encoding_, &message, &error)) {
      WarnThrottled("rgb_convert", "dropping RGB frame: " + error);
      return;
    }
    message.header.stamp = FrameTime(time);
    message.header.frame_id = rgb_frame_id_;
    rgb_publisher_->publish(std::move(message));
  }
  /** @brief 发布一个经时间匹配校验的 iTOF 深度或灰度帧。 */
  void PublishItof(const stereo_camera_frame_t &frame, bool depth,
                   const FrameTimestampResult &time) {
    if (!IsPublishableTofMatchState(frame.match_state)) {
      WarnThrottled("itof_publish_match",
                    "dropping iTOF frame with an invalid match state");
      return;
    }
    sensor_msgs::msg::Image message;
    std::string error;
    if (!CopyMono16Frame(frame, depth ? "16UC1" : "mono16", &message, &error)) {
      WarnThrottled(depth ? "itof_convert_depth" : "itof_convert_gray",
                    "dropping iTOF frame: " + error);
      return;
    }
    message.header.stamp = FrameTime(time);
    message.header.frame_id = itof_frame_id_;
    if (depth)
      itof_depth_publisher_->publish(std::move(message));
    else
      itof_gray_publisher_->publish(std::move(message));
  }
  /** @brief 保留 SDK 解码顺序并全量发布一批 IMU 样本。 */
  void PublishImu(const std::vector<stereo_camera_imu_data_t> &samples,
                  const FrameTimestampResult &time) {
    for (const stereo_camera_imu_data_t &sample : samples) {
      const ImuTimestampResult sample_time =
          imu_timestamp_mapper_->MapImuSample(sample.timestamp, time);
      if (sample_time.future_clamped)
        WarnThrottled("imu_sample_future",
                      "clamping future IMU timestamp to outer frame timestamp");
      sensor_msgs::msg::Imu message;
      message.header.stamp =
          rclcpp::Time(sample_time.system_ns, RCL_SYSTEM_TIME);
      message.header.frame_id = imu_frame_id_;
      message.orientation_covariance[0] = -1.0;
      message.angular_velocity.x =
          DegreesPerSecondToRadiansPerSecond(sample.gx);
      message.angular_velocity.y =
          DegreesPerSecondToRadiansPerSecond(sample.gy);
      message.angular_velocity.z =
          DegreesPerSecondToRadiansPerSecond(sample.gz);
      message.linear_acceleration.x = sample.ax;
      message.linear_acceleration.y = sample.ay;
      message.linear_acceleration.z = sample.az;
      imu_publisher_->publish(std::move(message));
    }
  }
  /** @brief 在专用线程中读取 SDK 帧，并以唯一复合帧推进同步器。 */
  void CaptureLoop() {
    while (running_.load()) {
      stereo_camera_frame_t *frame = stereo_camera_parse_frame(camera_);
      const std::int64_t receive_steady_ns = SteadyNowNs();
      if (frame == nullptr)
        continue;
      const auto log_timestamp = [&](TimestampMapper *mapper,
                                     const FrameTimestampResult &time,
                                     const char *label) {
        if (time.status == FrameTimestampStatus::kReset)
          WarnThrottled(std::string(label) + ":reset",
                        std::string("SDK timestamp discontinuity; stream=") +
                            label + " restarting timestamp calibration");
        if (time.status == FrameTimestampStatus::kDropped)
          WarnThrottled(std::string(label) + ":drop",
                        std::string("dropping frame: stream=") + label +
                            " monotonic timestamp conflicts with receive-time "
                            "upper bound");
        if (time.receive_clamped)
          WarnThrottled(
              std::string(label) + ":clamp",
              std::string("clamping future outer-frame timestamp: stream=") +
                  label + " to receive time");
        if (time.newly_locked)
          RCLCPP_INFO(this->get_logger(),
                      "timestamp synchronization locked: stream=%s "
                      "steady_offset_ns=%ld "
                      "system_offset_ns=%ld calibration_frames=%d "
                      "window_frames=%d max_slew_ppm=%.3f",
                      label, mapper->detected_epoch_offset_ns(),
                      mapper->detected_system_epoch_offset_ns(),
                      timestamp_calibration_frames_, timestamp_window_frames_,
                      timestamp_max_slew_ppm_);
      };
      TimestampMapper *mapper = nullptr;
      const char *stream_label = nullptr;
      if (frame->sourcetype == STEREO_SENSOR_RGB && frame->stream_id == 0 &&
          rgb_publisher_) {
        mapper = rgb_timestamp_mapper_.get();
        stream_label = "rgb";
      } else if (frame->sourcetype == STEREO_SENSOR_ITOF &&
                 frame->stream_id == kItofDepthStreamId &&
                 itof_depth_publisher_) {
        if (!IsPublishableTofMatchState(frame->match_state)) {
          WarnThrottled(
              "itof_depth_match",
              "dropping iTOF depth frame with an invalid match state");
          continue;
        }
        mapper = itof_depth_timestamp_mapper_.get();
        stream_label = "itof_depth";
      } else if (frame->sourcetype == STEREO_SENSOR_ITOF &&
                 frame->stream_id == kItofGrayStreamId &&
                 itof_gray_publisher_) {
        if (!IsPublishableTofMatchState(frame->match_state)) {
          WarnThrottled("itof_gray_match",
                        "dropping iTOF gray frame with an invalid match state");
          continue;
        }
        mapper = itof_gray_timestamp_mapper_.get();
        stream_label = "itof_gray";
      }
      if (frame->sourcetype == STEREO_SENSOR_IMU && imu_publisher_) {
        std::vector<stereo_camera_imu_data_t> samples;
        std::string error;
        if (!DecodeImuFrame(*frame, &samples, &error)) {
          WarnThrottled("imu_decode", "dropping IMU frame: " + error);
          continue;
        }
        if (!error.empty())
          WarnThrottled("imu_metadata", "IMU frame metadata: " + error);
        if (samples.empty())
          continue;
        std::int64_t observation_us = 0;
        for (const auto &sample : samples)
          if (sample.timestamp > observation_us &&
              sample.timestamp <=
                  std::numeric_limits<std::int64_t>::max() / 1000)
            observation_us = sample.timestamp;
        const FrameTimestampResult time = imu_timestamp_mapper_->ObserveFrame(
            static_cast<std::uint64_t>(observation_us), receive_steady_ns);
        log_timestamp(imu_timestamp_mapper_.get(), time, "imu");
        if (time.status == FrameTimestampStatus::kReady ||
            time.status == FrameTimestampStatus::kInvalidHostFallback)
          PublishImu(samples, time);
        continue;
      }
      if (mapper == nullptr)
        continue;
      const FrameTimestampResult time =
          mapper->ObserveFrame(frame->frame_timestamp, receive_steady_ns);
      if (time.status == FrameTimestampStatus::kInvalidHostFallback)
        WarnThrottled(std::string(stream_label) + ":invalid",
                      "invalid or overflowing SDK frame timestamp; using host "
                      "receive time");
      log_timestamp(mapper, time, stream_label);
      const bool publish =
          time.status == FrameTimestampStatus::kReady ||
          time.status == FrameTimestampStatus::kInvalidHostFallback;
      if (!publish)
        continue;
      if (frame->sourcetype == STEREO_SENSOR_RGB && rgb_publisher_)
        PublishRgb(*frame, time);
      else if (frame->sourcetype == STEREO_SENSOR_ITOF &&
               frame->stream_id == kItofDepthStreamId && itof_depth_publisher_)
        PublishItof(*frame, true, time);
      else if (frame->sourcetype == STEREO_SENSOR_ITOF &&
               frame->stream_id == kItofGrayStreamId && itof_gray_publisher_)
        PublishItof(*frame, false, time);
    }
  }

  /// SDK 打开的相机句柄，由节点独占并在关闭时释放。
  stereo_camera_t *camera_ = nullptr;
  /// 相机流是否已由当前节点成功启动。
  bool stream_started_ = false;
  // SDK 日志开关已配置，关闭节点时需要恢复为关闭状态。
  bool sdk_log_configured_ = false;
  /// 控制采集线程继续读取的原子标志。
  std::atomic<bool> running_{false};
  /// 执行阻塞 SDK 读取的采集线程。
  std::thread capture_thread_;
  /// SDK 微秒设备时钟到主机时钟的状态同步器。
  std::unique_ptr<TimestampMapper> rgb_timestamp_mapper_;
  std::unique_ptr<TimestampMapper> itof_depth_timestamp_mapper_;
  std::unique_ptr<TimestampMapper> itof_gray_timestamp_mapper_;
  std::unique_ptr<TimestampMapper> imu_timestamp_mapper_;
  /// 日志中记录的同步参数。
  int timestamp_calibration_frames_ = 30;
  int timestamp_window_frames_ = 120;
  double timestamp_max_slew_ppm_ = 200.0;
  /// 按稳定类别键保存上次告警时刻，避免不同流互相抑制。
  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
      last_warnings_;
  /// 写入 RGB 图像消息头的坐标系名称。
  std::string rgb_frame_id_;
  /// RGB 图像发布时采用的编码策略，仅在节点启动时设置。
  RgbOutputEncoding rgb_output_encoding_ = RgbOutputEncoding::kYuv422Yuy2;
  /// 写入 iTOF 图像消息头的坐标系名称。
  std::string itof_frame_id_;
  /// 写入 IMU 消息头的坐标系名称。
  std::string imu_frame_id_;
  /// 发布 RGB 图像的 ROS 发布器；为空时该话题关闭。
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_publisher_;
  /// 发布 iTOF 深度图的 ROS 发布器；为空时该话题关闭。
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr itof_depth_publisher_;
  /// 发布 iTOF 灰度图的 ROS 发布器；为空时该话题关闭。
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr itof_gray_publisher_;
  /// 发布原始 IMU 数据的 ROS 发布器；为空时该话题关闭。
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_publisher_;
};

} // namespace tof_stereo_camera

/**
 * @brief 初始化 ROS、运行相机节点并在退出时关闭 ROS。
 * @param[in] argc 命令行参数数量。
 * @param[in] argv 命令行参数数组。
 * @return 节点正常退出时返回 0；初始化或运行失败时返回 1。
 */
int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  int result = 0;
  try {
    rclcpp::spin(std::make_shared<tof_stereo_camera::StereoCameraNode>());
  } catch (const std::exception &error) {
    RCLCPP_ERROR(rclcpp::get_logger("tof_stereo_camera_node"), "%s",
                 error.what());
    result = 1;
  }
  rclcpp::shutdown();
  return result;
}
