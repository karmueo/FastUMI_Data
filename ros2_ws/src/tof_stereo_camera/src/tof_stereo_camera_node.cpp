/**
 * @file tof_stereo_camera_node.cpp
 * @brief 打包 stereo_camera SDK 的 ROS 2 图像与 IMU 发布节点。
 * @author 待确认
 * @date 创建：待确认
 * @date 修改：2026-08-13
 */
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/imu.hpp"

#include "stereo_camera/stereo_camera.h"
#include "tof_stereo_camera/frame_utils.hpp"

namespace tof_stereo_camera {
namespace {

/**
 * @brief 获取当前稳态时钟时间。
 * @return `std::chrono::steady_clock` 的纳秒计数，用作 SDK 时间映射锚点。
 */
std::int64_t SteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

/**
 * @class StereoCameraNode
 * @brief 管理 SDK 相机流并发布既有 ROS 图像和 IMU 话题。
 * @details 节点独占相机句柄和采集线程；SDK 帧 payload 只在下一次解析前有效。
 */
class StereoCameraNode final : public rclcpp::Node {
 public:
  /**
   * @brief 声明运行参数、打开相机并启动采集线程。
   * @throws std::runtime_error 相机打开、协商或启动失败时抛出。
   */
  StereoCameraNode()
      : Node("tof_stereo_camera_node"),
        timestamp_mapper_(SteadyNowNs(), this->now().nanoseconds()) {
    const bool enable_rgb = this->declare_parameter<bool>("enable_rgb", true);
    const bool enable_itof_depth =
        this->declare_parameter<bool>("enable_itof_depth", true);
    const bool enable_itof_gray =
        this->declare_parameter<bool>("enable_itof_gray", true);
    const bool enable_imu = this->declare_parameter<bool>("enable_imu", true);
    const int width = this->declare_parameter<int>("width", 1920);
    const int height = this->declare_parameter<int>("height", 2362);
    const std::string pixel_format =
        this->declare_parameter<std::string>("pixel_format", "YUYV");
    const std::string device_path =
        this->declare_parameter<std::string>("device_path", "");
    // 是否启用 SDK 面向真机排障的内部文件日志。
    const bool enable_sdk_log =
        this->declare_parameter<bool>("enable_sdk_log", false);
    // SDK 日志文件路径；空值交由 SDK 选择默认路径。
    const std::string sdk_log_path =
        this->declare_parameter<std::string>("sdk_log_path", "");
    rgb_frame_id_ = this->declare_parameter<std::string>(
        "rgb_frame_id", "tof_stereo_camera_rgb_optical_frame");
    itof_frame_id_ = this->declare_parameter<std::string>(
        "itof_frame_id", "tof_stereo_camera_itof_optical_frame");
    imu_frame_id_ = this->declare_parameter<std::string>(
        "imu_frame_id", "tof_stereo_camera_imu_frame");

    if (width <= 0 || height <= 0 ||
        (pixel_format != "YUYV" && pixel_format != "NV12")) {
      throw std::invalid_argument(
          "width and height must be positive; pixel_format must be YUYV or NV12");
    }

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
    if (stereo_camera_set_format(camera_, width, height, pixel_format.c_str()) != 0) {
      stereo_camera_close(camera_);
      camera_ = nullptr;
      throw std::runtime_error("failed to negotiate the requested camera format");
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
                "stereo camera stream started: device=%s, format=%s %dx%d",
                device_path.empty() ? "auto" : device_path.c_str(),
                actual_format, actual_width, actual_height);
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

/**
 * @brief 按五秒间隔输出告警，避免持续错误刷屏。
 * @param[in] message 待输出的告警文本。
 */
  void WarnThrottled(const std::string &message) {
    const auto now = std::chrono::steady_clock::now();
    if (now - last_warning_ >= std::chrono::seconds(5)) {
      RCLCPP_WARN(this->get_logger(), "%s", message.c_str());
      last_warning_ = now;
    }
  }

/**
 * @brief 取得 SDK 帧时间戳对应的 ROS 时间。
 * @param[in] frame SDK 输出的帧。
 * @return 映射后的 ROS 时间；无效时间戳回退到当前节点时间。
 */
  rclcpp::Time FrameTime(const stereo_camera_frame_t &frame) const {
    return timestamp_mapper_.Map(frame.frame_timestamp, this->now());
  }

/**
 * @brief 转换并发布一帧 RGB 图像。
 * @param[in] frame SDK 输出的 RGB 帧。
 */
  void PublishRgb(const stereo_camera_frame_t &frame) {
    sensor_msgs::msg::Image message;
    std::string error;
    if (!ConvertRgbFrame(frame, &message, &error)) {
      WarnThrottled("dropping RGB frame: " + error);
      return;
    }
    message.header.stamp = FrameTime(frame);
    message.header.frame_id = rgb_frame_id_;
    rgb_publisher_->publish(std::move(message));
  }

/**
 * @brief 发布一个经时间匹配校验的 iTOF 深度或灰度帧。
 * @param[in] frame SDK 输出的 iTOF 帧。
 * @param[in] depth 为 true 时发布深度图，为 false 时发布灰度图。
 */
  void PublishItof(const stereo_camera_frame_t &frame, bool depth) {
    if (!IsPublishableTofMatchState(frame.match_state)) {
      WarnThrottled("dropping iTOF frame with an invalid match state");
      return;
    }
    sensor_msgs::msg::Image message;
    std::string error;
    if (!CopyMono16Frame(frame, depth ? "16UC1" : "mono16", &message, &error)) {
      WarnThrottled("dropping iTOF frame: " + error);
      return;
    }
    message.header.stamp = FrameTime(frame);
    message.header.frame_id = itof_frame_id_;
    if (depth) {
      itof_depth_publisher_->publish(std::move(message));
    } else {
      itof_gray_publisher_->publish(std::move(message));
    }
  }

  /**
   * @brief 发布批量 IMU 样本，并仅报告非阻断性序列诊断。
   * @param[in] frame SDK 输出的 IMU 帧。
   */
  void PublishImu(const stereo_camera_frame_t &frame) {
    std::vector<stereo_camera_imu_data_t> samples;
    std::string error;
    if (!DecodeImuFrame(frame, &samples, &error)) {
      WarnThrottled("dropping IMU frame: " + error);
      return;
    }
    if (!error.empty()) {
      WarnThrottled("IMU frame metadata: " + error);
    }

    const rclcpp::Time frame_time = FrameTime(frame);
    for (const stereo_camera_imu_data_t &sample : samples) {
      sensor_msgs::msg::Imu message;
      message.header.stamp =
          timestamp_mapper_.MapImuSample(sample.timestamp, frame_time);
      message.header.frame_id = imu_frame_id_;
      message.orientation_covariance[0] = -1.0;
      message.angular_velocity.x = sample.gx;
      message.angular_velocity.y = sample.gy;
      message.angular_velocity.z = sample.gz;
      message.linear_acceleration.x = sample.ax;
      message.linear_acceleration.y = sample.ay;
      message.linear_acceleration.z = sample.az;
      imu_publisher_->publish(std::move(message));
    }
  }

/**
 * @brief 在专用线程中持续读取 SDK 帧并按传感器类型发布。
 * @note `stereo_camera_parse_frame()` 可能阻塞；`StopCamera()` 通过取消读取唤醒它。
 */
  void CaptureLoop() {
    while (running_.load()) {
      stereo_camera_frame_t *frame = stereo_camera_parse_frame(camera_);
      if (frame == nullptr) {
        continue;
      }
      if (frame->sourcetype == STEREO_SENSOR_RGB && rgb_publisher_) {
        PublishRgb(*frame);
      } else if (frame->sourcetype == STEREO_SENSOR_ITOF &&
                 frame->stream_id == kItofDepthStreamId &&
                 itof_depth_publisher_) {
        PublishItof(*frame, true);
      } else if (frame->sourcetype == STEREO_SENSOR_ITOF &&
                 frame->stream_id == kItofGrayStreamId &&
                 itof_gray_publisher_) {
        PublishItof(*frame, false);
      } else if (frame->sourcetype == STEREO_SENSOR_IMU && imu_publisher_) {
        PublishImu(*frame);
      }
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
  /// SDK 单调时钟到 ROS 时间的固定锚点映射器。
  TimestampMapper timestamp_mapper_;
  /// 上一次发出节流告警的稳态时间点。
  std::chrono::steady_clock::time_point last_warning_{};
  /// 写入 RGB 图像消息头的坐标系名称。
  std::string rgb_frame_id_;
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

}  // namespace tof_stereo_camera

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
    RCLCPP_ERROR(rclcpp::get_logger("tof_stereo_camera_node"), "%s", error.what());
    result = 1;
  }
  rclcpp::shutdown();
  return result;
}
