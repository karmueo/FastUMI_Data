/**
 * @file usb_camera_node.cpp
 * @brief 通过共享 V4L2 采集核心发布 USB 相机原始图像或原生 JPEG。
 */

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "fastumi_usb_camera/configuration.hpp"
#include "fastumi_usb_camera/image_message.hpp"
#include "fastumi_usb_camera/jpeg_frame.hpp"
#include "fastumi_usb_camera/latest_frame_buffer.hpp"
#include "fastumi_usb_camera/v4l2_camera.hpp"

namespace fastumi_usb_camera
{

using SteadyClock = std::chrono::steady_clock;

/** @brief 保存一帧原生 JPEG 及其主机接收时间。 */
struct CapturedJpeg
{
  builtin_interfaces::msg::Time stamp;  ///< ROS 时钟下的帧接收时间。
  std::vector<uint8_t> data;  ///< 从 V4L2 缓冲区复制出的完整 JPEG 字节。
};

/** @brief 发布 V4L2 USB 相机 raw 或原生 JPEG 图像的 ROS 2 节点。 */
class UsbCameraNode : public rclcpp::Node
{
public:
  /** @brief 声明只读配置并创建与输出模式对应的发布器。 */
  UsbCameraNode()
  : Node("usb_camera_node")
  {
    config_.width = declare_parameter<int64_t>("width", config_.width);
    config_.height = declare_parameter<int64_t>("height", config_.height);
    config_.fps = declare_parameter<int64_t>("fps", config_.fps);
    config_.frame_timeout_seconds = declare_parameter<double>(
      "frame_timeout_seconds", config_.frame_timeout_seconds);
    config_.video_device = declare_parameter<std::string>("video_device", "");
    config_.frame_id = declare_parameter<std::string>("frame_id", config_.frame_id);
    publish_compressed_ = declare_parameter<bool>("publish_compressed", false);
    validate_configuration(config_);

    rclcpp::QoS qos(rclcpp::KeepLast(5));  ///< 与既有接口一致的传感器 QoS。
    qos.best_effort();
    qos.durability_volatile();
    if (publish_compressed_) {
      compressed_publisher_ = create_publisher<sensor_msgs::msg::CompressedImage>(
        "image_raw/compressed", qos);
    } else {
      image_publisher_ = create_publisher<sensor_msgs::msg::Image>("image_raw", qos);
    }
    parameter_callback_ = add_on_set_parameters_callback(
      [](const std::vector<rclcpp::Parameter> &) {
        rcl_interfaces::msg::SetParametersResult result;  ///< 运行时参数拒绝结果。
        result.successful = false;
        result.reason = "USB camera parameters are read-only after startup";
        return result;
      });
    watchdog_timer_ = create_wall_timer(
      std::chrono::milliseconds(500), std::bind(&UsbCameraNode::check_camera, this));
    diagnostics_timer_ = create_wall_timer(
      std::chrono::seconds(5), std::bind(&UsbCameraNode::report_diagnostics, this));
  }

  /** @brief 停止相机并等待发布线程退出。 */
  ~UsbCameraNode() override
  {
    stop();
  }

  /** @brief 启动发布工作线程和共享 V4L2 相机。 */
  void start()
  {
    if (started_.exchange(true)) {
      throw std::logic_error("camera node has already started");
    }
    stopping_ = false;
    last_frame_steady_ns_.store(steady_now_ns());
    worker_ = std::thread(&UsbCameraNode::process_frames, this);
    try {
      camera_ = std::make_unique<V4l2Camera>(
        config_, [this](const uint8_t * data, size_t size) {capture_frame(data, size);});
    } catch (...) {
      frame_buffer_.close();
      worker_.join();
      started_ = false;
      throw;
    }
    RCLCPP_INFO(
      get_logger(), "Streaming V4L2 %s MJPEG %ldx%ld@%ld -> %s",
      camera_->device_path().c_str(), config_.width, config_.height, config_.fps,
      publish_compressed_ ? "image_raw/compressed" : "image_raw");
  }

  /** @brief 幂等停止采集，释放设备并结束发布线程。 */
  void stop() noexcept
  {
    if (started_.exchange(false)) {
      stopping_ = true;
      if (camera_) {
        camera_->stop();
        camera_.reset();
      }
      frame_buffer_.close();
      if (worker_.joinable()) {
        worker_.join();
      }
    }
  }

  /** @return 节点是否遇到需要退出的采集或发布错误。 */
  bool failed() const noexcept
  {
    return fatal_error_.load();
  }

private:
  /** @return 当前单调时钟纳秒数。 */
  static int64_t steady_now_ns()
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
      SteadyClock::now().time_since_epoch()).count();
  }

  /** @brief 在 V4L2 线程复制完整 JPEG 并覆盖旧待处理帧。 */
  void capture_frame(const uint8_t * data, size_t size) noexcept
  {
    if (stopping_ || data == nullptr || size == 0U) {
      callback_invalid_count_.fetch_add(1);
      return;
    }
    try {
      CapturedJpeg captured;  ///< 即将进入最新帧缓存的 JPEG 数据。
      captured.stamp = now();
      captured.data.assign(data, data + size);
      captured_count_.fetch_add(1);
      last_frame_steady_ns_.store(steady_now_ns());
      frame_buffer_.push(std::move(captured));
    } catch (...) {
      set_fatal_error("failed to copy a frame from the V4L2 capture thread");
    }
  }

  /** @brief 在独立线程解码或直接发布最新 JPEG 帧。 */
  void process_frames() noexcept
  {
    try {
      while (auto captured = frame_buffer_.wait_and_take()) {
        if (publish_compressed_) {
          sensor_msgs::msg::CompressedImage message = make_jpeg_message(
            std::move(captured->data), captured->stamp,
            config_.frame_id);  ///< 原生 JPEG ROS 消息。
          compressed_publisher_->publish(std::move(message));
          published_count_.fetch_add(1);
          continue;
        }

        cv::Mat bgr = decode_jpeg_frame(
          captured->data, static_cast<int>(config_.width),
          static_cast<int>(config_.height));  ///< 当前帧的 BGR 解码结果。
        if (bgr.empty()) {
          decode_failure_count_.fetch_add(1);
          continue;
        }
        sensor_msgs::msg::Image message = make_bgr_image_message(
          bgr, captured->stamp, config_.frame_id);  ///< 待发布的 raw 图像消息。
        image_publisher_->publish(std::move(message));
        published_count_.fetch_add(1);
      }
    } catch (const std::exception & error) {
      set_fatal_error(std::string("frame processing failed: ") + error.what());
    } catch (...) {
      set_fatal_error("frame processing failed with an unknown error");
    }
  }

  /** @brief 记录首个致命错误并请求主循环退出。 */
  void set_fatal_error(const std::string & message) noexcept
  {
    if (!fatal_error_.exchange(true)) {
      RCLCPP_FATAL(get_logger(), "%s", message.c_str());
    }
  }

  /** @brief 检查采集线程错误和连续无帧超时。 */
  void check_camera()
  {
    if (!started_ || stopping_) {
      return;
    }
    if (camera_ && camera_->failed()) {
      set_fatal_error("V4L2 camera capture failed: " + camera_->error_message());
      return;
    }
    const int64_t silence_ns =
      steady_now_ns() - last_frame_steady_ns_.load();  ///< 距离最后有效帧的时间。
    if (silence_ns > static_cast<int64_t>(config_.frame_timeout_seconds * 1.0e9)) {
      set_fatal_error(
        "no valid V4L2 camera frame received for " +
        std::to_string(config_.frame_timeout_seconds) + " seconds");
    }
  }

  /** @brief 周期记录采集、发布、覆盖、坏帧和解码失败数量。 */
  void report_diagnostics()
  {
    const uint64_t camera_invalid = camera_ ? camera_->invalid_count() : 0U;  ///< 驱动坏帧数。
    RCLCPP_INFO(
      get_logger(),
      "frames captured=%lu published=%lu overwritten=%lu invalid=%lu decode_failed=%lu",
      captured_count_.load(), published_count_.load(),
      frame_buffer_.overwritten_count(), camera_invalid + callback_invalid_count_.load(),
      decode_failure_count_.load());
  }

  CameraConfiguration config_;  ///< 设备选择、采集模式及消息坐标系配置。
  bool publish_compressed_{false};  ///< true 时直接发布相机原生 JPEG。
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;  ///< raw 发布器。
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr
    compressed_publisher_;  ///< 原生 JPEG 发布器。
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
    parameter_callback_;  ///< 拒绝运行时参数修改的回调句柄。
  LatestFrameBuffer<CapturedJpeg> frame_buffer_;  ///< 只保存最新待处理帧的缓存。
  std::unique_ptr<V4l2Camera> camera_;  ///< 唯一 V4L2 设备资源所有者。
  std::thread worker_;  ///< 图像解码和发布线程。
  rclcpp::TimerBase::SharedPtr watchdog_timer_;  ///< 采集异常和无帧超时检查定时器。
  rclcpp::TimerBase::SharedPtr diagnostics_timer_;  ///< 帧统计输出定时器。
  std::atomic<bool> started_{false};  ///< 节点是否已启动相机。
  std::atomic<bool> stopping_{false};  ///< 节点是否正在停止并拒绝新帧。
  std::atomic<bool> fatal_error_{false};  ///< 是否发生需要结束进程的错误。
  std::atomic<int64_t> last_frame_steady_ns_{0};  ///< 最后有效帧的单调时间。
  std::atomic<uint64_t> captured_count_{0};  ///< 已从 V4L2 接收的完整 JPEG 帧数量。
  std::atomic<uint64_t> published_count_{0};  ///< 已发布图像数量。
  std::atomic<uint64_t> callback_invalid_count_{0};  ///< 回调层拒绝的空帧数量。
  std::atomic<uint64_t> decode_failure_count_{0};  ///< JPEG 解码或尺寸校验失败数量。
};

}  // namespace fastumi_usb_camera

namespace
{

volatile std::sig_atomic_t stop_requested = 0;  ///< SIGINT/SIGTERM 请求的退出标志。

/** @brief 将进程终止信号转换为主循环可轮询的标志。 */
void request_stop(int signal_number)
{
  (void)signal_number;
  stop_requested = 1;
}

}  // namespace

/** @brief 创建 USB 相机节点并在退出前确定性释放 V4L2 设备。 */
int main(int argc, char ** argv)
{
  rclcpp::init(
    argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, request_stop);
  std::signal(SIGTERM, request_stop);
  try {
    bool failed = false;  ///< 节点退出码是否应表示运行失败。
    {
      auto node = std::make_shared<fastumi_usb_camera::UsbCameraNode>();  ///< 相机节点实例。
      node->start();
      rclcpp::executors::SingleThreadedExecutor executor;  ///< 定时器与参数回调执行器。
      executor.add_node(node);
      while (!stop_requested && !node->failed()) {
        executor.spin_once(std::chrono::milliseconds(100));
      }
      failed = node->failed();
      executor.remove_node(node);
      node->stop();
      node.reset();
    }
    rclcpp::shutdown();
    return failed ? 1 : 0;
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("usb_camera_node"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
}
