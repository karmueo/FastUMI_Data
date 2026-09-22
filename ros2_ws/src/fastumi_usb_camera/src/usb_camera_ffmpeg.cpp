/**
 * @file usb_camera_ffmpeg.cpp
 * @brief 通过共享 V4L2 核心采集 MJPEG，并用 FFmpeg image transport 发布。
 */

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <image_transport/publisher_plugin.hpp>
#include <opencv2/imgcodecs.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "fastumi_usb_camera/configuration.hpp"
#include "fastumi_usb_camera/image_message.hpp"
#include "fastumi_usb_camera/jpeg_frame.hpp"
#include "fastumi_usb_camera/latest_frame_buffer.hpp"
#include "fastumi_usb_camera/v4l2_camera.hpp"

namespace fastumi_usb_camera
{

using SteadyClock = std::chrono::steady_clock;

struct CapturedJpeg
{
  builtin_interfaces::msg::Time stamp;
  SteadyClock::time_point queued_at;
  std::vector<uint8_t> data;
};

class UsbCameraFfmpeg : public rclcpp::Node
{
public:
  explicit UsbCameraFfmpeg(
    pluginlib::ClassLoader<image_transport::PublisherPlugin> & loader)
  : Node("usb_camera_ffmpeg")
  {
    config_.width = declare_parameter<int64_t>("width", config_.width);
    config_.height = declare_parameter<int64_t>("height", config_.height);
    config_.fps = declare_parameter<int64_t>("fps", config_.fps);
    config_.frame_timeout_seconds = declare_parameter<double>(
      "frame_timeout_seconds", config_.frame_timeout_seconds);
    config_.video_device = declare_parameter<std::string>("video_device", "");
    config_.frame_id = declare_parameter<std::string>("frame_id", config_.frame_id);
    config_.topic = declare_parameter<std::string>("topic", config_.topic);
    validate_configuration(config_);

    encoder_ = loader.createUniqueInstance(
      image_transport::PublisherPlugin::getLookupName("ffmpeg"));
    auto qos = rmw_qos_profile_sensor_data;
    qos.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
    qos.depth = 1;
    qos.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
    qos.durability = RMW_QOS_POLICY_DURABILITY_VOLATILE;
    encoder_->advertise(this, config_.topic, qos);

    diagnostics_timer_ = create_wall_timer(
      std::chrono::seconds(1), std::bind(&UsbCameraFfmpeg::report_diagnostics, this));
  }

  ~UsbCameraFfmpeg() override
  {
    stop();
  }

  void start()
  {
    if (started_.exchange(true)) {
      throw std::logic_error("camera node has already started");
    }
    stopping_ = false;
    last_frame_steady_ns_.store(steady_now_ns());
    worker_ = std::thread(&UsbCameraFfmpeg::process_frames, this);
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
      get_logger(),
      "Streaming V4L2 %s MJPEG %ldx%ld@%ld -> %s/ffmpeg",
      camera_->device_path().c_str(),
      config_.width, config_.height, config_.fps,
      config_.topic.c_str());
  }

  void stop()
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
    if (encoder_) {
      encoder_->shutdown();
      encoder_.reset();
    }
  }

  bool failed() const
  {
    return fatal_error_.load();
  }

private:
  static int64_t steady_now_ns()
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
      SteadyClock::now().time_since_epoch()).count();
  }

  void capture_frame(const uint8_t * data, size_t size) noexcept
  {
    if (stopping_ || data == nullptr || size == 0U) {
      invalid_frame_count_.fetch_add(1);
      return;
    }
    try {
      CapturedJpeg captured;
      captured.stamp = now();
      captured.queued_at = SteadyClock::now();
      captured.data.assign(data, data + size);
      captured_count_.fetch_add(1);
      last_frame_steady_ns_.store(steady_now_ns());
      frame_buffer_.push(std::move(captured));
    } catch (...) {
      set_fatal_error("failed to copy a frame from the V4L2 capture thread");
    }
  }

  void process_frames() noexcept
  {
    try {
      while (auto captured = frame_buffer_.wait_and_take()) {
        if (encoder_->getNumSubscribers() == 0) {
          skipped_without_subscriber_count_.fetch_add(1);
          continue;
        }
        const auto dequeue_time = SteadyClock::now();
        queued_nanoseconds_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
            dequeue_time - captured->queued_at).count());

        cv::Mat bgr = decode_jpeg_frame(
          captured->data, static_cast<int>(config_.width), static_cast<int>(config_.height));
        const auto decoded_time = SteadyClock::now();
        decode_nanoseconds_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
            decoded_time - dequeue_time).count());
        if (bgr.empty()) {
          decode_failure_count_.fetch_add(1);
          continue;
        }

        sensor_msgs::msg::Image image = make_bgr_image_message(
          bgr, captured->stamp, config_.frame_id);

        encoder_->publish(image);
        publish_nanoseconds_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
            SteadyClock::now() - decoded_time).count());
        encoded_count_.fetch_add(1);
      }
    } catch (const std::exception & error) {
      set_fatal_error(std::string("frame processing failed: ") + error.what());
    } catch (...) {
      set_fatal_error("frame processing failed with an unknown error");
    }
  }

  void set_fatal_error(const std::string & message) noexcept
  {
    if (!fatal_error_.exchange(true)) {
      RCLCPP_FATAL(get_logger(), "%s", message.c_str());
    }
  }

  void report_diagnostics()
  {
    const auto captured = captured_count_.load();
    const auto encoded = encoded_count_.load();
    const auto overwritten = frame_buffer_.overwritten_count();
    const auto interval_encoded = encoded - previous_encoded_count_;
    const auto interval_queue_ns = queued_nanoseconds_.exchange(0);
    const auto interval_decode_ns = decode_nanoseconds_.exchange(0);
    const auto interval_publish_ns = publish_nanoseconds_.exchange(0);
    const double divisor = interval_encoded == 0 ? 1.0 : static_cast<double>(interval_encoded);

    RCLCPP_INFO(
      get_logger(),
      "frames captured=%lu encoded=%lu overwritten=%lu no_subscriber=%lu invalid=%lu "
      "decode_failed=%lu rate=%lu fps avg_queue=%.2f ms avg_decode=%.2f ms avg_publish=%.2f ms",
      captured, encoded, overwritten, skipped_without_subscriber_count_.load(),
      invalid_frame_count_.load() + (camera_ ? camera_->invalid_count() : 0U),
      decode_failure_count_.load(), interval_encoded,
      interval_queue_ns / divisor / 1.0e6, interval_decode_ns / divisor / 1.0e6,
      interval_publish_ns / divisor / 1.0e6);
    previous_encoded_count_ = encoded;

    if (started_ && !stopping_) {
      if (camera_ && camera_->failed()) {
        set_fatal_error("V4L2 camera capture failed: " + camera_->error_message());
        return;
      }
      const auto silence_ns = steady_now_ns() - last_frame_steady_ns_.load();
      if (silence_ns > static_cast<int64_t>(config_.frame_timeout_seconds * 1.0e9)) {
        set_fatal_error(
          "no valid V4L2 camera frame received for " +
          std::to_string(config_.frame_timeout_seconds) + " seconds");
      }
    }
  }

  CameraConfiguration config_;
  pluginlib::UniquePtr<image_transport::PublisherPlugin> encoder_;
  LatestFrameBuffer<CapturedJpeg> frame_buffer_;
  std::unique_ptr<V4l2Camera> camera_;
  std::thread worker_;
  rclcpp::TimerBase::SharedPtr diagnostics_timer_;
  std::atomic<bool> started_{false};
  std::atomic<bool> stopping_{false};
  std::atomic<bool> fatal_error_{false};
  std::atomic<int64_t> last_frame_steady_ns_{0};
  std::atomic<uint64_t> captured_count_{0};
  std::atomic<uint64_t> encoded_count_{0};
  std::atomic<uint64_t> skipped_without_subscriber_count_{0};
  std::atomic<uint64_t> invalid_frame_count_{0};
  std::atomic<uint64_t> decode_failure_count_{0};
  std::atomic<int64_t> queued_nanoseconds_{0};
  std::atomic<int64_t> decode_nanoseconds_{0};
  std::atomic<int64_t> publish_nanoseconds_{0};
  uint64_t previous_encoded_count_{0};
};

}  // namespace fastumi_usb_camera

namespace
{

volatile std::sig_atomic_t stop_requested = 0;

void request_stop(int signal_number)
{
  (void)signal_number;
  stop_requested = 1;
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(
    argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, request_stop);
  std::signal(SIGTERM, request_stop);
  try {
    bool failed = false;
    {
      pluginlib::ClassLoader<image_transport::PublisherPlugin> loader(
        "image_transport", "image_transport::PublisherPlugin");
      auto node = std::make_shared<fastumi_usb_camera::UsbCameraFfmpeg>(loader);
      node->start();
      rclcpp::executors::SingleThreadedExecutor executor;
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
    RCLCPP_FATAL(rclcpp::get_logger("usb_camera_ffmpeg"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
}
