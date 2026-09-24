/**
 * @file usb_camera_ffmpeg.cpp
 * @brief 按稳定 V4L2 物理端口采集 MJPEG，并用软件或 Jetson 硬件发布 H.264。
 */

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <linux/videodev2.h>
#include <memory>
#include <mutex>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include <ffmpeg_image_transport_msgs/msg/ffmpeg_packet.hpp>
#include <image_transport/publisher_plugin.hpp>
#include <opencv2/imgcodecs.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "fastumi_usb_camera/configuration.hpp"
#include "fastumi_usb_camera/hardware_h264_encoder.hpp"
#include "fastumi_usb_camera/jpeg_frame.hpp"
#include "fastumi_usb_camera/latest_frame_buffer.hpp"

namespace fastumi_usb_camera
{

using SteadyClock = std::chrono::steady_clock;

struct CapturedJpeg
{
  builtin_interfaces::msg::Time stamp;
  SteadyClock::time_point queued_at;
  std::vector<uint8_t> data;
};

namespace
{

constexpr int kPollTimeoutMilliseconds = 100;
constexpr uint32_t kCaptureBufferCount = 4;

/** @brief 执行可被信号中断的 V4L2 ioctl，仅对 EINTR 重试。 */
int ioctl_retry(int file_descriptor, unsigned long request, void * argument)
{
  int result;
  do {
    result = ioctl(file_descriptor, request, argument);
  } while (result < 0 && errno == EINTR);
  return result;
}

/** @brief 把当前 errno 和操作名组合为运行时错误。 */
std::runtime_error system_error(const std::string & action)
{
  return std::runtime_error(action + ": " + std::strerror(errno));
}

/** @brief 读取 sysfs 单词属性，失败时返回空字符串。 */
std::string read_trimmed(const std::filesystem::path & path)
{
  std::ifstream stream(path);
  std::string value;
  stream >> value;
  return value;
}

/** @brief 读取 sysfs 十六进制 USB 标识，失败时返回 -1。 */
int64_t read_hex_id(const std::filesystem::path & path)
{
  const std::string value = read_trimmed(path);
  if (value.empty()) {
    return -1;
  }
  try {
    return static_cast<int64_t>(std::stoul(value, nullptr, 16));
  } catch (const std::exception &) {
    return -1;
  }
}

/**
 * @brief 解析显式 by-path，或按 VID/PID 和序列号选择唯一主视频节点。
 * @throws std::runtime_error 设备不存在或匹配数不为一。
 */
std::string discover_video_device(const CameraConfiguration & config)
{
  if (!config.video_device.empty()) {
    if (!std::filesystem::exists(config.video_device)) {
      throw std::runtime_error("video device does not exist: " + config.video_device);
    }
    return config.video_device;
  }

  std::vector<std::string> matches;
  std::error_code error;
  const std::filesystem::path video_class("/sys/class/video4linux");
  for (std::filesystem::directory_iterator iterator(video_class, error), end;
    !error && iterator != end; iterator.increment(error))
  {
    const auto class_entry = iterator->path();
    if (read_trimmed(class_entry / "index") != "0") {
      continue;
    }
    const auto usb_interface = std::filesystem::canonical(class_entry / "device", error);
    if (error) {
      error.clear();
      continue;
    }
    const auto usb_device = usb_interface.parent_path();
    if (read_hex_id(usb_device / "idVendor") != config.vendor_id ||
      read_hex_id(usb_device / "idProduct") != config.product_id)
    {
      continue;
    }
    if (!config.serial_number.empty() &&
      read_trimmed(usb_device / "serial") != config.serial_number)
    {
      continue;
    }
    matches.push_back("/dev/" + class_entry.filename().string());
  }
  if (matches.size() != 1) {
    std::ostringstream message;
    message << "expected exactly one V4L2 camera matching " << std::hex <<
      std::setfill('0') << std::setw(4) << config.vendor_id << ':' <<
      std::setw(4) << config.product_id;
    if (!config.serial_number.empty()) {
      message << " (serial_number=" << config.serial_number << ')';
    }
    message << ", found " << std::dec << matches.size();
    throw std::runtime_error(message.str());
  }
  return matches.front();
}

}  // namespace

/** @brief 通过 uvcvideo 的 V4L2 mmap 队列采集原生 MJPEG 帧。 */
class V4l2Camera
{
public:
  using FrameCallback = void (*)(const uint8_t *, size_t, void *);

  /** @brief 打开设备、严格协商采集模式并启动读帧线程。 */
  V4l2Camera(const CameraConfiguration & config, FrameCallback callback, void * user_data)
  : device_path_(discover_video_device(config)), callback_(callback), user_data_(user_data)
  {
    open_and_start(config);
    running_ = true;
    try {
      capture_thread_ = std::thread(&V4l2Camera::capture_loop, this);
    } catch (...) {
      running_ = false;
      close_resources();
      throw;
    }
  }

  V4l2Camera(const V4l2Camera &) = delete;
  V4l2Camera & operator=(const V4l2Camera &) = delete;

  ~V4l2Camera()
  {
    stop();
  }

  /** @brief 可重复调用地停止读帧线程并释放 V4L2 资源。 */
  void stop()
  {
    running_ = false;
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    close_resources();
  }

  /** @brief 返回采集线程是否因不可恢复错误退出。 */
  bool failed() const
  {
    return failed_.load();
  }

  /** @brief 返回采集线程的最后错误说明。 */
  std::string error_message() const
  {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return error_message_;
  }

  /** @brief 返回驱动标记为错误或越界的帧数。 */
  uint64_t invalid_count() const
  {
    return invalid_count_.load();
  }

  /** @brief 返回实际打开的 V4L2 设备路径。 */
  const std::string & device_path() const
  {
    return device_path_;
  }

private:
  /** @brief 一个由 V4L2 驱动分配并映射到用户空间的缓冲区。 */
  struct MappedBuffer
  {
    void * address{MAP_FAILED};
    size_t length{0};
  };

  /** @brief 配置能力、MJPEG 格式、帧率、mmap 队列并开流。 */
  void open_and_start(const CameraConfiguration & config)
  {
    file_descriptor_ = open(device_path_.c_str(), O_RDWR | O_NONBLOCK);
    if (file_descriptor_ < 0) {
      throw system_error("open " + device_path_);
    }
    try {
      v4l2_capability capabilities{};
      if (ioctl_retry(file_descriptor_, VIDIOC_QUERYCAP, &capabilities) < 0) {
        throw system_error("VIDIOC_QUERYCAP");
      }
      const uint32_t device_capabilities =
        (capabilities.capabilities & V4L2_CAP_DEVICE_CAPS) != 0U ?
        capabilities.device_caps : capabilities.capabilities;
      if ((device_capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0U ||
        (device_capabilities & V4L2_CAP_STREAMING) == 0U)
      {
        throw std::runtime_error(device_path_ + " is not a streaming V4L2 capture device");
      }

      v4l2_format format{};
      format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      format.fmt.pix.width = static_cast<uint32_t>(config.width);
      format.fmt.pix.height = static_cast<uint32_t>(config.height);
      format.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
      format.fmt.pix.field = V4L2_FIELD_NONE;
      if (ioctl_retry(file_descriptor_, VIDIOC_S_FMT, &format) < 0) {
        throw system_error("VIDIOC_S_FMT");
      }
      if (format.fmt.pix.width != static_cast<uint32_t>(config.width) ||
        format.fmt.pix.height != static_cast<uint32_t>(config.height) ||
        format.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG)
      {
        throw std::runtime_error("camera did not accept the requested MJPEG dimensions");
      }

      v4l2_streamparm parameters{};
      parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      parameters.parm.capture.timeperframe.numerator = 1U;
      parameters.parm.capture.timeperframe.denominator = static_cast<uint32_t>(config.fps);
      if (ioctl_retry(file_descriptor_, VIDIOC_S_PARM, &parameters) < 0) {
        throw system_error("VIDIOC_S_PARM");
      }
      const auto numerator = parameters.parm.capture.timeperframe.numerator;
      const auto denominator = parameters.parm.capture.timeperframe.denominator;
      if (numerator == 0U || denominator != static_cast<uint32_t>(config.fps) * numerator) {
        throw std::runtime_error("camera did not accept the requested frame rate");
      }

      v4l2_requestbuffers request{};
      request.count = kCaptureBufferCount;
      request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      request.memory = V4L2_MEMORY_MMAP;
      if (ioctl_retry(file_descriptor_, VIDIOC_REQBUFS, &request) < 0) {
        throw system_error("VIDIOC_REQBUFS");
      }
      if (request.count < 2U) {
        throw std::runtime_error("V4L2 driver supplied fewer than two capture buffers");
      }
      buffers_.resize(request.count);
      for (uint32_t index = 0; index < request.count; ++index) {
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        if (ioctl_retry(file_descriptor_, VIDIOC_QUERYBUF, &buffer) < 0) {
          throw system_error("VIDIOC_QUERYBUF");
        }
        buffers_[index].length = buffer.length;
        buffers_[index].address = mmap(
          nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED,
          file_descriptor_, buffer.m.offset);
        if (buffers_[index].address == MAP_FAILED) {
          throw system_error("mmap");
        }
      }
      for (uint32_t index = 0; index < buffers_.size(); ++index) {
        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        if (ioctl_retry(file_descriptor_, VIDIOC_QBUF, &buffer) < 0) {
          throw system_error("VIDIOC_QBUF");
        }
      }
      v4l2_buf_type buffer_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      if (ioctl_retry(file_descriptor_, VIDIOC_STREAMON, &buffer_type) < 0) {
        throw system_error("VIDIOC_STREAMON");
      }
      streaming_ = true;
    } catch (...) {
      close_resources();
      throw;
    }
  }

  /** @brief 轮询并出队最新 MJPEG 帧，回调复制完成后立即重新入队。 */
  void capture_loop() noexcept
  {
    try {
      while (running_.load()) {
        pollfd descriptor{};
        descriptor.fd = file_descriptor_;
        descriptor.events = POLLIN | POLLPRI;
        const int poll_result = poll(&descriptor, 1, kPollTimeoutMilliseconds);
        if (poll_result < 0) {
          if (errno == EINTR) {
            continue;
          }
          throw system_error("poll");
        }
        if (poll_result == 0) {
          continue;
        }
        if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
          throw std::runtime_error("V4L2 camera poll returned an error event");
        }

        v4l2_buffer buffer{};
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        if (ioctl_retry(file_descriptor_, VIDIOC_DQBUF, &buffer) < 0) {
          if (errno == EAGAIN) {
            continue;
          }
          throw system_error("VIDIOC_DQBUF");
        }
        const bool valid = buffer.index < buffers_.size() && buffer.bytesused > 0U &&
          buffer.bytesused <= buffers_[buffer.index].length &&
          (buffer.flags & V4L2_BUF_FLAG_ERROR) == 0U;
        if (valid && running_.load()) {
          callback_(
            static_cast<const uint8_t *>(buffers_[buffer.index].address),
            buffer.bytesused, user_data_);
        } else if (!valid) {
          invalid_count_.fetch_add(1);
        }
        if (ioctl_retry(file_descriptor_, VIDIOC_QBUF, &buffer) < 0) {
          throw system_error("VIDIOC_QBUF");
        }
      }
    } catch (const std::exception & error) {
      {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_message_ = error.what();
      }
      failed_ = true;
      running_ = false;
    }
  }

  /** @brief 关流、解除映射并关闭视频节点；析构路径不抛异常。 */
  void close_resources() noexcept
  {
    if (streaming_ && file_descriptor_ >= 0) {
      v4l2_buf_type buffer_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      static_cast<void>(ioctl(file_descriptor_, VIDIOC_STREAMOFF, &buffer_type));
      streaming_ = false;
    }
    for (auto & buffer : buffers_) {
      if (buffer.address != MAP_FAILED) {
        static_cast<void>(munmap(buffer.address, buffer.length));
        buffer.address = MAP_FAILED;
      }
    }
    buffers_.clear();
    if (file_descriptor_ >= 0) {
      static_cast<void>(close(file_descriptor_));
      file_descriptor_ = -1;
    }
  }

  std::string device_path_;
  FrameCallback callback_;
  void * user_data_;
  int file_descriptor_{-1};
  bool streaming_{false};
  std::vector<MappedBuffer> buffers_;
  std::thread capture_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> failed_{false};
  std::atomic<uint64_t> invalid_count_{0};
  mutable std::mutex error_mutex_;
  std::string error_message_;
};

class UsbCameraFfmpeg : public rclcpp::Node
{
public:
  explicit UsbCameraFfmpeg(
    pluginlib::ClassLoader<image_transport::PublisherPlugin> & loader)
  : Node("usb_camera_ffmpeg")
  {
    config_.vendor_id = declare_parameter<int64_t>("vendor_id", config_.vendor_id);
    config_.product_id = declare_parameter<int64_t>("product_id", config_.product_id);
    config_.width = declare_parameter<int64_t>("width", config_.width);
    config_.height = declare_parameter<int64_t>("height", config_.height);
    config_.fps = declare_parameter<int64_t>("fps", config_.fps);
    config_.frame_timeout_seconds = declare_parameter<double>(
      "frame_timeout_seconds", config_.frame_timeout_seconds);
    config_.video_device = declare_parameter<std::string>("video_device", "");
    config_.serial_number = declare_parameter<std::string>("serial_number", "");
    config_.frame_id = declare_parameter<std::string>("frame_id", config_.frame_id);
    config_.topic = declare_parameter<std::string>("topic", config_.topic);
    h264_encoder_ = declare_parameter<std::string>("h264_encoder", "hardware");
    const auto publish_reliability = declare_parameter<std::string>(
      "publish_reliability", "reliable");
    validate_configuration(config_);
    if (h264_encoder_ != "hardware" && h264_encoder_ != "software") {
      throw std::invalid_argument("h264_encoder must be hardware or software");
    }
    if (publish_reliability != "reliable" && publish_reliability != "best_effort") {
      throw std::invalid_argument("publish_reliability must be reliable or best_effort");
    }

    auto qos = rmw_qos_profile_sensor_data;
    qos.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
    qos.depth = 20;
    qos.reliability = publish_reliability == "reliable" ?
      RMW_QOS_POLICY_RELIABILITY_RELIABLE : RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
    qos.durability = RMW_QOS_POLICY_DURABILITY_VOLATILE;
    if (h264_encoder_ == "software") {
      software_encoder_ = loader.createUniqueInstance(
        image_transport::PublisherPlugin::getLookupName("ffmpeg"));
      software_encoder_->advertise(this, config_.topic, qos);
    } else {
      auto image_qos = rclcpp::SensorDataQoS().keep_last(20);
      if (publish_reliability == "reliable") {
        image_qos.reliable();
      }
      hardware_publisher_ = create_publisher<
        ffmpeg_image_transport_msgs::msg::FFMPEGPacket>(
        config_.topic + "/ffmpeg", image_qos);
      HardwareH264Configuration hardware_configuration;
      hardware_configuration.width = static_cast<int>(config_.width);
      hardware_configuration.height = static_cast<int>(config_.height);
      hardware_configuration.fps = static_cast<int>(config_.fps);
      hardware_encoder_ = std::make_unique<HardwareH264Encoder>(
        hardware_configuration,
        [this](HardwareH264Packet && packet) {publish_hardware_packet(std::move(packet));});
    }

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
        config_, &UsbCameraFfmpeg::v4l2_callback, this);
    } catch (...) {
      frame_buffer_.close();
      worker_.join();
      started_ = false;
      throw;
    }
    RCLCPP_INFO(
      get_logger(),
      "Streaming V4L2 %s MJPEG %ldx%ld@%ld -> %s/ffmpeg encoder=%s",
      camera_->device_path().c_str(),
      config_.width, config_.height, config_.fps,
      config_.topic.c_str(), h264_encoder_.c_str());
  }

  void stop()
  {
    if (started_.exchange(false)) {
      stopping_ = true;
      if (camera_) {
        camera_->stop();
        camera_.reset();
      }
      if (hardware_encoder_) {
        hardware_encoder_->stop();
      }
      frame_buffer_.close();
      if (worker_.joinable()) {
        worker_.join();
      }
    }
    hardware_encoder_.reset();
    hardware_publisher_.reset();
    if (software_encoder_) {
      software_encoder_->shutdown();
      software_encoder_.reset();
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

  static void v4l2_callback(const uint8_t * data, size_t size, void * user_data) noexcept
  {
    static_cast<UsbCameraFfmpeg *>(user_data)->capture_frame(data, size);
  }

  static uint64_t stamp_nanoseconds(const builtin_interfaces::msg::Time & stamp)
  {
    if (stamp.sec < 0) {
      throw std::runtime_error("camera timestamp is negative");
    }
    return static_cast<uint64_t>(stamp.sec) * 1'000'000'000ULL + stamp.nanosec;
  }

  static builtin_interfaces::msg::Time time_from_nanoseconds(uint64_t nanoseconds)
  {
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(nanoseconds / 1'000'000'000ULL);
    stamp.nanosec = static_cast<uint32_t>(nanoseconds % 1'000'000'000ULL);
    return stamp;
  }

  bool has_subscribers() const
  {
    return h264_encoder_ == "hardware" ?
           hardware_publisher_->get_subscription_count() != 0U :
           software_encoder_->getNumSubscribers() != 0U;
  }

  void publish_hardware_packet(HardwareH264Packet && packet) noexcept
  {
    try {
      ffmpeg_image_transport_msgs::msg::FFMPEGPacket message;
      message.header.stamp = time_from_nanoseconds(packet.pts);
      message.header.frame_id = config_.frame_id;
      message.width = static_cast<int32_t>(config_.width);
      message.height = static_cast<int32_t>(config_.height);
      message.encoding = "h264;nv12;bgr8;bgr8";
      message.pts = packet.pts;
      message.flags = packet.keyframe ? 0x01U : 0x00U;
      message.is_bigendian = false;
      message.data = std::move(packet.data);
      hardware_publisher_->publish(std::move(message));
      pipeline_nanoseconds_.fetch_add(packet.latency_nanoseconds);
      encoded_count_.fetch_add(1);
    } catch (const std::exception & error) {
      set_fatal_error(std::string("hardware H.264 publish failed: ") + error.what());
    } catch (...) {
      set_fatal_error("hardware H.264 publish failed with an unknown error");
    }
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
        const bool subscribed = has_subscribers();
        const bool needs_hardware_probe = h264_encoder_ == "hardware" &&
          !hardware_probe_submitted_.load();
        if (!subscribed && !needs_hardware_probe) {
          skipped_without_subscriber_count_.fetch_add(1);
          continue;
        }
        const auto dequeue_time = SteadyClock::now();
        queued_nanoseconds_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
            dequeue_time - captured->queued_at).count());

        if (h264_encoder_ == "hardware") {
          if (!hardware_encoder_->push(
              stamp_nanoseconds(captured->stamp), captured->data, captured->queued_at))
          {
            if (stopping_) {
              return;
            }
            throw std::runtime_error(hardware_encoder_->error_message());
          }
          if (!subscribed) {
            hardware_probe_submitted_ = true;
          }
          continue;
        }

        cv::Mat bgr = decode_jpeg_frame(
          captured->data, static_cast<int>(config_.width), static_cast<int>(config_.height));
        const auto decoded_time = SteadyClock::now();
        decode_nanoseconds_.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
            decoded_time - dequeue_time).count());
        if (bgr.empty()) {
          decode_failure_count_.fetch_add(1);
          continue;
        }

        sensor_msgs::msg::Image image;
        image.header.stamp = captured->stamp;
        image.header.frame_id = config_.frame_id;
        image.height = static_cast<uint32_t>(bgr.rows);
        image.width = static_cast<uint32_t>(bgr.cols);
        image.encoding = sensor_msgs::image_encodings::BGR8;
        image.is_bigendian = false;
        image.step = static_cast<sensor_msgs::msg::Image::_step_type>(bgr.cols * bgr.elemSize());
        if (bgr.isContinuous()) {
          image.data.assign(bgr.datastart, bgr.dataend);
        } else {
          image.data.resize(static_cast<size_t>(image.step) * image.height);
          for (int row = 0; row < bgr.rows; ++row) {
            std::memcpy(
              image.data.data() + static_cast<size_t>(row) * image.step,
              bgr.ptr(row), image.step);
          }
        }

        software_encoder_->publish(image);
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
    const auto interval_pipeline_ns = pipeline_nanoseconds_.exchange(0);
    const double divisor = interval_encoded == 0 ? 1.0 : static_cast<double>(interval_encoded);

    RCLCPP_INFO(
      get_logger(),
      "encoder=%s frames captured=%lu encoded=%lu overwritten=%lu no_subscriber=%lu invalid=%lu "
      "decode_failed=%lu rate=%lu fps avg_queue=%.2f ms avg_decode=%.2f ms "
      "avg_publish=%.2f ms avg_pipeline=%.2f ms",
      h264_encoder_.c_str(),
      captured, encoded, overwritten, skipped_without_subscriber_count_.load(),
      invalid_frame_count_.load() + (camera_ ? camera_->invalid_count() : 0U),
      decode_failure_count_.load(), interval_encoded,
      interval_queue_ns / divisor / 1.0e6, interval_decode_ns / divisor / 1.0e6,
      interval_publish_ns / divisor / 1.0e6, interval_pipeline_ns / divisor / 1.0e6);
    previous_encoded_count_ = encoded;

    if (started_ && !stopping_) {
      if (hardware_encoder_ && hardware_encoder_->poll_error()) {
        set_fatal_error("NVIDIA H.264 hardware pipeline failed: " +
          hardware_encoder_->error_message());
        return;
      }
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
  std::string h264_encoder_;
  pluginlib::UniquePtr<image_transport::PublisherPlugin> software_encoder_;
  rclcpp::Publisher<ffmpeg_image_transport_msgs::msg::FFMPEGPacket>::SharedPtr hardware_publisher_;
  std::unique_ptr<HardwareH264Encoder> hardware_encoder_;
  LatestFrameBuffer<CapturedJpeg> frame_buffer_;
  std::unique_ptr<V4l2Camera> camera_;
  std::thread worker_;
  rclcpp::TimerBase::SharedPtr diagnostics_timer_;
  std::atomic<bool> started_{false};
  std::atomic<bool> stopping_{false};
  std::atomic<bool> fatal_error_{false};
  std::atomic<bool> hardware_probe_submitted_{false};
  std::atomic<int64_t> last_frame_steady_ns_{0};
  std::atomic<uint64_t> captured_count_{0};
  std::atomic<uint64_t> encoded_count_{0};
  std::atomic<uint64_t> skipped_without_subscriber_count_{0};
  std::atomic<uint64_t> invalid_frame_count_{0};
  std::atomic<uint64_t> decode_failure_count_{0};
  std::atomic<int64_t> queued_nanoseconds_{0};
  std::atomic<int64_t> decode_nanoseconds_{0};
  std::atomic<int64_t> publish_nanoseconds_{0};
  std::atomic<int64_t> pipeline_nanoseconds_{0};
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
