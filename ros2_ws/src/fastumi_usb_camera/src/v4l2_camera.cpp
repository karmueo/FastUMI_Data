/**
 * @file v4l2_camera.cpp
 * @brief 管理 V4L2 MJPEG 队列、采集线程及有界流恢复。
 */
#include "fastumi_usb_camera/v4l2_camera.hpp"

#include <atomic>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <linux/videodev2.h>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace fastumi_usb_camera
{
namespace
{

constexpr int kPollTimeoutMilliseconds = 100;
constexpr uint32_t kCaptureBufferCount = 4;
constexpr unsigned int kMaxConsecutiveRestarts = 3;

/** @brief 表示可通过重建采集队列恢复的驱动错误。 */
class RecoverableCaptureError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

/** @brief 记录 poll 的原始事件掩码，区分队列错误与设备挂断。 */
std::string poll_error_message(short events)
{
  std::ostringstream message;
  message << "V4L2 camera poll revents=0x" << std::hex << events;
  if ((events & POLLERR) != 0) {message << " POLLERR";}
  if ((events & POLLHUP) != 0) {message << " POLLHUP";}
  if ((events & POLLNVAL) != 0) {message << " POLLNVAL";}
  return message.str();
}

/** @brief 把当前 errno 和操作名组合为运行时错误。 */
std::runtime_error system_error(const std::string & action)
{
  const int error_number = errno;
  return std::runtime_error(
    action + ": " + std::strerror(error_number) +
    " (errno=" + std::to_string(error_number) + ")");
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
struct V4l2Camera::Impl
{
public:
  using FrameCallback = void (*)(const uint8_t *, size_t, void *);

  /** @brief 打开设备、严格协商采集模式并启动读帧线程。 */
  Impl(
    const CameraConfiguration & config, FrameCallback callback, void * user_data,
    RecoveryCallback recovery_callback, std::shared_ptr<V4l2Io> io)
  : device_path_(discover_video_device(config)), config_(config),
    recovery_callback_(std::move(recovery_callback)), io_(std::move(io)),
    callback_(callback), user_data_(user_data)
  {
    open_and_start(config);
    running_ = true;
    try {
      capture_thread_ = std::thread(&Impl::capture_loop, this);
    } catch (...) {
      running_ = false;
      close_resources();
      throw;
    }
  }

  ~Impl()
  {
    stop();
  }

  /** @brief 可重复调用地停止读帧线程并释放 V4L2 资源。 */
  void stop()
  {
    running_ = false;
    stop_condition_.notify_all();
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

  uint64_t recovery_count() const
  {
    return recovery_count_.load();
  }

private:
  /** @brief 执行可被信号中断的 V4L2 ioctl，仅对 EINTR 重试。 */
  int ioctl_retry(int file_descriptor, unsigned long request, void * argument)
  {
    int result;
    do {
      result = io_->control(file_descriptor, request, argument);
    } while (result < 0 && errno == EINTR);
    return result;
  }

  /** @brief 一个由 V4L2 驱动分配并映射到用户空间的缓冲区。 */
  struct MappedBuffer
  {
    void * address{MAP_FAILED};
    size_t length{0};
  };

  /** @brief 配置能力、MJPEG 格式、帧率、mmap 队列并开流。 */
  void open_and_start(const CameraConfiguration & config)
  {
    file_descriptor_ = io_->open_device(device_path_);
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
        buffers_[index].address = io_->map_buffer(file_descriptor_, buffer.length, buffer.m.offset);
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

  /** @brief 重建队列，最多连续尝试三次；停止信号可中断退避等待。 */
  void recover_stream(const std::string & reason)
  {
    if (++consecutive_restarts_ > kMaxConsecutiveRestarts) {
      throw std::runtime_error(reason + "; capture restart limit exhausted");
    }
    if (recovery_callback_) {
      recovery_callback_(
        device_path_ + ": " + reason + "; restarting V4L2 capture (" +
        std::to_string(consecutive_restarts_) + "/" +
        std::to_string(kMaxConsecutiveRestarts) + ")");
    }
    close_resources();
    {
      std::unique_lock<std::mutex> lock(stop_mutex_);
      stop_condition_.wait_for(
        lock, std::chrono::milliseconds(100), [this]() {return !running_.load();});
    }
    if (!running_.load()) {
      return;
    }
    check_frame_timeout();
    try {
      // 重开原设备路径，重新协商模式、分配 mmap 并入队所有缓冲区。
      open_and_start(config_);
      recovery_count_.fetch_add(1);
    } catch (const std::exception & error) {
      throw std::runtime_error(reason + "; capture restart failed: " + error.what());
    }
  }

  /** @brief 所有错误重试共享自最后有效帧起的超时预算。 */
  void check_frame_timeout() const
  {
    if (std::chrono::duration<double>(
        std::chrono::steady_clock::now() - last_valid_frame_).count() >
      config_.frame_timeout_seconds)
    {
      throw std::runtime_error(
        "no valid V4L2 camera frame received for " +
        std::to_string(config_.frame_timeout_seconds) + " seconds");
    }
  }

  /** @brief 轮询一个缓冲区；EIO 后不复用驱动可能已出队的缓冲区。 */
  void capture_one()
  {
    pollfd descriptor{};
    descriptor.fd = file_descriptor_;
    descriptor.events = POLLIN;
    const int poll_result = io_->wait(&descriptor, kPollTimeoutMilliseconds);
    if (!running_.load()) {
      return;
    }
    if (poll_result < 0) {
      if (errno == EINTR) {
        return;
      }
      throw system_error("poll");
    }
    if (poll_result == 0) {
      return;
    }
    if ((descriptor.revents & (POLLHUP | POLLNVAL)) != 0) {
      throw std::runtime_error(poll_error_message(descriptor.revents));
    }
    if ((descriptor.revents & POLLIN) == 0) {
      if ((descriptor.revents & POLLERR) != 0) {
        throw RecoverableCaptureError(poll_error_message(descriptor.revents));
      }
      return;
    }

    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    if (ioctl_retry(file_descriptor_, VIDIOC_DQBUF, &buffer) < 0) {
      if (errno == EAGAIN) {
        if ((descriptor.revents & POLLERR) != 0) {
          throw RecoverableCaptureError(poll_error_message(descriptor.revents));
        }
        return;
      }
      if (errno == EIO) {
        throw RecoverableCaptureError(system_error("VIDIOC_DQBUF").what());
      }
      throw system_error("VIDIOC_DQBUF");
    }
    if (buffer.index >= buffers_.size()) {
      throw std::runtime_error("VIDIOC_DQBUF returned an out-of-range buffer index");
    }
    const bool valid = buffer.bytesused > 0U &&
      buffer.bytesused <= buffers_[buffer.index].length &&
      (buffer.flags & V4L2_BUF_FLAG_ERROR) == 0U;
    if (valid && running_.load()) {
      callback_(
        static_cast<const uint8_t *>(buffers_[buffer.index].address),
        buffer.bytesused, user_data_);
      last_valid_frame_ = std::chrono::steady_clock::now();
      if (consecutive_restarts_ != 0 && recovery_callback_) {
        recovery_callback_(device_path_ + ": V4L2 capture recovered; valid frame received");
      }
      consecutive_restarts_ = 0;
    } else if (!valid) {
      invalid_count_.fetch_add(1);
    }
    if (ioctl_retry(file_descriptor_, VIDIOC_QBUF, &buffer) < 0) {
      if (errno == EIO) {
        throw RecoverableCaptureError(system_error("VIDIOC_QBUF").what());
      }
      throw system_error("VIDIOC_QBUF");
    }
  }

  /** @brief 采集线程统一处理可恢复错误，持续无帧或断开时保留失败退出语义。 */
  void capture_loop() noexcept
  {
    last_valid_frame_ = std::chrono::steady_clock::now();
    try {
      while (running_.load()) {
        check_frame_timeout();
        try {
          capture_one();
        } catch (const RecoverableCaptureError & error) {
          if (running_.load()) {
            recover_stream(error.what());
          }
        }
      }
    } catch (const std::exception & error) {
      if (running_.load()) {
        {
          std::lock_guard<std::mutex> lock(error_mutex_);
          error_message_ = device_path_ + ": " + error.what();
        }
        failed_ = true;
        running_ = false;
      }
    }
  }

  /** @brief 关流、解除映射并关闭视频节点；析构路径不抛异常。 */
  void close_resources() noexcept
  {
    if (streaming_ && file_descriptor_ >= 0) {
      v4l2_buf_type buffer_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      static_cast<void>(io_->control(file_descriptor_, VIDIOC_STREAMOFF, &buffer_type));
      streaming_ = false;
    }
    for (auto & buffer : buffers_) {
      if (buffer.address != MAP_FAILED) {
        static_cast<void>(io_->unmap_buffer(buffer.address, buffer.length));
        buffer.address = MAP_FAILED;
      }
    }
    buffers_.clear();
    if (file_descriptor_ >= 0) {
      static_cast<void>(io_->close_device(file_descriptor_));
      file_descriptor_ = -1;
    }
  }

  std::string device_path_;
  CameraConfiguration config_;
  RecoveryCallback recovery_callback_;
  std::shared_ptr<V4l2Io> io_;
  FrameCallback callback_;
  void * user_data_;
  int file_descriptor_{-1};
  bool streaming_{false};
  std::vector<MappedBuffer> buffers_;
  std::thread capture_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> failed_{false};
  std::atomic<uint64_t> invalid_count_{0};
  std::atomic<uint64_t> recovery_count_{0};
  unsigned int consecutive_restarts_{0};
  std::chrono::steady_clock::time_point last_valid_frame_;
  std::mutex stop_mutex_;
  std::condition_variable stop_condition_;
  mutable std::mutex error_mutex_;
  std::string error_message_;
};

V4l2Camera::V4l2Camera(
  const CameraConfiguration & config, FrameCallback callback, void * user_data,
  RecoveryCallback recovery_callback, std::shared_ptr<V4l2Io> io)
: implementation_(std::make_unique<Impl>(
    config, callback, user_data, std::move(recovery_callback), std::move(io)))
{
}

V4l2Camera::~V4l2Camera() = default;

void V4l2Camera::stop() {implementation_->stop();}
bool V4l2Camera::failed() const {return implementation_->failed();}
std::string V4l2Camera::error_message() const {return implementation_->error_message();}
uint64_t V4l2Camera::invalid_count() const {return implementation_->invalid_count();}
const std::string & V4l2Camera::device_path() const {return implementation_->device_path();}
uint64_t V4l2Camera::recovery_count() const {return implementation_->recovery_count();}

}  // namespace fastumi_usb_camera
