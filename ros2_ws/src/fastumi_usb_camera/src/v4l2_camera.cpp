/**
 * @file v4l2_camera.cpp
 * @brief 实现 USB MJPEG 相机的 V4L2 设备发现、mmap 采集与确定性释放。
 */

#include "fastumi_usb_camera/v4l2_camera.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <linux/videodev2.h>
#include <mutex>
#include <optional>
#include <poll.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace fastumi_usb_camera
{
namespace
{

constexpr int kPollTimeoutMilliseconds = 100;  ///< 停止采集时轮询线程的最大响应时间。
constexpr uint32_t kCaptureBufferCount = 4;  ///< 向 V4L2 驱动申请的 mmap 缓冲区数量。

/**
 * @brief 执行可被信号中断的 ioctl。
 * @param[in] file_descriptor 视频设备文件描述符。
 * @param[in] request ioctl 请求编号。
 * @param[in,out] argument ioctl 参数结构体。
 * @return ioctl 返回值，仅在 EINTR 时自动重试。
 */
int ioctl_retry(int file_descriptor, unsigned long request, void * argument)
{
  int result;  ///< 当前 ioctl 调用结果。
  do {
    result = ioctl(file_descriptor, request, argument);
  } while (result < 0 && errno == EINTR);
  return result;
}

/**
 * @brief 把当前 errno 和操作名组合为运行时错误。
 * @param[in] action 失败的系统操作名称。
 * @return 包含 errno 文本的异常对象。
 */
std::runtime_error system_error(const std::string & action)
{
  return std::runtime_error(action + ": " + std::strerror(errno));
}

/**
 * @brief 读取 sysfs 单词属性。
 * @param[in] path 属性文件路径。
 * @return 首个空白分隔字段，读取失败时为空字符串。
 */
std::string read_trimmed(const std::filesystem::path & path)
{
  std::ifstream stream(path);  ///< sysfs 属性输入流。
  std::string value;  ///< 读取到的属性文本。
  stream >> value;
  return value;
}

/**
 * @brief 解析 videoN 节点名称中的数字编号。
 * @param[in] name sysfs 视频节点名称。
 * @return 有效 videoN 名称的数字编号，其他名称返回空值。
 */
std::optional<uint64_t> video_node_number(const std::string & name)
{
  constexpr char kPrefix[] = "video";  ///< V4L2 视频节点名称前缀。
  const std::string prefix(kPrefix);  ///< 用于检查节点名称的字符串前缀。
  if (name.compare(0, prefix.size(), prefix) != 0 || name.size() == prefix.size()) {
    return std::nullopt;
  }
  const std::string number_text = name.substr(prefix.size());  ///< 节点名称中的数字部分。
  const bool all_digits = std::all_of(
    number_text.begin(), number_text.end(),
    [](unsigned char character) {return std::isdigit(character) != 0;});  ///< 数字部分是否合法。
  if (!all_digits) {
    return std::nullopt;
  }
  try {
    return std::stoull(number_text);
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

/**
 * @brief 从视频接口向上查找包含 USB 标识的设备目录。
 * @param[in] interface_path 视频节点关联的 sysfs 设备路径。
 * @return USB 设备目录，未找到时为空路径。
 */
std::filesystem::path find_usb_device(const std::filesystem::path & interface_path)
{
  std::filesystem::path current = interface_path;  ///< 当前检查的 sysfs 祖先目录。
  while (!current.empty()) {
    if (std::filesystem::is_regular_file(current / "idVendor") &&
      std::filesystem::is_regular_file(current / "idProduct"))
    {
      return current;
    }
    if (current == current.root_path()) {
      break;
    }
    current = current.parent_path();
  }
  return {};
}

}  // namespace

std::string discover_video_device(
  const CameraConfiguration & config,
  const std::filesystem::path & video_class_root,
  const std::filesystem::path & device_root)
{
  if (!config.video_device.empty()) {
    if (!std::filesystem::exists(config.video_device)) {
      throw std::runtime_error("video device does not exist: " + config.video_device);
    }
    return config.video_device;
  }

  using Candidate = std::pair<uint64_t, std::string>;  ///< 视频节点数字编号及其设备路径。
  std::vector<Candidate> candidates;  ///< 可用的 USB 主视频节点。
  std::error_code error;  ///< filesystem 遍历和链接解析错误。
  std::filesystem::directory_iterator iterator(video_class_root, error);  ///< sysfs 视频节点迭代器。
  const std::filesystem::directory_iterator end;  ///< sysfs 目录迭代终点。
  if (error) {
    throw std::runtime_error(
            "cannot enumerate V4L2 devices under " + video_class_root.string() +
            ": " + error.message());
  }
  for (; iterator != end; iterator.increment(error)) {
    if (error) {
      throw std::runtime_error(
              "cannot enumerate V4L2 devices under " + video_class_root.string() +
              ": " + error.message());
    }
    const std::filesystem::path class_entry = iterator->path();  ///< 当前 videoN sysfs 目录。
    const std::optional<uint64_t> node_number =
      video_node_number(class_entry.filename().string());  ///< 当前节点的数字编号。
    if (!node_number.has_value()) {
      continue;
    }
    if (read_trimmed(class_entry / "index") != "0") {
      continue;
    }
    const std::filesystem::path interface_path =
      std::filesystem::canonical(class_entry / "device", error);  ///< 视频接口真实路径。
    if (error) {
      error.clear();
      continue;
    }
    const std::filesystem::path usb_device = find_usb_device(interface_path);  ///< USB 设备目录。
    if (usb_device.empty()) {
      continue;
    }
    candidates.emplace_back(
      *node_number, (device_root / class_entry.filename()).string());
  }
  if (error) {
    throw std::runtime_error(
            "cannot enumerate V4L2 devices under " + video_class_root.string() +
            ": " + error.message());
  }

  if (candidates.empty()) {
    throw std::runtime_error(
            "no USB V4L2 primary video device found under " + video_class_root.string());
  }
  std::sort(
    candidates.begin(), candidates.end(),
    [](const Candidate & left, const Candidate & right) {
      return left.first < right.first;
    });
  return candidates.front().second;
}

/** @brief 保存单个 V4L2 mmap 缓冲区的地址与长度。 */
struct MappedBuffer
{
  void * address{MAP_FAILED};  ///< 映射到用户空间的起始地址。
  size_t length{0};  ///< 映射长度，单位为字节。
};

class V4l2Camera::Impl
{
public:
  /** @brief 打开设备、协商模式并启动采集线程。 */
  Impl(const CameraConfiguration & config, FrameCallback callback)
  : device_path_(discover_video_device(config)), callback_(std::move(callback))
  {
    if (!callback_) {
      throw std::invalid_argument("V4L2 frame callback must not be empty");
    }
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

  Impl(const Impl &) = delete;
  Impl & operator=(const Impl &) = delete;

  /** @brief 保证析构时释放流、映射和文件描述符。 */
  ~Impl()
  {
    stop();
  }

  /** @brief 幂等停止采集并释放资源。 */
  void stop() noexcept
  {
    running_ = false;
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    close_resources();
  }

  /** @return 采集线程是否异常退出。 */
  bool failed() const noexcept
  {
    return failed_.load();
  }

  /** @return 采集线程错误说明。 */
  std::string error_message() const
  {
    std::lock_guard<std::mutex> lock(error_mutex_);  ///< 保护错误字符串读取。
    return error_message_;
  }

  /** @return 无效 V4L2 帧计数。 */
  uint64_t invalid_count() const noexcept
  {
    return invalid_count_.load();
  }

  /** @return 实际设备路径。 */
  const std::string & device_path() const noexcept
  {
    return device_path_;
  }

private:
  /** @brief 配置能力、MJPEG 模式、帧率、mmap 队列并开启视频流。 */
  void open_and_start(const CameraConfiguration & config)
  {
    file_descriptor_ = open(device_path_.c_str(), O_RDWR | O_NONBLOCK);
    if (file_descriptor_ < 0) {
      throw system_error("open " + device_path_);
    }
    try {
      v4l2_capability capabilities{};  ///< 驱动报告的设备能力。
      if (ioctl_retry(file_descriptor_, VIDIOC_QUERYCAP, &capabilities) < 0) {
        throw system_error("VIDIOC_QUERYCAP");
      }
      const uint32_t device_capabilities =
        (capabilities.capabilities & V4L2_CAP_DEVICE_CAPS) != 0U ?
        capabilities.device_caps : capabilities.capabilities;  ///< 当前节点有效能力位。
      if ((device_capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0U ||
        (device_capabilities & V4L2_CAP_STREAMING) == 0U)
      {
        throw std::runtime_error(device_path_ + " is not a streaming V4L2 capture device");
      }

      v4l2_format format{};  ///< 请求并接收实际 MJPEG 图像格式。
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

      v4l2_streamparm parameters{};  ///< 请求并接收实际采集帧周期。
      parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      parameters.parm.capture.timeperframe.numerator = 1U;
      parameters.parm.capture.timeperframe.denominator = static_cast<uint32_t>(config.fps);
      if (ioctl_retry(file_descriptor_, VIDIOC_S_PARM, &parameters) < 0) {
        throw system_error("VIDIOC_S_PARM");
      }
      const uint32_t numerator =
        parameters.parm.capture.timeperframe.numerator;  ///< 驱动采用的帧周期分子。
      const uint32_t denominator =
        parameters.parm.capture.timeperframe.denominator;  ///< 驱动采用的帧周期分母。
      if (numerator == 0U || denominator != static_cast<uint32_t>(config.fps) * numerator) {
        throw std::runtime_error("camera did not accept the requested frame rate");
      }

      v4l2_requestbuffers request{};  ///< mmap 缓冲区申请参数与实际数量。
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
        v4l2_buffer buffer{};  ///< 当前待查询和映射的驱动缓冲区。
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
        v4l2_buffer buffer{};  ///< 当前准备加入驱动采集队列的缓冲区。
        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        if (ioctl_retry(file_descriptor_, VIDIOC_QBUF, &buffer) < 0) {
          throw system_error("VIDIOC_QBUF");
        }
      }
      v4l2_buf_type buffer_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;  ///< STREAMON 缓冲区类型。
      if (ioctl_retry(file_descriptor_, VIDIOC_STREAMON, &buffer_type) < 0) {
        throw system_error("VIDIOC_STREAMON");
      }
      streaming_ = true;
    } catch (...) {
      close_resources();
      throw;
    }
  }

  /** @brief 轮询、出队完整帧并在回调返回后重新入队。 */
  void capture_loop() noexcept
  {
    try {
      while (running_.load()) {
        pollfd descriptor{};  ///< 当前视频文件描述符的轮询状态。
        descriptor.fd = file_descriptor_;
        descriptor.events = POLLIN | POLLPRI;
        const int poll_result =
          poll(&descriptor, 1, kPollTimeoutMilliseconds);  ///< 本轮 poll 返回状态。
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

        v4l2_buffer buffer{};  ///< 驱动返回的已填充 mmap 缓冲区。
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
          (buffer.flags & V4L2_BUF_FLAG_ERROR) == 0U;  ///< 当前帧是否可安全读取。
        if (valid && running_.load()) {
          callback_(
            static_cast<const uint8_t *>(buffers_[buffer.index].address), buffer.bytesused);
        } else if (!valid) {
          invalid_count_.fetch_add(1);
        }
        if (ioctl_retry(file_descriptor_, VIDIOC_QBUF, &buffer) < 0) {
          throw system_error("VIDIOC_QBUF");
        }
      }
    } catch (const std::exception & error) {
      {
        std::lock_guard<std::mutex> lock(error_mutex_);  ///< 保护错误字符串写入。
        error_message_ = error.what();
      }
      failed_ = true;
      running_ = false;
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(error_mutex_);  ///< 保护未知错误字符串写入。
        error_message_ = "unknown V4L2 capture error";
      }
      failed_ = true;
      running_ = false;
    }
  }

  /** @brief 关流、解除全部映射并关闭文件描述符。 */
  void close_resources() noexcept
  {
    if (streaming_ && file_descriptor_ >= 0) {
      v4l2_buf_type buffer_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;  ///< STREAMOFF 缓冲区类型。
      static_cast<void>(ioctl(file_descriptor_, VIDIOC_STREAMOFF, &buffer_type));
      streaming_ = false;
    }
    for (MappedBuffer & buffer : buffers_) {
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

  std::string device_path_;  ///< 实际打开的 V4L2 视频节点路径。
  FrameCallback callback_;  ///< 完整 MJPEG 帧接收回调。
  int file_descriptor_{-1};  ///< 已打开的视频设备文件描述符。
  bool streaming_{false};  ///< 是否已成功执行 VIDIOC_STREAMON。
  std::vector<MappedBuffer> buffers_;  ///< 驱动分配并映射的采集缓冲区。
  std::thread capture_thread_;  ///< 轮询并递送 MJPEG 帧的工作线程。
  std::atomic<bool> running_{false};  ///< 采集线程继续运行标志。
  std::atomic<bool> failed_{false};  ///< 采集线程不可恢复错误标志。
  std::atomic<uint64_t> invalid_count_{0};  ///< 无效或错误 V4L2 帧数量。
  mutable std::mutex error_mutex_;  ///< 保护采集错误文本。
  std::string error_message_;  ///< 采集线程最后错误说明。
};

V4l2Camera::V4l2Camera(const CameraConfiguration & config, FrameCallback callback)
: impl_(std::make_unique<Impl>(config, std::move(callback)))
{
}

V4l2Camera::~V4l2Camera() = default;

void V4l2Camera::stop() noexcept
{
  impl_->stop();
}

bool V4l2Camera::failed() const noexcept
{
  return impl_->failed();
}

std::string V4l2Camera::error_message() const
{
  return impl_->error_message();
}

uint64_t V4l2Camera::invalid_count() const noexcept
{
  return impl_->invalid_count();
}

const std::string & V4l2Camera::device_path() const noexcept
{
  return impl_->device_path();
}

}  // namespace fastumi_usb_camera
