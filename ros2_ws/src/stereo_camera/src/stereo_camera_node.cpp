/**
 * @file stereo_camera_node.cpp
 * @brief 实现基于 V4L2 的 ROS 2 双目相机图像发布节点。
 * @details 采集横向拼接的 MJPEG 帧，拆分左右目图像并发布同步的 Image 与
 * CameraInfo。
 * @author 待确认
 * @date 创建：2026-09-07
 * @date 修改：2026-09-08
 */

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "rcl_interfaces/msg/parameter_descriptor.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "stereo_camera/stereo_calibration.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

namespace stereo_camera {
namespace {

constexpr int kPollTimeoutMs =
    1000; ///< 采集线程等待一帧的最长时间，单位为毫秒。
constexpr int kMinimumBufferCount = 2; ///< 允许配置的最小 mmap 缓冲区数量。
constexpr int kMaximumBufferCount = 32; ///< 允许配置的最大 mmap 缓冲区数量。
constexpr std::int64_t kNanosecondsPerSecond =
    1000000000LL; ///< 每秒包含的纳秒数。

/**
 * @brief 保存节点启动时固定的相机采集与 ROS 发布配置。
 */
struct CameraConfiguration {
  std::string device_path; ///< V4L2 图像设备路径。
  int capture_width = 0;   ///< 双目拼接帧宽度，单位为像素。
  int capture_height = 0;  ///< 双目拼接帧高度，单位为像素。
  int frame_rate = 0;      ///< 目标采集帧率，单位为 FPS。
  int buffer_count = 0;    ///< 向 V4L2 驱动申请的 mmap 缓冲区数量。
  int jpeg_quality = 0; ///< 左右目 JPEG 输出质量，允许范围为 1 到 100。
  std::string left_frame_id;  ///< 左目图像使用的 ROS 坐标系名称。
  std::string right_frame_id; ///< 右目图像使用的 ROS 坐标系名称。
  std::string calibration_file; ///< 统一双目 YAML 标定文件路径。
  int sensor_qos_depth = 0; ///< 图像和 CameraInfo 发布器的 KEEP_LAST 深度。
  std::string sensor_qos_reliability; ///< 传感器 QoS 可靠性名称。
};

/**
 * @brief 保存一块 V4L2 驱动内存映射区域。
 */
struct MappedBuffer {
  void *address = MAP_FAILED; ///< mmap 返回的用户空间起始地址。
  std::size_t length = 0U;    ///< 映射区域长度，单位为字节。
};

/**
 * @brief 保存一帧脱离驱动 mmap 生命周期的 MJPEG 数据及采集时间。
 */
struct CapturedFrame {
  std::vector<unsigned char> data; ///< MJPEG 压缩数据的独立副本。
  std::int64_t monotonic_timestamp_ns =
      0; ///< V4L2 单调时钟采集时间，单位为纳秒。
  bool timestamp_is_monotonic = false; ///< 驱动是否明确标记时间戳来自单调时钟。
};

/**
 * @brief 表示一次非阻塞帧读取的结果。
 */
enum class FrameReadResult {
  kSuccess, ///< 已取得有效 MJPEG 帧。
  kRetry,   ///< 当前无帧或等待超时，可继续读取。
  kInvalid, ///< 驱动返回了带错误标志或非法长度的帧。
};

/**
 * @brief 使用当前 `errno` 构造系统调用异常。
 * @param[in] operation 失败的系统调用或操作名称。
 * @return 包含系统错误文本的运行时异常。
 */
std::runtime_error SystemError(const std::string &operation) {
  /** 在其他库调用改变 `errno` 前保存原始错误码。 */
  const int error_number = errno;
  return std::runtime_error(operation + ": " + std::strerror(error_number));
}

/**
 * @brief 执行 ioctl，并在 `EINTR` 时自动重试。
 * @param[in] file_descriptor 目标设备文件描述符。
 * @param[in] request ioctl 请求编号。
 * @param[in,out] argument ioctl 参数与结果缓冲区。
 * @return ioctl 的原始返回值。
 */
int IoctlRetry(int file_descriptor, unsigned long request, void *argument) {
  /** 保存最近一次 ioctl 返回值。 */
  int result = -1;
  do {
    result = ioctl(file_descriptor, request, argument);
  } while (result < 0 && errno == EINTR);
  return result;
}

/**
 * @brief 获取 Linux `CLOCK_MONOTONIC` 当前时间。
 * @return 单调时钟纳秒计数。
 * @throws std::runtime_error 无法读取时钟时抛出。
 */
std::int64_t MonotonicNowNs() {
  /** 接收内核返回的单调时钟秒和纳秒字段。 */
  timespec timestamp{};
  if (clock_gettime(CLOCK_MONOTONIC, &timestamp) < 0) {
    throw SystemError("clock_gettime(CLOCK_MONOTONIC)");
  }
  return static_cast<std::int64_t>(timestamp.tv_sec) * kNanosecondsPerSecond +
         static_cast<std::int64_t>(timestamp.tv_nsec);
}

/**
 * @brief 创建只能在节点启动时赋值的参数描述符。
 * @param[in] description 参数用途说明。
 * @return 设置了只读标志的 ROS 参数描述符。
 */
rcl_interfaces::msg::ParameterDescriptor
ReadOnlyParameter(const std::string &description) {
  /** 保存待返回的参数约束和说明。 */
  rcl_interfaces::msg::ParameterDescriptor descriptor;
  descriptor.description = description;
  descriptor.read_only = true;
  return descriptor;
}

/**
 * @brief 声明、读取并校验节点配置参数。
 * @param[in,out] node 负责持有参数的 ROS 节点，不可为空。
 * @return 通过完整校验的启动配置。
 * @throws std::invalid_argument 参数取值不满足设备或 QoS 约束时抛出。
 */
CameraConfiguration ReadConfiguration(rclcpp::Node *node) {
  if (node == nullptr) {
    throw std::invalid_argument("节点指针不能为空");
  }

  /** 保存从 ROS 参数服务器读取的启动配置。 */
  CameraConfiguration configuration;
  configuration.device_path = node->declare_parameter<std::string>(
      "device_path", "/dev/video0", ReadOnlyParameter("V4L2 图像设备路径"));
  configuration.capture_width = node->declare_parameter<int>(
      "capture_width", 3840, ReadOnlyParameter("双目拼接帧宽度，单位为像素"));
  configuration.capture_height = node->declare_parameter<int>(
      "capture_height", 1080, ReadOnlyParameter("双目拼接帧高度，单位为像素"));
  configuration.frame_rate = node->declare_parameter<int>(
      "frame_rate", 50, ReadOnlyParameter("目标采集帧率，单位为 FPS"));
  configuration.buffer_count = node->declare_parameter<int>(
      "buffer_count", 8, ReadOnlyParameter("V4L2 mmap 缓冲区数量"));
  configuration.jpeg_quality = node->declare_parameter<int>(
      "jpeg_quality", 85,
      ReadOnlyParameter("左右目 JPEG 压缩质量，允许范围为 1 到 100"));
  configuration.left_frame_id = node->declare_parameter<std::string>(
      "left_frame_id", "stereo_camera_left_optical_frame",
      ReadOnlyParameter("左目图像 ROS 坐标系名称"));
  configuration.right_frame_id = node->declare_parameter<std::string>(
      "right_frame_id", "stereo_camera_right_optical_frame",
      ReadOnlyParameter("右目图像 ROS 坐标系名称"));
  configuration.calibration_file = node->declare_parameter<std::string>(
      "calibration_file", "",
      ReadOnlyParameter("统一双目鱼眼 YAML 标定文件路径"));
  configuration.sensor_qos_depth = node->declare_parameter<int>(
      "sensor_qos_depth", 5,
      ReadOnlyParameter("传感器 KEEP_LAST QoS 队列深度"));
  configuration.sensor_qos_reliability = node->declare_parameter<std::string>(
      "sensor_qos_reliability", "best_effort",
      ReadOnlyParameter("传感器 QoS 可靠性，可选 best_effort 或 reliable"));

  if (configuration.device_path.empty()) {
    throw std::invalid_argument("device_path 不能为空");
  }
  if (configuration.capture_width <= 0 ||
      configuration.capture_width % 2 != 0) {
    throw std::invalid_argument("capture_width 必须为正偶数");
  }
  if (configuration.capture_height <= 0) {
    throw std::invalid_argument("capture_height 必须为正数");
  }
  if (configuration.frame_rate <= 0) {
    throw std::invalid_argument("frame_rate 必须为正数");
  }
  if (configuration.buffer_count < kMinimumBufferCount ||
      configuration.buffer_count > kMaximumBufferCount) {
    throw std::invalid_argument("buffer_count 必须在 2 到 32 之间");
  }
  if (configuration.jpeg_quality < 1 || configuration.jpeg_quality > 100) {
    throw std::invalid_argument("jpeg_quality 必须在 1 到 100 之间");
  }
  if (configuration.left_frame_id.empty() ||
      configuration.right_frame_id.empty()) {
    throw std::invalid_argument("左右目 frame_id 均不能为空");
  }
  if (configuration.calibration_file.empty()) {
    throw std::invalid_argument("calibration_file 不能为空");
  }
  if (configuration.sensor_qos_depth < 1 ||
      configuration.sensor_qos_depth > 32) {
    throw std::invalid_argument("sensor_qos_depth 必须在 1 到 32 之间");
  }
  if (configuration.sensor_qos_reliability != "best_effort" &&
      configuration.sensor_qos_reliability != "reliable") {
    throw std::invalid_argument(
        "sensor_qos_reliability 必须为 best_effort 或 reliable");
  }
  return configuration;
}

/**
 * @brief 创建符合配置的传感器数据 QoS。
 * @param[in] configuration 已校验的节点配置。
 * @return 图像和 CameraInfo 发布器共用的 QoS。
 */
rclcpp::SensorDataQoS
CreateSensorQos(const CameraConfiguration &configuration) {
  /** 保存待配置的 ROS 传感器 QoS。 */
  rclcpp::SensorDataQoS qos;
  qos.keep_last(static_cast<std::size_t>(configuration.sensor_qos_depth));
  if (configuration.sensor_qos_reliability == "reliable") {
    qos.reliable();
  } else {
    qos.best_effort();
  }
  return qos;
}

/**
 * @class V4l2Camera
 * @brief 独占管理 V4L2 MJPEG 相机设备及 mmap 采集资源。
 * @details 实例只能由采集线程读取，析构时自动停止流并释放全部内核资源。
 */
class V4l2Camera {
public:
  /**
   * @brief 保存设备配置但暂不打开相机。
   * @param[in] configuration 已校验且在相机生命周期内保持不变的配置。
   */
  explicit V4l2Camera(const CameraConfiguration &configuration)
      : configuration_(configuration) {}

  /** @brief 停止流并释放 mmap 和文件描述符。 */
  ~V4l2Camera() { Close(); }

  V4l2Camera(const V4l2Camera &) = delete; ///< 禁止复制相机资源所有权。
  V4l2Camera &
  operator=(const V4l2Camera &) = delete; ///< 禁止复制赋值相机资源。

  /**
   * @brief 打开相机并完成能力、格式、帧率和 mmap 队列配置。
   * @throws std::runtime_error 任一设备配置步骤失败时抛出。
   */
  void OpenAndStart() {
    file_descriptor_ =
        open(configuration_.device_path.c_str(), O_RDWR | O_NONBLOCK);
    if (file_descriptor_ < 0) {
      throw SystemError("无法打开相机 " + configuration_.device_path);
    }

    try {
      ValidateCapabilities();
      ConfigureFormat();
      ConfigureFrameRate();
      AllocateBuffers();
      QueueAllBuffers();

      /** 指定待启动的 V4L2 视频采集缓冲区类型。 */
      v4l2_buf_type buffer_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      if (IoctlRetry(file_descriptor_, VIDIOC_STREAMON, &buffer_type) < 0) {
        throw SystemError("VIDIOC_STREAMON");
      }
      streaming_ = true;
    } catch (...) {
      Close();
      throw;
    }
  }

  /**
   * @brief 等待并复制一帧 MJPEG 数据，然后立即归还驱动缓冲区。
   * @param[out] output 接收帧数据和 V4L2 时间戳，不可为空。
   * @return 本次读取的状态，超时和错误帧允许调用方继续循环。
   * @throws std::runtime_error poll、出队或重新入队失败时抛出。
   */
  FrameReadResult ReadFrame(CapturedFrame *output) {
    if (output == nullptr) {
      throw std::invalid_argument("输出帧指针不能为空");
    }

    /** 配置 poll 需要监听的视频数据事件。 */
    pollfd descriptor{};
    descriptor.fd = file_descriptor_;
    descriptor.events = POLLIN | POLLPRI;
    /** 保存 poll 返回的就绪描述符数量。 */
    const int poll_result = poll(&descriptor, 1, kPollTimeoutMs);
    if (poll_result < 0) {
      if (errno == EINTR) {
        return FrameReadResult::kRetry;
      }
      throw SystemError("poll");
    }
    if (poll_result == 0) {
      return FrameReadResult::kRetry;
    }
    if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      throw std::runtime_error("相机设备 poll 返回错误事件");
    }

    /** 接收驱动返回的已填充 mmap 缓冲区描述。 */
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    if (IoctlRetry(file_descriptor_, VIDIOC_DQBUF, &buffer) < 0) {
      if (errno == EAGAIN) {
        return FrameReadResult::kRetry;
      }
      throw SystemError("VIDIOC_DQBUF");
    }

    /** 标记驱动返回的索引、长度和错误位是否允许安全复制。 */
    const bool frame_is_valid =
        buffer.index < buffers_.size() && buffer.bytesused > 0U &&
        buffer.bytesused <= buffers_[buffer.index].length &&
        (buffer.flags & V4L2_BUF_FLAG_ERROR) == 0U;
    if (frame_is_valid) {
      /** 指向当前 mmap 缓冲区内的 MJPEG 数据，重新入队后即失效。 */
      const auto *frame_begin =
          static_cast<const unsigned char *>(buffers_[buffer.index].address);
      output->data.assign(frame_begin, frame_begin + buffer.bytesused);
      output->monotonic_timestamp_ns =
          static_cast<std::int64_t>(buffer.timestamp.tv_sec) *
              kNanosecondsPerSecond +
          static_cast<std::int64_t>(buffer.timestamp.tv_usec) * 1000LL;
      output->timestamp_is_monotonic =
          (buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) ==
          V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC;
    } else {
      output->data.clear();
      output->monotonic_timestamp_ns = 0;
      output->timestamp_is_monotonic = false;
    }

    if (IoctlRetry(file_descriptor_, VIDIOC_QBUF, &buffer) < 0) {
      throw SystemError("VIDIOC_QBUF");
    }
    return frame_is_valid ? FrameReadResult::kSuccess
                          : FrameReadResult::kInvalid;
  }

private:
  /**
   * @brief 检查设备的视频采集和流式 I/O 能力。
   * @throws std::runtime_error 查询失败或能力不满足时抛出。
   */
  void ValidateCapabilities() const {
    /** 接收 V4L2 驱动公开的设备能力。 */
    v4l2_capability capabilities{};
    if (IoctlRetry(file_descriptor_, VIDIOC_QUERYCAP, &capabilities) < 0) {
      throw SystemError("VIDIOC_QUERYCAP");
    }
    /** 保存应检查的设备专属能力位集合。 */
    const std::uint32_t device_capabilities =
        (capabilities.capabilities & V4L2_CAP_DEVICE_CAPS) != 0U
            ? capabilities.device_caps
            : capabilities.capabilities;
    if ((device_capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0U) {
      throw std::runtime_error(configuration_.device_path +
                               " 不支持 V4L2 视频采集");
    }
    if ((device_capabilities & V4L2_CAP_STREAMING) == 0U) {
      throw std::runtime_error(configuration_.device_path +
                               " 不支持 V4L2 流式 I/O");
    }
  }

  /**
   * @brief 请求并严格验证 MJPEG 拼接帧格式。
   * @throws std::runtime_error 设置失败或驱动协商结果不一致时抛出。
   */
  void ConfigureFormat() const {
    /** 保存视频格式请求以及驱动协商后的实际结果。 */
    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width =
        static_cast<std::uint32_t>(configuration_.capture_width);
    format.fmt.pix.height =
        static_cast<std::uint32_t>(configuration_.capture_height);
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    format.fmt.pix.field = V4L2_FIELD_NONE;
    if (IoctlRetry(file_descriptor_, VIDIOC_S_FMT, &format) < 0) {
      throw SystemError("VIDIOC_S_FMT");
    }
    if (format.fmt.pix.width !=
            static_cast<std::uint32_t>(configuration_.capture_width) ||
        format.fmt.pix.height !=
            static_cast<std::uint32_t>(configuration_.capture_height) ||
        format.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
      throw std::runtime_error("相机未接受请求的 MJPEG 分辨率");
    }
  }

  /**
   * @brief 请求并严格验证目标采集帧率。
   * @throws std::runtime_error 设置失败或驱动返回不同帧率时抛出。
   */
  void ConfigureFrameRate() const {
    /** 保存目标帧率请求以及驱动返回的实际时间基。 */
    v4l2_streamparm parameters{};
    parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parameters.parm.capture.timeperframe.numerator = 1U;
    parameters.parm.capture.timeperframe.denominator =
        static_cast<std::uint32_t>(configuration_.frame_rate);
    if (IoctlRetry(file_descriptor_, VIDIOC_S_PARM, &parameters) < 0) {
      throw SystemError("VIDIOC_S_PARM");
    }

    /** 保存实际每帧时间的分子。 */
    const std::uint32_t numerator =
        parameters.parm.capture.timeperframe.numerator;
    /** 保存实际每帧时间的分母。 */
    const std::uint32_t denominator =
        parameters.parm.capture.timeperframe.denominator;
    if (numerator == 0U || denominator == 0U ||
        denominator !=
            static_cast<std::uint32_t>(configuration_.frame_rate) * numerator) {
      throw std::runtime_error("相机未接受请求的采集帧率");
    }
  }

  /**
   * @brief 申请并映射全部 V4L2 采集缓冲区。
   * @throws std::runtime_error 申请数量不足、查询或 mmap 失败时抛出。
   */
  void AllocateBuffers() {
    /** 保存 mmap 缓冲区申请参数及实际分配数量。 */
    v4l2_requestbuffers request{};
    request.count = static_cast<std::uint32_t>(configuration_.buffer_count);
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    if (IoctlRetry(file_descriptor_, VIDIOC_REQBUFS, &request) < 0) {
      throw SystemError("VIDIOC_REQBUFS");
    }
    if (request.count < static_cast<std::uint32_t>(kMinimumBufferCount)) {
      throw std::runtime_error("驱动提供的 mmap 缓冲区少于 2 个");
    }

    buffers_.resize(request.count);
    /** 遍历驱动分配的缓冲区并映射到当前进程。 */
    for (std::uint32_t index = 0U; index < request.count; ++index) {
      /** 保存当前缓冲区的驱动描述信息。 */
      v4l2_buffer buffer{};
      buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buffer.memory = V4L2_MEMORY_MMAP;
      buffer.index = index;
      if (IoctlRetry(file_descriptor_, VIDIOC_QUERYBUF, &buffer) < 0) {
        throw SystemError("VIDIOC_QUERYBUF");
      }
      buffers_[index].length = buffer.length;
      buffers_[index].address =
          mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED,
               file_descriptor_, buffer.m.offset);
      if (buffers_[index].address == MAP_FAILED) {
        throw SystemError("mmap");
      }
    }
  }

  /**
   * @brief 将所有已映射缓冲区加入 V4L2 采集队列。
   * @throws std::runtime_error 任一缓冲区入队失败时抛出。
   */
  void QueueAllBuffers() const {
    /** 遍历全部 mmap 缓冲区索引并逐个入队。 */
    for (std::size_t index = 0U; index < buffers_.size(); ++index) {
      /** 保存当前待入队缓冲区的索引和内存类型。 */
      v4l2_buffer buffer{};
      buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buffer.memory = V4L2_MEMORY_MMAP;
      buffer.index = static_cast<std::uint32_t>(index);
      if (IoctlRetry(file_descriptor_, VIDIOC_QBUF, &buffer) < 0) {
        throw SystemError("VIDIOC_QBUF");
      }
    }
  }

  /**
   * @brief 停止视频流、解除全部映射并关闭设备。
   * @note 可重复调用，析构期间只记录到标准错误且不抛异常。
   */
  void Close() noexcept {
    if (streaming_ && file_descriptor_ >= 0) {
      /** 指定需要停止的 V4L2 视频采集缓冲区类型。 */
      v4l2_buf_type buffer_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      static_cast<void>(
          ioctl(file_descriptor_, VIDIOC_STREAMOFF, &buffer_type));
      streaming_ = false;
    }
    /** 遍历所有成功建立的映射并释放用户空间地址。 */
    for (MappedBuffer &buffer : buffers_) {
      if (buffer.address != MAP_FAILED) {
        static_cast<void>(munmap(buffer.address, buffer.length));
        buffer.address = MAP_FAILED;
        buffer.length = 0U;
      }
    }
    buffers_.clear();
    if (file_descriptor_ >= 0) {
      static_cast<void>(close(file_descriptor_));
      file_descriptor_ = -1;
    }
  }

  CameraConfiguration configuration_; ///< 当前设备使用的不可变启动配置副本。
  int file_descriptor_ = -1; ///< V4L2 设备描述符，-1 表示尚未打开。
  bool streaming_ = false;   ///< 是否已成功执行 `VIDIOC_STREAMON`。
  std::vector<MappedBuffer> buffers_; ///< 由当前对象独占的 mmap 缓冲区。
};

/**
 * @brief 从拼接 BGR 图像的指定半幅构造 ROS JPEG 压缩图像消息。
 * @param[in] stereo_frame 已解码的连续或跨步 BGR 拼接帧。
 * @param[in] right_eye 为 `true` 时复制右半幅，否则复制左半幅。
 * @param[in] stamp 当前双目帧共用的 ROS 时间戳。
 * @param[in] frame_id 当前单目图像使用的坐标系名称。
 * @param[in] jpeg_quality JPEG 编码质量，允许范围为 1 到 100。
 * @param[out] message 接收拥有独立 JPEG 数据的压缩图像消息，不可为空。
 * @return JPEG 编码成功时返回 `true`，编码器拒绝当前图像时返回 `false`。
 * @throws std::runtime_error 输入图像尺寸或通道数不满足双目拆分要求时抛出。
 */
bool BuildCompressedImageMessage(const cv::Mat &stereo_frame, bool right_eye,
                                 const builtin_interfaces::msg::Time &stamp,
                                 const std::string &frame_id, int jpeg_quality,
                                 sensor_msgs::msg::CompressedImage *message) {
  if (message == nullptr) {
    throw std::invalid_argument("压缩图像输出消息指针不能为空");
  }
  if (stereo_frame.empty() || stereo_frame.type() != CV_8UC3 ||
      stereo_frame.cols % 2 != 0) {
    throw std::runtime_error("无法从当前 OpenCV 图像构造双目 JPEG 消息");
  }

  /** 保存单目图像宽度，单位为像素。 */
  const int eye_width = stereo_frame.cols / 2;
  /** 保存当前单目在拼接帧中的水平起始像素。 */
  const int column_offset = right_eye ? eye_width : 0;
  /** 引用当前拼接帧中的目标单目区域，不复制未压缩像素。 */
  const cv::Mat eye_frame =
      stereo_frame(cv::Rect(column_offset, 0, eye_width, stereo_frame.rows));
  /** 保存 OpenCV JPEG 编码器参数。 */
  const std::vector<int> encoding_parameters{cv::IMWRITE_JPEG_QUALITY,
                                             jpeg_quality};

  message->header.stamp = stamp;
  message->header.frame_id = frame_id;
  message->format = "bgr8; jpeg compressed bgr8";
  try {
    return cv::imencode(".jpg", eye_frame, message->data, encoding_parameters);
  } catch (const cv::Exception &) {
    message->data.clear();
    return false;
  }
}

} // namespace

/**
 * @class StereoCameraNode
 * @brief 采集 UVC 双目拼接帧并发布左右目 ROS 图像和标定信息。
 * @details 节点独占 V4L2 设备与采集线程；发布器可由采集线程安全调用。
 */
class StereoCameraNode final : public rclcpp::Node {
public:
  /**
   * @brief 声明参数、加载标定、创建发布器并启动相机采集线程。
   * @throws std::exception 参数、标定文件或相机初始化失败时抛出。
   */
  StereoCameraNode() : Node("stereo_camera_node") {
    configuration_ = ReadConfiguration(this);
    /** 保存单目输出宽度，单位为像素。 */
    const std::uint32_t eye_width =
        static_cast<std::uint32_t>(configuration_.capture_width / 2);
    /** 保存单目输出高度，单位为像素。 */
    const std::uint32_t eye_height =
        static_cast<std::uint32_t>(configuration_.capture_height);
    /** 保存统一 YAML 文件加载出的双目标定。 */
    const StereoCalibration calibration = LoadStereoCalibration(
        configuration_.calibration_file, eye_width, eye_height);
    /** 保存由鱼眼双目校正计算得到的左右 CameraInfo 模板。 */
    auto camera_info = BuildStereoCameraInfo(calibration);
    left_camera_info_ = std::move(camera_info.first);
    right_camera_info_ = std::move(camera_info.second);

    /** 保存四个传感器发布器共用的 QoS 配置。 */
    const rclcpp::SensorDataQoS sensor_qos = CreateSensorQos(configuration_);
    left_image_publisher_ =
        this->create_publisher<sensor_msgs::msg::CompressedImage>(
            "left/image_raw/compressed", sensor_qos);
    right_image_publisher_ =
        this->create_publisher<sensor_msgs::msg::CompressedImage>(
            "right/image_raw/compressed", sensor_qos);
    left_info_publisher_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
        "left/camera_info", sensor_qos);
    right_info_publisher_ =
        this->create_publisher<sensor_msgs::msg::CameraInfo>(
            "right/camera_info", sensor_qos);

    camera_ = std::make_unique<V4l2Camera>(configuration_);
    camera_->OpenAndStart();
    static_tf_broadcaster_ =
        std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
    /** 保存右相机在左相机坐标系中的静态位姿。 */
    const geometry_msgs::msg::TransformStamped right_camera_transform =
        BuildRightCameraTransform(calibration, this->now(),
                                  configuration_.left_frame_id,
                                  configuration_.right_frame_id);
    static_tf_broadcaster_->sendTransform(right_camera_transform);
    monotonic_anchor_ns_ = MonotonicNowNs();
    system_anchor_ns_ = rclcpp::Clock(RCL_SYSTEM_TIME).now().nanoseconds();
    running_.store(true);
    capture_thread_ = std::thread(&StereoCameraNode::CaptureLoop, this);

    RCLCPP_INFO(this->get_logger(),
                "双目相机已启动：device=%s format=%dx%d MJPG fps=%d eye=%ux%u "
                "calibration=%s",
                configuration_.device_path.c_str(),
                configuration_.capture_width, configuration_.capture_height,
                configuration_.frame_rate, eye_width, eye_height,
                configuration_.calibration_file.c_str());
  }

  /** @brief 请求采集线程退出并等待相机资源安全释放。 */
  ~StereoCameraNode() override {
    running_.store(false);
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }
    camera_.reset();
  }

private:
  /**
   * @brief 将 V4L2 单调时钟时间映射为 ROS 系统时间。
   * @param[in] frame 待转换时间戳的已采集帧。
   * @return 可同时用于左右目消息的 ROS 时间戳。
   */
  builtin_interfaces::msg::Time MapTimestamp(const CapturedFrame &frame) const {
    if (!frame.timestamp_is_monotonic || frame.monotonic_timestamp_ns <= 0) {
      return rclcpp::Clock(RCL_SYSTEM_TIME).now();
    }
    /** 保存依据启动锚点换算出的系统时钟纳秒值。 */
    const std::int64_t mapped_timestamp_ns =
        system_anchor_ns_ +
        (frame.monotonic_timestamp_ns - monotonic_anchor_ns_);
    return rclcpp::Time(mapped_timestamp_ns, RCL_SYSTEM_TIME);
  }

  /**
   * @brief 编码并发布一对时间戳严格一致的左右目压缩图像与 CameraInfo。
   * @param[in] stereo_frame 已解码且尺寸经过验证的 BGR 拼接帧。
   * @param[in] stamp 当前硬件双目帧对应的 ROS 时间戳。
   * @return 左右 JPEG 均编码并发布成功时返回 `true`，否则返回 `false`。
   */
  bool PublishStereoFrame(const cv::Mat &stereo_frame,
                          const builtin_interfaces::msg::Time &stamp) {
    /** 保存待发布的左目 JPEG 压缩图像消息。 */
    sensor_msgs::msg::CompressedImage left_image;
    /** 保存待发布的右目 JPEG 压缩图像消息。 */
    sensor_msgs::msg::CompressedImage right_image;
    if (!BuildCompressedImageMessage(
            stereo_frame, false, stamp, configuration_.left_frame_id,
            configuration_.jpeg_quality, &left_image) ||
        !BuildCompressedImageMessage(
            stereo_frame, true, stamp, configuration_.right_frame_id,
            configuration_.jpeg_quality, &right_image)) {
      return false;
    }
    /** 复制左目标定模板并填入当前帧头。 */
    sensor_msgs::msg::CameraInfo left_info = left_camera_info_;
    /** 复制右目标定模板并填入当前帧头。 */
    sensor_msgs::msg::CameraInfo right_info = right_camera_info_;
    left_info.header = left_image.header;
    right_info.header = right_image.header;

    left_image_publisher_->publish(std::move(left_image));
    left_info_publisher_->publish(std::move(left_info));
    right_image_publisher_->publish(std::move(right_image));
    right_info_publisher_->publish(std::move(right_info));
    return true;
  }

  /**
   * @brief 持续读取、解码、校验并发布双目帧。
   * @note 运行在独立线程；不可恢复异常会触发全局 ROS shutdown。
   */
  void CaptureLoop() noexcept {
    try {
      /** 保存每次从相机取得的独立 MJPEG 数据。 */
      CapturedFrame captured_frame;
      while (running_.load() && rclcpp::ok()) {
        /** 保存当前非阻塞读取操作的结果。 */
        const FrameReadResult read_result = camera_->ReadFrame(&captured_frame);
        if (read_result == FrameReadResult::kRetry) {
          continue;
        }
        if (read_result == FrameReadResult::kInvalid) {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                               "跳过 V4L2 驱动标记的错误帧");
          continue;
        }

        /** 保存 OpenCV 解码后的 3840x1080 BGR 拼接帧。 */
        const cv::Mat stereo_frame =
            cv::imdecode(captured_frame.data, cv::IMREAD_COLOR);
        if (stereo_frame.empty()) {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                               "跳过无法解码的 MJPEG 帧");
          continue;
        }
        if (stereo_frame.cols != configuration_.capture_width ||
            stereo_frame.rows != configuration_.capture_height ||
            stereo_frame.type() != CV_8UC3) {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                               "跳过尺寸或类型异常的解码帧：%dx%d type=%d",
                               stereo_frame.cols, stereo_frame.rows,
                               stereo_frame.type());
          continue;
        }

        /** 保存当前左右目消息共用的映射后系统时间戳。 */
        const builtin_interfaces::msg::Time stamp =
            MapTimestamp(captured_frame);
        if (!PublishStereoFrame(stereo_frame, stamp)) {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                               "跳过左右目 JPEG 编码不完整的双目帧");
        }
      }
    } catch (const std::exception &error) {
      RCLCPP_FATAL(this->get_logger(), "相机采集线程异常：%s", error.what());
      running_.store(false);
      rclcpp::shutdown();
    }
  }

  CameraConfiguration
      configuration_; ///< 节点生命周期内不可变的采集和发布配置。
  sensor_msgs::msg::CameraInfo left_camera_info_; ///< 左目标定信息模板。
  sensor_msgs::msg::CameraInfo right_camera_info_; ///< 右目标定信息模板。
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr
      left_image_publisher_; ///< 左目 JPEG 压缩图像发布器。
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr
      right_image_publisher_; ///< 右目 JPEG 压缩图像发布器。
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr
      left_info_publisher_; ///< 左目 CameraInfo 发布器。
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr
      right_info_publisher_; ///< 右目 CameraInfo 发布器。
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster>
      static_tf_broadcaster_; ///< 左目到右目的静态 TF 广播器。
  std::unique_ptr<V4l2Camera> camera_; ///< 当前节点独占的 V4L2 相机资源。
  std::thread capture_thread_; ///< 执行阻塞采集、解码和发布工作的线程。
  std::atomic<bool> running_{false}; ///< 控制采集线程退出的原子标志。
  std::int64_t monotonic_anchor_ns_ = 0; ///< 启动时单调时钟锚点，单位为纳秒。
  std::int64_t system_anchor_ns_ = 0; ///< 与单调锚点配对的 ROS 系统时钟纳秒值。
};

} // namespace stereo_camera

/**
 * @brief 初始化 ROS 2 并运行双目相机节点。
 * @param[in] argc 命令行参数数量。
 * @param[in] argv 命令行参数数组，所有权属于运行时。
 * @return 正常退出时返回 0，节点初始化或运行失败时返回 1。
 */
int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  try {
    /** 保存 ROS 2 双目相机节点实例。 */
    auto node = std::make_shared<stereo_camera::StereoCameraNode>();
    rclcpp::spin(node);
    node.reset();
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    return 0;
  } catch (const std::exception &error) {
    RCLCPP_FATAL(rclcpp::get_logger("stereo_camera_node"), "%s", error.what());
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    return 1;
  }
}
