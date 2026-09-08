/**
 * @file stereo_viewer.cpp
 * @brief 实现基于 V4L2 和 OpenCV 的双目相机实时预览程序。
 * @details 作为 example 示例从 UVC 相机采集 3840x1080 MJPEG 拼接帧，解码后分别显示左右目画面。
 * @author 待确认
 * @date 创建：2026-09-07
 * @date 修改：2026-09-07
 */

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <getopt.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr std::uint32_t kFrameWidth = 3840U;       ///< 双目拼接帧宽度，单位为像素。
constexpr std::uint32_t kFrameHeight = 1080U;      ///< 双目拼接帧高度，单位为像素。
constexpr std::uint32_t kFrameRate = 50U;          ///< 相机目标采集帧率，单位为 FPS。
constexpr std::uint32_t kBufferCount = 8U;         ///< V4L2 mmap 缓冲区申请数量。
constexpr int kPollTimeoutMs = 1000;               ///< 等待相机帧的单次超时时间，单位为毫秒。
constexpr int kPreviewWidth = 960;                 ///< 单个预览窗口的初始宽度，单位为像素。
constexpr int kPreviewHeight = 540;                ///< 单个预览窗口的初始高度，单位为像素。
constexpr int kEscapeKey = 27;                     ///< Esc 键的键值。
constexpr double kFpsUpdateSeconds = 1.0;           ///< 平均 FPS 的统计更新周期，单位为秒。
constexpr char kLeftWindowName[] = "Left Camera"; ///< 左目预览窗口名称。
constexpr char kRightWindowName[] = "Right Camera"; ///< 右目预览窗口名称。

volatile std::sig_atomic_t g_running = 1; ///< 主循环运行标志，由信号处理函数清零。

/**
 * @brief 保存命令行参数解析结果。
 */
struct ProgramOptions {
  std::string device_path = "/dev/video0"; ///< 待打开的 V4L2 图像设备路径。
  bool show_help = false;                   ///< 是否只显示命令行帮助。
};

/**
 * @brief 描述一块由 V4L2 驱动分配并映射到用户空间的缓冲区。
 */
struct MappedBuffer {
  void *address = MAP_FAILED; ///< mmap 返回的缓冲区起始地址。
  std::size_t length = 0U;    ///< 缓冲区映射长度，单位为字节。
};

/**
 * @brief 处理退出信号并请求主循环停止。
 * @param[in] signal_number 收到的 POSIX 信号编号，本函数不区分具体信号。
 * @note 本函数只修改 `sig_atomic_t` 标志，可安全用于异步信号上下文。
 */
void HandleSignal(int signal_number) {
  static_cast<void>(signal_number);
  g_running = 0;
}

/**
 * @brief 使用当前 `errno` 构造并抛出系统调用异常。
 * @param[in] operation 发生错误的系统调用或操作名称。
 * @throws std::runtime_error 始终抛出，消息包含 `errno` 对应的文本。
 */
[[noreturn]] void ThrowSystemError(const std::string &operation) {
  /** 在后续库调用改变 `errno` 前保存原始错误码。 */
  const int error_number = errno;
  throw std::runtime_error(operation + ": " + std::strerror(error_number));
}

/**
 * @brief 执行 ioctl，并在被信号中断时自动重试。
 * @param[in] file_descriptor 目标设备文件描述符。
 * @param[in] request ioctl 请求编号。
 * @param[in,out] argument ioctl 请求参数及结果缓冲区。
 * @return ioctl 的返回值。
 */
int IoctlRetry(int file_descriptor, unsigned long request, void *argument) {
  /** 保存最近一次 ioctl 的返回值。 */
  int result = -1;
  do {
    result = ioctl(file_descriptor, request, argument);
  } while (result < 0 && errno == EINTR && g_running != 0);
  return result;
}

/**
 * @brief 将 V4L2 FourCC 数值转换为四字符字符串。
 * @param[in] pixel_format V4L2 像素格式 FourCC 数值。
 * @return 对应的四字符字符串。
 */
std::string PixelFormatToString(std::uint32_t pixel_format) {
  /** 保存 FourCC 四个字符及字符串结束符。 */
  char characters[5] = {
      static_cast<char>(pixel_format & 0xffU),
      static_cast<char>((pixel_format >> 8U) & 0xffU),
      static_cast<char>((pixel_format >> 16U) & 0xffU),
      static_cast<char>((pixel_format >> 24U) & 0xffU), '\0'};
  return std::string(characters);
}

/**
 * @brief 打印程序命令行帮助。
 * @param[in] program_name 当前可执行文件名称。
 */
void PrintUsage(const char *program_name) {
  std::cout << "Usage: " << program_name << " [-d /dev/videoN] [-h]\n"
            << "  -d PATH  V4L2 video capture device (default: /dev/video0)\n"
            << "  -h       Show this help message\n"
            << "Press q, Esc, or Ctrl+C to exit.\n";
}

/**
 * @brief 解析命令行参数。
 * @param[in] argc 命令行参数数量。
 * @param[in] argv 命令行参数数组，元素由运行时管理。
 * @return 解析后的程序选项。
 * @throws std::invalid_argument 参数未知、缺失或设备路径为空时抛出。
 */
ProgramOptions ParseOptions(int argc, char *argv[]) {
  /** 保存解析得到的程序选项。 */
  ProgramOptions options;
  /** 保存 getopt 当前返回的选项字符。 */
  int option = 0;

  opterr = 0;
  while ((option = getopt(argc, argv, "d:h")) != -1) {
    switch (option) {
    case 'd':
      options.device_path = optarg;
      break;
    case 'h':
      options.show_help = true;
      break;
    default:
      throw std::invalid_argument("未知或不完整的命令行参数");
    }
  }

  if (optind != argc) {
    throw std::invalid_argument("存在未识别的位置参数");
  }
  if (options.device_path.empty()) {
    throw std::invalid_argument("相机设备路径不能为空");
  }
  return options;
}

/**
 * @brief 判断当前进程是否处于可创建图形窗口的环境。
 * @return 存在 X11 或 Wayland 显示环境变量时返回 `true`，否则返回 `false`。
 */
bool HasGraphicalDisplay() {
  /** 指向 X11 显示地址环境变量，所有权属于 C 运行库。 */
  const char *display = std::getenv("DISPLAY");
  /** 指向 Wayland 显示名称环境变量，所有权属于 C 运行库。 */
  const char *wayland_display = std::getenv("WAYLAND_DISPLAY");
  return (display != nullptr && display[0] != '\0') ||
         (wayland_display != nullptr && wayland_display[0] != '\0');
}

/**
 * @class V4l2Camera
 * @brief 管理单个 V4L2 MJPEG 相机的配置、mmap 缓冲区和采集生命周期。
 * @details 对设备文件描述符和映射内存拥有独占所有权，不允许复制。
 */
class V4l2Camera {
public:
  /**
   * @brief 构造尚未打开设备的相机对象。
   * @param[in] device_path 待打开的 V4L2 图像设备路径。
   */
  explicit V4l2Camera(std::string device_path)
      : device_path_(std::move(device_path)) {}

  /** @brief 停止视频流并释放所有 V4L2 资源。 */
  ~V4l2Camera() { Close(); }

  V4l2Camera(const V4l2Camera &) = delete; ///< 禁止复制设备资源所有权。
  V4l2Camera &operator=(const V4l2Camera &) = delete; ///< 禁止复制赋值设备资源。

  /**
   * @brief 打开设备并配置目标采集格式、帧率和 mmap 缓冲区。
   * @throws std::runtime_error 设备能力或任意 V4L2 配置步骤不满足要求时抛出。
   */
  void OpenAndStart() {
    file_descriptor_ = open(device_path_.c_str(), O_RDWR | O_NONBLOCK);
    if (file_descriptor_ < 0) {
      ThrowSystemError("无法打开相机 " + device_path_);
    }

    try {
      ValidateCapabilities();
      ConfigureFormat();
      ConfigureFrameRate();
      AllocateBuffers();
      QueueAllBuffers();

      /** 指定需要启动的视频采集缓冲区类型。 */
      v4l2_buf_type buffer_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      if (IoctlRetry(file_descriptor_, VIDIOC_STREAMON, &buffer_type) < 0) {
        ThrowSystemError("VIDIOC_STREAMON");
      }
      streaming_ = true;
    } catch (...) {
      Close();
      throw;
    }
  }

  /**
   * @brief 等待并读取一帧完整 MJPEG 数据。
   * @param[out] compressed_frame 接收压缩帧副本，返回后不依赖 mmap 缓冲区。
   * @return 成功取得可解码帧时返回 `true`；超时、可重试或驱动标记错误时返回 `false`。
   * @throws std::runtime_error poll 或 V4L2 出队、重新入队失败时抛出。
   */
  bool ReadCompressedFrame(std::vector<unsigned char> *compressed_frame) {
    if (compressed_frame == nullptr) {
      throw std::invalid_argument("输出帧指针不能为空");
    }

    /** 配置等待当前相机文件描述符上的图像数据。 */
    pollfd descriptor{};
    descriptor.fd = file_descriptor_;
    descriptor.events = POLLIN | POLLPRI;

    /** 保存 poll 返回的就绪描述符数量。 */
    const int poll_result = poll(&descriptor, 1, kPollTimeoutMs);
    if (poll_result < 0) {
      if (errno == EINTR && g_running == 0) {
        return false;
      }
      ThrowSystemError("poll");
    }
    if (poll_result == 0) {
      std::cerr << "警告：等待相机帧超时\n";
      return false;
    }
    if ((descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
      throw std::runtime_error("相机设备 poll 返回错误事件");
    }

    /** 接收驱动返回的已填充缓冲区描述。 */
    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    if (IoctlRetry(file_descriptor_, VIDIOC_DQBUF, &buffer) < 0) {
      if (errno == EAGAIN) {
        return false;
      }
      ThrowSystemError("VIDIOC_DQBUF");
    }

    /** 标记当前帧是否满足复制和解码条件。 */
    const bool frame_is_valid =
        buffer.index < buffers_.size() && buffer.bytesused > 0U &&
        buffer.bytesused <= buffers_[buffer.index].length &&
        (buffer.flags & V4L2_BUF_FLAG_ERROR) == 0U;

    if (frame_is_valid) {
      /** 指向驱动填充的 MJPEG 数据，仅在重新入队前有效。 */
      const auto *frame_begin = static_cast<const unsigned char *>(
          buffers_[buffer.index].address);
      compressed_frame->assign(frame_begin, frame_begin + buffer.bytesused);
    } else {
      compressed_frame->clear();
    }

    if (IoctlRetry(file_descriptor_, VIDIOC_QBUF, &buffer) < 0) {
      ThrowSystemError("VIDIOC_QBUF");
    }
    return frame_is_valid;
  }

private:
  /**
   * @brief 检查设备是否支持视频采集和流式 mmap I/O。
   * @throws std::runtime_error 查询失败或能力不满足时抛出。
   */
  void ValidateCapabilities() const {
    /** 接收设备公开的 V4L2 能力集合。 */
    v4l2_capability capabilities{};
    if (IoctlRetry(file_descriptor_, VIDIOC_QUERYCAP, &capabilities) < 0) {
      ThrowSystemError("VIDIOC_QUERYCAP");
    }

    /** 保存应检查的设备能力位；支持 DEVICE_CAPS 时使用其专属集合。 */
    const std::uint32_t device_capabilities =
        (capabilities.capabilities & V4L2_CAP_DEVICE_CAPS) != 0U
            ? capabilities.device_caps
            : capabilities.capabilities;
    if ((device_capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0U) {
      throw std::runtime_error(device_path_ + " 不支持 V4L2 视频采集");
    }
    if ((device_capabilities & V4L2_CAP_STREAMING) == 0U) {
      throw std::runtime_error(device_path_ + " 不支持 V4L2 流式 I/O");
    }
  }

  /**
   * @brief 请求并验证 3840x1080 MJPEG 视频格式。
   * @throws std::runtime_error 格式设置失败或驱动协商结果不匹配时抛出。
   */
  void ConfigureFormat() const {
    /** 保存视频格式请求及驱动返回的实际格式。 */
    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = kFrameWidth;
    format.fmt.pix.height = kFrameHeight;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    format.fmt.pix.field = V4L2_FIELD_NONE;

    if (IoctlRetry(file_descriptor_, VIDIOC_S_FMT, &format) < 0) {
      ThrowSystemError("VIDIOC_S_FMT");
    }
    if (format.fmt.pix.width != kFrameWidth ||
        format.fmt.pix.height != kFrameHeight ||
        format.fmt.pix.pixelformat != V4L2_PIX_FMT_MJPEG) {
      throw std::runtime_error(
          "相机格式协商失败，实际格式为 " +
          std::to_string(format.fmt.pix.width) + "x" +
          std::to_string(format.fmt.pix.height) + " " +
          PixelFormatToString(format.fmt.pix.pixelformat));
    }

    std::cout << "视频格式：" << format.fmt.pix.width << 'x'
              << format.fmt.pix.height << ' '
              << PixelFormatToString(format.fmt.pix.pixelformat) << '\n';
  }

  /**
   * @brief 请求 50 FPS 采集速率并验证驱动返回值。
   * @throws std::runtime_error 帧率设置失败或实际帧率不等于 50 FPS 时抛出。
   */
  void ConfigureFrameRate() const {
    /** 保存视频流参数请求及驱动返回的实际时间基。 */
    v4l2_streamparm parameters{};
    parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parameters.parm.capture.timeperframe.numerator = 1U;
    parameters.parm.capture.timeperframe.denominator = kFrameRate;

    if (IoctlRetry(file_descriptor_, VIDIOC_S_PARM, &parameters) < 0) {
      ThrowSystemError("VIDIOC_S_PARM");
    }

    /** 保存驱动返回的每帧时间分子。 */
    const std::uint32_t numerator =
        parameters.parm.capture.timeperframe.numerator;
    /** 保存驱动返回的每帧时间分母。 */
    const std::uint32_t denominator =
        parameters.parm.capture.timeperframe.denominator;
    if (numerator == 0U || denominator == 0U ||
        denominator != kFrameRate * numerator) {
      throw std::runtime_error(
          "相机帧率协商失败，实际时间基为 " +
          std::to_string(numerator) + "/" + std::to_string(denominator));
    }
    std::cout << "采集帧率：" << denominator / numerator << " FPS\n";
  }

  /**
   * @brief 向驱动申请并映射采集缓冲区。
   * @throws std::runtime_error 缓冲区数量不足、查询或 mmap 失败时抛出。
   */
  void AllocateBuffers() {
    /** 保存 V4L2 mmap 缓冲区申请参数。 */
    v4l2_requestbuffers request{};
    request.count = kBufferCount;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    if (IoctlRetry(file_descriptor_, VIDIOC_REQBUFS, &request) < 0) {
      ThrowSystemError("VIDIOC_REQBUFS");
    }
    if (request.count < 2U) {
      throw std::runtime_error("驱动提供的 V4L2 缓冲区少于 2 个");
    }

    buffers_.resize(request.count);
    /** 遍历驱动分配的所有缓冲区并建立用户空间映射。 */
    for (std::uint32_t index = 0U; index < request.count; ++index) {
      /** 保存当前待查询的 V4L2 缓冲区描述。 */
      v4l2_buffer buffer{};
      buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buffer.memory = V4L2_MEMORY_MMAP;
      buffer.index = index;
      if (IoctlRetry(file_descriptor_, VIDIOC_QUERYBUF, &buffer) < 0) {
        ThrowSystemError("VIDIOC_QUERYBUF");
      }

      buffers_[index].length = buffer.length;
      buffers_[index].address =
          mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED,
               file_descriptor_, buffer.m.offset);
      if (buffers_[index].address == MAP_FAILED) {
        ThrowSystemError("mmap");
      }
    }
  }

  /**
   * @brief 将全部 mmap 缓冲区加入驱动采集队列。
   * @throws std::runtime_error 任一缓冲区入队失败时抛出。
   */
  void QueueAllBuffers() const {
    /** 遍历全部映射缓冲区并按索引加入驱动队列。 */
    for (std::size_t index = 0U; index < buffers_.size(); ++index) {
      /** 保存当前待入队的 V4L2 缓冲区描述。 */
      v4l2_buffer buffer{};
      buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buffer.memory = V4L2_MEMORY_MMAP;
      buffer.index = static_cast<std::uint32_t>(index);
      if (IoctlRetry(file_descriptor_, VIDIOC_QBUF, &buffer) < 0) {
        ThrowSystemError("VIDIOC_QBUF");
      }
    }
  }

  /**
   * @brief 停止采集并释放映射内存和设备描述符。
   * @note 可重复调用，析构阶段不会抛出异常。
   */
  void Close() noexcept {
    if (streaming_ && file_descriptor_ >= 0) {
      /** 指定需要停止的视频采集缓冲区类型。 */
      v4l2_buf_type buffer_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      if (ioctl(file_descriptor_, VIDIOC_STREAMOFF, &buffer_type) < 0) {
        std::cerr << "警告：VIDIOC_STREAMOFF 失败："
                  << std::strerror(errno) << '\n';
      }
      streaming_ = false;
    }

    /** 遍历所有已映射缓冲区并解除映射。 */
    for (MappedBuffer &buffer : buffers_) {
      if (buffer.address != MAP_FAILED) {
        if (munmap(buffer.address, buffer.length) < 0) {
          std::cerr << "警告：munmap 失败：" << std::strerror(errno) << '\n';
        }
        buffer.address = MAP_FAILED;
        buffer.length = 0U;
      }
    }
    buffers_.clear();

    if (file_descriptor_ >= 0) {
      if (close(file_descriptor_) < 0) {
        std::cerr << "警告：close 失败：" << std::strerror(errno) << '\n';
      }
      file_descriptor_ = -1;
    }
  }

  std::string device_path_;             ///< 相机图像设备路径。
  int file_descriptor_ = -1;            ///< 已打开的设备描述符，-1 表示未打开。
  bool streaming_ = false;              ///< 是否已经成功执行 STREAMON。
  std::vector<MappedBuffer> buffers_;   ///< 当前设备拥有的 mmap 缓冲区集合。
};

/**
 * @brief 创建并排列左右目预览窗口。
 */
void CreatePreviewWindows() {
  cv::namedWindow(kLeftWindowName, cv::WINDOW_NORMAL);
  cv::namedWindow(kRightWindowName, cv::WINDOW_NORMAL);
  cv::resizeWindow(kLeftWindowName, kPreviewWidth, kPreviewHeight);
  cv::resizeWindow(kRightWindowName, kPreviewWidth, kPreviewHeight);
  cv::moveWindow(kLeftWindowName, 0, 0);
  cv::moveWindow(kRightWindowName, kPreviewWidth, 0);
}

/**
 * @brief 在单目画面顶部绘制实时平均 FPS。
 * @param[in,out] frame 接收文字叠加的 BGR 图像，必须具有可写像素数据。
 * @param[in] average_fps 最近统计周期内的平均显示帧率。
 */
void DrawFpsOverlay(cv::Mat *frame, double average_fps) {
  if (frame == nullptr || frame->empty()) {
    return;
  }

  /** 组装保留一位小数的 FPS 显示文本。 */
  std::ostringstream text_stream;
  text_stream << "AVG FPS: " << std::fixed << std::setprecision(1)
              << average_fps;
  /** 保存最终绘制到画面上的 FPS 文本。 */
  const std::string text = text_stream.str();
  /** 设置 FPS 文字在原始单目画面中的左上角基线位置。 */
  const cv::Point origin(30, 55);

  cv::putText(*frame, text, origin, cv::FONT_HERSHEY_SIMPLEX, 1.2,
              cv::Scalar(0, 0, 0), 5, cv::LINE_AA);
  cv::putText(*frame, text, origin, cv::FONT_HERSHEY_SIMPLEX, 1.2,
              cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}

/**
 * @brief 运行相机采集、MJPEG 解码和双窗口显示主循环。
 * @param[in] device_path 待打开的 V4L2 图像设备路径。
 * @return 用户正常退出时返回 0。
 * @throws std::runtime_error 相机、图像格式或显示操作失败时抛出。
 */
int RunViewer(const std::string &device_path) {
  /** 管理当前相机的 V4L2 资源和采集状态。 */
  V4l2Camera camera(device_path);
  camera.OpenAndStart();
  CreatePreviewWindows();

  std::cout << "正在预览 " << device_path
            << "，按 q、Esc 或 Ctrl+C 退出。\n";

  /** 保存从 V4L2 mmap 缓冲区复制出的 MJPEG 数据。 */
  std::vector<unsigned char> compressed_frame;
  /** 统计被驱动标记为错误或内容无效的帧数量。 */
  std::uint64_t invalid_frame_count = 0U;
  /** 统计 OpenCV 无法解码的 MJPEG 帧数量。 */
  std::uint64_t decode_failure_count = 0U;
  /** 保存当前 FPS 统计周期的起始时刻。 */
  std::chrono::steady_clock::time_point fps_period_start =
      std::chrono::steady_clock::now();
  /** 统计当前 FPS 周期内成功显示的帧数。 */
  std::uint64_t fps_period_frames = 0U;
  /** 保存最近一个完整统计周期的平均 FPS。 */
  double average_fps = 0.0;

  while (g_running != 0) {
    if (!camera.ReadCompressedFrame(&compressed_frame)) {
      ++invalid_frame_count;
      continue;
    }

    /** 保存 OpenCV 解码后的双目 BGR 拼接图像。 */
    cv::Mat stereo_frame = cv::imdecode(compressed_frame, cv::IMREAD_COLOR);
    if (stereo_frame.empty()) {
      ++decode_failure_count;
      std::cerr << "警告：MJPEG 帧解码失败，累计 " << decode_failure_count
                << " 帧\n";
      continue;
    }
    if (stereo_frame.cols != static_cast<int>(kFrameWidth) ||
        stereo_frame.rows != static_cast<int>(kFrameHeight) ||
        stereo_frame.cols % 2 != 0) {
      throw std::runtime_error(
          "解码帧尺寸不符合 3840x1080 双目横向拼接格式，实际为 " +
          std::to_string(stereo_frame.cols) + "x" +
          std::to_string(stereo_frame.rows));
    }

    /** 保存单目图像宽度，单位为像素。 */
    const int eye_width = stereo_frame.cols / 2;
    /** 引用拼接帧左半部分的左目图像，不复制像素数据。 */
    cv::Mat left_frame =
        stereo_frame(cv::Rect(0, 0, eye_width, stereo_frame.rows));
    /** 引用拼接帧右半部分的右目图像，不复制像素数据。 */
    cv::Mat right_frame = stereo_frame(
        cv::Rect(eye_width, 0, eye_width, stereo_frame.rows));

    ++fps_period_frames;
    /** 保存当前帧完成解码和拆分时的单调时钟时间。 */
    const std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
    /** 保存当前 FPS 统计周期已经持续的秒数。 */
    const double fps_elapsed_seconds =
        std::chrono::duration<double>(now - fps_period_start).count();
    if (fps_elapsed_seconds >= kFpsUpdateSeconds) {
      average_fps = static_cast<double>(fps_period_frames) /
                    fps_elapsed_seconds;
      fps_period_start = now;
      fps_period_frames = 0U;
    } else if (average_fps == 0.0 && fps_elapsed_seconds > 0.0) {
      average_fps = static_cast<double>(fps_period_frames) /
                    fps_elapsed_seconds;
    }

    DrawFpsOverlay(&left_frame, average_fps);
    DrawFpsOverlay(&right_frame, average_fps);

    cv::imshow(kLeftWindowName, left_frame);
    cv::imshow(kRightWindowName, right_frame);

    /** 保存 HighGUI 事件循环返回的低八位键值。 */
    const int key = cv::waitKey(1) & 0xff;
    if (key == kEscapeKey || key == 'q' || key == 'Q') {
      g_running = 0;
    }
  }

  cv::destroyAllWindows();
  std::cout << "预览结束，跳过无效帧 " << invalid_frame_count
            << "，解码失败帧 " << decode_failure_count << "。\n";
  return 0;
}

} // namespace

/**
 * @brief 程序入口，解析参数并启动双目实时预览。
 * @param[in] argc 命令行参数数量。
 * @param[in] argv 命令行参数数组，元素由运行时管理。
 * @return 正常退出或显示帮助时返回 0；参数、相机或显示错误时返回 1。
 */
int main(int argc, char *argv[]) {
  try {
    /** 保存解析得到的程序运行选项。 */
    const ProgramOptions options = ParseOptions(argc, argv);
    if (options.show_help) {
      PrintUsage(argv[0]);
      return 0;
    }
    if (!HasGraphicalDisplay()) {
      throw std::runtime_error(
          "未检测到图形显示环境，请在设置 DISPLAY 或 WAYLAND_DISPLAY 的桌面终端运行");
    }

    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);
    return RunViewer(options.device_path);
  } catch (const std::exception &error) {
    std::cerr << "错误：" << error.what() << '\n';
    PrintUsage(argv[0]);
    return 1;
  }
}
