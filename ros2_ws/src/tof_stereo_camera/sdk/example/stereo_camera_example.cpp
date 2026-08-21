/**
 * @file stereo_camera_example.cpp
 * @brief 直接调用随附 stereo_camera SDK 的原始帧元数据观察示例。
 */

#include "frame_display.hpp"
#include "stereo_camera/stereo_camera.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

namespace {

/// 示例默认请求的复合帧宽度。
constexpr int kDefaultWidth = 2048;

/// 示例默认请求的复合帧高度。
constexpr int kDefaultHeight = 2738;

/// 示例默认请求的像素格式。
constexpr const char *kDefaultFormat = "YUYV";

/// 连续无帧返回时的主机退避时长。
constexpr auto kNullFrameBackoff = std::chrono::milliseconds(1);

/// 显示窗口两次交换缓冲区之间的最短间隔。
constexpr auto kDisplayRefreshInterval = std::chrono::milliseconds(16);

/// RGB 视频流 ID。
constexpr int kRgbStreamId = 0;

/// iTOF 深度视频流 ID。
constexpr int kItofDepthStreamId = 2;

/// iTOF 灰度视频流 ID。
constexpr int kItofGrayStreamId = 7;

/// SDK 解析循环是否继续运行；该标志由信号处理函数设置。
volatile std::sig_atomic_t g_running = 1;

/**
 * @brief 保存命令行请求的设备、格式和输出选项。
 */
struct Options {
  /// 指定的设备路径；空字符串表示由 SDK 自动选择设备。
  std::string device;

  /// 请求的图像宽度。
  int width = kDefaultWidth;

  /// 请求的图像高度。
  int height = kDefaultHeight;

  /// 请求的四字符像素格式。
  std::string format = kDefaultFormat;

  /// 是否逐次输出成功解析的原始帧字段。
  bool verbose = false;

  /// 是否创建三路原始图像显示窗口。
  bool display = false;

  /// iTOF 深度图显示映射使用的最大原始值。
  int depth_max = 5000;

  /// iTOF 灰度图显示映射使用的最大原始值。
  int gray_max = 65535;

  /// 是否只显示帮助并在打开相机前退出。
  bool help = false;
};

/**
 * @brief 保存一条传感器类型和流 ID 路径的窗口统计数据。
 */
struct RouteStats {
  /// 当前统计窗口内 stereo_camera_parse_frame 的成功返回次数。
  std::uint64_t callback_count = 0;

  /// 当前统计窗口内的 IMU 样本数。
  std::uint64_t imu_sample_count = 0;
};

/// 以 sourcetype 和 stream_id 唯一标识一路 SDK 返回流。
using RouteKey = std::pair<int, int>;

/**
 * @brief 在 SDK 读取线程和主线程之间传递单次阻塞读取结果。
 *
 * 主线程处理完成前，读取线程不会发起下一次解析，确保 SDK 返回的帧指针
 * 在使用期间保持有效。
 */
struct FrameReadState {
  /// 保护读取结果和停止状态的互斥量。
  std::mutex mutex;

  /// 通知读取结果已就绪或线程应退出的条件变量。
  std::condition_variable condition;

  /// 最近一次 SDK 读取结果；允许为空以表示无帧或错误。
  stereo_camera_frame_t *frame = nullptr;

  /// 是否有一轮新的读取结果等待主线程处理。
  bool result_ready = false;

  /// 是否请求读取线程停止。
  bool stop_requested = false;
};

/**
 * @brief 记录 Ctrl-C 或终止信号，请主线程取消 SDK 阻塞读取并退出。
 *
 * @param signal_number 收到的信号编号；示例无需读取其具体值。
 * @note 信号处理函数只修改 sig_atomic_t 标志，不调用非异步信号安全接口。
 */
void handle_signal(int signal_number) {
  (void)signal_number;
  g_running = 0;
}

/**
 * @brief 持续执行 SDK 阻塞读取，并逐次将结果交给主线程。
 *
 * @param[in] camera 已启动视频流的 SDK 相机句柄。
 * @param[in,out] state 读取线程与主线程共享的同步状态。
 * @note 主线程通过 stereo_camera_cancel_read 唤醒阻塞读取后，本函数退出。
 */
void run_frame_reader(stereo_camera_t *camera, FrameReadState *state) {
  while (true) {
    stereo_camera_frame_t *frame =
        stereo_camera_parse_frame(camera); // 本轮 SDK 阻塞读取结果。
    std::unique_lock<std::mutex> lock(state->mutex); // 共享状态访问锁。
    if (state->stop_requested) {
      return;
    }
    state->frame = frame;
    state->result_ready = true;
    state->condition.notify_one();
    state->condition.wait(lock, [state] {
      return state->stop_requested || !state->result_ready;
    });
    if (state->stop_requested) {
      return;
    }
  }
}

/**
 * @brief 返回传感器类型的可读名称。
 *
 * @param sourcetype SDK 返回的传感器类型值。
 * @return 传感器类型名称；未知值返回 UNKNOWN。
 */
const char *sensor_name(int sourcetype) {
  switch (sourcetype) {
  case STEREO_SENSOR_RGB:
    return "RGB";
  case STEREO_SENSOR_ITOF:
    return "ITOF";
  case STEREO_SENSOR_DTOF:
    return "DTOF";
  case STEREO_SENSOR_IMU:
    return "IMU";
  default:
    return "UNKNOWN";
  }
}

/**
 * @brief 将 SDK 的 V4L2 FOURCC 数值转换为四字符显示文本。
 *
 * @param pixel_format SDK 返回的 FOURCC 数值。
 * @return 四字符文本；不可打印字节显示为点号。
 */
std::string fourcc_string(std::uint32_t pixel_format) {
  std::string value(4, '.');
  for (std::size_t index = 0; index < value.size(); ++index) {
    const unsigned char character =
        static_cast<unsigned char>((pixel_format >> (index * 8U)) & 0xffU);
    if (character >= 0x20U && character <= 0x7eU) {
      value[index] = static_cast<char>(character);
    }
  }
  return value;
}

/**
 * @brief 打印一条 SDK 原始帧的元数据。
 *
 * @param output 输出流。
 * @param frame SDK 返回的帧结构体；只读取结构体字段，不访问像素数据。
 * @param prefix 输出行前缀，用于区分首次帧和 verbose 输出。
 */
void print_frame_metadata(std::ostream &output,
                          const stereo_camera_frame_t &frame,
                          const char *prefix) {
  output << prefix << "sourcetype=" << frame.sourcetype << " ("
         << sensor_name(frame.sourcetype) << ")"
         << " stream_id=" << frame.stream_id
         << " frame_seqidx=" << frame.frame_seqidx
         << " frame_timestamp=" << frame.frame_timestamp << " us"
         << " width=" << frame.width << " height=" << frame.height
         << " FOURCC=" << fourcc_string(frame.pixel_format) << " (0x"
         << std::hex << std::setw(8) << std::setfill('0') << frame.pixel_format
         << std::dec << std::setfill(' ') << ")"
         << " data_size=" << frame.data_size << " bytes"
         << " match_state=" << frame.match_state
         << " frame_seq_count=" << frame.frame_seq_count << '\n';
}

/**
 * @brief 打印 iTOF 首帧按两种字节序解释后的 16 位数值范围。
 *
 * @param[in] frame iTOF 深度或灰度帧。
 * @note 仅应在一路流首次出现时调用，避免持续扫描 payload 增加采集开销。
 */
void print_itof_payload_diagnostics(const stereo_camera_frame_t &frame) {
  if (frame.data == nullptr || frame.data_size <= 0 ||
      (frame.data_size % 2) != 0) {
    std::cout << "[itof payload] stream_id=" << frame.stream_id
              << " cannot inspect empty or odd-sized payload\n";
    return;
  }
  std::uint16_t little_min = std::numeric_limits<std::uint16_t>::max();
  std::uint16_t little_max = 0; // 小端解释的最大值。
  std::uint16_t swapped_min = std::numeric_limits<std::uint16_t>::max();
  std::uint16_t swapped_max = 0;        // 字节交换解释的最大值。
  std::uint64_t little_zero_count = 0;  // 小端解释时零值数量。
  std::uint64_t swapped_zero_count = 0; // 字节交换解释时零值数量。
  const std::size_t value_count =
      static_cast<std::size_t>(frame.data_size) / 2U; // 16 位样本数量。
  for (std::size_t index = 0; index < value_count; ++index) {
    const std::uint16_t first_byte = frame.data[index * 2U]; // 第一个字节。
    const std::uint16_t second_byte =
        frame.data[index * 2U + 1U]; // 第二个字节。
    const std::uint16_t little_value =
        first_byte | static_cast<std::uint16_t>(second_byte << 8U);
    const std::uint16_t swapped_value =
        second_byte | static_cast<std::uint16_t>(first_byte << 8U);
    little_min = std::min(little_min, little_value);
    little_max = std::max(little_max, little_value);
    swapped_min = std::min(swapped_min, swapped_value);
    swapped_max = std::max(swapped_max, swapped_value);
    little_zero_count += little_value == 0 ? 1U : 0U;
    swapped_zero_count += swapped_value == 0 ? 1U : 0U;
  }
  const double little_zero_percent =
      100.0 * static_cast<double>(little_zero_count) /
      static_cast<double>(value_count); // 小端零值占比。
  const double swapped_zero_percent =
      100.0 * static_cast<double>(swapped_zero_count) /
      static_cast<double>(value_count); // 字节交换零值占比。
  std::cout << "[itof payload] stream_id=" << frame.stream_id
            << " values=" << value_count << " little_endian[min=" << little_min
            << ", max=" << little_max << ", zero=" << little_zero_percent
            << "%] swapped[min=" << swapped_min << ", max=" << swapped_max
            << ", zero=" << swapped_zero_percent << "%]\n";
}

/**
 * @brief 打印命令行用法和所有支持的选项。
 *
 * @param output 输出流。
 * @param program_name 程序名。
 */
void print_usage(std::ostream &output, const char *program_name) {
  output << "Usage: " << program_name << " [options]\n\n"
         << "Options:\n"
         << "  --device PATH   选择设备路径；默认由 SDK 自动选择\n"
         << "  --width N       请求复合帧宽度；默认 " << kDefaultWidth << "\n"
         << "  --height N      请求复合帧高度；默认 " << kDefaultHeight << "\n"
         << "  --format FOURCC 请求像素格式（YUYV 或 NV12）；默认 "
         << kDefaultFormat << "\n"
         << "  --display       显示 RGB、iTOF Depth 和 iTOF Gray；GPU "
            "直接解释原始 payload\n"
         << "  --depth-max N   16位深度后备路径的显示最大值；默认 5000\n"
         << "  --gray-max N    16位灰度后备路径的显示最大值；默认 65535\n"
         << "  --verbose       每次 parse_frame 成功返回都打印原始字段\n"
         << "  --help          显示此帮助并退出\n";
}

/**
 * @brief 读取需要一个后置值的命令行选项。
 *
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @param index 当前参数下标，成功时会前移到值参数。
 * @param option_name 选项名称。
 * @param value 输出的选项值。
 * @param error 输出的错误描述。
 * @return 成功读取返回 true，参数缺失返回 false。
 */
bool read_option_value(int argc, char *argv[], int *index,
                       const char *option_name, std::string *value,
                       std::string *error) {
  if (*index + 1 >= argc || argv[*index + 1] == nullptr ||
      std::strncmp(argv[*index + 1], "--", 2) == 0) {
    *error = std::string(option_name) + " requires a value";
    return false;
  }

  ++(*index);
  *value = argv[*index];
  return true;
}

/**
 * @brief 将命令行文本解析为正整数。
 *
 * @param text 待解析的文本。
 * @param option_name 选项名称。
 * @param value 输出的正整数。
 * @param error 输出的错误描述。
 * @return 文本合法且在 int 范围内时返回 true。
 */
bool parse_positive_int(const std::string &text, const char *option_name,
                        int *value, std::string *error) {
  if (text.empty()) {
    *error = std::string(option_name) + " expects a positive integer";
    return false;
  }

  std::size_t consumed = 0; // 已成功解析的字符数量。
  try {
    const int parsed = std::stoi(text, &consumed);
    if (consumed != text.size() || parsed <= 0) {
      *error =
          std::string(option_name) + " expects a positive integer: " + text;
      return false;
    }
    *value = parsed;
    return true;
  } catch (const std::exception &exception) {
    (void)exception;
    *error = std::string(option_name) + " expects a positive integer: " + text;
    return false;
  }
}

/**
 * @brief 解析示例支持的命令行参数。
 *
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @param options 输出的运行选项。
 * @param error 输出的参数错误描述。
 * @return 参数合法时返回 true；发现错误时返回 false。
 */
bool parse_options(int argc, char *argv[], Options *options,
                   std::string *error) {
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index]; // 当前命令行选项。
    std::string value;                        // 当前选项的后置值。

    if (argument == "--help") {
      options->help = true;
      return true;
    }
    if (argument == "--verbose") {
      options->verbose = true;
      continue;
    }
    if (argument == "--display") {
      options->display = true;
      continue;
    }
    if (argument == "--device") {
      if (!read_option_value(argc, argv, &index, "--device", &value, error)) {
        return false;
      }
      options->device = value;
      continue;
    }
    if (argument == "--width" || argument == "--height") {
      if (!read_option_value(argc, argv, &index, argument.c_str(), &value,
                             error)) {
        return false;
      }
      int *destination =
          argument == "--width" ? &options->width : &options->height;
      if (!parse_positive_int(value, argument.c_str(), destination, error)) {
        return false;
      }
      continue;
    }
    if (argument == "--depth-max" || argument == "--gray-max") {
      if (!read_option_value(argc, argv, &index, argument.c_str(), &value,
                             error)) {
        return false;
      }
      int *destination =
          argument == "--depth-max" ? &options->depth_max : &options->gray_max;
      if (!parse_positive_int(value, argument.c_str(), destination, error)) {
        return false;
      }
      continue;
    }
    if (argument == "--format") {
      if (!read_option_value(argc, argv, &index, "--format", &value, error)) {
        return false;
      }
      if (value.size() != 4) {
        *error = "--format expects exactly four characters: " + value;
        return false;
      }
      options->format = value;
      continue;
    }

    *error = "unknown option: " + argument;
    return false;
  }
  return true;
}

/**
 * @brief 为三路显示显式启用 RGB、iTOF Depth 和 iTOF Gray 设备流。
 *
 * @param[in] camera 已打开且已协商格式的 SDK 相机句柄。
 * @param[out] error XU 传输或设备 ACK 失败的诊断信息。
 * @return 设备接受流掩码时返回 true。
 */
bool configure_display_streams(stereo_camera_t *camera, std::string *error) {
  const std::uint32_t stream_mask =
      (std::uint32_t{1} << static_cast<unsigned int>(kRgbStreamId)) |
      (std::uint32_t{1} << static_cast<unsigned int>(kItofDepthStreamId)) |
      (std::uint32_t{1} << static_cast<unsigned int>(kItofGrayStreamId));
  const stereo_camera_xu_stream_mask_param_t parameter{
      stream_mask};         // 设备视频流选择掩码。
  int acknowledgement = -1; // 设备命令 ACK；零表示成功。
  const int result = stereo_camera_xu_command(
      camera, STEREO_CAMERA_XU_CMD_STREAM_MASK, &parameter, sizeof(parameter),
      &acknowledgement); // SDK XU 调用结果。
  if (result == 0 && acknowledgement == 0) {
    std::cout << "Display stream mask configured: 0x" << std::hex << stream_mask
              << std::dec << " (RGB + iTOF Depth + iTOF Gray)\n";
    return true;
  }
  std::ostringstream diagnostic; // XU 失败诊断文本。
  diagnostic << "stream mask XU command failed: result=" << result
             << ", ack=" << acknowledgement << ", mask=0x" << std::hex
             << stream_mask;
  *error = diagnostic.str();
  return false;
}

/**
 * @brief 打印各路在一个主机时间窗口内的回调和 IMU 样本速率。
 *
 * @param output 输出流。
 * @param routes 按传感器类型和流 ID 保存的统计数据。
 * @param window_seconds 窗口长度，单位为秒。
 * @param null_returns 当前窗口内 SDK 返回空指针的次数。
 */
void print_window_stats(std::ostream &output,
                        const std::map<RouteKey, RouteStats> &routes,
                        double window_seconds, std::uint64_t null_returns) {
  output << "[stats] window=" << std::fixed << std::setprecision(3)
         << window_seconds << " s null_returns=" << null_returns << '\n';
  for (const auto &entry : routes) {
    const RouteKey &route = entry.first;    // 传感器类型和流 ID。
    const RouteStats &stats = entry.second; // 当前路的窗口统计。
    const double callback_fps =
        static_cast<double>(stats.callback_count) / window_seconds;
    output << "  sourcetype=" << route.first << " (" << sensor_name(route.first)
           << ") stream_id=" << route.second << " callback_fps=" << callback_fps
           << " parse_frame_returns/s";
    if (route.first == STEREO_SENSOR_IMU) {
      const double sample_rate =
          static_cast<double>(stats.imu_sample_count) / window_seconds;
      output << " imu_sample_rate=" << sample_rate
             << " samples/s (frame_seq_count 优先)";
    }
    output << '\n';
  }
  output.flush();
}

/**
 * @brief 清空各路当前窗口的计数。
 *
 * @param routes 按传感器类型和流 ID 保存的统计数据。
 */
void reset_window_stats(std::map<RouteKey, RouteStats> *routes) {
  for (auto &entry : *routes) {
    entry.second.callback_count = 0;
    entry.second.imu_sample_count = 0;
  }
}

} // namespace
/**
 * @brief 运行直接调用 SDK 的元数据观察示例。
 *
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @return 0 表示正常结束；参数、SDK 或流控制错误返回非零。
 */
int main(int argc, char *argv[]) {
  Options options;         // 命令行解析后的运行选项。
  std::string parse_error; // 命令行错误描述。
  if (!parse_options(argc, argv, &options, &parse_error)) {
    std::cerr << "Error: " << parse_error << '\n';
    print_usage(std::cerr, argv[0]);
    return 2;
  }
  if (options.help) {
    print_usage(std::cout, argv[0]);
    return 0;
  }

  const char *device_path =
      options.device.empty() ? nullptr : options.device.c_str();
  stereo_camera_t *camera = stereo_camera_open(device_path); // SDK 相机句柄。
  if (camera == nullptr) {
    std::cerr << "Error: stereo_camera_open failed";
    if (device_path != nullptr) {
      std::cerr << ": " << device_path;
    }
    std::cerr << '\n';
    return 1;
  }

  if (stereo_camera_set_format(camera, options.width, options.height,
                               options.format.c_str()) != 0) {
    std::cerr << "Error: stereo_camera_set_format failed\n";
    stereo_camera_close(camera);
    return 1;
  }

  int actual_width = 0;       // SDK 实际协商的宽度。
  int actual_height = 0;      // SDK 实际协商的高度。
  char actual_format[5] = {}; // SDK 实际协商的 FOURCC 文本。
  if (stereo_camera_get_format(camera, &actual_width, &actual_height,
                               actual_format) != 0) {
    std::cerr << "Error: stereo_camera_get_format failed\n";
    stereo_camera_close(camera);
    return 1;
  }
  std::cout << "Negotiated format: " << actual_width << "x" << actual_height
            << " " << actual_format << '\n';

  FrameDisplay display;      // 可选的三路 GPU 显示窗口。
  std::string display_error; // 显示初始化或帧上传错误。
  if (options.display && !configure_display_streams(camera, &display_error)) {
    std::cerr << "Error: " << display_error << '\n';
    stereo_camera_close(camera);
    return 1;
  }
  if (options.display &&
      !display.Initialize({options.depth_max, options.gray_max},
                          &display_error)) {
    std::cerr << "Error: display initialization failed: " << display_error
              << '\n';
    stereo_camera_close(camera);
    return 1;
  }

  if (stereo_camera_start_stream(camera) != 0) {
    std::cerr << "Error: stereo_camera_start_stream failed\n";
    stereo_camera_close(camera);
    return 1;
  }

  g_running = 1;
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  std::map<RouteKey, RouteStats> routes; // 各传感器类型和流 ID 的统计。
  using Clock = std::chrono::steady_clock;
  Clock::time_point window_start = Clock::now(); // 当前主机时间窗口起点。
  int exit_code = 0;                             // 采集循环退出状态。
  std::uint64_t consecutive_null_frames = 0; // 连续无帧/错误返回次数。
  std::uint64_t null_returns = 0; // 当前统计窗口内的空指针返回次数。
  Clock::time_point last_success_frame_time = window_start; // 最近成功帧时间。
  Clock::time_point last_null_warning_time =
      window_start;              // 最近空返回提示时间。
  bool has_null_warning = false; // 是否已经输出过空返回提示。
  std::map<RouteKey, std::string> display_route_errors; // 各显示路最近错误。
  Clock::time_point last_display_time =
      window_start - kDisplayRefreshInterval; // 最近一次窗口刷新时间。

  FrameReadState read_state; // SDK 读取线程与主线程的同步状态。
  std::thread reader_thread(run_frame_reader, camera,
                            &read_state); // 可取消的 SDK 阻塞读取线程。

  while (g_running != 0) {
    if (options.display && !display.PollEvents()) {
      g_running = 0;
      break;
    }

    stereo_camera_frame_t *frame = nullptr; // 当前等待处理的 SDK 帧指针。
    bool result_ready = false; // 是否取得一轮 SDK 读取结果。
    {
      std::unique_lock<std::mutex> lock(read_state.mutex); // 读取状态访问锁。
      read_state.condition.wait_for(
          lock, kDisplayRefreshInterval,
          [&read_state] { return read_state.result_ready; });
      result_ready = read_state.result_ready;
      if (result_ready) {
        frame = read_state.frame;
      }
    }

    if (result_ready) {
      if (frame == nullptr) {
        ++null_returns;
        ++consecutive_null_frames;
        const Clock::time_point null_time =
            Clock::now(); // 本轮空返回的主机单调时钟时间。
        const double seconds_since_success =
            std::chrono::duration<double>(null_time - last_success_frame_time)
                .count();
        const bool warning_interval_elapsed =
            !has_null_warning ||
            std::chrono::duration<double>(null_time - last_null_warning_time)
                    .count() >= 1.0;
        if (seconds_since_success >= 1.0 && warning_interval_elapsed) {
          std::cerr << "Warning: stereo_camera_parse_frame returned no "
                       "frame/error (null_returns="
                    << null_returns << ")\n";
          last_null_warning_time = null_time;
          has_null_warning = true;
        }
        if (consecutive_null_frames > 1) {
          std::this_thread::sleep_for(kNullFrameBackoff);
        }
      } else {
        consecutive_null_frames = 0;
        const Clock::time_point frame_time =
            Clock::now(); // 成功帧到达的主机单调时钟时间。
        last_success_frame_time = frame_time;

        const RouteKey route_key{frame->sourcetype,
                                 frame->stream_id}; // 当前帧路由标识。
        const auto insertion = routes.emplace(route_key, RouteStats{});
        RouteStats &stats = insertion.first->second; // 当前路的统计对象。
        if (insertion.second || options.verbose) {
          print_frame_metadata(std::cout, *frame,
                               insertion.second ? "[frame first] "
                                                : "[frame] ");
        }
        if (insertion.second && frame->sourcetype == STEREO_SENSOR_ITOF) {
          print_itof_payload_diagnostics(*frame);
        }
        ++stats.callback_count;
        if (frame->sourcetype == STEREO_SENSOR_IMU) {
          std::uint64_t sample_count =
              frame->frame_seq_count; // 当前 IMU 回调包含的样本数。
          if (sample_count == 0 && frame->data_size > 0) {
            sample_count = static_cast<std::uint64_t>(frame->data_size) /
                           sizeof(stereo_camera_imu_data_t);
          }
          stats.imu_sample_count += sample_count;
        }

        if (options.display) {
          display_error.clear();
          const bool image_updated =
              display.Update(*frame, &display_error); // 是否更新了显示纹理。
          if (!display_error.empty()) {
            std::string &last_error =
                display_route_errors[route_key]; // 当前路最近一次显示错误。
            if (last_error != display_error) {
              std::cerr << "Warning: display skipped sourcetype="
                        << frame->sourcetype
                        << " stream_id=" << frame->stream_id << ": "
                        << display_error << '\n';
              last_error = display_error;
            }
          } else if (image_updated) {
            display_route_errors.erase(route_key);
            if (frame_time - last_display_time >= kDisplayRefreshInterval) {
              display.Render();
              last_display_time = frame_time;
            }
          }
        }
      }

      {
        std::lock_guard<std::mutex> lock(
            read_state.mutex); // 标记本轮结果已处理的状态锁。
        read_state.result_ready = false;
      }
      read_state.condition.notify_one();
    }

    const Clock::time_point now = Clock::now(); // 当前主机单调时钟时间。
    const double window_seconds =
        std::chrono::duration<double>(now - window_start).count();
    if (window_seconds >= 1.0) {
      print_window_stats(std::cout, routes, window_seconds, null_returns);
      reset_window_stats(&routes);
      null_returns = 0;
      window_start = now;
    }
  }

  {
    std::lock_guard<std::mutex> lock(
        read_state.mutex); // 发布读取线程停止请求的状态锁。
    read_state.stop_requested = true;
    read_state.result_ready = false;
  }
  stereo_camera_cancel_read(camera);
  read_state.condition.notify_all();
  reader_thread.join();

  if (stereo_camera_stop_stream(camera) != 0) {
    std::cerr << "Error: stereo_camera_stop_stream failed\n";
    exit_code = 1;
  }
  stereo_camera_close(camera);
  return exit_code;
}
