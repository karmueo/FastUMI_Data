/**
 * @file frame_utils.cpp
 * @brief ROS 2 图像与 IMU 帧转换工具的实现。
 * @author 待确认
 * @date 创建：待确认
 * @date 修改：2026-08-20
 */

#include "tof_stereo_camera/frame_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include <cstddef>
#include <cstring>
#include <limits>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <sensor_msgs/image_encodings.hpp>

namespace tof_stereo_camera {
namespace {

static_assert(sizeof(stereo_camera_imu_data_t) == 72,
              "stereo_camera_imu_data_t must match the public SDK ABI");
static_assert(offsetof(stereo_camera_imu_data_t, timestamp) == 0,
              "stereo_camera_imu_data_t::timestamp ABI mismatch");
static_assert(offsetof(stereo_camera_imu_data_t, idx) == 8,
              "stereo_camera_imu_data_t::idx ABI mismatch");
static_assert(offsetof(stereo_camera_imu_data_t, ax) == 16,
              "stereo_camera_imu_data_t::ax ABI mismatch");
static_assert(offsetof(stereo_camera_imu_data_t, gx) == 28,
              "stereo_camera_imu_data_t::gx ABI mismatch");
static_assert(offsetof(stereo_camera_imu_data_t, reverve) == 40,
              "stereo_camera_imu_data_t::reverve ABI mismatch");
static_assert(sizeof(stereo_camera_frame_t) == 56,
              "stereo_camera_frame_t must match the public SDK ABI");
static_assert(offsetof(stereo_camera_frame_t, frame_timestamp) == 0,
              "stereo_camera_frame_t::frame_timestamp ABI mismatch");
static_assert(offsetof(stereo_camera_frame_t, frame_seqidx) == 8,
              "stereo_camera_frame_t::frame_seqidx ABI mismatch");
static_assert(offsetof(stereo_camera_frame_t, stream_id) == 16,
              "stereo_camera_frame_t::stream_id ABI mismatch");
static_assert(offsetof(stereo_camera_frame_t, data) == 32,
              "stereo_camera_frame_t::data ABI mismatch");
static_assert(offsetof(stereo_camera_frame_t, frame_seq_count) == 52,
              "stereo_camera_frame_t::frame_seq_count ABI mismatch");

/// 每微秒包含的纳秒数。
constexpr std::int64_t kNanosecondsPerMicrosecond = 1'000;
/// 每度对应的弧度数。
constexpr double kRadiansPerDegree = 0.017453292519943295;

/**
 * @brief 按 V4L2 字节序构造 FOURCC 数值。
 * @param[in] a FOURCC 的第一个字符。
 * @param[in] b FOURCC 的第二个字符。
 * @param[in] c FOURCC 的第三个字符。
 * @param[in] d FOURCC 的第四个字符。
 * @return 对应的 32 位 FOURCC。
 */
constexpr std::uint32_t MakeFourcc(char a, char b, char c, char d) {
  return static_cast<std::uint32_t>(static_cast<unsigned char>(a)) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(b)) << 8U) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(c)) << 16U) |
         (static_cast<std::uint32_t>(static_cast<unsigned char>(d)) << 24U);
}

/// YUYV 图像的 V4L2 FOURCC。
constexpr std::uint32_t kFourccYuyv = MakeFourcc('Y', 'U', 'Y', 'V');
/// NV12 图像的 V4L2 FOURCC。
constexpr std::uint32_t kFourccNv12 = MakeFourcc('N', 'V', '1', '2');

/**
 * @brief 校验 SDK 图像帧的尺寸与 payload 是否匹配。
 * @param[in] frame 待校验的 SDK 图像帧。
 * @param[in] bytes 由图像格式推导出的期望 payload 字节数。
 * @param[out] error 接收校验失败原因，不可为 `nullptr`。
 * @return 尺寸和 payload 均有效时返回 true。
 */
bool ValidateFrame(const stereo_camera_frame_t &frame, std::uint64_t bytes,
                   std::string *error) {
  if (frame.width <= 0 || frame.height <= 0) {
    *error = "image dimensions must be positive";
    return false;
  }
  if (frame.data == nullptr || frame.data_size <= 0) {
    *error = "frame has no payload";
    return false;
  }
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
      frame.data_size != static_cast<int>(bytes)) {
    *error = "frame payload size does not match its dimensions";
    return false;
  }
  return true;
}

/**
 * @brief 计算图像帧的像素数量。
 * @param[in] frame 包含宽高的 SDK 图像帧。
 * @return 宽度与高度相乘得到的像素数。
 */
std::uint64_t PixelCount(const stereo_camera_frame_t &frame) {
  return static_cast<std::uint64_t>(frame.width) *
         static_cast<std::uint64_t>(frame.height);
}

} // namespace

/** @copydoc DegreesPerSecondToRadiansPerSecond */
double DegreesPerSecondToRadiansPerSecond(float degrees_per_second) {
  return static_cast<double>(degrees_per_second) * kRadiansPerDegree;
}

/** @copydoc IsPublishableTofMatchState */
bool IsPublishableTofMatchState(int match_state) {
  return match_state == STEREO_MATCH_EXACT ||
         match_state == STEREO_MATCH_APPROX ||
         match_state == STEREO_MATCH_STALE;
}

/** @copydoc ParseRgbOutputEncoding */
bool ParseRgbOutputEncoding(const std::string &value,
                            RgbOutputEncoding *output_encoding,
                            std::string *error) {
  if (value == sensor_msgs::image_encodings::YUV422_YUY2) {
    *output_encoding = RgbOutputEncoding::kYuv422Yuy2;
    return true;
  }
  if (value == sensor_msgs::image_encodings::BGR8) {
    *output_encoding = RgbOutputEncoding::kBgr8;
    return true;
  }
  *error = "rgb_output_encoding must be yuv422_yuy2 or bgr8";
  return false;
}

/** @copydoc ConvertRgbFrame */
bool ConvertRgbFrame(const stereo_camera_frame_t &frame,
                     RgbOutputEncoding output_encoding,
                     sensor_msgs::msg::Image *message, std::string *error) {
  const std::uint64_t pixels = PixelCount(frame);
  if (output_encoding == RgbOutputEncoding::kYuv422Yuy2) {
    if (frame.pixel_format != kFourccYuyv) {
      *error = "yuv422_yuy2 output requires a YUYV RGB frame";
      return false;
    }
    if ((frame.width % 2) != 0 || !ValidateFrame(frame, pixels * 2U, error)) {
      if ((frame.width % 2) != 0) {
        *error = "YUYV width must be even";
      }
      return false;
    }
    message->height = static_cast<std::uint32_t>(frame.height);
    message->width = static_cast<std::uint32_t>(frame.width);
    message->encoding = sensor_msgs::image_encodings::YUV422_YUY2;
    message->is_bigendian = 0;
    message->step = static_cast<std::uint32_t>(frame.width * 2);
    message->data.assign(frame.data, frame.data + frame.data_size);
    return true;
  }

  if (output_encoding != RgbOutputEncoding::kBgr8) {
    *error = "unsupported RGB output encoding strategy";
    return false;
  }

  cv::Mat bgr;
  if (frame.pixel_format == kFourccYuyv) {
    if ((frame.width % 2) != 0 || !ValidateFrame(frame, pixels * 2U, error)) {
      if ((frame.width % 2) != 0) {
        *error = "YUYV width must be even";
      }
      return false;
    }
    const cv::Mat yuyv(frame.height, frame.width, CV_8UC2, frame.data);
    cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUY2);
  } else if (frame.pixel_format == kFourccNv12) {
    if ((frame.width % 2) != 0 || (frame.height % 2) != 0 ||
        !ValidateFrame(frame, pixels * 3U / 2U, error)) {
      if ((frame.width % 2) != 0 || (frame.height % 2) != 0) {
        *error = "NV12 width and height must be even";
      }
      return false;
    }
    const cv::Mat nv12(frame.height * 3 / 2, frame.width, CV_8UC1, frame.data);
    cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
  } else {
    *error = "RGB frame FOURCC is neither YUYV nor NV12";
    return false;
  }

  message->height = static_cast<std::uint32_t>(frame.height);
  message->width = static_cast<std::uint32_t>(frame.width);
  message->encoding = sensor_msgs::image_encodings::BGR8;
  message->is_bigendian = 0;
  message->step = static_cast<std::uint32_t>(frame.width * 3);
  message->data.assign(bgr.datastart, bgr.dataend);
  return true;
}

/** @copydoc CopyMono16Frame */
bool CopyMono16Frame(const stereo_camera_frame_t &frame,
                     const std::string &encoding,
                     sensor_msgs::msg::Image *message, std::string *error) {
  const std::uint64_t bytes = PixelCount(frame) * 2U;
  if (!ValidateFrame(frame, bytes, error)) {
    return false;
  }

  message->height = static_cast<std::uint32_t>(frame.height);
  message->width = static_cast<std::uint32_t>(frame.width);
  message->encoding = encoding;
  message->is_bigendian = 0;
  message->step = static_cast<std::uint32_t>(frame.width * 2);
  message->data.assign(frame.data, frame.data + frame.data_size);
  return true;
}

/** @copydoc DecodeImuFrame */
bool DecodeImuFrame(const stereo_camera_frame_t &frame,
                    std::vector<stereo_camera_imu_data_t> *samples,
                    std::string *error) {
  samples->clear();
  error->clear();
  if (frame.data_size < 0) {
    *error = "IMU payload size is negative";
    return false;
  }
  if (frame.data_size == 0) {
    if (frame.frame_seq_count != 0U) {
      std::ostringstream diagnostic;
      diagnostic << "IMU sequence count mismatch: frame_seq_count="
                 << frame.frame_seq_count << ", decoded_samples=0";
      *error = diagnostic.str();
    }
    return true;
  }
  if (frame.data == nullptr) {
    *error = "IMU frame has a null payload with positive size";
    return false;
  }

  const std::size_t payload_size = static_cast<std::size_t>(frame.data_size);
  if (payload_size % sizeof(stereo_camera_imu_data_t) != 0U) {
    *error = "IMU payload size is not a multiple of the SDK sample size";
    return false;
  }

  const std::size_t sample_count =
      payload_size / sizeof(stereo_camera_imu_data_t);
  samples->resize(sample_count);
  for (std::size_t index = 0; index < sample_count; ++index) {
    std::memcpy(&(*samples)[index],
                frame.data + index * sizeof(stereo_camera_imu_data_t),
                sizeof(stereo_camera_imu_data_t));
  }

  std::ostringstream diagnostic;
  if (static_cast<std::size_t>(frame.frame_seq_count) != sample_count) {
    diagnostic << "IMU sequence count mismatch: frame_seq_count="
               << frame.frame_seq_count << ", decoded_samples=" << sample_count;
  }
  if (frame.frame_seqidx != 0U && samples->front().idx > 0 &&
      frame.frame_seqidx != static_cast<std::uint64_t>(samples->front().idx)) {
    if (diagnostic.tellp() > 0) {
      diagnostic << "; ";
    }
    diagnostic << "IMU sequence index mismatch: frame_seqidx="
               << frame.frame_seqidx
               << ", first_sample_idx=" << samples->front().idx;
  }
  *error = diagnostic.str();
  return true;
}
/** @brief 安全相加两个有符号纳秒数。 */
static bool AddNs(std::int64_t a, std::int64_t b, std::int64_t *out) {
  if ((b > 0 && a > std::numeric_limits<std::int64_t>::max() - b) ||
      (b < 0 && a < std::numeric_limits<std::int64_t>::min() - b))
    return false;
  *out = a + b;
  return true;
}
/** @brief 安全计算两个有符号纳秒数的差。 */
static bool SubNs(std::int64_t a, std::int64_t b, std::int64_t *out) {
  if ((b > 0 && a < std::numeric_limits<std::int64_t>::min() + b) ||
      (b < 0 && a > std::numeric_limits<std::int64_t>::max() + b))
    return false;
  *out = a - b;
  return true;
}
/** @copydoc TimestampMapper::TimestampMapper */
TimestampMapper::TimestampMapper(std::int64_t steady_anchor_ns,
                                 std::int64_t system_anchor_ns,
                                 std::size_t calibration_frames,
                                 std::size_t window_frames, double max_slew_ppm)
    : steady_anchor_ns_(steady_anchor_ns), system_anchor_ns_(system_anchor_ns),
      calibration_frames_(calibration_frames), window_frames_(window_frames),
      max_slew_ppm_(max_slew_ppm) {
  if (calibration_frames_ < 2U || window_frames_ < calibration_frames_ ||
      !std::isfinite(max_slew_ppm_) || max_slew_ppm_ <= 0.0 ||
      max_slew_ppm_ >= 1'000'000.0)
    throw std::invalid_argument("invalid timestamp synchronization parameters");
}
/** @copydoc TimestampMapper::PushCandidate */
void TimestampMapper::PushCandidate(std::int64_t offset) {
  candidates_.push_back(offset);
  while (candidates_.size() > window_frames_)
    candidates_.pop_front();
}
/** @copydoc TimestampMapper::MinimumCandidate */
std::int64_t TimestampMapper::MinimumCandidate() const {
  return *std::min_element(candidates_.begin(), candidates_.end());
}
/** @copydoc TimestampMapper::SystemFromSteady */
std::int64_t TimestampMapper::SystemFromSteady(std::int64_t steady_ns) const {
  std::int64_t delta = 0;
  std::int64_t result = system_anchor_ns_;
  if (SubNs(steady_ns, steady_anchor_ns_, &delta))
    AddNs(system_anchor_ns_, delta, &result);
  return result;
}
/** @copydoc TimestampMapper::StartEpoch */
void TimestampMapper::StartEpoch(std::int64_t sdk_ns, std::int64_t receive_ns,
                                 std::int64_t offset) {
  candidates_.clear();
  PushCandidate(offset);
  ready_ = false;
  have_previous_unique_ = true;
  previous_sdk_ns_ = sdk_ns;
  previous_receive_ns_ = receive_ns;
}
/** @copydoc TimestampMapper::RestartCalibration */
void TimestampMapper::RestartCalibration() {
  candidates_.clear();
  ready_ = false;
  have_previous_unique_ = false;
}
/** @copydoc TimestampMapper::HostFallback */
FrameTimestampResult TimestampMapper::HostFallback(std::int64_t receive_ns) {
  FrameTimestampResult result;
  result.steady_ns = receive_ns;
  result.system_ns = SystemFromSteady(receive_ns);
  if (have_previous_ready_ && receive_ns <= previous_ready_steady_ns_) {
    RestartCalibration();
    result.status = FrameTimestampStatus::kDropped;
    return result;
  }
  result.status = FrameTimestampStatus::kInvalidHostFallback;
  previous_ready_steady_ns_ = receive_ns;
  have_previous_ready_ = true;
  return result;
}
/** @copydoc TimestampMapper::DuplicateResult */
FrameTimestampResult TimestampMapper::DuplicateResult() const {
  FrameTimestampResult result = cached_result_;
  result.status = FrameTimestampStatus::kDuplicate;
  result.ready = false;
  result.newly_locked = false;
  return result;
}
/** @copydoc TimestampMapper::ObserveFrame */
FrameTimestampResult TimestampMapper::ObserveFrame(std::uint64_t timestamp_us,
                                                   std::int64_t receive_ns) {
  FrameTimestampResult result;
  const std::uint64_t max_us = static_cast<std::uint64_t>(
      std::numeric_limits<std::int64_t>::max() / kNanosecondsPerMicrosecond);
  if (timestamp_us == 0 || timestamp_us > max_us)
    return HostFallback(receive_ns);
  if (have_cached_result_ && timestamp_us == cached_sdk_timestamp_us_)
    return DuplicateResult();
  const std::int64_t sdk_ns =
      static_cast<std::int64_t>(timestamp_us) * kNanosecondsPerMicrosecond;
  std::int64_t candidate = 0;
  if (!SubNs(receive_ns, sdk_ns, &candidate))
    return HostFallback(receive_ns);
  bool reset = false;
  std::int64_t device_delta = 0;
  if (have_previous_unique_) {
    if (sdk_ns < previous_sdk_ns_)
      reset = true;
    else if (sdk_ns > previous_sdk_ns_) {
      std::int64_t receive_delta = 0;
      std::int64_t delta_error = 0;
      if (!SubNs(sdk_ns, previous_sdk_ns_, &device_delta) ||
          !SubNs(receive_ns, previous_receive_ns_, &receive_delta) ||
          !SubNs(device_delta, receive_delta, &delta_error) ||
          delta_error == std::numeric_limits<std::int64_t>::min() ||
          std::llabs(delta_error) > 1'000'000'000LL)
        reset = true;
    }
  }
  if (reset) {
    StartEpoch(sdk_ns, receive_ns, candidate);
    result.status = FrameTimestampStatus::kReset;
  } else {
    if (!have_previous_unique_)
      StartEpoch(sdk_ns, receive_ns, candidate);
    else {
      PushCandidate(candidate);
      previous_sdk_ns_ = sdk_ns;
      previous_receive_ns_ = receive_ns;
    }
    if (!ready_ && candidates_.size() == calibration_frames_) {
      applied_offset_ns_ = MinimumCandidate();
      ready_ = true;
      result.newly_locked = true;
    }
    if (!ready_)
      result.status = FrameTimestampStatus::kCalibrating;
  }
  if (ready_ && !reset) {
    const std::int64_t target = MinimumCandidate();
    if (!result.newly_locked && target != applied_offset_ns_) {
      const long double raw = static_cast<long double>(max_slew_ppm_) *
                              static_cast<long double>(device_delta) /
                              1'000'000.0L;
      std::int64_t limit =
          std::max<std::int64_t>(1, static_cast<std::int64_t>(raw));
      const std::int64_t difference = target - applied_offset_ns_;
      applied_offset_ns_ += std::clamp(difference, -limit, limit);
    }
    std::int64_t mapped = 0;
    if (AddNs(sdk_ns, applied_offset_ns_, &mapped)) {
      if (mapped > receive_ns) {
        result.future_by_ns = mapped - receive_ns;
        // 当前候选偏移是已观测到的因果上界，可安全立即向下校正。
        applied_offset_ns_ = candidate;
        mapped = receive_ns;
        result.receive_clamped = true;
      }
      if (!have_previous_ready_ || mapped > previous_ready_steady_ns_) {
        result.status = FrameTimestampStatus::kReady;
        result.ready = true;
        result.steady_ns = mapped;
        result.system_ns = SystemFromSteady(mapped);
        result.offset_ns = applied_offset_ns_;
        result.has_offset_snapshot = true;
        previous_ready_steady_ns_ = mapped;
        have_previous_ready_ = true;
      } else {
        // 冲突帧不能既被丢弃又作为新 epoch 的可用标定观测。
        RestartCalibration();
        result.status = FrameTimestampStatus::kDropped;
        result.newly_locked = false;
      }
    } else {
      return HostFallback(receive_ns);
    }
  }
  cached_sdk_timestamp_us_ = timestamp_us;
  cached_result_ = result;
  have_cached_result_ = true;
  return result;
}
/** @copydoc TimestampMapper::MapImuSample */
ImuTimestampResult
TimestampMapper::MapImuSample(std::int64_t timestamp_us,
                              const FrameTimestampResult &frame) const {
  ImuTimestampResult result{frame.steady_ns, frame.system_ns, false, false};
  if (timestamp_us <= 0 || !frame.has_offset_snapshot ||
      timestamp_us >
          std::numeric_limits<std::int64_t>::max() / kNanosecondsPerMicrosecond)
    return result;
  std::int64_t mapped = 0;
  if (!AddNs(timestamp_us * kNanosecondsPerMicrosecond, frame.offset_ns,
             &mapped))
    return result;
  result.valid_sample = true;
  if (mapped > frame.steady_ns) {
    mapped = frame.steady_ns;
    result.future_clamped = true;
  }
  result.steady_ns = mapped;
  result.system_ns = SystemFromSteady(mapped);
  return result;
}
/** @copydoc TimestampMapper::ready */
bool TimestampMapper::ready() const { return ready_; }
/** @copydoc TimestampMapper::detected_epoch_offset_ns */
std::int64_t TimestampMapper::detected_epoch_offset_ns() const {
  return applied_offset_ns_;
}
/** @copydoc TimestampMapper::detected_system_epoch_offset_ns */
std::int64_t TimestampMapper::detected_system_epoch_offset_ns() const {
  std::int64_t base = 0;
  return SubNs(system_anchor_ns_, steady_anchor_ns_, &base) &&
                 AddNs(base, applied_offset_ns_, &base)
             ? base
             : 0;
}

} // namespace tof_stereo_camera
