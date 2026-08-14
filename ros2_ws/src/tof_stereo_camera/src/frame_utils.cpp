/**
 * @file frame_utils.cpp
 * @brief ROS 2 图像与 IMU 帧转换工具的实现。
 * @author 待确认
 * @date 创建：待确认
 * @date 修改：2026-08-13
 */

#include "tof_stereo_camera/frame_utils.hpp"

#include <cstddef>
#include <cstring>
#include <limits>
#include <sstream>

#include <opencv2/imgproc.hpp>

namespace tof_stereo_camera {
namespace {

static_assert(sizeof(stereo_camera_imu_data_t) == 72,
              "stereo_camera_imu_data_t must match the public SDK ABI");
static_assert(offsetof(stereo_camera_imu_data_t, idx) == 8,
              "stereo_camera_imu_data_t::idx ABI mismatch");
static_assert(offsetof(stereo_camera_imu_data_t, ax) == 16,
              "stereo_camera_imu_data_t::ax ABI mismatch");
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

}  // namespace

/** @copydoc IsPublishableTofMatchState */
bool IsPublishableTofMatchState(int match_state) {
  return match_state == STEREO_MATCH_EXACT ||
         match_state == STEREO_MATCH_APPROX ||
         match_state == STEREO_MATCH_STALE;
}

/** @copydoc ConvertRgbFrame */
bool ConvertRgbFrame(const stereo_camera_frame_t &frame,
                     sensor_msgs::msg::Image *message,
                     std::string *error) {
  const std::uint64_t pixels = PixelCount(frame);
  cv::Mat bgr;
  if (frame.pixel_format == kFourccYuyv) {
    if ((frame.width % 2) != 0 ||
        !ValidateFrame(frame, pixels * 2U, error)) {
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
    const cv::Mat nv12(frame.height * 3 / 2, frame.width, CV_8UC1,
                       frame.data);
    cv::cvtColor(nv12, bgr, cv::COLOR_YUV2BGR_NV12);
  } else {
    *error = "RGB frame FOURCC is neither YUYV nor NV12";
    return false;
  }

  message->height = static_cast<std::uint32_t>(frame.height);
  message->width = static_cast<std::uint32_t>(frame.width);
  message->encoding = "bgr8";
  message->is_bigendian = 0;
  message->step = static_cast<std::uint32_t>(frame.width * 3);
  message->data.assign(bgr.datastart, bgr.dataend);
  return true;
}

/** @copydoc CopyMono16Frame */
bool CopyMono16Frame(const stereo_camera_frame_t &frame,
                     const std::string &encoding,
                     sensor_msgs::msg::Image *message,
                     std::string *error) {
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
bool DecodeImuFrame(
    const stereo_camera_frame_t &frame,
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

/** @copydoc TimestampMapper::TimestampMapper */
TimestampMapper::TimestampMapper(std::int64_t sdk_anchor_us,
                                 std::int64_t ros_anchor_ns)
    : sdk_anchor_us_(sdk_anchor_us), ros_anchor_ns_(ros_anchor_ns) {}

/** @copydoc TimestampMapper::Map */
rclcpp::Time TimestampMapper::Map(std::uint64_t sdk_timestamp_us,
                                  const rclcpp::Time &fallback) const {
  if (sdk_anchor_us_ < 0 || sdk_timestamp_us == 0 ||
      sdk_timestamp_us >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    return fallback;
  }

  const std::int64_t sdk_timestamp =
      static_cast<std::int64_t>(sdk_timestamp_us);
  const std::int64_t delta_us = sdk_timestamp - sdk_anchor_us_;
  if (delta_us >
          std::numeric_limits<std::int64_t>::max() /
              kNanosecondsPerMicrosecond ||
      delta_us <
          std::numeric_limits<std::int64_t>::min() /
              kNanosecondsPerMicrosecond) {
    return fallback;
  }
  const std::int64_t delta_ns = delta_us * kNanosecondsPerMicrosecond;
  if ((delta_ns > 0 &&
       ros_anchor_ns_ >
           std::numeric_limits<std::int64_t>::max() - delta_ns) ||
      (delta_ns < 0 &&
       ros_anchor_ns_ <
           std::numeric_limits<std::int64_t>::min() - delta_ns)) {
    return fallback;
  }
  return rclcpp::Time(ros_anchor_ns_ + delta_ns, fallback.get_clock_type());
}

/** @copydoc TimestampMapper::MapImuSample */
rclcpp::Time TimestampMapper::MapImuSample(
    std::int64_t sdk_timestamp_us, const rclcpp::Time &fallback) const {
  if (sdk_timestamp_us <= 0) {
    return fallback;
  }
  return Map(static_cast<std::uint64_t>(sdk_timestamp_us), fallback);
}

}  // namespace tof_stereo_camera
