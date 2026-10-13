/**
 * @file frame_metadata.hpp
 * @brief 定义 V4L2 缓冲区帧元数据，并按缓冲区标志区分时间戳时钟与采样点。
 *
 * ``v4l2_buffer.timestamp`` 的时钟由 ``V4L2_BUF_FLAG_TIMESTAMP_MASK`` 声明（单调时钟、
 * 复制或未知），采样点由 ``V4L2_BUF_FLAG_TSTAMP_SRC_MASK`` 声明（帧结束 EOF 或曝光开始
 * SOE）。SOE 只是驱动的语义声明，不能证明设备提供了真实曝光开始的硬件时间。
 */

#ifndef FASTUMI_USB_CAMERA__FRAME_METADATA_HPP_
#define FASTUMI_USB_CAMERA__FRAME_METADATA_HPP_

#include <linux/videodev2.h>
#include <sys/time.h>

#include <cstdint>
#include <string>

namespace fastumi_usb_camera
{

/** @brief 一帧出队时从 V4L2 缓冲区取得的源时间与序号。 */
struct FrameMetadata
{
  int64_t timestamp_ns{0};  ///< ``v4l2_buffer.timestamp`` 换算的纳秒值，时钟见 ``flags``。
  uint32_t sequence{0};  ///< 驱动帧序号，可用于发现驱动层丢帧。
  uint32_t flags{0};  ///< ``v4l2_buffer.flags`` 原值，保留时钟与 SOE/EOF 标志。
};

/** @brief 缓冲区时间戳使用的时钟。 */
enum class BufferClock
{
  kMonotonic,  ///< ``CLOCK_MONOTONIC``，可映射到 ROS 时间。
  kCopy,  ///< 输出设备复制的时间，采集设备不应出现。
  kUnknown,  ///< 驱动未声明时钟，不能用于映射。
};

/** @brief 缓冲区时间戳的采样点。 */
enum class BufferTimestampSource
{
  kEndOfFrame,  ///< EOF：帧最后一个数据到达时（标志位为 0 时的默认语义）。
  kStartOfExposure,  ///< SOE：驱动声明的曝光开始。
};

/** @return 由缓冲区标志声明的时间戳时钟。 */
inline BufferClock buffer_clock(uint32_t flags)
{
  switch (flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) {
    case V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC:
      return BufferClock::kMonotonic;
    case V4L2_BUF_FLAG_TIMESTAMP_COPY:
      return BufferClock::kCopy;
    default:
      return BufferClock::kUnknown;
  }
}

/** @return 由缓冲区标志声明的时间戳采样点。 */
inline BufferTimestampSource buffer_timestamp_source(uint32_t flags)
{
  return (flags & V4L2_BUF_FLAG_TSTAMP_SRC_MASK) == V4L2_BUF_FLAG_TSTAMP_SRC_SOE ?
         BufferTimestampSource::kStartOfExposure : BufferTimestampSource::kEndOfFrame;
}

/** @return 时钟名称：``monotonic``、``copy`` 或 ``unknown``。 */
inline std::string buffer_clock_name(uint32_t flags)
{
  switch (buffer_clock(flags)) {
    case BufferClock::kMonotonic:
      return "monotonic";
    case BufferClock::kCopy:
      return "copy";
    default:
      return "unknown";
  }
}

/** @return 采样点名称：``soe`` 或 ``eof``。 */
inline std::string buffer_timestamp_source_name(uint32_t flags)
{
  return buffer_timestamp_source(flags) == BufferTimestampSource::kStartOfExposure ? "soe" : "eof";
}

/** @return ``timeval`` 换算的纳秒值。 */
inline int64_t timeval_to_ns(const timeval & value)
{
  return static_cast<int64_t>(value.tv_sec) * 1000000000LL +
         static_cast<int64_t>(value.tv_usec) * 1000LL;
}

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__FRAME_METADATA_HPP_
