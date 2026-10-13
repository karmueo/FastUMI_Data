/**
 * @file test_frame_timestamper.cpp
 * @brief 验证缓冲区时间戳分类、映射、拒绝非单调时钟与兼容回调模式。
 */

#include <gtest/gtest.h>

#include <linux/videodev2.h>

#include <stdexcept>
#include <string>

#include "fastumi_usb_camera/clock_mapping.hpp"
#include "fastumi_usb_camera/frame_metadata.hpp"
#include "fastumi_usb_camera/frame_timestamper.hpp"

using fastumi_usb_camera::BufferClock;
using fastumi_usb_camera::BufferTimestampSource;
using fastumi_usb_camera::FrameMetadata;
using fastumi_usb_camera::FrameTimestamper;
using fastumi_usb_camera::MonotonicToRosClock;
using fastumi_usb_camera::TimestampMode;

namespace
{

constexpr uint32_t kMonotonicSoe =
  V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC | V4L2_BUF_FLAG_TSTAMP_SRC_SOE;  ///< uvcvideo 常见标志。
constexpr int64_t kOffset = 1700000000000000000;  ///< 测试用 ROS−单调时钟偏移。

/** @brief 偏移固定的映射，单调时钟每次读取前进 1 us。 */
struct FixedMapping
{
  int64_t monotonic{0};
  int64_t ros_now{0};
  MonotonicToRosClock mapper{
    [this]() {monotonic += 1000; return monotonic;},
    [this]() {return monotonic + kOffset;}};

  FixedMapping() {mapper.initialize();}
};

FrameMetadata frame(int64_t timestamp_ns, uint32_t flags = kMonotonicSoe, uint32_t sequence = 0)
{
  FrameMetadata metadata;
  metadata.timestamp_ns = timestamp_ns;
  metadata.flags = flags;
  metadata.sequence = sequence;
  return metadata;
}

}  // namespace

TEST(FrameMetadata, ClassifiesClockAndSource)
{
  EXPECT_EQ(fastumi_usb_camera::buffer_clock(kMonotonicSoe), BufferClock::kMonotonic);
  EXPECT_EQ(
    fastumi_usb_camera::buffer_timestamp_source(kMonotonicSoe),
    BufferTimestampSource::kStartOfExposure);
  EXPECT_EQ(
    fastumi_usb_camera::buffer_timestamp_source(V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC),
    BufferTimestampSource::kEndOfFrame);
  EXPECT_EQ(fastumi_usb_camera::buffer_clock(V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN), BufferClock::kUnknown);
  EXPECT_EQ(fastumi_usb_camera::buffer_clock(V4L2_BUF_FLAG_TIMESTAMP_COPY), BufferClock::kCopy);
  timeval value{12, 345678};
  EXPECT_EQ(fastumi_usb_camera::timeval_to_ns(value), 12345678000LL);
}

TEST(FrameTimestamper, MapsBufferTimeIndependentOfCallbackDelay)
{
  FixedMapping mapping;
  int64_t callback_ros = 0;
  FrameTimestamper stamper(
    TimestampMode::kV4l2Buffer, &mapping.mapper, [&]() {return callback_ros;});
  callback_ros = kOffset + 5000000000 + 40000000;  // 回调比缓冲区时间晚 40 ms。
  const auto stamped = stamper.stamp(frame(5000000000, kMonotonicSoe, 17));
  EXPECT_EQ(stamped.ros_ns, 5000000000 + mapping.mapper.offset_ns());
  EXPECT_EQ(stamped.v4l2_ns, 5000000000);
  EXPECT_EQ(stamped.sequence, 17U);
  EXPECT_EQ(stamped.callback_ros_ns, callback_ros);
  EXPECT_EQ(stamped.generation, 1U);
  // 人为增加出队延迟不改变源时间。
  callback_ros += 100000000;
  EXPECT_EQ(stamper.stamp(frame(5033333333)).ros_ns, 5033333333 + mapping.mapper.offset_ns());
  EXPECT_EQ(
    FrameTimestamper::source_label(TimestampMode::kV4l2Buffer, kMonotonicSoe),
    "v4l2_buffer_monotonic_soe_mapped_to_ros");
  EXPECT_EQ(
    FrameTimestamper::source_label(TimestampMode::kV4l2Buffer, V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC),
    "v4l2_buffer_monotonic_eof_mapped_to_ros");
}

TEST(FrameTimestamper, RejectsUnknownClockAndInvalidTimestamp)
{
  FixedMapping mapping;
  FrameTimestamper stamper(TimestampMode::kV4l2Buffer, &mapping.mapper, []() {return 0;});
  EXPECT_THROW(stamper.stamp(frame(5000000000, V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN)), std::runtime_error);
  EXPECT_THROW(stamper.stamp(frame(5000000000, V4L2_BUF_FLAG_TIMESTAMP_COPY)), std::runtime_error);
  EXPECT_THROW(stamper.stamp(frame(0)), std::runtime_error);
  EXPECT_THROW(
    FrameTimestamper(TimestampMode::kV4l2Buffer, nullptr, []() {return 0;}), std::invalid_argument);
}

TEST(FrameTimestamper, CountsNonMonotonicSourceTimes)
{
  FixedMapping mapping;
  FrameTimestamper stamper(TimestampMode::kV4l2Buffer, &mapping.mapper, []() {return 0;});
  stamper.stamp(frame(2000));
  stamper.stamp(frame(3000));
  stamper.stamp(frame(3000));
  stamper.stamp(frame(2500));
  EXPECT_EQ(stamper.non_monotonic_count(), 2U);
}

TEST(FrameTimestamper, CallbackModeIsExplicitAndLabelled)
{
  int64_t ros = 42;
  FrameTimestamper stamper(TimestampMode::kCallbackNow, nullptr, [&]() {return ros;});
  const auto stamped = stamper.stamp(frame(0, V4L2_BUF_FLAG_TIMESTAMP_UNKNOWN));
  EXPECT_EQ(stamped.ros_ns, 42);
  EXPECT_EQ(stamped.generation, 0U);
  EXPECT_EQ(FrameTimestamper::source_label(TimestampMode::kCallbackNow, 0U), "callback_now");
  EXPECT_THROW(fastumi_usb_camera::parse_timestamp_mode("now"), std::invalid_argument);
  EXPECT_EQ(fastumi_usb_camera::parse_timestamp_mode("callback_now"), TimestampMode::kCallbackNow);
}

TEST(FrameTimestamper, TimingTextContainsTraceFields)
{
  FixedMapping mapping;
  FrameTimestamper stamper(TimestampMode::kV4l2Buffer, &mapping.mapper, []() {return 7;});
  const auto stamped = stamper.stamp(frame(9000, kMonotonicSoe, 3));
  const std::string text = fastumi_usb_camera::timing_source_text(
    TimestampMode::kV4l2Buffer, stamped, 11, "unchanged(value=1)");
  for (const char * field : {"mode=v4l2_buffer", "clock=monotonic", "tsrc=soe", "seq=3",
      "recv_ns=7", "pub_ns=11", "gen=1", "exp_prio=unchanged(value=1)"})
  {
    EXPECT_NE(text.find(field), std::string::npos) << field;
  }
}
