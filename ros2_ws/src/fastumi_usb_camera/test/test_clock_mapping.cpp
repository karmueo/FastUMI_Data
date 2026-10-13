/**
 * @file test_clock_mapping.cpp
 * @brief 验证单调时钟→ROS 时间映射的成对采样、误差界、跟随与跳变检测。
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "fastumi_usb_camera/clock_mapping.hpp"

using fastumi_usb_camera::ClockCheckStatus;
using fastumi_usb_camera::ClockMappingConfig;
using fastumi_usb_camera::MonotonicToRosClock;
using fastumi_usb_camera::sample_clock_offset;

namespace
{

/** @brief 可控时钟：每次读取单调时钟前进 step 纳秒，ROS 读数对应两次单调读数的中点。 */
struct FakeClocks
{
  int64_t monotonic{1000000000};
  int64_t step{1000};
  int64_t ros_offset{1700000000000000000};

  int64_t read_monotonic()
  {
    monotonic += step;
    return monotonic;
  }

  int64_t read_ros() const {return monotonic + step / 2 + ros_offset;}
};

}  // namespace

TEST(ClockMapping, MidpointSampleRecoversOffsetAndBound)
{
  FakeClocks clocks;
  const auto sample = sample_clock_offset(
    [&]() {return clocks.read_monotonic();}, [&]() {return clocks.read_ros();}, 3);
  // ROS 读数位于两次单调读数中点：误差界为一个步长，偏移精确。
  EXPECT_EQ(sample.uncertainty_ns, clocks.step);
  EXPECT_EQ(sample.offset_ns, clocks.ros_offset);
  EXPECT_THROW(
    sample_clock_offset([]() {return 0;}, []() {return 0;}, 0), std::invalid_argument);
}

TEST(ClockMapping, MapsMonotonicTimestampsAndTracksSlowSlew)
{
  FakeClocks clocks;
  MonotonicToRosClock mapper(
    [&]() {return clocks.read_monotonic();}, [&]() {return clocks.read_ros();});
  mapper.initialize();
  EXPECT_EQ(mapper.generation(), 1U);
  EXPECT_EQ(mapper.to_ros_ns(5000000000), 5000000000 + clocks.ros_offset);
  clocks.ros_offset += 500000;  // NTP 缓慢校正 0.5 ms，低于 2 ms 跳变门限。
  const auto result = mapper.check();
  EXPECT_EQ(result.status, ClockCheckStatus::kOk);
  EXPECT_EQ(result.offset_change_ns, 500000);
  EXPECT_EQ(mapper.generation(), 1U);
  EXPECT_EQ(mapper.to_ros_ns(0), clocks.ros_offset);
}

TEST(ClockMapping, SystemTimeJumpRebuildsMappingWithNewGeneration)
{
  FakeClocks clocks;
  MonotonicToRosClock mapper(
    [&]() {return clocks.read_monotonic();}, [&]() {return clocks.read_ros();});
  mapper.initialize();
  clocks.ros_offset -= 3000000000;  // 系统时间回拨 3 s。
  const auto result = mapper.check();
  EXPECT_EQ(result.status, ClockCheckStatus::kJump);
  EXPECT_EQ(result.offset_change_ns, -3000000000);
  EXPECT_EQ(result.generation, 2U);
  EXPECT_EQ(mapper.to_ros_ns(0), clocks.ros_offset);
}

TEST(ClockMapping, UncertainSampleKeepsPreviousOffset)
{
  FakeClocks clocks;
  ClockMappingConfig config;
  config.max_uncertainty_ns = 10000;
  MonotonicToRosClock mapper(
    [&]() {return clocks.read_monotonic();}, [&]() {return clocks.read_ros();}, config);
  mapper.initialize();
  const int64_t offset = mapper.offset_ns();
  clocks.step = 50000;  // 调度抖动使区间宽度超过 10 us。
  clocks.ros_offset += 100000;
  const auto result = mapper.check();
  EXPECT_EQ(result.status, ClockCheckStatus::kUncertain);
  EXPECT_EQ(mapper.offset_ns(), offset);
  EXPECT_EQ(mapper.generation(), 1U);
}

TEST(ClockMapping, InitializationFailsWhenBoundExceedsLimit)
{
  FakeClocks clocks;
  clocks.step = 1000000;
  ClockMappingConfig config;
  config.max_uncertainty_ns = 100000;
  MonotonicToRosClock mapper(
    [&]() {return clocks.read_monotonic();}, [&]() {return clocks.read_ros();}, config);
  EXPECT_THROW(mapper.initialize(), std::runtime_error);
  EXPECT_THROW(mapper.check(), std::logic_error);
}

TEST(ClockMapping, RealClocksMapWithinMicroseconds)
{
  MonotonicToRosClock mapper(
    fastumi_usb_camera::monotonic_now_ns,
    []() {
      timespec value{};
      clock_gettime(CLOCK_REALTIME, &value);
      return static_cast<int64_t>(value.tv_sec) * 1000000000LL + value.tv_nsec;
    });
  mapper.initialize();
  EXPECT_LT(mapper.uncertainty_ns(), 200000);
  EXPECT_EQ(mapper.check().status, ClockCheckStatus::kOk);
}
