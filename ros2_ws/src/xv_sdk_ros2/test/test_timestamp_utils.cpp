/**
 * @file test_timestamp_utils.cpp
 * @brief 验证 SDK 时间戳对齐与 ROS 时间戳转换规则。
 */

#include "timestamp_utils.h"

#include <gtest/gtest.h>

/**
 * @brief 整数秒应转换为整秒时间戳。
 */
TEST(TimestampUtilsTest, ConvertsIntegerSeconds) {
  /** 转换后的 ROS 时间戳。 */
  const builtin_interfaces::msg::Time stamp =
      xv_ros2::timestamp::hostTimestampToRosTime(42.0);

  EXPECT_EQ(42, stamp.sec);
  EXPECT_EQ(0U, stamp.nanosec);
}

/**
 * @brief 小数秒应拆分为秒和纳秒。
 */
TEST(TimestampUtilsTest, ConvertsFractionalSeconds) {
  /** 转换后的 ROS 时间戳。 */
  const builtin_interfaces::msg::Time stamp =
      xv_ros2::timestamp::hostTimestampToRosTime(42.25);

  EXPECT_EQ(42, stamp.sec);
  EXPECT_EQ(250000000U, stamp.nanosec);
}

/**
 * @brief 过小和负数时间戳应按 0.1 秒保护。
 */
TEST(TimestampUtilsTest, ClampsSmallAndNegativeSeconds) {
  /** 过小正数转换后的 ROS 时间戳。 */
  const builtin_interfaces::msg::Time small_stamp =
      xv_ros2::timestamp::hostTimestampToRosTime(0.01);
  /** 负数转换后的 ROS 时间戳。 */
  const builtin_interfaces::msg::Time negative_stamp =
      xv_ros2::timestamp::hostTimestampToRosTime(-1.0);

  EXPECT_EQ(0, small_stamp.sec);
  EXPECT_EQ(100000000U, small_stamp.nanosec);
  EXPECT_EQ(0, negative_stamp.sec);
  EXPECT_EQ(100000000U, negative_stamp.nanosec);
}

/**
 * @brief 相同输入应生成完全一致的 ROS 时间戳。
 */
TEST(TimestampUtilsTest, SameInputProducesSameStamp) {
  /** 第一次转换后的 ROS 时间戳。 */
  const builtin_interfaces::msg::Time first_stamp =
      xv_ros2::timestamp::hostTimestampToRosTime(123.456);
  /** 第二次转换后的 ROS 时间戳。 */
  const builtin_interfaces::msg::Time second_stamp =
      xv_ros2::timestamp::hostTimestampToRosTime(123.456);

  EXPECT_EQ(first_stamp.sec, second_stamp.sec);
  EXPECT_EQ(first_stamp.nanosec, second_stamp.nanosec);
}

/**
 * @brief 已使用主机 steady_clock 的数据流应保持原始时间戳。
 */
TEST(StreamTimestampAlignerTest, PassesThroughHostClockTimestamp) {
  /** 使用三个样本完成标定的对齐器。 */
  xv_ros2::timestamp::StreamTimestampAligner aligner(3, 1.0);

  EXPECT_DOUBLE_EQ(100.0, aligner.align(100.0, 100.02));
  EXPECT_FALSE(aligner.compensationEnabled());
  EXPECT_TRUE(aligner.locked());
  EXPECT_DOUBLE_EQ(0.0, aligner.offsetSeconds());
}

/**
 * @brief 独立时间原点应使用启动样本中的最小偏移并在指定样本数后锁定。
 */
TEST(StreamTimestampAlignerTest, EstimatesAndLocksIndependentClockOffset) {
  /** 使用三个样本完成标定的对齐器。 */
  xv_ros2::timestamp::StreamTimestampAligner aligner(3, 1.0);

  EXPECT_NEAR(110.03, aligner.align(10.0, 110.03), 1e-9);
  EXPECT_NEAR(111.01, aligner.align(11.0, 111.01), 1e-9);
  EXPECT_NEAR(112.01, aligner.align(12.0, 112.02), 1e-9);
  EXPECT_TRUE(aligner.compensationEnabled());
  EXPECT_TRUE(aligner.locked());
  EXPECT_NEAR(100.01, aligner.offsetSeconds(), 1e-9);

  EXPECT_NEAR(113.01, aligner.align(13.0, 113.001), 1e-9);
  EXPECT_NEAR(100.01, aligner.offsetSeconds(), 1e-9);
}

/**
 * @brief 标定期间偏移下降时输出时间戳仍应保持单调递增。
 */
TEST(StreamTimestampAlignerTest, KeepsAlignedTimestampMonotonic) {
  /** 使用三个样本完成标定的对齐器。 */
  xv_ros2::timestamp::StreamTimestampAligner aligner(3, 1.0);
  /** 第一个对齐后的时间戳。 */
  const double first_timestamp = aligner.align(10.0, 110.1);
  /** 偏移候选值明显下降后的第二个时间戳。 */
  const double second_timestamp = aligner.align(10.001, 109.9);

  EXPECT_GT(second_timestamp, first_timestamp);
}

/**
 * @brief 无效源时间戳应回退到主机当前时间。
 */
TEST(StreamTimestampAlignerTest, InvalidSourceTimestampUsesHostNow) {
  /** 默认时间戳对齐器。 */
  xv_ros2::timestamp::StreamTimestampAligner aligner;

  EXPECT_DOUBLE_EQ(123.456, aligner.align(-1.0, 123.456));
}

/**
 * @brief 无效源时间戳回退后，后续有效样本也应保持严格单调递增。
 */
TEST(StreamTimestampAlignerTest, TracksFallbackTimestampForMonotonicity) {
  /** 使用三个样本完成标定的对齐器。 */
  xv_ros2::timestamp::StreamTimestampAligner aligner(3, 1.0);
  /** 初始化偏移估计后的时间戳。 */
  const double initial_timestamp = aligner.align(10.0, 110.1);
  /** 无效源时间戳触发的主机时间回退值。 */
  const double fallback_timestamp = aligner.align(0.0, 110.5);
  /** 传输延迟降低后的有效样本时间戳。 */
  const double resumed_timestamp = aligner.align(10.2, 110.2);

  EXPECT_GT(fallback_timestamp, initial_timestamp);
  EXPECT_GT(resumed_timestamp, fallback_timestamp);
}
