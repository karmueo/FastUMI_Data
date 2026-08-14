/**
 * @file test_frame_utils.cpp
 * @brief ROS 2 帧转换工具的回归测试。
 * @author 待确认
 * @date 创建：待确认
 * @date 修改：2026-08-14
 */
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

#include "tof_stereo_camera/frame_utils.hpp"

namespace tof_stereo_camera {
namespace {

/**
 * @brief 按 V4L2 字节序生成测试所需的 FOURCC 数值。
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

/**
 * @brief 将 IMU 样本编码为连续的 SDK payload 测试数据。
 * @param[in] samples 待编码的 IMU 样本。
 * @return 独立持有样本字节副本的 payload。
 */
std::vector<unsigned char> EncodeImuSamples(
    const std::vector<stereo_camera_imu_data_t> &samples) {
  std::vector<unsigned char> payload(
      samples.size() * sizeof(stereo_camera_imu_data_t));
  std::memcpy(payload.data(), samples.data(), payload.size());
  return payload;
}

/** @brief 验证 `FrameUtils::ConvertsYuyvToBgr8` 所覆盖的帧转换行为。 */
TEST(FrameUtils, ConvertsYuyvToBgr8) {
  std::vector<unsigned char> payload{16, 128, 235, 128};
  stereo_camera_frame_t frame{};
  frame.width = 2;
  frame.height = 1;
  frame.pixel_format = MakeFourcc('Y', 'U', 'Y', 'V');
  frame.data = payload.data();
  frame.data_size = static_cast<int>(payload.size());

  sensor_msgs::msg::Image image;
  std::string error;
  ASSERT_TRUE(ConvertRgbFrame(frame, &image, &error));
  EXPECT_EQ(image.encoding, "bgr8");
  EXPECT_EQ(image.step, 6U);
  EXPECT_EQ(image.data.size(), 6U);
}

/** @brief 验证 `FrameUtils::RejectsWrongMono16PayloadSize` 所覆盖的帧转换行为。 */
TEST(FrameUtils, RejectsWrongMono16PayloadSize) {
  std::vector<unsigned char> payload(7, 0);
  stereo_camera_frame_t frame{};
  frame.width = 2;
  frame.height = 2;
  frame.data = payload.data();
  frame.data_size = static_cast<int>(payload.size());

  sensor_msgs::msg::Image image;
  std::string error;
  EXPECT_FALSE(CopyMono16Frame(frame, "16UC1", &image, &error));
  EXPECT_FALSE(error.empty());
}

/** @brief 验证 `FrameUtils::CopiesMono16PayloadBeforeSdkBufferChanges` 所覆盖的帧转换行为。 */
TEST(FrameUtils, CopiesMono16PayloadBeforeSdkBufferChanges) {
  std::vector<unsigned char> payload{1, 2, 3, 4};
  stereo_camera_frame_t frame{};
  frame.width = 2;
  frame.height = 1;
  frame.data = payload.data();
  frame.data_size = static_cast<int>(payload.size());

  sensor_msgs::msg::Image image;
  std::string error;
  ASSERT_TRUE(CopyMono16Frame(frame, "mono16", &image, &error));
  payload[0] = 99;
  EXPECT_EQ(image.data[0], 1U);
}

/** @brief 验证 `FrameUtils::AcceptsOnlyDocumentedTofMatchStates` 所覆盖的帧转换行为。 */
TEST(FrameUtils, AcceptsOnlyDocumentedTofMatchStates) {
  EXPECT_TRUE(IsPublishableTofMatchState(STEREO_MATCH_EXACT));
  EXPECT_TRUE(IsPublishableTofMatchState(STEREO_MATCH_APPROX));
  EXPECT_TRUE(IsPublishableTofMatchState(STEREO_MATCH_STALE));
  EXPECT_FALSE(IsPublishableTofMatchState(STEREO_MATCH_NONE));
  EXPECT_FALSE(IsPublishableTofMatchState(STEREO_MATCH_LOST));
}

/** @brief 验证 SDK 微秒时间戳会正确映射为 ROS 纳秒时间。 */
TEST(FrameUtils, MapsSdkMonotonicMicrosecondsAndUsesFallbackForZero) {
  TimestampMapper mapper(1'000, 10'000);
  const rclcpp::Time fallback(static_cast<std::int64_t>(50'000),
                              RCL_SYSTEM_TIME);
  EXPECT_EQ(mapper.Map(1'500, fallback).nanoseconds(), 510'000);
  EXPECT_EQ(mapper.Map(0, fallback).nanoseconds(), fallback.nanoseconds());
}

/** @brief 验证 `FrameUtils::DecodesSingleImuSampleFields` 所覆盖的帧转换行为。 */
TEST(FrameUtils, DecodesSingleImuSampleFields) {
  stereo_camera_imu_data_t sample{};
  sample.timestamp = 1'234;
  sample.ax = 1.0F;
  sample.ay = 2.0F;
  sample.az = 3.0F;
  sample.gx = 4.0F;
  sample.gy = 5.0F;
  sample.gz = 6.0F;
  sample.idx = 42;

  std::vector<unsigned char> payload = EncodeImuSamples({sample});
  stereo_camera_frame_t frame{};
  frame.data = payload.data();
  frame.data_size = static_cast<int>(payload.size());
  frame.frame_seq_count = 1;
  frame.frame_seqidx = 42;

  std::vector<stereo_camera_imu_data_t> decoded;
  std::string error;
  ASSERT_TRUE(DecodeImuFrame(frame, &decoded, &error));
  EXPECT_TRUE(error.empty());
  ASSERT_EQ(decoded.size(), 1U);
  EXPECT_EQ(decoded[0].timestamp, 1'234);
  EXPECT_FLOAT_EQ(decoded[0].ax, 1.0F);
  EXPECT_FLOAT_EQ(decoded[0].ay, 2.0F);
  EXPECT_FLOAT_EQ(decoded[0].az, 3.0F);
  EXPECT_FLOAT_EQ(decoded[0].gx, 4.0F);
  EXPECT_FLOAT_EQ(decoded[0].gy, 5.0F);
  EXPECT_FLOAT_EQ(decoded[0].gz, 6.0F);
}


/** @brief 验证 `FrameUtils::DecodesAndCopiesMultipleImuSamples` 所覆盖的帧转换行为。 */
TEST(FrameUtils, DecodesAndCopiesMultipleImuSamples) {
  stereo_camera_imu_data_t first{};
  first.timestamp = 1'000;
  first.ax = 1.0F;
  first.ay = 2.0F;
  first.az = 3.0F;
  first.gx = 4.0F;
  first.gy = 5.0F;
  first.gz = 6.0F;

  stereo_camera_imu_data_t second{};
  second.timestamp = 2'000;
  second.ax = -1.0F;
  second.gz = -6.0F;

  std::vector<unsigned char> payload = EncodeImuSamples({first, second});
  stereo_camera_frame_t frame{};
  frame.data = payload.data();
  frame.data_size = static_cast<int>(payload.size());

  std::vector<stereo_camera_imu_data_t> decoded;
  std::string error;
  ASSERT_TRUE(DecodeImuFrame(frame, &decoded, &error));
  ASSERT_EQ(decoded.size(), 2U);
  EXPECT_EQ(decoded[0].timestamp, 1'000);
  EXPECT_FLOAT_EQ(decoded[0].ax, 1.0F);
  EXPECT_FLOAT_EQ(decoded[0].gy, 5.0F);
  EXPECT_EQ(decoded[1].timestamp, 2'000);
  EXPECT_FLOAT_EQ(decoded[1].ax, -1.0F);
  EXPECT_FLOAT_EQ(decoded[1].gz, -6.0F);

  payload.assign(payload.size(), 0U);
  EXPECT_EQ(decoded[0].timestamp, 1'000);
  EXPECT_FLOAT_EQ(decoded[1].gz, -6.0F);
}

/** @brief 验证 `FrameUtils::RejectsMalformedImuPayload` 所覆盖的帧转换行为。 */
TEST(FrameUtils, RejectsMalformedImuPayload) {
  std::vector<unsigned char> payload(
      sizeof(stereo_camera_imu_data_t) + 1U, 0U);
  stereo_camera_frame_t frame{};
  frame.data = payload.data();
  frame.data_size = static_cast<int>(payload.size());

  std::vector<stereo_camera_imu_data_t> decoded;
  std::string error;
  EXPECT_FALSE(DecodeImuFrame(frame, &decoded, &error));
  EXPECT_TRUE(decoded.empty());
  EXPECT_FALSE(error.empty());
}

/** @brief 验证 IMU 微秒时间戳映射以及无效值回退。 */
TEST(FrameUtils, MapsImuSampleTimeAndFallsBackForInvalidValues) {
  TimestampMapper mapper(1'000, 10'000);
  const rclcpp::Time frame_time(static_cast<std::int64_t>(50'000),
                                RCL_SYSTEM_TIME);
  EXPECT_EQ(mapper.MapImuSample(1'500, frame_time).nanoseconds(), 510'000);
  EXPECT_EQ(mapper.MapImuSample(0, frame_time).nanoseconds(),
            frame_time.nanoseconds());
  EXPECT_EQ(mapper.MapImuSample(-1, frame_time).nanoseconds(),
            frame_time.nanoseconds());
}

/** @brief 验证微秒换算、锚点和 ROS 时间加法溢出时均安全回退。 */
TEST(FrameUtils, FallsBackWhenMicrosecondTimeMappingOverflows) {
  const rclcpp::Time fallback(static_cast<std::int64_t>(50'000),
                              RCL_SYSTEM_TIME);
  const std::int64_t maximum = std::numeric_limits<std::int64_t>::max();

  TimestampMapper positive_delta_overflow(1, 0);
  EXPECT_EQ(
      positive_delta_overflow
          .Map(static_cast<std::uint64_t>(maximum), fallback)
          .nanoseconds(),
      fallback.nanoseconds());

  TimestampMapper negative_delta_overflow(maximum, 0);
  EXPECT_EQ(negative_delta_overflow.Map(1, fallback).nanoseconds(),
            fallback.nanoseconds());

  TimestampMapper ros_time_overflow(1'000, maximum - 500);
  EXPECT_EQ(ros_time_overflow.Map(1'001, fallback).nanoseconds(),
            fallback.nanoseconds());

  TimestampMapper invalid_anchor(-1, 0);
  EXPECT_EQ(invalid_anchor.Map(1, fallback).nanoseconds(),
            fallback.nanoseconds());
}

/** @brief 验证 `FrameUtils::AcceptsEmptyImuBatchWithoutDiagnostic` 所覆盖的帧转换行为。 */
TEST(FrameUtils, AcceptsEmptyImuBatchWithoutDiagnostic) {
  stereo_camera_frame_t frame{};
  frame.data_size = 0;
  frame.frame_seq_count = 0;

  std::vector<stereo_camera_imu_data_t> decoded;
  std::string error;
  EXPECT_TRUE(DecodeImuFrame(frame, &decoded, &error));
  EXPECT_TRUE(decoded.empty());
  EXPECT_TRUE(error.empty());
}

/** @brief 验证 `FrameUtils::RejectsNullOrNegativeImuPayload` 所覆盖的帧转换行为。 */
TEST(FrameUtils, RejectsNullOrNegativeImuPayload) {
  stereo_camera_frame_t frame{};
  frame.data = nullptr;
  frame.data_size = static_cast<int>(sizeof(stereo_camera_imu_data_t));

  std::vector<stereo_camera_imu_data_t> decoded;
  std::string error;
  EXPECT_FALSE(DecodeImuFrame(frame, &decoded, &error));
  EXPECT_FALSE(error.empty());

  frame.data_size = -1;
  EXPECT_FALSE(DecodeImuFrame(frame, &decoded, &error));
  EXPECT_FALSE(error.empty());
}

/** @brief 验证 `FrameUtils::ReportsSequenceMetadataMismatchWithoutDroppingSamples` 所覆盖的帧转换行为。 */
TEST(FrameUtils, ReportsSequenceMetadataMismatchWithoutDroppingSamples) {
  stereo_camera_imu_data_t sample{};
  sample.idx = 42;
  sample.gx = 1.25F;
  std::vector<unsigned char> payload = EncodeImuSamples({sample});
  stereo_camera_frame_t frame{};
  frame.data = payload.data();
  frame.data_size = static_cast<int>(payload.size());
  frame.frame_seq_count = 2;
  frame.frame_seqidx = 41;

  std::vector<stereo_camera_imu_data_t> decoded;
  std::string error;
  EXPECT_TRUE(DecodeImuFrame(frame, &decoded, &error));
  ASSERT_EQ(decoded.size(), 1U);
  EXPECT_FLOAT_EQ(decoded.front().gx, 1.25F);
  EXPECT_NE(error.find("count mismatch"), std::string::npos);
  EXPECT_NE(error.find("index mismatch"), std::string::npos);
}
/** @brief 验证缺少帧序号元数据时，结构有效的 IMU 样本仍可正常解码。 */
TEST(FrameUtils, AcceptsMissingSequenceIndexMetadata) {
  stereo_camera_imu_data_t sample{};
  sample.idx = 42;
  std::vector<unsigned char> payload = EncodeImuSamples({sample});
  stereo_camera_frame_t frame{};
  frame.data = payload.data();
  frame.data_size = static_cast<int>(payload.size());
  frame.frame_seq_count = 1;
  frame.frame_seqidx = 0;

  std::vector<stereo_camera_imu_data_t> decoded;
  std::string error;
  ASSERT_TRUE(DecodeImuFrame(frame, &decoded, &error));
  ASSERT_EQ(decoded.size(), 1U);
  EXPECT_EQ(decoded.front().idx, 42);
  EXPECT_TRUE(error.empty());
}
}  // namespace
}  // namespace tof_stereo_camera
