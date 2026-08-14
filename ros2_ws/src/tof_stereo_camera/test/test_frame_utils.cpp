/**
 * @file test_frame_utils.cpp
 * @brief ROS 2 帧转换工具的回归测试。
 * @author 待确认
 * @date 创建：待确认
 * @date 修改：2026-08-14
 */
#include <cmath>
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
std::vector<unsigned char>
EncodeImuSamples(const std::vector<stereo_camera_imu_data_t> &samples) {
  std::vector<unsigned char> payload(samples.size() *
                                     sizeof(stereo_camera_imu_data_t));
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

/** @brief 验证 `FrameUtils::RejectsWrongMono16PayloadSize` 所覆盖的帧转换行为。
 */
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

/** @brief 验证 `FrameUtils::CopiesMono16PayloadBeforeSdkBufferChanges`
 * 所覆盖的帧转换行为。 */
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

/** @brief 验证 `FrameUtils::AcceptsOnlyDocumentedTofMatchStates`
 * 所覆盖的帧转换行为。 */
TEST(FrameUtils, AcceptsOnlyDocumentedTofMatchStates) {
  EXPECT_TRUE(IsPublishableTofMatchState(STEREO_MATCH_EXACT));
  EXPECT_TRUE(IsPublishableTofMatchState(STEREO_MATCH_APPROX));
  EXPECT_TRUE(IsPublishableTofMatchState(STEREO_MATCH_STALE));
  EXPECT_FALSE(IsPublishableTofMatchState(STEREO_MATCH_NONE));
  EXPECT_FALSE(IsPublishableTofMatchState(STEREO_MATCH_LOST));
}

/** @brief 验证 `FrameUtils::DecodesSingleImuSampleFields` 所覆盖的帧转换行为。
 */
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

/** @brief 验证 `FrameUtils::DecodesAndCopiesMultipleImuSamples`
 * 所覆盖的帧转换行为。 */
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
  std::vector<unsigned char> payload(sizeof(stereo_camera_imu_data_t) + 1U, 0U);
  stereo_camera_frame_t frame{};
  frame.data = payload.data();
  frame.data_size = static_cast<int>(payload.size());

  std::vector<stereo_camera_imu_data_t> decoded;
  std::string error;
  EXPECT_FALSE(DecodeImuFrame(frame, &decoded, &error));
  EXPECT_TRUE(decoded.empty());
  EXPECT_FALSE(error.empty());
}

/** @brief 验证 `FrameUtils::AcceptsEmptyImuBatchWithoutDiagnostic`
 * 所覆盖的帧转换行为。 */
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

/** @brief 验证 `FrameUtils::RejectsNullOrNegativeImuPayload`
 * 所覆盖的帧转换行为。 */
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

/** @brief 验证
 * `FrameUtils::ReportsSequenceMetadataMismatchWithoutDroppingSamples`
 * 所覆盖的帧转换行为。 */
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
/** @brief 验证独立 25.5 秒 epoch 在精确三帧启动标定后被消除。 */
TEST(TimestampMapper, LocksAfterExactBootstrapAndRemovesEpoch) {
  TimestampMapper mapper(0, 1'000'000'000'000'000'000LL, 3, 10, 200.0);
  const auto first = mapper.ObserveFrame(25'500'000, 50'000'000);
  const auto second = mapper.ObserveFrame(25'550'000, 60'000'000);
  const auto third = mapper.ObserveFrame(25'600'000, 130'000'000);
  EXPECT_EQ(first.status, FrameTimestampStatus::kCalibrating);
  EXPECT_EQ(second.status, FrameTimestampStatus::kCalibrating);
  ASSERT_EQ(third.status, FrameTimestampStatus::kReady);
  EXPECT_TRUE(third.ready);
  EXPECT_NEAR(third.steady_ns, 110'000'000, 1);
}

/** @brief 验证有效复合帧重复值不会推进启动状态并复用快照。 */
TEST(TimestampMapper, DuplicateValidFrameReusesSnapshot) {
  TimestampMapper mapper(0, 0, 3, 10, 200.0);
  const auto first = mapper.ObserveFrame(1'000, 2'000'000);
  const auto duplicate = mapper.ObserveFrame(1'000, 9'000'000);
  EXPECT_EQ(duplicate.status, FrameTimestampStatus::kDuplicate);
  EXPECT_FALSE(duplicate.ready);
  EXPECT_EQ(duplicate.steady_ns, first.steady_ns);
  EXPECT_EQ(duplicate.offset_ns, first.offset_ns);
  EXPECT_EQ(mapper.ObserveFrame(2'000, 3'000'000).status,
            FrameTimestampStatus::kCalibrating);
  EXPECT_EQ(mapper.ObserveFrame(3'000, 4'000'000).status,
            FrameTimestampStatus::kReady);
  const auto locked_duplicate = mapper.ObserveFrame(3'000, 9'000'000);
  EXPECT_EQ(locked_duplicate.status, FrameTimestampStatus::kDuplicate);
  EXPECT_FALSE(locked_duplicate.ready);
}

/** @brief 验证无效和溢出外层时间戳每次使用当次接收时间。 */
TEST(TimestampMapper, InvalidTimestampsAlwaysUseCurrentReceiveFallback) {
  TimestampMapper mapper(0, 100, 3, 10, 200.0);
  const std::uint64_t overflow =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() /
                                 1'000) +
      1U;
  std::int64_t receive = 10;
  for (const std::uint64_t timestamp :
       {std::uint64_t{0}, std::uint64_t{0}, overflow, overflow}) {
    const auto result = mapper.ObserveFrame(timestamp, receive);
    EXPECT_EQ(result.status, FrameTimestampStatus::kInvalidHostFallback);
    EXPECT_EQ(result.steady_ns, receive);
    ++receive;
  }
  EXPECT_FALSE(mapper.ready());
}

/** @brief 验证无效观测不会覆盖最近有效复合帧的重复缓存。 */
TEST(TimestampMapper, InvalidFramesDoNotEvictValidDuplicateCache) {
  TimestampMapper mapper(0, 0, 3, 10, 200.0);
  const auto first = mapper.ObserveFrame(1'000, 2'000'000);
  const auto zero = mapper.ObserveFrame(0, 7'000'000);
  const std::uint64_t overflow =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() /
                                 1'000) +
      1U;
  const auto invalid = mapper.ObserveFrame(overflow, 8'000'000);
  EXPECT_EQ(zero.steady_ns, 7'000'000);
  EXPECT_EQ(invalid.steady_ns, 8'000'000);
  const auto duplicate = mapper.ObserveFrame(1'000, 9'000'000);
  EXPECT_EQ(duplicate.status, FrameTimestampStatus::kDuplicate);
  EXPECT_EQ(duplicate.steady_ns, first.steady_ns);
  EXPECT_EQ(duplicate.offset_ns, first.offset_ns);
}

/** @brief 验证零时间戳回退也会维护跨 epoch 的发布单调下界。 */
TEST(TimestampMapper, ZeroFallbackPreventsRecoveredTimestampRegression) {
  TimestampMapper mapper(0, 0, 2, 10, 200.0);
  mapper.ObserveFrame(1'000, 2'000'000);
  ASSERT_TRUE(mapper.ObserveFrame(2'000, 3'000'000).ready);
  const auto fallback = mapper.ObserveFrame(0, 4'000'000);
  ASSERT_EQ(fallback.status, FrameTimestampStatus::kInvalidHostFallback);
  EXPECT_EQ(fallback.steady_ns, 4'000'000);
  const auto recovered = mapper.ObserveFrame(3'000, 4'000'000);
  EXPECT_EQ(recovered.status, FrameTimestampStatus::kDropped);
  EXPECT_FALSE(mapper.ready());
  EXPECT_EQ(mapper.ObserveFrame(4'000, 5'000'000).status,
            FrameTimestampStatus::kCalibrating);
  const auto relocked = mapper.ObserveFrame(5'000, 6'000'000);
  EXPECT_EQ(relocked.status, FrameTimestampStatus::kReady);
  EXPECT_GT(relocked.steady_ns, fallback.steady_ns);
}

/** @brief 验证溢出时间戳回退也会维护跨 epoch 的发布单调下界。 */
TEST(TimestampMapper, OverflowFallbackPreventsRecoveredTimestampRegression) {
  TimestampMapper mapper(0, 0, 2, 10, 200.0);
  const std::uint64_t overflow =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() /
                                 1'000) +
      1U;
  mapper.ObserveFrame(1'000, 2'000'000);
  ASSERT_TRUE(mapper.ObserveFrame(2'000, 3'000'000).ready);
  const auto fallback = mapper.ObserveFrame(overflow, 4'000'000);
  ASSERT_EQ(fallback.status, FrameTimestampStatus::kInvalidHostFallback);
  EXPECT_EQ(fallback.steady_ns, 4'000'000);
  const auto recovered = mapper.ObserveFrame(3'000, 4'000'000);
  EXPECT_EQ(recovered.status, FrameTimestampStatus::kDropped);
  EXPECT_FALSE(mapper.ready());
  EXPECT_EQ(mapper.ObserveFrame(4'000, 5'000'000).status,
            FrameTimestampStatus::kCalibrating);
  const auto relocked = mapper.ObserveFrame(5'000, 6'000'000);
  EXPECT_EQ(relocked.status, FrameTimestampStatus::kReady);
  EXPECT_GT(relocked.steady_ns, fallback.steady_ns);
}

/** @brief 验证 IMU 有效、无效和乱序样本均可独立映射或回退。 */
TEST(TimestampMapper, MapsMixedAndOutOfOrderImuSamplesWithoutDropping) {
  TimestampMapper mapper(0, 0, 2, 10, 200.0);
  mapper.ObserveFrame(1'000, 2'000'000);
  const auto frame = mapper.ObserveFrame(2'000, 3'000'000);
  const auto later = mapper.MapImuSample(1'900, frame);
  const auto invalid = mapper.MapImuSample(0, frame);
  const auto earlier = mapper.MapImuSample(1'100, frame);
  EXPECT_TRUE(later.valid_sample);
  EXPECT_FALSE(invalid.valid_sample);
  EXPECT_TRUE(earlier.valid_sample);
  EXPECT_EQ(invalid.steady_ns, frame.steady_ns);
}

/** @brief 验证交错的旧 iTOF 时间戳不会影响 RGB 独立同步器。 */
TEST(TimestampMapper, IndependentMappersLockDespiteInterleavedOldStreams) {
  TimestampMapper rgb(0, 0, 3, 10, 200.0);
  TimestampMapper depth(0, 0, 3, 10, 200.0);
  EXPECT_EQ(rgb.ObserveFrame(1'000, 2'000'000).status,
            FrameTimestampStatus::kCalibrating);
  EXPECT_EQ(depth.ObserveFrame(100, 2'000'000).status,
            FrameTimestampStatus::kCalibrating);
  EXPECT_EQ(rgb.ObserveFrame(2'000, 3'000'000).status,
            FrameTimestampStatus::kCalibrating);
  EXPECT_EQ(depth.ObserveFrame(100, 3'000'000).status,
            FrameTimestampStatus::kDuplicate);
  EXPECT_EQ(rgb.ObserveFrame(3'000, 4'000'000).status,
            FrameTimestampStatus::kReady);
  EXPECT_EQ(depth.ObserveFrame(200, 4'000'000).status,
            FrameTimestampStatus::kCalibrating);
  EXPECT_EQ(depth.ObserveFrame(300, 5'000'000).status,
            FrameTimestampStatus::kReady);
  EXPECT_TRUE(rgb.ready());
  EXPECT_TRUE(depth.ready());
}

/** @brief 验证回退、失配重置以及固定快照 IMU 钳制。 */
TEST(TimestampMapper, ResetsRelocksAndMapsImuSnapshot) {
  TimestampMapper mapper(0, 0, 3, 10, 200.0);
  mapper.ObserveFrame(1'000, 2'000'000);
  mapper.ObserveFrame(2'000, 3'000'000);
  const auto before_reset = mapper.ObserveFrame(3'000, 4'000'000);
  ASSERT_EQ(before_reset.status, FrameTimestampStatus::kReady);
  EXPECT_EQ(mapper.ObserveFrame(2'000, 5'000'000).status,
            FrameTimestampStatus::kReset);
  mapper.ObserveFrame(3'000, 6'000'000);
  const auto relocked = mapper.ObserveFrame(4'000, 7'000'000);
  EXPECT_EQ(relocked.status, FrameTimestampStatus::kReady);
  EXPECT_GT(relocked.steady_ns, before_reset.steady_ns);
  const auto frame = mapper.ObserveFrame(5'000, 8'000'000);
  const auto earlier = mapper.MapImuSample(4'000, frame);
  EXPECT_TRUE(earlier.valid_sample);
  EXPECT_LT(earlier.steady_ns, frame.steady_ns);
  EXPECT_FALSE(mapper.MapImuSample(0, frame).valid_sample);
  const auto future = mapper.MapImuSample(9'000, frame);
  EXPECT_TRUE(future.future_clamped);
  EXPECT_EQ(future.steady_ns, frame.steady_ns);
}

/** @brief 验证参数边界会拒绝无效同步设置。 */
TEST(TimestampMapper, RejectsInvalidParameters) {
  EXPECT_THROW(TimestampMapper(0, 0, 1, 1, 200.0), std::invalid_argument);
  EXPECT_THROW(TimestampMapper(0, 0, 3, 2, 200.0), std::invalid_argument);
  EXPECT_THROW(TimestampMapper(0, 0, 3, 3, 0.0), std::invalid_argument);
  EXPECT_THROW(TimestampMapper(0, 0, 3, 3, -1.0), std::invalid_argument);
  EXPECT_THROW(TimestampMapper(0, 0, 3, 3, 1'000'000.0), std::invalid_argument);
  EXPECT_THROW(
      TimestampMapper(0, 0, 3, 3, std::numeric_limits<double>::infinity()),
      std::invalid_argument);
}

/** @brief 验证设备与接收间隔失配会重置并在剩余两帧后重新锁定。 */
TEST(TimestampMapper, DeltaMismatchResetsAndRelocks) {
  TimestampMapper mapper(0, 0, 3, 10, 200.0);
  mapper.ObserveFrame(1'000, 2'000'000);
  mapper.ObserveFrame(2'000, 3'000'000);
  ASSERT_TRUE(mapper.ObserveFrame(3'000, 4'000'000).ready);
  const auto reset = mapper.ObserveFrame(1'003'000, 3'004'000'000);
  EXPECT_EQ(reset.status, FrameTimestampStatus::kReset);
  EXPECT_FALSE(mapper.ready());
  EXPECT_EQ(mapper.ObserveFrame(1'004'000, 3'005'000'000).status,
            FrameTimestampStatus::kCalibrating);
  EXPECT_EQ(mapper.ObserveFrame(1'005'000, 3'006'000'000).status,
            FrameTimestampStatus::kReady);
}

/** @brief 验证单调下界与接收上界冲突会丢弃并重新开始标定。 */
TEST(TimestampMapper, MonotonicReceiveConflictDropsAndResets) {
  TimestampMapper mapper(0, 0, 2, 10, 200.0);
  mapper.ObserveFrame(1'000, 2'000'000);
  const auto locked = mapper.ObserveFrame(2'000, 3'000'000);
  ASSERT_TRUE(locked.ready);
  const auto dropped = mapper.ObserveFrame(2'001, 3'000'000);
  EXPECT_EQ(dropped.status, FrameTimestampStatus::kDropped);
  EXPECT_FALSE(mapper.ready());
  EXPECT_EQ(mapper.ObserveFrame(3'001, 4'000'000).status,
            FrameTimestampStatus::kCalibrating);
  EXPECT_EQ(mapper.ObserveFrame(4'001, 5'000'000).status,
            FrameTimestampStatus::kReady);
}

/** @brief 验证 -31 ppm 漂移在十分钟滚动同步中保持五毫秒内误差。 */
TEST(TimestampMapper, TracksNegativeThirtyOnePpmDriftForTenMinutes) {
  TimestampMapper mapper(0, 0, 30, 120, 200.0);
  for (int index = 0; index <= 12'000; ++index) {
    const std::int64_t event_ns = static_cast<std::int64_t>(index) * 50'000'000;
    const int jitter_ms = (index % 5) * 25;
    const std::int64_t receive_ns = event_ns + jitter_ms * 1'000'000LL;
    const std::int64_t device_us = static_cast<std::int64_t>(std::llround(
        25'500'000.0 + static_cast<double>(event_ns) * 0.999969 / 1'000.0));
    const auto result =
        mapper.ObserveFrame(static_cast<std::uint64_t>(device_us), receive_ns);
    if (result.ready) {
      EXPECT_LE(std::llabs(result.steady_ns - event_ns), 5'000'000);
      EXPECT_LE(result.steady_ns, receive_ns);
    }
    EXPECT_NE(result.status, FrameTimestampStatus::kReset);
    EXPECT_NE(result.status, FrameTimestampStatus::kDropped);
  }
}

} // namespace
} // namespace tof_stereo_camera
