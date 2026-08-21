/**
 * @file test_stream_profile.cpp
 * @brief 主码流、子码流复合帧格式映射的无硬件回归测试。
 * @author 待确认
 * @date 创建：2026-08-20
 */

#include <string>

#include "gtest/gtest.h"

#include "tof_stereo_camera/stream_profile.hpp"

namespace tof_stereo_camera {
namespace {

/** @brief 验证主码流映射到 2048 系完整复合帧。 */
TEST(StreamProfile, ResolvesMainProfile) {
  StreamProfile profile; ///< 接收主码流映射结果。
  std::string error;     ///< 接收意外错误。
  ASSERT_TRUE(ResolveStreamProfile("main", &profile, &error)) << error;
  EXPECT_EQ(profile.composite_width, 2048);
  EXPECT_EQ(profile.composite_height, 2738);
  EXPECT_EQ(profile.rgb_width, 2048);
  EXPECT_EQ(profile.rgb_height, 1536);

  const CaptureFormat rgb_only =
      SelectCaptureFormat(profile, false); ///< 主码流 RGB-only 采集尺寸。
  EXPECT_EQ(rgb_only.width, 2048);
  EXPECT_EQ(rgb_only.height, 1538);

  const CaptureFormat with_itof =
      SelectCaptureFormat(profile, true); ///< 主码流完整复合帧尺寸。
  EXPECT_EQ(with_itof.width, 2048);
  EXPECT_EQ(with_itof.height, 2738);
}

/** @brief 验证子码流映射到 1920 系完整复合帧。 */
TEST(StreamProfile, ResolvesSubProfile) {
  StreamProfile profile; ///< 接收子码流映射结果。
  std::string error;     ///< 接收意外错误。
  ASSERT_TRUE(ResolveStreamProfile("sub", &profile, &error)) << error;
  EXPECT_EQ(profile.composite_width, 1920);
  EXPECT_EQ(profile.composite_height, 2362);
  EXPECT_EQ(profile.rgb_width, 1920);
  EXPECT_EQ(profile.rgb_height, 1080);

  const CaptureFormat rgb_only =
      SelectCaptureFormat(profile, false); ///< 子码流 RGB-only 采集尺寸。
  EXPECT_EQ(rgb_only.width, 1920);
  EXPECT_EQ(rgb_only.height, 1082);

  const CaptureFormat with_itof =
      SelectCaptureFormat(profile, true); ///< 子码流完整复合帧尺寸。
  EXPECT_EQ(with_itof.width, 1920);
  EXPECT_EQ(with_itof.height, 2362);
}

/** @brief 验证非法档位名称会得到清晰诊断。 */
TEST(StreamProfile, RejectsUnknownProfile) {
  StreamProfile profile; ///< 接收不应生成的档位结果。
  std::string error;     ///< 接收非法档位诊断。
  EXPECT_FALSE(ResolveStreamProfile("1920x1080", &profile, &error));
  EXPECT_NE(error.find("main"), std::string::npos);
  EXPECT_NE(error.find("sub"), std::string::npos);
}

/** @brief 验证空输出参数被安全拒绝。 */
TEST(StreamProfile, RejectsNullOutputs) {
  StreamProfile profile; ///< 提供给空错误参数场景的有效输出对象。
  std::string error; ///< 提供给空档位参数场景的有效错误对象。
  EXPECT_FALSE(ResolveStreamProfile("main", nullptr, &error));
  EXPECT_FALSE(ResolveStreamProfile("main", &profile, nullptr));
}

} // namespace
} // namespace tof_stereo_camera
