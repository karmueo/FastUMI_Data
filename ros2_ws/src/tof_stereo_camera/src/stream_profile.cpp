/**
 * @file stream_profile.cpp
 * @brief 实现主码流、子码流档位到 SDK 复合帧格式的固定映射。
 * @author 待确认
 * @date 创建：2026-08-20
 */

#include "tof_stereo_camera/stream_profile.hpp"

namespace tof_stereo_camera {
namespace {

/** @brief RGB-only 复合帧在有效 RGB 图像之外包含的设备帧头行数。 */
constexpr int kRgbOnlyHeaderRows = 2;

} // namespace

/** @copydoc ResolveStreamProfile */
bool ResolveStreamProfile(const std::string &name, StreamProfile *profile,
                          std::string *error) {
  if (profile == nullptr || error == nullptr) {
    return false;
  }
  error->clear();
  if (name == "main") {
    *profile = StreamProfile{2048, 2738, 2048, 1536};
    return true;
  }
  if (name == "sub") {
    *profile = StreamProfile{1920, 2362, 1920, 1080};
    return true;
  }
  *error = "stream_profile must be 'main' or 'sub'";
  return false;
}

/** @copydoc SelectCaptureFormat */
CaptureFormat SelectCaptureFormat(const StreamProfile &profile,
                                  bool enable_itof) {
  if (enable_itof) {
    return CaptureFormat{profile.composite_width, profile.composite_height};
  }
  return CaptureFormat{profile.rgb_width,
                       profile.rgb_height + kRgbOnlyHeaderRows};
}

} // namespace tof_stereo_camera
