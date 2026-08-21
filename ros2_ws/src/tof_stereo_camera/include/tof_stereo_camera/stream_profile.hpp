/**
 * @file stream_profile.hpp
 * @brief 声明用户码流档位到 SDK 复合帧格式的映射接口。
 * @author 待确认
 * @date 创建：2026-08-20
 */

#ifndef TOF_STEREO_CAMERA__STREAM_PROFILE_HPP_
#define TOF_STEREO_CAMERA__STREAM_PROFILE_HPP_

#include <string>

namespace tof_stereo_camera {

/** @brief 保存一个码流档位对应的复合帧及 RGB 有效图像尺寸。 */
struct StreamProfile {
  int composite_width = 0;  ///< SDK 完整复合帧宽度。
  int composite_height = 0; ///< SDK 完整复合帧高度。
  int rgb_width = 0;        ///< 解析后的 RGB 有效图像宽度。
  int rgb_height = 0;       ///< 解析后的 RGB 有效图像高度。
};

/** @brief 保存传给 SDK 的实际 UVC 采集尺寸。 */
struct CaptureFormat {
  int width = 0;  ///< UVC 采集帧宽度。
  int height = 0; ///< UVC 采集帧高度。
};

/**
 * @brief 将 ROS 码流档位名称解析为固定的 SDK 复合帧格式。
 * @param[in] name 码流档位名称，允许 `main` 或 `sub`。
 * @param[out] profile 接收档位尺寸，不可为 `nullptr`。
 * @param[out] error 接收失败原因，不可为 `nullptr`。
 * @return 档位名称和输出参数有效时返回 true。
 */
bool ResolveStreamProfile(const std::string &name, StreamProfile *profile,
                          std::string *error);

/**
 * @brief 根据 iTOF 启用状态选择实际 UVC 采集尺寸。
 * @param[in] profile 已解析的主码流或子码流档位。
 * @param[in] enable_itof 是否启用任意 iTOF 图像流。
 * @return 启用 iTOF 时返回完整复合帧尺寸，否则返回带两行帧头的 RGB 尺寸。
 */
CaptureFormat SelectCaptureFormat(const StreamProfile &profile,
                                  bool enable_itof);

} // namespace tof_stereo_camera

#endif // TOF_STEREO_CAMERA__STREAM_PROFILE_HPP_
