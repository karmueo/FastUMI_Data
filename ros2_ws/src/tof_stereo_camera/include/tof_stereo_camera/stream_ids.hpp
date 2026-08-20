/**
 * @file stream_ids.hpp
 * @brief 定义 stereo_camera 设备协议使用的视频流 ID。
 * @author 待确认
 * @date 创建：2026-08-20
 */

#ifndef TOF_STEREO_CAMERA__STREAM_IDS_HPP_
#define TOF_STEREO_CAMERA__STREAM_IDS_HPP_

namespace tof_stereo_camera {

constexpr int kRgbStreamId = 0; ///< RGB 视频流在 SDK 中使用的流 ID。
constexpr int kItofDepthStreamId = 2; ///< iTOF 深度图在 SDK 中使用的流 ID。
constexpr int kItofGrayStreamId = 7; ///< iTOF 灰度图在 SDK 中使用的流 ID。

} // namespace tof_stereo_camera

#endif // TOF_STEREO_CAMERA__STREAM_IDS_HPP_
