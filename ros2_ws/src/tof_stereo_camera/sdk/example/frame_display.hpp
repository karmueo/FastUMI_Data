/**
 * @file frame_display.hpp
 * @brief 声明直接上传 SDK 原始图像并通过 GPU 显示的窗口组件。
 */

#ifndef STEREO_CAMERA_EXAMPLE_FRAME_DISPLAY_HPP_
#define STEREO_CAMERA_EXAMPLE_FRAME_DISPLAY_HPP_

#include "stereo_camera/stereo_camera.h"

#include <memory>
#include <string>

/**
 * @brief 保存显示布局及 iTOF 单通道图像的显示范围。
 */
struct FrameDisplayOptions {
  /// 深度图映射为可见灰度时采用的最大原始值。
  int depth_max = 5000;

  /// 灰度图映射为可见灰度时采用的最大原始值。
  int gray_max = 65535;

  /// 是否显示 iTOF Depth 和 iTOF Gray 窗格。
  bool show_itof = true;
};

/**
 * @brief 在 SDL2/OpenGL 窗口中显示 RGB 及可选的 iTOF 图像。
 *
 * RGB 与当前设备 iTOF 的 YUYV/NV12 payload 直接上传 GPU；其他固件返回
 * 16 位单通道 iTOF 时使用 R16UI 后备路径。颜色解释和灰度映射由 fragment
 * shader 完成，CPU 不生成 RGB/BGR 中间图像。
 */
class FrameDisplay {
public:
  /** @brief 构造尚未初始化的显示组件。 */
  FrameDisplay();

  /** @brief 释放 OpenGL、窗口和 SDL 资源。 */
  ~FrameDisplay();

  FrameDisplay(const FrameDisplay &) = delete;
  FrameDisplay &operator=(const FrameDisplay &) = delete;

  /**
   * @brief 创建窗口、OpenGL 上下文和 shader。
   *
   * @param[in] options 显示布局和单通道图像显示参数。
   * @param[out] error 初始化失败原因。
   * @return 初始化成功返回 true。
   */
  bool Initialize(const FrameDisplayOptions &options, std::string *error);

  /**
   * @brief 将一帧受支持的 SDK payload 直接上传到对应 GPU 纹理。
   *
   * @param[in] frame SDK 返回的当前帧，函数返回前完成 payload 上传。
   * @param[out] error 非空时接收被识别图像帧的校验或上传错误。
   * @return 帧属于 RGB/iTOF 显示路由且上传成功时返回 true；其他路返回
   * false 且 error 保持为空。
   */
  bool Update(const stereo_camera_frame_t &frame, std::string *error);

  /**
   * @brief 处理窗口事件。
   *
   * @return 用户关闭窗口或按下 Esc 时返回 false。
   */
  bool PollEvents();

  /** @brief 将当前已收到的启用纹理绘制到窗口。 */
  void Render();

private:
  /// 隐藏 SDL2/OpenGL 类型和实现细节。
  class Impl;

  /// 显示实现对象。
  std::unique_ptr<Impl> impl_;
};

#endif // STEREO_CAMERA_EXAMPLE_FRAME_DISPLAY_HPP_
