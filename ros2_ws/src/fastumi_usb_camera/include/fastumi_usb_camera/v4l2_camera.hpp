/**
 * @file v4l2_camera.hpp
 * @brief 定义基于内核 V4L2 mmap 的 USB MJPEG 相机资源管理接口。
 */

#ifndef FASTUMI_USB_CAMERA__V4L2_CAMERA_HPP_
#define FASTUMI_USB_CAMERA__V4L2_CAMERA_HPP_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "fastumi_usb_camera/configuration.hpp"

namespace fastumi_usb_camera
{

/**
 * @brief 解析显式视频路径，或选择编号最小的 USB 主视频节点。
 * @param[in] config 相机设备选择配置。
 * @param[in] video_class_root video4linux sysfs 类目录。
 * @param[in] device_root 视频字符设备目录。
 * @return 可由 V4L2 打开的设备路径。
 * @throws std::runtime_error 设备不存在、sysfs 不可读或未找到 USB 主视频节点。
 */
std::string discover_video_device(
  const CameraConfiguration & config,
  const std::filesystem::path & video_class_root = "/sys/class/video4linux",
  const std::filesystem::path & device_root = "/dev");

/** @brief 通过 uvcvideo 的 V4L2 mmap 队列采集原生 MJPEG 帧。 */
class V4l2Camera
{
public:
  /** @brief 完整 MJPEG 帧回调类型。 */
  using FrameCallback = std::function<void(const uint8_t *, size_t)>;

  /**
   * @brief 打开设备、严格协商模式并启动采集线程。
   * @param[in] config 相机设备及采集模式配置。
   * @param[in] callback 每个完整 MJPEG 帧到达时调用的函数。
   * @throws std::runtime_error 设备打开、模式协商或流启动失败。
   */
  V4l2Camera(const CameraConfiguration & config, FrameCallback callback);

  V4l2Camera(const V4l2Camera &) = delete;
  V4l2Camera & operator=(const V4l2Camera &) = delete;
  V4l2Camera(V4l2Camera &&) = delete;
  V4l2Camera & operator=(V4l2Camera &&) = delete;

  /** @brief 停止采集并释放所有 V4L2 资源。 */
  ~V4l2Camera();

  /** @brief 可重复调用地停止采集线程、关闭视频流并释放设备。 */
  void stop() noexcept;

  /** @return 采集线程是否因不可恢复错误退出。 */
  bool failed() const noexcept;

  /** @return 采集线程记录的最后错误说明。 */
  std::string error_message() const;

  /** @return 驱动标记错误或缓冲区越界的帧数。 */
  uint64_t invalid_count() const noexcept;

  /** @return 实际打开的 V4L2 设备路径。 */
  const std::string & device_path() const noexcept;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;  ///< 隐藏 Linux V4L2 资源及采集线程实现。
};

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__V4L2_CAMERA_HPP_
