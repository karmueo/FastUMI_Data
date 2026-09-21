/**
 * @file configuration.hpp
 * @brief 定义并校验 USB 相机采集与 FFmpeg 输出配置。
 */

#ifndef FASTUMI_USB_CAMERA__CONFIGURATION_HPP_
#define FASTUMI_USB_CAMERA__CONFIGURATION_HPP_

#include <cstdint>
#include <stdexcept>
#include <string>

#include "fastumi_usb_camera/physical_device.hpp"

namespace fastumi_usb_camera
{

/** @brief 保存 UVC 设备选择、采集模式与 FFmpeg 输出配置。 */
struct CameraConfiguration
{
  int64_t vendor_id{0x1BCF};  ///< 兼容选择模式使用的 USB 厂商标识。
  int64_t product_id{0x28C4};  ///< 兼容选择模式使用的 USB 产品标识。
  int64_t width{1280};  ///< 采集图像宽度，单位为像素。
  int64_t height{960};  ///< 采集图像高度，单位为像素。
  int64_t fps{30};  ///< 采集帧率。
  double frame_timeout_seconds{5.0};  ///< 连续无有效帧的错误门限，单位为秒。
  std::string video_device;  ///< 稳定的 V4L2 by-path 主视频节点。
  std::string serial_number;  ///< 未使用物理路径时的可选 USB 序列号。
  std::string frame_id{"usb_camera_optical_frame"};  ///< ROS 图像坐标系名称。
  std::string topic{"/usb_camera/image_raw"};  ///< FFmpeg transport 的绝对基础话题。
};

inline void validate_configuration(const CameraConfiguration & config)
{
  if (config.vendor_id <= 0 || config.vendor_id > 0xFFFF) {
    throw std::invalid_argument("vendor_id must be between 1 and 65535");
  }
  if (config.product_id <= 0 || config.product_id > 0xFFFF) {
    throw std::invalid_argument("product_id must be between 1 and 65535");
  }
  if (config.width <= 0 || config.height <= 0 || config.fps <= 0) {
    throw std::invalid_argument("width, height and fps must be greater than zero");
  }
  if (config.frame_timeout_seconds <= 0.0) {
    throw std::invalid_argument("frame_timeout_seconds must be greater than zero");
  }
  if (!config.video_device.empty() && !is_physical_video_device_path(config.video_device)) {
    throw std::invalid_argument(
            "video_device must use /dev/v4l/by-path/*-video-index0");
  }
  if (!config.video_device.empty() && !config.serial_number.empty()) {
    throw std::invalid_argument("video_device and serial_number are mutually exclusive");
  }
  if (config.frame_id.empty()) {
    throw std::invalid_argument("frame_id must not be empty");
  }
  if (config.topic.empty() || config.topic.front() != '/') {
    throw std::invalid_argument("topic must be a non-empty absolute ROS topic");
  }
  if (config.topic.size() > 1 && config.topic.back() == '/') {
    throw std::invalid_argument("topic must not end in a slash");
  }
}

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__CONFIGURATION_HPP_
