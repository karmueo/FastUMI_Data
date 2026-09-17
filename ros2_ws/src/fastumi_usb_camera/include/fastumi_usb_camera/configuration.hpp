// Copyright 2026 karmueo

#ifndef FASTUMI_USB_CAMERA__CONFIGURATION_HPP_
#define FASTUMI_USB_CAMERA__CONFIGURATION_HPP_

#include <cstdint>
#include <stdexcept>
#include <string>

namespace fastumi_usb_camera
{

struct CameraConfiguration
{
  int64_t vendor_id{0x1BCF};
  int64_t product_id{0x28C4};
  int64_t width{1280};
  int64_t height{960};
  int64_t fps{30};
  double frame_timeout_seconds{5.0};
  std::string serial_number;
  std::string frame_id{"usb_camera_optical_frame"};
  std::string topic{"/usb_camera/image_raw"};
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
