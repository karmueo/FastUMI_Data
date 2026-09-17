#ifndef FASTUMI_USB_CAMERA__JPEG_FRAME_HPP_
#define FASTUMI_USB_CAMERA__JPEG_FRAME_HPP_

#include <cstdint>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace fastumi_usb_camera
{

// Decode an MJPEG payload into an expected-size BGR frame. Empty means the
// payload was corrupt or did not match the negotiated UVC frame dimensions.
inline cv::Mat decode_jpeg_frame(
  const std::vector<uint8_t> & jpeg, int width, int height)
{
  if (jpeg.empty()) {
    return {};
  }
  cv::Mat bgr = cv::imdecode(jpeg, cv::IMREAD_COLOR);
  if (bgr.empty() || bgr.cols != width || bgr.rows != height || bgr.type() != CV_8UC3) {
    return {};
  }
  return bgr;
}

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__JPEG_FRAME_HPP_
