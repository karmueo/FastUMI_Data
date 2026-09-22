/**
 * @file image_message.hpp
 * @brief 提供 OpenCV BGR 图像到 ROS Image 消息的统一转换函数。
 */

#ifndef FASTUMI_USB_CAMERA__IMAGE_MESSAGE_HPP_
#define FASTUMI_USB_CAMERA__IMAGE_MESSAGE_HPP_

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <builtin_interfaces/msg/time.hpp>
#include <opencv2/core.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace fastumi_usb_camera
{

/**
 * @brief 将三通道 BGR 图像复制为 sensor_msgs Image。
 * @param[in] bgr 已解码的 CV_8UC3 图像。
 * @param[in] stamp 相机帧的主机接收时间戳。
 * @param[in] frame_id ROS 图像坐标系名称。
 * @return 拥有独立像素数据的 bgr8 图像消息。
 * @throws std::invalid_argument 输入为空或不是 CV_8UC3 时抛出。
 */
inline sensor_msgs::msg::Image make_bgr_image_message(
  const cv::Mat & bgr, const builtin_interfaces::msg::Time & stamp,
  const std::string & frame_id)
{
  if (bgr.empty() || bgr.type() != CV_8UC3) {
    throw std::invalid_argument("BGR image must be a non-empty CV_8UC3 matrix");
  }
  sensor_msgs::msg::Image image;  ///< 待返回的 ROS 原始图像消息。
  image.header.stamp = stamp;
  image.header.frame_id = frame_id;
  image.height = static_cast<uint32_t>(bgr.rows);
  image.width = static_cast<uint32_t>(bgr.cols);
  image.encoding = sensor_msgs::image_encodings::BGR8;
  image.is_bigendian = false;
  image.step = static_cast<sensor_msgs::msg::Image::_step_type>(bgr.cols * bgr.elemSize());
  if (bgr.isContinuous()) {
    image.data.assign(bgr.datastart, bgr.dataend);
  } else {
    image.data.resize(static_cast<size_t>(image.step) * image.height);
    for (int row = 0; row < bgr.rows; ++row) {
      std::memcpy(
        image.data.data() + static_cast<size_t>(row) * image.step,
        bgr.ptr(row), image.step);
    }
  }
  return image;
}

/**
 * @brief 将相机原生 JPEG 字节封装为 CompressedImage 消息。
 * @param[in] jpeg 完整 JPEG 字节，函数取得其所有权。
 * @param[in] stamp 相机帧的主机接收时间戳。
 * @param[in] frame_id ROS 图像坐标系名称。
 * @return format 为 jpeg 且拥有输入字节的压缩图像消息。
 * @throws std::invalid_argument JPEG 数据为空时抛出。
 */
inline sensor_msgs::msg::CompressedImage make_jpeg_message(
  std::vector<uint8_t> jpeg, const builtin_interfaces::msg::Time & stamp,
  const std::string & frame_id)
{
  if (jpeg.empty()) {
    throw std::invalid_argument("JPEG data must not be empty");
  }
  sensor_msgs::msg::CompressedImage message;  ///< 待返回的原生 JPEG 消息。
  message.header.stamp = stamp;
  message.header.frame_id = frame_id;
  message.format = "jpeg";
  message.data = std::move(jpeg);
  return message;
}

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__IMAGE_MESSAGE_HPP_
