/**
 * @file test_image_message.cpp
 * @brief 验证 BGR 图像转换后的 ROS 消息字段和像素所有权。
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>

#include "fastumi_usb_camera/image_message.hpp"

TEST(ImageMessage, CopiesContinuousBgrImageAndHeader)
{
  cv::Mat image(2, 3, CV_8UC3, cv::Scalar(1, 2, 3));  ///< 待转换的连续 BGR 图像。
  builtin_interfaces::msg::Time stamp;  ///< 测试使用的 ROS 时间戳。
  stamp.sec = 42;
  stamp.nanosec = 123U;

  const auto message = fastumi_usb_camera::make_bgr_image_message(
    image, stamp, "camera_frame");  ///< 转换后的图像消息。
  EXPECT_EQ(message.header.stamp.sec, 42);
  EXPECT_EQ(message.header.stamp.nanosec, 123U);
  EXPECT_EQ(message.header.frame_id, "camera_frame");
  EXPECT_EQ(message.height, 2U);
  EXPECT_EQ(message.width, 3U);
  EXPECT_EQ(message.encoding, "bgr8");
  EXPECT_EQ(message.step, 9U);
  ASSERT_EQ(message.data.size(), 18U);
  EXPECT_EQ(message.data[0], 1U);
  EXPECT_EQ(message.data[1], 2U);
  EXPECT_EQ(message.data[2], 3U);
}

TEST(ImageMessage, RejectsEmptyOrWrongTypeImages)
{
  builtin_interfaces::msg::Time stamp;  ///< 测试使用的空时间戳。
  EXPECT_THROW(
    fastumi_usb_camera::make_bgr_image_message(cv::Mat(), stamp, "frame"),
    std::invalid_argument);
  cv::Mat gray(2, 2, CV_8UC1);  ///< 不符合 bgr8 契约的灰度图像。
  EXPECT_THROW(
    fastumi_usb_camera::make_bgr_image_message(gray, stamp, "frame"),
    std::invalid_argument);
}

TEST(ImageMessage, PreservesNativeJpegAndHeader)
{
  builtin_interfaces::msg::Time stamp;  ///< 测试使用的 ROS 时间戳。
  stamp.sec = 7;
  const std::vector<uint8_t> jpeg{0xFFU, 0xD8U, 0x01U, 0xFFU, 0xD9U};  ///< 原生 JPEG 字节。
  const auto message = fastumi_usb_camera::make_jpeg_message(
    jpeg, stamp, "jpeg_frame");  ///< 转换后的压缩图像消息。
  EXPECT_EQ(message.header.stamp.sec, 7);
  EXPECT_EQ(message.header.frame_id, "jpeg_frame");
  EXPECT_EQ(message.format, "jpeg");
  EXPECT_EQ(message.data, jpeg);
  EXPECT_THROW(
    fastumi_usb_camera::make_jpeg_message({}, stamp, "jpeg_frame"),
    std::invalid_argument);
}
