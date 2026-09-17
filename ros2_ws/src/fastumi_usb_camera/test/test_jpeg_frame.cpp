#include <gtest/gtest.h>

#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "fastumi_usb_camera/jpeg_frame.hpp"

TEST(JpegFrame, DecodesOnlyValidExpectedSizeFrames)
{
  cv::Mat original(12, 16, CV_8UC3, cv::Scalar(30, 80, 140));
  std::vector<uint8_t> jpeg;
  ASSERT_TRUE(cv::imencode(".jpg", original, jpeg));
  auto decoded = fastumi_usb_camera::decode_jpeg_frame(jpeg, 16, 12);
  ASSERT_FALSE(decoded.empty());
  EXPECT_EQ(decoded.type(), CV_8UC3);
  EXPECT_EQ(decoded.cols, 16);
  EXPECT_EQ(decoded.rows, 12);
  EXPECT_TRUE(fastumi_usb_camera::decode_jpeg_frame(jpeg, 15, 12).empty());
  EXPECT_TRUE(fastumi_usb_camera::decode_jpeg_frame({1, 2, 3}, 16, 12).empty());
  EXPECT_TRUE(fastumi_usb_camera::decode_jpeg_frame({}, 16, 12).empty());
}
