/**
 * @file test_configuration.cpp
 * @brief 验证 USB 相机采集与输出配置校验规则。
 */

#include <gtest/gtest.h>

#include "fastumi_usb_camera/configuration.hpp"

using fastumi_usb_camera::CameraConfiguration;
using fastumi_usb_camera::validate_configuration;

TEST(CameraConfiguration, DefaultsAreValid)
{
  const auto config = CameraConfiguration{};  ///< 待检查的默认配置。
  EXPECT_NO_THROW(validate_configuration(config));
  EXPECT_EQ(config.frame_timeout_seconds, 5.0);
  EXPECT_EQ(config.topic, "/usb_camera/image_raw");
  EXPECT_EQ(config.frame_id, "usb_camera_optical_frame");
}

TEST(CameraConfiguration, RejectsInvalidDeviceAndModeValues)
{
  auto config = CameraConfiguration{};  ///< 待注入非法值的配置。
  config.width = -1;
  EXPECT_THROW(validate_configuration(config), std::invalid_argument);
  config = CameraConfiguration{};
  config.frame_timeout_seconds = 0.0;
  EXPECT_THROW(validate_configuration(config), std::invalid_argument);
}

TEST(CameraConfiguration, RequiresAbsoluteTopicAndFrameId)
{
  auto config = CameraConfiguration{};  ///< 待注入非法话题和坐标系的配置。
  config.topic = "relative/image";
  EXPECT_THROW(validate_configuration(config), std::invalid_argument);
  config = CameraConfiguration{};
  config.frame_id.clear();
  EXPECT_THROW(validate_configuration(config), std::invalid_argument);
  config = CameraConfiguration{};
  config.topic = "/usb_camera/image_raw/";
  EXPECT_THROW(validate_configuration(config), std::invalid_argument);
}

TEST(CameraConfiguration, RequiresStablePhysicalDevicePath)
{
  auto config = CameraConfiguration{};  ///< 待校验的相机配置。
  config.video_device = "/dev/video0";
  EXPECT_THROW(validate_configuration(config), std::invalid_argument);
  config.video_device =
    "/dev/v4l/by-path/pci-test-usb-0:2.4:1.0-video-index0";
  EXPECT_NO_THROW(validate_configuration(config));
}
