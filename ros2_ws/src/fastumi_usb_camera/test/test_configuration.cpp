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

TEST(CameraConfiguration, FrameRateMustMatchExactly)
{
  using fastumi_usb_camera::frame_rate_matches;
  EXPECT_TRUE(frame_rate_matches(30, 1U, 30U));
  EXPECT_TRUE(frame_rate_matches(30, 1001U, 30030U));
  // 设备静默协商为其他档位（例如请求 60 得到 90 或 30）时拒绝。
  EXPECT_FALSE(frame_rate_matches(60, 1U, 90U));
  EXPECT_FALSE(frame_rate_matches(30, 1U, 20U));
  EXPECT_FALSE(frame_rate_matches(30, 0U, 30U));
}

TEST(CameraConfiguration, ExposurePriorityRequestIsValidated)
{
  auto config = CameraConfiguration{};
  EXPECT_EQ(config.exposure_auto_priority, "leave");
  config.exposure_auto_priority = "disable";
  EXPECT_NO_THROW(validate_configuration(config));
  config.exposure_auto_priority = "off";
  EXPECT_THROW(validate_configuration(config), std::invalid_argument);
}
