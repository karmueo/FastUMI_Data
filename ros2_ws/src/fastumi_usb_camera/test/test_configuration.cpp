// Copyright 2026 karmueo

#include <gtest/gtest.h>

#include "fastumi_usb_camera/configuration.hpp"

using fastumi_usb_camera::CameraConfiguration;
using fastumi_usb_camera::validate_configuration;

TEST(CameraConfiguration, DefaultsAreValid)
{
  const auto config = CameraConfiguration{};
  EXPECT_NO_THROW(validate_configuration(config));
  EXPECT_EQ(config.frame_timeout_seconds, 5.0);
  EXPECT_EQ(config.topic, "/usb_camera/image_raw");
  EXPECT_EQ(config.frame_id, "usb_camera_optical_frame");
}

TEST(CameraConfiguration, RejectsInvalidDeviceAndModeValues)
{
  auto config = CameraConfiguration{};
  config.vendor_id = 0;
  EXPECT_THROW(validate_configuration(config), std::invalid_argument);
  config = CameraConfiguration{};
  config.width = -1;
  EXPECT_THROW(validate_configuration(config), std::invalid_argument);
  config = CameraConfiguration{};
  config.frame_timeout_seconds = 0.0;
  EXPECT_THROW(validate_configuration(config), std::invalid_argument);
}

TEST(CameraConfiguration, RequiresAbsoluteTopicAndFrameId)
{
  auto config = CameraConfiguration{};
  config.topic = "relative/image";
  EXPECT_THROW(validate_configuration(config), std::invalid_argument);
  config = CameraConfiguration{};
  config.frame_id.clear();
  EXPECT_THROW(validate_configuration(config), std::invalid_argument);
  config = CameraConfiguration{};
  config.topic = "/usb_camera/image_raw/";
  EXPECT_THROW(validate_configuration(config), std::invalid_argument);
}
