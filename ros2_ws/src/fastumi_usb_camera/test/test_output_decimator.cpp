/**
 * @file test_output_decimator.cpp
 * @brief 验证 H.264 接收器的解码后输出降频计数。
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "fastumi_usb_camera/output_decimator.hpp"

using fastumi_usb_camera::OutputDecimator;

TEST(OutputDecimator, SelectsEveryNthDecodedFrame)
{
  OutputDecimator all(1);
  for (int index = 0; index < 5; ++index) {
    EXPECT_TRUE(all.take());
  }
  EXPECT_EQ(all.skipped(), 0U);
  OutputDecimator third(3);
  std::vector<bool> selected;
  for (int index = 0; index < 7; ++index) {
    selected.push_back(third.take());
  }
  EXPECT_EQ(selected, (std::vector<bool>{true, false, false, true, false, false, true}));
  EXPECT_EQ(third.skipped(), 4U);
  EXPECT_THROW(OutputDecimator(0), std::invalid_argument);
}
