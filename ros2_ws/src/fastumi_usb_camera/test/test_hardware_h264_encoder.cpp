/** @file test_hardware_h264_encoder.cpp @brief Jetson hardware H.264 pipeline smoke test. */

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "fastumi_usb_camera/hardware_h264_encoder.hpp"

using namespace std::chrono_literals;

TEST(HardwareH264Encoder, ProducesTimestampedAccessUnitsAndKeyframes)
{
  std::string missing;
  if (!fastumi_usb_camera::HardwareH264Encoder::is_available(&missing)) {
    GTEST_SKIP() << "NVIDIA GStreamer element is unavailable: " << missing;
  }

  constexpr int width = 640;
  constexpr int height = 480;
  cv::Mat image(height, width, CV_8UC3, cv::Scalar(32, 96, 160));
  std::vector<uint8_t> jpeg;
  ASSERT_TRUE(cv::imencode(".jpg", image, jpeg));

  std::mutex mutex;
  std::condition_variable condition;
  std::vector<fastumi_usb_camera::HardwareH264Packet> packets;
  fastumi_usb_camera::HardwareH264Configuration configuration;
  configuration.width = width;
  configuration.height = height;
  configuration.fps = 30;
  configuration.bitrate = 1'000'000;
  configuration.gop_size = 2;
  fastumi_usb_camera::HardwareH264Encoder encoder(
    configuration,
    [&](fastumi_usb_camera::HardwareH264Packet && packet) {
      std::lock_guard<std::mutex> lock(mutex);
      packets.push_back(std::move(packet));
      condition.notify_all();
    });

  std::vector<uint64_t> input_pts;
  for (uint64_t index = 0; index < 6; ++index) {
    const uint64_t pts =
      1'000'000'000ULL + index * 33'333'333ULL + (index % 3U) * 1'000'003ULL;
    input_pts.push_back(pts);
    ASSERT_TRUE(encoder.push(
      pts, jpeg, fastumi_usb_camera::HardwareH264Encoder::SteadyClock::now()));
  }
  {
    std::unique_lock<std::mutex> lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, 5s, [&packets]() {return packets.size() >= 6U;}));
  }
  ASSERT_FALSE(encoder.poll_error()) << encoder.error_message();
  ASSERT_EQ(packets.size(), 6U);
  bool keyframe_seen = false;
  for (size_t index = 0; index < packets.size(); ++index) {
    const auto & packet = packets[index];
    EXPECT_FALSE(packet.data.empty());
    EXPECT_EQ(packet.pts, input_pts[index]);
    EXPECT_GT(packet.latency_nanoseconds, 0);
    keyframe_seen = keyframe_seen || packet.keyframe;
  }
  EXPECT_TRUE(keyframe_seen);
}

TEST(HardwareH264Encoder, ReportsThrowingPacketCallback)
{
  std::string missing;
  if (!fastumi_usb_camera::HardwareH264Encoder::is_available(&missing)) {
    GTEST_SKIP() << "NVIDIA GStreamer element is unavailable: " << missing;
  }

  cv::Mat image(480, 640, CV_8UC3, cv::Scalar(32, 96, 160));
  std::vector<uint8_t> jpeg;
  ASSERT_TRUE(cv::imencode(".jpg", image, jpeg));

  fastumi_usb_camera::HardwareH264Configuration configuration;
  configuration.width = image.cols;
  configuration.height = image.rows;
  configuration.fps = 30;
  fastumi_usb_camera::HardwareH264Encoder encoder(
    configuration,
    [](fastumi_usb_camera::HardwareH264Packet &&) {
      throw std::runtime_error("test callback failure");
    });

  ASSERT_TRUE(encoder.push(
    1'000'000'000ULL, jpeg, fastumi_usb_camera::HardwareH264Encoder::SteadyClock::now()));
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!encoder.poll_error() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  EXPECT_TRUE(encoder.poll_error());
  EXPECT_NE(encoder.error_message().find("test callback failure"), std::string::npos);
}
