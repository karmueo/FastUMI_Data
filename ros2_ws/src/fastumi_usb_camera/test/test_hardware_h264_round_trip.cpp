/** @file test_hardware_h264_round_trip.cpp @brief NVIDIA encoder to FFmpeg transport decoder test. */

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <ffmpeg_image_transport_msgs/msg/ffmpeg_packet.hpp>
#include <gtest/gtest.h>
#include <image_transport/subscriber_plugin.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "fastumi_usb_camera/hardware_h264_encoder.hpp"

using namespace std::chrono_literals;

TEST(HardwareH264RoundTrip, DecodesAndRecoversAfterResubscribe)
{
  std::string missing;
  if (!fastumi_usb_camera::HardwareH264Encoder::is_available(&missing)) {
    GTEST_SKIP() << "NVIDIA GStreamer element is unavailable: " << missing;
  }
  rclcpp::init(0, nullptr);
  const std::string topic = "/hardware_round_trip/image";
  auto publisher_node = std::make_shared<rclcpp::Node>("hardware_packet_publisher");
  auto subscriber_node = std::make_shared<rclcpp::Node>("hardware_packet_subscriber");
  auto publisher = publisher_node->create_publisher<
    ffmpeg_image_transport_msgs::msg::FFMPEGPacket>(
    topic + "/ffmpeg", rclcpp::SensorDataQoS().keep_last(20));

  constexpr int width = 640;
  constexpr int height = 480;
  fastumi_usb_camera::HardwareH264Configuration configuration;
  configuration.width = width;
  configuration.height = height;
  configuration.fps = 30;
  configuration.bitrate = 1'000'000;
  configuration.gop_size = 2;
  auto encoder = std::make_unique<fastumi_usb_camera::HardwareH264Encoder>(
    configuration,
    [publisher](fastumi_usb_camera::HardwareH264Packet && packet) {
      ffmpeg_image_transport_msgs::msg::FFMPEGPacket message;
      message.header.stamp.sec = static_cast<int32_t>(packet.pts / 1'000'000'000ULL);
      message.header.stamp.nanosec = static_cast<uint32_t>(packet.pts % 1'000'000'000ULL);
      message.header.frame_id = "hardware_camera";
      message.width = width;
      message.height = height;
      message.encoding = "h264;nv12;bgr8;bgr8";
      message.pts = packet.pts;
      message.flags = packet.keyframe ? 0x01U : 0x00U;
      message.data = std::move(packet.data);
      publisher->publish(std::move(message));
    });

  cv::Mat image(height, width, CV_8UC3, cv::Scalar(20, 100, 180));
  std::vector<uint8_t> jpeg;
  ASSERT_TRUE(cv::imencode(".jpg", image, jpeg));
  pluginlib::ClassLoader<image_transport::SubscriberPlugin> loader(
    "image_transport", "image_transport::SubscriberPlugin");
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(publisher_node);
  executor.add_node(subscriber_node);

  uint64_t sequence = 0;
  auto subscribe_and_receive = [&]() {
      bool received = false;
      sensor_msgs::msg::Image::ConstSharedPtr decoded;
      auto subscriber = loader.createUniqueInstance(
        image_transport::SubscriberPlugin::getLookupName("ffmpeg"));
      subscriber->subscribe(
        subscriber_node.get(), topic,
        [&](sensor_msgs::msg::Image::ConstSharedPtr value) {
          received = true;
          decoded = std::move(value);
        }, rmw_qos_profile_sensor_data);
      const auto deadline = std::chrono::steady_clock::now() + 5s;
      while (!received && std::chrono::steady_clock::now() < deadline) {
        const uint64_t pts = 2'000'000'000ULL + sequence++ * 33'333'333ULL;
        ASSERT_TRUE(encoder->push(
          pts, jpeg, fastumi_usb_camera::HardwareH264Encoder::SteadyClock::now()));
        for (int spin = 0; spin < 10 && !received; ++spin) {
          executor.spin_some();
          std::this_thread::sleep_for(10ms);
        }
      }
      subscriber->shutdown();
      ASSERT_TRUE(received);
      ASSERT_NE(decoded, nullptr);
      EXPECT_EQ(decoded->width, width);
      EXPECT_EQ(decoded->height, height);
      EXPECT_EQ(decoded->encoding, sensor_msgs::image_encodings::BGR8);
      EXPECT_EQ(decoded->header.frame_id, "hardware_camera");
      EXPECT_GE(rclcpp::Time(decoded->header.stamp).nanoseconds(), 2'000'000'000LL);
    };

  subscribe_and_receive();
  subscribe_and_receive();
  EXPECT_FALSE(encoder->poll_error()) << encoder->error_message();
  executor.remove_node(subscriber_node);
  executor.remove_node(publisher_node);
  encoder.reset();
  publisher.reset();
  subscriber_node.reset();
  publisher_node.reset();
  rclcpp::shutdown();
}
