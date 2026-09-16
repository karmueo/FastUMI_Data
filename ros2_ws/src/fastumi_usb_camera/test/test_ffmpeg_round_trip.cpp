// Copyright 2026 karmueo

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <ffmpeg_image_transport_msgs/msg/ffmpeg_packet.hpp>
#include <image_transport/publisher_plugin.hpp>
#include <image_transport/subscriber_plugin.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace
{

void require(bool condition, const std::string & message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

sensor_msgs::msg::Image make_image(int64_t nanoseconds)
{
  sensor_msgs::msg::Image image;
  image.header.stamp = rclcpp::Time(nanoseconds);
  image.header.frame_id = "test_camera";
  image.width = 160;
  image.height = 120;
  image.encoding = sensor_msgs::image_encodings::BGR8;
  image.step = image.width * 3;
  image.data.resize(image.step * image.height);
  for (size_t index = 0; index < image.data.size(); ++index) {
    image.data[index] = static_cast<uint8_t>((index + nanoseconds) % 251);
  }
  return image;
}

void run_check()
{
  const std::string topic = "/round_trip/image";
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("round_trip.image.ffmpeg.encoder", "libx264"),
    rclcpp::Parameter(
      "round_trip.image.ffmpeg.encoder_av_options",
      "preset:ultrafast,tune:zerolatency,profile:baseline"),
    rclcpp::Parameter("round_trip.image.ffmpeg.pixel_format", "yuv420p"),
    rclcpp::Parameter("round_trip.image.ffmpeg.bit_rate", int64_t{500000}),
    rclcpp::Parameter("round_trip.image.ffmpeg.gop_size", int64_t{2}),
    rclcpp::Parameter("round_trip.image.ffmpeg.max_b_frames", int64_t{0}),
  });
  auto publisher_node = std::make_shared<rclcpp::Node>("round_trip_publisher", options);
  auto subscriber_node = std::make_shared<rclcpp::Node>("round_trip_subscriber");
  auto packet_node = std::make_shared<rclcpp::Node>("round_trip_packet_observer");

  pluginlib::ClassLoader<image_transport::PublisherPlugin> publisher_loader(
    "image_transport", "image_transport::PublisherPlugin");
  pluginlib::ClassLoader<image_transport::SubscriberPlugin> subscriber_loader(
    "image_transport", "image_transport::SubscriberPlugin");
  auto publisher = publisher_loader.createUniqueInstance(
    image_transport::PublisherPlugin::getLookupName("ffmpeg"));

  auto qos = rmw_qos_profile_sensor_data;
  qos.depth = 1;
  qos.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
  publisher->advertise(publisher_node.get(), topic, qos);

  bool keyframe_seen = false;
  auto packet_subscription = packet_node->create_subscription<
    ffmpeg_image_transport_msgs::msg::FFMPEGPacket>(
    topic + "/ffmpeg", rclcpp::SensorDataQoS().keep_last(20),
    [&keyframe_seen](ffmpeg_image_transport_msgs::msg::FFMPEGPacket::ConstSharedPtr packet) {
      keyframe_seen = keyframe_seen || packet->flags != 0;
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(publisher_node);
  executor.add_node(subscriber_node);
  executor.add_node(packet_node);

  auto subscribe_and_receive = [&](int64_t stamp) {
      bool received = false;
      sensor_msgs::msg::Image::ConstSharedPtr decoded;
      auto subscriber = subscriber_loader.createUniqueInstance(
        image_transport::SubscriberPlugin::getLookupName("ffmpeg"));
      subscriber->subscribe(
        subscriber_node.get(), topic,
        [&](sensor_msgs::msg::Image::ConstSharedPtr image) {
          received = true;
          decoded = std::move(image);
        }, qos);

      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      int sequence = 0;
      while (!received && std::chrono::steady_clock::now() < deadline) {
        publisher->publish(make_image(stamp + sequence++));
        executor.spin_some();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      subscriber->shutdown();
      require(received && decoded != nullptr, "decoder did not produce an image");
      require(decoded->width == 160 && decoded->height == 120, "decoded dimensions changed");
      require(
        decoded->encoding == sensor_msgs::image_encodings::BGR8,
        "decoded image encoding changed");
      require(decoded->header.frame_id == "test_camera", "frame_id was not preserved");
      require(
        rclcpp::Time(decoded->header.stamp).nanoseconds() >= stamp,
        "timestamp was not preserved");
    };

  subscribe_and_receive(1000000);
  subscribe_and_receive(2000000);
  require(keyframe_seen, "no H.264 keyframe packet was observed");

  packet_subscription.reset();
  publisher->shutdown();
  publisher.reset();
  executor.remove_node(packet_node);
  executor.remove_node(subscriber_node);
  executor.remove_node(publisher_node);
  packet_node.reset();
  subscriber_node.reset();
  publisher_node.reset();
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    run_check();
    rclcpp::shutdown();
    std::cout << "FFmpeg round-trip and resubscribe checks passed" << std::endl;
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "FFmpeg round-trip check failed: " << error.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  }
}
