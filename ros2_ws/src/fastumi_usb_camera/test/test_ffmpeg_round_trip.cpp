/**
 * @file test_ffmpeg_round_trip.cpp
 * @brief 验证 FFmpeg 往返传输及使用本地 PTS 时的原始时间戳恢复。
 */
// Copyright 2026 karmueo

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <ffmpeg_encoder_decoder/decoder.hpp>
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

/** @brief 验证插件往返、重新订阅及本地 PTS 解码后的消息头。 */
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
  // 保存实际编码后的包，以验证高位 PTS 的接收端解码方式。
  std::vector<ffmpeg_image_transport_msgs::msg::FFMPEGPacket> captured_packets;
  auto packet_subscription = packet_node->create_subscription<
    ffmpeg_image_transport_msgs::msg::FFMPEGPacket>(
    topic + "/ffmpeg", rclcpp::SensorDataQoS().keep_last(20),
    [&keyframe_seen, &captured_packets](
      ffmpeg_image_transport_msgs::msg::FFMPEGPacket::ConstSharedPtr packet) {
      keyframe_seen = keyframe_seen || packet->flags != 0;
      captured_packets.push_back(*packet);
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

  // 模拟 Jetson 包中纳秒级的原始 PTS，直接解码时仅传入本地小序号。
  // 用软件解码器验证与 CUDA 路径相同的 PTS 和消息头映射行为。
  ffmpeg_encoder_decoder::Decoder normalized_decoder;
  normalized_decoder.setOutputMessageEncoding("bgr8");
  // 记录实际输入帧的 ROS 时间戳，用于核对解码回调。
  std::unordered_set<int64_t> expected_stamps;
  // 已成功恢复的图像数量及下一个解码器本地 PTS。
  size_t normalized_frames = 0;
  uint64_t local_pts = 1;
  // 首个关键帧出现后才初始化解码器。
  bool initialized = false;
  for (auto & packet : captured_packets) {
    packet.pts = 1700000000000000000ULL + local_pts;
    if (!initialized) {
      if ((packet.flags & AV_PKT_FLAG_KEY) == 0) {
        continue;
      }
      initialized = normalized_decoder.initialize(
        packet.encoding,
        [&expected_stamps, &normalized_frames](
          const ffmpeg_encoder_decoder::ImageConstPtr & image, bool, const std::string &) {
          require(
            expected_stamps.count(rclcpp::Time(image->header.stamp).nanoseconds()) != 0,
            "normalized decoder changed the original ROS timestamp");
          require(image->header.frame_id == "test_camera", "normalized frame_id changed");
          require(image->encoding == sensor_msgs::image_encodings::BGR8,
            "normalized image encoding changed");
          ++normalized_frames;
        }, "h264");
      require(initialized, "software decoder initialization failed");
    }
    expected_stamps.insert(rclcpp::Time(packet.header.stamp).nanoseconds());
    require(
      normalized_decoder.decodePacket(
        packet.encoding, packet.data.data(), packet.data.size(), local_pts++,
        packet.header.frame_id, rclcpp::Time(packet.header.stamp)),
      "normalized packet decoding failed");
  }
  require(normalized_frames > 0, "normalized decoder produced no image");

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
