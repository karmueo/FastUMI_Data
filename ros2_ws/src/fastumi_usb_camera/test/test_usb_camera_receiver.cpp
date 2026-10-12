/**
 * @file test_usb_camera_receiver.cpp
 * @brief 验证接收节点 auto 后端：关键帧起点、消息头保持、QoS 和软件回退。
 */
// Copyright 2026 karmueo

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <ffmpeg_image_transport_msgs/msg/ffmpeg_packet.hpp>
#include <image_transport/publisher_plugin.hpp>
#include <image_transport/subscriber_plugin.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "fastumi_usb_camera/h264_access_unit.hpp"
#include "fastumi_usb_camera/usb_camera_receiver.hpp"

namespace
{

using ffmpeg_image_transport_msgs::msg::FFMPEGPacket;

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
    image.data[index] = static_cast<uint8_t>((index + nanoseconds / 1000) % 251);
  }
  return image;
}

// Encode 12 frames with libx264 (GOP 5, no B frames) and return the packets.
std::vector<FFMPEGPacket> encode_packets(pluginlib::ClassLoader<image_transport::PublisherPlugin> & loader)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("enc.image.ffmpeg.encoder", "libx264"),
    rclcpp::Parameter(
      "enc.image.ffmpeg.encoder_av_options",
      "preset:ultrafast,tune:zerolatency,profile:baseline"),
    rclcpp::Parameter("enc.image.ffmpeg.pixel_format", "yuv420p"),
    rclcpp::Parameter("enc.image.ffmpeg.bit_rate", int64_t{500000}),
    rclcpp::Parameter("enc.image.ffmpeg.gop_size", int64_t{5}),
    rclcpp::Parameter("enc.image.ffmpeg.max_b_frames", int64_t{0}),
  });
  auto encoder_node = std::make_shared<rclcpp::Node>("receiver_test_encoder", options);
  auto observer_node = std::make_shared<rclcpp::Node>("receiver_test_observer");
  auto publisher = loader.createUniqueInstance(
    image_transport::PublisherPlugin::getLookupName("ffmpeg"));
  auto qos = rmw_qos_profile_sensor_data;
  qos.depth = 30;
  qos.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
  publisher->advertise(encoder_node.get(), "/enc/image", qos);
  std::vector<FFMPEGPacket> packets;
  auto subscription = observer_node->create_subscription<FFMPEGPacket>(
    "/enc/image/ffmpeg", rclcpp::SensorDataQoS().keep_last(30),
    [&packets](FFMPEGPacket::ConstSharedPtr packet) {packets.push_back(*packet);});
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(encoder_node);
  executor.add_node(observer_node);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  int64_t sequence = 0;
  while (packets.size() < 12 && std::chrono::steady_clock::now() < deadline) {
    // 先等观察者与编码插件匹配，再按固定时间戳递增发布。
    if (observer_node->count_publishers("/enc/image/ffmpeg") > 0) {
      publisher->publish(make_image(1000000000 + sequence++ * 33000000));
    }
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  publisher->shutdown();
  require(packets.size() >= 12, "encoder did not produce enough packets");
  return packets;
}

struct Result
{
  std::vector<sensor_msgs::msg::Image> frames;
  std::string active;
  bool software_locked = false;
  rmw_qos_reliability_policy_t reliability = RMW_QOS_POLICY_RELIABILITY_UNKNOWN;
  size_t depth = 0;
};

// Run a receiver with `parameters`, feed it the packets and collect its output.
Result run_receiver(
  const std::vector<FFMPEGPacket> & packets, size_t first_packet,
  const std::string & backend, const std::string & reliability, int64_t depth,
  const std::string & hardware_decoder = "h264_cuvid")
{
  rclcpp::NodeOptions options;
  options.parameter_overrides({
    rclcpp::Parameter("input_topic", "/rx/image"),
    rclcpp::Parameter("output_topic", "/rx/decoded"),
    rclcpp::Parameter("decoder_backend", backend),
    rclcpp::Parameter("output_queue_depth", int64_t{30}),
    rclcpp::Parameter("input_reliability", reliability),
    rclcpp::Parameter("input_queue_depth", depth),
    rclcpp::Parameter("hardware_decoder", hardware_decoder),
  });
  pluginlib::ClassLoader<image_transport::SubscriberPlugin> loader(
    "image_transport", "image_transport::SubscriberPlugin");
  auto receiver = std::make_shared<fastumi_usb_camera::UsbCameraReceiver>(loader, options);
  auto feeder = std::make_shared<rclcpp::Node>("receiver_test_feeder");
  auto sink = std::make_shared<rclcpp::Node>("receiver_test_sink");
  auto packet_publisher = feeder->create_publisher<FFMPEGPacket>(
    "/rx/image/ffmpeg", rclcpp::QoS(30).reliable());
  Result result;
  auto subscription = sink->create_subscription<sensor_msgs::msg::Image>(
    "/rx/decoded", rclcpp::QoS(30).reliable(),
    [&result](sensor_msgs::msg::Image::ConstSharedPtr image) {
      result.frames.push_back(*image);
    });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(receiver);
  executor.add_node(feeder);
  executor.add_node(sink);

  // 等待订阅匹配，再逐包发布。
  const auto match_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while ((packet_publisher->get_subscription_count() == 0 ||
    sink->count_publishers("/rx/decoded") == 0) &&
    std::chrono::steady_clock::now() < match_deadline)
  {
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  require(packet_publisher->get_subscription_count() > 0, "receiver did not subscribe");
  // 订阅端实际 QoS；发现信息不携带队列深度，故直接读取订阅本身。
  result.reliability = receiver->input_qos().get_rmw_qos_profile().reliability;
  result.depth = receiver->input_qos().get_rmw_qos_profile().depth;
  for (const auto & info : feeder->get_subscriptions_info_by_topic("/rx/image/ffmpeg")) {
    require(
      info.qos_profile().get_rmw_qos_profile().reliability == result.reliability,
      "advertised reliability differs from the configured one");
  }
  for (size_t index = first_packet; index < packets.size(); ++index) {
    packet_publisher->publish(packets[index]);
    for (int spin = 0; spin < 3; ++spin) {
      executor.spin_some();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  const auto drain_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
  while (std::chrono::steady_clock::now() < drain_deadline) {
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  result.active = receiver->active_decoder();
  result.software_locked = receiver->software_locked();
  executor.remove_node(receiver);
  return result;
}

// 每帧都带有某个输入包的原始时间戳与 frame_id，且像素格式未变。
void check_headers(
  const std::vector<sensor_msgs::msg::Image> & frames, const std::set<int64_t> & expected)
{
  for (const auto & frame : frames) {
    require(frame.header.frame_id == "test_camera", "frame_id was not preserved");
    require(
      expected.count(rclcpp::Time(frame.header.stamp).nanoseconds()) != 0,
      "timestamp was not preserved");
    require(frame.encoding == sensor_msgs::image_encodings::BGR8, "output encoding changed");
    require(frame.width == 160 && frame.height == 120, "output size changed");
  }
}

void run_checks()
{
  pluginlib::ClassLoader<image_transport::PublisherPlugin> publisher_loader(
    "image_transport", "image_transport::PublisherPlugin");
  const auto packets = encode_packets(publisher_loader);

  // 找到第一个和第二个合格关键帧，并确认编码器输出包含内联参数集。
  std::vector<size_t> keyframes;
  for (size_t index = 0; index < packets.size(); ++index) {
    if (fastumi_usb_camera::analyze_access_unit(
        packets[index].data.data(), packets[index].data.size()).qualified_keyframe())
    {
      keyframes.push_back(index);
    }
  }
  require(keyframes.size() >= 2, "encoder packets carry no inline SPS/PPS keyframes");
  const size_t mid_gop = keyframes[0] + 2;

  // 从 GOP 中段开始接入：关键帧前的包被跳过，之后每包输出一帧并保持消息头。
  std::set<int64_t> expected;
  for (size_t index = keyframes[1]; index < packets.size(); ++index) {
    expected.insert(rclcpp::Time(packets[index].header.stamp).nanoseconds());
  }
  // 硬件解码器（本机有 GPU 时为 h264_cuvid）自带若干帧流水线延迟，只要求输出
  // 的帧不多于输入、消息头原样保留；软件解码零延迟，帧数必须精确。
  const auto automatic = run_receiver(packets, mid_gop, "auto", "best_effort", 33);
  require(
    !automatic.frames.empty() && automatic.frames.size() <= expected.size(),
    "auto backend produced no frames or invented frames");
  check_headers(automatic.frames, expected);
  require(
    automatic.active == "h264" || automatic.active == "h264_cuvid",
    "auto backend opened no decoder");
  if (automatic.active == "h264_cuvid") {
    require(!automatic.software_locked, "software lock without a hardware failure");
  } else {
    require(automatic.software_locked, "software decoder used without being locked");
  }
  // 压缩包订阅使用参数指定的 QoS。
  require(automatic.reliability == RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT, "QoS not best effort");
  require(automatic.depth == 33, "input_queue_depth was not applied");

  const auto reliable = run_receiver(packets, keyframes[0], "auto", "reliable", 7);
  require(reliable.reliability == RMW_QOS_POLICY_RELIABILITY_RELIABLE, "QoS not reliable");
  require(reliable.depth == 7, "reliable input_queue_depth was not applied");
  require(
    !reliable.frames.empty() && reliable.frames.size() <= packets.size() - keyframes[0],
    "reliable run produced no or invented frames");

  // 硬件解码器无法打开：auto 用同一关键帧改用软件，精确输出每一帧并保持消息头；
  // cuda 后端不回退，首个关键帧即明确失败。
  const auto fallback = run_receiver(
    packets, mid_gop, "auto", "best_effort", 33, "h264_does_not_exist");
  require(fallback.active == "h264" && fallback.software_locked, "no software fallback");
  require(fallback.frames.size() == expected.size(), "fallback dropped or invented frames");
  check_headers(fallback.frames, expected);
  bool cuda_failed = false;
  try {
    run_receiver(packets, mid_gop, "cuda", "best_effort", 33, "h264_does_not_exist");
  } catch (const std::runtime_error & error) {
    cuda_failed = std::string(error.what()).find("h264_does_not_exist") != std::string::npos;
  }
  require(cuda_failed, "cuda backend did not fail on an unavailable hardware decoder");

  // 软件后端的 auto 与显式 transport 参数校验。
  rclcpp::NodeOptions bad;
  bad.parameter_overrides({rclcpp::Parameter("decoder_backend", "gpu")});
  pluginlib::ClassLoader<image_transport::SubscriberPlugin> loader(
    "image_transport", "image_transport::SubscriberPlugin");
  bool rejected = false;
  try {
    fastumi_usb_camera::UsbCameraReceiver invalid(loader, bad);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "invalid decoder_backend was accepted");
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    run_checks();
    rclcpp::shutdown();
    std::cout << "Receiver auto backend checks passed" << std::endl;
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "Receiver auto backend check failed: " << error.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  }
}
