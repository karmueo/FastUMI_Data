#ifndef FASTUMI_USB_CAMERA__USB_CAMERA_RECEIVER_HPP_
#define FASTUMI_USB_CAMERA__USB_CAMERA_RECEIVER_HPP_

/**
 * @file usb_camera_receiver.hpp
 * @brief 接收 FFmpeg 视频包，并通过插件、CUDA 或自动回退解码后发布原始图像。
 */
// Copyright 2026 karmueo

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include <ffmpeg_encoder_decoder/decoder.hpp>
#include <ffmpeg_image_transport_msgs/msg/ffmpeg_packet.hpp>
#include <image_transport/subscriber_plugin.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "fastumi_usb_camera/decoder_policy.hpp"
#include "fastumi_usb_camera/h264_access_unit.hpp"

namespace fastumi_usb_camera
{

/** @brief 一个统计周期内的接收与发布计数。 */
struct PacketIntervalStats
{
  /** 收到的 H.264 压缩包数。 */
  uint64_t packets = 0;
  /** 收到的压缩字节数。 */
  uint64_t bytes = 0;
  /** 收到的合格关键帧（含 SPS、PPS 和 IDR）数。 */
  uint64_t keyframes = 0;
  /** 已提交发布的解码图像数。 */
  uint64_t frames = 0;
  /** 解码失败的压缩包数。 */
  uint64_t decode_failures = 0;
  /** 两个相邻包的源时间戳间隔超过 50 毫秒的次数。 */
  uint64_t stamp_gaps = 0;
  /** 当前周期中最大的源时间戳间隔，单位毫秒。 */
  double max_stamp_gap_ms = 0.0;
  /** 当前周期中最大的单包解码耗时，单位毫秒。 */
  double max_decode_ms = 0.0;
  /** 当前周期中最大的单帧发布耗时，单位毫秒。 */
  double max_publish_ms = 0.0;
};

/**
 * @brief 把 ffmpeg_encoder_decoder::Decoder 适配为 PacketDecoder。
 *
 * 解码器使用本地递增 PTS：上游包的 PTS 是纳秒级时间值，解码库内部时间基
 * 需要较小的序号；原始 ROS 时间戳和 frame_id 随每个包传入并原样输出。
 */
class FfmpegPacketDecoder : public PacketDecoder
{
public:
  using ImageCallback = std::function<void(const sensor_msgs::msg::Image::ConstSharedPtr &)>;

  /**
   * @param callback 每解出一帧调用一次，图像带有输入包的消息头。
   */
  explicit FfmpegPacketDecoder(ImageCallback callback)
  : callback_(std::move(callback))
  {
    decoder_.setOutputMessageEncoding("bgr8");
  }

  /** @brief 设置下一次 initialize/decode 使用的压缩包，调用方须保证其存活。 */
  void set_packet(const ffmpeg_image_transport_msgs::msg::FFMPEGPacket * packet)
  {
    packet_ = packet;
  }

  /** @brief 设置日志输出。 */
  void set_logger(const rclcpp::Logger & logger) {decoder_.setLogger(logger);}

  bool initialized() const override {return decoder_.isInitialized();}

  bool initialize(const std::string & decoder_name) override
  {
    return decoder_.initialize(
      packet_->encoding,
      [this](const ffmpeg_encoder_decoder::ImageConstPtr & image, bool, const std::string &) {
        callback_(image);
      }, decoder_name);
  }

  bool decode() override
  {
    return decoder_.decodePacket(
      packet_->encoding, packet_->data.data(), packet_->data.size(), next_pts_++,
      packet_->header.frame_id, rclcpp::Time(packet_->header.stamp));
  }

  void reset() override {decoder_.reset();}

private:
  /** 回调，把解码图像交给节点发布。 */
  ImageCallback callback_;
  /** 底层 FFmpeg 解码器。 */
  ffmpeg_encoder_decoder::Decoder decoder_;
  /** 当前正在处理的压缩包。 */
  const ffmpeg_image_transport_msgs::msg::FFMPEGPacket * packet_ = nullptr;
  /** 下一个本地解码 PTS，独立于发送端消息中的 PTS。 */
  uint64_t next_pts_ = 1;
};

/** @brief 将 FFmpeg 压缩图像解码到本机原始图像话题。 */
class UsbCameraReceiver : public rclcpp::Node
{
public:
  /**
   * @brief 读取话题、QoS 及解码后端参数，建立相应的订阅和发布。
   * @param loader 用于加载 FFmpeg 订阅插件的类加载器。
   * @param options 节点选项，测试可用它覆盖参数。
   */
  explicit UsbCameraReceiver(
    pluginlib::ClassLoader<image_transport::SubscriberPlugin> & loader,
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("usb_camera_receiver", options)
  {
    // 输入和输出分别对应压缩图像的基础话题与解码后的原始图像话题。
    const auto input_topic = declare_parameter<std::string>(
      "input_topic", "/usb_camera/image_raw");
    const auto output_topic = declare_parameter<std::string>(
      "output_topic", "/usb_camera/image_decoded");
    // transport 保持原插件行为；cuda 使用本机 NVIDIA H.264 解码器；
    // auto 优先 cuda，初始化或运行失败时回退到软件解码并保持到重启。
    const auto backend = declare_parameter<std::string>("decoder_backend", "transport");
    // 发布队列深度可用于评估大图像在可靠订阅端的排队与丢帧。
    const auto output_queue_depth = declare_parameter<int64_t>("output_queue_depth", 1);
    // cuda/auto 优先使用的硬件解码器名称，默认 NVIDIA 的 h264_cuvid。
    const auto hardware_decoder = declare_parameter<std::string>(
      "hardware_decoder", "h264_cuvid");
    // 压缩包订阅的可靠性与队列深度，默认 BEST_EFFORT/20 兼容当前发布端。
    const auto input_reliability = declare_parameter<std::string>(
      "input_reliability", "best_effort");
    const auto input_queue_depth = declare_parameter<int64_t>("input_queue_depth", 20);
    if (input_topic.empty() || input_topic.front() != '/' ||
      input_topic.back() == '/' || output_topic.empty() ||
      output_topic.front() != '/' || output_topic.back() == '/')
    {
      throw std::invalid_argument("input_topic and output_topic must be absolute ROS topics");
    }
    if (backend != "transport" && backend != "cuda" && backend != "auto") {
      throw std::invalid_argument("decoder_backend must be transport, cuda or auto");
    }
    if (output_queue_depth < 1 || input_queue_depth < 1) {
      throw std::invalid_argument("output_queue_depth and input_queue_depth must be positive");
    }
    if (input_reliability != "best_effort" && input_reliability != "reliable") {
      throw std::invalid_argument("input_reliability must be best_effort or reliable");
    }

    output_ = create_publisher<sensor_msgs::msg::Image>(
      output_topic,
      rclcpp::QoS(rclcpp::KeepLast(static_cast<size_t>(output_queue_depth)))
      .reliable().durability_volatile());
    // 压缩包保持 VOLATILE；可靠性与深度由参数决定。
    auto input_qos = rclcpp::QoS(rclcpp::KeepLast(static_cast<size_t>(input_queue_depth)))
      .durability_volatile();
    if (input_reliability == "reliable") {
      input_qos.reliable();
    } else {
      input_qos.best_effort();
    }
    if (backend == "transport") {
      auto qos = input_qos.get_rmw_qos_profile();
      decoder_ = loader.createUniqueInstance(
        image_transport::SubscriberPlugin::getLookupName("ffmpeg"));
      decoder_->subscribe(
        this, input_topic,
        [this](const sensor_msgs::msg::Image::ConstSharedPtr & image) {
          output_->publish(*image);
        }, qos);
    } else {
      packet_decoder_ = std::make_unique<FfmpegPacketDecoder>(
        [this](const sensor_msgs::msg::Image::ConstSharedPtr & image) {publish_image(*image);});
      packet_decoder_->set_logger(get_logger());
      policy_ = std::make_unique<DecoderPolicy>(
        *packet_decoder_, backend == "auto", hardware_decoder);
      packet_subscription_ = create_subscription<
        ffmpeg_image_transport_msgs::msg::FFMPEGPacket>(
        input_topic + "/ffmpeg", input_qos,
        [this](const ffmpeg_image_transport_msgs::msg::FFMPEGPacket::ConstSharedPtr packet) {
          receivePacket(*packet);
        });
      stats_started_at_ = std::chrono::steady_clock::now();
      stats_timer_ = create_wall_timer(
        std::chrono::seconds(5), [this]() {logStats();});
    }

    RCLCPP_INFO(
      get_logger(), "Decoding %s/ffmpeg -> %s with %s backend",
      input_topic.c_str(), output_topic.c_str(), backend.c_str());
  }

  /** @brief 关闭订阅并释放发布端。 */
  ~UsbCameraReceiver() override
  {
    stats_timer_.reset();
    packet_subscription_.reset();
    policy_.reset();
    packet_decoder_.reset();
    if (decoder_) {
      decoder_->shutdown();
      decoder_.reset();
    }
    output_.reset();
  }

  /** @brief 当前打开的解码器名称，未打开时为空；仅 cuda/auto 后端有效。 */
  std::string active_decoder() const
  {
    return policy_ ? policy_->active_decoder() : std::string();
  }

  /** @brief 压缩包订阅实际使用的 QoS；仅 cuda/auto 后端有效。 */
  rclcpp::QoS input_qos() const
  {
    return packet_subscription_ ? packet_subscription_->get_actual_qos() : rclcpp::QoS(1);
  }

  /** @brief 是否已回退到软件解码。 */
  bool software_locked() const {return policy_ && policy_->software_locked();}

private:
  /** @brief 发布解码图像并记录发布耗时。 */
  void publish_image(const sensor_msgs::msg::Image & image)
  {
    const auto publish_started_at = std::chrono::steady_clock::now();
    output_->publish(image);
    const double publish_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - publish_started_at).count();
    stats_.max_publish_ms = std::max(stats_.max_publish_ms, publish_ms);
    ++stats_.frames;
  }

  /**
   * @brief 使用本地递增 PTS 解码 H.264，保留原消息头中的时间戳与坐标系。
   *
   * 只有同时含 SPS、PPS 和 IDR 的关键帧才能启动解码。auto 后端在 CUDA 初始化
   * 失败时用同一关键帧尝试软件解码，运行失败后等待下一个合格关键帧再切换。
   * @param packet 接收到的压缩视频包。
   * @throws std::runtime_error 编码类型错误或没有任何解码器可初始化时抛出。
   */
  void receivePacket(const ffmpeg_image_transport_msgs::msg::FFMPEGPacket & packet)
  {
    ++stats_.packets;
    stats_.bytes += packet.data.size();
    // 源时间戳可用于发现发送端停顿或接收端遗漏的包。
    const int64_t stamp_ns = rclcpp::Time(packet.header.stamp).nanoseconds();
    if (last_packet_stamp_ns_ != 0 && stamp_ns > last_packet_stamp_ns_) {
      const double gap_ms = static_cast<double>(stamp_ns - last_packet_stamp_ns_) / 1.0e6;
      stats_.max_stamp_gap_ms = std::max(stats_.max_stamp_gap_ms, gap_ms);
      if (gap_ms > 50.0) {
        ++stats_.stamp_gaps;
      }
    }
    last_packet_stamp_ns_ = stamp_ns;

    if (packet.encoding != "h264" && packet.encoding.rfind("h264;", 0) != 0) {
      throw std::runtime_error(
        "H.264 receiver requires H.264 packets, got '" + packet.encoding + "'");
    }
    if (packet.data.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Ignoring empty H.264 packet");
      return;
    }
    const bool qualified = analyze_access_unit(
      packet.data.data(), packet.data.size()).qualified_keyframe();
    if (qualified) {
      ++stats_.keyframes;
    }
    const std::string previous = policy_->active_decoder();
    packet_decoder_->set_packet(&packet);
    const auto decode_started_at = std::chrono::steady_clock::now();
    const auto status = policy_->process(qualified);
    const double decode_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - decode_started_at).count();
    stats_.max_decode_ms = std::max(stats_.max_decode_ms, decode_ms);
    if (policy_->active_decoder() != previous && !policy_->active_decoder().empty()) {
      RCLCPP_INFO(get_logger(), "Using %s H.264 decoder", policy_->active_decoder().c_str());
    }
    if (status == PacketStatus::kDecodeFailed) {
      ++stats_.decode_failures;
      RCLCPP_WARN(
        get_logger(), "H.264 packet decode failed; waiting for next keyframe%s",
        policy_->software_locked() ? " (software decoder)" : "");
    }
  }

  /** @brief 每五秒记录节点自身的压缩包接收与图像发布速率。 */
  void logStats()
  {
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - stats_started_at_).count();
    RCLCPP_INFO(
      get_logger(),
      "H.264 rx %.1f pkt/s, %.2f MB/s, keyframes %lu, stamp gaps %lu (max %.1f ms); "
      "publish %.1f frame/s, decode failures %lu, max decode %.1f ms, max publish %.1f ms, "
      "decoder %s",
      stats_.packets / seconds, stats_.bytes / seconds / 1.0e6,
      static_cast<unsigned long>(stats_.keyframes),
      static_cast<unsigned long>(stats_.stamp_gaps), stats_.max_stamp_gap_ms,
      stats_.frames / seconds, static_cast<unsigned long>(stats_.decode_failures),
      stats_.max_decode_ms, stats_.max_publish_ms,
      policy_->active_decoder().empty() ? "none" : policy_->active_decoder().c_str());
    stats_ = {};
    stats_started_at_ = now;
  }

  /** 原有 FFmpeg image_transport 订阅插件。 */
  pluginlib::UniquePtr<image_transport::SubscriberPlugin> decoder_;
  /** cuda/auto 模式下直接订阅的压缩视频包。 */
  rclcpp::Subscription<ffmpeg_image_transport_msgs::msg::FFMPEGPacket>::SharedPtr
    packet_subscription_;
  /** cuda/auto 模式下使用的 FFmpeg 解码器适配。 */
  std::unique_ptr<FfmpegPacketDecoder> packet_decoder_;
  /** 选择 CUDA 或软件解码器的策略。 */
  std::unique_ptr<DecoderPolicy> policy_;
  /** 当前五秒统计周期内的收包与发布数据。 */
  PacketIntervalStats stats_;
  /** 上一个压缩包的源时间戳，用于统计包间断档。 */
  int64_t last_packet_stamp_ns_ = 0;
  /** 当前统计周期的起点。 */
  std::chrono::steady_clock::time_point stats_started_at_;
  /** 定期输出接收统计的计时器。 */
  rclcpp::TimerBase::SharedPtr stats_timer_;
  /** 解码后的原始图像发布端。 */
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr output_;
};

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__USB_CAMERA_RECEIVER_HPP_
