/**
 * @file usb_camera_receiver.cpp
 * @brief 接收 FFmpeg 视频包，并通过插件或 CUDA 解码后发布原始图像。
 */
// Copyright 2026 karmueo

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include <ffmpeg_encoder_decoder/decoder.hpp>
#include <ffmpeg_image_transport_msgs/msg/ffmpeg_packet.hpp>
#include <image_transport/subscriber_plugin.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace fastumi_usb_camera
{

/** @brief 一个统计周期内的 CUDA 接收与发布计数。 */
struct CudaIntervalStats
{
  /** 收到的 H.264 压缩包数。 */
  uint64_t packets = 0;
  /** 收到的压缩字节数。 */
  uint64_t bytes = 0;
  /** 收到的关键帧包数。 */
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

/** @brief 将 FFmpeg 压缩图像解码到本机原始图像话题。 */
class UsbCameraReceiver : public rclcpp::Node
{
public:
  /**
   * @brief 读取话题及解码后端参数，建立相应的订阅和发布。
   * @param loader 用于加载 FFmpeg 订阅插件的类加载器。
   */
  explicit UsbCameraReceiver(
    pluginlib::ClassLoader<image_transport::SubscriberPlugin> & loader)
  : Node("usb_camera_receiver")
  {
    // 输入和输出分别对应压缩图像的基础话题与解码后的原始图像话题。
    const auto input_topic = declare_parameter<std::string>(
      "input_topic", "/usb_camera/image_raw");
    const auto output_topic = declare_parameter<std::string>(
      "output_topic", "/usb_camera/image_decoded");
    // transport 保持原插件行为；cuda 使用本机 NVIDIA H.264 解码器。
    const auto backend = declare_parameter<std::string>("decoder_backend", "transport");
    // 发布队列深度可用于评估大图像在可靠订阅端的排队与丢帧。
    const auto output_queue_depth = declare_parameter<int64_t>("output_queue_depth", 1);
    if (input_topic.empty() || input_topic.front() != '/' ||
      input_topic.back() == '/' || output_topic.empty() ||
      output_topic.front() != '/' || output_topic.back() == '/')
    {
      throw std::invalid_argument("input_topic and output_topic must be absolute ROS topics");
    }
    if (backend != "transport" && backend != "cuda") {
      throw std::invalid_argument("decoder_backend must be transport or cuda");
    }
    if (output_queue_depth < 1) {
      throw std::invalid_argument("output_queue_depth must be positive");
    }

    output_ = create_publisher<sensor_msgs::msg::Image>(
      output_topic,
      rclcpp::QoS(rclcpp::KeepLast(static_cast<size_t>(output_queue_depth)))
      .reliable().durability_volatile());
    // 保持接收端原有的 BEST_EFFORT、VOLATILE 和 20 包缓存。
    auto qos = rmw_qos_profile_sensor_data;
    qos.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
    qos.depth = 20;
    qos.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
    qos.durability = RMW_QOS_POLICY_DURABILITY_VOLATILE;
    if (backend == "transport") {
      decoder_ = loader.createUniqueInstance(
        image_transport::SubscriberPlugin::getLookupName("ffmpeg"));
      decoder_->subscribe(
        this, input_topic,
        [this](const sensor_msgs::msg::Image::ConstSharedPtr & image) {
          output_->publish(*image);
        }, qos);
    } else {
      cuda_decoder_.setLogger(get_logger());
      cuda_decoder_.setOutputMessageEncoding("bgr8");
      packet_subscription_ = create_subscription<
        ffmpeg_image_transport_msgs::msg::FFMPEGPacket>(
        input_topic + "/ffmpeg",
        rclcpp::QoS(rclcpp::KeepLast(20)).best_effort().durability_volatile(),
        [this](const ffmpeg_image_transport_msgs::msg::FFMPEGPacket::ConstSharedPtr packet) {
          receiveCudaPacket(*packet);
        });
      stats_started_at_ = std::chrono::steady_clock::now();
      stats_timer_ = create_wall_timer(
        std::chrono::seconds(5), [this]() {logCudaStats();});
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
    if (decoder_) {
      decoder_->shutdown();
      decoder_.reset();
    }
    output_.reset();
  }

private:
  /**
   * @brief 使用本地递增 PTS 解码 H.264，保留原消息头中的时间戳与坐标系。
   * @param packet 接收到的压缩视频包。
   * @throws std::runtime_error 编码类型错误或 CUDA 解码器初始化失败时抛出。
   */
  void receiveCudaPacket(const ffmpeg_image_transport_msgs::msg::FFMPEGPacket & packet)
  {
    ++cuda_stats_.packets;
    cuda_stats_.bytes += packet.data.size();
    if ((packet.flags & AV_PKT_FLAG_KEY) != 0) {
      ++cuda_stats_.keyframes;
    }
    // 源时间戳可用于发现发送端停顿或接收端遗漏的包。
    const int64_t stamp_ns = rclcpp::Time(packet.header.stamp).nanoseconds();
    if (last_packet_stamp_ns_ != 0 && stamp_ns > last_packet_stamp_ns_) {
      const double gap_ms = static_cast<double>(stamp_ns - last_packet_stamp_ns_) / 1.0e6;
      cuda_stats_.max_stamp_gap_ms = std::max(cuda_stats_.max_stamp_gap_ms, gap_ms);
      if (gap_ms > 50.0) {
        ++cuda_stats_.stamp_gaps;
      }
    }
    last_packet_stamp_ns_ = stamp_ns;

    if (packet.encoding != "h264" && packet.encoding.rfind("h264;", 0) != 0) {
      throw std::runtime_error(
        "CUDA receiver requires H.264 packets, got '" + packet.encoding + "'");
    }
    if (packet.data.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Ignoring empty H.264 packet");
      return;
    }
    if (!cuda_decoder_.isInitialized()) {
      if ((packet.flags & AV_PKT_FLAG_KEY) == 0) {
        return;
      }
      // 回调中的图像已带有 decodePacket 提供的原始 ROS 消息头。
      if (!cuda_decoder_.initialize(
          packet.encoding,
          [this](const ffmpeg_encoder_decoder::ImageConstPtr & image, bool, const std::string &) {
            const auto publish_started_at = std::chrono::steady_clock::now();
            output_->publish(*image);
            const double publish_ms = std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - publish_started_at).count();
            cuda_stats_.max_publish_ms = std::max(cuda_stats_.max_publish_ms, publish_ms);
            ++cuda_stats_.frames;
          }, "h264_cuvid"))
      {
        throw std::runtime_error("Failed to initialize h264_cuvid CUDA decoder");
      }
    }

    // 上游包的 PTS 是纳秒级时间值；解码库内部时间基需要较小的序号。
    const uint64_t local_pts = next_pts_++;
    const auto decode_started_at = std::chrono::steady_clock::now();
    const bool decoded = cuda_decoder_.decodePacket(
        packet.encoding, packet.data.data(), packet.data.size(), local_pts,
        packet.header.frame_id, rclcpp::Time(packet.header.stamp));
    const double decode_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - decode_started_at).count();
    cuda_stats_.max_decode_ms = std::max(cuda_stats_.max_decode_ms, decode_ms);
    if (!decoded) {
      ++cuda_stats_.decode_failures;
      RCLCPP_WARN(get_logger(), "H.264 packet decode failed; waiting for next keyframe");
      cuda_decoder_.reset();
    }
  }

  /** @brief 每五秒记录节点自身的压缩包接收与图像发布速率。 */
  void logCudaStats()
  {
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - stats_started_at_).count();
    RCLCPP_INFO(
      get_logger(),
      "CUDA rx %.1f pkt/s, %.2f MB/s, keyframes %lu, stamp gaps %lu (max %.1f ms); "
      "publish %.1f frame/s, decode failures %lu, max decode %.1f ms, max publish %.1f ms",
      cuda_stats_.packets / seconds, cuda_stats_.bytes / seconds / 1.0e6,
      static_cast<unsigned long>(cuda_stats_.keyframes),
      static_cast<unsigned long>(cuda_stats_.stamp_gaps), cuda_stats_.max_stamp_gap_ms,
      cuda_stats_.frames / seconds, static_cast<unsigned long>(cuda_stats_.decode_failures),
      cuda_stats_.max_decode_ms, cuda_stats_.max_publish_ms);
    cuda_stats_ = {};
    stats_started_at_ = now;
  }

  /** 原有 FFmpeg image_transport 订阅插件。 */
  pluginlib::UniquePtr<image_transport::SubscriberPlugin> decoder_;
  /** CUDA 模式下直接订阅的压缩视频包。 */
  rclcpp::Subscription<ffmpeg_image_transport_msgs::msg::FFMPEGPacket>::SharedPtr
    packet_subscription_;
  /** CUDA 模式下使用的 FFmpeg 解码器。 */
  ffmpeg_encoder_decoder::Decoder cuda_decoder_;
  /** 下一个本地解码 PTS，独立于发送端消息中的 PTS。 */
  uint64_t next_pts_ = 1;
  /** 当前五秒统计周期内的收包与发布数据。 */
  CudaIntervalStats cuda_stats_;
  /** 上一个压缩包的源时间戳，用于统计包间断档。 */
  int64_t last_packet_stamp_ns_ = 0;
  /** 当前统计周期的起点。 */
  std::chrono::steady_clock::time_point stats_started_at_;
  /** 定期输出 CUDA 接收统计的计时器。 */
  rclcpp::TimerBase::SharedPtr stats_timer_;
  /** 解码后的原始图像发布端。 */
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr output_;
};

}  // namespace fastumi_usb_camera

namespace
{

/** 信号处理器设置的停止标记。 */
volatile std::sig_atomic_t stop_requested = 0;

/**
 * @brief 在收到终止信号时请求退出。
 * @param signal_number 收到的信号编号。
 */
void request_stop(int signal_number)
{
  (void)signal_number;
  stop_requested = 1;
}

}  // namespace

/**
 * @brief 启动接收节点并处理退出信号。
 * @param argc 参数个数。
 * @param argv 参数数组。
 * @return 进程退出码。
 */
int main(int argc, char ** argv)
{
  rclcpp::init(
    argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, request_stop);
  std::signal(SIGTERM, request_stop);
  try {
    {
      pluginlib::ClassLoader<image_transport::SubscriberPlugin> loader(
        "image_transport", "image_transport::SubscriberPlugin");
      auto node = std::make_shared<fastumi_usb_camera::UsbCameraReceiver>(loader);
      rclcpp::executors::SingleThreadedExecutor executor;
      executor.add_node(node);
      while (!stop_requested) {
        executor.spin_once(std::chrono::milliseconds(100));
      }
      executor.remove_node(node);
      node.reset();
    }
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("usb_camera_receiver"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
}
