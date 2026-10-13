/**
 * @file camera_timing.hpp
 * @brief raw/JPEG 与 H.264 采集节点共用的时间戳组件：参数、时钟映射、周期检查与诊断话题。
 *
 * 每帧发布图像后，在 ``<图像基础话题>/time_reference`` 发布
 * ``sensor_msgs/TimeReference``：``header.stamp`` 等于图像 Header 时间，``time_ref`` 为
 * 原始 V4L2 缓冲区时间，``source`` 为 :func:`timing_source_text` 键值文本（含回调与
 * 发布时的 ROS 时间、映射代号、偏移、误差界和曝光优先状态）。采集节点录制该话题，
 * 并在映射代号变化时让当前采集失败。本地实时采集要求 ``use_sim_time=false``。
 */

#ifndef FASTUMI_USB_CAMERA__CAMERA_TIMING_HPP_
#define FASTUMI_USB_CAMERA__CAMERA_TIMING_HPP_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/time_reference.hpp>

#include "fastumi_usb_camera/clock_mapping.hpp"
#include "fastumi_usb_camera/frame_metadata.hpp"
#include "fastumi_usb_camera/frame_timestamper.hpp"

namespace fastumi_usb_camera
{

/** @return 纳秒整数对应的 ROS 时间消息。 */
inline builtin_interfaces::msg::Time to_time_msg(int64_t nanoseconds)
{
  builtin_interfaces::msg::Time stamp;
  stamp.sec = static_cast<int32_t>(nanoseconds / 1000000000LL);
  stamp.nanosec = static_cast<uint32_t>(nanoseconds % 1000000000LL);
  return stamp;
}

/** @brief 节点内的时间戳与映射管理。 */
class CameraTiming
{
public:
  /**
   * @brief 声明参数、拒绝模拟时间、建立映射并创建检查定时器和诊断发布器。
   * @param node 所属节点。
   * @param time_reference_topic 诊断话题（相对或绝对）。
   * @throws std::invalid_argument 参数非法或启用了模拟时间。
   * @throws std::runtime_error 映射误差界超限。
   */
  CameraTiming(rclcpp::Node & node, const std::string & time_reference_topic)
  : node_(node)
  {
    mode_ = parse_timestamp_mode(
      node.declare_parameter<std::string>("timestamp_mode", "v4l2_buffer"));
    ClockMappingConfig config;
    config.attempts = static_cast<int>(
      node.declare_parameter<int64_t>("clock_mapping_attempts", config.attempts));
    config.max_uncertainty_ns = static_cast<int64_t>(
      node.declare_parameter<double>("clock_mapping_max_uncertainty_us", 200.0) * 1000.0);
    config.max_step_ns = static_cast<int64_t>(
      node.declare_parameter<double>("clock_mapping_max_step_us", 2000.0) * 1000.0);
    const double check_period_s = node.declare_parameter<double>("clock_check_period_s", 1.0);
    const bool publish_reference = node.declare_parameter<bool>("publish_time_reference", true);
    if (config.attempts <= 0 || config.max_uncertainty_ns <= 0 || config.max_step_ns <= 0 ||
      check_period_s <= 0.0)
    {
      throw std::invalid_argument("clock mapping parameters must be positive");
    }
    if (node.get_parameter("use_sim_time").as_bool()) {
      throw std::invalid_argument(
              "local camera capture requires use_sim_time:=false; V4L2 timestamps map to system time");
    }
    const auto ros_now = [this]() {return node_.get_clock()->now().nanoseconds();};
    if (mode_ == TimestampMode::kV4l2Buffer) {
      mapper_ = std::make_unique<MonotonicToRosClock>(monotonic_now_ns, ros_now, config);
      mapper_->initialize();
      RCLCPP_INFO(
        node.get_logger(),
        "Image timestamps: V4L2 monotonic buffer time mapped to ROS time "
        "(offset %ld ns, uncertainty %ld ns, max step %ld ns, check every %.2f s)",
        mapper_->offset_ns(), mapper_->uncertainty_ns(), config.max_step_ns, check_period_s);
      check_timer_ = node.create_wall_timer(
        std::chrono::duration<double>(check_period_s), [this]() {check_mapping();});
    } else {
      RCLCPP_WARN(
        node.get_logger(),
        "Image timestamps: timestamp_mode=callback_now; Header time includes USB receive, "
        "dequeue and scheduling latency");
    }
    timestamper_ = std::make_unique<FrameTimestamper>(mode_, mapper_.get(), ros_now);
    if (publish_reference) {
      rclcpp::QoS qos(rclcpp::KeepLast(30));
      qos.reliable().durability_volatile();
      reference_publisher_ = node.create_publisher<sensor_msgs::msg::TimeReference>(
        time_reference_topic, qos);
    }
  }

  /** @brief 生成一帧时间；无效缓冲区时间抛出 std::runtime_error。 */
  StampedFrame stamp(const FrameMetadata & metadata)
  {
    const StampedFrame frame = timestamper_->stamp(metadata);
    std::lock_guard<std::mutex> lock(label_mutex_);
    const std::string label = FrameTimestamper::source_label(mode_, metadata.flags);
    if (label != source_label_) {
      // 首帧或驱动标志变化（例如 SOE→EOF）时记录，不把不同语义混为一谈。
      RCLCPP_INFO(node_.get_logger(), "Image timestamp source: %s", label.c_str());
      source_label_ = label;
    }
    return frame;
  }

  /** @brief 图像发布后发布对应的时间诊断。 */
  void publish_reference(
    const StampedFrame & frame, const std::string & frame_id, const std::string & exposure)
  {
    if (!reference_publisher_) {
      return;
    }
    sensor_msgs::msg::TimeReference message;
    message.header.stamp = to_time_msg(frame.ros_ns);
    message.header.frame_id = frame_id;
    message.time_ref = to_time_msg(frame.v4l2_ns);
    message.source = timing_source_text(
      mode_, frame, node_.get_clock()->now().nanoseconds(), exposure);
    reference_publisher_->publish(message);
  }

  /** @return 周期诊断摘要。 */
  std::string diagnostics() const
  {
    std::string label;
    {
      std::lock_guard<std::mutex> lock(label_mutex_);
      label = source_label_.empty() ? std::string("pending") : source_label_;
    }
    std::string text = "timestamp=" + label;
    if (mapper_) {
      text += " offset_ns=" + std::to_string(mapper_->offset_ns()) +
        " uncertainty_ns=" + std::to_string(mapper_->uncertainty_ns()) +
        " generation=" + std::to_string(mapper_->generation()) +
        " jumps=" + std::to_string(jumps_.load()) +
        " uncertain_checks=" + std::to_string(uncertain_checks_.load());
    }
    return text + " non_monotonic=" + std::to_string(timestamper_->non_monotonic_count());
  }

private:
  /** @brief 周期检查映射；跳变时重建映射并以 ERROR 记录。 */
  void check_mapping()
  {
    const ClockCheckResult result = mapper_->check();
    if (result.status == ClockCheckStatus::kJump) {
      jumps_.fetch_add(1);
      RCLCPP_ERROR(
        node_.get_logger(),
        "System/ROS time jumped by %ld ns relative to CLOCK_MONOTONIC; clock mapping rebuilt "
        "(generation %lu). Current collection must fail and a new episode must start.",
        result.offset_change_ns, static_cast<unsigned long>(result.generation));
    } else if (result.status == ClockCheckStatus::kUncertain) {
      uncertain_checks_.fetch_add(1);
      RCLCPP_WARN_THROTTLE(
        node_.get_logger(), *node_.get_clock(), 5000,
        "Clock mapping sample uncertainty %ld ns exceeds limit; keeping previous offset",
        result.uncertainty_ns);
    }
  }

  rclcpp::Node & node_;
  TimestampMode mode_{TimestampMode::kV4l2Buffer};
  std::unique_ptr<MonotonicToRosClock> mapper_;
  std::unique_ptr<FrameTimestamper> timestamper_;
  rclcpp::Publisher<sensor_msgs::msg::TimeReference>::SharedPtr reference_publisher_;
  rclcpp::TimerBase::SharedPtr check_timer_;
  mutable std::mutex label_mutex_;
  std::string source_label_;
  std::atomic<uint64_t> jumps_{0};
  std::atomic<uint64_t> uncertain_checks_{0};
};

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__CAMERA_TIMING_HPP_
