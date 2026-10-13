/**
 * @file frame_timestamper.hpp
 * @brief 为 raw、JPEG 与 H.264 采集节点统一生成图像 Header 时间并记录时序诊断。
 *
 * 默认 ``v4l2_buffer`` 模式使用出队缓冲区的单调时钟时间戳，经
 * :class:`MonotonicToRosClock` 映射为 ROS 时间；缓冲区未声明单调时钟或时间戳为 0 时
 * 明确报错，不静默回退，也不把 EOF 或未知采样点标成 SOE。``callback_now`` 是显式
 * 兼容模式：在回调中读取 ROS 时间，包含 USB 接收、出队与调度时延，来源单独标记。
 * 解码、发布和降频都沿用这里生成的时间，不重新打时间戳。
 */

#ifndef FASTUMI_USB_CAMERA__FRAME_TIMESTAMPER_HPP_
#define FASTUMI_USB_CAMERA__FRAME_TIMESTAMPER_HPP_

#include <atomic>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "fastumi_usb_camera/clock_mapping.hpp"
#include "fastumi_usb_camera/frame_metadata.hpp"

namespace fastumi_usb_camera
{

/** @brief 图像 Header 时间来源模式。 */
enum class TimestampMode
{
  kV4l2Buffer,  ///< V4L2 缓冲区单调时钟时间映射到 ROS 时间（默认）。
  kCallbackNow,  ///< 兼容模式：回调中读取 ROS 时间。
};

/**
 * @brief 解析 ``timestamp_mode`` 参数。
 * @throws std::invalid_argument 取值不是 ``v4l2_buffer`` 或 ``callback_now``。
 */
inline TimestampMode parse_timestamp_mode(const std::string & text)
{
  if (text == "v4l2_buffer") {
    return TimestampMode::kV4l2Buffer;
  }
  if (text == "callback_now") {
    return TimestampMode::kCallbackNow;
  }
  throw std::invalid_argument("timestamp_mode must be v4l2_buffer or callback_now");
}

/** @brief 一帧的时间信息：发布用 ROS 时间及追溯所需的原始量。 */
struct StampedFrame
{
  int64_t ros_ns{0};  ///< 写入图像 Header 的 ROS 纳秒时间。
  int64_t v4l2_ns{0};  ///< 原始缓冲区时间戳（时钟见 flags）。
  uint32_t sequence{0};  ///< 驱动帧序号。
  uint32_t flags{0};  ///< 缓冲区标志原值。
  int64_t callback_ros_ns{0};  ///< 回调执行时的 ROS 时间，用于计算到达延迟。
  uint64_t generation{0};  ///< 使用的映射代号；兼容模式为 0。
  int64_t offset_ns{0};  ///< 使用的映射偏移；兼容模式为 0。
  int64_t uncertainty_ns{0};  ///< 使用的映射误差界；兼容模式为 0。
};

/** @brief 把缓冲区元数据转换为 ROS 时间并统计源时间单调性。 */
class FrameTimestamper
{
public:
  /**
   * @param mode 时间来源模式。
   * @param mapper ``kV4l2Buffer`` 时必须提供且已初始化的映射，生命周期由调用方保证。
   * @param ros_now 读取 ROS 时间的函数。
   */
  FrameTimestamper(TimestampMode mode, const MonotonicToRosClock * mapper, ClockReader ros_now)
  : mode_(mode), mapper_(mapper), ros_now_(std::move(ros_now))
  {
    if (!ros_now_) {
      throw std::invalid_argument("ROS clock reader must not be empty");
    }
    if (mode_ == TimestampMode::kV4l2Buffer && mapper_ == nullptr) {
      throw std::invalid_argument("v4l2_buffer timestamp mode requires a clock mapping");
    }
  }

  /**
   * @brief 生成一帧的时间。
   * @throws std::runtime_error ``v4l2_buffer`` 模式下缓冲区未使用单调时钟或时间戳无效。
   */
  StampedFrame stamp(const FrameMetadata & metadata)
  {
    StampedFrame frame;
    frame.v4l2_ns = metadata.timestamp_ns;
    frame.sequence = metadata.sequence;
    frame.flags = metadata.flags;
    frame.callback_ros_ns = ros_now_();
    if (mode_ == TimestampMode::kCallbackNow) {
      frame.ros_ns = frame.callback_ros_ns;
    } else {
      if (buffer_clock(metadata.flags) != BufferClock::kMonotonic) {
        throw std::runtime_error(
                "V4L2 buffer timestamp clock is " + buffer_clock_name(metadata.flags) +
                ", not monotonic; set timestamp_mode:=callback_now explicitly to use callback time");
      }
      if (metadata.timestamp_ns <= 0) {
        throw std::runtime_error("V4L2 buffer timestamp is zero or negative");
      }
      frame.generation = mapper_->generation();
      frame.offset_ns = mapper_->offset_ns();
      frame.uncertainty_ns = mapper_->uncertainty_ns();
      frame.ros_ns = metadata.timestamp_ns + frame.offset_ns;
    }
    const int64_t source = mode_ == TimestampMode::kCallbackNow ? frame.ros_ns : frame.v4l2_ns;
    const int64_t previous = last_source_ns_.exchange(source);
    if (previous != 0 && source <= previous) {
      non_monotonic_count_.fetch_add(1);
    }
    return frame;
  }

  /** @return 源时间不严格递增的帧数。 */
  uint64_t non_monotonic_count() const {return non_monotonic_count_.load();}

  TimestampMode mode() const {return mode_;}

  /**
   * @return 写入采集快照与诊断的时间来源标签，例如
   * ``v4l2_buffer_monotonic_soe_mapped_to_ros`` 或 ``callback_now``。
   */
  static std::string source_label(TimestampMode mode, uint32_t flags)
  {
    if (mode == TimestampMode::kCallbackNow) {
      return "callback_now";
    }
    return "v4l2_buffer_" + buffer_clock_name(flags) + "_" +
           buffer_timestamp_source_name(flags) + "_mapped_to_ros";
  }

private:
  TimestampMode mode_;
  const MonotonicToRosClock * mapper_;
  ClockReader ros_now_;
  std::atomic<int64_t> last_source_ns_{0};
  std::atomic<uint64_t> non_monotonic_count_{0};
};

/**
 * @brief 生成 ``sensor_msgs/TimeReference.source`` 中的分号分隔键值诊断文本。
 *
 * 字段：``mode``、``clock``、``tsrc``、``seq``、``recv_ns``（回调 ROS 时间）、
 * ``pub_ns``（发布前 ROS 时间）、``gen``、``offset_ns``、``unc_ns``、``exp_prio``。
 */
inline std::string timing_source_text(
  TimestampMode mode, const StampedFrame & frame, int64_t publish_ros_ns,
  const std::string & exposure_priority)
{
  std::ostringstream text;
  text << "mode=" << (mode == TimestampMode::kCallbackNow ? "callback_now" : "v4l2_buffer")
       << ";clock=" << buffer_clock_name(frame.flags)
       << ";tsrc=" << buffer_timestamp_source_name(frame.flags)
       << ";seq=" << frame.sequence
       << ";recv_ns=" << frame.callback_ros_ns
       << ";pub_ns=" << publish_ros_ns
       << ";gen=" << frame.generation
       << ";offset_ns=" << frame.offset_ns
       << ";unc_ns=" << frame.uncertainty_ns
       << ";exp_prio=" << exposure_priority;
  return text.str();
}

}  // namespace fastumi_usb_camera

#endif  // FASTUMI_USB_CAMERA__FRAME_TIMESTAMPER_HPP_
