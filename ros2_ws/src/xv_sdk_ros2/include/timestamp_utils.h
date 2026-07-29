/**
 * @file timestamp_utils.h
 * @brief 声明 SDK 数据流时间戳对齐与 ROS 时间戳转换工具。
 */

#ifndef __TIMESTAMP_UTILS_H__
#define __TIMESTAMP_UTILS_H__

#include <builtin_interfaces/msg/time.hpp>

#include <cstddef>
#include <mutex>

namespace xv_ros2 {
namespace timestamp {
/**
 * @brief 将 SDK 秒级 hostTimestamp 转换为 ROS Time 消息。
 * @param host_timestamp_seconds SDK 秒级 hostTimestamp。
 * @return ROS Time 消息；输入小于等于 0.1 秒时按 0.1 秒保护。
 */
builtin_interfaces::msg::Time
hostTimestampToRosTime(double host_timestamp_seconds);

/**
 * @brief 将 steady_clock 时间戳映射到 Unix system_clock 时间域。
 * @param steady_timestamp_seconds 待转换的 steady_clock 秒级时间戳。
 * @param steady_now_seconds 采样时刻的 steady_clock 秒数。
 * @param system_now_seconds 同一采样时刻的 Unix system_clock 秒数。
 * @return 映射后的 Unix 秒级时间戳；输入无效时回退到 system_now_seconds。
 */
double steadyTimestampToSystemSeconds(double steady_timestamp_seconds,
                                      double steady_now_seconds,
                                      double system_now_seconds);

/**
 * @brief 将 steady_clock 秒级时间戳转换为 Unix 时间域的 ROS Time。
 * @param steady_timestamp_seconds steady_clock 秒级时间戳。
 * @return Unix system_clock 时间域的 ROS Time。
 */
builtin_interfaces::msg::Time
steadyTimestampToRosTime(double steady_timestamp_seconds);

/**
 * @brief 将 Unix system_clock 时间戳映射回 steady_clock 时间域。
 * @param system_timestamp_seconds 待转换的 Unix 秒级时间戳。
 * @param steady_now_seconds 采样时刻的 steady_clock 秒数。
 * @param system_now_seconds 同一采样时刻的 Unix system_clock 秒数。
 * @return 映射后的 steady_clock 秒数；输入无效时回退到 steady_now_seconds。
 */
double systemTimestampToSteadySeconds(double system_timestamp_seconds,
                                      double steady_now_seconds,
                                      double system_now_seconds);

/**
 * @brief 将 Unix system_clock 秒级时间戳转换为 steady_clock 秒数。
 * @param system_timestamp_seconds Unix system_clock 秒级时间戳。
 * @return steady_clock 时间域的秒数。
 */
double systemTimestampToSteadySeconds(double system_timestamp_seconds);

/**
 * @brief 将具有独立时间原点的数据流时间戳映射到主机 steady_clock。
 *
 * 对已经位于主机时间基准的数据流保持透传。检测到明显的时间原点偏差后，
 * 使用启动阶段观测到的最小传输延迟估算固定偏移，并保证输出单调递增。
 */
class StreamTimestampAligner {
public:
  /**
   * @brief 创建数据流时间戳对齐器。
   * @param calibration_sample_count 锁定偏移前使用的样本数。
   * @param passthrough_threshold_seconds 判定时间戳已在主机基准内的最大偏差。
   */
  explicit StreamTimestampAligner(std::size_t calibration_sample_count = 30,
                                  double passthrough_threshold_seconds = 1.0);

  /**
   * @brief 将数据流时间戳对齐到主机 steady_clock。
   * @param source_timestamp_seconds 数据流原始秒级时间戳。
   * @param host_now_seconds 回调时刻的主机 steady_clock 秒数。
   * @return 对齐后的秒级时间戳；源时间戳无效时返回主机当前时间。
   */
  double align(double source_timestamp_seconds, double host_now_seconds);

  /** @brief 返回当前是否启用了偏移补偿。 */
  bool compensationEnabled() const;

  /** @brief 返回偏移估计是否已经锁定。 */
  bool locked() const;

  /** @brief 返回当前估算的时间偏移，单位秒。 */
  double offsetSeconds() const;

private:
  /** 保护对齐状态。 */
  mutable std::mutex m_mutex;
  /** 锁定偏移前需要收集的样本数。 */
  std::size_t m_calibrationSampleCount;
  /** 已参与偏移估计的样本数。 */
  std::size_t m_observedSampleCount{0};
  /** 判断是否需要补偿的偏差阈值，单位秒。 */
  double m_passthroughThresholdSeconds;
  /** 当前使用的时间偏移，单位秒。 */
  double m_offsetSeconds{0.0};
  /** 上一次输出时间戳，单位秒。 */
  double m_lastAlignedTimestampSeconds{0.0};
  /** 是否已经确定数据流的时间基准模式。 */
  bool m_initialized{false};
  /** 是否需要对数据流应用偏移。 */
  bool m_compensationEnabled{false};
  /** 偏移估计是否已经锁定。 */
  bool m_locked{false};
  /** 是否已经产生过有效输出。 */
  bool m_hasLastAlignedTimestamp{false};
};
} // namespace timestamp
} // namespace xv_ros2

#endif // __TIMESTAMP_UTILS_H__
