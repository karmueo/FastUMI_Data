/**
 * @file timestamp_utils.cpp
 * @brief 实现 SDK 数据流时间戳对齐与 ROS 时间戳转换工具。
 */

#include "timestamp_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>

namespace {
/** hostTimestamp 过小或为负时使用的最小安全秒数。 */
constexpr double kMinimumSafeHostTimestampSeconds = 0.1;
/** 一秒包含的纳秒数。 */
constexpr double kNanosecondsPerSecond = 1000000000.0;
/** 单调保护使用的最小时间增量，单位秒。 */
constexpr double kMinimumTimestampStepSeconds = 1e-6;
} // namespace

namespace xv_ros2 {
namespace timestamp {
builtin_interfaces::msg::Time
hostTimestampToRosTime(double host_timestamp_seconds) {
  /** 转换后的 ROS 时间戳。 */
  builtin_interfaces::msg::Time stamp;
  /** 归一化后的 SDK hostTimestamp。 */
  const double safe_seconds =
      host_timestamp_seconds > kMinimumSafeHostTimestampSeconds
          ? host_timestamp_seconds
          : kMinimumSafeHostTimestampSeconds;

#ifdef HAVE_TRUNC
  stamp.sec = static_cast<std::int32_t>(std::trunc(safe_seconds));
#else
  if (safe_seconds >= 0.0) {
    stamp.sec = static_cast<std::int32_t>(std::floor(safe_seconds));
  } else {
    stamp.sec = static_cast<std::int32_t>(std::floor(safe_seconds)) + 1;
  }
#endif
  stamp.nanosec = static_cast<std::uint32_t>(
      (safe_seconds - static_cast<double>(stamp.sec)) * kNanosecondsPerSecond);

  return stamp;
}

double steadyTimestampToSystemSeconds(double steady_timestamp_seconds,
                                      double steady_now_seconds,
                                      double system_now_seconds) {
  if (!std::isfinite(system_now_seconds)) {
    return steady_timestamp_seconds;
  }
  if (!std::isfinite(steady_timestamp_seconds) ||
      steady_timestamp_seconds <= kMinimumSafeHostTimestampSeconds ||
      !std::isfinite(steady_now_seconds)) {
    return system_now_seconds;
  }

  /** 当前主机 system_clock 与 steady_clock 的时间原点偏移。 */
  const double clock_epoch_offset_seconds =
      system_now_seconds - steady_now_seconds;
  /** 映射到 Unix system_clock 时间域的事件时间戳。 */
  const double system_timestamp_seconds =
      steady_timestamp_seconds + clock_epoch_offset_seconds;
  return std::isfinite(system_timestamp_seconds) &&
                 system_timestamp_seconds > 0.0
             ? system_timestamp_seconds
             : system_now_seconds;
}

builtin_interfaces::msg::Time
steadyTimestampToRosTime(double steady_timestamp_seconds) {
  /** system_clock 采样前的 steady_clock 时刻。 */
  const auto steady_before = std::chrono::steady_clock::now();
  /** 与当前 steady_clock 采样相邻的 Unix system_clock 时刻。 */
  const auto system_now = std::chrono::system_clock::now();
  /** system_clock 采样后的 steady_clock 时刻。 */
  const auto steady_after = std::chrono::steady_clock::now();
  /** 两次 steady_clock 采样的中点秒数。 */
  const double steady_now_seconds =
      std::chrono::duration<double>(
          steady_before.time_since_epoch() +
          (steady_after - steady_before) / 2)
          .count();
  /** 当前 Unix system_clock 秒数。 */
  const double system_now_seconds =
      std::chrono::duration<double>(system_now.time_since_epoch()).count();
  /** 映射后的 Unix 秒级时间戳。 */
  const double system_timestamp_seconds = steadyTimestampToSystemSeconds(
      steady_timestamp_seconds, steady_now_seconds, system_now_seconds);
  return hostTimestampToRosTime(system_timestamp_seconds);
}

double systemTimestampToSteadySeconds(double system_timestamp_seconds,
                                      double steady_now_seconds,
                                      double system_now_seconds) {
  if (!std::isfinite(steady_now_seconds)) {
    return system_timestamp_seconds;
  }
  if (!std::isfinite(system_timestamp_seconds) ||
      system_timestamp_seconds <= kMinimumSafeHostTimestampSeconds ||
      !std::isfinite(system_now_seconds)) {
    return steady_now_seconds;
  }

  /** 当前主机 steady_clock 与 system_clock 的时间原点偏移。 */
  const double clock_epoch_offset_seconds =
      steady_now_seconds - system_now_seconds;
  /** 映射到 steady_clock 时间域的事件时间戳。 */
  const double steady_timestamp_seconds =
      system_timestamp_seconds + clock_epoch_offset_seconds;
  return std::isfinite(steady_timestamp_seconds) &&
                 steady_timestamp_seconds > 0.0
             ? steady_timestamp_seconds
             : steady_now_seconds;
}

double systemTimestampToSteadySeconds(double system_timestamp_seconds) {
  /** system_clock 采样前的 steady_clock 时刻。 */
  const auto steady_before = std::chrono::steady_clock::now();
  /** 与当前 steady_clock 采样相邻的 Unix system_clock 时刻。 */
  const auto system_now = std::chrono::system_clock::now();
  /** system_clock 采样后的 steady_clock 时刻。 */
  const auto steady_after = std::chrono::steady_clock::now();
  /** 两次 steady_clock 采样的中点秒数。 */
  const double steady_now_seconds =
      std::chrono::duration<double>(
          steady_before.time_since_epoch() +
          (steady_after - steady_before) / 2)
          .count();
  /** 当前 Unix system_clock 秒数。 */
  const double system_now_seconds =
      std::chrono::duration<double>(system_now.time_since_epoch()).count();
  return systemTimestampToSteadySeconds(
      system_timestamp_seconds, steady_now_seconds, system_now_seconds);
}

StreamTimestampAligner::StreamTimestampAligner(
    std::size_t calibration_sample_count, double passthrough_threshold_seconds)
    : m_calibrationSampleCount(
          std::max(calibration_sample_count, std::size_t{1})),
      m_passthroughThresholdSeconds(
          std::max(passthrough_threshold_seconds, 0.0)) {}

double StreamTimestampAligner::align(double source_timestamp_seconds,
                                     double host_now_seconds) {
  if (!std::isfinite(host_now_seconds)) {
    return source_timestamp_seconds;
  }
  std::lock_guard<std::mutex> lock(m_mutex);

  /** 应用当前偏移或无效源时间戳回退策略后的时间戳。 */
  double aligned_timestamp_seconds = host_now_seconds;
  if (std::isfinite(source_timestamp_seconds) &&
      source_timestamp_seconds > 0.0) {
    /** 当前样本对应的主机与数据流时间原点偏移候选值。 */
    const double candidate_offset_seconds =
        host_now_seconds - source_timestamp_seconds;
    if (!m_initialized) {
      m_initialized = true;
      m_compensationEnabled =
          std::abs(candidate_offset_seconds) > m_passthroughThresholdSeconds;
      m_offsetSeconds = m_compensationEnabled ? candidate_offset_seconds : 0.0;
      m_observedSampleCount = 1;
      m_locked = !m_compensationEnabled || m_calibrationSampleCount == 1;
    } else if (m_compensationEnabled && !m_locked) {
      m_offsetSeconds = std::min(m_offsetSeconds, candidate_offset_seconds);
      ++m_observedSampleCount;
      m_locked = m_observedSampleCount >= m_calibrationSampleCount;
    }

    aligned_timestamp_seconds = source_timestamp_seconds + m_offsetSeconds;
  }

  if (m_hasLastAlignedTimestamp &&
      aligned_timestamp_seconds <= m_lastAlignedTimestampSeconds) {
    aligned_timestamp_seconds =
        m_lastAlignedTimestampSeconds + kMinimumTimestampStepSeconds;
  }
  m_lastAlignedTimestampSeconds = aligned_timestamp_seconds;
  m_hasLastAlignedTimestamp = true;
  return aligned_timestamp_seconds;
}

bool StreamTimestampAligner::compensationEnabled() const {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_compensationEnabled;
}

bool StreamTimestampAligner::locked() const {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_locked;
}

double StreamTimestampAligner::offsetSeconds() const {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_offsetSeconds;
}
} // namespace timestamp
} // namespace xv_ros2
