/**
 * @file timestamp_utils.cpp
 * @brief 实现 SDK 数据流时间戳对齐与 ROS 时间戳转换工具。
 */

#include "timestamp_utils.h"

#include <algorithm>
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
