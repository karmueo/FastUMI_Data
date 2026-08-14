/**
 * @file time_sync.cpp
 * @brief 实现 Tracker 主机查询时间估计、单调时间戳和轨迹限频工具。
 */

#include "vive_tracker/time_sync.hpp"

#include <cmath>
#include <limits>

namespace vive_tracker {
namespace {

/**
 * @brief 安全计算两个有序时间值的非负跨度。
 * @param later_ns 较晚的时间值，单位为纳秒。
 * @param earlier_ns 较早的时间值，单位为纳秒。
 * @return 可表示的非负跨度；逆序或跨度溢出时返回空值。
 */
std::optional<std::int64_t> TryCalculateNonnegativeDeltaNs(
    std::int64_t later_ns, std::int64_t earlier_ns) noexcept {
  if (later_ns < earlier_ns) {
    return std::nullopt;
  }
  if (earlier_ns < 0 &&
      later_ns > std::numeric_limits<std::int64_t>::max() + earlier_ns) {
    return std::nullopt;
  }
  return later_ns - earlier_ns;
}

} // namespace

/**
 * @brief 验证区间后计算其中点，避免有符号整数溢出。
 * @param before_ns 区间起点，单位为纳秒。
 * @param after_ns 区间终点，单位为纳秒。
 * @return 有效区间的整数中点；逆序或溢出时返回空值。
 */
std::optional<std::int64_t> TryCalculateMidpointNs(
    std::int64_t before_ns, std::int64_t after_ns) noexcept {
  /** 已验证不会溢出的非负时间跨度。 */
  const std::optional<std::int64_t> interval_ns =
      TryCalculateNonnegativeDeltaNs(after_ns, before_ns);
  if (!interval_ns.has_value()) {
    return std::nullopt;
  }
  return before_ns + *interval_ns / 2;
}

/**
 * @brief 将稳定时钟时间映射到节点启动时固定的系统时钟域。
 * @param steady_time_ns 待映射的稳定时钟时间，单位为纳秒。
 * @param anchor 节点启动时捕获的时钟锚点。
 * @return 映射后的 Unix 纳秒；计算溢出时返回空值。
 */
std::optional<std::int64_t> TryMapSteadyToSystemNs(
    std::int64_t steady_time_ns, const ClockAnchor &anchor) noexcept {
  /** 稳定时钟相对锚点的有符号方向。 */
  const bool is_forward = steady_time_ns >= anchor.steady_time_ns;
  /** 已在执行减法前验证过可表示性的稳定时钟跨度。 */
  const std::optional<std::int64_t> delta_ns =
      is_forward ? TryCalculateNonnegativeDeltaNs(steady_time_ns,
                                                   anchor.steady_time_ns)
                 : TryCalculateNonnegativeDeltaNs(anchor.steady_time_ns,
                                                   steady_time_ns);
  if (!delta_ns.has_value()) {
    return std::nullopt;
  }
  if (is_forward) {
    if (anchor.system_time_ns >
        std::numeric_limits<std::int64_t>::max() - *delta_ns) {
      return std::nullopt;
    }
    return anchor.system_time_ns + *delta_ns;
  }
  if (anchor.system_time_ns <
      std::numeric_limits<std::int64_t>::min() + *delta_ns) {
    return std::nullopt;
  }
  return anchor.system_time_ns - *delta_ns;
}

/**
 * @brief 估计一次 OpenVR 查询的主机 Unix 时间戳。
 * @param timing 包围同一 OpenVR 查询的稳定/系统时钟区间。
 * @param anchor 节点启动时捕获的时钟锚点。
 * @return 优先使用稳定时钟中点映射；无效时使用系统时钟中点。
 */
std::optional<std::int64_t> EstimateQuerySystemTimeNs(
    const TrackerQueryTiming &timing, const ClockAnchor &anchor) noexcept {
  /** 稳定时钟调用区间的中点。 */
  const std::optional<std::int64_t> steady_midpoint_ns =
      TryCalculateMidpointNs(timing.steady_before_ns, timing.steady_after_ns);
  if (steady_midpoint_ns.has_value()) {
    /** 中点映射到固定系统时钟域后的候选时间。 */
    const std::optional<std::int64_t> mapped_time_ns =
        TryMapSteadyToSystemNs(*steady_midpoint_ns, anchor);
    if (mapped_time_ns.has_value()) {
      return mapped_time_ns;
    }
  }
  return TryCalculateMidpointNs(timing.system_before_ns,
                                timing.system_after_ns);
}

/**
 * @brief 保证批次时间戳相对前一批严格递增。
 * @param candidate_ns 当前查询得到的候选 Unix 纳秒。
 * @param previous_ns 已发布的前一批 Unix 纳秒；无前值时传入空值。
 * @return 可发布的时间戳；前值已到最大整数时返回空值。
 */
std::optional<std::int64_t> MakeStrictlyMonotonicStampNs(
    std::int64_t candidate_ns,
    const std::optional<std::int64_t> &previous_ns) noexcept {
  if (!previous_ns.has_value() || candidate_ns > *previous_ns) {
    return candidate_ns;
  }
  if (*previous_ns == std::numeric_limits<std::int64_t>::max()) {
    return std::nullopt;
  }
  return *previous_ns + 1;
}

/**
 * @brief 判断有效位姿是否到达 Path 的下一次追加和发布时机。
 * @param sample_time_ns 已严格单调的查询批次时间戳，单位为 Unix 纳秒。
 * @param previous_path_time_ns 前一次 Path 更新的时间戳；无前值时传入空值。
 * @param path_publish_rate_hz Path 最大更新频率，单位为 Hz。
 * @return 首个有效位姿或已满足最小间隔时返回 true。
 */
bool IsPathUpdateDue(std::int64_t sample_time_ns,
                     const std::optional<std::int64_t> &previous_path_time_ns,
                     double path_publish_rate_hz) noexcept {
  if (!std::isfinite(path_publish_rate_hz) || path_publish_rate_hz <= 0.0) {
    return false;
  }
  if (!previous_path_time_ns.has_value()) {
    return true;
  }
  /** 当前有效位姿与上次 Path 更新之间的安全单调时间跨度。 */
  const std::optional<std::int64_t> elapsed_ns =
      TryCalculateNonnegativeDeltaNs(sample_time_ns,
                                     *previous_path_time_ns);
  if (!elapsed_ns.has_value()) {
    return sample_time_ns > *previous_path_time_ns;
  }
  /** Path 两次更新之间的最小时间间隔，单位为纳秒。 */
  const double minimum_interval_ns = 1000000000.0 / path_publish_rate_hz;
  return static_cast<double>(*elapsed_ns) >= minimum_interval_ns;
}

} // namespace vive_tracker
