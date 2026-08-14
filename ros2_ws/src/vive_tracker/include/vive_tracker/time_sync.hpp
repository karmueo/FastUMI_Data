/**
 * @file time_sync.hpp
 * @brief 声明 Tracker 主机查询时间估计、单调时间戳和轨迹限频工具。
 */

#pragma once

#include <cstdint>
#include <optional>

#include "vive_tracker/tracker_pose_types.hpp"

namespace vive_tracker {

/** @brief 节点启动时固定的稳定时钟与系统时钟对应锚点。 */
struct ClockAnchor {
  /** 稳定时钟锚点，单位为纳秒。 */
  std::int64_t steady_time_ns{0};
  /** 系统时钟锚点，单位为 Unix 纳秒。 */
  std::int64_t system_time_ns{0};
};

/**
 * @brief 验证区间后计算其中点，避免有符号整数溢出。
 * @param before_ns 区间起点，单位为纳秒。
 * @param after_ns 区间终点，单位为纳秒。
 * @return 有效区间的整数中点；逆序或溢出时返回空值。
 */
std::optional<std::int64_t> TryCalculateMidpointNs(
    std::int64_t before_ns, std::int64_t after_ns) noexcept;

/**
 * @brief 将稳定时钟时间映射到节点启动时固定的系统时钟域。
 * @param steady_time_ns 待映射的稳定时钟时间，单位为纳秒。
 * @param anchor 节点启动时捕获的时钟锚点。
 * @return 映射后的 Unix 纳秒；计算溢出时返回空值。
 */
std::optional<std::int64_t> TryMapSteadyToSystemNs(
    std::int64_t steady_time_ns, const ClockAnchor &anchor) noexcept;

/**
 * @brief 估计一次 OpenVR 查询的主机 Unix 时间戳。
 * @param timing 包围同一 OpenVR 查询的稳定/系统时钟区间。
 * @param anchor 节点启动时捕获的时钟锚点。
 * @return 优先使用稳定时钟中点映射；无效时使用系统时钟中点。
 */
std::optional<std::int64_t> EstimateQuerySystemTimeNs(
    const TrackerQueryTiming &timing, const ClockAnchor &anchor) noexcept;

/**
 * @brief 保证批次时间戳相对前一批严格递增。
 * @param candidate_ns 当前查询得到的候选 Unix 纳秒。
 * @param previous_ns 已发布的前一批 Unix 纳秒；无前值时传入空值。
 * @return 可发布的时间戳；前值已到最大整数时返回空值。
 */
std::optional<std::int64_t> MakeStrictlyMonotonicStampNs(
    std::int64_t candidate_ns,
    const std::optional<std::int64_t> &previous_ns) noexcept;

/**
 * @brief 判断有效位姿是否到达 Path 的下一次追加和发布时机。
 * @param sample_time_ns 已严格单调的查询批次时间戳，单位为 Unix 纳秒。
 * @param previous_path_time_ns 前一次 Path 更新的时间戳；无前值时传入空值。
 * @param path_publish_rate_hz Path 最大更新频率，单位为 Hz。
 * @return 首个有效位姿或已满足最小间隔时返回 true。
 */
bool IsPathUpdateDue(std::int64_t sample_time_ns,
                     const std::optional<std::int64_t> &previous_path_time_ns,
                     double path_publish_rate_hz) noexcept;

} // namespace vive_tracker
