/**
 * @file tracker_pose_types.hpp
 * @brief 定义 VIVE Tracker ROS 2 节点使用的公共位姿数据类型。
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vive_tracker {

/**
 * @brief SteamVR 位姿查询使用的跟踪原点类型。
 */
enum class TrackingOrigin {
  /** SteamVR 站立坐标系。 */
  kStanding,
  /** SteamVR 就座坐标系。 */
  kSeated,
  /** Lighthouse 驱动原始坐标系。 */
  kRaw,
};

/**
 * @brief OpenVR 返回的跟踪状态。
 */
enum class TrackingState {
  /** 设备尚未初始化。 */
  kUninitialized,
  /** 设备正在标定。 */
  kCalibratingInProgress,
  /** 标定过程中设备超出跟踪范围。 */
  kCalibratingOutOfRange,
  /** 设备正常跟踪。 */
  kRunningOk,
  /** 运行过程中设备超出跟踪范围。 */
  kRunningOutOfRange,
  /** 当前只能提供旋转跟踪。 */
  kFallbackRotationOnly,
  /** OpenVR 返回了当前程序未识别的状态。 */
  kUnknown,
};

/**
 * @brief 三维位置，单位为米。
 */
struct Vector3 {
  /** X 轴位置，单位为米。 */
  double x{0.0};
  /** Y 轴位置，单位为米。 */
  double y{0.0};
  /** Z 轴位置，单位为米。 */
  double z{0.0};
};

/**
 * @brief 采用 xyzw 顺序的单位四元数。
 */
struct Quaternion {
  /** 四元数 X 分量。 */
  double x{0.0};
  /** 四元数 Y 分量。 */
  double y{0.0};
  /** 四元数 Z 分量。 */
  double z{0.0};
  /** 四元数 W 分量。 */
  double w{1.0};
};

/**
 * @brief Tracker 在指定 SteamVR 跟踪坐标系中的刚体位姿。
 */
struct Pose {
  /** Tracker 本体原点的位置。 */
  Vector3 position{};
  /** Tracker 本体坐标系的方向。 */
  Quaternion orientation{};
};

/**
 * @brief 单个 Tracker 在一次采样中的状态和位姿。
 */
struct TrackerPoseSample {
  /** OpenVR 在当前会话中分配的设备索引。 */
  std::uint32_t device_index{0};
  /** Tracker 的 LHR 序列号。 */
  std::string serial_number{};
  /** Tracker 是否连接到 SteamVR。 */
  bool device_connected{false};
  /** 当前位姿是否有效。 */
  bool pose_valid{false};
  /** OpenVR 报告的跟踪状态。 */
  TrackingState tracking_state{TrackingState::kUninitialized};
  /** 当前位姿；仅在 pose_valid 为 true 时使用。 */
  Pose pose{};
};

/** @brief 包围单次 OpenVR 查询的主机时钟读数，单位均为纳秒。 */
struct TrackerQueryTiming {
  /** OpenVR 调用前读取的稳定时钟时间。 */
  std::int64_t steady_before_ns{0};
  /** OpenVR 调用后读取的稳定时钟时间。 */
  std::int64_t steady_after_ns{0};
  /** OpenVR 调用前读取的系统时钟 Unix 时间。 */
  std::int64_t system_before_ns{0};
  /** OpenVR 调用后读取的系统时钟 Unix 时间。 */
  std::int64_t system_after_ns{0};
};

/** @brief 一次 OpenVR 查询得到的统一时间上下文和全部 Tracker 样本。 */
struct TrackerPoseBatch {
  /** 当前查询的主机时钟区间，用于为所有样本生成同一批次时间戳。 */
  TrackerQueryTiming timing{};
  /** 当前 OpenVR 会话中所有 Generic Tracker 的采样结果。 */
  std::vector<TrackerPoseSample> samples{};
};

/**
 * @brief 将跟踪状态转换为稳定的输出字符串。
 * @param state 跟踪状态。
 * @return 对应的英文小写状态字符串。
 */
std::string_view TrackingStateToString(TrackingState state) noexcept;

} // namespace vive_tracker
