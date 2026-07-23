/**
 * @file tracker_pose_reader.hpp
 * @brief 声明基于 OpenVR 的 VIVE Tracker 位姿读取器。
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vive_tracker/tracker_pose_types.hpp"

namespace vive_tracker {

/**
 * @brief 管理 OpenVR 生命周期并读取所有 Generic Tracker 位姿。
 */
class TrackerPoseReader {
public:
  /**
   * @brief 创建尚未初始化的 Tracker 位姿读取器。
   */
  TrackerPoseReader();

  /**
   * @brief 关闭已经初始化的 OpenVR 会话。
   */
  ~TrackerPoseReader();

  TrackerPoseReader(const TrackerPoseReader &) = delete;
  TrackerPoseReader &operator=(const TrackerPoseReader &) = delete;

  /**
   * @brief 以后台应用类型连接到已经运行的 SteamVR。
   * @param error_message 初始化失败时接收可读错误信息；允许传入 nullptr。
   * @return 初始化成功或读取器已经初始化时返回 true。
   */
  bool Initialize(std::string *error_message);

  /**
   * @brief 查询读取器是否已经成功连接 OpenVR。
   * @return 已初始化返回 true。
   */
  bool IsInitialized() const noexcept;

  /**
   * @brief 读取当前所有 Generic Tracker 的状态和位姿。
   * @param origin 本次查询使用的 SteamVR 跟踪原点。
   * @return 当前会话中所有 Generic Tracker 的采样结果；未初始化时返回空数组。
   */
  std::vector<TrackerPoseSample> ReadPoses(TrackingOrigin origin) const;

private:
  /** @brief 隐藏 OpenVR 类型和运行时状态的内部实现。 */
  class Impl;

  /** OpenVR 读取器的内部实现。 */
  std::unique_ptr<Impl> impl_;
};

} // namespace vive_tracker
