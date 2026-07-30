/**
 * @file pose_math.hpp
 * @brief 声明 OpenVR 位姿解析、ROS 坐标转换和首帧相对位姿接口。
 */

#pragma once

#include "vive_tracker/tracker_pose_types.hpp"

namespace vr {
struct HmdMatrix34_t;
} // namespace vr

namespace vive_tracker {

/**
 * @brief 将 OpenVR 的 3×4 刚体变换矩阵转换为原始 OpenVR 位姿。
 * @param matrix OpenVR 返回的设备到绝对跟踪坐标系变换矩阵。
 * @return 位置单位为米、四元数顺序为 xyzw 的 OpenVR 位姿。
 */
Pose ConvertOpenVrMatrixToPose(const vr::HmdMatrix34_t &matrix) noexcept;

/**
 * @brief 将 OpenVR 位姿转换到轴向重排后的 ROS 跟踪坐标系。
 * @param openvr_pose 使用 VIVE Tracker 3.0 原始坐标轴的 OpenVR 位姿。
 * @return 将父坐标系原始正 Z、正 X、正 Y 依次映射到 ROS 正 X、正 Y、正 Z
 * 的位姿。
 */
Pose ConvertOpenVrPoseToRosPose(const Pose &openvr_pose) noexcept;

/**
 * @brief 根据发布配置选择 OpenVR 原始位姿或轴向重排位姿。
 * @param openvr_pose 使用 OpenVR 原始坐标定义的 Tracker 位姿。
 * @param reorder_pose_axes 是否将全局坐标轴重排为 ROS 跟踪坐标。
 * @return 关闭重排时返回原始位姿，开启时返回 ROS 跟踪坐标位姿。
 */
Pose SelectPublishedPose(const Pose &openvr_pose,
                         bool reorder_pose_axes) noexcept;

/**
 * @brief 计算当前位姿相对于参考位姿所定义坐标系的刚体变换。
 * @param reference_pose 作为相对坐标系原点和轴向的参考位姿。
 * @param current_pose 与参考位姿表达在同一父坐标系中的当前位姿。
 * @return 满足 reference_pose × relative_pose = current_pose 的相对位姿。
 */
Pose CalculateRelativePose(const Pose &reference_pose,
                           const Pose &current_pose) noexcept;

/**
 * @brief 获取轴向重排后的 ROS 跟踪坐标系在 OpenVR 全局坐标系中的方向。
 * @return 用于发布 OpenVR 全局坐标系到 ROS 跟踪坐标系静态 TF 的单位四元数。
 */
Quaternion GetRosTrackingFrameOrientationInOpenVr() noexcept;

} // namespace vive_tracker
