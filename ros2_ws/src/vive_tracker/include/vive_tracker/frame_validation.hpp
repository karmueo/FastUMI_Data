/**
 * @file frame_validation.hpp
 * @brief 声明 VIVE Tracker 节点坐标系参数校验接口。
 */

#pragma once

#include <string>

namespace vive_tracker {

/**
 * @brief 校验当前坐标模式实际使用的 frame ID 及其唯一性。
 * @param openvr_frame OpenVR 原始全局坐标系。
 * @param parent_frame 轴向重排后的全局坐标系。
 * @param odom_frame 第一条有效位姿定义的里程计坐标系。
 * @param child_frame Tracker 子坐标系。
 * @param reorder_pose_axes 是否启用全局坐标轴重排。
 * @throws std::invalid_argument 实际使用的 frame ID 无效或相互重名时抛出。
 */
void ValidateFrameConfiguration(const std::string &openvr_frame,
                                const std::string &parent_frame,
                                const std::string &odom_frame,
                                const std::string &child_frame,
                                bool reorder_pose_axes);

} // namespace vive_tracker
