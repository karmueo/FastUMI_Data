/**
 * @file frame_validation.cpp
 * @brief 实现 VIVE Tracker 节点坐标系参数校验。
 */

#include "vive_tracker/frame_validation.hpp"

#include <stdexcept>
#include <string>

namespace vive_tracker {
namespace {

/**
 * @brief 验证 frame ID 非空且不带前导斜杠。
 * @param value 待验证的 frame ID。
 * @param parameter_name 对应的 ROS 参数名称。
 * @throws std::invalid_argument frame ID 无效时抛出。
 */
void ValidateFrameId(const std::string &value,
                     const std::string &parameter_name) {
  if (value.empty()) {
    throw std::invalid_argument(parameter_name + " must not be empty");
  }
  if (value.front() == '/') {
    throw std::invalid_argument(parameter_name +
                                " must not start with a slash");
  }
}

} // namespace

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
                                bool reorder_pose_axes) {
  ValidateFrameId(openvr_frame, "openvr_frame");
  ValidateFrameId(odom_frame, "odom_frame");
  ValidateFrameId(child_frame, "child_frame");

  if (openvr_frame == odom_frame || openvr_frame == child_frame ||
      odom_frame == child_frame) {
    throw std::invalid_argument(
        "openvr_frame, odom_frame, and child_frame must be different");
  }

  if (!reorder_pose_axes) {
    return;
  }

  ValidateFrameId(parent_frame, "parent_frame");
  if (parent_frame == openvr_frame || parent_frame == odom_frame ||
      parent_frame == child_frame) {
    throw std::invalid_argument(
        "openvr_frame, parent_frame, odom_frame, and child_frame must be "
        "different");
  }
}

} // namespace vive_tracker
