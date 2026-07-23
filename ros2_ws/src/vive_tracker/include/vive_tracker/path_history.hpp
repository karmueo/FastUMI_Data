/**
 * @file path_history.hpp
 * @brief 声明 ROS 2 位姿轨迹的有界追加工具。
 */

#pragma once

#include <cstddef>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>

namespace vive_tracker {

/**
 * @brief 将位姿追加到轨迹，并移除超出上限的最旧采样。
 * @param pose 待追加的有效位姿。
 * @param max_points 轨迹允许保留的最大点数，必须大于零。
 * @param path 接收更新后的轨迹消息，不能为空。
 */
void AppendPoseToBoundedPath(const geometry_msgs::msg::PoseStamped &pose,
                             std::size_t max_points, nav_msgs::msg::Path *path);

} // namespace vive_tracker
