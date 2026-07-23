/**
 * @file path_history.cpp
 * @brief 实现 ROS 2 位姿轨迹的有界追加工具。
 */

#include "vive_tracker/path_history.hpp"

#include <cstddef>
#include <stdexcept>

namespace vive_tracker {

/**
 * @brief 将位姿追加到轨迹，并移除超出上限的最旧采样。
 * @param pose 待追加的有效位姿。
 * @param max_points 轨迹允许保留的最大点数，必须大于零。
 * @param path 接收更新后的轨迹消息，不能为空。
 */
void AppendPoseToBoundedPath(const geometry_msgs::msg::PoseStamped &pose,
                             std::size_t max_points,
                             nav_msgs::msg::Path *path) {
  if (path == nullptr) {
    throw std::invalid_argument("path must not be null");
  }
  if (max_points == 0) {
    throw std::invalid_argument("max_points must be greater than zero");
  }

  path->poses.push_back(pose);
  if (path->poses.size() <= max_points) {
    return;
  }

  /** 超出上限、需要从轨迹头部移除的点数。 */
  const std::size_t excess_points = path->poses.size() - max_points;
  path->poses.erase(path->poses.begin(),
                    path->poses.begin() +
                        static_cast<std::ptrdiff_t>(excess_points));
}

} // namespace vive_tracker
