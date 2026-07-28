/**
 * @file image_snapshot_node.h
 * @brief 声明订阅指定图像话题并保存首帧的一次性 ROS 2 节点。
 */

#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "image_snapshot_utils.h"

namespace xv_ros2::image_snapshot {

/** @brief 订阅指定图像话题并保存首个有效帧的一次性节点。 */
class ImageSnapshotNode : public rclcpp::Node {
public:
  /**
   * @brief 创建截图节点并读取参数。
   * @param options ROS 2 节点选项，用于传入参数覆盖。
   */
  explicit ImageSnapshotNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

  /**
   * @brief 查询节点是否已完成保存或失败退出。
   * @return 节点已完成时返回 true。
   */
  bool finished() const;

  /**
   * @brief 获取节点退出状态码。
   * @return 0 表示保存成功，非零表示参数、超时或保存失败。
   */
  int exitCode() const;

  /**
   * @brief 获取成功保存的完整文件路径。
   * @return 成功时返回输出路径，失败或未完成时为空。
   */
  std::filesystem::path outputPath() const;

private:
  /**
   * @brief 处理收到的首个图像消息并结束节点。
   * @param image 收到的 ROS 图像消息。
   */
  void handleImage(const sensor_msgs::msg::Image::ConstSharedPtr &image);

  /** @brief 处理等待首帧超时并结束节点。 */
  void handleTimeout();

  /**
   * @brief 记录失败状态并停止后续订阅和定时器事件。
   * @param message 需要写入日志的失败原因。
   * @param exit_code 进程退出状态码。
   */
  void finishWithError(const std::string &message, int exit_code);

  /** 图像订阅器。 */
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  /** 等待首帧的超时定时器。 */
  rclcpp::TimerBase::SharedPtr timeout_timer_;
  /** 图像保存选项。 */
  SaveOptions save_options_;
  /** 节点是否已经结束。 */
  std::atomic<bool> finished_{false};
  /** 节点退出状态码。 */
  std::atomic<int> exit_code_{1};
  /** 成功保存的输出路径。 */
  std::filesystem::path output_path_;
};

} // namespace xv_ros2::image_snapshot
