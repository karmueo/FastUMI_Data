/**
 * @file image_snapshot_main.cpp
 * @brief 提供一次性 ROS 2 图像截图节点的进程入口。
 */

#include "image_snapshot_node.h"

#include <chrono>
#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

/**
 * @brief 初始化并运行一次性截图节点。
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @return 0 表示截图成功，非零表示参数、超时或保存失败。
 */
int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  /** 截图节点退出状态码。 */
  int exit_code = 1;
  try {
    /** 一次性截图节点。 */
    auto node = std::make_shared<xv_ros2::image_snapshot::ImageSnapshotNode>();
    /** 驱动订阅和定时器回调的单线程执行器。 */
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    while (rclcpp::ok() && !node->finished()) {
      executor.spin_once(std::chrono::milliseconds(100));
    }
    exit_code = node->exitCode();
    executor.remove_node(node);
  } catch (const std::exception &exception) {
    RCLCPP_ERROR(rclcpp::get_logger("image_snapshot"), "节点启动失败：%s",
                 exception.what());
    exit_code = 5;
  }
  rclcpp::shutdown();
  return exit_code;
}
