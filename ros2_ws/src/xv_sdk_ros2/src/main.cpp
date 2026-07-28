/**
 * @file main.cpp
 * @brief XV SDK ROS 2 节点入口。
 */

#include "xv_ros2_node.h"
using namespace xv;
using namespace std::chrono_literals;

/**
 * @brief 初始化 ROS 2、创建 XV SDK 节点并进入多线程执行器。
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @return 程序退出状态码。
 */
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<xvision_ros2_node>();
    node->init();
    rclcpp::executors::MultiThreadedExecutor executor(
    rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}
