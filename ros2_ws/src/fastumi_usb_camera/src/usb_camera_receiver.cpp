/**
 * @file usb_camera_receiver.cpp
 * @brief 启动 FFmpeg 视频包接收节点（transport、cuda 或 auto 解码后端）。
 */
// Copyright 2026 karmueo

#include <chrono>
#include <csignal>
#include <memory>

#include <image_transport/subscriber_plugin.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>

#include "fastumi_usb_camera/usb_camera_receiver.hpp"

namespace
{

/** 信号处理器设置的停止标记。 */
volatile std::sig_atomic_t stop_requested = 0;

/**
 * @brief 在收到终止信号时请求退出。
 * @param signal_number 收到的信号编号。
 */
void request_stop(int signal_number)
{
  (void)signal_number;
  stop_requested = 1;
}

}  // namespace

/**
 * @brief 启动接收节点并处理退出信号。
 * @param argc 参数个数。
 * @param argv 参数数组。
 * @return 进程退出码。
 */
int main(int argc, char ** argv)
{
  rclcpp::init(
    argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, request_stop);
  std::signal(SIGTERM, request_stop);
  try {
    {
      pluginlib::ClassLoader<image_transport::SubscriberPlugin> loader(
        "image_transport", "image_transport::SubscriberPlugin");
      auto node = std::make_shared<fastumi_usb_camera::UsbCameraReceiver>(loader);
      rclcpp::executors::SingleThreadedExecutor executor;
      executor.add_node(node);
      while (!stop_requested) {
        executor.spin_once(std::chrono::milliseconds(100));
      }
      executor.remove_node(node);
      node.reset();
    }
    rclcpp::shutdown();
    return 0;
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("usb_camera_receiver"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
}
