/**
 * @file image_snapshot_node.cpp
 * @brief 实现订阅指定图像话题并保存首帧的一次性 ROS 2 节点。
 */

#include "image_snapshot_node.h"

#include <chrono>
#include <functional>
#include <utility>

namespace xv_ros2::image_snapshot {

/**
 * @brief 创建截图节点并读取参数。
 * @param options ROS 2 节点选项，用于传入参数覆盖。
 */
ImageSnapshotNode::ImageSnapshotNode(const rclcpp::NodeOptions &options)
    : Node("image_snapshot", options) {
  /** 待订阅的完整图像话题名称。 */
  const std::string image_topic =
      declare_parameter<std::string>("image_topic", "");
  /** 截图输出目录参数。 */
  const std::string output_dir =
      declare_parameter<std::string>("output_dir", "/tmp/xv_sdk_snapshots");
  /** 可选输出文件名参数。 */
  save_options_.filename = declare_parameter<std::string>("filename", "");
  /** 已有文件覆盖开关。 */
  save_options_.overwrite = declare_parameter<bool>("overwrite", false);
  /** 等待首帧的超时秒数。 */
  const double timeout_sec = declare_parameter<double>("timeout_sec", 30.0);
  save_options_.output_dir = output_dir;

  if (image_topic.empty() || image_topic.front() != '/') {
    finishWithError("image_topic 必须是以 / 开头的完整图像话题名称", 2);
    return;
  }
  if (save_options_.output_dir.empty()) {
    finishWithError("output_dir 不能为空", 2);
    return;
  }
  if (timeout_sec < 0.0) {
    finishWithError("timeout_sec 不能小于 0", 2);
    return;
  }

  subscription_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic, rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&ImageSnapshotNode::handleImage, this, std::placeholders::_1));

  if (timeout_sec > 0.0) {
    /** 超时秒数转换得到的墙上时钟时长。 */
    const auto timeout_duration = std::chrono::duration<double>(timeout_sec);
    timeout_timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(timeout_duration),
        std::bind(&ImageSnapshotNode::handleTimeout, this));
  }

  RCLCPP_INFO(get_logger(), "等待话题 %s 的首个有效图像帧",
              image_topic.c_str());
}

/**
 * @brief 查询节点是否已完成保存或失败退出。
 * @return 节点已完成时返回 true。
 */
bool ImageSnapshotNode::finished() const { return finished_.load(); }

/**
 * @brief 获取节点退出状态码。
 * @return 0 表示保存成功，非零表示参数、超时或保存失败。
 */
int ImageSnapshotNode::exitCode() const { return exit_code_.load(); }

/**
 * @brief 获取成功保存的完整文件路径。
 * @return 成功时返回输出路径，失败或未完成时为空。
 */
std::filesystem::path ImageSnapshotNode::outputPath() const {
  return output_path_;
}

/**
 * @brief 处理收到的首个图像消息并结束节点。
 * @param image 收到的 ROS 图像消息。
 */
void ImageSnapshotNode::handleImage(
    const sensor_msgs::msg::Image::ConstSharedPtr &image) {
  if (finished_.exchange(true)) {
    return;
  }

  if (timeout_timer_) {
    timeout_timer_->cancel();
  }
  subscription_.reset();

  /** 当前图像的保存结果。 */
  const SaveResult result = saveImage(*image, save_options_);
  if (!result.success) {
    exit_code_.store(3);
    RCLCPP_ERROR(get_logger(), "截图失败：%s", result.error.c_str());
    return;
  }

  output_path_ = result.output_path;
  exit_code_.store(0);
  RCLCPP_INFO(get_logger(), "截图已保存：%s", output_path_.c_str());
}

/** @brief 处理等待首帧超时并结束节点。 */
void ImageSnapshotNode::handleTimeout() {
  finishWithError("等待首个图像帧超时", 4);
}

/**
 * @brief 记录失败状态并停止后续订阅和定时器事件。
 * @param message 需要写入日志的失败原因。
 * @param exit_code 进程退出状态码。
 */
void ImageSnapshotNode::finishWithError(const std::string &message,
                                        int exit_code) {
  if (finished_.exchange(true)) {
    return;
  }
  exit_code_.store(exit_code);
  if (timeout_timer_) {
    timeout_timer_->cancel();
  }
  subscription_.reset();
  RCLCPP_ERROR(get_logger(), "%s", message.c_str());
}

} // namespace xv_ros2::image_snapshot
