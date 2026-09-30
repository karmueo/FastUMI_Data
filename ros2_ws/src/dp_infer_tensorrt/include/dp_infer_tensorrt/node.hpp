#pragma once
// ROS 回调使用单线程 executor，模型只由一个持久工作线程调用。
#include "dp_infer_tensorrt/core.hpp"
#include <rclcpp/rclcpp.hpp>

namespace dp_infer_tensorrt {
class DpInferenceNode final : public rclcpp::Node {
public:
    explicit DpInferenceNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions(),
                             std::shared_ptr<InferenceBackend> backend = nullptr);
    ~DpInferenceNode() override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace dp_infer_tensorrt
