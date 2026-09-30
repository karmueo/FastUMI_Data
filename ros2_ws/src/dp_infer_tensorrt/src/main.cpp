// C++ 推理入口；默认 executor 保证所有 ROS 状态操作串行执行。
#include "dp_infer_tensorrt/node.hpp"
#include <iostream>

int main(int argc,char **argv) {
    rclcpp::init(argc,argv);
    try {
        auto node = std::make_shared<dp_infer_tensorrt::DpInferenceNode>();
        rclcpp::spin(node);
        node.reset();
        rclcpp::shutdown();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "dp_infer_tensorrt startup/runtime failed: " << error.what() << '\n';
        rclcpp::shutdown();
        return 1;
    }
}
