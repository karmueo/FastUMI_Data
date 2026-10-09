// Test-only source: native ROS Image publication mirrors the H264 C++ decoder.
// Publishes a fixed recording frame; no robot driver or motion interfaces.
#include <chrono>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

class MockCamera : public rclcpp::Node {
 public:
  MockCamera(const std::string& filename, int height, int width)
      : Node("dexgraspvla_mock_camera") {
    std::ifstream stream(filename, std::ios::binary);
    pixels_ = std::vector<uint8_t>(std::istreambuf_iterator<char>(stream), {});
    if (pixels_.size() != static_cast<size_t>(height * width * 3)) {
      throw std::runtime_error("Invalid fixed BGR frame");
    }
    height_ = height;
    width_ = width;
    publisher_ = create_publisher<sensor_msgs::msg::Image>(
        "/wrist_camera/image_decoded", rclcpp::SensorDataQoS().keep_last(1));
    timer_ = create_wall_timer(std::chrono::nanoseconds(33333333), [this]() {
      sensor_msgs::msg::Image image;
      image.header.stamp = now();
      image.header.frame_id = "wrist_camera_optical_frame";
      image.height = height_;
      image.width = width_;
      image.encoding = "bgr8";
      image.is_bigendian = false;
      image.step = width_ * 3;
      image.data = pixels_;
      publisher_->publish(image);
    });
  }

 private:
  std::vector<uint8_t> pixels_;
  int height_, width_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MockCamera>(argv[1], std::stoi(argv[2]), std::stoi(argv[3])));
  rclcpp::shutdown();
  return 0;
}
