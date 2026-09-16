// Copyright 2026 karmueo

#include <chrono>
#include <csignal>
#include <memory>
#include <string>

#include <image_transport/subscriber_plugin.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace fastumi_usb_camera
{

class UsbCameraReceiver : public rclcpp::Node
{
public:
  explicit UsbCameraReceiver(
    pluginlib::ClassLoader<image_transport::SubscriberPlugin> & loader)
  : Node("usb_camera_receiver")
  {
    const auto input_topic = declare_parameter<std::string>(
      "input_topic", "/usb_camera/image_raw");
    const auto output_topic = declare_parameter<std::string>(
      "output_topic", "/usb_camera/image_decoded");
    if (input_topic.empty() || input_topic.front() != '/' ||
      input_topic.back() == '/' || output_topic.empty() ||
      output_topic.front() != '/' || output_topic.back() == '/')
    {
      throw std::invalid_argument("input_topic and output_topic must be absolute ROS topics");
    }

    output_ = create_publisher<sensor_msgs::msg::Image>(
      output_topic, rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile());
    decoder_ = loader.createUniqueInstance(
      image_transport::SubscriberPlugin::getLookupName("ffmpeg"));

    auto qos = rmw_qos_profile_sensor_data;
    qos.history = RMW_QOS_POLICY_HISTORY_KEEP_LAST;
    qos.depth = 20;
    qos.reliability = RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT;
    qos.durability = RMW_QOS_POLICY_DURABILITY_VOLATILE;
    decoder_->subscribe(
      this, input_topic,
      [this](const sensor_msgs::msg::Image::ConstSharedPtr & image) {
        output_->publish(*image);
      }, qos);

    RCLCPP_INFO(
      get_logger(), "Decoding %s/ffmpeg -> %s",
      input_topic.c_str(), output_topic.c_str());
  }

  ~UsbCameraReceiver() override
  {
    if (decoder_) {
      decoder_->shutdown();
      decoder_.reset();
    }
    output_.reset();
  }

private:
  pluginlib::UniquePtr<image_transport::SubscriberPlugin> decoder_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr output_;
};

}  // namespace fastumi_usb_camera

namespace
{

volatile std::sig_atomic_t stop_requested = 0;

void request_stop(int signal_number)
{
  (void)signal_number;
  stop_requested = 1;
}

}  // namespace

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
