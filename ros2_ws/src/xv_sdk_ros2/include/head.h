#pragma once

#include <chrono>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <deque>

#include <xv-sdk.h>
#include <xv-sdk-ex.h>

#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "xv_ros2_msgs/msg/orientation_stamped.hpp"
#include "xv_ros2_msgs/msg/color_depth.hpp"
#include "xv_ros2_msgs/msg/controller.hpp"
#include "xv_ros2_msgs/msg/event_data.hpp"
#include "xv_ros2_msgs/msg/button_msg.hpp"
#include "xv_ros2_msgs/msg/clamp.hpp"

#include "std_srvs/srv/set_bool.hpp"

#include "xv-sdk-private.h"

#include "rclcpp/rclcpp.hpp"

using rosImage = sensor_msgs::msg::Image;
using rosCamInfo = sensor_msgs::msg::CameraInfo;
using rosImu = sensor_msgs::msg::Imu;
using rosOrientationStamped = xv_ros2_msgs::msg::OrientationStamped;
using rosColorDepth = xv_ros2_msgs::msg::ColorDepth;
using rosController = xv_ros2_msgs::msg::Controller;
using rosEventData = xv_ros2_msgs::msg::EventData;
using rosButtonMsg = xv_ros2_msgs::msg::ButtonMsg;
using rosPointCloud2 = sensor_msgs::msg::PointCloud2;
using rosClamp = xv_ros2_msgs::msg::Clamp;
