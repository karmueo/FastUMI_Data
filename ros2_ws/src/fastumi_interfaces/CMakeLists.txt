# 本文件用于生成 FastUMI 数据采集流程共享的 ROS 2 消息接口。

cmake_minimum_required(VERSION 3.16)
project(fastumi_interfaces VERSION 0.1.0)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  "msg/EpisodeEvent.msg"
  "msg/GripperState.msg"
  "msg/TrackerStatus.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
