# 本文件用于生成 FastUMI 数据采集流程共享的 ROS 2 消息接口。

cmake_minimum_required(VERSION 3.16)
project(fastumi_interfaces VERSION 0.1.0)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(
  ${PROJECT_NAME}
  "msg/EpisodeAnnotation.msg"
  "msg/EpisodeEvent.msg"
  "msg/FrameSequence.msg"
  "msg/GripperState.msg"
  "msg/PolicyActionSequence.msg"
  "msg/TrackerStatus.msg"
  "srv/DeleteEpisodeAnnotation.srv"
  "srv/ListEpisodeAnnotations.srv"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_frame_sequence_interface
    test/test_frame_sequence_interface.py)
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
