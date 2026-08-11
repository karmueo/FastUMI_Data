# 本文件构建并导出 FastUMI 回放标注 RViz2 Panel 插件。
cmake_minimum_required(VERSION 3.8)
project(fastumi_rviz_plugins)

find_package(ament_cmake REQUIRED)
find_package(fastumi_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_interfaces REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(rviz_common REQUIRED)
find_package(std_srvs REQUIRED)

set(CMAKE_AUTOMOC ON)
add_library(${PROJECT_NAME} SHARED
  src/replay_annotation_panel.cpp
  include/fastumi_rviz_plugins/replay_annotation_panel.hpp
  include/fastumi_rviz_plugins/panel_state.hpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} PUBLIC include)
ament_target_dependencies(${PROJECT_NAME}
  fastumi_interfaces geometry_msgs pluginlib rclcpp rosbag2_interfaces
  rosgraph_msgs rviz_common std_srvs)
pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
install(FILES plugin_description.xml DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_panel_gating test/test_panel_gating.cpp)
  target_include_directories(test_panel_gating PRIVATE include)
  target_compile_definitions(test_panel_gating PRIVATE
    FASTUMI_RVIZ_PLUGIN_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
endif()

ament_package()
