# 本文件用于构建并安装 stereo_camera ROS 2 双目相机驱动包。
cmake_minimum_required(VERSION 3.16)
project(stereo_camera VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "stereo_camera requires Linux V4L2 support")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS calib3d core imgcodecs)

add_library(stereo_calibration src/stereo_calibration.cpp)
target_include_directories(stereo_calibration PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(stereo_calibration yaml-cpp ${OpenCV_LIBS})
ament_target_dependencies(stereo_calibration
  geometry_msgs
  sensor_msgs
  tf2)

add_library(stereo_imu src/imu.cpp)
target_include_directories(stereo_imu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(stereo_imu sensor_msgs)

add_executable(stereo_camera_node src/stereo_camera_node.cpp)
target_include_directories(stereo_camera_node PRIVATE include)
target_link_libraries(stereo_camera_node stereo_calibration stereo_imu ${OpenCV_LIBS})
ament_target_dependencies(stereo_camera_node
  geometry_msgs
  rcl_interfaces
  rclcpp
  sensor_msgs
  tf2_ros)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(stereo_imu PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(stereo_calibration PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(stereo_camera_node PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS stereo_calibration stereo_imu stereo_camera_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/
  DESTINATION include)
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE)
install(FILES README.md
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_gtest(test_imu test/test_imu.cpp)
  target_link_libraries(test_imu stereo_imu)
  ament_add_gtest(test_stereo_calibration test/test_stereo_calibration.cpp)
  target_compile_definitions(test_stereo_calibration PRIVATE
    STEREO_CALIBRATION_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/calib.yaml")
  target_link_libraries(test_stereo_calibration stereo_calibration)
  ament_target_dependencies(test_stereo_calibration geometry_msgs sensor_msgs tf2)
  ament_add_pytest_test(test_launch test/test_launch.py)
endif()

ament_export_include_directories(include)
ament_export_dependencies(geometry_msgs sensor_msgs tf2)
ament_export_libraries(stereo_calibration stereo_imu)

ament_package()
