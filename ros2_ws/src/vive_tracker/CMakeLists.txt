# 本文件用于构建和安装 VIVE Tracker ROS 2 位姿发布节点及其测试。

cmake_minimum_required(VERSION 3.16)
project(vive_tracker VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)

set(OPENVR_SDK_ROOT "" CACHE PATH "Path to the OpenVR SDK root")

find_path(
  OPENVR_INCLUDE_DIR
  NAMES openvr.h
  HINTS "${OPENVR_SDK_ROOT}" "$ENV{OPENVR_SDK_ROOT}"
  PATH_SUFFIXES headers include
)

find_library(
  OPENVR_LIBRARY
  NAMES openvr_api
  HINTS "${OPENVR_SDK_ROOT}" "$ENV{OPENVR_SDK_ROOT}"
  PATH_SUFFIXES lib/linux64 bin/linux64 lib
)

if(NOT OPENVR_INCLUDE_DIR OR NOT OPENVR_LIBRARY)
  message(
    FATAL_ERROR
    "OpenVR SDK was not found. Configure with -DOPENVR_SDK_ROOT=/path/to/openvr"
  )
endif()

add_library(
  vive_tracker_core STATIC
  src/path_history.cpp
  src/pose_math.cpp
  src/tracker_pose_reader.cpp
)
target_compile_features(vive_tracker_core PUBLIC cxx_std_17)
target_include_directories(
  vive_tracker_core
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:include>"
    "${OPENVR_INCLUDE_DIR}"
)
ament_target_dependencies(vive_tracker_core geometry_msgs nav_msgs)
target_link_libraries(vive_tracker_core "${OPENVR_LIBRARY}")

add_executable(vive_tracker_node src/vive_tracker_node.cpp)
target_compile_features(vive_tracker_node PRIVATE cxx_std_17)
ament_target_dependencies(
  vive_tracker_node
  geometry_msgs
  nav_msgs
  rclcpp
  tf2_ros
)
target_link_libraries(vive_tracker_node vive_tracker_core)
set_target_properties(
  vive_tracker_node
  PROPERTIES BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN"
)

add_custom_command(
  TARGET vive_tracker_node POST_BUILD
  COMMAND
    "${CMAKE_COMMAND}" -E copy_if_different "${OPENVR_LIBRARY}"
    "$<TARGET_FILE_DIR:vive_tracker_node>"
  COMMENT "Copying the OpenVR runtime library next to vive_tracker_node"
)

install(
  TARGETS vive_tracker_core
  ARCHIVE DESTINATION lib
)
install(
  TARGETS vive_tracker_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(
  FILES "${OPENVR_LIBRARY}"
  DESTINATION lib/${PROJECT_NAME}
)
install(
  DIRECTORY include/
  DESTINATION include
)
install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)
install(
  FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  # 保留仓库现有 clang-format 风格，跳过使用不同样式规则的检查器。
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(
    vive_tracker_core_test
    tests/vive_tracker_core_test.cpp
  )
  if(TARGET vive_tracker_core_test)
    target_compile_features(vive_tracker_core_test PRIVATE cxx_std_17)
    target_include_directories(
      vive_tracker_core_test
      PRIVATE "${OPENVR_INCLUDE_DIR}"
    )
    ament_target_dependencies(
      vive_tracker_core_test
      geometry_msgs
      nav_msgs
    )
    target_link_libraries(vive_tracker_core_test vive_tracker_core)
  endif()

  ament_add_pytest_test(
    vive_tracker_launch_test
    tests/test_vive_tracker_launch.py
  )
endif()

ament_package()
