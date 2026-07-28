cmake_minimum_required(VERSION 3.8)
project(xv_sdk_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -g)
endif()

add_compile_options(-Wno-pedantic)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(xv_ros2_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)

# XV SDK 3.2.0 is installed from the SDK Debian package before building.
find_package(xvsdk 3.2.0 EXACT CONFIG QUIET)
if(NOT xvsdk_FOUND)
  message(FATAL_ERROR
    "XV SDK 3.2.0 was not found. Install "
    "/home/scl/work/UMI/FastUMI_Hardware_SDK/xv/sdk/1217/"
    "XVSDK_jammy_amd64_1217.deb, then rebuild this workspace.")
endif()

# xvsdkConfig.cmake provides these variables without imported CMake targets.
set(XVSDK_INCLUDE_DIRS ${xvsdk_INCLUDE_DIRS}/xvsdk)
set(XVSDK_LIBRARIES ${xvsdk_LIBRARIES})
find_path(XVSDK_EXTENDED_INCLUDE_DIR
  NAMES xv-sdk-ex.h
  PATHS /usr/include2
)
if(NOT XVSDK_EXTENDED_INCLUDE_DIR)
  message(FATAL_ERROR "XV SDK extended headers are missing from /usr/include2.")
endif()
message(STATUS "XV SDK ${xvsdk_VERSION}: ${XVSDK_LIBRARY_DIRS}")

# Ubuntu 24.04 的系统 SuiteSparse 与 SDK 自带 Ceres/SuiteSparse ABI 不完全一致。
# 显式链接 SDK 随包库，避免链接器优先选择 /usr/lib/x86_64-linux-gnu 下的系统库。
set(XVSDK_BUNDLED_LIBRARIES
  /usr/lib/libceres.so
  /usr/lib/libspqr.so
  /usr/lib/libcholmod.so
  /usr/lib/liblapack.so
  /usr/lib/libblas.so
  /usr/lib/libcxsparse.so
  /usr/lib/libsuitesparseconfig.so
)
foreach(XVSDK_BUNDLED_LIBRARY IN LISTS XVSDK_BUNDLED_LIBRARIES)
  if(NOT EXISTS ${XVSDK_BUNDLED_LIBRARY})
    message(FATAL_ERROR "XV SDK runtime library is missing: ${XVSDK_BUNDLED_LIBRARY}")
  endif()
endforeach()

include_directories(
  include
  ${XVSDK_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${XVSDK_EXTENDED_INCLUDE_DIR}
)

add_executable(xv_cameras
  src/main.cpp
  src/xv_ros2_node.cpp
  src/xv_dev_wrapper.cpp
  src/factory_rgbd_utils.cpp
  src/rgb_registered_utils.cpp
  src/rgb_fisheye_undistort_utils.cpp
  src/rgbd_image_utils.cpp
  src/timestamp_utils.cpp
)
set_target_properties(xv_cameras PROPERTIES
  BUILD_RPATH "/usr/lib"
  INSTALL_RPATH "/usr/lib"
  LINK_FLAGS "-Wl,--disable-new-dtags"
)
target_include_directories(xv_cameras PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(xv_cameras PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(xv_cameras
  rclcpp
  std_msgs
  sensor_msgs
  xv_ros2_msgs
  std_srvs
  builtin_interfaces
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  image_transport
  ament_index_cpp)

target_link_libraries(xv_cameras ${XVSDK_LIBRARIES} ${XVSDK_BUNDLED_LIBRARIES} pthread yaml-cpp ${OpenCV_LIBS})

add_executable(image_snapshot
  src/image_snapshot_main.cpp
  src/image_snapshot_node.cpp
  src/image_snapshot_utils.cpp
)
target_include_directories(image_snapshot PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(image_snapshot PUBLIC cxx_std_17)
ament_target_dependencies(image_snapshot
  rclcpp
  sensor_msgs
  cv_bridge)
target_link_libraries(image_snapshot ${OpenCV_LIBS})

install(TARGETS xv_cameras image_snapshot
  DESTINATION lib/${PROJECT_NAME})

# Install launch files.
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)
install(FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_add_gtest(test_rgb_registered_utils
    test/test_rgb_registered_utils.cpp
    src/rgb_registered_utils.cpp
    src/rgb_fisheye_undistort_utils.cpp
  )
  target_include_directories(test_rgb_registered_utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${XVSDK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${XVSDK_EXTENDED_INCLUDE_DIR}
  )
  ament_target_dependencies(test_rgb_registered_utils
    sensor_msgs
    builtin_interfaces)
  target_link_libraries(test_rgb_registered_utils
    ${XVSDK_LIBRARIES}
    ${XVSDK_BUNDLED_LIBRARIES}
    yaml-cpp
    ${OpenCV_LIBS})

  ament_add_gtest(test_factory_rgbd_utils
    test/test_factory_rgbd_utils.cpp
    src/factory_rgbd_utils.cpp
    src/rgb_registered_utils.cpp
    src/rgb_fisheye_undistort_utils.cpp
  )
  target_include_directories(test_factory_rgbd_utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${XVSDK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${XVSDK_EXTENDED_INCLUDE_DIR}
  )
  target_compile_features(test_factory_rgbd_utils PUBLIC c_std_99 cxx_std_17)
  target_compile_definitions(test_factory_rgbd_utils PRIVATE
    XV_SDK_ROS2_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/fixtures"
  )
  ament_target_dependencies(test_factory_rgbd_utils
    sensor_msgs
    builtin_interfaces)
  target_link_libraries(test_factory_rgbd_utils
    ${XVSDK_LIBRARIES}
    ${XVSDK_BUNDLED_LIBRARIES}
    yaml-cpp
    ${OpenCV_LIBS})

  ament_add_gtest(test_rgbd_image_utils
    test/test_rgbd_image_utils.cpp
    src/rgbd_image_utils.cpp
    src/timestamp_utils.cpp
  )
  target_include_directories(test_rgbd_image_utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${XVSDK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${XVSDK_EXTENDED_INCLUDE_DIR}
  )
  target_compile_features(test_rgbd_image_utils PUBLIC c_std_99 cxx_std_17)
  ament_target_dependencies(test_rgbd_image_utils
    sensor_msgs
    xv_ros2_msgs)

  ament_add_gtest(test_rgb_fisheye_undistort_utils
    test/test_rgb_fisheye_undistort_utils.cpp
    src/rgb_fisheye_undistort_utils.cpp
  )
  target_include_directories(test_rgb_fisheye_undistort_utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${OpenCV_INCLUDE_DIRS}
  )
  target_compile_features(test_rgb_fisheye_undistort_utils PUBLIC c_std_99 cxx_std_17)
  target_compile_definitions(test_rgb_fisheye_undistort_utils PRIVATE
    XV_SDK_ROS2_TEST_CONFIG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/config"
  )
  ament_target_dependencies(test_rgb_fisheye_undistort_utils
    sensor_msgs
    builtin_interfaces)
  target_link_libraries(test_rgb_fisheye_undistort_utils yaml-cpp ${OpenCV_LIBS})

  ament_add_gtest(test_timestamp_utils
    test/test_timestamp_utils.cpp
    src/timestamp_utils.cpp
  )
  target_include_directories(test_timestamp_utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_compile_features(test_timestamp_utils PUBLIC c_std_99 cxx_std_17)
  ament_target_dependencies(test_timestamp_utils
    builtin_interfaces)

  ament_add_gtest(test_image_snapshot
    test/test_image_snapshot.cpp
    src/image_snapshot_node.cpp
    src/image_snapshot_utils.cpp
  )
  target_include_directories(test_image_snapshot PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${OpenCV_INCLUDE_DIRS}
  )
  target_compile_features(test_image_snapshot PUBLIC cxx_std_17)
  ament_target_dependencies(test_image_snapshot
    rclcpp
    sensor_msgs
    cv_bridge)
  target_link_libraries(test_image_snapshot ${OpenCV_LIBS})

  ament_add_gtest(test_xv_ros2_node_publishers
    test/test_xv_ros2_node_publishers.cpp
    test/test_xv_ros2_node_stubs.cpp
    src/timestamp_utils.cpp
    src/xv_ros2_node.cpp
  )
  target_include_directories(test_xv_ros2_node_publishers PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${XVSDK_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${XVSDK_EXTENDED_INCLUDE_DIR}
  )
  target_compile_features(test_xv_ros2_node_publishers PUBLIC c_std_99 cxx_std_17)
  ament_target_dependencies(test_xv_ros2_node_publishers
    rclcpp
    std_msgs
    sensor_msgs
    xv_ros2_msgs
    std_srvs
    builtin_interfaces
    geometry_msgs
    nav_msgs
    tf2
    tf2_ros
    image_transport
    ament_index_cpp)
  target_link_libraries(test_xv_ros2_node_publishers ${OpenCV_LIBS})
  # 导入的历史 C++ 代码保留原有格式；功能测试和其余静态检查继续执行。
  set(AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_cpplint
    ament_cmake_uncrustify
  )
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
