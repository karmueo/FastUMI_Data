cmake_minimum_required(VERSION 3.16)
project(tof_stereo_camera LANGUAGES C CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "tof_stereo_camera requires Linux and the packaged SDK.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

set(STEREO_CAMERA_SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/sdk")
set(STEREO_CAMERA_LIBRARY "${STEREO_CAMERA_SDK_ROOT}/lib/libstereo_camera.so")
set(STEREO_CAMERA_DEBUG_LIBRARY
  "${STEREO_CAMERA_SDK_ROOT}/lib/libstereo_camera.so.debug")
set(STEREO_CAMERA_INCLUDE_DIR "${STEREO_CAMERA_SDK_ROOT}/include")
if(NOT EXISTS "${STEREO_CAMERA_LIBRARY}" OR
   NOT EXISTS "${STEREO_CAMERA_INCLUDE_DIR}/stereo_camera/stereo_camera.h")
  message(FATAL_ERROR "The packaged stereo_camera SDK was not found at ${STEREO_CAMERA_SDK_ROOT}.")
endif()

add_library(stereo_camera_sdk SHARED IMPORTED GLOBAL)
set_target_properties(stereo_camera_sdk PROPERTIES
  IMPORTED_LOCATION "${STEREO_CAMERA_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${STEREO_CAMERA_INCLUDE_DIR}"
)

add_library(frame_utils STATIC src/frame_utils.cpp)
target_include_directories(frame_utils PUBLIC include)
target_link_libraries(frame_utils PUBLIC stereo_camera_sdk ${OpenCV_LIBS})
ament_target_dependencies(frame_utils PUBLIC rclcpp sensor_msgs)

add_executable(tof_stereo_camera_node src/tof_stereo_camera_node.cpp)
target_link_libraries(tof_stereo_camera_node frame_utils stereo_camera_sdk)
ament_target_dependencies(tof_stereo_camera_node rclcpp sensor_msgs)
set_target_properties(tof_stereo_camera_node PROPERTIES
  BUILD_RPATH "${STEREO_CAMERA_SDK_ROOT}/lib"
  INSTALL_RPATH "$ORIGIN"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(frame_utils PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(tof_stereo_camera_node PRIVATE -Wall -Wextra -Wpedantic)
endif()

# 保留可执行文件和 SDK 二进制的真实安装副本，使其在 --symlink-install 下仍可自包含运行。
if(AMENT_CMAKE_SYMLINK_INSTALL)
  _install(TARGETS tof_stereo_camera_node
    DESTINATION lib/${PROJECT_NAME})
  _install(FILES "${STEREO_CAMERA_LIBRARY}" "${STEREO_CAMERA_DEBUG_LIBRARY}"
    DESTINATION lib/${PROJECT_NAME})
else()
  install(TARGETS tof_stereo_camera_node
    DESTINATION lib/${PROJECT_NAME})
  install(FILES "${STEREO_CAMERA_LIBRARY}" "${STEREO_CAMERA_DEBUG_LIBRARY}"
    DESTINATION lib/${PROJECT_NAME})
endif()
install(DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME})
install(DIRECTORY include/
  DESTINATION include)
install(DIRECTORY "${STEREO_CAMERA_INCLUDE_DIR}/"
  DESTINATION include)
install(FILES README.md
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_frame_utils test/test_frame_utils.cpp)
  if(TARGET test_frame_utils)
    target_link_libraries(test_frame_utils frame_utils stereo_camera_sdk)
    ament_target_dependencies(test_frame_utils rclcpp sensor_msgs)
    set_property(TARGET test_frame_utils PROPERTY EXCLUDE_FROM_ALL FALSE)
  endif()
endif()

ament_package()
