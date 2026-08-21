# 本文件用于独立构建直接调用随附 stereo_camera SDK 的元数据观察示例。

cmake_minimum_required(VERSION 3.16)
project(stereo_camera_example LANGUAGES CXX)

find_package(OpenGL REQUIRED)
find_package(SDL2 REQUIRED)
find_package(Threads REQUIRED)

set(SDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(SDK_INCLUDE_DIR "${SDK_ROOT}/include")
set(SDK_LIBRARY "${SDK_ROOT}/lib/libstereo_camera.so")

if(NOT EXISTS "${SDK_INCLUDE_DIR}/stereo_camera/stereo_camera.h")
  message(FATAL_ERROR "stereo_camera SDK header not found: ${SDK_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${SDK_LIBRARY}")
  message(FATAL_ERROR "stereo_camera SDK library not found: ${SDK_LIBRARY}")
endif()

add_executable(
  stereo_camera_example
  stereo_camera_example.cpp
  frame_display.cpp
)
target_compile_features(stereo_camera_example PRIVATE cxx_std_17)
target_compile_options(stereo_camera_example PRIVATE -Wall -Wextra -Wpedantic)
target_include_directories(stereo_camera_example PRIVATE "${SDK_INCLUDE_DIR}")
target_link_libraries(
  stereo_camera_example
  PRIVATE
    "${SDK_LIBRARY}"
    OpenGL::GL
    SDL2::SDL2
    Threads::Threads
)

# 构建目录直接使用源码树中的随附动态库；安装时保持同级 lib 目录可运行。
set_target_properties(
  stereo_camera_example
  PROPERTIES
    BUILD_RPATH "${SDK_ROOT}/lib"
    INSTALL_RPATH "$ORIGIN/../lib"
)
