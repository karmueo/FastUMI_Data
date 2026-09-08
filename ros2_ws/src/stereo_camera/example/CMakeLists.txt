# 本文件用于在 example 目录构建基于 V4L2 和 OpenCV 的双目相机实时预览程序。
cmake_minimum_required(VERSION 3.16)

project(stereo_camera_viewer LANGUAGES CXX)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc)

add_executable(stereo_viewer stereo_viewer.cpp)

target_compile_features(stereo_viewer PRIVATE cxx_std_17)
target_compile_options(stereo_viewer PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(stereo_viewer PRIVATE ${OpenCV_LIBS})
