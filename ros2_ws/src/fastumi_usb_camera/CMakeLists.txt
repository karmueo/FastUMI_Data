cmake_minimum_required(VERSION 3.16)
project(fastumi_usb_camera)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

# Use the interpreter selected by colcon (including an active virtualenv) for
# both the installed Python entry point and ament_cmake_python.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(image_transport REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs)
find_path(LIBUVC_INCLUDE_DIR NAMES libuvc/libuvc.h REQUIRED)
find_library(LIBUVC_LIBRARY NAMES uvc REQUIRED)

ament_python_install_package(${PROJECT_NAME})
configure_file(scripts/usb_camera_node.in
  ${CMAKE_CURRENT_BINARY_DIR}/usb_camera_node @ONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/usb_camera_node
  DESTINATION lib/${PROJECT_NAME})

add_library(${PROJECT_NAME}_core INTERFACE)
target_include_directories(${PROJECT_NAME}_core INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")

add_executable(usb_camera_ffmpeg src/usb_camera_ffmpeg.cpp)
target_include_directories(usb_camera_ffmpeg PRIVATE
  ${LIBUVC_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(usb_camera_ffmpeg
  ${PROJECT_NAME}_core ${LIBUVC_LIBRARY} ${OpenCV_LIBRARIES})
ament_target_dependencies(usb_camera_ffmpeg
  image_transport pluginlib rclcpp sensor_msgs)

add_executable(usb_camera_receiver src/usb_camera_receiver.cpp)
ament_target_dependencies(usb_camera_receiver
  image_transport pluginlib rclcpp sensor_msgs)

install(TARGETS usb_camera_ffmpeg usb_camera_receiver
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ffmpeg_image_transport_msgs REQUIRED)

  ament_add_pytest_test(test_python test)

  ament_add_gtest(test_configuration test/test_configuration.cpp)
  target_link_libraries(test_configuration ${PROJECT_NAME}_core)

  ament_add_gtest(test_physical_device test/test_physical_device.cpp)
  target_link_libraries(test_physical_device ${PROJECT_NAME}_core)

  ament_add_gtest(test_latest_frame_buffer test/test_latest_frame_buffer.cpp)
  target_link_libraries(test_latest_frame_buffer ${PROJECT_NAME}_core)

  ament_add_gtest(test_jpeg_frame test/test_jpeg_frame.cpp)
  target_link_libraries(test_jpeg_frame ${PROJECT_NAME}_core ${OpenCV_LIBRARIES})
  target_include_directories(test_jpeg_frame PRIVATE ${OpenCV_INCLUDE_DIRS})

  add_executable(test_ffmpeg_round_trip test/test_ffmpeg_round_trip.cpp)
  ament_target_dependencies(test_ffmpeg_round_trip
    ffmpeg_image_transport_msgs image_transport pluginlib rclcpp sensor_msgs)
  add_test(NAME test_ffmpeg_round_trip COMMAND test_ffmpeg_round_trip)
  set_tests_properties(test_ffmpeg_round_trip PROPERTIES TIMEOUT 30)
endif()

ament_package()
