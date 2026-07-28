cmake_minimum_required(VERSION 3.8)
project(xv_ros2_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/OrientationStamped.msg"
  "msg/ColorDepth.msg"
  "srv/GetOrientation.srv"
  "srv/GetOrientationAt.srv"
  "srv/GetPose.srv"
  "srv/GetPoseAt.srv"
  "srv/LoadMapAndSwitchCslam.srv"
  "srv/SaveMapAndSwitchCslam.srv"
  "srv/ControllerStart.srv"
  "srv/ControllerStop.srv"
  "msg/Controller.msg"
  "msg/EventData.msg"
  "msg/ButtonMsg.msg"
  "msg/Clamp.msg"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # 导入接口文件尚未附带上游版权声明，保留原始内容并跳过版权检查。
  set(AMENT_LINT_AUTO_EXCLUDE ament_cmake_copyright)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
