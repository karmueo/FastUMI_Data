"""启动 FastUMI 夹爪开合度估计节点并加载默认参数。"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """创建加载默认 YAML 配置的 ROS 2 LaunchDescription。

    Returns:
        包含夹爪开合度估计节点的启动描述。
    """
    # 估计包安装后的共享目录。
    package_share = get_package_share_directory("fastumi_gripper_estimator")
    # ToF 相机安装后的共享目录。
    tof_camera_share = get_package_share_directory("tof_stereo_camera")
    # 默认节点参数文件路径。
    parameter_file = os.path.join(
        package_share, "config", "gripper_openness.yaml"
    )
    # ToF 驱动随包安装的默认相机标定文件路径。
    default_calibration_file = os.path.join(
        tof_camera_share, "config", "calibration.yaml"
    )
    # 输入话题参数默认选择 ToF 三维鱼眼解算使用的原始 RGB 图像。
    image_topic_argument = DeclareLaunchArgument(
        "image_topic",
        default_value="/tof_stereo_camera/rgb/image_raw",
        description="用于三维夹爪距离估计的原始 sensor_msgs/Image 话题",
    )
    # 调试图像开关允许通过 launch 命令行覆盖 YAML 默认值。
    debug_argument = DeclareLaunchArgument(
        "publish_debug_image",
        default_value="false",
        description="是否发布带 ArUco 标注的调试图像",
    )
    # 相机标定允许在启动时替换，默认值不依赖源码工作区位置。
    calibration_argument = DeclareLaunchArgument(
        "camera_calibration_path",
        default_value=default_calibration_file,
        description="ToF 或 Kalibr 鱼眼相机标定 YAML 路径",
    )
    # 夹爪估计节点启动动作。
    estimator_node = Node(
        package="fastumi_gripper_estimator",
        executable="gripper_openness_node",
        name="gripper_openness_estimator",
        output="screen",
        parameters=[
            parameter_file,
            {
                "image_topic": LaunchConfiguration("image_topic"),
                "publish_debug_image": LaunchConfiguration(
                    "publish_debug_image"
                ),
                "camera_calibration_path": LaunchConfiguration(
                    "camera_calibration_path"
                ),
            },
        ],
    )
    return LaunchDescription(
        [
            image_topic_argument,
            debug_argument,
            calibration_argument,
            estimator_node,
        ]
    )
