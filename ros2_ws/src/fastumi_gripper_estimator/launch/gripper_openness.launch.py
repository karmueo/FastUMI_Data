"""启动 FastUMI 夹爪开合度估计节点并加载默认参数。"""

import os
from pathlib import Path

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
    # 夹爪标定包安装后的共享目录，供估计节点复用同一组标定文件。
    calibration_share = get_package_share_directory("gripper_openness")
    # 默认节点参数文件路径。
    parameter_file = os.path.join(
        package_share, "config", "gripper_openness.yaml"
    )
    # 夹爪标定节点默认读取的相机标定文件路径。
    default_calibration_file = os.path.join(
        calibration_share, "config", "calib.yaml"
    )
    # 优先使用用户新生成的标定，首次运行时保留包内示例标定。
    user_gripper_calibration = Path.home() / "fastumi_gripper_calibration.yaml"
    default_gripper_calibration_file = (
        str(user_gripper_calibration)
        if user_gripper_calibration.is_file()
        else os.path.join(calibration_share, "config", "calibration.yaml")
    )
    # 默认订阅与相机标定对应的 USB 原始图像。
    image_topic_argument = DeclareLaunchArgument(
        "image_topic",
        default_value="/usb_camera/image_raw",
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
        description="相机标定 YAML 路径",
    )
    # 夹爪范围标定允许在启动时指定其他文件。
    gripper_calibration_argument = DeclareLaunchArgument(
        "gripper_calibration_path",
        default_value=default_gripper_calibration_file,
        description="夹爪范围标定 YAML 路径",
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
                "gripper_calibration_path": LaunchConfiguration(
                    "gripper_calibration_path"
                ),
            },
        ],
    )
    return LaunchDescription(
        [
            image_topic_argument,
            debug_argument,
            calibration_argument,
            gripper_calibration_argument,
            estimator_node,
        ]
    )
