"""启动夹爪范围标定节点并映射常用输入参数。"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """创建夹爪范围标定 LaunchDescription。"""
    package_share = get_package_share_directory("gripper_openness")
    parameter_file = os.path.join(package_share, "config", "gripper_calibration.yaml")
    # 相机内参来自包目录；生成的夹爪标定写入用户可写目录。
    camera_calibration_file = os.path.join(package_share, "config", "calib.yaml")
    output_file = str(Path.home() / "fastumi_gripper_calibration.yaml")
    arguments = [
        DeclareLaunchArgument("source_mode", default_value="topic", description="topic 或 video"),
        DeclareLaunchArgument(
            "image_topic", default_value="/usb_camera/image_raw", description="ROS 图像话题"
        ),
        DeclareLaunchArgument("video_path", default_value="", description="本地视频路径"),
        DeclareLaunchArgument(
            "camera_calibration_path", default_value=camera_calibration_file,
            description="相机标定 YAML",
        ),
        DeclareLaunchArgument(
            "output_path", default_value=output_file,
            description="夹爪标定输出 YAML",
        ),
        DeclareLaunchArgument("overwrite", default_value="false", description="是否覆盖已有输出"),
        DeclareLaunchArgument("publish_debug_image", default_value="false", description="发布调试图像"),
    ]
    node = Node(
        package="gripper_openness",
        executable="gripper_calibration_node",
        name="gripper_calibration",
        output="screen",
        parameters=[
            parameter_file,
            {
                "source_mode": LaunchConfiguration("source_mode"),
                "image_topic": LaunchConfiguration("image_topic"),
                "video_path": LaunchConfiguration("video_path"),
                "camera_calibration_path": LaunchConfiguration("camera_calibration_path"),
                "output_path": LaunchConfiguration("output_path"),
                "overwrite": LaunchConfiguration("overwrite"),
                "publish_debug_image": LaunchConfiguration("publish_debug_image"),
            },
        ],
    )
    return LaunchDescription(arguments + [node])
