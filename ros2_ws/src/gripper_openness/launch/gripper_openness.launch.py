"""启动夹爪开合度预测节点并加载相机、夹爪范围标定。"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """创建夹爪开合度预测 LaunchDescription。"""
    package_share = get_package_share_directory("gripper_openness")
    parameter_file = os.path.join(package_share, "config", "gripper_openness.yaml")
    user_gripper_calibration = Path.home() / "fastumi_gripper_calibration.yaml"
    default_gripper_calibration = (
        str(user_gripper_calibration)
        if user_gripper_calibration.is_file()
        else os.path.join(package_share, "config", "calibration.yaml")
    )
    arguments = [
        DeclareLaunchArgument(
            "image_topic", default_value="/umi_camera/image_raw", description="ROS 图像话题"
        ),
        DeclareLaunchArgument(
            "camera_calibration_path",
            default_value=os.path.join(package_share, "config", "calib.yaml"),
            description="相机标定 YAML",
        ),
        DeclareLaunchArgument(
            "gripper_calibration_path",
            default_value=default_gripper_calibration,
            description="夹爪范围标定 YAML",
        ),
        DeclareLaunchArgument("publish_debug_image", default_value="false", description="发布调试图像"),
        DeclareLaunchArgument(
            "roi_padding_pixels", default_value="50", description="标定 ROI 向外扩展像素"
        ),
        DeclareLaunchArgument("smoothing_alpha", default_value="1.0", description="指数平滑权重"),
    ]
    node = Node(
        package="gripper_openness",
        executable="gripper_openness_node",
        name="gripper_openness",
        output="screen",
        parameters=[
            parameter_file,
            {
                "image_topic": LaunchConfiguration("image_topic"),
                "camera_calibration_path": LaunchConfiguration("camera_calibration_path"),
                "gripper_calibration_path": LaunchConfiguration("gripper_calibration_path"),
                "publish_debug_image": LaunchConfiguration("publish_debug_image"),
                "roi_padding_pixels": LaunchConfiguration("roi_padding_pixels"),
                "smoothing_alpha": LaunchConfiguration("smoothing_alpha"),
            },
        ],
    )
    return LaunchDescription(arguments + [node])
