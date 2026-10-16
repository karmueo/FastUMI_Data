"""启动夹爪范围标定节点并映射常用输入参数。"""

import os

from ament_index_python.packages import get_package_share_directory
from gripper_openness.paths import default_gripper_calibration_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """创建夹爪范围标定 LaunchDescription。"""
    package_share = get_package_share_directory("gripper_openness")
    parameter_file = os.path.join(package_share, "config", "gripper_calibration_params.yaml")
    # 内参由节点选择，夹爪标定默认写回本包 config/gripper_calibration.yaml。
    output_file = default_gripper_calibration_path()
    arguments = [
        DeclareLaunchArgument(
            "source_mode", default_value="topic", description="topic、video 或 mcap"
        ),
        DeclareLaunchArgument(
            "image_topic", default_value="/umi_camera/image_raw", description="ROS 图像话题"
        ),
        DeclareLaunchArgument("video_path", default_value="", description="本地视频路径"),
        DeclareLaunchArgument("mcap_path", default_value="", description="MCAP 文件、bag 目录或录制目录"),
        DeclareLaunchArgument(
            "mcap_image_topic", default_value="", description="留空时自动选择录包图像话题"
        ),
        DeclareLaunchArgument(
            "roi_ratios", default_value="[0.0, 0.0, 1.0, 1.0]",
            description="检测区域比例 [左, 上, 右, 下]",
        ),
        DeclareLaunchArgument(
            "camera_calibration_path", default_value="",
            description="相机标定 YAML，留空优先读取 MCAP 录制快照，否则读取包内默认文件",
        ),
        DeclareLaunchArgument(
            "output_path", default_value=output_file,
            description="夹爪标定输出 YAML",
        ),
        DeclareLaunchArgument("overwrite", default_value="true", description="是否覆盖已有输出"),
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
                "mcap_path": LaunchConfiguration("mcap_path"),
                "mcap_image_topic": LaunchConfiguration("mcap_image_topic"),
                "roi_ratios": LaunchConfiguration("roi_ratios"),
                "camera_calibration_path": LaunchConfiguration("camera_calibration_path"),
                "output_path": LaunchConfiguration("output_path"),
                "overwrite": LaunchConfiguration("overwrite"),
                "publish_debug_image": LaunchConfiguration("publish_debug_image"),
            },
        ],
    )
    return LaunchDescription(arguments + [node])
