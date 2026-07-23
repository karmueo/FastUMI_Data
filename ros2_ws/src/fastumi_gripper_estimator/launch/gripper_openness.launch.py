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
    # 安装后的包共享目录。
    package_share = get_package_share_directory("fastumi_gripper_estimator")
    # 默认节点参数文件路径。
    parameter_file = os.path.join(
        package_share, "config", "gripper_openness.yaml"
    )
    # 输入话题参数默认选择与标定数据一致的畸变校正图像。
    image_topic_argument = DeclareLaunchArgument(
        "image_topic",
        default_value=(
            "/xv_sdk/SN250801DR48FB26001253/rgb_fisheye_undistorted/image"
        ),
        description="用于夹爪开合度估计的 sensor_msgs/Image 话题",
    )
    # 调试图像开关允许通过 launch 命令行覆盖 YAML 默认值。
    debug_argument = DeclareLaunchArgument(
        "publish_debug_image",
        default_value="false",
        description="是否发布带 ArUco 标注的调试图像",
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
                "publish_debug_image": LaunchConfiguration("publish_debug_image"),
            },
        ],
    )
    return LaunchDescription(
        [
            image_topic_argument,
            debug_argument,
            estimator_node,
        ]
    )
