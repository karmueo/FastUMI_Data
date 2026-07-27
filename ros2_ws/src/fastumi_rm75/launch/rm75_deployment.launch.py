"""启动 RM75 策略桥和通用平行夹爪适配器。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """创建支持配置文件覆盖的 RM75 部署启动描述。"""
    default_config = PathJoinSubstitution(
        [
            FindPackageShare("fastumi_rm75"),
            "config",
            "rm75_deployment.yaml",
        ]
    )
    config_argument = DeclareLaunchArgument(
        "config_file", default_value=default_config
    )
    arm_bridge = Node(
        package="fastumi_rm75",
        executable="rm75_policy_bridge",
        name="rm75_policy_bridge",
        output="screen",
        parameters=[LaunchConfiguration("config_file")],
    )
    gripper_bridge = Node(
        package="fastumi_rm75",
        executable="gripper_bridge",
        name="gripper_bridge",
        output="screen",
        parameters=[LaunchConfiguration("config_file")],
    )
    return LaunchDescription(
        [config_argument, arm_bridge, gripper_bridge]
    )
