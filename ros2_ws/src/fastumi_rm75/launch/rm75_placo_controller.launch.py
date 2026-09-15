"""启动单个 RM75 Placo 关节控制节点。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """创建支持 YAML 覆盖的单节点启动描述。"""
    default_config = PathJoinSubstitution(
        [FindPackageShare("fastumi_rm75"), "config", "rm75_placo_controller.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("python_executable", default_value="python3"),
            ExecuteProcess(
                cmd=[
                    LaunchConfiguration("python_executable"),
                    "-m",
                    "fastumi_rm75.rm75_placo_controller",
                    "--ros-args",
                    "--params-file",
                    LaunchConfiguration("config_file"),
                ],
                output="screen",
            ),
        ]
    )
