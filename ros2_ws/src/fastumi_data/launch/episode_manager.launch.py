"""启动 FastUMI episode 管理节点并允许覆盖任务和会话标识。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """创建 episode 管理节点的启动描述。"""
    task_argument = DeclareLaunchArgument("task_name", default_value="test")
    session_argument = DeclareLaunchArgument("session_id", default_value="")
    manager = Node(
        package="fastumi_data",
        executable="episode_manager",
        name="episode_manager",
        output="screen",
        parameters=[
            {
                "task_name": LaunchConfiguration("task_name"),
                "session_id": LaunchConfiguration("session_id"),
            }
        ],
    )
    return LaunchDescription([task_argument, session_argument, manager])
