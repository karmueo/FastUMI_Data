"""启动 RM75 Tracker 遥操控制节点和独立终端中的键盘节点。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """创建支持配置覆盖和可选键盘终端的启动描述。"""
    # 包内默认遥操参数文件。
    default_config = PathJoinSubstitution(
        [
            FindPackageShare("tracker_teleoperated"),
            "config",
            "tracker_teleoperated.yaml",
        ]
    )
    # 调用者可替换的配置文件路径。
    config_file = LaunchConfiguration("config_file")
    # 是否由 launch 创建独立键盘终端。
    use_keyboard = LaunchConfiguration("use_keyboard")
    # 创建交互终端所用的命令前缀。
    keyboard_prefix = LaunchConfiguration("keyboard_prefix")

    config_argument = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description="Tracker 遥操 ROS 参数文件路径。",
    )
    keyboard_argument = DeclareLaunchArgument(
        "use_keyboard",
        default_value="true",
        description="是否在独立终端中启动键盘启停节点。",
    )
    keyboard_prefix_argument = DeclareLaunchArgument(
        "keyboard_prefix",
        default_value="xterm -fa Monospace -fs 20 -e",
        description="启动交互式键盘节点的终端命令前缀，默认使用 20 号等宽字体。",
    )
    control_node = Node(
        package="tracker_teleoperated",
        executable="tracker_teleop_node",
        name="tracker_teleop",
        output="screen",
        parameters=[config_file],
    )
    keyboard_node = Node(
        package="tracker_teleoperated",
        executable="tracker_teleop_keyboard",
        name="tracker_teleop_keyboard",
        output="screen",
        emulate_tty=True,
        prefix=keyboard_prefix,
        condition=IfCondition(use_keyboard),
    )
    # 键盘退出前会请求暂停；随后关闭整个 launch，避免控制节点单独残留。
    keyboard_exit_handler = RegisterEventHandler(
        OnProcessExit(
            target_action=keyboard_node,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="Tracker 遥操键盘节点已退出")
                )
            ],
        )
    )
    return LaunchDescription(
        [
            config_argument,
            keyboard_argument,
            keyboard_prefix_argument,
            keyboard_exit_handler,
            control_node,
            keyboard_node,
        ]
    )
