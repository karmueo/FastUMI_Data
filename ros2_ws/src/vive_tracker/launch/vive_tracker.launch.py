"""同时启动 VIVE Tracker 位姿发布节点与 RViz2."""

import functools
import signal

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnShutdown
from launch.events import matches_action
from launch.events.process import SignalProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _build_tracker_parameters(context):
    """根据 launch 上下文构造 Tracker 节点参数列表."""
    # 启动时可替换的节点参数文件。
    config_file = LaunchConfiguration('config_file')
    # 非空值表示调用者明确要求覆盖参数文件中的序列号。
    serial_override = LaunchConfiguration('serial').perform(context)
    # 非空值表示调用者明确要求覆盖参数文件中的坐标重排开关。
    reorder_override = LaunchConfiguration(
        'reorder_pose_axes', default=''
    ).perform(context)
    # 非空值表示调用者明确要求覆盖参数文件中的采样频率。
    publish_rate_override = LaunchConfiguration(
        'publish_rate_hz', default=''
    ).perform(context)
    # 非空值表示调用者明确要求覆盖参数文件中的 Path 更新频率。
    path_publish_rate_override = LaunchConfiguration(
        'path_publish_rate_hz', default=''
    ).perform(context)

    # 节点参数默认完全沿用参数文件。
    tracker_parameters = [config_file]
    if serial_override:
        tracker_parameters.append({'serial': serial_override})
    if reorder_override:
        tracker_parameters.append(
            {'reorder_pose_axes': reorder_override == 'true'}
        )
    if publish_rate_override:
        tracker_parameters.append(
            {'publish_rate_hz': float(publish_rate_override)}
        )
    if path_publish_rate_override:
        tracker_parameters.append(
            {'path_publish_rate_hz': float(path_publish_rate_override)}
        )
    return tracker_parameters


def _forward_terminal_interrupt(event, context, tracker_node):
    """向独立会话中的 Tracker 节点转发一次终端中断信号."""
    if not event.due_to_sigint or context.noninteractive:
        return []
    # 只匹配当前 Tracker 节点的中断事件。
    tracker_interrupt = SignalProcess(
        signal_number=signal.SIGINT,
        process_matcher=matches_action(tracker_node),
    )
    return [EmitEvent(event=tracker_interrupt)]


def _create_tracker_node(context):
    """按需应用序列号和频率覆盖并创建 Tracker 节点."""
    # 根据调用者是否显式传入覆盖值构造节点参数。
    tracker_parameters = _build_tracker_parameters(context)

    # 在固定命名空间中运行的 Tracker 位姿发布节点。
    tracker_node = Node(
        package='vive_tracker',
        executable='vive_tracker_node',
        namespace='vive_tracker',
        name='pose_publisher',
        output='screen',
        prefix='setsid',
        parameters=tracker_parameters,
    )
    # 终端中断时主动通知已脱离原终端进程组的 Tracker 节点。
    terminal_interrupt_handler = RegisterEventHandler(
        OnShutdown(
            on_shutdown=functools.partial(
                _forward_terminal_interrupt,
                tracker_node=tracker_node,
            )
        )
    )
    return [tracker_node, terminal_interrupt_handler]


def generate_launch_description():
    """创建支持参数覆盖的 VIVE Tracker 可视化启动描述."""
    # 已安装 vive_tracker 包的共享资源目录。
    package_share = FindPackageShare('vive_tracker')
    # 包内默认节点参数文件。
    default_config = PathJoinSubstitution(
        [package_share, 'config', 'vive_tracker.yaml']
    )
    # 包内默认 RViz2 显示配置。
    default_rviz_config = PathJoinSubstitution(
        [package_share, 'config', 'vive_tracker.rviz']
    )

    # 启动时可替换的 RViz2 配置文件。
    rviz_config = LaunchConfiguration('rviz_config')
    # 控制是否启动 RViz2 的布尔参数。
    use_rviz = LaunchConfiguration('use_rviz')

    # 目标 Tracker 的序列号声明。
    serial_argument = DeclareLaunchArgument(
        'serial',
        default_value='',
        description=(
            'Target VIVE Tracker serial number. Leave empty to use the value '
            'from config_file.'
        ),
    )
    # Tracker 位姿全局坐标轴重排开关声明。
    reorder_pose_axes_argument = DeclareLaunchArgument(
        'reorder_pose_axes',
        default_value='',
        choices=['', 'true', 'false'],
        description=(
            'Whether to reorder OpenVR pose axes. Leave empty to use the '
            'value from config_file.'
        ),
    )
    # 采样频率覆盖参数声明。
    publish_rate_argument = DeclareLaunchArgument(
        'publish_rate_hz',
        default_value='',
        description=(
            'Tracker sample rate in Hz. Leave empty to use the value from '
            'config_file.'
        ),
    )
    # Path 更新频率覆盖参数声明。
    path_publish_rate_argument = DeclareLaunchArgument(
        'path_publish_rate_hz',
        default_value='',
        description=(
            'Path update rate in Hz. Leave empty to use the value from '
            'config_file.'
        ),
    )
    # 节点参数文件路径声明。
    config_file_argument = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to the vive_tracker ROS parameter file.',
    )
    # RViz2 配置文件路径声明。
    rviz_config_argument = DeclareLaunchArgument(
        'rviz_config',
        default_value=default_rviz_config,
        description='Path to the RViz2 configuration file.',
    )
    # RViz2 启用开关声明。
    use_rviz_argument = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to start RViz2.',
    )

    # 参数声明完成后，根据调用者是否传入覆盖值创建节点。
    tracker_node_action = OpaqueFunction(function=_create_tracker_node)
    # 使用预配置显示项启动的 RViz2 进程。
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='vive_tracker_rviz',
        output='screen',
        arguments=['-d', rviz_config],
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        [
            serial_argument,
            reorder_pose_axes_argument,
            publish_rate_argument,
            path_publish_rate_argument,
            config_file_argument,
            rviz_config_argument,
            use_rviz_argument,
            tracker_node_action,
            rviz_node,
        ]
    )
