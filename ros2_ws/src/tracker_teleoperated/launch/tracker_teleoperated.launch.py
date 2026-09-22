"""启动会话管理器和 RViz 面板，由管理器负责硬件、遥操及保存收尾。"""

import os
from pathlib import Path
import signal
import tempfile

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown, matches_action
from launch.events.process import SignalProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import yaml

from fastumi_usb_camera.capture import is_physical_video_device_path


def configured_rviz(template_path, config_path, manager_config_path=None):
    """从控制、记录及管理配置同步两路视频和里程计显示，返回配置字典。"""
    display = yaml.safe_load(Path(template_path).read_text(encoding="utf-8"))
    parameters = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    control = parameters["tracker_teleop"]["ros__parameters"]
    # UMI 使用本机原始图像，末端显示使用本机 H.264 解码输出。
    management = (yaml.safe_load(Path(manager_config_path).read_text(encoding="utf-8"))
                  if manager_config_path else {})
    umi = management.get('components', {}).get('umi_camera', {}).get('parameters', {})
    decoder = management.get('components', {}).get('wrist_decoder', {}).get('parameters', {})
    topics = {
        'umi_camera': '/' + umi.get('namespace', 'umi_camera').strip('/') + '/image_raw',
        'wrist_decoder': decoder.get('output_topic', '/wrist_camera/image_decoded'),
    }
    manager = display["Visualization Manager"]
    manager["Global Options"]["Fixed Frame"] = control.get("odom_frame", "vive_tracker_odom")
    for item in manager["Displays"]:
        if item["Class"] == "rviz_default_plugins/Image":
            item["Topic"]["Value"] = topics['umi_camera' if item['Name'] == 'UMI 视频' else 'wrist_decoder']
        if item["Class"] == "rviz_default_plugins/Odometry":
            item["Topic"]["Value"] = control.get("tracker_odom_topic", "/vive_tracker/odom")
    return display


def _launch(context):
    """创建管理器，按需生成 RViz 配置并接入退出信号转发。"""
    config = LaunchConfiguration("config_file").perform(context)
    share = FindPackageShare("tracker_teleoperated").perform(context)
    # 在自动启动前读取面板保存的设备；旧 Topic 字段不改变业务输入。
    rviz_config = LaunchConfiguration("rviz_config").perform(context)
    video_parameters = {}
    if rviz_config:
        saved = yaml.safe_load(Path(rviz_config).read_text(encoding="utf-8"))
        for panel in saved.get('Panels', []):
            if panel.get('Class') == 'tracker_teleoperated/TeleopPanel':
                for field, parameter in (('UmiVideoDevice', 'umi_video_device'),):
                    if is_physical_video_device_path(panel.get(field, '')):
                        video_parameters[parameter] = panel[field]
                break
    manager = Node(
        package="tracker_teleoperated", executable="tracker_component_manager",
        name="tracker_component_manager", output="screen", sigterm_timeout="320", sigkill_timeout="10",
        parameters=[{
            "config_file": config,
            "manager_config": LaunchConfiguration("manager_config").perform(context),
            "autostart": LaunchConfiguration("autostart").perform(context) == "true",
            "use_recorder": LaunchConfiguration("use_recorder").perform(context) == "true",
            **video_parameters,
        }],
    )
    actions = [RegisterEventHandler(OnProcessExit(
        target_action=manager,
        on_exit=[EmitEvent(event=Shutdown(reason="遥操会话管理器已退出"))],
    )), manager]
    if LaunchConfiguration("use_rviz").perform(context) == "true":
        rviz_config = LaunchConfiguration("rviz_config").perform(context)
        if not rviz_config:
            # 仅修改运行时副本，包内配置始终保持可重用。
            descriptor, rviz_config = tempfile.mkstemp(prefix="tracker-teleop-", suffix=".rviz")
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                yaml.safe_dump(configured_rviz(
                    Path(share) / "config/tracker_teleoperated.rviz", config,
                    LaunchConfiguration("manager_config").perform(context)), stream, allow_unicode=True)

            def cleanup(_context, path=rviz_config):
                """会话退出时删除临时显示配置。"""
                Path(path).unlink(missing_ok=True)
                return []

            actions.append(RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=cleanup)])))
        rviz = Node(package="rviz2", executable="rviz2", name="tracker_teleop_rviz",
                    arguments=["-d", rviz_config], output="screen")
        # 窗口退出时只通知管理器；等待管理器保存完成后才关闭整个 launch。
        actions.extend([RegisterEventHandler(OnProcessExit(
            target_action=rviz, on_exit=[EmitEvent(event=SignalProcess(
                signal_number=signal.SIGINT, process_matcher=matches_action(manager),
            ))],
        )), rviz])
    return actions


def generate_launch_description():
    """声明原有参数和面板管理参数，默认自动启动并显示 RViz。"""
    share = FindPackageShare("tracker_teleoperated")
    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value=PathJoinSubstitution([share, "config", "tracker_teleoperated.yaml"])),
        DeclareLaunchArgument("manager_config", default_value=PathJoinSubstitution([share, "config", "component_manager.yaml"])),
        DeclareLaunchArgument("use_recorder", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("autostart", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("use_rviz", default_value="true", choices=["true", "false"]),
        DeclareLaunchArgument("rviz_config", default_value=""),
        OpaqueFunction(function=_launch),
    ])
