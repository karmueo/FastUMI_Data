"""启动 Jetson 机械臂、夹爪、末端相机，回位成功后提供录制服务。"""

import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription,
    OpaqueFunction, RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.logging import get_logger
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from fastumi_usb_camera.capture import is_physical_video_device_path


INITIAL_JOINT_POSITIONS = [math.radians(value) for value in (0, 20, 0, 70, 0, 90, 90)]
CAMERA_MODES = {
    'raw': ('false', 'false', '/wrist_camera/image_raw', 'raw'),
    'jpeg': ('false', 'true', '/wrist_camera/image_raw/compressed', 'jpeg'),
    'h264': ('true', 'false', '/wrist_camera/image_raw/ffmpeg', 'ffmpeg'),
}


def _enabled(context, name):
    """解析严格布尔开关。"""
    value = LaunchConfiguration(name).perform(context).lower()
    if value not in ('true', 'false'):
        raise ValueError(f'{name} 只能是 true 或 false')
    return value == 'true'


def _include(package, launch_file, arguments=None):
    """创建已安装包的 launch 包含动作。"""
    source = Path(get_package_share_directory(package)) / 'launch' / launch_file
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(str(source)),
                                    launch_arguments=(arguments or {}).items())


def _after_home(event, context, recorder):
    """回位失败或超时以非零状态退出，成功才启动录制节点。"""
    if context.is_shutdown:
        return []
    if event.returncode != 0:
        reason = f'机械臂初始回位失败（退出码 {event.returncode}），统一启动退出'
        get_logger('fastumi_bringup').error(reason)
        return [EmitEvent(event=Shutdown(reason=reason))]
    return [recorder] if recorder is not None else []


def _launch_hardware(context):
    """先验证设备配置，再启动硬件；UMI 不属于此启动入口。"""
    mode = LaunchConfiguration('wrist_camera_mode').perform(context).strip().lower()
    if mode not in CAMERA_MODES:
        raise ValueError('wrist_camera_mode 只能是 raw、jpeg 或 h264')
    enable_ffmpeg, publish_compressed, default_topic, default_transport = CAMERA_MODES[mode]
    camera_enabled = _enabled(context, 'start_wrist_camera')
    if mode != 'h264' and _enabled(context, 'enable_decoder'):
        raise ValueError('enable_decoder=true 仅支持 wrist_camera_mode=h264')
    image_topic = LaunchConfiguration('image_topic').perform(context).strip()
    image_transport = LaunchConfiguration('image_transport').perform(context).strip().lower()
    if camera_enabled and ((image_topic and image_topic != default_topic) or
                           (image_transport and image_transport != default_transport)):
        raise ValueError('image_topic/image_transport 必须与 wrist_camera_mode 对应的相机输出一致')
    image_topic = image_topic or default_topic
    image_transport = image_transport or default_transport
    actions = []
    if camera_enabled:
        device = LaunchConfiguration('wrist_video_device').perform(context)
        if not is_physical_video_device_path(device) or not Path(device).exists():
            raise ValueError(f'末端相机必须指定存在的 /dev/v4l/by-path/*-video-index0: {device}')
        share = Path(get_package_share_directory('fastumi_usb_camera'))
        camera_arguments = {
            'config': str(share / 'config/usb_camera_1280_960.yaml'),
            'namespace': 'wrist_camera', 'video_device': device,
            'width': LaunchConfiguration('wrist_width'),
            'height': LaunchConfiguration('wrist_height'),
            'fps': LaunchConfiguration('camera_fps'),
            'frame_id': 'wrist_camera_optical_frame',
            'enable_ffmpeg': enable_ffmpeg,
            'publish_compressed': publish_compressed,
            'enable_decoder': LaunchConfiguration('enable_decoder'),
            'topic': '/wrist_camera/image_raw',
            'decoded_topic': '/wrist_camera/image_decoded',
        }
        if mode == 'h264':
            camera_arguments['h264_encoder'] = LaunchConfiguration('h264_encoder')
        actions.append(_include('fastumi_usb_camera', 'usb_camera.launch.py', camera_arguments))
    if _enabled(context, 'start_gripper'):
        arguments = {'network_interface': LaunchConfiguration('gripper_network_interface')}
        config = LaunchConfiguration('gripper_config_file').perform(context)
        if config:
            arguments['config_file'] = str(Path(config).expanduser().resolve())
        actions.append(_include('unitree_gripper', 'gripper.launch.py', arguments))
    recorder = None
    if _enabled(context, 'start_recorder'):
        recorder_arguments = {
            name: LaunchConfiguration(name) for name in
            ('dataset_root', 'dir_name', 'name', 'camera_fps', 'record_camera',
             'shutdown_save_timeout')
        }
        recorder_arguments.update(image_topic=image_topic, image_transport=image_transport)
        recorder = _include('fastumi_recorder', 'recorder.launch.py', recorder_arguments)
    if _enabled(context, 'start_arm'):
        actions.append(_include('rm_driver', 'rm_75_driver.launch.py'))
        if _enabled(context, 'move_to_initial_pose'):
            home = Node(package='rm_bringup', executable='rm_75_initial_pose',
                        output='screen', parameters=[{'initial_joint_positions': INITIAL_JOINT_POSITIONS}])
            actions.extend([
                RegisterEventHandler(OnProcessExit(
                    target_action=home,
                    on_exit=lambda event, ctx: _after_home(event, ctx, recorder))),
                home,
            ])
            return actions
    if recorder is not None:
        actions.append(recorder)
    return actions


def generate_launch_description():
    """默认启动四个组件并执行一次初始回位，提供维护开关。"""
    gripper_default_config = str(
        Path(get_package_share_directory('unitree_gripper')) / 'config/gripper.yaml'
    )
    defaults = dict(
        start_arm='true', start_gripper='true', start_wrist_camera='true', start_recorder='true',
        move_to_initial_pose='true', enable_decoder='false', h264_encoder='hardware',
        wrist_camera_mode='jpeg',
        wrist_video_device='', wrist_width='1280',
        wrist_height='960', camera_fps='30', gripper_config_file=gripper_default_config,
        gripper_network_interface='',
        dataset_root='', dir_name='test', name='default_test', record_camera='true',
        image_topic='', image_transport='',
        shutdown_save_timeout='120.0',
    )
    return LaunchDescription([
        *[DeclareLaunchArgument(key, default_value=value) for key, value in defaults.items()],
        OpaqueFunction(function=_launch_hardware),
    ])
