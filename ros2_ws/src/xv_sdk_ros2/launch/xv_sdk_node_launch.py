"""启动 XV SDK ROS 2 节点并配置 SDK 动态库搜索路径."""

from datetime import datetime
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """生成 XV SDK ROS 2 节点的 launch 描述."""
    # 普通 RGB 图像发布开关，默认启用原始 RGB8。
    rgb_enable = LaunchConfiguration('rgb_enable', default='true')
    # ToF 深度图像和原始 IR 强度图发布开关。
    tof_enable = LaunchConfiguration('tof_enable', default='false')
    # Kalibr RGB 鱼眼校正输出开关，默认关闭。
    rgb_fisheye_undistort_enable = LaunchConfiguration(
        'rgb_fisheye_undistort_enable', default='false')
    # Kalibr RGB 鱼眼标定文件路径。
    default_rgb_fisheye_calibration_path = PathJoinSubstitution([
        FindPackageShare('xv_sdk_ros2'),
        'config',
        'kalibr_data-camchain-imucam.yaml',
    ])
    rgb_fisheye_calibration_path = LaunchConfiguration(
        'rgb_fisheye_calibration_path',
        default=default_rgb_fisheye_calibration_path,
    )
    # bag 录制开关，打开后随 launch 启动 ros2 bag record。
    record_bag = LaunchConfiguration('record_bag', default='false')
    # 默认 bag 输出目录，使用时间戳避免覆盖已有录制结果。
    bag_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_bag_output_dir = str(
        Path('/home/scl/datasets/ros2bag') / f'xv_sdk_ros2_{bag_timestamp}')
    # bag 输出目录，可通过 bag_output_dir:=... 覆盖。
    bag_output_dir = LaunchConfiguration('bag_output_dir', default=default_bag_output_dir)
    # 一次性截图节点开关，默认关闭。
    snapshot_enable = LaunchConfiguration('snapshot_enable', default='false')
    # 一次性截图订阅的完整图像话题名称。
    snapshot_topic = LaunchConfiguration('snapshot_topic', default='')
    # 一次性截图输出目录。
    snapshot_output_dir = LaunchConfiguration(
        'snapshot_output_dir', default='/tmp/xv_sdk_snapshots'
    )
    # 一次性截图文件名；为空时由节点按接收时间生成。
    snapshot_filename = LaunchConfiguration('snapshot_filename', default='')
    # 一次性截图等待首帧的超时秒数。
    snapshot_timeout_sec = LaunchConfiguration('snapshot_timeout_sec', default='30.0')

    return LaunchDescription([
        DeclareLaunchArgument(
            'record_bag',
            default_value='false',
            description='是否随 xv_cameras 节点启动 ros2 bag record -a。',
        ),
        DeclareLaunchArgument(
            'bag_output_dir',
            default_value=default_bag_output_dir,
            description='ros2 bag 输出目录；record_bag 为 true 时生效。',
        ),
        DeclareLaunchArgument(
            'snapshot_enable',
            default_value='false',
            description='是否随设备节点启动一次性截图节点。',
        ),
        DeclareLaunchArgument(
            'snapshot_topic',
            default_value='',
            description='一次性截图订阅的完整 sensor_msgs/msg/Image 话题。',
        ),
        DeclareLaunchArgument(
            'snapshot_output_dir',
            default_value='/tmp/xv_sdk_snapshots',
            description='一次性截图的输出目录。',
        ),
        DeclareLaunchArgument(
            'snapshot_filename',
            default_value='',
            description='一次性截图文件名；留空时自动生成。',
        ),
        DeclareLaunchArgument(
            'snapshot_timeout_sec',
            default_value='30.0',
            description='等待首个图像帧的超时秒数；0 表示一直等待。',
        ),
        DeclareLaunchArgument(
            'rgb_fisheye_undistort_enable',
            default_value='false',
            description='是否发布基于 Kalibr equidistant 标定校正后的 RGB8 图像和相机内参。',
        ),
        DeclareLaunchArgument(
            'rgb_fisheye_calibration_path',
            default_value=default_rgb_fisheye_calibration_path,
            description='Kalibr camchain YAML 路径，用于 RGB fisheye 校正。',
        ),
        DeclareLaunchArgument(
            'rgb_enable',
            default_value='true',
            description='是否发布普通 RGB8 图像和相机内参。',
        ),
        DeclareLaunchArgument(
            'tof_enable',
            default_value='false',
            description='是否发布 ToF 深度图、原始 mono16 IR 强度图和相机内参。',
        ),
        SetEnvironmentVariable(
            name='LD_LIBRARY_PATH',
            value=['/usr/lib:', EnvironmentVariable('LD_LIBRARY_PATH', default_value='')],
        ),
        Node(
            package='xv_sdk_ros2',
            executable='xv_cameras',
            name='xv_cameras',
            output='screen',
            emulate_tty=True,
            arguments=['--ros-args', '--disable-external-lib-logs'],
            parameters=[
                {'imu_optical_frame': 'xv_sdk/imu_optical_frame'},
                {'rgb_optical_frame': 'rgb_optical_frame'},
                {'tof_optical_frame': 'tof_optical_frame'},
                {'rgb_enable': rgb_enable},
                {'tof_enable': tof_enable},
                {'rgb_fisheye_undistort_enable': rgb_fisheye_undistort_enable},
                {'rgb_fisheye_calibration_path': rgb_fisheye_calibration_path},
            ]
        ),
        Node(
            package='xv_sdk_ros2',
            executable='image_snapshot',
            name='image_snapshot',
            output='screen',
            emulate_tty=True,
            condition=IfCondition(snapshot_enable),
            parameters=[
                {'image_topic': snapshot_topic},
                {'output_dir': snapshot_output_dir},
                {'filename': snapshot_filename},
                {'timeout_sec': snapshot_timeout_sec},
            ],
        ),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'record', '-a', '-o', bag_output_dir],
            output='screen',
            emulate_tty=True,
            condition=IfCondition(record_bag),
        )
    ])
