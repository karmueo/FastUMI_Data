"""一键启动外参标定使用的鱼眼相机、VIVE Tracker 和 RViz2。"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EqualsSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """创建可在 ToF 与 XV 相机之间选择的外参采集启动描述。

    Returns:
        包含一路鱼眼相机、VIVE Tracker 和默认 RViz2 可视化的启动描述。
    """
    # 相机类型决定启用 ToF 或 XV 子 launch，同一时刻只启动一路相机。
    camera_type = LaunchConfiguration("camera_type")
    # ToF 相机的可选 UVC 设备路径，留空时由驱动自动发现。
    device_path = LaunchConfiguration("device_path")
    # ToF 相机码流档位，主码流与子码流分别对应不同 RGB 分辨率。
    stream_profile = LaunchConfiguration("stream_profile")
    # ToF 传感器话题的 DDS 可靠性，可靠模式用于避免录包传输层丢帧。
    sensor_qos_reliability = LaunchConfiguration(
        "sensor_qos_reliability"
    )
    # XV 原始鱼眼图像的可选去畸变输出开关。
    rgb_fisheye_undistort_enable = LaunchConfiguration(
        "rgb_fisheye_undistort_enable"
    )
    # Tracker 序列号覆盖值，留空时沿用参数文件配置。
    tracker_serial = LaunchConfiguration("tracker_serial")
    # RViz2 启用开关，默认由 Tracker 子 launch 启动一份实例。
    use_rviz = LaunchConfiguration("use_rviz")

    # 外参采集入口公开的常用参数声明。
    launch_arguments = [
        DeclareLaunchArgument(
            "camera_type",
            default_value="tof",
            choices=["tof", "xv"],
            description="鱼眼相机类型：tof 或 xv。",
        ),
        DeclareLaunchArgument(
            "device_path",
            default_value="",
            description="可选 ToF UVC 设备路径；留空时自动发现。",
        ),
        DeclareLaunchArgument(
            "stream_profile",
            default_value="main",
            choices=["main", "sub"],
            description=(
                "ToF 码流档位：main 为 2048x1536 RGB，"
                "sub 为 1920x1080 RGB。"
            ),
        ),
        DeclareLaunchArgument(
            "sensor_qos_reliability",
            default_value="reliable",
            choices=["best_effort", "reliable"],
            description="ToF 传感器话题的 DDS 可靠性。",
        ),
        DeclareLaunchArgument(
            "rgb_fisheye_undistort_enable",
            default_value="false",
            choices=["true", "false"],
            description="是否启用 XV RGB 鱼眼去畸变输出。",
        ),
        DeclareLaunchArgument(
            "tracker_serial",
            default_value="",
            description="可选 Tracker 序列号；留空时使用参数文件配置。",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            choices=["true", "false"],
            description="是否启动 Tracker RViz2 可视化。",
        ),
    ]

    # ToF 相机包的安装后共享目录。
    tof_camera_share = FindPackageShare("tof_stereo_camera")
    # XV 相机包的安装后共享目录。
    xv_camera_share = FindPackageShare("xv_sdk_ros2")
    # VIVE Tracker 包的安装后共享目录。
    tracker_share = FindPackageShare("vive_tracker")

    # ToF 标定采集只启用 RGB，并关闭深度、灰度、IMU 和相机自带 RViz2。
    tof_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [tof_camera_share, "launch", "tof_stereo_camera.launch.py"]
            )
        ),
        launch_arguments={
            "device_path": device_path,
            "stream_profile": stream_profile,
            "sensor_qos_reliability": sensor_qos_reliability,
            "enable_rgb": "true",
            "enable_itof_depth": "false",
            "enable_itof_gray": "false",
            "enable_imu": "false",
            "enable_imu_filter": "false",
            "enable_rviz": "false",
        }.items(),
        condition=IfCondition(EqualsSubstitution(camera_type, "tof")),
    )
    # XV 标定采集启用 RGB、关闭 ToF 和驱动自带录包功能。
    xv_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [xv_camera_share, "launch", "xv_sdk_node_launch.py"]
            )
        ),
        launch_arguments={
            "record_bag": "false",
            "rgb_enable": "true",
            "tof_enable": "false",
            "rgb_fisheye_undistort_enable": (
                rgb_fisheye_undistort_enable
            ),
        }.items(),
        condition=IfCondition(EqualsSubstitution(camera_type, "xv")),
    )
    # Tracker 子 launch 同时提供唯一的 RViz2 实例，默认开启。
    tracker_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [tracker_share, "launch", "vive_tracker.launch.py"]
            )
        ),
        launch_arguments={
            "serial": tracker_serial,
            "use_rviz": use_rviz,
        }.items(),
    )

    return LaunchDescription(
        [
            *launch_arguments,
            tof_camera_launch,
            xv_camera_launch,
            tracker_launch,
        ]
    )
