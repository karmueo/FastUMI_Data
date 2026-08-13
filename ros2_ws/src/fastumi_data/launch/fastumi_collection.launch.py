"""统一启动 FastUMI 设备数据节点，并可选自动录制 MCAP。"""

from datetime import datetime, timezone

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


# 统一采集入口使用的 ToF 原始 RGB 话题。
TOF_RGB_IMAGE_TOPIC = "/tof_stereo_camera/rgb/image_raw"


def generate_launch_description() -> LaunchDescription:
    """创建设备数据节点和可选 MCAP 录制器的统一启动描述。

    Returns:
        包含 ToF 相机、VIVE Tracker 和夹爪开合度估计的 ROS 2
        启动描述；启用录制时还包含 MCAP 录制进程。
    """
    # 子包的安装后共享目录，用于复用现有 launch 和配置。
    camera_share = FindPackageShare("tof_stereo_camera")
    tracker_share = FindPackageShare("vive_tracker")
    gripper_share = FindPackageShare("fastumi_gripper_estimator")

    # ToF 设备路径支持临时指定，留空时由驱动自动发现设备。
    device_path = LaunchConfiguration("device_path")

    # Tracker 和调试显示的可选覆盖。
    tracker_serial = LaunchConfiguration("tracker_serial")
    use_rviz = LaunchConfiguration("use_rviz")
    publish_debug_image = LaunchConfiguration("publish_debug_image")

    # MCAP 录制开关和数据集保存根目录。
    record_mcap = LaunchConfiguration("record_mcap")
    dataset_root = LaunchConfiguration("dataset_root")
    # 每次 launch 使用独立 UTC 时间戳目录，避免覆盖已有数据。
    recording_timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    mcap_output_dir = PathJoinSubstitution(
        [dataset_root, f"fastumi_{recording_timestamp}"]
    )

    # Tracker 子节点使用安装后的默认配置。
    tracker_config = PathJoinSubstitution(
        [tracker_share, "config", "vive_tracker.yaml"]
    )
    # 统一入口对外暴露的启动参数。
    launch_arguments = [
        DeclareLaunchArgument(
            "device_path",
            default_value="",
            description="可选 ToF 设备路径；留空时由驱动自动发现设备。",
        ),
        DeclareLaunchArgument(
            "tracker_serial",
            default_value="",
            description="可选 Tracker 序列号；留空时使用 vive_tracker 配置。",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            choices=["true", "false"],
            description="是否启动 Tracker RViz2 可视化。",
        ),
        DeclareLaunchArgument(
            "publish_debug_image",
            default_value="false",
            choices=["true", "false"],
            description="是否发布夹爪 ArUco 调试图像。",
        ),
        DeclareLaunchArgument(
            "record_mcap",
            default_value="false",
            choices=["true", "false"],
            description="是否随设备节点启动并立即录制 MCAP。",
        ),
        DeclareLaunchArgument(
            "dataset_root",
            default_value="dataset",
            description="MCAP 保存根目录；每次录制会创建时间戳子目录。",
        ),
    ]

    # 终端 1：复用 ToF 相机 launch，并避免与 Tracker 重复启动 RViz2。
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [camera_share, "launch", "tof_stereo_camera.launch.py"]
            )
        ),
        launch_arguments={
            "device_path": device_path,
            "enable_rviz": "false",
        }.items(),
    )
    # 终端 2：复用 VIVE Tracker launch，采集时默认不启动 RViz2。
    tracker_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [tracker_share, "launch", "vive_tracker.launch.py"]
            )
        ),
        launch_arguments={
            "serial": tracker_serial,
            "config_file": tracker_config,
            "use_rviz": use_rviz,
        }.items(),
    )
    # 终端 3：夹爪开合度估计订阅固定 ToF RGB 话题。
    gripper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [gripper_share, "launch", "gripper_openness.launch.py"]
            )
        ),
        launch_arguments={
            "image_topic": TOF_RGB_IMAGE_TOPIC,
            "publish_debug_image": publish_debug_image,
        }.items(),
    )
    # 可选录制进程使用 MCAP 原生快速 Zstd 块压缩并持续发现全部话题。
    mcap_recorder = ExecuteProcess(
        cmd=[
            "ros2",
            "bag",
            "record",
            "--storage",
            "mcap",
            "--storage-preset-profile",
            "zstd_fast",
            "--disable-keyboard-controls",
            "--output",
            mcap_output_dir,
            "--all-topics",
        ],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(record_mcap),
    )

    return LaunchDescription(
        [
            *launch_arguments,
            camera_launch,
            tracker_launch,
            gripper_launch,
            mcap_recorder,
        ]
    )
