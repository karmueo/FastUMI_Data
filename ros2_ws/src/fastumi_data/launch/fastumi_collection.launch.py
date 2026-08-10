"""统一启动 FastUMI 相机、Tracker 和夹爪开合度估计。"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


# 默认 XV 相机序列号，与采集和处理配置保持一致。
DEFAULT_CAMERA_SERIAL = "SN250801DR48FB26001253"


def generate_launch_description() -> LaunchDescription:
    """创建三个设备数据节点的统一启动描述。

    Returns:
        包含 XV 相机、VIVE Tracker 和夹爪开合度估计的 ROS 2
        启动描述。
    """
    # 子包的安装后共享目录，用于复用现有 launch 和配置。
    camera_share = FindPackageShare("xv_sdk_ros2")
    tracker_share = FindPackageShare("vive_tracker")
    gripper_share = FindPackageShare("fastumi_gripper_estimator")

    # 相机序列号决定夹爪估计订阅的图像话题。
    camera_serial = LaunchConfiguration("camera_serial")
    image_topic = ["/xv_sdk/", camera_serial, "/rgb/image"]

    # Tracker 和调试显示的可选覆盖。
    tracker_serial = LaunchConfiguration("tracker_serial")
    use_rviz = LaunchConfiguration("use_rviz")
    publish_debug_image = LaunchConfiguration("publish_debug_image")

    # Tracker 子节点使用安装后的默认配置。
    tracker_config = PathJoinSubstitution(
        [tracker_share, "config", "vive_tracker.yaml"]
    )
    # 统一入口对外暴露的启动参数。
    launch_arguments = [
        DeclareLaunchArgument(
            "camera_serial",
            default_value=DEFAULT_CAMERA_SERIAL,
            description="XV 相机序列号，用于构造 RGB 图像话题。",
        ),
        DeclareLaunchArgument(
            "tracker_serial",
            default_value="",
            description="可选 Tracker 序列号；留空时使用 vive_tracker 配置。",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            choices=["true", "false"],
            description="是否启动 Tracker RViz2 可视化。",
        ),
        DeclareLaunchArgument(
            "publish_debug_image",
            default_value="false",
            choices=["true", "false"],
            description="是否发布夹爪 ArUco 调试图像。",
        ),
    ]

    # 终端 1：复用 XV SDK 相机 launch。
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [camera_share, "launch", "xv_sdk_node_launch.py"]
            )
        ),
        launch_arguments={"record_bag": "false"}.items(),
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
    # 终端 3：夹爪开合度估计共享同一个相机图像话题。
    gripper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [gripper_share, "launch", "gripper_openness.launch.py"]
            )
        ),
        launch_arguments={
            "image_topic": image_topic,
            "publish_debug_image": publish_debug_image,
        }.items(),
    )

    return LaunchDescription(
        [
            *launch_arguments,
            camera_launch,
            tracker_launch,
            gripper_launch,
        ]
    )
