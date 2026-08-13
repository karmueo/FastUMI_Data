"""启动 ToF 双目相机节点、可选 IMU 滤波器和 RViz 的入口。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    AndSubstitution,
    EqualsSubstitution,
    LaunchConfiguration,
)
from launch_ros.actions import Node


def generate_launch_description():
    """声明相机、SDK 日志与可选 IMU 滤波组件的启动描述。"""
    # 已安装包的共享目录，用于定位随包发布的 RViz 配置。
    package_share = Path(get_package_share_directory("tof_stereo_camera"))
    # RViz 启动时加载的预置显示配置。
    rviz_config = package_share / "rviz" / "tof_stereo_camera.rviz"

    # 相机节点、SDK 日志和可选可视化组件的启动参数。
    arguments = [
        DeclareLaunchArgument("enable_rgb", default_value="true"),
        DeclareLaunchArgument("enable_itof_depth", default_value="true"),
        DeclareLaunchArgument("enable_itof_gray", default_value="true"),
        DeclareLaunchArgument("enable_imu", default_value="true"),
        DeclareLaunchArgument("enable_imu_filter", default_value="true"),
        DeclareLaunchArgument(
            "imu_filter_type",
            default_value="madgwick",
            choices=["madgwick", "complementary"],
        ),
        DeclareLaunchArgument("enable_rviz", default_value="true"),
        # 2048 系完整复合帧对应 2048x1536 的 RGB 标定分辨率。
        DeclareLaunchArgument("width", default_value="2048"),
        DeclareLaunchArgument("height", default_value="2738"),
        DeclareLaunchArgument("enable_sdk_log", default_value="false"),
        DeclareLaunchArgument("sdk_log_path", default_value=""),
        DeclareLaunchArgument("pixel_format", default_value="YUYV"),
        DeclareLaunchArgument("device_path", default_value=""),
    ]

    # 发布既有 RGB、iTOF 和原始 IMU 话题的主节点。
    camera_node = Node(
        package="tof_stereo_camera",
        executable="tof_stereo_camera_node",
        name="tof_stereo_camera_node",
        namespace="tof_stereo_camera",
        output="screen",
        parameters=[{
            "enable_rgb": LaunchConfiguration("enable_rgb"),
            "enable_itof_depth": LaunchConfiguration("enable_itof_depth"),
            "enable_itof_gray": LaunchConfiguration("enable_itof_gray"),
            "enable_imu": LaunchConfiguration("enable_imu"),
            "width": LaunchConfiguration("width"),
            "height": LaunchConfiguration("height"),
            "pixel_format": LaunchConfiguration("pixel_format"),
            "device_path": LaunchConfiguration("device_path"),
            "enable_sdk_log": LaunchConfiguration("enable_sdk_log"),
            "sdk_log_path": LaunchConfiguration("sdk_log_path"),
        }],
    )

    # 仅在原始 IMU 与滤波器均启用时启动姿态滤波器。
    imu_filter_enabled = AndSubstitution(
        LaunchConfiguration("enable_imu"),
        LaunchConfiguration("enable_imu_filter"),
    )
    # 使滤波器订阅端与相机的 SensorDataQoS 可靠性匹配。
    imu_input_qos = {
        "qos_overrides./tof_stereo_camera/imu/data_raw.subscription.reliability":
            "best_effort",
    }
    # 可选的 Madgwick 姿态滤波节点。
    madgwick_filter_node = Node(
        package="imu_filter_madgwick",
        executable="imu_filter_madgwick_node",
        name="imu_filter_madgwick",
        namespace="tof_stereo_camera",
        output="screen",
        parameters=[{
            "stateless": False,
            "use_mag": False,
            "publish_tf": False,
            "world_frame": "enu",
            "gain": 0.1,
            "zeta": 0.0,
            **imu_input_qos,
        }],
        condition=IfCondition(AndSubstitution(
            imu_filter_enabled,
            EqualsSubstitution(
                LaunchConfiguration("imu_filter_type"),
                "madgwick",
            ),
        )),
    )
    # 可选的互补滤波姿态节点。
    complementary_filter_node = Node(
        package="imu_complementary_filter",
        executable="complementary_filter_node",
        name="imu_complementary_filter",
        namespace="tof_stereo_camera",
        output="screen",
        parameters=[{
            "use_mag": False,
            "do_bias_estimation": True,
            "do_adaptive_gain": True,
            "publish_tf": False,
            "constant_dt": 0.0,
            **imu_input_qos,
        }],
        condition=IfCondition(AndSubstitution(
            imu_filter_enabled,
            EqualsSubstitution(
                LaunchConfiguration("imu_filter_type"),
                "complementary",
            ),
        )),
    )
    # 可选的 RViz 可视化节点。
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="tof_stereo_camera_rviz",
        arguments=["-d", str(rviz_config)],
        condition=IfCondition(LaunchConfiguration("enable_rviz")),
    )
    return LaunchDescription(arguments + [
        camera_node,
        madgwick_filter_node,
        complementary_filter_node,
        rviz_node,
    ])
