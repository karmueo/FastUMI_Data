"""启动 V4L2 双目相机 ROS 2 节点并加载统一双目标定。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """创建支持设备路径和统一双目标定文件覆盖的启动描述。"""
    # 已安装 stereo_camera 包的共享资源目录。
    package_share = Path(get_package_share_directory("stereo_camera"))
    # 节点默认使用的统一双目标定文件路径。
    default_calibration = package_share / "config" / "calib.yaml"

    # 可从命令行覆盖的相机设备路径参数。
    device_argument = DeclareLaunchArgument(
        "device_path", default_value="/dev/video0"
    )
    # 可从命令行覆盖的统一双目标定文件路径参数。
    calibration_argument = DeclareLaunchArgument(
        "calibration_file", default_value=str(default_calibration)
    )

    # 负责采集拼接帧并发布左右目消息的驱动节点。
    camera_node = Node(
        package="stereo_camera",
        executable="stereo_camera_node",
        name="stereo_camera_node",
        namespace="stereo_camera",
        output="screen",
        parameters=[
            {
                "device_path": LaunchConfiguration("device_path"),
                "calibration_file": LaunchConfiguration("calibration_file"),
            },
        ],
    )

    return LaunchDescription(
        [
            device_argument,
            calibration_argument,
            camera_node,
        ]
    )
