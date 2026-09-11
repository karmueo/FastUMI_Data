"""启动 V4L2 双目相机 ROS 2 节点及 IMU 并加载统一双目标定。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


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

    # IMU 开关、主机轮询频率及原始轴向坐标系。
    imu_arguments = [
        DeclareLaunchArgument("enable_imu", default_value="true"),
        DeclareLaunchArgument("imu_poll_rate_hz", default_value="200.0"),
        DeclareLaunchArgument(
            "imu_frame_id", default_value="stereo_camera_imu_frame"
        ),
    ]

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
                "enable_imu": ParameterValue(
                    LaunchConfiguration("enable_imu"), value_type=bool
                ),
                "imu_poll_rate_hz": ParameterValue(
                    LaunchConfiguration("imu_poll_rate_hz"), value_type=float
                ),
                "imu_frame_id": ParameterValue(
                    LaunchConfiguration("imu_frame_id"), value_type=str
                ),
                "calibration_file": LaunchConfiguration("calibration_file"),
            },
        ],
    )

    return LaunchDescription(
        [
            device_argument,
            calibration_argument,
            *imu_arguments,
            camera_node,
        ]
    )
