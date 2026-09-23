"""Start direct UVC MJPEG capture and H.264 FFmpeg transport publishing."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    """Load the sender configuration from the package or an override path."""
    default_config = Path(get_package_share_directory("fastumi_usb_camera")) / \
        "config" / "ffmpeg.yaml"
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=str(default_config)),
        DeclareLaunchArgument("h264_encoder", default_value="hardware"),
        Node(
            package="fastumi_usb_camera",
            executable="usb_camera_ffmpeg",
            name="usb_camera_ffmpeg",
            output="screen",
            parameters=[LaunchConfiguration("config"), {
                "h264_encoder": ParameterValue(
                    LaunchConfiguration("h264_encoder"), value_type=str),
            }],
        ),
    ])
