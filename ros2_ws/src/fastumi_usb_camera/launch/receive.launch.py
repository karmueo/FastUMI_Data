"""Start the FFmpeg transport decoder and publish a local raw BGR image."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Load the receiver configuration from the package or an override path."""
    default_config = Path(get_package_share_directory("fastumi_usb_camera")) / \
        "config" / "ffmpeg.yaml"
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=str(default_config)),
        DeclareLaunchArgument("input_topic", default_value="/usb_camera/image_raw"),
        DeclareLaunchArgument("output_topic", default_value="/usb_camera/image_decoded"),
        Node(
            package="fastumi_usb_camera",
            executable="usb_camera_receiver",
            name="usb_camera_receiver",
            output="screen",
            parameters=[LaunchConfiguration("config"), {
                "input_topic": LaunchConfiguration("input_topic"),
                "output_topic": LaunchConfiguration("output_topic"),
            }],
        ),
    ])
