"""启动 USB 单目相机，并允许使用参数文件或启动参数覆盖默认配置。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _camera_node(context):
    """在启动时仅应用用户明确提供的参数覆盖。"""
    # 参数文件可替换整套默认值，非空启动参数覆盖该文件中的单项值。
    config = LaunchConfiguration("config").perform(context)
    overrides = {}
    for name in ("vendor_id", "product_id", "width", "height", "fps"):
        value = LaunchConfiguration(name).perform(context)
        if value:
            overrides[name] = int(value, 0)
    frame_id = LaunchConfiguration("frame_id").perform(context)
    if frame_id:
        overrides["frame_id"] = frame_id
    compressed = LaunchConfiguration("publish_compressed").perform(context)
    if compressed:
        if compressed.lower() not in ("true", "false"):
            raise ValueError("publish_compressed 只能是 true 或 false")
        overrides["publish_compressed"] = compressed.lower() == "true"
    return [Node(
        package="fastumi_usb_camera",
        executable="usb_camera_node",
        name="usb_camera_node",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[config, overrides],
    )]


def generate_launch_description() -> LaunchDescription:
    """声明配置路径和相机参数，并生成独立节点启动动作。"""
    # 安装后包共享目录中的默认参数文件。
    default_config = Path(get_package_share_directory("fastumi_usb_camera")) / \
        "config" / "usb_camera.yaml"
    arguments = [
        DeclareLaunchArgument("config", default_value=str(default_config)),
        DeclareLaunchArgument("namespace", default_value="usb_camera"),
    ]
    arguments.extend(
        DeclareLaunchArgument(name, default_value="")
        for name in (
            "vendor_id", "product_id", "width", "height", "fps",
            "frame_id", "publish_compressed",
        )
    )
    return LaunchDescription(arguments + [OpaqueFunction(function=_camera_node)])
