"""按启动参数选择 USB 图像或 FFmpeg 传输，并覆盖相机配置。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _camera_node(context):
    """每次只启动一个采集节点，并应用显式给出的相机参数。"""
    # 默认相机配置使用 /** 作用域，可同时覆盖 Python 和 C++ 采集节点。
    config = LaunchConfiguration("config").perform(context)
    overrides = {}
    for name in ("vendor_id", "product_id", "width", "height", "fps"):
        value = LaunchConfiguration(name).perform(context)
        if value:
            overrides[name] = int(value, 0)
    frame_id = LaunchConfiguration("frame_id").perform(context)
    if frame_id:
        overrides["frame_id"] = frame_id
    device_uid = LaunchConfiguration("device_uid").perform(context)
    video_device = LaunchConfiguration("video_device").perform(context)
    if device_uid and video_device:
        raise ValueError("device_uid 和 video_device 只能指定其中一个")
    enable_ffmpeg = LaunchConfiguration("enable_ffmpeg").perform(context).lower()
    if enable_ffmpeg not in ("true", "false"):
        raise ValueError("enable_ffmpeg 只能是 true 或 false")
    compressed = LaunchConfiguration("publish_compressed").perform(context).lower()
    if compressed and compressed not in ("true", "false"):
        raise ValueError("publish_compressed 只能是 true 或 false")

    if enable_ffmpeg == "true":
        if device_uid or video_device:
            raise ValueError("device_uid 和 video_device 仅支持 Python 相机模式")
        # 先加载编码与话题配置，再由相机配置和命令行相机参数逐层覆盖。
        return [Node(
            package="fastumi_usb_camera",
            executable="usb_camera_ffmpeg",
            name="usb_camera_ffmpeg",
            output="screen",
            parameters=[LaunchConfiguration("ffmpeg_config"), config, overrides],
        )]

    if compressed:
        overrides["publish_compressed"] = compressed == "true"
    if device_uid:
        overrides["device_uid"] = device_uid
        overrides["video_device"] = ""
    if video_device:
        overrides["video_device"] = video_device
    elif not device_uid and any(
        name in overrides for name in ("vendor_id", "product_id")
    ):
        # 显式指定 VID/PID 时关闭 YAML 中的默认视频设备路径。
        overrides["video_device"] = ""
    return [Node(
        package="fastumi_usb_camera",
        executable="usb_camera_node",
        name="usb_camera_node",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[config, overrides],
    )]


def generate_launch_description() -> LaunchDescription:
    """声明配置与模式参数，按模式生成一个相机节点。"""
    # 安装后包共享目录中的默认参数文件。
    default_config = Path(get_package_share_directory("fastumi_usb_camera")) / \
        "config" / "usb_camera.yaml"
    default_ffmpeg_config = default_config.with_name("ffmpeg.yaml")
    arguments = [
        DeclareLaunchArgument("config", default_value=str(default_config)),
        DeclareLaunchArgument("ffmpeg_config", default_value=str(default_ffmpeg_config)),
        DeclareLaunchArgument("namespace", default_value="usb_camera"),
        DeclareLaunchArgument("enable_ffmpeg", default_value="false"),
    ]
    arguments.extend(
        DeclareLaunchArgument(name, default_value="")
        for name in (
            "vendor_id", "product_id", "width", "height", "fps",
            "frame_id", "publish_compressed", "device_uid", "video_device",
        )
    )
    return LaunchDescription(arguments + [OpaqueFunction(function=_camera_node)])
