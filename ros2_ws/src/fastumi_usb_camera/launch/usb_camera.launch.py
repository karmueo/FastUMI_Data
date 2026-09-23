"""按启动参数选择 USB 图像或 FFmpeg 传输，并覆盖相机配置。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction
from launch.events import Shutdown
from launch.logging import get_logger
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from fastumi_usb_camera.capture import is_physical_video_device_path


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
    if video_device and not is_physical_video_device_path(video_device):
        raise ValueError(
            "video_device 必须使用 /dev/v4l/by-path/*-video-index0 物理端口路径"
        )
    if device_uid and video_device:
        raise ValueError("device_uid 和 video_device 只能指定其中一个")
    enable_ffmpeg = LaunchConfiguration("enable_ffmpeg").perform(context).lower()
    if enable_ffmpeg not in ("true", "false"):
        raise ValueError("enable_ffmpeg 只能是 true 或 false")
    enable_decoder = LaunchConfiguration("enable_decoder").perform(context).lower()
    if enable_decoder not in ("true", "false"):
        raise ValueError("enable_decoder 只能是 true 或 false")
    if enable_decoder == "true" and enable_ffmpeg != "true":
        raise ValueError("enable_decoder=true 仅支持 FFmpeg 模式")
    h264_encoder = LaunchConfiguration("h264_encoder").perform(context)
    if h264_encoder not in ("hardware", "software"):
        raise ValueError("h264_encoder 只能是 hardware 或 software")
    compressed = LaunchConfiguration("publish_compressed").perform(context).lower()
    if compressed and compressed not in ("true", "false"):
        raise ValueError("publish_compressed 只能是 true 或 false")

    if enable_ffmpeg == "true":
        if device_uid:
            raise ValueError("device_uid 仅支持 Python 相机模式")
        if video_device:
            overrides["video_device"] = video_device
        elif any(name in overrides for name in ("vendor_id", "product_id")):
            overrides["video_device"] = ""
        topic = LaunchConfiguration("topic").perform(context)
        decoded_topic = LaunchConfiguration("decoded_topic").perform(context)
        if not topic.startswith("/") or topic.endswith("/"):
            raise ValueError("topic 必须是不以 / 结尾的绝对话题")
        if not decoded_topic.startswith("/") or decoded_topic.endswith("/"):
            raise ValueError("decoded_topic 必须是不以 / 结尾的绝对话题")
        overrides["topic"] = topic
        overrides["h264_encoder"] = h264_encoder
        prefix = topic.lstrip("/").replace("/", ".") + ".ffmpeg."
        overrides.update({
            prefix + "encoder": "libx264",
            prefix + "encoder_av_options":
                "preset:ultrafast,tune:zerolatency,profile:baseline",
            prefix + "pixel_format": "yuv420p",
            prefix + "bit_rate": 4_000_000,
            prefix + "gop_size": 10,
            prefix + "max_b_frames": 0,
        })
        # 先加载编码与话题配置，再由相机配置和命令行相机参数逐层覆盖。
        nodes = [Node(
            package="fastumi_usb_camera",
            executable="usb_camera_ffmpeg",
            name="usb_camera_ffmpeg",
            output="screen",
            parameters=[LaunchConfiguration("ffmpeg_config"), config, overrides],
            on_exit=_shutdown_on_camera_exit,
        )]
        if enable_decoder == "true":
            nodes.append(Node(
                package="fastumi_usb_camera",
                executable="usb_camera_receiver",
                name="usb_camera_receiver",
                output="screen",
                parameters=[LaunchConfiguration("ffmpeg_config"), {
                    "input_topic": topic,
                    "output_topic": decoded_topic,
                }],
            ))
        return nodes

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


def _shutdown_on_camera_exit(event, context):
    """H.264 相机意外退出时关闭包含它的统一启动。"""
    if context.is_shutdown:
        return []
    reason = f"H.264 相机节点退出（退出码 {event.returncode}），统一启动退出"
    get_logger("fastumi_usb_camera").error(reason)
    return [EmitEvent(event=Shutdown(reason=reason))]


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
        DeclareLaunchArgument("enable_decoder", default_value="false"),
        DeclareLaunchArgument("h264_encoder", default_value="hardware"),
        DeclareLaunchArgument("topic", default_value="/usb_camera/image_raw"),
        DeclareLaunchArgument("decoded_topic", default_value="/usb_camera/image_decoded"),
    ]
    arguments.extend(
        DeclareLaunchArgument(name, default_value="")
        for name in (
            "vendor_id", "product_id", "width", "height", "fps",
            "frame_id", "publish_compressed", "device_uid", "video_device",
        )
    )
    return LaunchDescription(arguments + [OpaqueFunction(function=_camera_node)])
