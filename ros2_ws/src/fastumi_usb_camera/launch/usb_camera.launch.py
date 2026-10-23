"""按启动参数选择 USB 图像或 FFmpeg 传输，并覆盖相机配置。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction
from launch.events import Shutdown
from launch.logging import get_logger
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.utilities import make_namespace_absolute, prefix_namespace
from launch_ros.utilities.to_parameters_list import is_node_name_matched
import yaml

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
    publish_reliability = LaunchConfiguration("publish_reliability").perform(context).strip().lower()
    if publish_reliability not in ("reliable", "best_effort"):
        raise ValueError("publish_reliability 只能是 reliable 或 best_effort")
    compressed = LaunchConfiguration("publish_compressed").perform(context).lower()
    if compressed and compressed not in ("true", "false"):
        raise ValueError("publish_compressed 只能是 true 或 false")

    overlay = _alignment_overlay_settings(context)

    if enable_ffmpeg == "true":
        if overlay is not None and enable_decoder != "true":
            raise ValueError("alignment_overlay=true 在 FFmpeg 模式下需要 enable_decoder=true")
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
        overrides["publish_reliability"] = publish_reliability
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
            if overlay is not None:
                # 解码后的原始图像叠加；输出与 Python 模式同名，位于相机话题的父命名空间。
                parent = topic.rsplit("/", 1)[0] or ""
                nodes.append(_alignment_overlay_node(
                    overlay, decoded_topic, False, f"{parent}/image_alignment/compressed"))
        return nodes

    if compressed:
        overrides["publish_compressed"] = compressed == "true"
    overrides["publish_reliability"] = publish_reliability
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
    nodes = [Node(
        package="fastumi_usb_camera",
        executable="usb_camera_node",
        name="usb_camera_node",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[config, overrides],
    )]
    if overlay is not None:
        # publish_compressed 未显式给出时沿用相机配置文件中的值。
        is_compressed = compressed == "true" if compressed else \
            _configured_publish_compressed(context, config)
        nodes.append(_alignment_overlay_node(
            overlay, "image_raw/compressed" if is_compressed else "image_raw",
            is_compressed, "image_alignment/compressed"))
    return nodes


def _matching_parameters(document, node_fqn, prefix=""):
    """按 YAML 顺序合并匹配完整节点名的参数，支持嵌套命名空间与 *、** 通配符。"""
    parameters = {}
    for name, value in document.items():
        if name == "ros__parameters":
            if is_node_name_matched(prefix, node_fqn):
                parameters.update(value)
        elif isinstance(value, dict):
            parameters.update(_matching_parameters(
                value, node_fqn, f"{prefix}/{name.strip('/')}"))
    return parameters


def _configured_publish_compressed(context, config):
    """与相机 Node 一致地展开命名空间；未配置输出模式时沿用节点默认 raw。"""
    namespace = make_namespace_absolute(prefix_namespace(
        context.launch_configurations.get("ros_namespace"),
        LaunchConfiguration("namespace").perform(context)))
    node_fqn = prefix_namespace(namespace, "usb_camera_node")
    with open(config, encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    return bool(_matching_parameters(document, node_fqn).get("publish_compressed", False))


def _alignment_overlay_settings(context):
    """解析安装对齐叠加参数；未开启时返回 None。"""
    enabled = LaunchConfiguration("alignment_overlay").perform(context).strip().lower()
    if enabled not in ("true", "false"):
        raise ValueError("alignment_overlay 只能是 true 或 false")
    if enabled == "false":
        return None
    mask = Path(LaunchConfiguration("alignment_mask").perform(context))
    if not mask.is_file():
        raise ValueError(f"alignment_mask 文件不存在: {mask}")
    alpha = float(LaunchConfiguration("alignment_alpha").perform(context))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alignment_alpha 必须在 [0, 1] 内")
    return {"mask_file": str(mask), "alpha": alpha}


def _alignment_overlay_node(overlay, input_topic, compressed, output_topic):
    """叠加节点只订阅相机输出，不影响原始话题、录制和推理输入。"""
    return Node(
        package="fastumi_usb_camera",
        executable="alignment_overlay_node",
        name="alignment_overlay",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        parameters=[{
            **overlay,
            "input_topic": input_topic,
            "compressed": compressed,
            "output_topic": output_topic,
        }],
    )


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
        DeclareLaunchArgument("publish_reliability", default_value="reliable"),
        DeclareLaunchArgument("topic", default_value="/usb_camera/image_raw"),
        DeclareLaunchArgument("decoded_topic", default_value="/usb_camera/image_decoded"),
        # 末端相机安装对齐：把 UMI 夹爪 mask 按透明度叠加后发布到 image_alignment/compressed。
        DeclareLaunchArgument("alignment_overlay", default_value="false"),
        DeclareLaunchArgument(
            "alignment_mask", default_value=str(default_config.with_name("umi_gripper_mask.png"))),
        DeclareLaunchArgument("alignment_alpha", default_value="0.5"),
    ]
    arguments.extend(
        DeclareLaunchArgument(name, default_value="")
        for name in (
            "vendor_id", "product_id", "width", "height", "fps",
            "frame_id", "publish_compressed", "device_uid", "video_device",
        )
    )
    return LaunchDescription(arguments + [OpaqueFunction(function=_camera_node)])
