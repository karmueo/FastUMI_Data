"""启动 UMI 服务化采集后端和带采集面板的 RViz2，并可选启动设备节点。

默认只启动 ``collection_node`` 与 RViz2，设备按已有节点连接。相机、Tracker
和夹爪估计可分别用 ``start_camera``、``start_tracker``、``start_gripper``
按需启动。采集由 ``/fastumi/collection`` 服务或 RViz2 面板开始，不再随
launch 自动录包。

``image_mode`` 选择相机输出：``auto``（默认）、``raw``、``compressed``（JPEG/PNG）
或 ``h264``（无 B 帧）。``auto`` 启动本地相机时使用 compressed（相机 JPEG 原样录制），
连接外部相机时按实际消息类型订阅。MCAP 始终保存编码话题；夹爪估计和 RViz2 订阅
原始图像，编码模式下由在线解码节点发布到 ``decoded_image_topic``。

``camera_fps``（默认 30）透传给本地相机，并作为标称源帧率写入采集快照；外部相机
的帧率由发布端设置，需用 ``external_camera_fps`` 声明其标称值。在线解码默认处理
全部帧（``decode_every_n_frames=1``），夹爪期望频率按 ``camera_fps / N`` 计算。
"""

import math
import os
from pathlib import Path
import tempfile
import time
from typing import List, Optional, Tuple

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import yaml

from fastumi_usb_camera.capture import is_physical_video_device_path

from fastumi_data.image_modes import (
    DEFAULT_DECODED_IMAGE_TOPIC,
    DEFAULT_H264_START_TIMEOUT_S,
    DEFAULT_IMAGE_QOS_DEPTH,
    DEFAULT_IMAGE_RELIABILITY,
    DEFAULT_IMAGE_TOPICS,
    IMAGE_MODES,
    IMAGE_RELIABILITIES,
    MODE_COMPRESSED,
    MODE_H264,
    MODE_RAW,
    mode_for_message_type,
    resolve_image_topic,
    validate_image_mode,
    validate_image_qos_depth,
    validate_image_reliability,
)


# UMI 相机默认原始图像话题；相机命名空间由它推导。
DEFAULT_IMAGE_TOPIC = DEFAULT_IMAGE_TOPICS[MODE_RAW]
# 各模式的图像话题必须以该后缀结尾，才能推导出 USB 相机节点的命名空间。
IMAGE_TOPIC_SUFFIXES = {
    MODE_RAW: "/image_raw",
    MODE_COMPRESSED: "/image_raw/compressed",
    MODE_H264: "/image_raw/ffmpeg",
}
# H.264 接收端订阅的话题后缀，输入参数是去掉该后缀的基础话题。
H264_TOPIC_SUFFIX = "/ffmpeg"
# image_mode 取该值时按相机实际发布的消息类型决定模式。
MODE_AUTO = "auto"
# 自动检测时等待发布端出现的默认秒数。
DEFAULT_PROBE_TIMEOUT_S = 5.0
# 本地相机默认请求的帧率（Hz）；采集、在线解码和离线数据均为 30 Hz。
DEFAULT_CAMERA_FPS = 30
# 本地 USB 相机节点给图像 Header 打时间戳的方式，写入采集快照。
LOCAL_CAMERA_TIMESTAMP_SOURCE = "v4l2_buffer_monotonic_mapped_to_ros"
# 本地相机每帧时间诊断（sensor_msgs/TimeReference）相对图像基础话题的后缀。
TIME_REFERENCE_SUFFIX = "/image_raw/time_reference"
# 相机时间戳模式与曝光优先控制的合法取值。
CAMERA_TIMESTAMP_MODES = ("v4l2_buffer", "callback_now")
EXPOSURE_PRIORITY_CHOICES = ("leave", "disable")
# 在线 H.264 解码后端，由 usb_camera_receiver 实现。
DECODER_BACKENDS = ("transport", "cuda", "auto")
# 包内 collection.rviz 中 UMI 视频显示默认订阅的话题。
RVIZ_DEFAULT_IMAGE_TOPIC = DEFAULT_IMAGE_TOPIC
# 采集 RViz2 默认固定坐标系，即 vive_tracker 发布位姿所在的 OpenVR 原点。
DEFAULT_FIXED_FRAME = "steamvr_tracking"
# Fast DDS 默认参与者配置的环境变量名。
FASTDDS_PROFILE_ENV = "FASTRTPS_DEFAULT_PROFILES_FILE"
# 旧 record_mcap 参数的迁移提示。
RECORD_MCAP_MIGRATION = (
    "record_mcap 已移除：launch 不再自动连续录包。请在 RViz2 的“数据采集”面板"
    "点击开始，或调用 /fastumi/collection/start 服务；每次采集会独立保存 MCAP。"
    "仍需连续录包时使用 `ros2 run fastumi_data record_session`。"
)


def _flag(context: LaunchContext, name: str) -> bool:
    """读取 true/false 启动参数，其他取值视为错误。"""
    value = LaunchConfiguration(name).perform(context).strip().lower()
    if value not in ("true", "false"):
        raise ValueError(f"{name} 只能是 true 或 false，当前为 {value!r}")
    return value == "true"


def _text(context: LaunchContext, name: str) -> str:
    """读取字符串启动参数并去除首尾空白。"""
    return LaunchConfiguration(name).perform(context).strip()


def _positive_int(context: LaunchContext, name: str) -> int:
    """读取正整数启动参数。"""
    text = _text(context, name)
    try:
        value = int(text, 10)
    except ValueError as error:
        raise ValueError(f"{name} 必须是正整数，当前为 {text!r}") from error
    if value <= 0:
        raise ValueError(f"{name} 必须是正整数，当前为 {value}")
    return value


def _optional_positive_float(context: LaunchContext, name: str) -> float:
    """读取可留空的正有限数参数；留空返回 0。"""
    text = _text(context, name)
    if not text:
        return 0.0
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{name} 必须是正数，当前为 {text!r}") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} 必须是有限正数，当前为 {text!r}")
    return value


def _probe_image_source(
    topic: str, timeout_s: float
) -> Optional[Tuple[str, str]]:
    """在 ROS 图中等待图像发布端，按其消息类型返回 ``(模式, 话题)``。

    ``topic`` 非空时只看该话题，否则依次看三种模式的默认话题。超时没有发现
    任何发布端返回 ``None``。使用独立的 rclpy 上下文，不影响 launch 本身。
    """
    import rclpy
    from rclpy.context import Context

    candidates = [topic] if topic else list(DEFAULT_IMAGE_TOPICS.values())
    context = Context()
    rclpy.init(context=context)
    try:
        node = rclpy.create_node("fastumi_image_probe", context=context)
        deadline = time.monotonic() + timeout_s
        while True:
            known = dict(node.get_topic_names_and_types())
            for candidate in candidates:
                for type_name in known.get(candidate, []):
                    mode = mode_for_message_type(type_name)
                    if mode is not None and node.count_publishers(candidate) > 0:
                        return mode, candidate
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.2)
    finally:
        context.try_shutdown()


def _mode_from_topic_name(topic: str) -> str:
    """无法探测时按话题后缀推断图像模式。"""
    if topic.endswith("/compressed"):
        return MODE_COMPRESSED
    if topic.endswith(H264_TOPIC_SUFFIX):
        return MODE_H264
    return MODE_RAW


def _resolve_image_source(
    requested_mode: str, requested_topic: str, start_camera: bool, timeout_s: float
) -> Tuple[str, str]:
    """确定实际的图像模式和话题。

    ``auto`` 且启动本地相机时使用 compressed（相机 JPEG 原样录制）；连接已有设备
    时按发布端的消息类型订阅，探测不到发布端则退回话题后缀或 raw 默认值并给出提示。
    """
    mode = requested_mode.strip().lower()
    if mode != MODE_AUTO:
        mode = validate_image_mode(mode)
        return mode, resolve_image_topic(mode, requested_topic)
    if start_camera:
        return MODE_COMPRESSED, resolve_image_topic(MODE_COMPRESSED, requested_topic)
    found = _probe_image_source(requested_topic, timeout_s)
    if found is not None:
        print(f"[fastumi_collection] 检测到图像话题 {found[1]}，模式 {found[0]}")
        return found
    fallback = _mode_from_topic_name(requested_topic) if requested_topic else MODE_RAW
    topic = resolve_image_topic(fallback, requested_topic)
    print(
        f"[fastumi_collection] {timeout_s:g} s 内没有发现图像发布端，按 {fallback} 订阅 {topic}；"
        "设备稍后启动时若类型不同，开始前检查会提示不一致"
    )
    return fallback, topic


def _camera_namespace(image_topic: str, image_mode: str = MODE_RAW) -> str:
    """从 ``/<namespace>/image_raw[/compressed|/ffmpeg]`` 推导 USB 相机命名空间。"""
    suffix = IMAGE_TOPIC_SUFFIXES[image_mode]
    if not image_topic.startswith("/") or not image_topic.endswith(suffix):
        raise ValueError(
            f"启动相机时 image_topic 必须形如 /<namespace>{suffix}，"
            f"当前为 {image_topic!r}"
        )
    namespace = image_topic[1 : -len(suffix)]
    if not namespace:
        raise ValueError("启动相机时 image_topic 必须带命名空间")
    return namespace


def _h264_base_topic(image_topic: str) -> str:
    """H.264 编码话题去掉 ``/ffmpeg`` 后得到接收器和编码器使用的基础话题。"""
    if not image_topic.endswith(H264_TOPIC_SUFFIX) or image_topic == H264_TOPIC_SUFFIX:
        raise ValueError(
            f"h264 的 image_topic 必须形如 /<基础话题>{H264_TOPIC_SUFFIX}，"
            f"当前为 {image_topic!r}"
        )
    return image_topic[: -len(H264_TOPIC_SUFFIX)]


def _transport_prefix(base_topic: str) -> str:
    """ffmpeg_image_transport 参数前缀：去掉开头 ``/``，余下 ``/`` 改为 ``.``。"""
    return base_topic.lstrip("/").replace("/", ".") + ".ffmpeg."


def _write_ffmpeg_config(base_topic: str) -> str:
    """为指定基础话题生成本地 H.264 编码配置，返回临时 YAML 路径。

    以包内 ``ffmpeg.yaml`` 为模板：话题改为 ``base_topic``，编码插件参数前缀
    按文档规则同步改名，其余编码参数（libx264、GOP、无 B 帧）保持不变。
    """
    source = (
        Path(get_package_share_directory("fastumi_usb_camera")) / "config" / "ffmpeg.yaml"
    )
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    parameters = document["usb_camera_ffmpeg"]["ros__parameters"]
    old_prefix = _transport_prefix(parameters["topic"])
    new_prefix = _transport_prefix(base_topic)
    rewritten = {}
    for key, value in parameters.items():
        if key == "topic":
            rewritten[key] = base_topic
        elif key.startswith(old_prefix):
            rewritten[new_prefix + key[len(old_prefix):]] = value
        else:
            rewritten[key] = value
    document["usb_camera_ffmpeg"]["ros__parameters"] = rewritten
    target = Path(tempfile.mkdtemp(prefix="fastumi_ffmpeg_")) / "ffmpeg.yaml"
    target.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return str(target)


def _rviz_config_for(config: str, view_topic: str) -> str:
    """让 RViz2 的 UMI 视频显示订阅实际的原始图像话题。

    只改订阅包内默认话题的 Image 显示；话题已相同或用户自定义配置指向
    其他话题时原样使用，不生成临时文件。
    """
    if view_topic == RVIZ_DEFAULT_IMAGE_TOPIC:
        return config
    document = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    changed = False
    for display in document.get("Visualization Manager", {}).get("Displays", []):
        topic = display.get("Topic")
        if (
            display.get("Class") == "rviz_default_plugins/Image"
            and isinstance(topic, dict)
            and topic.get("Value") == RVIZ_DEFAULT_IMAGE_TOPIC
        ):
            topic["Value"] = view_topic
            changed = True
    if not changed:
        return config
    target = Path(tempfile.mkdtemp(prefix="fastumi_rviz_")) / Path(config).name
    target.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return str(target)


def _gripper_snapshot_files(
    camera_calibration: str, gripper_calibration: str
) -> List[str]:
    """解析夹爪估计实际使用的内参和夹爪标定文件，供采集快照。

    与 ``gripper_openness.launch.py`` 的默认规则一致：夹爪标定优先使用
    用户主目录下新生成的文件，否则使用包内示例。
    """
    share = Path(get_package_share_directory("gripper_openness")) / "config"
    camera = camera_calibration or str(share / "calib.yaml")
    user_calibration = Path.home() / "fastumi_gripper_calibration.yaml"
    gripper = gripper_calibration or (
        str(user_calibration)
        if user_calibration.is_file()
        else str(share / "calibration.yaml")
    )
    return [camera, gripper]


def _decoder_node(
    image_mode: str,
    image_topic: str,
    decoded_image_topic: str,
    image_reliability: str,
    image_qos_depth: int,
    decoder_backend: str,
    decode_every_n_frames: int = 1,
) -> Node:
    """创建在线解码节点：JPEG/PNG 用可配置 QoS 的解码节点，H.264 用 C++ 接收器。

    ``image_transport republish`` 的订阅固定为 RELIABLE，无法匹配 BEST_EFFORT 的
    相机发布端，因此 JPEG/PNG 不使用它。
    """
    if image_mode == MODE_COMPRESSED:
        return Node(
            package="fastumi_data",
            executable="image_decoder",
            name="image_decoder",
            output="screen",
            parameters=[
                {
                    "image_mode": image_mode,
                    "input_topic": image_topic,
                    "output_topic": decoded_image_topic,
                    "image_reliability": image_reliability,
                    "image_qos_depth": image_qos_depth,
                    "decode_every_n_frames": decode_every_n_frames,
                }
            ],
        )
    base_topic = _h264_base_topic(image_topic)
    return Node(
        package="fastumi_usb_camera",
        executable="usb_camera_receiver",
        name="usb_camera_receiver",
        output="screen",
        parameters=[
            {
                "input_topic": base_topic,
                "output_topic": decoded_image_topic,
                "decoder_backend": decoder_backend,
                "output_queue_depth": 5,
                "input_reliability": image_reliability,
                "input_queue_depth": image_qos_depth,
                # 全部访问单元都送入解码器，只按解码后帧序号降频发布。
                "output_every_n_frames": decode_every_n_frames,
                # 软件解码，避免 cuvid 解码帧的 PTS 无法匹配 Jetson 流。
                f"{_transport_prefix(base_topic)}decoders.h264": "h264",
            }
        ],
    )


def _launch_setup(context: LaunchContext) -> list:
    """校验参数并按需展开采集节点、设备子 launch 和 RViz2。"""
    if _text(context, "record_mcap").lower() == "true":
        raise ValueError(RECORD_MCAP_MIGRATION)
    start_camera = _flag(context, "start_camera")
    start_tracker = _flag(context, "start_tracker")
    start_gripper = _flag(context, "start_gripper")
    image_mode, image_topic = _resolve_image_source(
        _text(context, "image_mode"),
        _text(context, "image_topic"),
        start_camera,
        float(_text(context, "image_probe_timeout_s")),
    )
    decoded_image_topic = _text(context, "decoded_image_topic")
    image_reliability = validate_image_reliability(_text(context, "image_reliability"))
    image_qos_depth = validate_image_qos_depth(int(_text(context, "image_qos_depth")))
    h264_start_timeout_s = float(_text(context, "h264_start_timeout_s"))
    camera_fps = _positive_int(context, "camera_fps")
    external_camera_fps = _optional_positive_float(context, "external_camera_fps")
    decode_every_n_frames = _positive_int(context, "decode_every_n_frames")
    camera_timestamp_mode = _text(context, "camera_timestamp_mode")
    if camera_timestamp_mode not in CAMERA_TIMESTAMP_MODES:
        raise ValueError(
            f"camera_timestamp_mode 必须是 {'、'.join(CAMERA_TIMESTAMP_MODES)} 之一，"
            f"当前为 {camera_timestamp_mode!r}"
        )
    exposure_auto_priority = _text(context, "exposure_auto_priority")
    if exposure_auto_priority not in EXPOSURE_PRIORITY_CHOICES:
        raise ValueError(
            f"exposure_auto_priority 必须是 {'、'.join(EXPOSURE_PRIORITY_CHOICES)} 之一，"
            f"当前为 {exposure_auto_priority!r}"
        )
    decoder_backend = _text(context, "decoder_backend")
    if decoder_backend not in DECODER_BACKENDS:
        raise ValueError(
            f"decoder_backend 必须是 {'、'.join(DECODER_BACKENDS)} 之一，当前为 {decoder_backend!r}"
        )
    if not decoded_image_topic.startswith("/"):
        raise ValueError(f"decoded_image_topic 必须是绝对话题，当前为 {decoded_image_topic!r}")
    # 夹爪估计和 RViz2 订阅原始图像：raw 直接用相机话题，编码模式用解码话题。
    view_topic = image_topic if image_mode == MODE_RAW else decoded_image_topic
    video_device = _text(context, "video_device")
    camera_calibration = _text(context, "camera_calibration_path")
    gripper_calibration = _text(context, "gripper_calibration_path")

    actions: list = []
    profile = _text(context, "fastdds_profile")
    # 默认 SHM 段装不下 6 MB 的原始图像，会回退到 UDP 并在实机上丢约 7% 的帧。
    # 用户已自行设置该环境变量时不覆盖；本 launch 启动的全部子进程都继承它。
    if profile and FASTDDS_PROFILE_ENV not in os.environ:
        actions.append(SetEnvironmentVariable(FASTDDS_PROFILE_ENV, profile))
    if start_camera:
        # video_device 留空时由相机节点自动选择编号最小的 USB 主视频节点。
        if video_device and not is_physical_video_device_path(video_device):
            raise ValueError(
                "video_device 必须使用 /dev/v4l/by-path/*-video-index0 物理端口路径"
            )
        # 本地相机三种输出对应现有节点：raw、仅 JPEG、软件 H.264；连接 Jetson 等
        # 外部发布的话题时不启动本地相机，直接给出对应的 image_mode 与话题。
        namespace = _camera_namespace(image_topic, image_mode)
        camera_arguments = {
            "namespace": namespace,
            "fps": str(camera_fps),
            "timestamp_mode": camera_timestamp_mode,
            "exposure_auto_priority": exposure_auto_priority,
        }
        if video_device:
            camera_arguments["video_device"] = video_device
        if image_mode == MODE_COMPRESSED:
            camera_arguments["publish_compressed"] = "true"
        elif image_mode == MODE_H264:
            camera_arguments["enable_ffmpeg"] = "true"
            camera_arguments["ffmpeg_config"] = _write_ffmpeg_config(
                _h264_base_topic(image_topic)
            )
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("fastumi_usb_camera"),
                            "launch",
                            "usb_camera.launch.py",
                        ]
                    )
                ),
                launch_arguments=camera_arguments.items(),
            )
        )
    if start_tracker:
        # 采集面板自带 RViz2，关闭 Tracker 子 launch 自带的那个。
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("vive_tracker"),
                            "launch",
                            "vive_tracker.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "serial": _text(context, "tracker_serial"),
                    "use_rviz": "false",
                }.items(),
            )
        )
    # 本 launch 声明了同名的标定参数（默认空），会遮蔽子 launch 的默认值使估计节点
    # 回退到 ToF 标定，因此始终把按子 launch 规则解析出的实际路径显式传入。
    resolved_camera, resolved_gripper = _gripper_snapshot_files(
        camera_calibration, gripper_calibration
    )
    gripper_arguments = {
        "image_topic": view_topic,
        "publish_debug_image": "true" if _flag(context, "publish_debug_image") else "false",
        "camera_calibration_path": resolved_camera,
        "gripper_calibration_path": resolved_gripper,
    }
    if start_gripper:
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("fastumi_gripper_estimator"),
                            "launch",
                            "gripper_openness.launch.py",
                        ]
                    )
                ),
                launch_arguments=gripper_arguments.items(),
            )
        )

    use_rviz = _flag(context, "use_rviz")
    if image_mode != MODE_RAW and _flag(context, "start_decoder") and (
        start_gripper or use_rviz
    ):
        actions.append(
            _decoder_node(
                image_mode,
                image_topic,
                decoded_image_topic,
                image_reliability,
                image_qos_depth,
                decoder_backend,
                decode_every_n_frames,
            )
        )

    # 标称源帧率：本地相机为确认透传的 camera_fps；外部相机须显式声明，不用采集
    # 入口的默认值冒充设备实际配置。
    if start_camera:
        nominal_fps, nominal_source = float(camera_fps), "camera_launch"
    elif external_camera_fps > 0.0:
        nominal_fps, nominal_source = external_camera_fps, "external_declared"
    else:
        nominal_fps, nominal_source = 0.0, "unknown"
        print(
            "[fastumi_collection] 外部相机未声明 external_camera_fps，采集快照的标称源帧率"
            "记为未知；离线转换时需用 --source-camera-fps 声明"
        )
    # 编码模式的夹爪估计订阅解码话题，期望频率随在线降频按 N 降低。
    gripper_rate = nominal_fps if image_mode == MODE_RAW else nominal_fps / decode_every_n_frames
    parameters: dict = {
        "dataset_root": _text(context, "dataset_root"),
        "nominal_source_fps": nominal_fps,
        "nominal_source_fps_source": nominal_source,
        "expected_image_rate_hz": nominal_fps,
        "expected_gripper_rate_hz": gripper_rate,
        "image_timestamp_source": (
            (
                LOCAL_CAMERA_TIMESTAMP_SOURCE
                if camera_timestamp_mode == "v4l2_buffer"
                else "callback_now"
            )
            if start_camera
            else "external_publisher"
        ),
        # 本地相机录制每帧时间诊断，并在时钟映射重建时让当前采集失败。
        "image_timing_topic": (
            f"/{_camera_namespace(image_topic, image_mode)}{TIME_REFERENCE_SUFFIX}"
            if start_camera
            else ""
        ),
        "image_mode": image_mode,
        "image_topic": image_topic,
        "decoded_image_topic": decoded_image_topic,
        "image_reliability": image_reliability,
        "image_qos_depth": image_qos_depth,
        "h264_start_timeout_s": h264_start_timeout_s,
    }
    for launch_name, parameter in (
        ("extrinsic_path", "extrinsic_path"),
        ("processing_config", "processing_config"),
    ):
        if _text(context, launch_name):
            parameters[parameter] = _text(context, launch_name)
    if start_gripper or camera_calibration or gripper_calibration:
        snapshots = [resolved_camera, resolved_gripper]
        if len({Path(path).name for path in snapshots}) != len(snapshots):
            raise ValueError(
                "相机内参与夹爪标定文件同名，无法同时写入采集快照: "
                + ", ".join(snapshots)
            )
        parameters["snapshot_files"] = snapshots
    actions.append(
        Node(
            package="fastumi_data",
            executable="collection_node",
            name="collection_node",
            output="screen",
            parameters=[
                str(
                    Path(get_package_share_directory("fastumi_data"))
                    / "config"
                    / "collection.yaml"
                ),
                parameters,
            ],
        )
    )
    # 在此解析而不是用运行时条件：Tracker 子 launch 的 use_rviz:=false 参数会泄漏到
    # 同一上下文，使稍后求值的 IfCondition 误判为关闭。
    if use_rviz:
        actions.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="fastumi_collection_rviz",
                output="screen",
                arguments=[
                    "-d",
                    _rviz_config_for(_text(context, "rviz_config"), view_topic),
                    "-f",
                    _text(context, "fixed_frame"),
                ],
            )
        )
    return actions


def generate_launch_description() -> LaunchDescription:
    """创建采集后端、RViz2 和可选设备节点的启动描述。"""
    data_share = FindPackageShare("fastumi_data")
    flag_choices = ["true", "false"]
    arguments = [
        DeclareLaunchArgument(
            "dataset_root",
            default_value="dataset",
            description="数据集根目录；内含 <task>/<UTC>_<UUID>、.staging 和 .trash。",
        ),
        DeclareLaunchArgument(
            "camera_fps",
            default_value=str(DEFAULT_CAMERA_FPS),
            description=(
                "本地相机请求帧率（正整数，默认 30），透传给相机子 launch 的 fps，"
                "设备无法达到时明确报错；同时作为标称源帧率写入采集快照。"
                "对 raw、JPEG、H.264 三种输出均生效；调整需重启相机节点。"
            ),
        ),
        DeclareLaunchArgument(
            "external_camera_fps",
            default_value="",
            description=(
                "start_camera:=false 时外部相机的标称帧率（如 30、29.97）；留空记为未知，"
                "此时不检查图像绝对速率，离线转换需显式声明。"
            ),
        ),
        DeclareLaunchArgument(
            "camera_timestamp_mode",
            default_value="v4l2_buffer",
            choices=list(CAMERA_TIMESTAMP_MODES),
            description=(
                "本地相机 Header 时间：v4l2_buffer（默认）把 V4L2 缓冲区单调时间映射到 ROS 时间；"
                "callback_now 为兼容模式，在回调中读取 ROS 时间，来源写入快照。"
            ),
        ),
        DeclareLaunchArgument(
            "exposure_auto_priority",
            default_value="leave",
            choices=list(EXPOSURE_PRIORITY_CHOICES),
            description=(
                "V4L2 曝光优先控制：leave（默认）只读取并记录；disable 写 0 并读回确认，"
                "失败如实记录、不放宽帧率门限。当前设备无法可靠关闭，须统一稳定照明。"
            ),
        ),
        DeclareLaunchArgument(
            "decode_every_n_frames",
            default_value="1",
            description=(
                "在线解码降负载：每 N 帧解码发布一帧（默认 1 即全部 30 Hz）。只影响夹爪估计"
                "和预览；MCAP 仍录制完整编码流，H.264 所有包仍送入解码器。"
            ),
        ),
        DeclareLaunchArgument(
            "image_mode",
            default_value=MODE_AUTO,
            choices=[MODE_AUTO, *IMAGE_MODES],
            description=(
                "相机输出：auto（默认）按发布端实际消息类型订阅，启动本地相机时为 compressed；"
                "raw（sensor_msgs/Image）、compressed（JPEG/PNG 的 CompressedImage）"
                "或 h264（无 B 帧的 FFMPEGPacket）可显式指定。MCAP 原样保存收到的消息。"
            ),
        ),
        DeclareLaunchArgument(
            "image_probe_timeout_s",
            default_value=str(DEFAULT_PROBE_TIMEOUT_S),
            description="image_mode:=auto 连接已有设备时等待图像发布端出现的秒数。",
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="",
            description=(
                "录制的图像话题；留空按 image_mode 使用 /umi_camera/image_raw、"
                "/umi_camera/image_raw/compressed 或 /umi_camera/image_raw/ffmpeg。"
                "连接 Jetson 等外部发布端时显式给出其话题。"
            ),
        ),
        DeclareLaunchArgument(
            "decoded_image_topic",
            default_value=DEFAULT_DECODED_IMAGE_TOPIC,
            description="编码模式下在线解码后发布的原始图像话题，供夹爪估计和 RViz2 订阅。",
        ),
        DeclareLaunchArgument(
            "image_reliability",
            default_value=DEFAULT_IMAGE_RELIABILITY,
            choices=list(IMAGE_RELIABILITIES),
            description="图像订阅可靠性；默认 best_effort 兼容当前相机发布端。",
        ),
        DeclareLaunchArgument(
            "image_qos_depth",
            default_value=str(DEFAULT_IMAGE_QOS_DEPTH),
            description="图像订阅 KEEP_LAST 队列深度。",
        ),
        DeclareLaunchArgument(
            "h264_start_timeout_s",
            default_value=str(DEFAULT_H264_START_TIMEOUT_S),
            description="h264 模式下开始采集等待首个 SPS/PPS+IDR 关键帧的最长秒数。",
        ),
        DeclareLaunchArgument(
            "start_decoder",
            default_value="true",
            choices=flag_choices,
            description=(
                "编码模式下是否启动在线解码节点；夹爪估计或 RViz2 启用时才会创建。"
                "已有外部解码节点发布 decoded_image_topic 时设为 false。"
            ),
        ),
        DeclareLaunchArgument(
            "decoder_backend",
            default_value="auto",
            choices=list(DECODER_BACKENDS),
            description=(
                "h264 在线解码后端（usb_camera_receiver）：auto 优先 GPU（h264_cuvid），"
                "初始化失败用同一关键帧改用软件，运行失败后在下一关键帧切换软件并保持到重启；"
                "transport 为 image_transport 软件插件；cuda 仅 GPU，失败即退出。"
            ),
        ),
        DeclareLaunchArgument(
            "extrinsic_path",
            default_value="",
            description="可选 Tracker 到 TCP 外参；留空时记录标记为未标定。",
        ),
        DeclareLaunchArgument(
            "processing_config",
            default_value="",
            description="可选处理配置模板；留空使用 fastumi_data 默认配置。",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            choices=flag_choices,
            description="是否启动带采集面板的 RViz2。",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=PathJoinSubstitution([data_share, "config", "collection.rviz"]),
            description="RViz2 配置文件，默认含采集面板、Image 和 Tracker Pose。",
        ),
        DeclareLaunchArgument(
            "fixed_frame",
            default_value=DEFAULT_FIXED_FRAME,
            description="RViz2 固定坐标系，覆盖配置文件中的值。",
        ),
        DeclareLaunchArgument(
            "start_camera",
            default_value="false",
            choices=flag_choices,
            description="是否启动 USB 相机；video_device 留空时自动选择第一个 USB 相机。",
        ),
        DeclareLaunchArgument(
            "video_device",
            default_value="",
            description=(
                "相机 /dev/v4l/by-path/*-video-index0 物理端口路径；留空则自动选择"
                "第一个可用的 USB 相机（多相机时不保证端口稳定，建议显式指定）。"
            ),
        ),
        DeclareLaunchArgument(
            "start_tracker",
            default_value="false",
            choices=flag_choices,
            description="是否启动 VIVE Tracker 位姿节点（不含其自带 RViz2）。",
        ),
        DeclareLaunchArgument(
            "tracker_serial",
            default_value="",
            description="可选 Tracker 序列号；留空使用 vive_tracker 配置。",
        ),
        DeclareLaunchArgument(
            "start_gripper",
            default_value="false",
            choices=flag_choices,
            description="是否启动夹爪开合度估计节点。",
        ),
        DeclareLaunchArgument(
            "publish_debug_image",
            default_value="false",
            choices=flag_choices,
            description="是否发布夹爪 ArUco 调试图像。",
        ),
        DeclareLaunchArgument(
            "camera_calibration_path",
            default_value="",
            description="可选相机内参 YAML；留空沿用夹爪估计默认内参。",
        ),
        DeclareLaunchArgument(
            "gripper_calibration_path",
            default_value="",
            description="可选夹爪范围标定 YAML；留空沿用夹爪估计默认标定。",
        ),
        DeclareLaunchArgument(
            "fastdds_profile",
            default_value=PathJoinSubstitution(
                [data_share, "config", "fastdds_large_images.xml"]
            ),
            description=(
                "Fast DDS 配置，放大共享内存段以无丢帧传输 1080p 图像；"
                "留空禁用。仅作用于本 launch 启动的进程，外部设备进程需自行设置 "
                "FASTRTPS_DEFAULT_PROFILES_FILE。"
            ),
        ),
        DeclareLaunchArgument(
            "record_mcap",
            default_value="false",
            description="已移除；为 true 时报迁移提示。",
        ),
    ]
    return LaunchDescription([*arguments, OpaqueFunction(function=_launch_setup)])
