"""定义采集、转换、标定和回放共用的图像模式、消息类型和默认参数。

支持三种相机输出：原始 ``sensor_msgs/Image``、JPEG/PNG 的
``sensor_msgs/CompressedImage``，以及无 B 帧、每包一个访问单元的 H.264
``ffmpeg_image_transport_msgs/FFMPEGPacket``。本模块不依赖 ROS 运行时。
"""

from __future__ import annotations

from typing import Optional


# 三种图像模式的取值。
MODE_RAW = "raw"
MODE_COMPRESSED = "compressed"
MODE_H264 = "h264"
IMAGE_MODES = (MODE_RAW, MODE_COMPRESSED, MODE_H264)
DEFAULT_IMAGE_MODE = MODE_RAW

# 各模式对应的 ROS 2 消息类型。
IMAGE_MESSAGE_TYPES = {
    MODE_RAW: "sensor_msgs/msg/Image",
    MODE_COMPRESSED: "sensor_msgs/msg/CompressedImage",
    MODE_H264: "ffmpeg_image_transport_msgs/msg/FFMPEGPacket",
}

# 各模式在 UMI 相机命名空间下的默认话题；显式配置的话题优先。
DEFAULT_IMAGE_TOPICS = {
    MODE_RAW: "/umi_camera/image_raw",
    MODE_COMPRESSED: "/umi_camera/image_raw/compressed",
    MODE_H264: "/umi_camera/image_raw/ffmpeg",
}
# 在线解码后发布的原始图像话题，夹爪估计和 RViz 订阅它。
DEFAULT_DECODED_IMAGE_TOPIC = "/umi_camera/image_decoded"

# 图像订阅 QoS 与 H.264 起点等待的默认值。
RELIABILITY_BEST_EFFORT = "best_effort"
RELIABILITY_RELIABLE = "reliable"
IMAGE_RELIABILITIES = (RELIABILITY_BEST_EFFORT, RELIABILITY_RELIABLE)
DEFAULT_IMAGE_RELIABILITY = RELIABILITY_BEST_EFFORT
DEFAULT_IMAGE_QOS_DEPTH = 30
DEFAULT_H264_START_TIMEOUT_S = 5.0


def validate_image_mode(mode: str) -> str:
    """规范化并校验图像模式。

    Raises:
        ValueError: 取值不是 ``raw``、``compressed`` 或 ``h264``。
    """
    normalized = str(mode).strip().lower()
    if normalized not in IMAGE_MODES:
        raise ValueError(
            f"image_mode 必须是 {'、'.join(IMAGE_MODES)} 之一，当前为 {mode!r}"
        )
    return normalized


def message_type_for_mode(mode: str) -> str:
    """返回模式对应的 ROS 2 消息类型名。"""
    return IMAGE_MESSAGE_TYPES[validate_image_mode(mode)]


def mode_for_message_type(type_name: str) -> Optional[str]:
    """返回消息类型对应的模式；不是图像类型时返回 ``None``。"""
    for mode, expected in IMAGE_MESSAGE_TYPES.items():
        if expected == type_name:
            return mode
    return None


def resolve_image_topic(mode: str, topic: str = "") -> str:
    """返回显式配置的图像话题；留空时使用该模式的默认话题。"""
    explicit = str(topic).strip()
    return explicit or DEFAULT_IMAGE_TOPICS[validate_image_mode(mode)]


def check_topic_type(topic: str, actual_type: str, mode: str) -> None:
    """要求话题的实际消息类型与图像模式一致。

    Raises:
        ValueError: 类型与模式不一致，说明里同时给出两者。
    """
    expected = message_type_for_mode(mode)
    if actual_type != expected:
        raise ValueError(
            f"话题 {topic} 的消息类型为 {actual_type}，与 image_mode="
            f"{validate_image_mode(mode)} 要求的 {expected} 不一致"
        )


def validate_image_reliability(value: str) -> str:
    """规范化并校验图像订阅可靠性。"""
    normalized = str(value).strip().lower()
    if normalized not in IMAGE_RELIABILITIES:
        raise ValueError(
            f"image_reliability 必须是 {'、'.join(IMAGE_RELIABILITIES)} 之一，"
            f"当前为 {value!r}"
        )
    return normalized


def validate_image_qos_depth(depth: int) -> int:
    """校验图像订阅队列深度为正整数。"""
    if int(depth) <= 0:
        raise ValueError(f"image_qos_depth 必须为正整数，当前为 {depth!r}")
    return int(depth)
