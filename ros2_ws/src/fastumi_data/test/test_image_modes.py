"""验证图像模式、消息类型映射和显式话题解析。"""

import pytest

from fastumi_data.image_modes import (
    DEFAULT_IMAGE_TOPICS,
    check_topic_type,
    message_type_for_mode,
    mode_for_message_type,
    resolve_image_topic,
    validate_image_mode,
    validate_image_qos_depth,
    validate_image_reliability,
)


def test_modes_map_to_message_types_and_back() -> None:
    """三种模式与三种消息类型一一对应。"""
    for mode, type_name in (
        ("raw", "sensor_msgs/msg/Image"),
        ("compressed", "sensor_msgs/msg/CompressedImage"),
        ("h264", "ffmpeg_image_transport_msgs/msg/FFMPEGPacket"),
    ):
        assert message_type_for_mode(mode) == type_name
        assert mode_for_message_type(type_name) == mode
    assert mode_for_message_type("std_msgs/msg/String") is None


def test_mode_is_normalized_and_validated() -> None:
    """模式忽略大小写和空白，未知取值被拒绝。"""
    assert validate_image_mode(" H264 ") == "h264"
    with pytest.raises(ValueError, match="image_mode"):
        validate_image_mode("mjpeg")


def test_topic_defaults_follow_mode_and_explicit_topic_wins() -> None:
    """留空时使用模式默认话题，显式话题原样保留。"""
    assert resolve_image_topic("raw") == "/umi_camera/image_raw"
    assert resolve_image_topic("compressed", "") == DEFAULT_IMAGE_TOPICS["compressed"]
    assert resolve_image_topic("h264", "/jetson/camera/ffmpeg") == "/jetson/camera/ffmpeg"


def test_topic_type_mismatch_names_both_types() -> None:
    """话题类型与模式不一致时说明实际类型和期望类型。"""
    check_topic_type("/cam", "sensor_msgs/msg/CompressedImage", "compressed")
    with pytest.raises(ValueError, match="CompressedImage.*image_mode=h264.*FFMPEGPacket"):
        check_topic_type("/cam", "sensor_msgs/msg/CompressedImage", "h264")


def test_qos_values_are_validated() -> None:
    """可靠性和队列深度只接受文档中的取值。"""
    assert validate_image_reliability("Reliable") == "reliable"
    with pytest.raises(ValueError, match="image_reliability"):
        validate_image_reliability("lossy")
    assert validate_image_qos_depth(30) == 30
    with pytest.raises(ValueError, match="image_qos_depth"):
        validate_image_qos_depth(0)
