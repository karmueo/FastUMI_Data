"""验证 CDR 头部解析与 rclpy 序列化结果一致。"""

from fastumi_interfaces.msg import GripperState
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image

from fastumi_data.cdr_header import read_header_stamp_ns, read_image_summary


def _image(frame_id: str, width: int, height: int, step_pad: int = 0) -> Image:
    """构造带指定 frame_id 的 bgr8 测试图像。"""
    message = Image()
    message.header.stamp.sec = 1_700_000_123
    message.header.stamp.nanosec = 456_789_012
    message.header.frame_id = frame_id
    message.height = height
    message.width = width
    message.encoding = "bgr8"
    message.step = width * 3 + step_pad
    message.data = bytes(message.step * height)
    return message


def test_header_stamp_matches_for_message_types_with_header() -> None:
    """验证位姿和夹爪状态都能读到与消息一致的时间戳。"""
    expected_ns = 1_700_000_123 * 1_000_000_000 + 456_789_012
    pose = PoseStamped()
    pose.header.stamp.sec = 1_700_000_123
    pose.header.stamp.nanosec = 456_789_012
    pose.header.frame_id = "steamvr_tracking"
    gripper = GripperState()
    gripper.header.stamp = pose.header.stamp

    assert read_header_stamp_ns(serialize_message(pose)) == expected_ns
    assert read_header_stamp_ns(serialize_message(gripper)) == expected_ns


def test_image_summary_handles_frame_id_alignment_variants() -> None:
    """验证不同 frame_id 长度引入的对齐填充不会影响尺寸读取。"""
    for frame_id in ("", "a", "ab", "abc", "usb_camera_optical_frame"):
        message = _image(frame_id, width=7, height=3)
        summary = read_image_summary(serialize_message(message))

        assert summary is not None
        assert summary.stamp_ns == 1_700_000_123 * 1_000_000_000 + 456_789_012
        assert (summary.height, summary.width) == (3, 7)
        assert summary.step == 21
        assert summary.data_length == 63
        assert summary.valid


def test_image_summary_flags_inconsistent_payload_as_invalid() -> None:
    """验证空数据和长度不符的图像被判为无效。"""
    empty = _image("cam", width=4, height=2)
    empty.data = b""
    short = _image("cam", width=4, height=2)
    short.data = bytes(5)

    assert not read_image_summary(serialize_message(empty)).valid
    assert not read_image_summary(serialize_message(short)).valid


def test_truncated_data_returns_none() -> None:
    """验证被截断或过短的字节不会抛出异常。"""
    serialized = serialize_message(_image("cam", width=4, height=2))

    assert read_header_stamp_ns(b"\x00\x01") is None
    assert read_header_stamp_ns(serialized[:8]) is None
    assert read_image_summary(serialized[:20]) is None
