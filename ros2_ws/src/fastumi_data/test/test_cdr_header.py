"""验证 CDR 头部解析与 rclpy 序列化结果一致。"""

import struct

from fastumi_interfaces.msg import GripperState
from geometry_msgs.msg import PoseStamped
import pytest
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image

from fastumi_data.cdr_header import (
    observe_image_cdr,
    read_compressed_image_summary,
    read_ffmpeg_packet_summary,
    read_header_stamp_ns,
    read_image_summary,
)
from video_fixtures import (
    compressed_message,
    ffmpeg_message,
    h264_packets,
    stamp_of,
)


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


# ---------------------------------------------------------------------------
# CompressedImage 与 FFMPEGPacket。
# ---------------------------------------------------------------------------

class _CdrWriter:
    """按 CDR 对齐规则手工序列化字段，用于生成大端数据。"""

    def __init__(self, big_endian: bool) -> None:
        """写入 4 字节封装头：0 表示大端，1 表示小端。"""
        self._order = ">" if big_endian else "<"
        self._buffer = bytearray(b"\x00\x00\x00\x00" if big_endian else b"\x00\x01\x00\x00")

    def _align(self, size: int) -> None:
        """相对封装头之后的起点补齐到 ``size`` 的倍数。"""
        self._buffer.extend(b"\x00" * ((-(len(self._buffer) - 4)) % size))

    def number(self, fmt: str, value: int) -> None:
        """对齐后写入一个定长整数。"""
        self._align(struct.calcsize(fmt))
        self._buffer.extend(struct.pack(self._order + fmt, value))

    def text(self, value: str) -> None:
        """写入带长度前缀和结尾空字符的字符串。"""
        encoded = value.encode() + b"\x00"
        self.number("I", len(encoded))
        self._buffer.extend(encoded)

    def octets(self, value: bytes) -> None:
        """写入 ``uint8[]``。"""
        self.number("I", len(value))
        self._buffer.extend(value)

    def bytes(self) -> bytes:
        """返回完整 CDR 字节。"""
        return bytes(self._buffer)


def _header(writer: _CdrWriter, index: int, frame_id: str) -> None:
    """写入 ``std_msgs/Header``。"""
    stamp = stamp_of(index)
    writer.number("i", stamp // 1_000_000_000)
    writer.number("I", stamp % 1_000_000_000)
    writer.text(frame_id)


def test_compressed_summary_matches_rclpy_serialization() -> None:
    """JPEG 与 PNG 的概要读取结果与 rclpy 序列化一致，frame_id 对齐不影响解析。"""
    for codec in ("jpeg", "png"):
        for frame_id in ("", "a", "abc", "usb_camera_optical_frame"):
            message = compressed_message(3, codec)
            message.header.frame_id = frame_id
            summary = read_compressed_image_summary(serialize_message(message))

            assert summary is not None
            assert summary.stamp_ns == stamp_of(3)
            assert summary.codec == codec
            assert summary.data_length == len(message.data)
            assert summary.valid


def test_compressed_summary_flags_bad_payloads() -> None:
    """空数据、未知文件头和与声明矛盾的格式都判为无效。"""
    empty = compressed_message(0)
    empty.data = b""
    garbage = compressed_message(0)
    garbage.data = b"not an image at all"
    contradictory = compressed_message(0, "png")
    contradictory.format = "jpeg"

    assert not read_compressed_image_summary(serialize_message(empty)).valid
    assert not read_compressed_image_summary(serialize_message(garbage)).valid
    assert not read_compressed_image_summary(serialize_message(contradictory)).valid
    assert read_compressed_image_summary(serialize_message(garbage)).detail


def test_ffmpeg_summary_matches_rclpy_serialization() -> None:
    """FFMPEGPacket 概要的尺寸、PTS、关键帧和有效性与消息一致。"""
    packets = h264_packets(12, gop_size=10)
    for index, data in enumerate(packets[:12]):
        message = ffmpeg_message(index, data, pts=1000 + index)
        message.header.frame_id = "frame" * index
        summary = read_ffmpeg_packet_summary(serialize_message(message))

        assert summary is not None
        assert summary.stamp_ns == stamp_of(index)
        assert (summary.width, summary.height) == (64, 48)
        assert summary.pts == 1000 + index
        assert summary.codec == "h264"
        assert summary.data_length == len(data)
        assert summary.keyframe == (index in (0, 10))
        assert summary.valid


def test_ffmpeg_summary_flags_b_frames_and_wrong_codec() -> None:
    """B 帧码流和非 h264 编码的包无效，原因可读。"""
    b_packets = h264_packets(12, b_frames=2)
    summaries = [
        read_ffmpeg_packet_summary(serialize_message(ffmpeg_message(i, data)))
        for i, data in enumerate(b_packets)
    ]
    assert any(not s.valid and "B 帧" in s.detail for s in summaries)

    hevc = ffmpeg_message(0, h264_packets(1)[0])
    hevc.encoding = "hevc;bgr8;bgr8;bgr8"
    assert "hevc" in read_ffmpeg_packet_summary(serialize_message(hevc)).detail


def test_big_endian_cdr_is_parsed() -> None:
    """大端封装的 CompressedImage 和 FFMPEGPacket 解析结果与小端一致。"""
    payload = compressed_message(2).data
    h264 = h264_packets(1)[0]
    for big_endian in (False, True):
        writer = _CdrWriter(big_endian)
        _header(writer, 2, "cam")
        writer.text("jpeg")
        writer.octets(bytes(payload))
        compressed = read_compressed_image_summary(writer.bytes())
        assert compressed is not None
        assert compressed.stamp_ns == stamp_of(2)
        assert compressed.format == "jpeg"
        assert compressed.data_length == len(payload)
        assert compressed.valid

        writer = _CdrWriter(big_endian)
        _header(writer, 5, "cam")
        writer.number("i", 64)
        writer.number("i", 48)
        writer.text("h264;bgr8;bgr8;bgr8")
        writer.number("Q", 777)
        writer.number("B", 1)
        writer.number("B", int(big_endian))
        writer.octets(h264)
        packet = read_ffmpeg_packet_summary(writer.bytes())
        assert packet is not None
        assert packet.stamp_ns == stamp_of(5)
        assert (packet.width, packet.height, packet.pts, packet.flags) == (64, 48, 777, 1)
        assert packet.keyframe and packet.valid


def test_truncated_encoded_messages_return_none() -> None:
    """任意位置截断的压缩图像和 H.264 消息都返回 None 而不抛异常。"""
    compressed = serialize_message(compressed_message(1))
    ffmpeg = serialize_message(ffmpeg_message(1, h264_packets(1)[0]))

    for cut in (0, 3, 8, 14, 20, len(compressed) - 1):
        assert read_compressed_image_summary(compressed[:cut]) is None
    for cut in (0, 3, 8, 14, 24, 40, len(ffmpeg) - 1):
        assert read_ffmpeg_packet_summary(ffmpeg[:cut]) is None


def test_observe_image_cdr_dispatches_by_mode() -> None:
    """统一观测按模式返回时间戳、有效性和关键帧标记。"""
    raw = observe_image_cdr(serialize_message(_image("cam", width=4, height=2)), "raw")
    assert raw.valid and not raw.keyframe and raw.label == "raw 4x2"

    jpeg = observe_image_cdr(serialize_message(compressed_message(4)), "compressed")
    assert jpeg.valid and jpeg.stamp_ns == stamp_of(4) and jpeg.label == "jpeg"

    key = observe_image_cdr(
        serialize_message(ffmpeg_message(0, h264_packets(2)[0])), "h264"
    )
    delta = observe_image_cdr(
        serialize_message(ffmpeg_message(1, h264_packets(2)[1])), "h264"
    )
    assert key.keyframe and key.label == "h264 64x48"
    assert delta.valid and not delta.keyframe

    assert observe_image_cdr(b"\x00\x01", "h264") is None
    with pytest.raises(ValueError, match="未知图像模式"):
        observe_image_cdr(b"", "mjpeg")
