"""在不反序列化整条消息的前提下，从 CDR 字节读取 Header 时间戳和图像概要。

采集节点以原始字节订阅并直接写入 MCAP，避免对每帧数 MB 的图像做
Python 对象转换。这里只解析 ``std_msgs/Header`` 之后的少量字段，覆盖原始
``Image``、``CompressedImage`` 和 H.264 ``FFMPEGPacket`` 三种图像消息。
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Optional, Tuple

from fastumi_data.h264_stream import analyze_annex_b
from fastumi_data.image_modes import MODE_COMPRESSED, MODE_H264, MODE_RAW


# CDR 封装头长度：2 字节表示表示格式，2 字节选项。
_ENCAPSULATION_BYTES = 4
# 表示格式第二字节：1 表示 CDR 小端，0 表示大端。
_LITTLE_ENDIAN_FLAG = 1


@dataclass(frozen=True)
class ImageSummary:
    """保存 ``sensor_msgs/Image`` 的时间戳和有效性检查所需字段。"""

    stamp_ns: int
    """Header 源时间戳，Unix 纳秒。"""
    height: int
    width: int
    step: int
    data_length: int
    """像素数据字节数。"""

    @property
    def valid(self) -> bool:
        """尺寸为正且数据长度与 ``step * height`` 一致时认为图像有效。"""
        return (
            self.height > 0
            and self.width > 0
            and self.step > 0
            and self.data_length == self.step * self.height
        )


class _Reader:
    """按 CDR 对齐规则顺序读取定长字段的最小读取器。"""

    def __init__(self, data: bytes) -> None:
        """检查封装头并确定字节序。"""
        if len(data) < _ENCAPSULATION_BYTES:
            raise ValueError("CDR 数据短于封装头")
        self._data = data
        self._endian = "<" if data[1] == _LITTLE_ENDIAN_FLAG else ">"
        self._offset = _ENCAPSULATION_BYTES

    def _align(self, size: int) -> None:
        """相对封装头之后的起点按字段大小对齐。"""
        relative = self._offset - _ENCAPSULATION_BYTES
        self._offset += (-relative) % size

    def read(self, fmt: str) -> int:
        """读取并返回一个对齐后的整数字段。"""
        size = struct.calcsize(fmt)
        self._align(size)
        end = self._offset + size
        if end > len(self._data):
            raise ValueError("CDR 数据被截断")
        (value,) = struct.unpack_from(self._endian + fmt, self._data, self._offset)
        self._offset = end
        return int(value)

    def skip_string(self) -> None:
        """跳过一个带长度前缀和结尾空字符的 CDR 字符串。"""
        length = self.read("I")
        end = self._offset + length
        if end > len(self._data):
            raise ValueError("CDR 字符串被截断")
        self._offset = end

    def read_string(self) -> str:
        """读取一个 CDR 字符串，去掉结尾空字符。"""
        length = self.read("I")
        end = self._offset + length
        if end > len(self._data):
            raise ValueError("CDR 字符串被截断")
        text = bytes(self._data[self._offset : end]).rstrip(b"\x00")
        self._offset = end
        return text.decode("utf-8", errors="replace")

    def read_byte_sequence(self) -> Tuple[int, int]:
        """读取 ``uint8[]`` 的长度前缀并跳过内容，返回 ``(起点, 长度)``。"""
        length = self.read("I")
        start = self._offset
        if start + length > len(self._data):
            raise ValueError("CDR 字节序列被截断")
        self._offset = start + length
        return start, length


def read_header_stamp_ns(data: bytes) -> Optional[int]:
    """读取以 ``std_msgs/Header`` 开头的消息的源时间戳。

    Args:
        data: 完整 CDR 序列化字节，包含 4 字节封装头。

    Returns:
        Unix 纳秒；数据不足以解析时返回 ``None``。
    """
    try:
        reader = _Reader(data)
        seconds = reader.read("i")
        nanoseconds = reader.read("I")
    except ValueError:
        return None
    return seconds * 1_000_000_000 + nanoseconds


def read_image_summary(data: bytes) -> Optional[ImageSummary]:
    """读取 ``sensor_msgs/Image`` 的时间戳、尺寸和数据长度。

    Args:
        data: 完整 CDR 序列化字节。

    Returns:
        图像概要；数据被截断或格式异常时返回 ``None``。
    """
    try:
        reader = _Reader(data)
        seconds = reader.read("i")
        nanoseconds = reader.read("I")
        reader.skip_string()  # header.frame_id
        height = reader.read("I")
        width = reader.read("I")
        reader.skip_string()  # encoding
        reader.read("B")  # is_bigendian
        step = reader.read("I")
        data_length = reader.read("I")
    except ValueError:
        return None
    return ImageSummary(
        stamp_ns=seconds * 1_000_000_000 + nanoseconds,
        height=height,
        width=width,
        step=step,
        data_length=data_length,
    )


# 压缩图像的文件头魔数。
_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def codec_from_magic(head: bytes) -> str:
    """按文件头判断压缩图像编码，返回 ``jpeg``、``png`` 或空串。"""
    if head.startswith(_JPEG_MAGIC):
        return "jpeg"
    if head.startswith(_PNG_MAGIC):
        return "png"
    return ""


def _declared_codec(format_text: str) -> str:
    """从 ``CompressedImage.format`` 文本推断声明的编码，无法判断返回空串。"""
    lowered = format_text.lower()
    if "jpeg" in lowered or "jpg" in lowered:
        return "jpeg"
    if "png" in lowered:
        return "png"
    return ""


@dataclass(frozen=True)
class CompressedImageSummary:
    """保存 ``sensor_msgs/CompressedImage`` 的时间戳和有效性检查字段。"""

    stamp_ns: int
    format: str
    """消息声明的格式文本，例如 ``jpeg``。"""
    data_length: int
    codec: str
    """按文件头识别出的编码；无法识别为空串。"""

    @property
    def valid(self) -> bool:
        """数据非空、文件头可识别且与声明格式不矛盾时认为有效。"""
        declared = _declared_codec(self.format)
        return (
            self.data_length > 0
            and bool(self.codec)
            and (not declared or declared == self.codec)
        )

    @property
    def detail(self) -> str:
        """无效原因；有效时为空串。"""
        if self.valid:
            return ""
        if self.data_length <= 0:
            return "压缩图像数据为空"
        if not self.codec:
            return "压缩图像不是 JPEG 或 PNG"
        return f"压缩图像声明格式 {self.format!r} 与实际 {self.codec} 不一致"


def read_compressed_image_summary(data: bytes) -> Optional[CompressedImageSummary]:
    """读取 ``sensor_msgs/CompressedImage`` 的时间戳、格式和数据长度。

    Args:
        data: 完整 CDR 序列化字节。

    Returns:
        概要；数据被截断时返回 ``None``。
    """
    try:
        reader = _Reader(data)
        seconds = reader.read("i")
        nanoseconds = reader.read("I")
        reader.skip_string()  # header.frame_id
        format_text = reader.read_string()
        start, length = reader.read_byte_sequence()
    except ValueError:
        return None
    return CompressedImageSummary(
        stamp_ns=seconds * 1_000_000_000 + nanoseconds,
        format=format_text,
        data_length=length,
        codec=codec_from_magic(bytes(data[start : start + 8])),
    )


@dataclass(frozen=True)
class H264PacketSummary:
    """保存 ``FFMPEGPacket`` 的时间戳、编码信息和访问单元结构。"""

    stamp_ns: int
    width: int
    height: int
    encoding: str
    """``codec;源像素格式;...`` 形式的编码描述。"""
    pts: int
    flags: int
    data_length: int
    annex_b: bool
    keyframe: bool
    """同时含 SPS、PPS 和 IDR。"""
    has_b_slice: bool

    @property
    def codec(self) -> str:
        """编码描述的第一个分号分隔项。"""
        return self.encoding.split(";", 1)[0].strip().lower()

    @property
    def valid(self) -> bool:
        """尺寸为正、编码为 H.264 的 Annex B 数据且不含 B 帧时认为有效。"""
        return (
            self.width > 0
            and self.height > 0
            and self.data_length > 0
            and self.codec == "h264"
            and self.annex_b
            and not self.has_b_slice
        )

    @property
    def detail(self) -> str:
        """无效原因；有效时为空串。"""
        if self.valid:
            return ""
        if self.data_length <= 0:
            return "H.264 数据为空"
        if self.width <= 0 or self.height <= 0:
            return "H.264 包的图像尺寸无效"
        if self.codec != "h264":
            return f"编码为 {self.codec or '未知'}，不是 h264"
        if not self.annex_b:
            return "H.264 数据不是 Annex B 格式"
        return "H.264 包含 B 帧，协议只支持无 B 帧码流"


def read_ffmpeg_packet_summary(data: bytes) -> Optional[H264PacketSummary]:
    """读取 ``ffmpeg_image_transport_msgs/FFMPEGPacket`` 的概要并分析访问单元。

    Args:
        data: 完整 CDR 序列化字节。

    Returns:
        概要；数据被截断时返回 ``None``。
    """
    try:
        reader = _Reader(data)
        seconds = reader.read("i")
        nanoseconds = reader.read("I")
        reader.skip_string()  # header.frame_id
        width = reader.read("i")
        height = reader.read("i")
        encoding = reader.read_string()
        pts = reader.read("Q")
        flags = reader.read("B")
        reader.read("B")  # is_bigendian
        start, length = reader.read_byte_sequence()
    except ValueError:
        return None
    info = analyze_annex_b(memoryview(data)[start : start + length])
    return H264PacketSummary(
        stamp_ns=seconds * 1_000_000_000 + nanoseconds,
        width=width,
        height=height,
        encoding=encoding,
        pts=pts,
        flags=flags,
        data_length=length,
        annex_b=info.annex_b,
        keyframe=info.keyframe,
        has_b_slice=info.has_b_slice,
    )


@dataclass(frozen=True)
class ImageObservation:
    """三种图像消息统一的健康观测结果。"""

    stamp_ns: int
    valid: bool
    detail: str
    """无效原因；有效时为空串。"""
    keyframe: bool
    """仅 H.264：该包含 SPS/PPS 和 IDR，可作为解码起点。"""
    label: str
    """格式简述，例如 ``h264 1280x960``。"""


def observe_image_cdr(data: bytes, mode: str) -> Optional[ImageObservation]:
    """按图像模式解析一帧原始 CDR 字节，返回统一观测；解析失败返回 ``None``。"""
    if mode == MODE_RAW:
        raw = read_image_summary(data)
        if raw is None:
            return None
        return ImageObservation(
            stamp_ns=raw.stamp_ns,
            valid=raw.valid,
            detail="" if raw.valid else "图像尺寸或数据长度异常",
            keyframe=False,
            label=f"raw {raw.width}x{raw.height}",
        )
    if mode == MODE_COMPRESSED:
        compressed = read_compressed_image_summary(data)
        if compressed is None:
            return None
        return ImageObservation(
            stamp_ns=compressed.stamp_ns,
            valid=compressed.valid,
            detail=compressed.detail,
            keyframe=False,
            label=compressed.codec or "compressed",
        )
    if mode == MODE_H264:
        packet = read_ffmpeg_packet_summary(data)
        if packet is None:
            return None
        return ImageObservation(
            stamp_ns=packet.stamp_ns,
            valid=packet.valid,
            detail=packet.detail,
            keyframe=packet.keyframe,
            label=f"h264 {packet.width}x{packet.height}",
        )
    raise ValueError(f"未知图像模式: {mode!r}")
