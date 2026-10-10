"""在不反序列化整条消息的前提下，从 CDR 字节读取 Header 时间戳和图像概要。

采集节点以原始字节订阅并直接写入 MCAP，避免对每帧数 MB 的图像做
Python 对象转换。这里只解析 ``std_msgs/Header`` 之后的少量定长字段。
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Optional


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
