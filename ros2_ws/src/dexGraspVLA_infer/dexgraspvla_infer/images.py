"""构造 ROS 图像时使用字节缓冲，避免 uint8 序列 setter 的逐像素检查。"""

from array import array
import struct

import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Header


def image_message(pixels, encoding, header):
    """uint8 mono8/bgr8 数组转 Image，保留源 header 与逐字节内容。"""
    pixels = np.asarray(pixels)
    expected = 2 if encoding == "mono8" else 3
    if encoding not in ("mono8", "bgr8") or pixels.dtype != np.uint8 or pixels.ndim != expected:
        raise ValueError("image must be uint8 mono8 or bgr8")
    if encoding == "bgr8" and pixels.shape[2] != 3:
        raise ValueError("BGR image needs three channels")
    message = Image()
    message.header = header
    message.height, message.width = pixels.shape[:2]
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = message.width * (1 if encoding == "mono8" else 3)
    message.data = array("B", pixels.tobytes())
    return message


def serialized_image(data):
    """解析 Humble CDR1 Image 为 BGR view，避免生成数百万个 Python 整数。

    原始 header 保持不变；支持 uint8 bgr8/rgb8/mono8 及行填充。
    未知封装、尺寸和截断数据均拒绝，不猜测布局。
    """
    data = memoryview(data)
    if len(data) < 4 or bytes(data[:2]) not in (b"\x00\x00", b"\x00\x01"):
        raise ValueError("unsupported Image CDR encapsulation (expected CDR1)")
    endian = "<" if data[1] == 1 else ">"
    offset = 4

    def number(code):
        nonlocal offset
        offset = (offset + 3) & ~3
        if offset + 4 > len(data):
            raise ValueError("truncated Image CDR scalar")
        value = struct.unpack_from(endian + code, data, offset)[0]
        offset += 4
        return value

    def string():
        nonlocal offset
        length = number("I")
        if length < 1 or offset + length > len(data) or data[offset + length - 1] != 0:
            raise ValueError("invalid Image CDR string")
        value = bytes(data[offset:offset + length - 1]).decode("utf-8")
        offset += length
        return value

    sec, nanosec = number("i"), number("I")
    frame = string()
    height, width = number("I"), number("I")
    encoding = string()
    if offset >= len(data):
        raise ValueError("truncated Image endianness flag")
    offset += 1
    step, size = number("I"), number("I")
    channels = 1 if encoding == "mono8" else 3
    if (encoding not in ("mono8", "rgb8", "bgr8") or not 0 <= nanosec < 10**9
            or height == 0 or width == 0 or step < width * channels
            or size != height * step or offset + size > len(data)):
        raise ValueError("invalid Image encoding, dimensions or buffer length")
    pixels = np.ndarray((height, width, channels), np.uint8, buffer=data, offset=offset,
                        strides=(step, channels, 1))
    if encoding == "rgb8":
        pixels = pixels[:, :, ::-1]
    elif encoding == "mono8":
        pixels = np.repeat(pixels, 3, axis=2)
    header = Header(frame_id=frame)
    header.stamp.sec, header.stamp.nanosec = sec, nanosec
    return header, pixels
