"""生成合成的 JPEG/PNG/H.264 图像消息，供压缩视频相关测试共用。

每帧是纯色底加一条随帧号移动的色带，因此解码后可以从像素读回帧号，
用来核对时间映射、抽帧和 RGB/BGR 通道顺序。
"""

from __future__ import annotations

from fractions import Fraction
from typing import List

import cv2
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
import numpy as np
from sensor_msgs.msg import CompressedImage, Image

# 合成图像尺寸，宽高均为 16 的倍数以避免编码器填充。
WIDTH = 64
HEIGHT = 48
# 合成帧间隔，约 30 Hz。
FRAME_PERIOD_NS = 33_333_333
# 第 0 帧的源时间戳。
BASE_STAMP_NS = 1_700_000_000_000_000_000


def frame_pixels_bgr(index: int) -> np.ndarray:
    """返回第 ``index`` 帧的 BGR 像素：蓝色底，红色带随帧号右移。"""
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, :] = (200, 40, 40)  # B, G, R 的底色偏蓝。
    column = (index * 4) % (WIDTH - 8)
    image[:, column : column + 8] = (30, 30, 230)  # 红色带。
    return image


def band_column(image_bgr: np.ndarray) -> int:
    """从解码图像读回红色带的起始列，供核对帧号。"""
    red = image_bgr[HEIGHT // 2, :, 2].astype(int)
    return int(np.argmax(red > 150))


def stamp_of(index: int) -> int:
    """第 ``index`` 帧的源时间戳（纳秒）。"""
    return BASE_STAMP_NS + index * FRAME_PERIOD_NS


def _set_header(message, index: int, frame_id: str = "cam") -> None:
    """写入第 ``index`` 帧的 Header。"""
    stamp = stamp_of(index)
    message.header.stamp.sec = stamp // 1_000_000_000
    message.header.stamp.nanosec = stamp % 1_000_000_000
    message.header.frame_id = frame_id


def raw_message(index: int) -> Image:
    """构造第 ``index`` 帧的 bgr8 原始图像。"""
    pixels = frame_pixels_bgr(index)
    message = Image()
    _set_header(message, index)
    message.height, message.width = HEIGHT, WIDTH
    message.encoding = "bgr8"
    message.step = WIDTH * 3
    message.data = pixels.tobytes()
    return message


def compressed_message(index: int, codec: str = "jpeg") -> CompressedImage:
    """构造第 ``index`` 帧的 JPEG 或 PNG 压缩图像。"""
    extension = ".jpg" if codec == "jpeg" else ".png"
    ok, encoded = cv2.imencode(extension, frame_pixels_bgr(index))
    assert ok
    message = CompressedImage()
    _set_header(message, index)
    message.format = codec
    message.data = encoded.tobytes()
    return message


def h264_packets(
    count: int, gop_size: int = 10, b_frames: int = 0
) -> List[bytes]:
    """编码 ``count`` 帧，返回每个访问单元的 Annex B 字节。

    ``b_frames`` 大于 0 时生成含 B 帧的码流，用于验证拒绝路径。
    """
    import av

    context = av.CodecContext.create("libx264", "w")
    context.width, context.height, context.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    context.time_base = Fraction(1, 30)
    context.gop_size = gop_size
    if b_frames:
        context.options = {
            "preset": "medium",
            "profile": "main",
            "x264-params": f"bframes={b_frames}:b-adapt=0:scenecut=0",
        }
    else:
        context.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            "x264-params": "scenecut=0",
        }
    context.open()
    packets: List[bytes] = []
    for index in range(count):
        frame = av.VideoFrame.from_ndarray(
            frame_pixels_bgr(index), format="bgr24"
        )
        frame.pts = index
        packets.extend(bytes(packet) for packet in context.encode(frame))
    packets.extend(bytes(packet) for packet in context.encode(None))
    return packets


def ffmpeg_message(index: int, data: bytes, pts: int | None = None) -> FFMPEGPacket:
    """把一个访问单元包装为 ``FFMPEGPacket``。"""
    message = FFMPEGPacket()
    _set_header(message, index)
    message.width, message.height = WIDTH, HEIGHT
    message.encoding = "h264;bgr8;bgr8;bgr8"
    message.pts = index if pts is None else pts
    message.flags = 0
    message.is_bigendian = False
    message.data = data
    return message
