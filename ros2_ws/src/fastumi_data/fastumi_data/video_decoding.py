"""JPEG/PNG 与 H.264 的共享解码状态，不依赖 ROS 消息或话题 I/O。

解码器逐包输入、以列表返回帧，并提供排空和重置：

* :class:`StillImageDecoder` 无状态，每个压缩图像对应一帧。
* :class:`H264Decoder` 维护参考帧。解码只能从同时含 SPS/PPS 和 IDR 的关键帧
  开始，此前的包被跳过并计数；协议保证无 B 帧、每包一个访问单元，因此
  每个包必须立即输出一帧。输出帧通过递增的 PTS 映射回输入包，保留源
  Header 时间、``frame_id`` 和 bag 时间。

H.264 依赖系统 ``python3-av``，仅在首次解码时导入。
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Dict, List, Optional, Protocol

import numpy as np

from fastumi_data.cdr_header import codec_from_magic
from fastumi_data.h264_stream import analyze_annex_b
from fastumi_data.image_modes import MODE_COMPRESSED, MODE_H264, MODE_RAW


# 解码输出的通道顺序。
COLOR_BGR = "bgr"
COLOR_RGB = "rgb"
COLOR_ORDERS = (COLOR_BGR, COLOR_RGB)


class VideoDecodeError(ValueError):
    """输入数据损坏、不符合协议或无法映射时间时抛出，调用方应明确失败。"""


@dataclass(frozen=True)
class EncodedPacket:
    """一个已适配的编码图像包。"""

    mode: str
    """``compressed`` 或 ``h264``。"""
    data: bytes
    stamp_ns: int
    """源 Header 时间戳，Unix 纳秒。"""
    frame_id: str = ""
    bag_timestamp_ns: int = 0
    """MCAP 日志时间；非 bag 来源为 0。"""


@dataclass(frozen=True)
class DecodedFrame:
    """一帧解码后的像素及其源时间信息。"""

    stamp_ns: int
    frame_id: str
    bag_timestamp_ns: int
    image: np.ndarray
    """``(高, 宽, 3)`` 的 uint8 数组，通道顺序见 ``color_order``。"""
    color_order: str


@dataclass
class DecodeStats:
    """解码过程的累计统计，用于报告被跳过的启动前缀。"""

    packets: int = 0
    frames: int = 0
    keyframes: int = 0
    skipped_before_keyframe: int = 0
    """首个（或重置后下一个）关键帧之前被跳过的包数。"""

    def summary(self) -> str:
        """返回一行可读统计。"""
        return (
            f"输入 {self.packets} 包，输出 {self.frames} 帧，"
            f"关键帧 {self.keyframes}，关键帧前跳过 {self.skipped_before_keyframe} 包"
        )


class ImageDecoder(Protocol):
    """压缩图像解码器接口。"""

    stats: DecodeStats

    def feed(self, packet: EncodedPacket) -> List[DecodedFrame]:
        """输入一个包，返回此时可输出的全部帧。"""

    def drain(self) -> List[DecodedFrame]:
        """文件结束时排空缓存并关闭解码器。"""

    def reset(self) -> None:
        """丢弃解码状态；视频流解码器需要等待下一个关键帧。"""


def _convert_color(image_bgr: np.ndarray, color_order: str) -> np.ndarray:
    """把 BGR 数组转换为请求的通道顺序（RGB 通过翻转通道得到连续数组）。"""
    if color_order == COLOR_BGR:
        return image_bgr
    return np.ascontiguousarray(image_bgr[:, :, ::-1])


def _check_color_order(color_order: str) -> str:
    """校验输出通道顺序。"""
    if color_order not in COLOR_ORDERS:
        raise ValueError(f"color_order 必须是 {COLOR_ORDERS} 之一")
    return color_order


class StillImageDecoder:
    """解码 JPEG/PNG 的 ``CompressedImage``，每包输出一帧。"""

    def __init__(self, color_order: str = COLOR_BGR) -> None:
        """记录输出通道顺序。"""
        self._color_order = _check_color_order(color_order)
        self.stats = DecodeStats()

    def feed(self, packet: EncodedPacket) -> List[DecodedFrame]:
        """解码一张 JPEG/PNG；文件头未知或解码失败时抛出 ``VideoDecodeError``。"""
        import cv2

        self.stats.packets += 1
        data = bytes(packet.data)
        if not codec_from_magic(data[:8]):
            raise VideoDecodeError(
                f"压缩图像（源时间 {packet.stamp_ns} ns）不是 JPEG 或 PNG"
            )
        decoded = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if decoded is None:
            raise VideoDecodeError(
                f"压缩图像（源时间 {packet.stamp_ns} ns）解码失败"
            )
        self.stats.frames += 1
        return [
            DecodedFrame(
                stamp_ns=packet.stamp_ns,
                frame_id=packet.frame_id,
                bag_timestamp_ns=packet.bag_timestamp_ns,
                image=_convert_color(decoded, self._color_order),
                color_order=self._color_order,
            )
        ]

    def drain(self) -> List[DecodedFrame]:
        """无缓存帧。"""
        return []

    def reset(self) -> None:
        """无状态，无需处理。"""


class H264Decoder:
    """无 B 帧 H.264 的有状态解码器，从关键帧开始并按包输出帧。"""

    def __init__(
        self, color_order: str = COLOR_BGR, zero_latency: bool = True
    ) -> None:
        """创建尚未打开的解码器。

        Args:
            color_order: 输出通道顺序。
            zero_latency: 为真时，已启动的解码器每输入一包必须立即输出一帧，
                否则抛出 ``VideoDecodeError``；这是转换和标定按包归属时间的前提。
        """
        self._color_order = _check_color_order(color_order)
        self._zero_latency = zero_latency
        self._context = None
        self._av = None
        self._next_sequence = 0
        self._pending: Dict[int, EncodedPacket] = {}
        self.stats = DecodeStats()

    @property
    def started(self) -> bool:
        """是否已从关键帧开始解码。"""
        return self._context is not None

    @property
    def pending(self) -> int:
        """已输入但尚未输出帧的包数。"""
        return len(self._pending)

    def _open(self):
        """创建低延迟、单线程的 H.264 软件解码上下文。"""
        try:
            import av
            from av.codec.context import Flags
        except ImportError as error:
            raise VideoDecodeError(
                "H.264 解码需要 PyAV，请安装系统依赖 python3-av"
            ) from error
        self._av = av
        context = av.CodecContext.create("h264", "r")
        context.thread_count = 1
        context.flags |= Flags.LOW_DELAY
        # 默认的错误掩盖会把损坏的 slice 悄悄补成一帧；explode 让它明确报错。
        context.options = {"err_detect": "explode"}
        context.open()
        return context

    def feed(self, packet: EncodedPacket) -> List[DecodedFrame]:
        """输入一个访问单元。

        Returns:
            已解码的帧；关键帧之前的启动前缀返回空列表并计入统计。

        Raises:
            VideoDecodeError: 不是 Annex B、含 B 帧、解码失败、无法映射时间，
                或零延迟模式下该包没有立即输出帧。
        """
        self.stats.packets += 1
        info = analyze_annex_b(packet.data)
        if not info.annex_b:
            raise VideoDecodeError(
                f"H.264 包（源时间 {packet.stamp_ns} ns）不是 Annex B 格式"
            )
        if info.has_b_slice:
            raise VideoDecodeError(
                f"H.264 包（源时间 {packet.stamp_ns} ns）含 B 帧；"
                "协议只支持无 B 帧码流"
            )
        if self._context is None:
            if not info.keyframe:
                self.stats.skipped_before_keyframe += 1
                return []
            self._context = self._open()
        if info.keyframe:
            self.stats.keyframes += 1
        sequence = self._next_sequence
        self._next_sequence += 1
        self._pending[sequence] = packet
        av_packet = self._av.Packet(bytes(packet.data))
        av_packet.pts = sequence
        av_packet.dts = sequence
        try:
            decoded = self._context.decode(av_packet)
        except self._av.error.FFmpegError as error:
            raise VideoDecodeError(
                f"H.264 包（源时间 {packet.stamp_ns} ns）解码失败: {error}"
            ) from error
        frames = self._collect(decoded)
        if self._zero_latency and self._pending:
            raise VideoDecodeError(
                f"H.264 包（源时间 {packet.stamp_ns} ns）没有立即输出图像帧：码流"
                "可能含 B 帧（协议只支持无 B 帧）、参考帧缺失或不是每包一个访问单元"
            )
        return frames

    def _collect(self, decoded) -> List[DecodedFrame]:
        """把 PyAV 帧转换为带源时间的输出，并核销对应输入包。"""
        frames = []
        for frame in decoded:
            source = self._pending.pop(frame.pts, None)
            if source is None:
                raise VideoDecodeError("解码帧的 PTS 无法映射回任何输入包的源时间")
            pixel_format = "bgr24" if self._color_order == COLOR_BGR else "rgb24"
            frames.append(
                DecodedFrame(
                    stamp_ns=source.stamp_ns,
                    frame_id=source.frame_id,
                    bag_timestamp_ns=source.bag_timestamp_ns,
                    image=frame.to_ndarray(format=pixel_format),
                    color_order=self._color_order,
                )
            )
        self.stats.frames += len(frames)
        return frames

    def drain(self) -> List[DecodedFrame]:
        """排空解码器缓存并关闭；仍有包没有对应输出帧时抛出异常。"""
        if self._context is None:
            return []
        try:
            decoded = self._context.decode(None)
        except self._av.error.FFmpegError as error:
            raise VideoDecodeError(f"H.264 排空解码器失败: {error}") from error
        frames = self._collect(decoded)
        leftover = len(self._pending)
        self._close()
        if leftover:
            raise VideoDecodeError(
                f"排空解码器后仍有 {leftover} 个 H.264 包没有对应的输出帧"
            )
        return frames

    def reset(self) -> None:
        """丢弃参考帧和待映射包；此后需要等待下一个关键帧。"""
        self._close()

    def _close(self) -> None:
        """释放解码上下文并清空待映射表。"""
        self._context = None
        self._pending.clear()


def create_decoder(mode: str, color_order: str = COLOR_BGR) -> ImageDecoder:
    """按编码图像模式创建解码器；原始图像不需要解码器。"""
    if mode == MODE_COMPRESSED:
        return StillImageDecoder(color_order)
    if mode == MODE_H264:
        return H264Decoder(color_order)
    if mode == MODE_RAW:
        raise ValueError("raw 图像不需要解码器")
    raise ValueError(f"未知图像模式: {mode!r}")


class DecodeReadiness:
    """持续喂入编码包直到首次成功解码，用于首个 episode 的启动门控。

    线程安全：订阅回调线程调用 :meth:`feed`，服务线程读取 :attr:`ready`。
    解码一旦成功就不再解码，避免为状态查询持续消耗 CPU。
    """

    def __init__(self, mode: str) -> None:
        """``raw`` 无需解码，立即就绪。"""
        self._mode = mode
        self._lock = threading.Lock()
        self._decoder: Optional[ImageDecoder] = (
            None if mode == MODE_RAW else create_decoder(mode)
        )
        self._ready = mode == MODE_RAW
        self._last_error = ""

    @property
    def ready(self) -> bool:
        """视频是否已经可以解码。"""
        with self._lock:
            return self._ready

    @property
    def status(self) -> str:
        """未就绪时的原因，供服务拒绝说明使用。"""
        with self._lock:
            if self._ready:
                return ""
            if self._last_error:
                return f"视频尚不可解码: {self._last_error}"
            if self._mode == MODE_H264:
                return "尚未收到含 SPS/PPS 的 H.264 关键帧"
            return "尚未收到可解码的图像"

    def feed(self, packet: EncodedPacket) -> None:
        """喂入一个包；解码错误记录为原因并重置，等待后续关键帧。"""
        with self._lock:
            if self._ready or self._decoder is None:
                return
            try:
                if self._decoder.feed(packet):
                    self._ready = True
                    self._last_error = ""
                    self._decoder = None
            except VideoDecodeError as error:
                self._last_error = str(error)
                self._decoder.reset()
