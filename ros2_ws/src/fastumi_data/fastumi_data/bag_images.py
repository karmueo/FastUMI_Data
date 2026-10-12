"""把 MCAP 中的原始、压缩或 H.264 图像消息解码为带源时间的像素帧。

这是共享解码状态（:mod:`fastumi_data.video_decoding`）与 rosbag2 之间的 I/O
层：负责反序列化消息、选择解码器并校验话题类型。转换、标定和回放都经过它，
因此三种模式的帧归属和错误行为一致。
"""

from __future__ import annotations

from typing import List, Mapping

from cv_bridge import CvBridge
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from fastumi_data.image_messages import packet_from_message, stamp_to_ns
from fastumi_data.image_modes import (
    MODE_RAW,
    check_topic_type,
    message_type_for_mode,
    validate_image_mode,
)
from fastumi_data.video_decoding import (
    COLOR_BGR,
    COLOR_ORDERS,
    DecodedFrame,
    DecodeStats,
    create_decoder,
)


def check_bag_image_topic(
    topic_types: Mapping[str, str], topic: str, mode: str
) -> None:
    """要求 bag 中存在图像话题且其消息类型与图像模式一致。

    Raises:
        ValueError: 话题缺失，或消息类型与 ``mode`` 不一致。
    """
    if topic not in topic_types:
        raise ValueError(f"MCAP 缺少话题: {topic}")
    check_topic_type(topic, topic_types[topic], mode)


class BagImageDecoder:
    """按图像模式把 bag 中的序列化图像消息解码为帧。

    ``raw`` 通过 ``cv_bridge`` 转换，``compressed`` 和 ``h264`` 使用共享
    解码器。解码状态（H.264 参考帧）跨调用保留，调用方应按 bag 顺序喂入
    话题上的每一条消息，包括 episode 之外的消息。
    """

    def __init__(self, mode: str, color_order: str = COLOR_BGR) -> None:
        """创建指定模式和输出通道顺序的解码器。"""
        if color_order not in COLOR_ORDERS:
            raise ValueError(f"color_order 必须是 {COLOR_ORDERS} 之一")
        self._mode = validate_image_mode(mode)
        self._color_order = color_order
        self._message_class = get_message(message_type_for_mode(self._mode))
        self._bridge = CvBridge() if self._mode == MODE_RAW else None
        self._decoder = (
            None if self._mode == MODE_RAW
            else create_decoder(self._mode, color_order)
        )
        self._raw_stats = DecodeStats()

    @property
    def mode(self) -> str:
        """图像模式。"""
        return self._mode

    @property
    def stats(self) -> DecodeStats:
        """累计解码统计，含关键帧之前被跳过的包数。"""
        return self._raw_stats if self._decoder is None else self._decoder.stats

    def feed(self, serialized: bytes, bag_timestamp_ns: int) -> List[DecodedFrame]:
        """解码一条序列化图像消息，返回此时可输出的帧。"""
        message = deserialize_message(bytes(serialized), self._message_class)
        if self._decoder is None:
            return [self._decode_raw(message, bag_timestamp_ns)]
        packet = packet_from_message(message, self._mode, bag_timestamp_ns)
        return self._decoder.feed(packet)

    def _decode_raw(self, message, bag_timestamp_ns: int) -> DecodedFrame:
        """用 cv_bridge 把原始图像转换为请求的通道顺序。"""
        encoding = "bgr8" if self._color_order == COLOR_BGR else "rgb8"
        pixels = self._bridge.imgmsg_to_cv2(message, desired_encoding=encoding)
        self._raw_stats.packets += 1
        self._raw_stats.frames += 1
        return DecodedFrame(
            stamp_ns=stamp_to_ns(message.header.stamp),
            frame_id=str(message.header.frame_id),
            bag_timestamp_ns=int(bag_timestamp_ns),
            image=np.asarray(pixels).copy(),
            color_order=self._color_order,
        )

    def drain(self) -> List[DecodedFrame]:
        """文件结束时排空解码器缓存。"""
        return [] if self._decoder is None else self._decoder.drain()
