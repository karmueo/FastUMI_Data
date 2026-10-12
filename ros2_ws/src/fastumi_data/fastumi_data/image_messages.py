"""把 ROS 图像消息适配为共享解码器的输入包，不依赖 ROS 运行时。

适配函数只读取消息的字段（鸭子类型），因此既可处理 rclpy 反序列化得到的
对象，也可处理测试替身。
"""

from __future__ import annotations

from typing import Any

from fastumi_data.image_modes import MODE_COMPRESSED, MODE_H264
from fastumi_data.video_decoding import EncodedPacket


def stamp_to_ns(stamp: Any) -> int:
    """把 ``builtin_interfaces/Time`` 转换为 Unix 纳秒整数。"""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def packet_from_compressed_message(
    message: Any, bag_timestamp_ns: int = 0
) -> EncodedPacket:
    """适配 ``sensor_msgs/CompressedImage``。"""
    return EncodedPacket(
        mode=MODE_COMPRESSED,
        data=bytes(message.data),
        stamp_ns=stamp_to_ns(message.header.stamp),
        frame_id=str(message.header.frame_id),
        bag_timestamp_ns=int(bag_timestamp_ns),
    )


def packet_from_ffmpeg_message(
    message: Any, bag_timestamp_ns: int = 0
) -> EncodedPacket:
    """适配 ``ffmpeg_image_transport_msgs/FFMPEGPacket``。"""
    return EncodedPacket(
        mode=MODE_H264,
        data=bytes(message.data),
        stamp_ns=stamp_to_ns(message.header.stamp),
        frame_id=str(message.header.frame_id),
        bag_timestamp_ns=int(bag_timestamp_ns),
    )


def packet_from_message(
    message: Any, mode: str, bag_timestamp_ns: int = 0
) -> EncodedPacket:
    """按图像模式适配压缩消息；``raw`` 不产生编码包。"""
    if mode == MODE_COMPRESSED:
        return packet_from_compressed_message(message, bag_timestamp_ns)
    if mode == MODE_H264:
        return packet_from_ffmpeg_message(message, bag_timestamp_ns)
    raise ValueError(f"模式 {mode!r} 没有编码包适配器")
