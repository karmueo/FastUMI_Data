"""验证在线解码节点：BEST_EFFORT 发布端可匹配，Header 保持，H.264 失败后等关键帧恢复。"""

import os
import random
import time

# 在导入 rclpy 前隔离 DDS，避免与真实设备或并行测试串话。
os.environ["ROS_DOMAIN_ID"] = str(random.randint(150, 232))
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import rclpy  # noqa: E402
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import CompressedImage, Image  # noqa: E402

from fastumi_data.image_decoder_node import ImageDecoderNode  # noqa: E402
from video_fixtures import (  # noqa: E402
    band_column,
    compressed_message,
    ffmpeg_message,
    frame_pixels_bgr,
    h264_packets,
    stamp_of,
)


class _Rig:
    """同进程的解码节点、BEST_EFFORT 发布端和输出订阅者。"""

    def __init__(self, mode: str, input_topic: str, message_type) -> None:
        """用命令行参数配置解码节点并创建收发两端。"""
        rclpy.init(args=[
            "--ros-args", "-p", f"image_mode:={mode}",
            "-p", f"input_topic:={input_topic}", "-p", "output_topic:=/test/decoded",
        ])
        self.node = ImageDecoderNode()
        self.helper = rclpy.create_node("decoder_test_helper")
        self.publisher = self.helper.create_publisher(
            message_type, input_topic,
            QoSProfile(depth=30, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.frames = []
        self.helper.create_subscription(
            Image, "/test/decoded", self.frames.append,
            QoSProfile(depth=30, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.helper)
        deadline = time.monotonic() + 10
        while self.publisher.get_subscription_count() == 0:
            assert time.monotonic() < deadline, "解码节点没有匹配 BEST_EFFORT 发布端"
            self.executor.spin_once(timeout_sec=0.05)

    def send(self, message, settle_s: float = 0.03) -> None:
        """发布一条消息并让节点处理。"""
        self.publisher.publish(message)
        end = time.monotonic() + settle_s
        while time.monotonic() < end:
            self.executor.spin_once(timeout_sec=0.01)

    def wait_frames(self, count: int) -> None:
        """等待输出帧数达到 ``count``。"""
        deadline = time.monotonic() + 5
        while len(self.frames) < count:
            assert time.monotonic() < deadline, f"只收到 {len(self.frames)} 帧"
            self.executor.spin_once(timeout_sec=0.05)

    def close(self) -> None:
        """销毁节点并关闭 rclpy。"""
        self.executor.shutdown()
        self.node.destroy_node()
        self.helper.destroy_node()
        rclpy.shutdown()


@pytest.fixture()
def jpeg_rig():
    """JPEG 解码台架。"""
    rig = _Rig("compressed", "/test/compressed", CompressedImage)
    yield rig
    rig.close()


@pytest.fixture()
def h264_rig():
    """H.264 解码台架。"""
    rig = _Rig("h264", "/test/ffmpeg", FFMPEGPacket)
    yield rig
    rig.close()


def test_jpeg_is_decoded_to_bgr8_with_original_header(jpeg_rig) -> None:
    """JPEG 解码为 bgr8，Header 的时间戳和 frame_id 原样保留。"""
    for index in range(3):
        jpeg_rig.send(compressed_message(index))
    jpeg_rig.wait_frames(3)

    frame = jpeg_rig.frames[2]
    assert frame.encoding == "bgr8" and (frame.width, frame.height) == (64, 48)
    stamp = frame.header.stamp
    assert stamp.sec * 1_000_000_000 + stamp.nanosec == stamp_of(2)
    assert frame.header.frame_id == "cam"
    pixels = bytes(frame.data)
    assert pixels[2] < 100 and pixels[0] > 150  # 左上角蓝色底（BGR 第 0 通道）。


def test_corrupt_jpeg_is_dropped_without_stopping(jpeg_rig) -> None:
    """损坏的 JPEG 只丢弃该帧，后续帧继续输出。"""
    bad = compressed_message(0)
    bad.data = bytes(bad.data)[:20]
    jpeg_rig.send(bad)
    jpeg_rig.send(compressed_message(1))
    jpeg_rig.wait_frames(1)

    assert len(jpeg_rig.frames) == 1


def test_h264_waits_for_keyframe_and_recovers_after_corruption(h264_rig) -> None:
    """H.264：关键帧前的包被跳过；损坏后重置并等待下一关键帧再恢复输出。"""
    packets = h264_packets(30, gop_size=10)
    for index in range(4, 13):  # 从 GOP 中段接入，关键帧在第 10 帧。
        h264_rig.send(ffmpeg_message(index, packets[index]))
    h264_rig.wait_frames(3)
    assert [
        f.header.stamp.sec * 1_000_000_000 + f.header.stamp.nanosec
        for f in h264_rig.frames
    ] == [stamp_of(10), stamp_of(11), stamp_of(12)]
    first = np.frombuffer(bytes(h264_rig.frames[0].data), np.uint8).reshape(48, 64, 3)
    assert abs(band_column(first) - band_column(frame_pixels_bgr(10))) <= 2

    data = packets[13]
    h264_rig.send(ffmpeg_message(13, data[:12] + bytes(len(data) - 12)))
    for index in (14, 15, 16):  # 重置后的非关键帧不输出。
        h264_rig.send(ffmpeg_message(index, packets[index]))
    assert len(h264_rig.frames) == 3
    for index in range(20, 23):  # 下一关键帧恢复。
        h264_rig.send(ffmpeg_message(index, packets[index]))
    h264_rig.wait_frames(6)
    assert h264_rig.frames[3].header.stamp.sec * 1_000_000_000 + h264_rig.frames[
        3
    ].header.stamp.nanosec == stamp_of(20)


def test_unsupported_mode_is_rejected() -> None:
    """raw 不需要解码节点，启动时明确失败。"""
    rclpy.init(args=["--ros-args", "-p", "image_mode:=raw"])
    try:
        with pytest.raises(ValueError, match="compressed 或 h264"):
            ImageDecoderNode()
    finally:
        rclpy.shutdown()
