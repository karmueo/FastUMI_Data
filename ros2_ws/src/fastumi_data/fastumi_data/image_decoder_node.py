"""把 JPEG/PNG 或 H.264 图像话题在线解码为 ``sensor_msgs/Image``。

``image_transport republish`` 的订阅固定为 RELIABLE，无法匹配 BEST_EFFORT 的
相机发布端（实测不会收到任何消息），因此这里用共享解码模块实现可配置 QoS
的解码节点。输出保留源 Header 的时间戳和 ``frame_id``，供夹爪估计、RViz 和
回放使用。H.264 解码失败时丢弃参考帧并等待下一个关键帧，不会退出。

``decode_every_n_frames``（默认 1，即全部解码）是可选降负载设置：JPEG/PNG 按输入
消息序号只解码 ``0, N, 2N, …``，其余在解码前跳过并计为“主动跳过”；H.264 的所有包
都必须送入解码器以维持参考帧，只按解码后的帧序号选择发布。主动跳过不是丢帧，
与解码失败分别计数。
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image

from fastumi_data.image_messages import packet_from_message
from fastumi_data.image_modes import (
    DEFAULT_DECODED_IMAGE_TOPIC,
    DEFAULT_IMAGE_QOS_DEPTH,
    DEFAULT_IMAGE_RELIABILITY,
    MODE_COMPRESSED,
    MODE_H264,
    message_type_for_mode,
    resolve_image_topic,
    validate_image_mode,
)
from fastumi_data.image_qos import image_qos_profile
from fastumi_data.video_decoding import (
    COLOR_BGR,
    VideoDecodeError,
    create_decoder,
)


# 统计日志周期，单位秒。
STATS_PERIOD_S = 5.0


class FrameDecimator:
    """按序号选择 ``0, N, 2N, …``：JPEG 用输入消息序号，H.264 用解码帧序号。"""

    def __init__(self, every_n: int) -> None:
        """``every_n`` 必须为正整数；1 表示全部选中。"""
        if isinstance(every_n, bool) or int(every_n) != every_n or every_n <= 0:
            raise ValueError(f"decode_every_n_frames 必须是正整数，当前为 {every_n!r}")
        self.every_n = int(every_n)
        self._index = 0

    def take(self) -> bool:
        """返回当前序号是否被选中，并前进一个序号。"""
        selected = self._index % self.every_n == 0
        self._index += 1
        return selected


def _stamp(stamp_ns: int) -> Time:
    """纳秒整数转 ROS Time。"""
    return Time(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)


class ImageDecoderNode(Node):
    """订阅编码图像话题，解码后发布 ``bgr8`` 原始图像。"""

    def __init__(self) -> None:
        """读取参数并创建订阅与发布。"""
        super().__init__("image_decoder")
        self.declare_parameter("image_mode", MODE_COMPRESSED)
        self.declare_parameter("input_topic", "")
        self.declare_parameter("output_topic", DEFAULT_DECODED_IMAGE_TOPIC)
        self.declare_parameter("image_reliability", DEFAULT_IMAGE_RELIABILITY)
        self.declare_parameter("image_qos_depth", DEFAULT_IMAGE_QOS_DEPTH)
        self.declare_parameter("output_queue_depth", 5)
        self.declare_parameter("decode_every_n_frames", 1)
        mode = validate_image_mode(str(self.get_parameter("image_mode").value))
        if mode not in (MODE_COMPRESSED, MODE_H264):
            raise ValueError("image_decoder 只解码 compressed 或 h264 图像")
        self._mode = mode
        input_topic = resolve_image_topic(
            mode, str(self.get_parameter("input_topic").value)
        )
        output_topic = str(self.get_parameter("output_topic").value).strip()
        if not output_topic:
            raise ValueError("output_topic 不能为空")
        self._decoder = create_decoder(mode, COLOR_BGR)
        self._decimator = FrameDecimator(
            int(self.get_parameter("decode_every_n_frames").value)
        )
        self._bridge = CvBridge()
        self._failures = 0
        self._received = 0
        self._published = 0
        self._skipped = 0
        self._total_received = 0
        self._total_published = 0
        self._total_skipped = 0
        self._last_report = time.monotonic()
        # 输出 RELIABLE，同时兼容 BEST_EFFORT（夹爪估计、RViz）与 RELIABLE 订阅。
        self._publisher = self.create_publisher(
            Image,
            output_topic,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=int(self.get_parameter("output_queue_depth").value),
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )
        self.create_subscription(
            get_message(message_type_for_mode(mode)),
            input_topic,
            self._on_message,
            image_qos_profile(
                str(self.get_parameter("image_reliability").value),
                int(self.get_parameter("image_qos_depth").value),
            ),
        )
        self.create_timer(STATS_PERIOD_S, self._report)
        self.get_logger().info(
            f"解码 {input_topic}（{mode}）-> {output_topic}，"
            f"每 {self._decimator.every_n} 帧发布一帧"
        )

    def _on_message(self, message) -> None:
        """解码一条消息并发布；损坏数据只丢弃该帧，H.264 重置并等待关键帧。"""
        self._received += 1
        self._total_received += 1
        if self._mode != MODE_H264 and not self._decimator.take():
            self._skipped += 1
            self._total_skipped += 1
            return
        try:
            frames = self._decoder.feed(packet_from_message(message, self._mode))
        except VideoDecodeError as error:
            self._failures += 1
            self._decoder.reset()
            self.get_logger().warning(
                f"解码失败，丢弃该帧{'并等待下一关键帧' if self._mode == MODE_H264 else ''}: "
                f"{error}",
                throttle_duration_sec=5.0,
            )
            return
        for frame in frames:
            if self._mode == MODE_H264 and not self._decimator.take():
                self._skipped += 1
                self._total_skipped += 1
                continue
            output = self._bridge.cv2_to_imgmsg(frame.image, encoding="bgr8")
            # 由解码帧携带的源时间与 frame_id 构造 Header，不重新打时间戳。
            output.header.stamp = _stamp(frame.stamp_ns)
            output.header.frame_id = frame.frame_id
            self._publisher.publish(output)
            self._published += 1
            self._total_published += 1

    def _report(self) -> None:
        """周期记录输入/发布速率、主动跳过数、解码失败数和 H.264 启动前缀。"""
        now = time.monotonic()
        elapsed = max(now - self._last_report, 1e-6)
        stats = self._decoder.stats
        self.get_logger().info(
            f"输入 {self._received / elapsed:.1f} fps，解码发布 "
            f"{self._published / elapsed:.1f} fps，主动跳过 {self._skipped}"
            f"（累计 {self._total_skipped}），累计失败 {self._failures}，"
            f"{stats.summary()}"
        )
        self._received = 0
        self._published = 0
        self._skipped = 0
        self._last_report = now


def main(args: Optional[Sequence[str]] = None) -> None:
    """运行解码节点，Ctrl+C 时退出。"""
    rclpy.init(args=list(args) if args is not None else None)
    node = ImageDecoderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
