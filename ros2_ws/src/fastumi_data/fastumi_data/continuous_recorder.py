"""用共享订阅连续录制编码图像，并依据已写入 MCAP 的消息门控 episode。

本模块负责 ROS 订阅、话题发现和 MCAP 写入。边界服务由 EpisodeManager
提供，通过同步事件回调及共享 DecodeReadiness 协作。写包成功后才更新图像
就绪状态，避免管理节点收到关键帧而独立录包订阅漏收时提前接受 START。
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from fastumi_interfaces.msg import EpisodeEvent
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message, serialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import yaml

from fastumi_data.episode_manager import EpisodeManager
from fastumi_data.image_messages import packet_from_message
from fastumi_data.image_modes import (
    DEFAULT_IMAGE_MODE,
    DEFAULT_IMAGE_QOS_DEPTH,
    DEFAULT_IMAGE_RELIABILITY,
    MODE_RAW,
    message_type_for_mode,
    resolve_image_topic,
    validate_image_mode,
)
from fastumi_data.image_qos import image_qos_profile
from fastumi_data.video_decoding import DecodeReadiness


# EpisodeManager 直接交给写入器的边界事件话题，避免重复订阅写入。
EVENT_TOPIC = "/fastumi/episode/events"
# 等待其他设备启动时的话题发现周期，单位秒。
DISCOVERY_PERIOD_S = 0.2


def _storage_qos_profiles(profiles: Sequence[QoSProfile]):
    """把发布端 QoS 转成 rosbag2 元数据，保留静态 TF 的持久性等回放信息。"""
    document = []  # 版本 8 QoS YAML 使用整数枚举，转换结果由 rosbag2 按当前格式保存。
    for profile in profiles:
        fields = {
            name: int(getattr(profile, name))
            for name in ("history", "depth", "reliability", "durability", "liveliness")
        }
        for name in ("lifespan", "deadline", "liveliness_lease_duration"):
            seconds, nanoseconds = divmod(getattr(profile, name).nanoseconds, 1_000_000_000)
            fields[name] = {"sec": seconds, "nsec": nanoseconds}
        fields["avoid_ros_namespace_conventions"] = profile.avoid_ros_namespace_conventions
        document.append(fields)
    return rosbag2_py.to_rclcpp_qos_vector(yaml.safe_dump(document), 8)


class ContinuousRecorder(Node):
    """连续记录选定话题，并共享实际录包图像的解码就绪状态。

    图像回调、事件服务和发现定时器须运行在同一个单线程执行器中，保证
    SequentialWriter 的线程安全和日志顺序。其他话题启动后动态建立订阅。
    """

    def __init__(self, bag_uri: str, topics: List[str]) -> None:
        """读取图像参数，打开 MCAP，并建立唯一的图像订阅。

        Args:
            bag_uri: 尚不存在的 bag 目录。
            topics: 需要记录的绝对话题名，必须包含图像话题。
        Raises:
            ValueError: 图像配置或话题列表无效。
            RuntimeError: MCAP 无法打开。
        """
        super().__init__("continuous_recorder")
        self.declare_parameter("image_mode", DEFAULT_IMAGE_MODE)
        self.declare_parameter("image_topic", "")
        self.declare_parameter("image_reliability", DEFAULT_IMAGE_RELIABILITY)
        self.declare_parameter("image_qos_depth", DEFAULT_IMAGE_QOS_DEPTH)
        # 解析后的图像模式、话题和消息类。
        self._mode = validate_image_mode(str(self.get_parameter("image_mode").value))
        self._image_topic = resolve_image_topic(
            self._mode, str(self.get_parameter("image_topic").value)
        )
        if self._mode == MODE_RAW:
            raise ValueError("连续编码录制器只支持 compressed 或 h264")
        if self._image_topic not in topics:
            raise ValueError("录制话题必须包含所选图像话题")
        image_qos = image_qos_profile(
            str(self.get_parameter("image_reliability").value),
            int(self.get_parameter("image_qos_depth").value),
        )
        self._image_class = get_message(message_type_for_mode(self._mode))
        # 就绪状态只由成功写入的图像消息更新。
        self.readiness = DecodeReadiness(self._mode)
        # 选定话题、已注册类型和订阅，确保每个话题只创建一次。
        self._topics = set(topics)
        self._types = {}
        self._subscriptions_by_topic = {}
        self._last_log_ns = 0  # 最近一次 MCAP 日志时间，保持非递减。
        self._error: Optional[Exception] = None  # 首次写入故障，之后明确停止。
        self._writer = rosbag2_py.SequentialWriter()  # 本执行器独占的 MCAP 写入器。
        self._writer.open(
            rosbag2_py.StorageOptions(
                uri=bag_uri, storage_id="mcap", storage_preset_profile="zstd_fast"
            ),
            rosbag2_py.ConverterOptions("", ""),
        )
        self._register_topic(
            self._image_topic, message_type_for_mode(self._mode), [image_qos]
        )
        if EVENT_TOPIC in self._topics:
            self._register_topic(
                EVENT_TOPIC, "fastumi_interfaces/msg/EpisodeEvent",
                [QoSProfile(
                    depth=20, reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                )],
            )
        self._subscriptions_by_topic[self._image_topic] = self.create_subscription(
            self._image_class, self._image_topic, self._on_image, image_qos, raw=True
        )
        self._discovery_timer = self.create_timer(
            DISCOVERY_PERIOD_S, self._discover_topics
        )

    def _register_topic(
        self, topic: str, type_name: str, qos_profiles: Sequence[QoSProfile] = (),
    ) -> None:
        """在 MCAP 中注册类型和 QoS；同一话题出现不同类型时明确失败。"""
        if topic in self._types:
            if self._types[topic] != type_name:
                raise ValueError(f"话题 {topic} 的消息类型发生变化")
            return
        self._writer.create_topic(
            rosbag2_py.TopicMetadata(
                id=len(self._types), name=topic, type=type_name,
                serialization_format="cdr",
                offered_qos_profiles=_storage_qos_profiles(qos_profiles),
            )
        )
        self._types[topic] = type_name

    def _discover_topics(self) -> None:
        """发现尚未订阅的设备话题，按发布端可靠性和持久性适配 QoS。"""
        if self._error is not None:
            raise RuntimeError(f"连续录制写入失败: {self._error}") from self._error
        known = dict(self.get_topic_names_and_types())
        for topic in sorted(self._topics - self._subscriptions_by_topic.keys()):
            if topic == EVENT_TOPIC:
                continue  # 本进程的边界事件通过同步回调记录。
            infos = self.get_publishers_info_by_topic(topic)
            types = known.get(topic, [])
            if not infos or not types:
                continue
            if len(types) != 1:
                raise ValueError(f"话题 {topic} 存在多个消息类型")
            self._register_topic(topic, types[0], [i.qos_profile for i in infos])
            qos = QoSProfile(
                depth=100,
                reliability=(
                    ReliabilityPolicy.RELIABLE
                    if all(i.qos_profile.reliability == ReliabilityPolicy.RELIABLE
                           for i in infos)
                    else ReliabilityPolicy.BEST_EFFORT
                ),
                durability=(
                    DurabilityPolicy.TRANSIENT_LOCAL
                    if all(i.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
                           for i in infos)
                    else DurabilityPolicy.VOLATILE
                ),
            )

            def on_message(data: bytes, selected_topic: str = topic) -> None:
                """将原始 CDR 字节写入闭包捕获的话题。"""
                self._write(selected_topic, bytes(data))

            self._subscriptions_by_topic[topic] = self.create_subscription(
                get_message(types[0]), topic, on_message, qos, raw=True
            )

    def _write(self, topic: str, data: bytes) -> None:
        """同步写包；失败后保留故障，禁止随后事件提交。"""
        if self._error is not None:
            raise RuntimeError(f"连续录制写入失败: {self._error}") from self._error
        timestamp_ns = max(self.get_clock().now().nanoseconds, self._last_log_ns)
        try:
            self._writer.write(topic, data, timestamp_ns)
        except Exception as error:
            self._error = error
            raise
        self._last_log_ns = timestamp_ns

    def _on_image(self, data: bytes) -> None:
        """先保存本订阅收到的图像，再用同一消息确认可解码起点。"""
        serialized = bytes(data)
        self._write(self._image_topic, serialized)
        if not self.readiness.ready:
            message = deserialize_message(serialized, self._image_class)
            self.readiness.feed(packet_from_message(message, self._mode))

    def record_event(self, event: EpisodeEvent) -> None:
        """在 EpisodeManager 提交边界状态前同步写入事件，写入失败则拒绝服务。"""
        if EVENT_TOPIC in self._topics:
            self._write(EVENT_TOPIC, serialize_message(event))

    def close(self) -> None:
        """在执行器停止后关闭 MCAP，完成索引与元数据写入。"""
        self._discovery_timer.cancel()
        self._writer.close()


def main(argv: Optional[List[str]] = None) -> None:
    """在同一个执行器中运行录包订阅和 episode 服务，退出时关闭 MCAP。"""
    parser = argparse.ArgumentParser(description="共享图像订阅的连续编码 MCAP 录制")
    parser.add_argument("--bag-uri", required=True)
    parser.add_argument("--topic", action="append", required=True, dest="topics")
    arguments, ros_arguments = parser.parse_known_args(argv)
    rclpy.init(args=ros_arguments)
    recorder = None  # 已创建的录包节点，供异常时清理。
    manager = None  # 已创建的边界服务节点。
    executor = SingleThreadedExecutor()  # 串行访问写入器及边界服务。
    try:
        recorder = ContinuousRecorder(arguments.bag_uri, arguments.topics)
        manager = EpisodeManager(
            event_observer=recorder.record_event, image_readiness=recorder.readiness
        )
        executor.add_node(recorder)
        executor.add_node(manager)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        try:
            if recorder is not None:
                recorder.close()
        finally:
            if manager is not None:
                manager.destroy_node()
            if recorder is not None:
                recorder.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
