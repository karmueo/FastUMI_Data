"""回放 MCAP、收集人工 episode 事件并原子生成标注后的新包。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
from typing import Callable, Iterator, Optional, Sequence

from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time
from fastumi_interfaces.msg import EpisodeAnnotation, EpisodeEvent
from fastumi_interfaces.srv import (
    DeleteEpisodeAnnotation,
    ListEpisodeAnnotations,
)
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message, serialize_message
import rosbag2_py
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Trigger

from fastumi_data.episode_manager import EpisodeManager, EpisodeSnapshot


EVENT_TOPIC = "/fastumi/episode/events"
TRACKER_TOPIC = "/vive_tracker/pose"
GRIPPER_TOPIC = "/gripper/state"
IMAGE_TYPE = "sensor_msgs/msg/Image"
EVENT_TYPE = "fastumi_interfaces/msg/EpisodeEvent"


def _stamp_to_ns(stamp: Time) -> int:
    """把 ROS 时间消息转换为无符号纳秒整数。"""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _ns_to_stamp(timestamp_ns: int) -> Time:
    """把绝对纳秒整数转换为 ROS 时间消息。"""
    return Time(
        sec=int(timestamp_ns) // 1_000_000_000,
        nanosec=int(timestamp_ns) % 1_000_000_000,
    )


@dataclass(frozen=True)
class PlaybackBounds:
    """保存 RViz2 时间轴使用的绝对起始时间和持续时间。"""

    start_time_ns: int
    """MCAP 第一条消息的绝对时间，单位为纳秒。"""
    duration_ns: int
    """MCAP 首尾消息之间的持续时间，单位为纳秒。"""

    def __post_init__(self) -> None:
        """拒绝无法构造有效时间轴的元数据。"""
        if self.start_time_ns <= 0 or self.duration_ns <= 0:
            raise ValueError("回放元数据必须包含正起始时间和正持续时间")

    @property
    def end_time_ns(self) -> int:
        """返回最后一条消息对应的绝对纳秒时间。"""
        return self.start_time_ns + self.duration_ns


@dataclass(frozen=True)
class CachedEvent:
    """保存一个以 header 时间为唯一合并键的序列化 episode 事件。"""

    timestamp_ns: int
    """事件 header.stamp 对应的纳秒时间。"""
    serialized: bytes
    """原始 CDR 序列化字节，写入时不再反序列化。"""
    event_type: int
    """START、STOP 或 ABORT 枚举值，用于同刻排序。"""


@dataclass(frozen=True)
class AnnotationCollectorSnapshot:
    """保存与 EpisodeManager revision 对应的事件缓存权威摘要。"""

    episodes: tuple[EpisodeAnnotation, ...]
    """全部由 STOP 正常闭合的 episode。"""
    last_event_time_ns: int
    """最后一条 START、STOP 或 ABORT 的绝对纳秒时间。"""
    has_annotations: bool
    """缓存中是否存在任意边界事件。"""


class EpisodeEventCollector:
    """同步缓存事件，并记录回放时钟是否曾到达有效非零时间。"""

    def __init__(self) -> None:
        """初始化锁、事件列表和时钟有效标志。"""
        self._lock = threading.Lock()
        self._events: list[CachedEvent] = []
        self._clock_nonzero = False

    def observe_event(self, event: EpisodeEvent) -> None:
        """在 EpisodeManager 持锁时同步序列化并缓存事件。

        Args:
            event: 即将发布且尚未提交状态的事件。

        Raises:
            ValueError: 回放时钟尚未有效或事件 header 时间为零。
        """
        timestamp_ns = _stamp_to_ns(event.header.stamp)
        with self._lock:
            if not self._clock_nonzero or timestamp_ns <= 0:
                raise ValueError("回放时钟尚未到达有效非零时间")
            self._events.append(
                CachedEvent(
                    timestamp_ns=timestamp_ns,
                    serialized=serialize_message(event),
                    event_type=int(event.event_type),
                )
            )

    def observe_clock(self, message: Clock) -> None:
        """接收 /clock 并在其值大于零后永久开放事件标注。"""
        if _stamp_to_ns(message.clock) > 0:
            with self._lock:
                self._clock_nonzero = True

    @property
    def clock_nonzero(self) -> bool:
        """返回是否已经观察到有效非零 bag 时钟。"""
        with self._lock:
            return self._clock_nonzero

    def events(self) -> tuple[CachedEvent, ...]:
        """返回保持服务线性化插入顺序的缓存副本。"""
        with self._lock:
            return tuple(self._events)

    def clear(self) -> None:
        """删除全部缓存边界，同时保留已经有效的回放时钟状态。"""
        with self._lock:
            self._events.clear()

    def completed_annotations(self) -> tuple[EpisodeAnnotation, ...]:
        """返回全部由 STOP 正常闭合的 episode 权威摘要。"""
        with self._lock:
            return self._completed_annotations_unlocked()

    def authoritative_snapshot(self) -> AnnotationCollectorSnapshot:
        """在单次缓存锁内返回列表、末次边界和非空状态。"""
        with self._lock:
            return AnnotationCollectorSnapshot(
                episodes=self._completed_annotations_unlocked(),
                last_event_time_ns=(
                    self._events[-1].timestamp_ns if self._events else 0
                ),
                has_annotations=bool(self._events),
            )

    def _completed_annotations_unlocked(
        self,
    ) -> tuple[EpisodeAnnotation, ...]:
        """在调用方持有缓存锁时构造全部正常闭合 episode 摘要。"""
        annotations: list[EpisodeAnnotation] = []
        start_event: Optional[EpisodeEvent] = None
        for cached in self._events:
            event = deserialize_message(cached.serialized, EpisodeEvent)
            if event.event_type == EpisodeEvent.START:
                start_event = event
                continue
            if (
                event.event_type == EpisodeEvent.STOP
                and start_event is not None
            ):
                annotation = EpisodeAnnotation()
                annotation.episode_index = int(event.episode_index)
                annotation.start_time = _ns_to_stamp(
                    _stamp_to_ns(start_event.header.stamp)
                )
                annotation.end_time = _ns_to_stamp(
                    _stamp_to_ns(event.header.stamp)
                )
                annotations.append(annotation)
            start_event = None
        return tuple(annotations)

    def delete_completed_episode(
        self, episode_index: int
    ) -> Optional[tuple[tuple[EpisodeEvent, ...], int]]:
        """删除指定 STOP episode，并连续重编号所有后续边界。

        Args:
            episode_index: 当前事件序列中的 episode 索引。

        Returns:
            删除后的消息和下一索引；目标不存在或目标为 ABORT 时返回
            ``None``。调用方必须在 EpisodeManager 空闲状态锁内执行。
        """
        with self._lock:
            start_position: Optional[int] = None
            delete_positions: Optional[tuple[int, int]] = None
            for position, cached in enumerate(self._events):
                event = deserialize_message(cached.serialized, EpisodeEvent)
                if event.event_type == EpisodeEvent.START:
                    start_position = position
                    continue
                if (
                    event.event_type == EpisodeEvent.STOP
                    and int(event.episode_index) == int(episode_index)
                    and start_position is not None
                ):
                    delete_positions = (start_position, position)
                    break
                start_position = None
            if delete_positions is None:
                return None

            retained = [
                cached
                for position, cached in enumerate(self._events)
                if position not in delete_positions
            ]
            replacement_events: list[CachedEvent] = []
            replacement_messages: list[EpisodeEvent] = []
            next_episode_index = 0
            active = False
            for cached in retained:
                event = deserialize_message(cached.serialized, EpisodeEvent)
                event.episode_index = next_episode_index
                replacement_messages.append(event)
                replacement_events.append(
                    CachedEvent(
                        timestamp_ns=cached.timestamp_ns,
                        serialized=serialize_message(event),
                        event_type=int(event.event_type),
                    )
                )
                if event.event_type == EpisodeEvent.START:
                    active = True
                else:
                    active = False
                    next_episode_index += 1
            if active:
                raise ValueError("活动 episode 不能执行单条删除")
            self._events = replacement_events
            return tuple(replacement_messages), next_episode_index

    def last_event_time_ns(self) -> int:
        """返回最后一条缓存边界的绝对时间；无标记时返回零。"""
        with self._lock:
            return self._events[-1].timestamp_ns if self._events else 0

    def has_annotations(self) -> bool:
        """返回当前缓存是否包含任意 START、STOP 或 ABORT。"""
        with self._lock:
            return bool(self._events)


class EpisodeAnnotationsController:
    """提供已完成 episode 权威列表和单条原子删除服务。"""

    def __init__(
        self,
        manager: EpisodeManager,
        collector: EpisodeEventCollector,
    ) -> None:
        """保存唯一状态机和对应事件缓存。"""
        self._manager = manager
        self._collector = collector

    def handle_list(
        self,
        request: ListEpisodeAnnotations.Request,
        response: ListEpisodeAnnotations.Response,
    ) -> ListEpisodeAnnotations.Response:
        """返回全部正常闭合 episode；活动和 ABORT 项不进入列表。"""
        del request
        snapshot, collector_snapshot = self._manager.read_consistent_state(
            self._collector.authoritative_snapshot
        )
        response.episodes = list(collector_snapshot.episodes)
        response.next_episode_index = snapshot.episode_index
        response.episode_active = snapshot.active
        response.last_event_time = _ns_to_stamp(
            collector_snapshot.last_event_time_ns
        )
        response.has_annotations = collector_snapshot.has_annotations
        response.revision = snapshot.revision
        return response

    def handle_delete(
        self,
        request: DeleteEpisodeAnnotation.Request,
        response: DeleteEpisodeAnnotation.Response,
    ) -> DeleteEpisodeAnnotation.Response:
        """在状态机锁内删除选中 episode 并返回替换后的权威状态。"""
        target_index = int(request.episode_index)
        snapshot = self._manager.try_replace_annotations(
            lambda: self._collector.delete_completed_episode(target_index)
        )
        response.success = snapshot is not None
        response.message = (
            f"episode {target_index} 已删除，后续编号已更新"
            if response.success
            else "无法删除：episode 正在活动、标注已冻结或目标不存在"
        )
        current_snapshot, collector_snapshot = self._manager.read_consistent_state(
            self._collector.authoritative_snapshot
        )
        response.episodes = list(collector_snapshot.episodes)
        response.next_episode_index = current_snapshot.episode_index
        response.episode_active = current_snapshot.active
        response.last_event_time = _ns_to_stamp(
            collector_snapshot.last_event_time_ns
        )
        response.has_annotations = collector_snapshot.has_annotations
        response.revision = current_snapshot.revision
        return response


class ClearAnnotationsController:
    """把清空服务线性化到 EpisodeManager 与事件缓存。"""

    def __init__(
        self,
        manager: EpisodeManager,
        collector: EpisodeEventCollector,
    ) -> None:
        """保存唯一状态机和对应事件缓存。"""
        self._manager = manager
        self._collector = collector

    def handle_request(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """清空全部标记；结束保存冻结后拒绝修改。"""
        del request
        response.success = self._manager.try_clear_annotations(
            self._collector.clear
        )
        response.message = (
            "已删除所有标记，episode 编号已重置为 0"
            if response.success
            else "episode 管理器已冻结，不能删除标记"
        )
        return response


class ManualFinishController:
    """协调手动结束服务、episode 原子冻结和播放器停止请求。"""

    def __init__(self, manager: EpisodeManager) -> None:
        """保存 episode 管理器并初始化一次性结束状态。

        Args:
            manager: 当前回放会话唯一的 episode 状态机。
        """
        self._manager = manager
        self._lock = threading.Lock()
        self._requested = threading.Event()
        self._snapshot: Optional[EpisodeSnapshot] = None

    @property
    def requested(self) -> threading.Event:
        """返回供播放器监督循环等待的线程事件。"""
        return self._requested

    @property
    def snapshot(self) -> Optional[EpisodeSnapshot]:
        """返回手动结束成功时取得的冻结快照。"""
        with self._lock:
            return self._snapshot

    def handle_request(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """接受空闲状态下的首次结束请求并保持重复调用幂等。"""
        del request
        with self._lock:
            if self._snapshot is not None:
                response.success = True
                response.message = "结束保存请求已接受，请等待输出完成"
                return response
            snapshot = self._manager.try_freeze_idle()
            if snapshot is None:
                response.success = False
                response.message = "存在活动 episode，请先结束或放弃后再保存"
                return response
            self._snapshot = snapshot
            response.success = True
            response.message = "已冻结标注，正在停止回放并保存"
            self._requested.set()
            return response


class MergeProgressReporter:
    """按固定百分比步长报告大 MCAP 的顺序复制进度。"""

    def __init__(
        self,
        total_messages: int,
        stage: str,
        step_percent: int = 5,
    ) -> None:
        """初始化总消息数、已处理计数和下一个报告阈值。"""
        self._total_messages = max(1, int(total_messages))
        self._stage = stage
        self._step_percent = max(1, int(step_percent))
        self._processed_messages = 0
        self._last_reported_percent = -self._step_percent

    def advance(self) -> None:
        """记录一条已读取源消息，并在越过阈值时输出进度。"""
        self._processed_messages += 1
        percent = min(
            100,
            self._processed_messages * 100 // self._total_messages,
        )
        report_percent = percent - percent % self._step_percent
        if report_percent > self._last_reported_percent:
            self._last_reported_percent = report_percent
            print(
                f"[保存] {self._stage}：{report_percent}% "
                f"({self._processed_messages}/{self._total_messages} 条消息)",
                flush=True,
            )

    def complete(self) -> None:
        """确保复制结束时始终显示 100%。"""
        if self._last_reported_percent < 100:
            self._last_reported_percent = 100
            print(
                f"[保存] {self._stage}：100% "
                f"({self._processed_messages}/{self._total_messages} 条消息)",
                flush=True,
            )


def validate_event_sequence(events: Sequence[CachedEvent]) -> None:
    """严格验证缓存事件为连续、完整且索引匹配的 START/结束对。"""
    expected_index = 0
    active_index: Optional[int] = None
    previous_timestamp_ns: Optional[int] = None
    for cached in events:
        event = deserialize_message(cached.serialized, EpisodeEvent)
        if _stamp_to_ns(event.header.stamp) != cached.timestamp_ns:
            raise ValueError("事件缓存键必须等于 header.stamp")
        if (
            previous_timestamp_ns is not None
            and cached.timestamp_ns < previous_timestamp_ns
        ):
            raise ValueError("episode 事件 header 时间必须非递减")
        previous_timestamp_ns = cached.timestamp_ns
        if event.event_type == EpisodeEvent.START:
            if active_index is not None or event.episode_index != expected_index:
                raise ValueError("episode START 顺序或索引无效")
            active_index = int(event.episode_index)
        elif event.event_type in (EpisodeEvent.STOP, EpisodeEvent.ABORT):
            if active_index is None or event.episode_index != active_index:
                raise ValueError("episode STOP/ABORT 没有匹配的 START")
            active_index = None
            expected_index += 1
        else:
            raise ValueError("未知 episode 事件类型")
    if active_index is not None:
        raise ValueError("存在未结束的活动 episode")


def _open_reader(uri: Path) -> rosbag2_py.SequentialReader:
    """以只读 MCAP 顺序读取器打开指定包目录。"""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(uri), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    return reader


def _build_annotated_storage_options(uri: Path) -> rosbag2_py.StorageOptions:
    """构造启用 MCAP 原生快速 Zstd 块压缩的补标输出选项。"""
    return rosbag2_py.StorageOptions(
        uri=str(uri),
        storage_id="mcap",
        storage_preset_profile="zstd_fast",
    )


def _finalize_annotated_metadata(uri: Path, output_name: str) -> None:
    """规范化单分片文件名，并修正 Jazzy Writer 的单文件消息计数。"""
    metadata_io = rosbag2_py.MetadataIo()
    metadata = metadata_io.read_metadata(str(uri))
    if len(metadata.relative_file_paths) != 1:
        raise RuntimeError("补标输出必须恰好包含一个 MCAP 分片")
    source_name = metadata.relative_file_paths[0]
    source_path = uri / source_name
    destination_name = f"{output_name}_0.mcap"
    destination_path = uri / destination_name
    if not source_path.is_file() or destination_path.exists():
        raise RuntimeError("补标输出 MCAP 分片状态无效")
    source_path.rename(destination_path)
    metadata.relative_file_paths = [destination_name]
    metadata.files = [
        rosbag2_py.FileInformation(
            destination_name,
            metadata.starting_time,
            metadata.duration,
            metadata.message_count,
        )
    ]
    metadata.bag_size = destination_path.stat().st_size
    metadata_io.write_metadata(str(uri), metadata)


def _close_if_supported(resource: object) -> None:
    """在当前 rosbag2 版本提供 close() 时显式释放其文件句柄。"""
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def _iter_non_event_messages(
    reader: rosbag2_py.SequentialReader,
    on_read: Optional[Callable[[], None]] = None,
) -> Iterator[tuple[str, bytes, int]]:
    """惰性跳过旧事件并逐条产出源消息，内存只保留当前消息。"""
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        if on_read is not None:
            on_read()
        if topic != EVENT_TOPIC:
            yield topic, bytes(serialized), int(timestamp_ns)


def source_topics(uri: Path) -> dict[str, object]:
    """读取源包元数据，返回名称到 rosbag TopicMetadata 的映射。"""
    reader = _open_reader(uri)
    try:
        return {
            metadata.name: metadata
            for metadata in reader.get_all_topics_and_types()
        }
    finally:
        _close_if_supported(reader)


def detect_image_topic(uri: Path, requested_topic: str = "") -> str:
    """优先检测 XV RGB 图像，或校验用户指定的 Image 话题。"""
    metadata = source_topics(uri)
    image_topics = [
        name
        for name, topic_metadata in metadata.items()
        if topic_metadata.type == IMAGE_TYPE
    ]
    if requested_topic:
        if requested_topic not in image_topics:
            raise ValueError("--image-topic 必须是源 MCAP 内的 sensor_msgs/msg/Image")
        return requested_topic
    xv_topics = [
        name for name in image_topics
        if name.startswith("/xv_sdk/") and name.endswith("/rgb/image")
    ]
    if len(xv_topics) == 1:
        return xv_topics[0]
    if len(xv_topics) > 1:
        raise ValueError("检测到多个 XV RGB 图像话题，请使用 --image-topic 选择")
    if len(image_topics) != 1:
        raise ValueError("源 MCAP 必须包含唯一 Image，或使用 --image-topic 选择")
    return image_topics[0]


def validate_source_bag(uri: Path, requested_image_topic: str = "") -> str:
    """检查源目录、必要话题及消息类型，并返回检测到的图像话题。"""
    if not uri.is_dir():
        raise ValueError(f"输入 MCAP 目录不存在: {uri}")
    metadata = source_topics(uri)
    required = {
        TRACKER_TOPIC: "geometry_msgs/msg/PoseStamped",
        GRIPPER_TOPIC: "fastumi_interfaces/msg/GripperState",
    }
    missing = [name for name in required if name not in metadata]
    wrong_type = [
        name for name, message_type in required.items()
        if name in metadata and metadata[name].type != message_type
    ]
    if missing or wrong_type:
        details = [*(f"缺少 {name}" for name in missing), *(f"类型不兼容 {name}" for name in wrong_type)]
        raise ValueError("源 MCAP 不满足回放标注要求: " + "；".join(details))
    return detect_image_topic(uri, requested_image_topic)


def read_playback_bounds(uri: Path) -> PlaybackBounds:
    """从 rosbag2 元数据读取 RViz2 时间轴需要的精确纳秒边界。"""
    metadata = rosbag2_py.Info().read_metadata(str(uri), "mcap")
    return PlaybackBounds(
        start_time_ns=int(metadata.starting_time.nanoseconds),
        duration_ns=int(metadata.duration.nanoseconds),
    )


def merge_annotated_bag(
    source_uri: Path,
    output_uri: Path,
    events: Sequence[CachedEvent],
) -> None:
    """离线合并源包和新事件，在验证后原子发布以前不存在的输出目录。

    源消息的 CDR 字节和包时间保持不变；原有 episode 事件被替换。临时
    目录仅在本函数失败时移除，源路径和用户路径从不删除。
    """
    source = source_uri.resolve()
    output = output_uri.resolve()
    if output.exists():
        raise FileExistsError(f"输出路径已存在: {output}")
    if output == source:
        raise ValueError("输出路径不能与输入 MCAP 相同")
    if not output.parent.is_dir():
        raise ValueError(f"输出父目录不存在: {output.parent}")
    validate_event_sequence(events)
    source_metadata = source_topics(source)
    source_bag_metadata = rosbag2_py.Info().read_metadata(str(source), "mcap")
    total_messages = int(source_bag_metadata.message_count)
    progress = MergeProgressReporter(total_messages, "正在复制源 MCAP")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.replay-", dir=str(output.parent))
    )
    # rosbag2 只接受不存在的 URI；名称由 mkdtemp 原子保留后立即释放。
    temporary.rmdir()
    try:
        writer = rosbag2_py.SequentialWriter()
        writer.open(
            _build_annotated_storage_options(temporary),
            rosbag2_py.ConverterOptions("", ""),
        )
        for metadata in source_metadata.values():
            writer.create_topic(metadata)
        if EVENT_TOPIC not in source_metadata and events:
            writer.create_topic(
                rosbag2_py.TopicMetadata(
                    id=0,
                    name=EVENT_TOPIC,
                    type=EVENT_TYPE,
                    serialization_format="cdr",
                )
            )

        reader = _open_reader(source)
        try:
            source_iterator = _iter_non_event_messages(reader, progress.advance)
            source_message = next(source_iterator, None)
            for event in events:
                while (
                    source_message is not None
                    and source_message[2] < event.timestamp_ns
                ):
                    writer.write(*source_message)
                    source_message = next(source_iterator, None)
                if event.event_type != EpisodeEvent.START:
                    while (
                        source_message is not None
                        and source_message[2] == event.timestamp_ns
                    ):
                        writer.write(*source_message)
                        source_message = next(source_iterator, None)
                writer.write(EVENT_TOPIC, event.serialized, event.timestamp_ns)
            while source_message is not None:
                writer.write(*source_message)
                source_message = next(source_iterator, None)
        finally:
            _close_if_supported(reader)
        del writer

        progress.complete()
        print("[保存] 源消息复制完成，正在逐字节校验输出 MCAP…", flush=True)
        validation_progress = MergeProgressReporter(total_messages, "正在校验输出 MCAP")
        _validate_merged_bag(
            source,
            temporary,
            tuple(events),
            validation_progress.advance,
        )
        validation_progress.complete()
        _finalize_annotated_metadata(temporary, output.name)
        os.replace(temporary, output)
        print(f"[保存] 标注结果已安全写入：{output}", flush=True)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _validate_merged_bag(
    source_uri: Path,
    output_uri: Path,
    expected_events: Sequence[CachedEvent],
    on_source_read: Optional[Callable[[], None]] = None,
) -> None:
    """流式核对临时包，始终只保留一条源消息与小型事件列表。"""
    metadata = source_topics(output_uri)
    if expected_events and EVENT_TOPIC not in metadata:
        raise RuntimeError("临时包未创建 episode 事件话题")
    source_reader = _open_reader(source_uri)
    output_reader = _open_reader(output_uri)
    actual_events: list[CachedEvent] = []
    try:
        expected_iterator = _iter_non_event_messages(source_reader, on_source_read)
        expected_source = next(expected_iterator, None)
        while output_reader.has_next():
            topic, serialized, timestamp_ns = output_reader.read_next()
            if topic == EVENT_TOPIC:
                message = deserialize_message(serialized, EpisodeEvent)
                actual_events.append(CachedEvent(
                    _stamp_to_ns(message.header.stamp), bytes(serialized),
                    int(message.event_type)))
                continue
            actual_source = (topic, bytes(serialized), int(timestamp_ns))
            if actual_source != expected_source:
                raise RuntimeError("临时包未逐字节保留源消息")
            expected_source = next(expected_iterator, None)
        if expected_source is not None:
            raise RuntimeError("临时包缺少源消息")
    finally:
        _close_if_supported(source_reader)
        _close_if_supported(output_reader)
    if actual_events != list(expected_events):
        raise RuntimeError("临时包的新 episode 事件序列不匹配")


def build_player_command(bag: Path, image_topic: str, rate: float) -> list[str]:
    """构造以模拟时钟启动、禁用键盘并重映射图像的 rosbag2 回放命令。"""
    return [
        "ros2", "bag", "play", str(bag), "--clock", "--start-paused",
        "--disable-keyboard-controls", "--exclude-topics", EVENT_TOPIC,
        "--rate", str(rate), "--remap", f"{image_topic}:=/fastumi/replay/image",
    ]


def build_rviz_command(
    config: Path,
    bounds: PlaybackBounds,
    initial_rate: float,
) -> list[str]:
    """构造携带回放时间轴边界和初始倍率的 RViz2 命令。"""
    return [
        "rviz2", "-d", str(config), "--ros-args",
        "-p", "use_sim_time:=true",
        "-p", f"replay_start_time_ns:={bounds.start_time_ns}",
        "-p", f"replay_duration_ns:={bounds.duration_ns}",
        "-p", f"replay_initial_rate:={initial_rate}",
    ]


def _build_parser() -> argparse.ArgumentParser:
    """构造回放标注命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="回放 MCAP 并人工标注 episode")
    parser.add_argument("--bag", required=True, help="输入 MCAP 目录")
    parser.add_argument("--output", required=True, help="新的输出 MCAP 目录")
    parser.add_argument("--task", default="", help="事件任务名，默认由输入目录推导")
    parser.add_argument("--session-id", default="", help="事件会话 ID，默认生成 UTC 时间")
    parser.add_argument("--rate", type=float, default=1.0, help="回放倍率")
    parser.add_argument(
        "--image-topic", default="",
        help="可选源 Image 话题；默认优先唯一 XV RGB 话题",
    )
    parser.add_argument("--rviz-config", default="", help="可选 RViz 配置路径")
    return parser


def _terminate(process: subprocess.Popen[bytes]) -> None:
    """依次 SIGINT、terminate、kill 回收工具创建的子进程。"""
    for action in (lambda: process.send_signal(signal.SIGINT), process.terminate, process.kill):
        if process.poll() is not None:
            return
        action()
        try:
            process.wait(timeout=5.0)
            return
        except subprocess.TimeoutExpired:
            continue


def _wait_for_playback(
    player: subprocess.Popen[bytes],
    rviz: subprocess.Popen[bytes],
    manual_finish_requested: threading.Event,
) -> bool:
    """监督回放和 RViz，并响应已经通过服务校验的手动结束请求。"""
    while True:
        if manual_finish_requested.is_set():
            print("[保存] 已收到手动结束请求，正在停止回放并冻结标注…", flush=True)
            _terminate(player)
            return True
        if player.poll() is not None:
            break
        rviz_code = rviz.poll()
        if rviz_code is not None:
            raise RuntimeError(f"RViz 在回放完成前退出，退出码为 {rviz_code}")
        manual_finish_requested.wait(timeout=0.2)
    if manual_finish_requested.is_set():
        return True
    if player.returncode != 0:
        raise RuntimeError(f"ros2 bag play 退出码为 {player.returncode}")
    if rviz.poll() is not None:
        raise RuntimeError("RViz 在回放完成、事件冻结前退出")
    return False


def main(argv: Optional[Sequence[str]] = None) -> None:
    """启动回放标注会话，成功时才原子发布完成校验的新 MCAP。"""
    arguments = _build_parser().parse_args(argv)
    bag = Path(arguments.bag).resolve()
    output = Path(arguments.output).resolve()
    if arguments.rate <= 0.0:
        raise ValueError("--rate 必须大于零")
    if output.exists() or output == bag:
        raise ValueError("--output 必须是不同于输入且尚不存在的路径")
    if not output.parent.is_dir():
        raise ValueError(f"输出父目录不存在: {output.parent}")
    rviz_config = Path(arguments.rviz_config).resolve() if arguments.rviz_config else (
        Path(get_package_share_directory("fastumi_rviz_plugins"))
        / "config" / "replay_annotation.rviz"
    )
    if not rviz_config.is_file():
        raise ValueError(f"RViz 配置不存在: {rviz_config}")
    image_topic = validate_source_bag(bag, arguments.image_topic.strip())
    playback_bounds = read_playback_bounds(bag)
    task = arguments.task.strip() or bag.name
    session_id = arguments.session_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rclpy.init(args=["--ros-args", "-p", f"task_name:={task}", "-p", f"session_id:={session_id}", "-p", "use_sim_time:=true"])
    collector = EpisodeEventCollector()
    manager = EpisodeManager(event_observer=collector.observe_event)
    finish_controller = ManualFinishController(manager)
    clear_controller = ClearAnnotationsController(manager, collector)
    annotations_controller = EpisodeAnnotationsController(manager, collector)
    clock_node = Node("replay_annotation_clock", parameter_overrides=[])
    clock_node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
    clock_qos = QoSProfile(
        depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    clock_node.create_subscription(
        Clock, "/clock", collector.observe_clock, clock_qos
    )
    finish_service = clock_node.create_service(
        Trigger,
        "/fastumi/replay/finish_and_save",
        finish_controller.handle_request,
    )
    clear_service = clock_node.create_service(
        Trigger,
        "/fastumi/replay/clear_annotations",
        clear_controller.handle_request,
    )
    list_annotations_service = clock_node.create_service(
        ListEpisodeAnnotations,
        "/fastumi/replay/list_episode_annotations",
        annotations_controller.handle_list,
    )
    delete_annotation_service = clock_node.create_service(
        DeleteEpisodeAnnotation,
        "/fastumi/replay/delete_episode_annotation",
        annotations_controller.handle_delete,
    )
    executor = MultiThreadedExecutor()
    executor.add_node(manager)
    executor.add_node(clock_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    player: Optional[subprocess.Popen[bytes]] = None
    rviz: Optional[subprocess.Popen[bytes]] = None
    snapshot: Optional[EpisodeSnapshot] = None
    try:
        player = subprocess.Popen(build_player_command(bag, image_topic, arguments.rate))
        rviz_command = build_rviz_command(
            rviz_config, playback_bounds, arguments.rate
        )
        rviz = subprocess.Popen(rviz_command)
        manual_finish = _wait_for_playback(
            player, rviz, finish_controller.requested
        )
        snapshot = finish_controller.snapshot if manual_finish else manager.freeze()
        if snapshot is None:
            raise RuntimeError("手动结束请求缺少冻结快照")
        if not collector.clock_nonzero:
            raise RuntimeError("回放期间未接收到有效非零 /clock")
        if snapshot.active:
            raise RuntimeError("回放结束时仍有未关闭的 episode")
        validate_event_sequence(collector.events())
        if rviz.poll() is None:
            _terminate(rviz)
        print(
            f"[保存] 已冻结 {len(collector.events())} 条边界事件，开始生成输出 MCAP：{output}",
            flush=True,
        )
        merge_annotated_bag(bag, output, collector.events())
    finally:
        del delete_annotation_service
        del list_annotations_service
        del clear_service
        del finish_service
        if rviz is not None and rviz.poll() is None:
            _terminate(rviz)
        if player is not None and player.poll() is None:
            _terminate(player)
        executor.shutdown()
        manager.destroy_node()
        clock_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
