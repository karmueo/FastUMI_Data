"""验证回放标注的事件校验、命令构造和真实 MCAP 离线合并。"""

from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

rosbag2_py = pytest.importorskip("rosbag2_py")
from builtin_interfaces.msg import Time
from fastumi_interfaces.msg import EpisodeEvent
from fastumi_interfaces.srv import DeleteEpisodeAnnotation, ListEpisodeAnnotations
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import deserialize_message, serialize_message
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image

from fastumi_data.replay_annotation import (
    CachedEvent,
    ClearAnnotationsController,
    EpisodeAnnotationsController,
    EpisodeEventCollector,
    EVENT_TOPIC,
    ManualFinishController,
    PlaybackBounds,
    _build_annotated_storage_options,
    _wait_for_playback,
    _iter_non_event_messages,
    build_player_command,
    build_rviz_command,
    detect_image_topic,
    merge_annotated_bag,
    read_playback_bounds,
    validate_event_sequence,
)
from fastumi_data.episode_manager import EpisodeSnapshot
from std_srvs.srv import Trigger


def test_annotated_writer_uses_fast_zstd_mcap(tmp_path: Path) -> None:
    """验证补标输出固定使用 MCAP 原生快速 Zstd 块压缩。"""
    output = tmp_path / "annotated"

    options = _build_annotated_storage_options(output)

    assert options.uri == str(output)
    assert options.storage_id == "mcap"
    assert options.storage_preset_profile == "zstd_fast"


def _event(timestamp_ns: int, event_type: int, index: int) -> CachedEvent:
    """构造具有给定 header 时间和类型的序列化 episode 事件。"""
    message = EpisodeEvent()
    message.header.stamp = Time(sec=timestamp_ns // 1_000_000_000, nanosec=timestamp_ns % 1_000_000_000)
    message.episode_index = index
    message.event_type = event_type
    return CachedEvent(timestamp_ns, serialize_message(message), event_type)


def _create_source(uri: Path) -> list[tuple[str, bytes, int]]:
    """创建含旧事件、图像、Tracker 和夹爪话题的微型真实 MCAP。"""
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(uri), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    topics = [
        ("/camera/image", "sensor_msgs/msg/Image"),
        ("/vive_tracker/pose", "geometry_msgs/msg/PoseStamped"),
        ("/gripper/state", "fastumi_interfaces/msg/GripperState"),
        (EVENT_TOPIC, "fastumi_interfaces/msg/EpisodeEvent"),
    ]
    for index, (name, message_type) in enumerate(topics):
        writer.create_topic(rosbag2_py.TopicMetadata(id=index, name=name, type=message_type, serialization_format="cdr"))
    image = Image(height=1, width=1, encoding="rgb8", step=3, data=[1, 2, 3])
    pose = PoseStamped()
    pose.pose.orientation.w = 1.0
    records = [
        ("/camera/image", serialize_message(image), 10),
        ("/vive_tracker/pose", serialize_message(pose), 20),
        (EVENT_TOPIC, _event(25, EpisodeEvent.START, 0).serialized, 25),
    ]
    for record in records:
        writer.write(*record)
    del writer
    return records[:2]


def test_merge_replaces_source_events_preserves_bytes_and_orders_ties(tmp_path: Path) -> None:
    """验证新 START、源同刻消息、STOP 的排序及原始 CDR 字节保留。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    original = _create_source(source)
    events = (_event(10, EpisodeEvent.START, 0), _event(20, EpisodeEvent.STOP, 0))

    merge_annotated_bag(source, output, events)

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(output), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    actual = []
    while reader.has_next():
        actual.append(reader.read_next())
    assert [(topic, bytes(data), stamp) for topic, data, stamp in actual if topic != EVENT_TOPIC] == original
    assert [topic for topic, _, _ in actual] == [EVENT_TOPIC, "/camera/image", "/vive_tracker/pose", EVENT_TOPIC]
    assert not source.joinpath("metadata.yaml").samefile(output.joinpath("metadata.yaml"))
    metadata = rosbag2_py.MetadataIo().read_metadata(str(output))
    assert metadata.relative_file_paths == ["output_0.mcap"]
    assert len(metadata.files) == 1
    assert metadata.files[0].path == "output_0.mcap"
    assert metadata.files[0].message_count == metadata.message_count == 4


def test_strict_event_validation_and_player_options() -> None:
    """验证未闭合 pair 被拒绝，播放器包含关键回放和话题选项。"""
    with pytest.raises(ValueError, match="未结束"):
        validate_event_sequence([_event(1, EpisodeEvent.START, 0)])
    command = build_player_command(Path("bag"), "/xv_sdk/SN/rgb/image", 1.5)
    assert "--clock" in command and "--start-paused" in command
    assert "--disable-keyboard-controls" in command
    assert command[command.index("--exclude-topics") + 1] == EVENT_TOPIC
    assert "/xv_sdk/SN/rgb/image:=/fastumi/replay/image" in command


def test_playback_bounds_and_rviz_parameters_use_exact_nanoseconds(
    tmp_path: Path,
) -> None:
    """验证真实 MCAP 边界读取及 RViz2 内部参数保持纳秒精度。"""
    source = tmp_path / "source"
    _create_source(source)

    bounds = read_playback_bounds(source)

    assert bounds == PlaybackBounds(start_time_ns=10, duration_ns=15)
    command = build_rviz_command(Path("config.rviz"), bounds, 1.5)
    assert command[:3] == ["rviz2", "-d", "config.rviz"]
    assert "replay_start_time_ns:=10" in command
    assert "replay_duration_ns:=15" in command
    assert "replay_initial_rate:=1.5" in command
    assert "use_sim_time:=true" in command


@pytest.mark.parametrize(
    ("start_time_ns", "duration_ns"),
    [(0, 1), (1, 0), (1, -1)],
)
def test_playback_bounds_reject_invalid_metadata(
    start_time_ns: int,
    duration_ns: int,
) -> None:
    """验证零起点、零时长和负时长不会进入 RViz2 控制链路。"""
    with pytest.raises(ValueError, match="回放元数据"):
        PlaybackBounds(start_time_ns=start_time_ns, duration_ns=duration_ns)


def test_same_timestamp_episodes_preserve_linearized_order_and_source(
    tmp_path: Path,
) -> None:
    """验证暂停时刻的两个完整 episode 不会按类型重排。"""
    source = tmp_path / "source"
    output = tmp_path / "output"
    original = _create_source(source)
    events = (
        _event(10, EpisodeEvent.START, 0),
        _event(10, EpisodeEvent.STOP, 0),
        _event(10, EpisodeEvent.START, 1),
        _event(10, EpisodeEvent.STOP, 1),
    )
    collector = EpisodeEventCollector()
    collector.observe_clock(Clock(clock=Time(sec=1)))
    for cached in events:
        collector.observe_event(
            deserialize_message(cached.serialized, EpisodeEvent)
        )
    assert collector.events() == events

    merge_annotated_bag(source, output, collector.events())

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(output), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    actual_events = []
    source_records = []
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        if topic == EVENT_TOPIC:
            actual_events.append(bytes(serialized))
        else:
            source_records.append((topic, bytes(serialized), timestamp_ns))
    assert actual_events == [event.serialized for event in events]
    assert source_records == original


def test_image_topic_prefers_xv_and_iterator_uses_single_message_lookahead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证调试图像存在时优先 XV RGB，且源迭代不会累积整包消息。"""
    metadata = {
        "/gripper/openness/debug_image": SimpleNamespace(type="sensor_msgs/msg/Image"),
        "/xv_sdk/SN/rgb/image": SimpleNamespace(type="sensor_msgs/msg/Image"),
    }
    monkeypatch.setattr(
        "fastumi_data.replay_annotation.source_topics", lambda uri: metadata
    )
    assert detect_image_topic(Path("bag")) == "/xv_sdk/SN/rgb/image"
    assert detect_image_topic(Path("bag"), "/gripper/openness/debug_image") == "/gripper/openness/debug_image"

    class FakeReader:
        """记录读取次数的轻量假读取器。"""

        def __init__(self) -> None:
            """初始化大量源消息和读取计数。"""
            self.records = iter(("/data", b"x", index) for index in range(10000))
            self.read_count = 0

        def has_next(self) -> bool:
            """报告测试记录是否仍未读完。"""
            return self.read_count < 10000

        def read_next(self) -> tuple[str, bytes, int]:
            """读取且只返回一条测试记录。"""
            self.read_count += 1
            return next(self.records)

    reader = FakeReader()
    iterator = _iter_non_event_messages(reader)
    assert next(iterator) == ("/data", b"x", 0)
    assert reader.read_count == 1


def test_manual_finish_rejects_active_then_freezes_idle() -> None:
    """验证手动结束只接受已闭合 episode，并保持重复请求幂等。"""

    class FakeManager:
        """按预设顺序返回原子冻结结果。"""

        def __init__(self) -> None:
            """初始化活动拒绝和空闲接受两次结果。"""
            self.results = [
                None,
                EpisodeSnapshot(active=False, episode_index=3, revision=6),
            ]

        def try_freeze_idle(self) -> EpisodeSnapshot | None:
            """返回下一次预设冻结结果。"""
            return self.results.pop(0)

    controller = ManualFinishController(FakeManager())
    rejected = controller.handle_request(Trigger.Request(), Trigger.Response())
    accepted = controller.handle_request(Trigger.Request(), Trigger.Response())
    repeated = controller.handle_request(Trigger.Request(), Trigger.Response())

    assert not rejected.success
    assert "活动 episode" in rejected.message
    assert accepted.success and repeated.success
    assert controller.requested.is_set()
    assert controller.snapshot == EpisodeSnapshot(
        active=False, episode_index=3, revision=6
    )


def test_clear_annotations_controller_clears_collector_and_reports_result() -> None:
    """验证清空服务调用管理器原子重置并删除全部缓存事件。"""

    class FakeManager:
        """模拟一次成功和一次冻结拒绝。"""

        def __init__(self) -> None:
            """初始化清空调用计数。"""
            self.calls = 0

        def try_clear_annotations(self, on_clear) -> bool:
            """首次执行回调，第二次模拟保存冻结。"""
            self.calls += 1
            if self.calls > 1:
                return False
            on_clear()
            return True

    collector = EpisodeEventCollector()
    collector._events.extend([
        _event(1, EpisodeEvent.START, 0),
        _event(2, EpisodeEvent.STOP, 0),
    ])
    controller = ClearAnnotationsController(FakeManager(), collector)

    accepted = controller.handle_request(Trigger.Request(), Trigger.Response())
    rejected = controller.handle_request(Trigger.Request(), Trigger.Response())

    assert accepted.success
    assert collector.events() == ()
    assert not rejected.success
    assert "冻结" in rejected.message


def test_episode_list_excludes_abort_and_preserves_exact_boundaries() -> None:
    """验证权威列表只返回 STOP 闭合项及其精确 START/STOP 时间。"""
    collector = EpisodeEventCollector()
    collector._events.extend([
        _event(10, EpisodeEvent.START, 0),
        _event(20, EpisodeEvent.STOP, 0),
        _event(30, EpisodeEvent.START, 1),
        _event(40, EpisodeEvent.ABORT, 1),
        _event(50, EpisodeEvent.START, 2),
        _event(80, EpisodeEvent.STOP, 2),
    ])

    episodes = collector.completed_annotations()

    assert [episode.episode_index for episode in episodes] == [0, 2]
    assert [episode.start_time.nanosec for episode in episodes] == [10, 50]
    assert [episode.end_time.nanosec for episode in episodes] == [20, 80]


def test_delete_episode_removes_pair_and_reindexes_all_later_attempts() -> None:
    """验证删除完整 episode 后保留 ABORT，并连续重编号所有后续边界。"""
    collector = EpisodeEventCollector()
    collector._events.extend([
        _event(10, EpisodeEvent.START, 0),
        _event(20, EpisodeEvent.STOP, 0),
        _event(30, EpisodeEvent.START, 1),
        _event(40, EpisodeEvent.ABORT, 1),
        _event(50, EpisodeEvent.START, 2),
        _event(80, EpisodeEvent.STOP, 2),
    ])

    replacement = collector.delete_completed_episode(0)

    assert replacement is not None
    messages, next_episode_index = replacement
    assert next_episode_index == 2
    assert [(message.episode_index, message.event_type) for message in messages] == [
        (0, EpisodeEvent.START),
        (0, EpisodeEvent.ABORT),
        (1, EpisodeEvent.START),
        (1, EpisodeEvent.STOP),
    ]
    assert [episode.episode_index for episode in collector.completed_annotations()] == [1]
    validate_event_sequence(collector.events())
    assert collector.delete_completed_episode(0) is None


def test_episode_annotation_services_list_and_delete_authoritative_state() -> None:
    """验证列表和删除服务返回服务端重编号后的权威快照。"""

    class FakeManager:
        """执行替换回调并返回新的空闲状态快照。"""

        def __init__(self) -> None:
            """初始化两条完整 episode 对应的状态与 revision。"""
            self.episode_index = 2
            self.revision = 4

        def try_replace_annotations(self, on_replace):
            """执行删除并把回调的新索引包装成管理器快照。"""
            replacement = on_replace()
            if replacement is None:
                return None
            _, next_episode_index = replacement
            self.episode_index = next_episode_index
            self.revision += 1
            return EpisodeSnapshot(
                active=False,
                episode_index=next_episode_index,
                revision=self.revision,
            )

        def read_consistent_state(self, on_read):
            """同步返回当前状态及 collector 权威 payload。"""
            return (
                EpisodeSnapshot(
                    active=False,
                    episode_index=self.episode_index,
                    revision=self.revision,
                ),
                on_read(),
            )

    collector = EpisodeEventCollector()
    collector._events.extend([
        _event(10, EpisodeEvent.START, 0),
        _event(20, EpisodeEvent.STOP, 0),
        _event(30, EpisodeEvent.START, 1),
        _event(40, EpisodeEvent.STOP, 1),
    ])
    controller = EpisodeAnnotationsController(FakeManager(), collector)

    listed = controller.handle_list(
        ListEpisodeAnnotations.Request(), ListEpisodeAnnotations.Response()
    )
    request = DeleteEpisodeAnnotation.Request(episode_index=0)
    deleted = controller.handle_delete(request, DeleteEpisodeAnnotation.Response())

    assert [episode.episode_index for episode in listed.episodes] == [0, 1]
    assert listed.revision == 4
    assert deleted.success and deleted.next_episode_index == 1
    assert deleted.episode_active is False
    assert deleted.revision == 5
    assert deleted.has_annotations
    assert deleted.last_event_time.nanosec == 40
    assert [episode.episode_index for episode in deleted.episodes] == [0]


def test_wait_for_playback_accepts_valid_manual_finish() -> None:
    """验证结束事件会主动停止播放器，并跳过人工终止退出码检查。"""

    class FakePlayer:
        """模拟收到 SIGINT 后以负退出码结束的播放器。"""

        def __init__(self) -> None:
            """初始化运行状态和收到的信号列表。"""
            self.returncode = None
            self.signals = []

        def poll(self):
            """返回当前退出状态。"""
            return self.returncode

        def send_signal(self, received_signal) -> None:
            """记录信号并模拟播放器退出。"""
            self.signals.append(received_signal)
            self.returncode = -2

        def terminate(self) -> None:
            """提供进程回收接口。"""

        def kill(self) -> None:
            """提供进程强制回收接口。"""

        def wait(self, timeout=None):
            """返回已经设置的退出码。"""
            del timeout
            return self.returncode

    class FakeRviz:
        """模拟始终存活的 RViz。"""

        def poll(self):
            """报告进程仍在运行。"""
            return None

    requested = threading.Event()
    requested.set()
    player = FakePlayer()

    assert _wait_for_playback(player, FakeRviz(), requested)
    assert player.signals
