"""验证有界队列 MCAP 写入器的顺序、溢出、故障和真实 MCAP 输出。"""

from pathlib import Path
import threading
from typing import List, Optional, Sequence, Tuple

import pytest
import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from std_msgs.msg import String

from fastumi_data.collection_writer import (
    ERROR_FINISH_TIMEOUT,
    ERROR_QUEUE_OVERFLOW,
    ERROR_WRITE_FAILED,
    QueuedBagWriter,
    RosbagMcapBackend,
    TopicSpec,
)


class _FakeBackend:
    """记录写入内容和线程，并可按需失败或阻塞。"""

    def __init__(
        self,
        fail_on_write: Optional[int] = None,
        fail_on_close: bool = False,
        gate: Optional[threading.Event] = None,
    ) -> None:
        """配置故障注入。"""
        self.opened_with: Optional[Tuple[str, Sequence[TopicSpec]]] = None
        self.writes: List[Tuple[str, bytes, int]] = []
        self.threads = set()
        self.closed = False
        self._fail_on_write = fail_on_write
        self._fail_on_close = fail_on_close
        self._gate = gate

    def open(self, uri: str, topics: Sequence[TopicSpec]) -> None:
        """记录打开参数。"""
        self.opened_with = (uri, topics)

    def write(self, topic: str, data: bytes, timestamp_ns: int) -> None:
        """记录写入；按配置失败或等待放行。"""
        if self._gate is not None:
            self._gate.wait(timeout=10)
        self.threads.add(threading.get_ident())
        if self._fail_on_write is not None and len(self.writes) == self._fail_on_write:
            raise OSError("磁盘已满")
        self.writes.append((topic, data, timestamp_ns))

    def close(self) -> None:
        """记录关闭，可按配置失败。"""
        self.threads.add(threading.get_ident())
        if self._fail_on_close:
            raise OSError("索引写出失败")
        self.closed = True


_TOPICS = [TopicSpec("/a", "std_msgs/msg/String")]


def test_messages_are_written_in_order_by_one_thread() -> None:
    """验证多线程入队的消息由单一写入线程按入队顺序写出并计数。"""
    backend = _FakeBackend()
    writer = QueuedBagWriter(backend, 100, 10_000)
    writer.start("uri", _TOPICS)

    for index in range(10):
        assert writer.offer("/a", bytes([index]), index)
    result = writer.finish(5.0)

    assert result.ok
    assert result.message_counts == {"/a": 10}
    assert [item[2] for item in backend.writes] == list(range(10))
    assert backend.closed
    assert len(backend.threads) == 1
    assert threading.get_ident() not in backend.threads
    assert backend.opened_with == ("uri", _TOPICS)


def test_finish_drains_everything_accepted_before_stop() -> None:
    """验证停止接受后已入队消息全部写出，之后的入队被拒绝。"""
    gate = threading.Event()
    backend = _FakeBackend(gate=gate)
    writer = QueuedBagWriter(backend, 100, 10_000)
    writer.start("uri", _TOPICS)
    for index in range(5):
        writer.offer("/a", b"x", index)
    writer.stop_accepting()
    assert writer.offer("/a", b"late", 99) is False
    gate.set()

    result = writer.finish(5.0)

    assert result.ok and result.message_counts == {"/a": 5}


def test_queue_overflow_reports_error_once_and_fails_result() -> None:
    """验证队列满时报告 QUEUE_OVERFLOW，后续入队被拒绝且结果失败。"""
    gate = threading.Event()
    errors: List[Tuple[str, str]] = []
    writer = QueuedBagWriter(
        _FakeBackend(gate=gate), 2, 10_000, on_error=lambda c, m: errors.append((c, m))
    )
    writer.start("uri", _TOPICS)

    accepted = [writer.offer("/a", b"x", index) for index in range(6)]
    gate.set()
    result = writer.finish(5.0)

    assert accepted.count(False) >= 1
    assert len(errors) == 1 and errors[0][0] == ERROR_QUEUE_OVERFLOW
    assert result.error_code == ERROR_QUEUE_OVERFLOW
    assert not result.ok
    assert writer.offer("/a", b"x", 99) is False


def test_byte_limit_overflows_but_single_oversized_message_is_allowed() -> None:
    """验证按字节限流，空队列时允许单条超限消息。"""
    gate = threading.Event()
    writer = QueuedBagWriter(_FakeBackend(gate=gate), 100, 10)
    writer.start("uri", _TOPICS)

    assert writer.offer("/a", bytes(50), 1) is True  # 空队列允许超限单条。
    assert writer.offer("/a", bytes(1), 2) is False  # 随后超过字节上限。
    gate.set()
    result = writer.finish(5.0)

    assert result.error_code == ERROR_QUEUE_OVERFLOW


def test_write_failure_is_reported_and_marks_result_failed() -> None:
    """验证后端写入异常被捕获为 WRITE_FAILED 并通知一次。"""
    errors: List[Tuple[str, str]] = []
    writer = QueuedBagWriter(
        _FakeBackend(fail_on_write=2),
        100,
        10_000,
        on_error=lambda c, m: errors.append((c, m)),
    )
    writer.start("uri", _TOPICS)

    for index in range(5):
        writer.offer("/a", b"x", index)
    result = writer.finish(5.0)

    assert len(errors) == 1 and errors[0][0] == ERROR_WRITE_FAILED
    assert "磁盘已满" in errors[0][1]
    assert result.error_code == ERROR_WRITE_FAILED
    assert result.message_counts == {"/a": 2}


def test_close_failure_marks_result_failed() -> None:
    """验证关闭 MCAP 失败不会被当作成功。"""
    writer = QueuedBagWriter(_FakeBackend(fail_on_close=True), 100, 10_000)
    writer.start("uri", _TOPICS)
    writer.offer("/a", b"x", 1)

    result = writer.finish(5.0)

    assert result.error_code == ERROR_WRITE_FAILED
    assert "索引写出失败" in result.error_message


def test_finish_timeout_is_reported_when_backend_is_stuck() -> None:
    """验证写入线程卡住时 finish 在超时后返回明确错误。"""
    gate = threading.Event()
    writer = QueuedBagWriter(_FakeBackend(gate=gate), 100, 10_000)
    writer.start("uri", _TOPICS)
    writer.offer("/a", b"x", 1)

    result = writer.finish(0.2)

    assert result.error_code == ERROR_FINISH_TIMEOUT
    gate.set()
    writer.abort()


def test_abort_discards_queued_messages() -> None:
    """验证取消时未写出的消息被丢弃且仍会关闭后端。"""
    gate = threading.Event()
    backend = _FakeBackend(gate=gate)
    writer = QueuedBagWriter(backend, 100, 10_000)
    writer.start("uri", _TOPICS)
    for index in range(20):
        writer.offer("/a", b"x", index)

    gate.set()
    writer.abort()

    assert len(backend.writes) < 20
    assert backend.closed


def test_abort_timeout_reports_live_thread_and_allows_retry() -> None:
    """验证关闭阻塞时取消返回超时，线程退出后重试可确认安全清理。"""
    close_started = threading.Event()
    release_close = threading.Event()

    class BlockedCloseBackend(_FakeBackend):
        """让关闭等待测试显式放行。"""

        def close(self):
            """进入关闭后等待放行，并记录最终关闭。"""
            close_started.set()
            if not release_close.wait(timeout=5):
                raise TimeoutError("测试未放行关闭")
            super().close()

    backend = BlockedCloseBackend()
    writer = QueuedBagWriter(backend, 100, 10_000)
    writer.start("uri", _TOPICS)
    try:
        result = writer.abort(0.01)
        assert close_started.wait(timeout=1)
        assert not result.thread_stopped and not result.ok
        assert result.error_code == ERROR_FINISH_TIMEOUT
        assert not backend.closed
        release_close.set()
        retried = writer.abort(1)
        assert retried.thread_stopped and backend.closed
        assert retried.error_code == ERROR_FINISH_TIMEOUT
    finally:
        release_close.set()
        writer.abort(1)


def test_invalid_capacity_is_rejected() -> None:
    """验证非正容量在构造时报错。"""
    with pytest.raises(ValueError):
        QueuedBagWriter(_FakeBackend(), 0, 10)


def test_real_mcap_backend_roundtrip_preserves_topics_stamps_and_counts(
    tmp_path: Path,
) -> None:
    """验证真实 rosbag2 MCAP 写出后可读回话题、时间戳、数量和内容。"""
    uri = tmp_path / "bag"
    writer = QueuedBagWriter(RosbagMcapBackend(), 100, 1_000_000)
    writer.start(
        str(uri),
        [
            TopicSpec("/one", "std_msgs/msg/String"),
            TopicSpec("/two", "std_msgs/msg/String"),
        ],
    )
    expected = []
    for index in range(6):
        topic = "/one" if index % 2 == 0 else "/two"
        message = String(data=f"m{index}")
        stamp = 1_700_000_000_000_000_000 + index * 1_000_000
        assert writer.offer(topic, serialize_message(message), stamp)
        expected.append((topic, f"m{index}", stamp))
    result = writer.finish(10.0)

    assert result.ok
    assert result.message_counts == {"/one": 3, "/two": 3}
    info = rosbag2_py.Info().read_metadata(str(uri), "mcap")
    assert info.message_count == 6
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(uri), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    assert types == {
        "/one": "std_msgs/msg/String",
        "/two": "std_msgs/msg/String",
    }
    read_back = []
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        read_back.append((topic, deserialize_message(data, String).data, stamp))
    assert read_back == expected
