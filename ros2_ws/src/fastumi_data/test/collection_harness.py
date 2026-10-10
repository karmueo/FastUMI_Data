"""为采集控制器测试提供可控时钟、内存 MCAP 后端和就绪的控制器。"""

from dataclasses import dataclass, field
import threading
import uuid as uuid_module
from typing import List, Optional, Sequence, Tuple

from fastumi_data.collection_health import (
    GRIPPER,
    IMAGE,
    TRACKER,
    CollectionHealthMonitor,
    HealthConfig,
)
from fastumi_data.collection_state import (
    EVENT_START,
    EVENT_STOP,
    CollectionConfig,
    CollectionController,
)
from fastumi_data.collection_storage import CollectionStorage
from fastumi_data.collection_writer import TopicSpec


# 毫秒到纳秒的换算。
MS = 1_000_000
# 基准源时间戳（墙钟）。
T0 = 1_700_000_000_000_000_000


class Clock:
    """同时驱动单调时钟和墙钟的可控时钟。"""

    def __init__(self) -> None:
        """单调时钟从 0 开始，墙钟与源时间戳同域。"""
        self.mono = 0
        self.wall = T0

    def advance(self, milliseconds: float) -> None:
        """两个时钟同步前进。"""
        self.mono += int(milliseconds * MS)
        self.wall += int(milliseconds * MS)


class MemoryBackend:
    """记录写入内容，可按需失败或阻塞的内存后端。"""

    def __init__(
        self,
        fail_on_write: Optional[int] = None,
        fail_on_close: bool = False,
        write_gate: Optional[threading.Event] = None,
        close_gate: Optional[threading.Event] = None,
    ) -> None:
        """配置故障注入点。"""
        self.writes: List[Tuple[str, bytes, int]] = []
        self.closed = False
        self.uri = ""
        self.topics: Sequence[TopicSpec] = ()
        self._fail_on_write = fail_on_write
        self._fail_on_close = fail_on_close
        self._write_gate = write_gate
        self._close_gate = close_gate

    def open(self, uri: str, topics: Sequence[TopicSpec]) -> None:
        """记录 URI 与话题。"""
        self.uri, self.topics = uri, topics

    def write(self, topic: str, data: bytes, timestamp_ns: int) -> None:
        """记录写入，按配置失败或等待放行。"""
        if self._write_gate is not None:
            self._write_gate.wait(timeout=10)
        if self._fail_on_write is not None and len(self.writes) >= self._fail_on_write:
            raise OSError("磁盘已满")
        self.writes.append((topic, data, timestamp_ns))

    def close(self) -> None:
        """记录关闭，按配置失败或等待放行。"""
        if self._close_gate is not None:
            self._close_gate.wait(timeout=10)
        if self._fail_on_close:
            raise OSError("索引写出失败")
        self.closed = True


def encode_event(event_type: int, session_id: str, task: str, stamp_ns: int) -> bytes:
    """测试用事件编码：可读文本，便于断言 START/STOP 顺序。"""
    name = {EVENT_START: "START", EVENT_STOP: "STOP"}[event_type]
    return f"{name}|{session_id}|{task}|{stamp_ns}".encode()


@dataclass
class Harness:
    """组合好的控制器及其可观察的外部依赖。"""

    controller: CollectionController
    storage: CollectionStorage
    clock: Clock
    health: CollectionHealthMonitor
    backends: List[MemoryBackend] = field(default_factory=list)
    pending_backends: List[MemoryBackend] = field(default_factory=list)
    """按顺序被后端工厂取用的预置后端，用于故障注入。"""
    states: List[str] = field(default_factory=list)
    discovered: dict = field(
        default_factory=lambda: {IMAGE: True, TRACKER: True, GRIPPER: True}
    )

    def feed(self, frames: int = 3) -> None:
        """喂入若干帧健康的三路数据，使预检通过。"""
        self.health.observe_tracker_status(True, True, 3)
        for _ in range(frames):
            stamp = self.clock.wall
            self.health.observe_image(stamp, True)
            self.health.observe_tracker_pose(stamp, True)
            self.health.observe_gripper(stamp, True, 0.5)
            self.clock.advance(33)
            self.health.observe_tracker_status(True, True, 3)


def make_harness(
    tmp_path,
    config: Optional[CollectionConfig] = None,
    backend_factory=None,
    uuids: Optional[list] = None,
) -> Harness:
    """创建已取得根目录锁、预检即可通过的控制器。"""
    clock = Clock()
    health = CollectionHealthMonitor(
        HealthConfig(), lambda: clock.mono, lambda: clock.wall
    )
    storage = CollectionStorage(tmp_path / "dataset")
    storage.acquire()
    harness = Harness(
        controller=None,  # type: ignore[arg-type]
        storage=storage,
        clock=clock,
        health=health,
    )

    def default_factory() -> MemoryBackend:
        """默认工厂：优先取预置后端，否则新建并登记。"""
        backend = (
            harness.pending_backends.pop(0)
            if harness.pending_backends
            else MemoryBackend()
        )
        harness.backends.append(backend)
        return backend

    uuid_iter = iter(uuids) if uuids is not None else None
    controller = CollectionController(
        storage=storage,
        config=config or CollectionConfig(min_free_disk_bytes=0),
        health=health,
        backend_factory=backend_factory or default_factory,
        event_encoder=encode_event,
        wall_clock_ns=lambda: clock.wall,
        mono_clock_ns=lambda: clock.mono,
        discovery=lambda: dict(harness.discovered),
        uuid_factory=(
            (lambda: uuid_module.UUID(next(uuid_iter)))
            if uuid_iter is not None
            else uuid_module.uuid4
        ),
        on_change=lambda: harness.states.append(
            harness.controller.status_snapshot().state
        ),
    )
    harness.controller = controller
    harness.feed()
    return harness
