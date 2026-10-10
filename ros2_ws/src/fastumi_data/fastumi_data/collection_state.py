"""UMI 采集的生命周期状态机，编排存储、MCAP 写入和健康监控。

状态流转::

    idle -> starting -> recording -> stopping -> pending -> saving -> idle
                              |                      \\-> cancelling -> idle
                              \\-> cancelling -> idle     (取消经清理返回 idle)
    任何写入故障 -> error (只能取消，失败数据不会被保存为成功记录)

组合操作由本控制器串行执行：``stop_and_save`` 直接 ``stopping -> saving``，
``stop_and_cancel`` 直接 ``cancelling``，期间不会暴露可被其他请求插入的中间
状态。本模块不依赖 ROS，时钟、存储后端和事件编码均可注入。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import enum
import hashlib
import math
from pathlib import Path
import shutil
import threading
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
import uuid as uuid_module

import yaml

from fastumi_data.cdr_header import read_image_summary
from fastumi_data.collection_health import (
    STREAMS,
    CollectionHealthMonitor,
    HealthSnapshot,
    preflight_issues,
)
from fastumi_data.collection_storage import (
    CollectionStorage,
    SavedRecord,
    StagedCollection,
    StorageError,
    atomic_write_yaml,
    directory_size_bytes,
    parse_uuid,
    validate_collection_name,
    validate_task_name,
)
from fastumi_data.collection_writer import (
    BagBackend,
    QueuedBagWriter,
    TopicSpec,
    WriterResult,
)
from fastumi_data.extrinsic import (
    calibration_passes_acceptance,
    load_tracker_tcp_extrinsic,
)


# 与 fastumi_interfaces/EpisodeEvent 一致的事件类型。
EVENT_START = 1
EVENT_STOP = 2
# 会话清单版本；3 起以 collection_uuid 标识单条采集。
SESSION_SCHEMA_VERSION = 3
# 写入队列使用率超过该比例时报警。
QUEUE_HIGH_RATIO = 0.8


class CollectionState(str, enum.Enum):
    """采集后端的权威生命周期状态。"""

    IDLE = "idle"
    STARTING = "starting"
    RECORDING = "recording"
    STOPPING = "stopping"
    PENDING = "pending"
    SAVING = "saving"
    CANCELLING = "cancelling"
    ERROR = "error"


@dataclass(frozen=True)
class TopicMap:
    """采集使用的话题映射；映射同时写入处理配置快照。"""

    image: str = "/umi_camera/image_raw"
    tracker_pose: str = "/vive_tracker/pose"
    tracker_status: str = "/vive_tracker/status"
    gripper_state: str = "/gripper/state"
    episode_event: str = "/fastumi/episode/events"
    tf_static: str = "/tf_static"

    def specs(self) -> List[TopicSpec]:
        """返回写入 MCAP 的全部话题及消息类型。"""
        return [
            TopicSpec(self.image, "sensor_msgs/msg/Image"),
            TopicSpec(self.tracker_pose, "geometry_msgs/msg/PoseStamped"),
            TopicSpec(self.tracker_status, "fastumi_interfaces/msg/TrackerStatus"),
            TopicSpec(self.gripper_state, "fastumi_interfaces/msg/GripperState"),
            TopicSpec(self.episode_event, "fastumi_interfaces/msg/EpisodeEvent"),
            TopicSpec(self.tf_static, "tf2_msgs/msg/TFMessage"),
        ]

    def processing_topics(self) -> Dict[str, str]:
        """返回转换器 ``processing.yaml`` 中的 ``topics`` 映射。"""
        return {
            "image": self.image,
            "tracker_pose": self.tracker_pose,
            "tracker_status": self.tracker_status,
            "gripper_state": self.gripper_state,
            "episode_event": self.episode_event,
        }


@dataclass(frozen=True)
class CollectionConfig:
    """控制器配置。"""

    topics: TopicMap = field(default_factory=TopicMap)
    queue_max_messages: int = 600
    queue_max_bytes: int = 768 * 1024 * 1024
    finish_timeout_s: float = 60.0
    min_free_disk_bytes: int = 2 * 1024 ** 3
    processing_template: str = ""
    """默认处理配置路径；为空时使用内置最小模板。"""
    extrinsic_path: str = ""
    """可选 Tracker 到 TCP 外参；为空时记录为未标定。"""
    snapshot_files: Tuple[str, ...] = ()
    """额外复制到标定快照目录的文件，例如相机内参和夹爪标定。"""
    idempotency_cache_size: int = 256
    default_list_limit: int = 50
    max_list_limit: int = 200


@dataclass(frozen=True)
class OperationResult:
    """修改类操作的统一结果。"""

    request_id: str
    accepted: bool
    completed: bool
    code: str
    message: str
    collection_uuid: str
    state: str
    state_version: int


@dataclass(frozen=True)
class ListResult:
    """已保存记录列表结果。"""

    success: bool
    code: str
    message: str
    records: Tuple[SavedRecord, ...] = ()
    total: int = 0


@dataclass(frozen=True)
class StatusSnapshot:
    """权威状态快照，节点据此生成 ``CollectionStatus``。"""

    state_version: int
    state: str
    collection_uuid: str
    task_name: str
    name: str
    last_request_id: str
    last_operation: str
    last_result_code: str
    last_error: str
    can_start: bool
    can_stop: bool
    can_save: bool
    can_cancel: bool
    can_stop_and_save: bool
    can_stop_and_cancel: bool
    start_blockers: Tuple[str, ...]
    duration_s: float
    health: HealthSnapshot
    image_messages: int
    tracker_messages: int
    gripper_messages: int
    writer_queue_depth: int
    writer_queue_capacity: int
    free_disk_bytes: int
    calibration_status: str
    alarms: Tuple[str, ...]


@dataclass
class _Running:
    """在线累计 n/均值/最小/最大。"""

    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        """加入一个有限样本，NaN 被忽略。"""
        if not math.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def to_dict(self) -> Optional[Dict[str, float]]:
        """无样本时返回 ``None``。"""
        if not self.count:
            return None
        return {
            "samples": self.count,
            "mean": round(self.total / self.count, 3),
            "min": round(self.minimum, 3),
            "max": round(self.maximum, 3),
        }


class QualityAccumulator:
    """采集期间累计报警和时间指标，停止时写入 ``quality.yaml``。"""

    def __init__(self) -> None:
        """创建空累计器。"""
        self._alarms: Dict[str, Dict[str, Any]] = {}
        self._active: set = set()
        self._timing: Dict[str, _Running] = {}

    def update(
        self, elapsed_s: float, snapshot: HealthSnapshot, extra_alarms: Sequence[str]
    ) -> None:
        """累计一次周期检查的报警跃迁和时间指标。"""
        current = set(snapshot.alarms) | set(extra_alarms)
        for code in current - self._active:
            entry = self._alarms.setdefault(
                code, {"count": 0, "first_s": round(elapsed_s, 3)}
            )
            entry["count"] += 1
        for code in current:
            self._alarms[code]["last_s"] = round(elapsed_s, 3)
        self._active = current
        for name in STREAMS:
            stream = snapshot.stream(name)
            for metric in (
                "source_delta_ms",
                "message_age_ms",
                "pair_arrival_delta_ms",
                "rate_hz",
            ):
                self._timing.setdefault(
                    f"{name}.{metric}", _Running()
                ).add(getattr(stream, metric))
        self._timing.setdefault("gripper.lag_ms", _Running()).add(
            snapshot.gripper_lag_ms
        )

    @property
    def has_alarms(self) -> bool:
        """采集期间是否出现过任何报警。"""
        return bool(self._alarms)

    def to_dict(self) -> Dict[str, Any]:
        """导出可直接写入 YAML 的统计。"""
        return {
            "has_alarms": self.has_alarms,
            "alarms": dict(sorted(self._alarms.items())),
            "timing": {
                key: stats
                for key, running in sorted(self._timing.items())
                if (stats := running.to_dict()) is not None
            },
            "timing_note": (
                "pair_arrival_delta_ms 为配对双方到达采集节点的时间差，"
                "包含传输与排队，不是纯算法耗时。"
            ),
        }


@dataclass
class _Active:
    """当前活动或待保存采集的内部状态。"""

    collection_uuid: str
    task_name: str
    name: str
    staged: StagedCollection
    created_at: datetime
    started_wall_ns: int
    started_mono_ns: int
    writer: Optional[QueuedBagWriter] = None
    stopped_mono_ns: Optional[int] = None
    stopped_wall_ns: Optional[int] = None
    last_log_ns: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    quality: QualityAccumulator = field(default_factory=QualityAccumulator)
    calibration_status: str = "uncalibrated"
    manifest: Dict[str, Any] = field(default_factory=dict)


def _hash_file(path: Path) -> str:
    """计算文件 SHA-256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CollectionController:
    """采集生命周期控制器；所有公共方法线程安全。"""

    def __init__(
        self,
        storage: CollectionStorage,
        config: CollectionConfig,
        health: CollectionHealthMonitor,
        backend_factory: Callable[[], BagBackend],
        event_encoder: Callable[[int, str, str, int], bytes],
        wall_clock_ns: Callable[[], int],
        mono_clock_ns: Callable[[], int],
        discovery: Callable[[], Mapping[str, bool]],
        uuid_factory: Callable[[], uuid_module.UUID] = uuid_module.uuid4,
        on_change: Optional[Callable[[], None]] = None,
    ) -> None:
        """注入全部外部依赖，不执行任何 IO。

        Args:
            storage: 已持有（或稍后持有）根目录锁的存储。
            config: 控制器配置。
            health: 健康监控；本类用内部锁串行化对它的访问。
            backend_factory: 每次采集创建一个新的 MCAP 后端。
            event_encoder: ``(事件类型, session_id, 任务名, 纳秒时间戳) -> CDR 字节``。
            wall_clock_ns: 与消息源时间戳同域的墙钟。
            mono_clock_ns: 单调时钟。
            discovery: 返回三路输入当前是否存在发布者。
            uuid_factory: UUID 生成器，测试中可固定。
            on_change: 状态版本变化后的回调，在锁外执行。
        """
        self._storage = storage
        self._config = config
        self._health = health
        self._backend_factory = backend_factory
        self._encode_event = event_encoder
        self._wall_ns = wall_clock_ns
        self._mono_ns = mono_clock_ns
        self._discovery = discovery
        self._uuid_factory = uuid_factory
        self._on_change = on_change
        # 状态锁：保护状态、版本、结果缓存和活动采集引用。
        self._lock = threading.RLock()
        # 数据锁：保护“允许写入”标志与入队的原子性。
        self._data_lock = threading.Lock()
        # 健康锁：健康监控本身不线程安全。
        self._health_lock = threading.Lock()
        # 质量锁：周期 tick 与停止时的补记可能并发更新累计器。
        self._quality_lock = threading.Lock()
        self._state = CollectionState.IDLE
        self._version = 1
        self._active: Optional[_Active] = None
        self._admitting = False
        self._results: "OrderedDict[str, Tuple[Tuple, OperationResult]]" = (
            OrderedDict()
        )
        self._inflight: Dict[str, Tuple] = {}
        self._last = ("", "", "", "")
        self._writer_error: Optional[Tuple[str, str]] = None
        self._tf_static: List[bytes] = []
        self._extra_alarms: Tuple[str, ...] = ()

    # ------------------------------------------------------------------
    # 数据入口：订阅回调调用。
    # ------------------------------------------------------------------
    def on_image(self, data: bytes) -> None:
        """处理一帧原始图像字节：更新健康并在采集中入队。"""
        summary = read_image_summary(data)
        if summary is None:
            return
        with self._health_lock:
            self._health.observe_image(summary.stamp_ns, summary.valid)
        self._submit(self._config.topics.image, data)

    def on_tracker_pose(self, data: bytes, stamp_ns: int, finite: bool) -> None:
        """处理一条 Tracker 位姿字节。"""
        with self._health_lock:
            self._health.observe_tracker_pose(stamp_ns, finite)
        self._submit(self._config.topics.tracker_pose, data)

    def on_tracker_status(
        self,
        data: bytes,
        device_connected: bool,
        pose_valid: bool,
        tracking_state: int,
    ) -> None:
        """处理一条 Tracker 状态字节。"""
        with self._health_lock:
            self._health.observe_tracker_status(
                device_connected, pose_valid, tracking_state
            )
        self._submit(self._config.topics.tracker_status, data)

    def on_gripper_state(
        self, data: bytes, stamp_ns: int, valid: bool, openness: float
    ) -> None:
        """处理一条夹爪状态字节。"""
        with self._health_lock:
            self._health.observe_gripper(stamp_ns, valid, openness)
        self._submit(self._config.topics.gripper_state, data)

    def on_tf_static(self, data: bytes) -> None:
        """缓存静态 TF，并在采集中写入；每条新采集开始时回放缓存。"""
        with self._lock:
            if data not in self._tf_static:
                self._tf_static.append(data)
                del self._tf_static[:-64]
        self._submit(self._config.topics.tf_static, data)

    def _submit(self, topic: str, data: bytes) -> None:
        """仅在允许写入时把消息放入写入队列。"""
        with self._data_lock:
            if not self._admitting:
                return
            active = self._active
            if active is None or active.writer is None:
                return
            log_ns = max(self._wall_ns(), active.last_log_ns)
            if active.writer.offer(topic, data, log_ns):
                active.last_log_ns = log_ns
                active.counts[topic] = active.counts.get(topic, 0) + 1

    def _on_writer_error(self, code: str, message: str) -> None:
        """写入线程或入队线程报告故障；只记录，状态切换延后到锁内同步。"""
        if self._writer_error is None:
            self._writer_error = (code, message)

    # ------------------------------------------------------------------
    # 状态与结果辅助。
    # ------------------------------------------------------------------
    def _sync_writer_error(self) -> bool:
        """把后台故障应用到状态；返回是否发生了状态切换。"""
        notify = False
        with self._lock:
            error = self._writer_error
            if (
                error is not None
                and self._state is CollectionState.RECORDING
                and self._active is not None
            ):
                with self._data_lock:
                    self._admitting = False
                self._state = CollectionState.ERROR
                self._last = (*self._last[:2], error[0], error[1])
                self._version += 1
                notify = True
        if notify:
            self._notify()
        return notify

    def _notify(self) -> None:
        """在锁外通知观察者状态已变化。"""
        if self._on_change is not None:
            self._on_change()

    def _set_state(self, state: CollectionState) -> None:
        """切换状态并递增版本；调用方负责随后调用 ``_notify``。"""
        with self._lock:
            self._state = state
            self._version += 1

    def _result(
        self,
        request_id: str,
        accepted: bool,
        completed: bool,
        code: str,
        message: str,
        collection_uuid: str = "",
    ) -> OperationResult:
        """用当前权威状态构造结果。"""
        with self._lock:
            return OperationResult(
                request_id=request_id,
                accepted=accepted,
                completed=completed,
                code=code,
                message=message,
                collection_uuid=collection_uuid,
                state=self._state.value,
                state_version=self._version,
            )

    def _reject(
        self, request_id: str, code: str, message: str, collection_uuid: str = ""
    ) -> OperationResult:
        """构造被拒绝的结果。"""
        return self._result(request_id, False, False, code, message, collection_uuid)

    def _run(
        self,
        request_id: str,
        operation: str,
        signature: Tuple,
        body: Callable[[], OperationResult],
    ) -> OperationResult:
        """串行化重复请求：缓存首次结果，冲突与在途请求返回明确码。"""
        request_id = request_id.strip()
        if not request_id:
            return self._reject("", "INVALID_REQUEST_ID", "request_id 不能为空")
        self._sync_writer_error()
        with self._lock:
            cached = self._results.get(request_id)
            if cached is not None:
                if cached[0] != signature:
                    return self._reject(
                        request_id,
                        "REQUEST_ID_CONFLICT",
                        "request_id 已用于另一个不同的请求",
                    )
                return cached[1]
            inflight = self._inflight.get(request_id)
            if inflight is not None:
                if inflight != signature:
                    return self._reject(
                        request_id,
                        "REQUEST_ID_CONFLICT",
                        "request_id 已用于另一个不同的请求",
                    )
                return self._result(
                    request_id, True, False, "IN_PROGRESS", "相同请求仍在执行"
                )
            self._inflight[request_id] = signature
        try:
            result = body()
        except Exception as error:  # 防止未预期异常使请求永远在途。
            result = self._result(
                request_id, True, False, "INTERNAL_ERROR", f"内部错误: {error}"
            )
        with self._lock:
            self._inflight.pop(request_id, None)
            result = replace(
                result, state=self._state.value, state_version=self._version
            )
            self._results[request_id] = (signature, result)
            while len(self._results) > self._config.idempotency_cache_size:
                self._results.popitem(last=False)
            failed = not (result.accepted and result.completed)
            self._last = (
                request_id,
                operation,
                result.code,
                result.message if failed else "",
            )
            self._version += 1
            result = replace(result, state_version=self._version)
            self._results[request_id] = (signature, result)
        self._notify()
        return result

    # ------------------------------------------------------------------
    # 开始。
    # ------------------------------------------------------------------
    def start(self, request_id: str, task_name: str, name: str) -> OperationResult:
        """开始新采集，成功后进入 ``recording``。"""
        signature = ("start", task_name.strip(), name.strip())
        return self._run(
            request_id,
            "start",
            signature,
            lambda: self._start(request_id.strip(), task_name, name),
        )

    def _start(self, request_id: str, task_name: str, name: str) -> OperationResult:
        """执行开始请求的校验、预检、暂存与首批写入。"""
        try:
            task = validate_task_name(task_name)
            label = validate_collection_name(name)
        except StorageError as error:
            return self._reject(request_id, error.code, error.message)
        rejection = self._start_state_rejection(request_id)
        if rejection is not None:
            return rejection
        blockers = self._start_blockers()
        if blockers:
            return self._reject(
                request_id, "PREFLIGHT_FAILED", "；".join(m for _, m in blockers)
            )
        with self._lock:
            rejection = self._start_state_rejection(request_id)
            if rejection is not None:
                return rejection
            collection_uuid = str(self._uuid_factory())
            self._set_state(CollectionState.STARTING)
        self._notify()
        staged: Optional[StagedCollection] = None
        writer: Optional[QueuedBagWriter] = None
        try:
            staged = self._storage.create_staging(collection_uuid, task)
            created_at = datetime.now(timezone.utc)
            manifest = self._write_snapshots(staged, task, label, created_at)
            writer = QueuedBagWriter(
                self._backend_factory(),
                self._config.queue_max_messages,
                self._config.queue_max_bytes,
                on_error=self._on_writer_error,
            )
            with self._lock:
                self._writer_error = None
            writer.start(str(staged.bag_uri), self._config.topics.specs())
            started_wall = self._wall_ns()
            active = _Active(
                collection_uuid=collection_uuid,
                task_name=task,
                name=label,
                staged=staged,
                created_at=created_at,
                started_wall_ns=started_wall,
                started_mono_ns=self._mono_ns(),
                writer=writer,
                last_log_ns=started_wall,
                calibration_status=manifest["calibration_status"],
                manifest=manifest,
            )
            with self._lock:
                self._active = active
            # START 先于任何传感器数据入队，随后回放缓存的静态 TF。
            start_event = self._encode_event(
                EVENT_START, collection_uuid, task, started_wall
            )
            if not writer.offer(
                self._config.topics.episode_event, start_event, started_wall
            ):
                raise RuntimeError("START 事件未能写入队列")
            with self._lock:
                tf_cache = list(self._tf_static)
            active.counts[self._config.topics.episode_event] = 1
            for data in tf_cache:
                if writer.offer(self._config.topics.tf_static, data, started_wall):
                    tf_topic = self._config.topics.tf_static
                    active.counts[tf_topic] = active.counts.get(tf_topic, 0) + 1
            with self._data_lock:
                self._admitting = True
            self._set_state(CollectionState.RECORDING)
        except Exception as error:
            with self._data_lock:
                self._admitting = False
            if writer is not None:
                if not writer.abort().thread_stopped:
                    self._set_state(CollectionState.ERROR)
                    return self._result(
                        request_id, True, False, "START_FAILED",
                        f"无法开始采集，写入线程仍未退出，请稍后取消: {error}",
                        collection_uuid,
                    )
            if staged is not None:
                try:
                    self._storage.discard(staged)
                except StorageError:
                    pass
            with self._data_lock:
                self._admitting = False
            with self._lock:
                self._active = None
            self._set_state(CollectionState.IDLE)
            code = error.code if isinstance(error, StorageError) else "START_FAILED"
            return self._result(
                request_id, True, False, code, f"无法开始采集: {error}"
            )
        return self._result(
            request_id, True, True, "OK", "采集已开始", collection_uuid
        )

    def _start_state_rejection(self, request_id: str) -> Optional[OperationResult]:
        """开始请求与当前状态冲突时返回拒绝结果。"""
        with self._lock:
            if self._state is CollectionState.IDLE:
                return None
            if self._state is CollectionState.PENDING:
                return self._reject(
                    request_id,
                    "PENDING_RECORD",
                    "存在待保存采集，请先保存或取消",
                    self._active.collection_uuid if self._active else "",
                )
            if self._state is CollectionState.ERROR:
                return self._reject(
                    request_id, "ERROR_STATE", "上一次采集失败，请先取消并清理"
                )
            return self._reject(
                request_id,
                "STATE_CONFLICT",
                f"当前状态为 {self._state.value}，不能开始新采集",
            )

    def _start_blockers(self) -> List[Tuple[str, str]]:
        """汇总开始前检查：三路输入、磁盘空间。"""
        with self._health_lock:
            health = self._health.snapshot(self._discovery())
        try:
            free = self._storage.free_disk_bytes()
        except OSError:
            free = 0
        return self._blockers_from(health, free)

    def _blockers_from(
        self, health: HealthSnapshot, free: int
    ) -> List[Tuple[str, str]]:
        """用已计算的健康快照和磁盘空间生成阻塞原因，避免重复采样。"""
        blockers = [
            (issue.code, issue.message)
            for issue in preflight_issues(health, self._health.config)
        ]
        if free < self._config.min_free_disk_bytes:
            blockers.append(
                (
                    "DISK_LOW",
                    f"磁盘可用空间 {free / 1024 ** 3:.1f} GiB 低于 "
                    f"{self._config.min_free_disk_bytes / 1024 ** 3:.1f} GiB",
                )
            )
        return blockers

    def _write_snapshots(
        self,
        staged: StagedCollection,
        task: str,
        name: str,
        created_at: datetime,
    ) -> Dict[str, Any]:
        """写入实际处理配置、外参及额外标定快照，返回会话清单骨架。"""
        topics = self._config.topics
        processing: Dict[str, Any] = {}
        template = self._config.processing_template
        if template:
            loaded = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                processing = loaded
        processing["topics"] = {
            **(processing.get("topics") or {}),
            **topics.processing_topics(),
        }
        processing_path = staged.snapshot_dir / "processing.yaml"
        atomic_write_yaml(processing_path, processing)
        manifest: Dict[str, Any] = {
            "schema_version": SESSION_SCHEMA_VERSION,
            "collection_uuid": staged.uuid,
            "session_id": staged.uuid,
            "episode_index": 0,
            "task_name": task,
            "name": name,
            "dir_name": staged.dir_name,
            "created_at": created_at.isoformat(),
            "created_at_unix": created_at.timestamp(),
            "storage_id": "mcap",
            "bag_uri": "raw/bag",
            "topics": [spec.name for spec in topics.specs()],
            "topic_map": topics.processing_topics(),
            "processing_config": {
                "path": "calibration_snapshot/processing.yaml",
                "sha256": _hash_file(processing_path),
            },
            "calibration_status": "uncalibrated",
            "tracker_to_tcp": None,
            "additional_snapshots": [],
            "missing_snapshots": [],
        }
        extrinsic_path = self._config.extrinsic_path
        if extrinsic_path:
            extrinsic = load_tracker_tcp_extrinsic(extrinsic_path)
            if not calibration_passes_acceptance(extrinsic.metadata):
                raise StorageError(
                    "CALIBRATION_INVALID", "Tracker 到 TCP 外参未通过 2 mm、1° 验收"
                )
            destination = staged.snapshot_dir / "tracker_to_tcp.yaml"
            shutil.copy2(extrinsic_path, destination)
            manifest["calibration_status"] = "calibrated"
            manifest["tracker_to_tcp"] = {
                "path": "calibration_snapshot/tracker_to_tcp.yaml",
                "sha256": _hash_file(destination),
            }
        used_names = {"processing.yaml", "tracker_to_tcp.yaml"}
        for source_text in self._config.snapshot_files:
            source = Path(source_text)
            if not source.is_file():
                manifest["missing_snapshots"].append(source_text)
                continue
            target_name = source.name
            if target_name in used_names:
                raise StorageError(
                    "SNAPSHOT_CONFLICT", f"快照文件重名: {target_name}"
                )
            used_names.add(target_name)
            destination = staged.snapshot_dir / target_name
            shutil.copy2(source, destination)
            manifest["additional_snapshots"].append(
                {
                    "path": f"calibration_snapshot/{target_name}",
                    "sha256": _hash_file(destination),
                }
            )
        return manifest

    # ------------------------------------------------------------------
    # 停止、保存、取消。
    # ------------------------------------------------------------------
    def _uuid_ops_signature(self, operation: str, collection_uuid: str) -> Tuple:
        """修改类请求的幂等签名。"""
        try:
            return (operation, parse_uuid(collection_uuid))
        except StorageError:
            return (operation, collection_uuid)

    def _check_target(
        self,
        request_id: str,
        collection_uuid: str,
        allowed: Sequence[CollectionState],
        error_allowed: bool = False,
    ) -> Tuple[Optional[OperationResult], str]:
        """校验目标 UUID 与当前状态，返回拒绝结果或规范化 UUID。"""
        try:
            identifier = parse_uuid(collection_uuid)
        except StorageError as error:
            return self._reject(request_id, error.code, error.message), ""
        with self._lock:
            state = self._state
            active = self._active
            if state is CollectionState.IDLE or active is None:
                return (
                    self._reject(
                        request_id, "NO_ACTIVE_COLLECTION", "当前没有活动或待保存采集"
                    ),
                    identifier,
                )
            if identifier != active.collection_uuid:
                return (
                    self._reject(
                        request_id,
                        "UUID_MISMATCH",
                        "目标 UUID 与当前采集不一致",
                        active.collection_uuid,
                    ),
                    identifier,
                )
            if state in allowed or (error_allowed and state is CollectionState.ERROR):
                return None, identifier
            if state is CollectionState.ERROR:
                return (
                    self._reject(
                        request_id,
                        "ERROR_STATE",
                        "采集已失败，只能取消并清理",
                        identifier,
                    ),
                    identifier,
                )
            return (
                self._reject(
                    request_id,
                    "STATE_CONFLICT",
                    f"当前状态为 {state.value}，不能执行该操作",
                    identifier,
                ),
                identifier,
            )

    def stop(self, request_id: str, collection_uuid: str) -> OperationResult:
        """停止采集并进入 ``pending``，等待保存或取消。"""
        return self._run(
            request_id,
            "stop",
            self._uuid_ops_signature("stop", collection_uuid),
            lambda: self._stop(request_id.strip(), collection_uuid, then="pending"),
        )

    def stop_and_save(self, request_id: str, collection_uuid: str) -> OperationResult:
        """停止并保存，中间不暴露 ``pending``。"""
        return self._run(
            request_id,
            "stop_and_save",
            self._uuid_ops_signature("stop_and_save", collection_uuid),
            lambda: self._stop(request_id.strip(), collection_uuid, then="save"),
        )

    def stop_and_cancel(self, request_id: str, collection_uuid: str) -> OperationResult:
        """停止并取消；记录中直接放弃，失败状态则清理。"""
        return self._run(
            request_id,
            "stop_and_cancel",
            self._uuid_ops_signature("stop_and_cancel", collection_uuid),
            lambda: self._cancel(
                request_id.strip(),
                collection_uuid,
                allowed=(CollectionState.RECORDING,),
            ),
        )

    def cancel(self, request_id: str, collection_uuid: str) -> OperationResult:
        """取消待保存或失败的采集，清理暂存数据。"""
        return self._run(
            request_id,
            "cancel",
            self._uuid_ops_signature("cancel", collection_uuid),
            lambda: self._cancel(
                request_id.strip(),
                collection_uuid,
                allowed=(CollectionState.PENDING,),
            ),
        )

    def save(self, request_id: str, collection_uuid: str) -> OperationResult:
        """保存待保存采集，发布为正式记录。"""
        return self._run(
            request_id,
            "save",
            self._uuid_ops_signature("save", collection_uuid),
            lambda: self._save(request_id.strip(), collection_uuid),
        )

    def _stop(self, request_id: str, collection_uuid: str, then: str) -> OperationResult:
        """关闭 MCAP；``then`` 为 ``pending`` 或 ``save``。"""
        with self._lock:
            rejection, identifier = self._check_target(
                request_id, collection_uuid, (CollectionState.RECORDING,)
            )
            if rejection is not None:
                return rejection
            active = self._active
            self._set_state(CollectionState.STOPPING)
        self._notify()
        failure = self._close_recording(active)
        if failure is not None:
            return self._result(
                request_id, True, False, failure[0], failure[1], identifier
            )
        if then == "pending":
            self._set_state(CollectionState.PENDING)
            self._notify()
            return self._result(
                request_id, True, True, "OK", "采集已停止，等待保存或取消", identifier
            )
        self._set_state(CollectionState.SAVING)
        self._notify()
        return self._publish(request_id, active)

    def _close_recording(self, active: _Active) -> Optional[Tuple[str, str]]:
        """停止接纳、写 STOP、排空并关闭；成功后写出质量与清单。

        Returns:
            成功为 ``None``；失败为 ``(错误码, 说明)``，状态已置为 ``error``。
        """
        writer = active.writer
        assert writer is not None
        with self._data_lock:
            self._admitting = False
            # 在录制边界固定时长，排空和关闭耗时不计入采集统计。
            active.stopped_mono_ns = self._mono_ns()
            stop_wall = max(self._wall_ns(), active.last_log_ns)
            active.stopped_wall_ns = stop_wall
            stop_event = self._encode_event(
                EVENT_STOP, active.collection_uuid, active.task_name, stop_wall
            )
            queued = writer.offer(
                self._config.topics.episode_event, stop_event, stop_wall
            )
            if queued:
                active.counts[self._config.topics.episode_event] = (
                    active.counts.get(self._config.topics.episode_event, 0) + 1
                )
            active.last_log_ns = stop_wall
        self._accumulate_quality(active)
        result = writer.finish(self._config.finish_timeout_s)
        failure = self._verify_writer(active, result, queued)
        if failure is None:
            try:
                self._write_session_files(active)
            except OSError as error:
                failure = ("WRITE_FAILED", f"无法写出质量统计与会话清单: {error}")
        if failure is not None:
            with self._lock:
                self._writer_error = self._writer_error or failure
                self._last = (*self._last[:2], failure[0], failure[1])
            self._set_state(CollectionState.ERROR)
            self._notify()
        return failure

    def _verify_writer(
        self, active: _Active, result: WriterResult, stop_queued: bool
    ) -> Optional[Tuple[str, str]]:
        """确认写入器无故障且每条入队消息都已落盘。"""
        if not result.ok:
            return result.error_code, result.error_message
        if self._writer_error is not None:
            return self._writer_error
        if not stop_queued:
            return "WRITE_FAILED", "STOP 事件未能写入队列"
        if result.message_counts != active.counts:
            return (
                "COUNT_MISMATCH",
                f"写入消息数 {result.message_counts} 与入队消息数 {active.counts} 不一致",
            )
        return None

    def _write_session_files(self, active: _Active) -> None:
        """停止后写出 ``quality.yaml`` 和待保存的 ``session.yaml``。"""
        staged = active.staged
        duration_s = self._duration_s(active)
        topics = self._config.topics
        message_counts = {
            "image": active.counts.get(topics.image, 0),
            "tracker_pose": active.counts.get(topics.tracker_pose, 0),
            "tracker_status": active.counts.get(topics.tracker_status, 0),
            "gripper_state": active.counts.get(topics.gripper_state, 0),
            "episode_event": active.counts.get(topics.episode_event, 0),
            "tf_static": active.counts.get(topics.tf_static, 0),
        }
        rates = {
            key: round(count / duration_s, 3) if duration_s > 0 else 0.0
            for key, count in message_counts.items()
            if key in ("image", "tracker_pose", "tracker_status", "gripper_state")
        }
        with self._quality_lock:
            quality = active.quality.to_dict()
            has_alarms = active.quality.has_alarms
        quality.update({"message_counts": message_counts, "mean_rate_hz": rates})
        atomic_write_yaml(staged.root / "quality.yaml", quality)
        active.manifest.update(
            {
                "started_at": datetime.fromtimestamp(
                    active.started_wall_ns / 1e9, timezone.utc
                ).isoformat(),
                "stopped_at": datetime.fromtimestamp(
                    (active.stopped_wall_ns or active.started_wall_ns) / 1e9,
                    timezone.utc,
                ).isoformat(),
                "duration_s": round(duration_s, 3),
                "message_counts": message_counts,
                "quality": {
                    "path": "quality.yaml",
                    "has_alarms": has_alarms,
                },
                "state": "pending",
            }
        )
        atomic_write_yaml(staged.root / "session.yaml", active.manifest)

    def _save(self, request_id: str, collection_uuid: str) -> OperationResult:
        """保存待保存采集。"""
        with self._lock:
            rejection, identifier = self._check_target(
                request_id, collection_uuid, (CollectionState.PENDING,)
            )
            if rejection is not None:
                return rejection
            active = self._active
            self._set_state(CollectionState.SAVING)
        self._notify()
        return self._publish(request_id, active)

    def _publish(self, request_id: str, active: _Active) -> OperationResult:
        """补全清单并把暂存目录原子重命名为正式记录。"""
        try:
            saved_at = datetime.now(timezone.utc)
            active.manifest.update(
                {
                    "state": "saved",
                    "saved_at": saved_at.isoformat(),
                    "saved_at_unix": saved_at.timestamp(),
                }
            )
            active.manifest["size_bytes"] = directory_size_bytes(active.staged.root)
            atomic_write_yaml(active.staged.root / "session.yaml", active.manifest)
            destination = self._storage.publish(active.staged)
        except (StorageError, OSError) as error:
            code = error.code if isinstance(error, StorageError) else "SAVE_FAILED"
            with self._lock:
                self._last = (*self._last[:2], code, str(error))
            self._set_state(CollectionState.ERROR)
            self._notify()
            return self._result(
                request_id, True, False, code, f"保存失败: {error}",
                active.collection_uuid,
            )
        with self._lock:
            self._active = None
        self._set_state(CollectionState.IDLE)
        self._notify()
        relative = destination.relative_to(self._storage.root)
        return self._result(
            request_id, True, True, "OK", f"已保存到 {relative}", active.collection_uuid
        )

    def _cancel(
        self,
        request_id: str,
        collection_uuid: str,
        allowed: Sequence[CollectionState],
    ) -> OperationResult:
        """放弃当前采集并清理暂存数据；失败状态也允许清理。"""
        with self._lock:
            rejection, identifier = self._check_target(
                request_id, collection_uuid, allowed, error_allowed=True
            )
            if rejection is not None:
                return rejection
            active = self._active
            self._set_state(CollectionState.CANCELLING)
        self._notify()
        with self._data_lock:
            self._admitting = False
        try:
            if active.writer is not None:
                if not active.writer.abort().thread_stopped:
                    raise StorageError(
                        "CANCEL_TIMEOUT", "写入线程仍未退出，请稍后重试取消"
                    )
            self._storage.discard(active.staged)
        except (StorageError, OSError) as error:
            with self._lock:
                code = (
                    error.code if isinstance(error, StorageError) else "CANCEL_FAILED"
                )
                self._last = (*self._last[:2], code, str(error))
            self._set_state(CollectionState.ERROR)
            self._notify()
            return self._result(
                request_id, True, False, code, f"清理失败: {error}",
                identifier,
            )
        with self._lock:
            self._active = None
            self._writer_error = None
        self._set_state(CollectionState.IDLE)
        self._notify()
        return self._result(
            request_id, True, True, "OK", "采集已取消并清理", identifier
        )

    # ------------------------------------------------------------------
    # 删除与列表。
    # ------------------------------------------------------------------
    def delete(self, request_id: str, collection_uuid: str) -> OperationResult:
        """把已保存记录移入回收区；活动或待保存采集不可删除。"""
        return self._run(
            request_id,
            "delete",
            self._uuid_ops_signature("delete", collection_uuid),
            lambda: self._delete(request_id.strip(), collection_uuid),
        )

    def _delete(self, request_id: str, collection_uuid: str) -> OperationResult:
        """执行删除，仅接受已保存记录。"""
        try:
            identifier = parse_uuid(collection_uuid)
        except StorageError as error:
            return self._reject(request_id, error.code, error.message)
        with self._lock:
            if self._active is not None and self._active.collection_uuid == identifier:
                return self._reject(
                    request_id,
                    "NOT_SAVED",
                    "该采集尚未保存，请使用取消而不是删除",
                    identifier,
                )
        try:
            destination = self._storage.delete(identifier)
        except StorageError as error:
            return self._reject(request_id, error.code, error.message, identifier)
        return self._result(
            request_id,
            True,
            True,
            "OK",
            f"已移入回收区 {destination.name}",
            identifier,
        )

    def list_records(
        self, task_name: str = "", offset: int = 0, limit: int = 0
    ) -> ListResult:
        """分页列出已保存记录；limit 为 0 时使用默认值并受上限约束。"""
        effective = limit if limit > 0 else self._config.default_list_limit
        effective = min(effective, self._config.max_list_limit)
        try:
            records, total = self._storage.list_records(task_name, offset, effective)
        except StorageError as error:
            return ListResult(False, error.code, error.message)
        except OSError as error:
            return ListResult(False, "LIST_FAILED", f"无法读取数据集目录: {error}")
        return ListResult(True, "OK", "", tuple(records), total)

    # ------------------------------------------------------------------
    # 状态、周期检查与退出。
    # ------------------------------------------------------------------
    def _duration_s(self, active: Optional[_Active]) -> float:
        """采集已持续的秒数；停止后固定为停止时刻。"""
        if active is None:
            return 0.0
        end = (
            active.stopped_mono_ns
            if active.stopped_mono_ns is not None
            else self._mono_ns()
        )
        return max(0.0, (end - active.started_mono_ns) / 1e9)

    def tick(self) -> StatusSnapshot:
        """周期调用：应用后台故障、累计质量统计并返回最新状态。"""
        self._sync_writer_error()
        snapshot = self.status_snapshot()
        with self._lock:
            active = self._active
            recording = self._state is CollectionState.RECORDING
        if recording and active is not None:
            with self._quality_lock:
                active.quality.update(
                    self._duration_s(active), snapshot.health, self._extra_alarms
                )
        return snapshot

    def _accumulate_quality(self, active: _Active) -> None:
        """停止时补记一次质量统计，避免漏掉最后一个周期内的报警。"""
        with self._health_lock:
            health = self._health.snapshot(self._discovery())
        with self._quality_lock:
            active.quality.update(
                self._duration_s(active),
                health,
                self._extra_alarms,
            )

    def status_snapshot(self) -> StatusSnapshot:
        """生成权威状态快照，按钮可用性由状态决定。"""
        self._sync_writer_error()
        with self._health_lock:
            health = self._health.snapshot(self._discovery())
        try:
            free = self._storage.free_disk_bytes()
        except OSError:
            free = 0
        with self._lock:
            state = self._state
            active = self._active
            writer = active.writer if active else None
            version = self._version
            last = self._last
            if active is not None:
                # 锁序固定为 _lock -> _data_lock；计数随入队更新，复制需互斥。
                with self._data_lock:
                    counts = dict(active.counts)
            else:
                counts = {}
        depth = writer.depth if writer is not None else 0
        capacity = writer.capacity if writer is not None else self._config.queue_max_messages
        extra: List[str] = []
        if free < self._config.min_free_disk_bytes:
            extra.append("DISK_LOW")
        if state is CollectionState.RECORDING and depth > capacity * QUEUE_HIGH_RATIO:
            extra.append("WRITER_QUEUE_HIGH")
        self._extra_alarms = tuple(extra)
        blockers: List[str] = []
        if state is CollectionState.IDLE:
            blockers = [message for _, message in self._blockers_from(health, free)]
        topics = self._config.topics
        return StatusSnapshot(
            state_version=version,
            state=state.value,
            collection_uuid=active.collection_uuid if active else "",
            task_name=active.task_name if active else "",
            name=active.name if active else "",
            last_request_id=last[0],
            last_operation=last[1],
            last_result_code=last[2],
            last_error=last[3],
            can_start=state is CollectionState.IDLE and not blockers,
            can_stop=state is CollectionState.RECORDING,
            can_save=state is CollectionState.PENDING,
            can_cancel=state in (CollectionState.PENDING, CollectionState.ERROR),
            can_stop_and_save=state is CollectionState.RECORDING,
            can_stop_and_cancel=state is CollectionState.RECORDING,
            start_blockers=tuple(blockers),
            duration_s=self._duration_s(active),
            health=health,
            image_messages=counts.get(topics.image, 0),
            tracker_messages=counts.get(topics.tracker_pose, 0),
            gripper_messages=counts.get(topics.gripper_state, 0),
            writer_queue_depth=depth,
            writer_queue_capacity=capacity,
            free_disk_bytes=free,
            calibration_status=(
                active.calibration_status
                if active
                else ("calibrated" if self._config.extrinsic_path else "uncalibrated")
            ),
            alarms=tuple(sorted(set(health.alarms) | set(extra))),
        )

    def shutdown(self) -> List[str]:
        """有序退出：放弃未保存采集并释放根目录锁。

        Returns:
            被清理的暂存目录名。

        Raises:
            StorageError: 写入线程未退出时保留暂存数据和根目录锁，允许重试退出。
        """
        with self._data_lock:
            self._admitting = False
        with self._lock:
            active = self._active
        if active is not None and active.writer is not None:
            if not active.writer.abort().thread_stopped:
                self._set_state(CollectionState.ERROR)
                raise StorageError("CANCEL_TIMEOUT", "写入线程仍未退出，保留暂存数据")
        with self._lock:
            self._active = None
        removed = self._storage.release()
        with self._lock:
            self._state = CollectionState.IDLE
            self._version += 1
        return removed
