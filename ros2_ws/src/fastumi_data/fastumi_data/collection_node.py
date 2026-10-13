"""UMI 采集节点：编排服务接口、原始字节订阅、MCAP 写入和权威状态发布。

生命周期、存储、写入和健康逻辑位于不依赖 ROS 的模块；本节点只负责把
ROS 接口映射到 :class:`CollectionController`。图像等消息以原始 CDR 字节
订阅并直接写入 MCAP，避免对每帧数 MB 数据做 Python 对象转换。
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from ament_index_python.packages import get_package_share_directory
from fastumi_interfaces.msg import (
    CollectionRecord,
    CollectionResult,
    CollectionStatus,
    CollectionStreamHealth,
    EpisodeEvent,
    GripperState,
    TrackerStatus,
)
from fastumi_interfaces.srv import (
    CollectionCancel,
    CollectionDelete,
    CollectionGetStatus,
    CollectionList,
    CollectionSave,
    CollectionStart,
    CollectionStop,
    CollectionStopAndCancel,
    CollectionStopAndSave,
)
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import (
    ExternalShutdownException,
    MultiThreadedExecutor,
    SingleThreadedExecutor,
)
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import TimeReference
from tf2_msgs.msg import TFMessage

from fastumi_data.collection_health import (
    GRIPPER,
    IMAGE,
    TRACKER,
    CollectionHealthMonitor,
    HealthConfig,
    StreamHealth,
)
from fastumi_data.collection_state import (
    CollectionConfig,
    CollectionController,
    ListResult,
    OperationResult,
    StatusSnapshot,
    TopicMap,
)
from fastumi_data.collection_storage import CollectionStorage, SavedRecord, StorageError
from fastumi_data.collection_writer import RosbagMcapBackend
from fastumi_data.image_qos import image_qos_profile
from fastumi_data.image_modes import (
    DEFAULT_DECODED_IMAGE_TOPIC,
    DEFAULT_H264_START_TIMEOUT_S,
    DEFAULT_IMAGE_MODE,
    DEFAULT_IMAGE_QOS_DEPTH,
    DEFAULT_IMAGE_RELIABILITY,
    RELIABILITY_RELIABLE,
    message_type_for_mode,
    resolve_image_topic,
    validate_image_mode,
    validate_image_qos_depth,
    validate_image_reliability,
)


# 采集服务与状态话题的命名空间。
NAMESPACE = "/fastumi/collection"
# 状态发布周期内的健康刷新频率默认值。
DEFAULT_STATUS_RATE_HZ = 2.0
# 服务执行器线程数；停止等长耗时请求会各占一个线程。
SERVICE_THREADS = 4
# Quaternion 范数偏离 1 的容忍度。
_QUATERNION_TOLERANCE = 1e-3


def stream_to_msg(stream: StreamHealth) -> CollectionStreamHealth:
    """把单路健康快照转换为消息。"""
    message = CollectionStreamHealth()
    message.name = stream.name
    message.discovered = stream.discovered
    message.fresh = stream.fresh
    message.valid = stream.valid
    message.matched = stream.matched
    message.age_s = float(stream.age_s)
    message.rate_hz = float(stream.rate_hz)
    message.source_delta_ms = float(stream.source_delta_ms)
    message.message_age_ms = float(stream.message_age_ms)
    message.pair_arrival_delta_ms = float(stream.pair_arrival_delta_ms)
    message.received = int(stream.received)
    message.out_of_order = int(stream.out_of_order)
    message.detail = stream.detail or _stream_format_text(stream)
    return message


def _stream_format_text(stream: StreamHealth) -> str:
    """没有异常时，图像流的 detail 携带格式和 h264 关键帧等待信息。"""
    if not stream.format:
        return ""
    if stream.awaiting_keyframe:
        return f"{stream.format}，尚未收到 SPS/PPS+IDR 关键帧"
    if stream.keyframes:
        return f"{stream.format}，已收到 {stream.keyframes} 个关键帧"
    return stream.format


def status_to_msg(snapshot: StatusSnapshot, stamp: Any = None) -> CollectionStatus:
    """把状态快照转换为 ``CollectionStatus`` 消息。"""
    message = CollectionStatus()
    if stamp is not None:
        message.header.stamp = stamp
    message.header.frame_id = "system_time"
    message.state_version = snapshot.state_version
    message.state = snapshot.state
    message.collection_uuid = snapshot.collection_uuid
    message.task_name = snapshot.task_name
    message.name = snapshot.name
    message.last_request_id = snapshot.last_request_id
    message.last_operation = snapshot.last_operation
    message.last_result_code = snapshot.last_result_code
    message.last_error = snapshot.last_error
    message.can_start = snapshot.can_start
    message.can_stop = snapshot.can_stop
    message.can_save = snapshot.can_save
    message.can_cancel = snapshot.can_cancel
    message.can_stop_and_save = snapshot.can_stop_and_save
    message.can_stop_and_cancel = snapshot.can_stop_and_cancel
    message.start_blockers = list(snapshot.start_blockers)
    message.duration_s = snapshot.duration_s
    message.image = stream_to_msg(snapshot.health.image)
    message.tracker = stream_to_msg(snapshot.health.tracker)
    message.gripper = stream_to_msg(snapshot.health.gripper)
    message.gripper_percent = float(snapshot.health.gripper_percent)
    message.gripper_valid = snapshot.health.gripper_valid
    message.tracker_tracking_ok = snapshot.health.tracker_tracking_ok
    message.tracker_tracking_state = int(snapshot.health.tracker_tracking_state)
    message.image_messages = snapshot.image_messages
    message.tracker_messages = snapshot.tracker_messages
    message.gripper_messages = snapshot.gripper_messages
    message.writer_queue_depth = snapshot.writer_queue_depth
    message.writer_queue_capacity = snapshot.writer_queue_capacity
    message.free_disk_bytes = snapshot.free_disk_bytes
    message.calibration_status = snapshot.calibration_status
    message.alarms = list(snapshot.alarms)
    message.image_mode = snapshot.image_mode
    message.image_format = snapshot.image_format
    return message


def result_to_msg(result: OperationResult) -> CollectionResult:
    """把操作结果转换为 ``CollectionResult`` 消息。"""
    message = CollectionResult()
    message.request_id = result.request_id
    message.accepted = result.accepted
    message.completed = result.completed
    message.code = result.code
    message.message = result.message
    message.collection_uuid = result.collection_uuid
    message.state = result.state
    message.state_version = result.state_version
    return message


def record_to_msg(record: SavedRecord) -> CollectionRecord:
    """把已保存记录摘要转换为 ``CollectionRecord`` 消息。"""
    message = CollectionRecord()
    message.collection_uuid = record.uuid
    message.task_name = record.task_name
    message.name = record.name
    message.dir_name = record.dir_name
    message.relative_path = record.relative_path
    message.created_at = record.created_at
    message.saved_at = record.saved_at
    message.duration_s = record.duration_s
    message.image_messages = record.image_messages
    message.tracker_messages = record.tracker_messages
    message.gripper_messages = record.gripper_messages
    message.size_bytes = record.size_bytes
    message.calibrated = record.calibrated
    message.has_alarms = record.has_alarms
    return message


def encode_episode_event(
    event_type: int, session_id: str, task_name: str, stamp_ns: int
) -> bytes:
    """序列化 START/STOP 事件，session_id 使用采集 UUID，episode_index 为 0。"""
    message = EpisodeEvent()
    message.header.stamp.sec = int(stamp_ns // 1_000_000_000)
    message.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
    message.header.frame_id = "system_time"
    message.session_id = session_id
    message.task_name = task_name
    message.episode_index = 0
    message.event_type = event_type
    return serialize_message(message)


def timing_generation(source: str) -> int:
    """从 ``TimeReference.source`` 的 ``gen=N`` 字段读取时钟映射代号，缺失为 0。"""
    for item in str(source).split(";"):
        key, _, value = item.partition("=")
        if key.strip() == "gen":
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


def _pose_is_finite(message: PoseStamped) -> bool:
    """位姿各分量有限且四元数近似单位长度。"""
    pose = message.pose
    values = (
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    if not all(math.isfinite(value) for value in values):
        return False
    norm = math.sqrt(sum(value * value for value in values[3:]))
    return abs(norm - 1.0) <= _QUATERNION_TOLERANCE


class CollectionNode(Node):
    """提供 ``/fastumi/collection`` 服务和权威状态的采集节点。"""

    def __init__(self, parameter_overrides: Optional[Sequence[Any]] = None) -> None:
        """声明参数、取得数据集根目录锁并创建全部 ROS 接口。

        Args:
            parameter_overrides: 可选参数覆盖，主要供测试在不修改全局 ROS
                参数的前提下配置节点。
        """
        super().__init__(
            "collection_node",
            parameter_overrides=(
                list(parameter_overrides) if parameter_overrides else None
            ),
        )
        self._declare_parameters()
        image_mode = validate_image_mode(self._text("image_mode"))
        self._image_reliability = validate_image_reliability(
            self._text("image_reliability")
        )
        self._image_qos_depth = validate_image_qos_depth(
            int(self.get_parameter("image_qos_depth").value)
        )
        topics = TopicMap(
            image_mode=image_mode,
            image=resolve_image_topic(image_mode, self._text("image_topic")),
            tracker_pose=self._text("tracker_pose_topic"),
            tracker_status=self._text("tracker_status_topic"),
            gripper_state=self._text("gripper_state_topic"),
            episode_event=self._text("episode_event_topic"),
            tf_static=self._text("tf_static_topic"),
            image_timing=self._text("image_timing_topic"),
        )
        self._topics = topics
        nominal_fps = float(self.get_parameter("nominal_source_fps").value)
        if not math.isfinite(nominal_fps) or nominal_fps < 0.0:
            raise ValueError(
                f"nominal_source_fps 必须为非负有限数，当前为 {nominal_fps!r}"
            )
        processing_template = self._text("processing_config") or str(
            Path(get_package_share_directory("fastumi_data"))
            / "config"
            / "processing.yaml"
        )
        config = CollectionConfig(
            topics=topics,
            queue_max_messages=int(self.get_parameter("queue_max_messages").value),
            queue_max_bytes=int(
                float(self.get_parameter("queue_max_mib").value) * 1024 * 1024
            ),
            finish_timeout_s=float(self.get_parameter("finish_timeout_s").value),
            h264_start_timeout_s=float(
                self.get_parameter("h264_start_timeout_s").value
            ),
            decoded_image_topic=self._text("decoded_image_topic"),
            image_reliability=self._image_reliability,
            image_qos_depth=self._image_qos_depth,
            nominal_source_fps=nominal_fps,
            nominal_source_fps_source=(
                self._text("nominal_source_fps_source") if nominal_fps > 0.0 else "unknown"
            ),
            image_timestamp_source=self._text("image_timestamp_source") or "unrecorded",
            min_free_disk_bytes=int(
                float(self.get_parameter("min_free_disk_gib").value) * 1024 ** 3
            ),
            processing_template=processing_template,
            extrinsic_path=self._text("extrinsic_path"),
            snapshot_files=tuple(
                path
                for path in self.get_parameter("snapshot_files").value
                if path.strip()
            ),
        )
        health = CollectionHealthMonitor(
            HealthConfig(
                image_timeout_s=float(self.get_parameter("image_timeout_s").value),
                tracker_timeout_s=float(self.get_parameter("tracker_timeout_s").value),
                gripper_timeout_s=float(self.get_parameter("gripper_timeout_s").value),
                match_window_s=float(self.get_parameter("match_window_s").value),
                rate_window_s=float(self.get_parameter("rate_window_s").value),
                image_mode=image_mode,
                expected_image_rate_hz=float(
                    self.get_parameter("expected_image_rate_hz").value
                ),
                expected_gripper_rate_hz=float(
                    self.get_parameter("expected_gripper_rate_hz").value
                ),
                min_rate_ratio=float(self.get_parameter("min_rate_ratio").value),
                low_rate_persist_s=float(
                    self.get_parameter("low_rate_persist_s").value
                ),
            ),
            mono_clock_ns=time.monotonic_ns,
            wall_clock_ns=lambda: self.get_clock().now().nanoseconds,
        )
        self._storage = CollectionStorage(Path(self._text("dataset_root")))
        try:
            removed = self._storage.acquire()
        except StorageError as error:
            raise RuntimeError(f"[{error.code}] {error.message}") from error
        if removed:
            self.get_logger().warning(f"已清理上次遗留的未保存采集: {removed}")
        self._controller = CollectionController(
            storage=self._storage,
            config=config,
            health=health,
            backend_factory=RosbagMcapBackend,
            event_encoder=encode_episode_event,
            wall_clock_ns=lambda: self.get_clock().now().nanoseconds,
            mono_clock_ns=time.monotonic_ns,
            discovery=self._discover,
            on_change=self._publish_status,
            image_check=self._check_image_publishers,
        )
        self._known_alarms: frozenset = frozenset()
        self._create_interfaces()
        status_period = 1.0 / max(
            0.1, float(self.get_parameter("status_rate_hz").value)
        )
        self._status_timer = self.create_timer(
            status_period, self._on_timer, callback_group=ReentrantCallbackGroup()
        )
        self.get_logger().info(
            f"采集服务已就绪，数据集根目录 {self._storage.root}，"
            f"图像话题 {topics.image}（{image_mode}，{self._image_reliability}，"
            f"深度 {self._image_qos_depth}）"
        )

    # ------------------------------------------------------------------
    # 参数与接口创建。
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        """声明全部节点参数及默认值。"""
        defaults = {
            "dataset_root": "dataset",
            "image_mode": DEFAULT_IMAGE_MODE,
            # 留空时按 image_mode 使用默认话题；显式配置优先。
            "image_topic": "",
            "decoded_image_topic": DEFAULT_DECODED_IMAGE_TOPIC,
            "image_reliability": DEFAULT_IMAGE_RELIABILITY,
            "image_qos_depth": DEFAULT_IMAGE_QOS_DEPTH,
            "h264_start_timeout_s": DEFAULT_H264_START_TIMEOUT_S,
            "tracker_pose_topic": TopicMap().tracker_pose,
            "tracker_status_topic": TopicMap().tracker_status,
            "gripper_state_topic": TopicMap().gripper_state,
            "episode_event_topic": TopicMap().episode_event,
            "tf_static_topic": TopicMap().tf_static,
            # 本地相机的时间诊断话题（sensor_msgs/TimeReference），为空不录制。
            "image_timing_topic": "",
            "processing_config": "",
            "extrinsic_path": "",
            "snapshot_files": [""],
            "image_timeout_s": 1.0,
            "tracker_timeout_s": 0.25,
            "gripper_timeout_s": 0.25,
            "match_window_s": 0.030,
            # 开始前与录制中的绝对频率检查；期望频率为 0 时不检查该路。
            "rate_window_s": 2.0,
            "expected_image_rate_hz": 30.0,
            "expected_gripper_rate_hz": 30.0,
            "min_rate_ratio": 0.95,
            "low_rate_persist_s": 2.0,
            # 标称源帧率及来源写入采集快照；0.0 表示未知。
            "nominal_source_fps": 0.0,
            "nominal_source_fps_source": "unknown",
            "image_timestamp_source": "unrecorded",
            "queue_max_messages": 600,
            "queue_max_mib": 768.0,
            "finish_timeout_s": 60.0,
            "min_free_disk_gib": 2.0,
            "status_rate_hz": DEFAULT_STATUS_RATE_HZ,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _text(self, name: str) -> str:
        """读取字符串参数并去除首尾空白。"""
        return str(self.get_parameter(name).value).strip()

    @property
    def controller(self) -> CollectionController:
        """生命周期控制器，供输入节点投递数据。"""
        return self._controller

    @property
    def topics(self) -> TopicMap:
        """当前话题映射。"""
        return self._topics

    def _create_interfaces(self) -> None:
        """创建服务和状态发布器；传感器订阅位于独立的输入节点。"""
        service_group = ReentrantCallbackGroup()
        for service_type, name, handler in (
            (CollectionStart, "start", self._start),
            (CollectionStop, "stop", self._stop),
            (CollectionSave, "save", self._save),
            (CollectionCancel, "cancel", self._cancel),
            (CollectionStopAndSave, "stop_and_save", self._stop_and_save),
            (CollectionStopAndCancel, "stop_and_cancel", self._stop_and_cancel),
            (CollectionDelete, "delete", self._delete),
            (CollectionList, "list", self._list),
            (CollectionGetStatus, "get_status", self._get_status),
        ):
            self.create_service(
                service_type,
                f"{NAMESPACE}/{name}",
                handler,
                callback_group=service_group,
            )
        self._status_publisher = self.create_publisher(
            CollectionStatus,
            f"{NAMESPACE}/status",
            # 周期状态使用 BEST_EFFORT：同进程可靠订阅会使 publish 阻塞数百毫秒
            # 并占用 GIL，拖慢全部传感器回调。状态每 0.5 s 重发，权威结果以
            # 服务响应和 get_status 为准，丢失单条状态不影响正确性。
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )

    # ------------------------------------------------------------------
    # 发现与订阅回调。
    # ------------------------------------------------------------------
    def _discover(self) -> dict:
        """返回三路输入当前是否存在发布者；Tracker 需位姿和状态都存在。"""
        topics = self._topics
        return {
            IMAGE: self.count_publishers(topics.image) > 0,
            TRACKER: (
                self.count_publishers(topics.tracker_pose) > 0
                and self.count_publishers(topics.tracker_status) > 0
            ),
            GRIPPER: self.count_publishers(topics.gripper_state) > 0,
        }

    def _check_image_publishers(self) -> Optional[Tuple[str, str]]:
        """核对图像发布端的消息类型和可靠性是否与配置兼容。

        Returns:
            不兼容时返回 ``(错误码, 说明)``；没有发布者或兼容返回 ``None``，
            前者由发现检查另行报告。
        """
        topic = self._topics.image
        mode = self._topics.image_mode
        infos = self.get_publishers_info_by_topic(topic)
        if not infos:
            return None
        expected = message_type_for_mode(mode)
        types = sorted({info.topic_type for info in infos})
        if expected not in types:
            return (
                "IMAGE_TYPE_MISMATCH",
                f"图像话题 {topic} 的发布类型为 {', '.join(types)}，"
                f"与 image_mode={mode} 要求的 {expected} 不一致",
            )
        if self._image_reliability == RELIABILITY_RELIABLE and not any(
            info.topic_type == expected
            and info.qos_profile.reliability == ReliabilityPolicy.RELIABLE
            for info in infos
        ):
            return (
                "IMAGE_QOS_INCOMPATIBLE",
                f"图像话题 {topic} 的发布端不是 RELIABLE，"
                "无法匹配 image_reliability=reliable 的订阅；"
                "请改用 best_effort 或让发布端使用 reliable",
            )
        return None

    @property
    def image_qos(self) -> QoSProfile:
        """图像订阅使用的 QoS。"""
        return image_qos_profile(self._image_reliability, self._image_qos_depth)

    # ------------------------------------------------------------------
    # 状态发布。
    # ------------------------------------------------------------------
    def _publish_status(self) -> None:
        """立即发布最新权威状态；节点销毁后忽略。"""
        try:
            self._status_publisher.publish(
                status_to_msg(
                    self._controller.status_snapshot(),
                    self.get_clock().now().to_msg(),
                )
            )
        except Exception as error:  # 关闭阶段发布器可能已失效。
            if rclpy.ok():
                self.get_logger().debug(f"发布采集状态失败: {error}")

    def _on_timer(self) -> None:
        """周期刷新健康、累计质量统计、发布状态并记录报警跃迁。"""
        snapshot = self._controller.tick()
        self._status_publisher.publish(
            status_to_msg(snapshot, self.get_clock().now().to_msg())
        )
        alarms = frozenset(snapshot.alarms)
        if snapshot.state == "recording":
            for code in sorted(alarms - self._known_alarms):
                self.get_logger().warning(f"采集报警: {code}")
        self._known_alarms = alarms

    # ------------------------------------------------------------------
    # 服务处理。
    # ------------------------------------------------------------------
    def _log_result(self, operation: str, result: OperationResult) -> None:
        """记录每次修改请求的受理与完成情况，便于现场排查。"""
        text = (
            f"{operation} request_id={result.request_id} accepted={result.accepted} "
            f"completed={result.completed} code={result.code} state={result.state}"
        )
        if result.accepted and result.completed:
            self.get_logger().info(text)
        else:
            self.get_logger().warning(f"{text} message={result.message}")

    def _start(self, request: Any, response: Any) -> Any:
        """开始采集服务。"""
        result = self._controller.start(
            request.request_id, request.task_name, request.name
        )
        self._log_result("start", result)
        response.result = result_to_msg(result)
        return response

    def _modify(self, operation: str, request: Any, response: Any) -> Any:
        """调用携带 UUID 的修改操作并填充统一结果。"""
        method = getattr(self._controller, operation)
        result = method(request.request_id, request.collection_uuid)
        self._log_result(operation, result)
        response.result = result_to_msg(result)
        return response

    def _stop(self, request: Any, response: Any) -> Any:
        """停止服务。"""
        return self._modify("stop", request, response)

    def _save(self, request: Any, response: Any) -> Any:
        """保存服务。"""
        return self._modify("save", request, response)

    def _cancel(self, request: Any, response: Any) -> Any:
        """取消服务。"""
        return self._modify("cancel", request, response)

    def _stop_and_save(self, request: Any, response: Any) -> Any:
        """停止并保存服务。"""
        return self._modify("stop_and_save", request, response)

    def _stop_and_cancel(self, request: Any, response: Any) -> Any:
        """停止并取消服务。"""
        return self._modify("stop_and_cancel", request, response)

    def _delete(self, request: Any, response: Any) -> Any:
        """删除已保存记录服务。"""
        return self._modify("delete", request, response)

    def _list(self, request: Any, response: Any) -> Any:
        """分页列出已保存记录。"""
        result: ListResult = self._controller.list_records(
            request.task_name, int(request.offset), int(request.limit)
        )
        response.success = result.success
        response.code = result.code
        response.message = result.message
        response.records = [record_to_msg(record) for record in result.records]
        response.total = result.total
        return response

    def _get_status(self, request: Any, response: Any) -> Any:
        """查询当前权威状态。"""
        del request
        response.status = status_to_msg(
            self._controller.status_snapshot(), self.get_clock().now().to_msg()
        )
        return response

    def shutdown(self) -> None:
        """有序退出：放弃未保存采集并释放根目录锁。"""
        removed = self._controller.shutdown()
        if removed:
            self.get_logger().warning(f"退出时已清理未保存采集: {removed}")


class CollectionInputs(Node):
    """订阅三路输入、Tracker 状态和静态 TF，并把原始字节交给控制器。

    传感器订阅独占一个节点，由单线程执行器处理：数百条/秒的小消息在
    多线程 Python 执行器中会因 GIL 竞争产生秒级积压；服务与周期状态则
    运行在另一个执行器，长耗时的停止请求不会阻塞传感器回调。
    """

    def __init__(
        self,
        controller: CollectionController,
        topics: TopicMap,
        image_qos: Optional[QoSProfile] = None,
    ) -> None:
        """创建全部订阅。

        Args:
            image_qos: 图像订阅 QoS；默认 BEST_EFFORT、深度 30。
        """
        super().__init__("collection_inputs")
        self._controller = controller
        self._topics = topics
        sensor_qos = image_qos or image_qos_profile(
            DEFAULT_IMAGE_RELIABILITY, DEFAULT_IMAGE_QOS_DEPTH
        )
        image_type = get_message(message_type_for_mode(topics.image_mode))

        def reliable_qos(depth: int) -> QoSProfile:
            """Tracker 与夹爪发布器为 RELIABLE，订阅端使用匹配的深度。"""
            return QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=depth,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        for message_type, topic, callback, qos in (
            (image_type, topics.image, self._on_image, sensor_qos),
            (PoseStamped, topics.tracker_pose, self._on_pose, reliable_qos(200)),
            (TrackerStatus, topics.tracker_status, self._on_tracker_status, reliable_qos(200)),
            (GripperState, topics.gripper_state, self._on_gripper, reliable_qos(100)),
            (TFMessage, topics.tf_static, self._on_tf_static, latched_qos),
        ):
            self.create_subscription(message_type, topic, callback, qos, raw=True)
        if topics.image_timing:
            self.create_subscription(
                TimeReference, topics.image_timing, self._on_image_timing,
                reliable_qos(60), raw=True,
            )

    def _on_image(self, data: bytes) -> None:
        """图像（原始、压缩或 H.264 包）原始字节：解析头部后更新健康并按需写入。"""
        self._controller.on_image(bytes(data))

    def _on_pose(self, data: bytes) -> None:
        """Tracker 位姿原始字节：反序列化小消息以检查有效性。"""
        raw = bytes(data)
        message = deserialize_message(raw, PoseStamped)
        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        self._controller.on_tracker_pose(raw, stamp_ns, _pose_is_finite(message))

    def _on_tracker_status(self, data: bytes) -> None:
        """Tracker 状态原始字节。"""
        raw = bytes(data)
        message = deserialize_message(raw, TrackerStatus)
        self._controller.on_tracker_status(
            raw,
            bool(message.device_connected),
            bool(message.pose_valid),
            int(message.tracking_state),
        )

    def _on_gripper(self, data: bytes) -> None:
        """夹爪状态原始字节。"""
        raw = bytes(data)
        message = deserialize_message(raw, GripperState)
        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        self._controller.on_gripper_state(
            raw, stamp_ns, bool(message.valid), float(message.filtered_openness)
        )

    def _on_image_timing(self, data: bytes) -> None:
        """相机时间诊断原始字节：解析映射代号并录制。"""
        raw = bytes(data)
        message = deserialize_message(raw, TimeReference)
        self._controller.on_image_timing(raw, timing_generation(message.source))

    def _on_tf_static(self, data: bytes) -> None:
        """静态 TF 原始字节：缓存并在采集中写入。"""
        self._controller.on_tf_static(bytes(data))


class CollectionRuntime:
    """持有采集节点、输入节点和两个执行器，统一启动与有序退出。"""

    def __init__(self, parameter_overrides: Optional[Sequence[Any]] = None) -> None:
        """创建节点与执行器；订阅线程在 :meth:`start` 中启动。"""
        self.node = CollectionNode(parameter_overrides)
        self.inputs = CollectionInputs(
            self.node.controller, self.node.topics, self.node.image_qos
        )
        self.service_executor = MultiThreadedExecutor(num_threads=SERVICE_THREADS)
        self.input_executor = SingleThreadedExecutor()
        self.service_executor.add_node(self.node)
        self.input_executor.add_node(self.inputs)
        self._input_thread = threading.Thread(
            target=self._spin_inputs, name="collection-inputs", daemon=True
        )

    def _spin_inputs(self) -> None:
        """运行输入执行器；外部关闭（SIGINT）时静默退出。"""
        try:
            self.input_executor.spin()
        except ExternalShutdownException:
            pass

    def start(self) -> None:
        """启动传感器订阅线程；服务执行器由调用方在合适线程中 spin。"""
        self._input_thread.start()

    def shutdown(self) -> None:
        """停止执行器、放弃未保存采集并销毁节点。"""
        self.input_executor.shutdown()
        self.service_executor.shutdown()
        if self._input_thread.is_alive():
            self._input_thread.join(timeout=5.0)
        self.node.shutdown()
        self.inputs.destroy_node()
        self.node.destroy_node()


def main(args: Optional[Sequence[str]] = None) -> None:
    """运行采集节点，Ctrl+C 时有序清理。"""
    rclpy.init(args=list(args) if args is not None else None)
    runtime: Optional[CollectionRuntime] = None
    try:
        runtime = CollectionRuntime()
        runtime.start()
        runtime.service_executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if runtime is not None:
            runtime.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
