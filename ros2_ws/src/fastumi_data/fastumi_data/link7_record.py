"""单遍读取一条 MCAP 记录，按 episode 同步并写出 ``fastumi_ros2_link7_v1`` HDF5。

* H.264 按 bag 顺序全部解码，参考帧跨 episode 保留；raw/JPEG 在活动与后滚动期间
  解码，其余消息只计数。每个解码帧获得递增的 ``source_image_index``。
* episode 内、时间位于 START/STOP 事件之间的解码帧逐帧写入磁盘暂存，内存有界。
* Tracker 位姿、状态和夹爪在 START 前保留短暂预滚动；STOP 后在 ``POST_ROLL_NS``
  内继续收集支撑样本和 Header 时间仍在该 episode 内的迟到图像，再处理。下一条
  START 不提前结束上一条的后滚动。
* Tracker 状态序列号与外参绑定不一致属于硬错误；图像解码失败同样中止。
* 夹爪开度字段与取值方式由 :class:`~fastumi_data.gripper_source.GripperSource` 决定；
  二值化记录的夹爪消息总数须与 ``binarization.yaml`` 一致，否则中止。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from fastumi_interfaces.msg import EpisodeEvent
import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message

from fastumi_data.bag_images import BagImageDecoder, check_bag_image_topic
from fastumi_data.frame_spool import FrameSpool
from fastumi_data.gripper_source import DEFAULT_GRIPPER_SOURCE, GripperSource
from fastumi_data.image_modes import MODE_H264, MODE_RAW, mode_for_message_type
from fastumi_data.json_report import write_json_report
from fastumi_data.link7_calibration import CalibrationError, Link7Calibration
from fastumi_data.link7_config import ResolvedProcessing
from fastumi_data.link7_hdf5 import write_link7_episode_hdf5
from fastumi_data.link7_sync import synchronize_link7_episode
from fastumi_data.models import GripperSample, PoseSample, TrackerStatusSample
from fastumi_data.video_decoding import COLOR_RGB, VideoDecodeError


# START 之前保留的位姿、状态、夹爪样本时长。
PRE_ROLL_NS = 500_000_000
# STOP 之后继续收集位姿、状态、夹爪的时长。
POST_ROLL_NS = 500_000_000
TRACKER_STATUS_TYPE = "fastumi_interfaces/msg/TrackerStatus"


def stamp_to_ns(stamp) -> int:
    """把 builtin_interfaces/Time 转换为 Unix 纳秒。"""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def open_reader(bag_uri: Path, topics: Optional[List[str]] = None) -> rosbag2_py.SequentialReader:
    """以 MCAP 存储打开 rosbag2，可选只读取指定话题。"""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_uri), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    if topics:
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    return reader


def precheck_tracker_identity(
    bag_uri: Path, status_topic: str, expected_serial: str
) -> Dict[str, Any]:
    """只读 Tracker 状态话题，核对序列号非空、唯一且等于标定绑定。

    Raises:
        CalibrationError: 缺少状态话题或消息、空序列号、多个序列号或不匹配。
    """
    reader = open_reader(bag_uri)
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if status_topic not in topic_types:
        raise CalibrationError(f"{bag_uri} 缺少 Tracker 状态话题 {status_topic}")
    if topic_types[status_topic] != TRACKER_STATUS_TYPE:
        raise CalibrationError(
            f"{bag_uri} 的 {status_topic} 类型为 {topic_types[status_topic]}，"
            f"不是 {TRACKER_STATUS_TYPE}"
        )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[status_topic]))
    message_type = get_message(TRACKER_STATUS_TYPE)
    count = 0
    empty = 0
    serials: Dict[str, int] = {}
    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic != status_topic:
            continue
        status = deserialize_message(serialized, message_type)
        count += 1
        serial = str(status.serial_number).strip()
        if not serial:
            empty += 1
        else:
            serials[serial] = serials.get(serial, 0) + 1
    if count == 0:
        raise CalibrationError(f"{bag_uri} 的 {status_topic} 没有任何状态消息")
    if empty:
        raise CalibrationError(f"{bag_uri} 的 {status_topic} 有 {empty} 条空序列号")
    if len(serials) != 1:
        raise CalibrationError(
            f"{bag_uri} 的 {status_topic} 含多个 Tracker 序列号 {sorted(serials)}；"
            "不同 Tracker 应分别使用配套外参转换"
        )
    serial = next(iter(serials))
    if serial != expected_serial:
        raise CalibrationError(
            f"{bag_uri} 的 Tracker 序列号 {serial} 与标定绑定 {expected_serial} 不一致"
        )
    return {"status_messages": count, "observed_tracker_serial": serial}


@dataclass
class _ActiveEpisode:
    """一条正在积累的 episode。"""

    start: Any
    spool: FrameSpool
    poses: List[PoseSample] = field(default_factory=list)
    grippers: List[GripperSample] = field(default_factory=list)
    statuses: List[TrackerStatusSample] = field(default_factory=list)
    stop_ns: Optional[int] = None
    stop_event: Any = None
    decoded_in_window: int = 0


@dataclass
class RecordSummary:
    """一条来源记录的转换结果。"""

    accepted: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class Link7RecordConverter:
    """把一条记录转换为逐 episode 的 Link7 HDF5 与 JSON 报告。"""

    def __init__(
        self,
        record_id: str,
        bag_uri: Path,
        output_dir: Path,
        work_dir: Path,
        processing: ResolvedProcessing,
        calibration: Link7Calibration,
        nominal_source_fps: float,
        base_metadata: Dict[str, Any],
        log: Callable[[str], None] = print,
        gripper_source: GripperSource = DEFAULT_GRIPPER_SOURCE,
        reject_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        """保存转换参数；``output_dir`` 下写 ``episodes/`` 与 ``reports/``。

        ``log`` 接收过程行；``reject_log`` 单独接收 episode 被拒绝的行，缺省沿用 ``log``。
        """
        self._record_id = record_id
        self._bag_uri = Path(bag_uri)
        self._output_dir = Path(output_dir)
        self._work_dir = Path(work_dir)
        self._processing = processing
        self._topics = processing.topics
        self._calibration = calibration
        self._fps = float(nominal_source_fps)
        self._base_metadata = dict(base_metadata)
        self._log = log
        self._reject_log = reject_log or log
        self._gripper_source = gripper_source
        self._gripper_count = 0
        self._offset_ns = int(round(calibration.tracker_time_offset_ms * 1.0e6))
        self._active: Optional[_ActiveEpisode] = None
        # 等待后滚动结束的 episode，允许与下一条活动 episode 并存。
        self._pending: Deque[_ActiveEpisode] = deque()
        self._pre_poses: Deque[PoseSample] = deque()
        self._pre_grippers: Deque[GripperSample] = deque()
        self._pre_statuses: Deque[TrackerStatusSample] = deque()
        self._summary = RecordSummary()
        self._image_index = 0
        self._image_mode = ""
        self._names: set = set()

    # ------------------------------------------------------------------ 输出
    def _episode_name(self, index: int) -> str:
        """返回来源记录与原 episode 编号组合的唯一名称。"""
        name = f"{self._record_id}/episode_{index:04d}"
        if name in self._names:
            raise ValueError(f"记录 {self._record_id} 内 episode 编号 {index} 重复")
        self._names.add(name)
        return name

    def _reject(self, start, reasons: List[Dict], warnings: List[Dict], stats: Dict) -> None:
        """写入拒绝报告并记入摘要；质量拒绝不中止批次。"""
        index = int(start.episode_index)
        name = f"{self._record_id}/episode_{index:04d}"
        report = {
            "accepted": False,
            "name": name,
            "record_id": self._record_id,
            "episode_index": index,
            "task_name": start.task_name,
            "session_id": start.session_id,
            "rejections": reasons,
            "warnings": warnings,
            "quality": stats,
        }
        report.update(self._base_metadata)
        path = self._output_dir / "reports" / f"episode_{index:04d}.json"
        if path.exists():
            raise FileExistsError(f"{path} 已存在：记录内 episode 编号重复")
        write_json_report(str(path), report)
        self._summary.rejected.append(
            {"name": name, "report": str(path), "rejections": [r["code"] for r in reasons]}
        )
        self._reject_log(f"  拒绝 {name}: " + "；".join(item["message"] for item in reasons))

    def _finish(self, episode: _ActiveEpisode) -> None:
        """同步、检查并写出一条 episode，随后删除帧暂存。"""
        try:
            spool = episode.spool
            start = episode.start
            if episode.stop_event is not None and (
                episode.stop_event.episode_index != start.episode_index
                or episode.stop_event.session_id != start.session_id
            ):
                self._reject(
                    start,
                    [{"code": "EVENT_MISMATCH", "message": "START 与 STOP 的 session 或 episode 编号不一致"}],
                    [],
                    {},
                )
                return
            timestamps = spool.timestamps_ns
            if episode.stop_ns is not None:
                inside = timestamps <= episode.stop_ns
                # STOP 之后的帧通常位于末尾；只在其构成后缀时截掉，乱序交给质量检查。
                count = int(np.count_nonzero(inside))
                if inside[:count].all():
                    timestamps = timestamps[:count]
            result = synchronize_link7_episode(
                timestamps,
                episode.poses,
                episode.grippers,
                episode.statuses,
                self._calibration.tracker_to_link7,
                self._fps,
                self._processing.sync,
                tracker_time_offset_ns=self._offset_ns,
                gripper_field=self._gripper_source.field,
                gripper_interpolation=self._gripper_source.interpolation,
            )
            stats = dict(result.stats)
            stats["decoded_frames_in_episode"] = int(len(timestamps))
            if result.episode is None:
                self._reject(start, result.rejections, result.warnings, stats)
                return
            index = int(start.episode_index)
            name = self._episode_name(index)
            metadata = dict(self._base_metadata)
            metadata.update(
                {
                    "episode_name": name,
                    "record_id": self._record_id,
                    "task_name": start.task_name,
                    "session_id": start.session_id,
                    "episode_index": index,
                    "measured_source_fps": stats.get("effective_source_rate_hz"),
                    "fitted_frame_rate_hz": stats.get("fitted_frame_rate_hz"),
                    "frame_drop_count": stats.get("output_dropped_frame_total", 0),
                    "frame_drop_warnings": [
                        item for item in result.warnings if item["code"] == "FRAME_DROP"
                    ],
                    "quality": stats,
                    "image_mode": self._image_mode,
                }
            )
            path = self._output_dir / "episodes" / f"episode_{index:04d}.hdf5"
            write_link7_episode_hdf5(path, result.episode, spool, metadata)
            report = {"accepted": True, "name": name, "hdf5": str(path), "warnings": result.warnings}
            report.update(metadata)
            report_path = self._output_dir / "reports" / f"episode_{index:04d}.json"
            write_json_report(str(report_path), report)
            self._summary.accepted.append(
                {
                    "name": name,
                    "hdf5": str(path),
                    "report": str(report_path),
                    "frames": len(result.episode),
                    "metadata": metadata,
                }
            )
            drops = stats.get("output_dropped_frame_total", 0)
            self._log(
                f"  写出 {name}: {len(result.episode)} 帧，路径 {result.episode.selection.path}"
                + (f"，丢帧提示 {drops} 帧" if drops else "")
            )
        finally:
            episode.spool.close()

    # ------------------------------------------------------------------ 输入
    def _new_episode(self, event) -> _ActiveEpisode:
        """创建 episode 并带入预滚动样本。"""
        spool = FrameSpool(
            self._work_dir / f"spool_{event.episode_index:04d}_{id(event):x}.h5"
        )
        episode = _ActiveEpisode(start=event, spool=spool)
        episode.poses.extend(self._pre_poses)
        episode.grippers.extend(self._pre_grippers)
        episode.statuses.extend(self._pre_statuses)
        return episode

    def _flush_pending(self, bag_time_ns: Optional[int]) -> None:
        """后滚动结束（或强制）时处理等待中的 episode。"""
        while self._pending:
            if (
                bag_time_ns is not None
                and bag_time_ns <= self._pending[0].stop_ns + POST_ROLL_NS
            ):
                break
            pending = self._pending.popleft()
            self._finish(pending)

    def _handle_event(self, message) -> None:
        """更新 episode 状态机。"""
        event_type = int(message.event_type)
        if event_type == EpisodeEvent.START:
            if self._active is not None:
                self._reject(
                    self._active.start,
                    [{"code": "EVENT_SEQUENCE", "message": "在活动 episode 内再次收到 START"}],
                    [],
                    {},
                )
                self._active.spool.close()
            self._active = self._new_episode(message)
        elif event_type == EpisodeEvent.STOP:
            if self._active is None:
                self._reject(
                    message,
                    [{"code": "EVENT_SEQUENCE", "message": "收到 STOP，但此前没有 START"}],
                    [],
                    {},
                )
                return
            self._active.stop_ns = stamp_to_ns(message.header.stamp)
            self._active.stop_event = message
            self._pending.append(self._active)
            self._active = None
        elif event_type == EpisodeEvent.ABORT:
            if self._active is None:
                self._reject(
                    message,
                    [{"code": "EVENT_SEQUENCE", "message": "收到 ABORT，但此前没有 START"}],
                    [],
                    {},
                )
            else:
                self._reject(
                    self._active.start,
                    [{"code": "ABORTED", "message": "操作员主动放弃 episode"}],
                    [],
                    {},
                )
                self._active.spool.close()
                self._active = None

    def _append_sample(self, kind: str, sample) -> None:
        """把位姿/夹爪/状态样本加入活动或后滚动 episode，并维护预滚动窗口。"""
        targets = []
        if self._active is not None:
            targets.append(self._active)
        targets.extend(
            episode for episode in self._pending
            if sample.timestamp_ns <= episode.stop_ns + POST_ROLL_NS
        )
        for episode in targets:
            getattr(episode, kind).append(sample)
        buffer = {
            "poses": self._pre_poses,
            "grippers": self._pre_grippers,
            "statuses": self._pre_statuses,
        }[kind]
        buffer.append(sample)
        while buffer and buffer[0].timestamp_ns < sample.timestamp_ns - PRE_ROLL_NS:
            buffer.popleft()

    def _gripper_sample(self, message) -> GripperSample:
        """把第 N 条夹爪消息转为样本；``raw`` 二值化记录按 ``original_valid`` 恢复原有效性。"""
        ordinal = self._gripper_count
        self._gripper_count += 1
        valid = bool(message.valid)
        original_valid = self._gripper_source.original_valid
        if original_valid is not None:
            if ordinal >= len(original_valid):
                raise ValueError(
                    f"{self._bag_uri} 的夹爪消息多于 binarization.yaml 的 "
                    f"frame_count {len(original_valid)}"
                )
            valid = original_valid[ordinal]
        return GripperSample(
            stamp_to_ns(message.header.stamp),
            float(message.raw_openness),
            float(message.filtered_openness),
            int(message.detected_marker_count),
            valid,
        )

    def _accept_frames(self, frames) -> None:
        """为解码帧分配源序号，并把 START/STOP 时间窗内的帧写入暂存。"""
        for frame in frames:
            index = self._image_index
            self._image_index += 1
            # 后滚动 episode 与当前活动 episode 共用源序号，按 Header 时间筛选归属。
            targets = list(self._pending)
            if self._active is not None:
                targets.append(self._active)
            for episode in targets:
                start_ns = stamp_to_ns(episode.start.header.stamp)
                if frame.stamp_ns < start_ns:
                    continue
                if episode.stop_ns is not None and frame.stamp_ns > episode.stop_ns:
                    continue
                episode.spool.append(frame.stamp_ns, index, frame.image)

    def _handle_image(self, decoder: BagImageDecoder, serialized: bytes, bag_ns: int) -> None:
        """解码活动与后滚动期间的图像；H.264 始终解码以维持参考帧。"""
        if self._image_mode != MODE_H264 and self._active is None and not self._pending:
            self._image_index += 1
            return
        skipped = decoder.stats.skipped_before_keyframe
        try:
            frames = decoder.feed(serialized, bag_ns)
        except VideoDecodeError as error:
            raise VideoDecodeError(
                f"{self._bag_uri} 图像解码失败（bag 时间 {bag_ns} ns）: {error}"
            ) from error
        if self._active is not None and decoder.stats.skipped_before_keyframe != skipped:
            raise VideoDecodeError(
                f"{self._bag_uri} episode {self._active.start.episode_index} 内的图像包"
                "位于首个 H.264 关键帧之前，无法解码"
            )
        self._accept_frames(frames)

    def convert(self) -> RecordSummary:
        """读取整条记录并返回接受/拒绝摘要。"""
        reader = open_reader(self._bag_uri)
        topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
        image_topic = self._topics["image"]
        if image_topic not in topic_types:
            raise ValueError(f"{self._bag_uri} 缺少图像话题 {image_topic}")
        detected = mode_for_message_type(topic_types[image_topic])
        if detected is None:
            raise ValueError(f"{image_topic} 的类型 {topic_types[image_topic]} 不是支持的图像消息")
        self._image_mode = self._processing.image_mode or detected
        check_bag_image_topic(topic_types, image_topic, self._image_mode)
        for key in ("tracker_pose", "tracker_status", "gripper_state", "episode_event"):
            if self._topics[key] not in topic_types:
                raise ValueError(f"{self._bag_uri} 缺少话题 {self._topics[key]}")
        wanted = {self._topics[key] for key in (
            "image", "tracker_pose", "tracker_status", "gripper_state", "episode_event"
        )}
        reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(wanted)))
        message_types = {topic: get_message(topic_types[topic]) for topic in wanted}
        decoder = BagImageDecoder(self._image_mode, COLOR_RGB)
        expected_serial = self._calibration.tracker_serial
        while reader.has_next():
            topic, serialized, bag_ns = reader.read_next()
            self._flush_pending(bag_ns)
            if topic == image_topic:
                self._handle_image(decoder, serialized, bag_ns)
                continue
            message = deserialize_message(serialized, message_types[topic])
            if topic == self._topics["episode_event"]:
                self._handle_event(message)
            elif topic == self._topics["tracker_status"]:
                serial = str(message.serial_number).strip()
                if serial != expected_serial:
                    raise CalibrationError(
                        f"{self._bag_uri} 的 Tracker 序列号 {serial!r} 与绑定 {expected_serial} 不一致"
                    )
                self._append_sample(
                    "statuses",
                    TrackerStatusSample(
                        stamp_to_ns(message.header.stamp),
                        bool(message.device_connected),
                        bool(message.pose_valid),
                        int(message.tracking_state),
                    ),
                )
            elif topic == self._topics["tracker_pose"]:
                pose = message.pose
                self._append_sample(
                    "poses",
                    PoseSample(
                        stamp_to_ns(message.header.stamp),
                        np.asarray([pose.position.x, pose.position.y, pose.position.z]),
                        np.asarray(
                            [pose.orientation.x, pose.orientation.y,
                             pose.orientation.z, pose.orientation.w]
                        ),
                    ),
                )
            elif topic == self._topics["gripper_state"]:
                self._append_sample("grippers", self._gripper_sample(message))
        try:
            self._accept_frames(decoder.drain())
        except VideoDecodeError as error:
            raise VideoDecodeError(f"{self._bag_uri} 排空解码器失败: {error}") from error
        expected = self._gripper_source.frame_count
        if expected is not None and self._gripper_count != expected:
            raise ValueError(
                f"{self._bag_uri} 的夹爪消息数 {self._gripper_count} 与 "
                f"binarization.yaml 的 frame_count {expected} 不一致"
            )
        self._flush_pending(None)
        if self._active is not None:
            self._reject(
                self._active.start,
                [{"code": "EVENT_SEQUENCE", "message": "MCAP 结束时 episode 尚未 STOP"}],
                [],
                {},
            )
            self._active.spool.close()
            self._active = None
        stats = decoder.stats
        self._summary.stats = {
            "image_mode": self._image_mode,
            "decoded_images": self._image_index,
            "decoder_packets": stats.packets,
            "decoder_frames": stats.frames,
            "skipped_before_keyframe": stats.skipped_before_keyframe,
        }
        return self._summary
