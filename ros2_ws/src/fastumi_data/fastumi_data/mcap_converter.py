"""``convert_mcap``：把 MCAP 记录批量转换为 Link7 HDF5，并合并导出 30 Hz Zarr。

命令行入口执行 MCAP→``fastumi_ros2_link7_v1`` HDF5→``fastumi-rm75-link7-v1`` Zarr：
负责来源发现、配置与标称源帧率解析、外参设备身份预检、逐记录转换编排、
清单和发布。单条记录的读取与同步见 :mod:`fastumi_data.link7_record`。

``McapEpisodeConverter`` 保留旧 ``fastumi_ros2_v2``（TCP、按处理配置采样率）库接口，
供连续录制等旧路径和测试使用，其行为不变；命令行不再提供旧 ``--extrinsic`` 用法。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Dict, Optional

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from fastumi_interfaces.msg import EpisodeEvent, TrackerStatus
from geometry_msgs.msg import PoseStamped
import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import yaml

from fastumi_data.bag_images import BagImageDecoder, check_bag_image_topic
from fastumi_data.extrinsic import load_tracker_tcp_extrinsic
from fastumi_data.hdf5_writer import (
    build_quality_report,
    write_episode_hdf5,
    write_json_report,
)
from fastumi_data.models import (
    EpisodeBuffer,
    EpisodeEventRecord,
    GripperSample,
    ImageSample,
    PoseSample,
    ProcessingConfig,
    TrackerStatusSample,
)
from fastumi_data.frame_selection import selection_path_for
from fastumi_data.image_modes import MODE_RAW, validate_image_mode
from fastumi_data.link7_calibration import (
    CalibrationError,
    TIME_OFFSET_CONVENTIONS,
    check_timestamp_source,
    load_link7_calibration,
    normalize_timestamp_source,
)
from fastumi_data.link7_config import ConfigError, resolve_processing, resolve_source_fps
from fastumi_data.link7_hdf5 import LINK7_SCHEMA_VERSION
from fastumi_data.link7_record import Link7RecordConverter, precheck_tracker_identity
from fastumi_data.link7_zarr import ZARR_FORMAT, ZarrEpisodeSource, export_link7_zarr
from fastumi_data.synchronizer import synchronize_episode
from fastumi_data.video_decoding import COLOR_RGB, VideoDecodeError


def _stamp_to_ns(stamp) -> int:
    """把 builtin_interfaces/Time 转换为 Unix 纳秒整数。"""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _load_processing_document(path: str) -> tuple[ProcessingConfig, Dict]:
    """加载同步参数、话题映射和图像模式（缺省为 raw）。"""
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取处理配置 {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("处理配置必须是 YAML 映射")
    topics = document.get("topics")
    if not isinstance(topics, dict):
        raise ValueError("处理配置缺少 topics 映射")
    required_topics = ("image", "tracker_pose", "gripper_state", "episode_event")
    missing = [name for name in required_topics if not topics.get(name)]
    if missing:
        raise ValueError(f"处理配置缺少话题: {', '.join(missing)}")
    config = ProcessingConfig(
        sample_rate_hz=float(document.get("sample_rate_hz", 20.0)),
        max_image_delta_s=float(document.get("max_image_delta_s", 0.03)),
        max_pose_gap_s=float(document.get("max_pose_gap_s", 0.1)),
        max_gripper_gap_s=float(document.get("max_gripper_gap_s", 0.2)),
        minimum_samples=int(document.get("minimum_samples", 10)),
        require_tracker_status=bool(
            document.get("require_tracker_status", True)
        ),
        image_mode=validate_image_mode(document.get("image_mode", MODE_RAW)),
    )
    if config.sample_rate_hz <= 0.0 or config.minimum_samples <= 0:
        raise ValueError("采样率和最小样本数必须为正数")
    if config.require_tracker_status and not topics.get("tracker_status"):
        raise ValueError(
            "require_tracker_status=true 时必须配置 tracker_status 话题"
        )
    return config, topics


class McapEpisodeConverter:
    """单遍读取 MCAP，并在 STOP 事件到达时处理当前 episode。"""

    def __init__(
        self,
        bag_uri: str,
        output_dir: str,
        processing_config: ProcessingConfig,
        topics: Dict,
        extrinsic_path: str,
        force: bool,
    ) -> None:
        """保存转换配置并初始化消息解码器。"""
        self._bag_uri = str(Path(bag_uri).resolve())
        self._output_dir = Path(output_dir).resolve()
        self._config = processing_config
        self._topics = topics
        self._extrinsic = load_tracker_tcp_extrinsic(extrinsic_path)
        self._force = force
        self._active: Optional[EpisodeBuffer] = None
        self._converted_count = 0
        self._rejected_count = 0

    def _write_rejection(
        self,
        event: EpisodeEventRecord,
        reasons: list[str],
        warnings: Optional[list[str]] = None,
    ) -> None:
        """写入被拒绝或主动放弃 episode 的报告。"""
        report_path = (
            self._output_dir
            / "reports"
            / f"episode_{event.episode_index:04d}.json"
        )
        report = {
            "accepted": False,
            "task_name": event.task_name,
            "session_id": event.session_id,
            "episode_index": event.episode_index,
            "rejection_reasons": reasons,
            "warnings": warnings or [],
            "calibration_sha256": self._extrinsic.calibration_hash,
            "tracker_time_offset_ms": self._extrinsic.time_offset_ms,
            "source_calibration_sha256": (
                self._extrinsic.source_calibration_sha256
            ),
            "calibration_method": self._extrinsic.calibration_method,
            "aruco_config_sha256": self._extrinsic.aruco_config_sha256,
        }
        write_json_report(str(report_path), report)
        self._rejected_count += 1

    def _finish_episode(self, stop_event: EpisodeEventRecord) -> None:
        """同步、检查并写入当前 episode。"""
        if self._active is None:
            self._write_rejection(
                stop_event, ["收到 STOP，但此前没有 START"]
            )
            return
        start_event = self._active.start_event
        if (
            start_event.episode_index != stop_event.episode_index
            or start_event.session_id != stop_event.session_id
        ):
            self._write_rejection(
                start_event, ["START 与 STOP 的 session 或 episode 编号不一致"]
            )
            self._active = None
            return
        tracker_time_offset_ns = int(
            round(self._extrinsic.time_offset_ms * 1.0e6)
        )
        result = synchronize_episode(
            self._active,
            stop_event.timestamp_ns,
            self._extrinsic.matrix,
            self._config,
            tracker_time_offset_ns=tracker_time_offset_ns,
        )
        if result.episode is None:
            self._write_rejection(
                start_event, result.rejection_reasons, result.warnings
            )
            self._active = None
            return
        episode_path = (
            self._output_dir
            / "episodes"
            / f"episode_{start_event.episode_index:04d}.hdf5"
        )
        if episode_path.exists() and not self._force:
            raise FileExistsError(
                f"{episode_path} 已存在；使用 --force 明确覆盖"
            )
        write_episode_hdf5(
            str(episode_path),
            result.episode,
            task_name=start_event.task_name,
            session_id=start_event.session_id,
            episode_index=start_event.episode_index,
            sample_rate_hz=self._config.sample_rate_hz,
            calibration_hash=self._extrinsic.calibration_hash,
            tracker_time_offset_ms=self._extrinsic.time_offset_ms,
            source_calibration_sha256=(
                self._extrinsic.source_calibration_sha256
            ),
            calibration_method=self._extrinsic.calibration_method,
            aruco_config_sha256=self._extrinsic.aruco_config_sha256,
        )
        report = build_quality_report(
            result.episode,
            result.warnings,
            self._extrinsic.calibration_hash,
            tracker_time_offset_ms=self._extrinsic.time_offset_ms,
            source_calibration_sha256=(
                self._extrinsic.source_calibration_sha256
            ),
            calibration_method=self._extrinsic.calibration_method,
            aruco_config_sha256=self._extrinsic.aruco_config_sha256,
        )
        report.update(
            {
                "task_name": start_event.task_name,
                "session_id": start_event.session_id,
                "episode_index": start_event.episode_index,
            }
        )
        write_json_report(
            str(
                self._output_dir
                / "reports"
                / f"episode_{start_event.episode_index:04d}.json"
            ),
            report,
        )
        self._converted_count += 1
        self._active = None

    def _handle_event(self, message: EpisodeEvent) -> None:
        """更新 episode 状态机并处理边界冲突。"""
        event = EpisodeEventRecord(
            timestamp_ns=_stamp_to_ns(message.header.stamp),
            session_id=message.session_id,
            task_name=message.task_name,
            episode_index=int(message.episode_index),
            event_type=int(message.event_type),
        )
        if event.event_type == EpisodeEvent.START:
            if self._active is not None:
                self._write_rejection(
                    self._active.start_event, ["在活动 episode 内再次收到 START"]
                )
            self._active = EpisodeBuffer(start_event=event)
        elif event.event_type == EpisodeEvent.STOP:
            self._finish_episode(event)
        elif event.event_type == EpisodeEvent.ABORT:
            if self._active is None:
                self._write_rejection(event, ["收到 ABORT，但此前没有 START"])
            else:
                self._write_rejection(
                    self._active.start_event, ["操作员主动放弃 episode"]
                )
                self._active = None

    def convert(self) -> Dict[str, int]:
        """读取全部消息并返回接收、转换和拒绝计数。"""
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(
                uri=self._bag_uri, storage_id="mcap"
            ),
            rosbag2_py.ConverterOptions("", ""),
        )
        topic_types = {
            item.name: item.type for item in reader.get_all_topics_and_types()
        }
        message_types = {
            topic: get_message(type_name)
            for topic, type_name in topic_types.items()
        }
        image_topic = self._topics["image"]
        if image_topic in topic_types:
            check_bag_image_topic(
                topic_types, image_topic, self._config.image_mode
            )
        image_decoder = BagImageDecoder(self._config.image_mode, COLOR_RGB)
        message_count = 0
        while reader.has_next():
            topic, serialized, bag_timestamp_ns = reader.read_next()
            if topic not in message_types:
                continue
            if topic == image_topic:
                message_count += 1
                self._handle_image(image_decoder, serialized, bag_timestamp_ns)
                continue
            message = deserialize_message(serialized, message_types[topic])
            message_count += 1
            if topic == self._topics["episode_event"]:
                self._handle_event(message)
                continue
            if topic == self._topics.get("tracker_status"):
                status: TrackerStatus = message
                if (
                    status.serial_number
                    and status.serial_number
                    != self._extrinsic.tracker_serial
                ):
                    raise ValueError(
                        "MCAP Tracker 序列号与外参不一致: "
                        f"{status.serial_number} != "
                        f"{self._extrinsic.tracker_serial}"
                    )
                if self._active is not None:
                    self._active.tracker_statuses.append(
                        TrackerStatusSample(
                            _stamp_to_ns(status.header.stamp),
                            bool(status.device_connected),
                            bool(status.pose_valid),
                            int(status.tracking_state),
                        )
                    )
                continue
            if self._active is None:
                continue
            if topic == self._topics["tracker_pose"]:
                pose_message: PoseStamped = message
                pose = pose_message.pose
                self._active.poses.append(
                    PoseSample(
                        _stamp_to_ns(pose_message.header.stamp),
                        np.asarray(
                            [
                                pose.position.x,
                                pose.position.y,
                                pose.position.z,
                            ],
                            dtype=np.float64,
                        ),
                        np.asarray(
                            [
                                pose.orientation.x,
                                pose.orientation.y,
                                pose.orientation.z,
                                pose.orientation.w,
                            ],
                            dtype=np.float64,
                        ),
                    )
                )
            elif topic == self._topics["gripper_state"]:
                self._active.grippers.append(
                    GripperSample(
                        _stamp_to_ns(message.header.stamp),
                        float(message.raw_openness),
                        float(message.filtered_openness),
                        int(message.detected_marker_count),
                        bool(message.valid),
                    )
                )
        try:
            self._append_frames(image_decoder.drain())
        except VideoDecodeError as error:
            raise VideoDecodeError(
                f"图像话题 {image_topic} 在文件结束排空解码器时失败: {error}"
            ) from error
        if self._active is not None:
            self._write_rejection(
                self._active.start_event, ["MCAP 结束时 episode 尚未 STOP"]
            )
            self._active = None
        summary = {
            "messages": message_count,
            "converted": self._converted_count,
            "rejected": self._rejected_count,
        }
        if self._config.image_mode != MODE_RAW:
            stats = image_decoder.stats
            summary.update(
                {
                    "image_mode": self._config.image_mode,
                    "image_frames_decoded": stats.frames,
                    "image_packets_skipped_before_keyframe": (
                        stats.skipped_before_keyframe
                    ),
                }
            )
        return summary

    def _append_frames(self, frames) -> None:
        """把解码帧按源时间加入活动 episode；episode 之外的帧只用于维持参考。"""
        if self._active is None:
            return
        for frame in frames:
            self._active.images.append(
                ImageSample(frame.stamp_ns, frame.image)
            )

    def _handle_image(
        self, decoder: BagImageDecoder, serialized: bytes, bag_timestamp_ns: int
    ) -> None:
        """解码一条图像消息并归属到活动 episode。

        压缩视频的解码状态跨越 episode：参考包持续输入，帧在解码后才按
        episode 筛选。episode 内出现无法解码的包（例如关键帧之前的前缀）
        会明确失败，而不是悄悄缺帧。
        """
        if decoder.mode == MODE_RAW and self._active is None:
            return
        skipped_before = decoder.stats.skipped_before_keyframe
        try:
            frames = decoder.feed(serialized, bag_timestamp_ns)
        except VideoDecodeError as error:
            raise VideoDecodeError(
                f"图像话题 {self._topics['image']} 解码失败"
                f"（bag 时间 {bag_timestamp_ns} ns）: {error}"
            ) from error
        if (
            self._active is not None
            and decoder.stats.skipped_before_keyframe != skipped_before
        ):
            raise VideoDecodeError(
                f"episode {self._active.start_event.episode_index} 内的图像包"
                f"（bag 时间 {bag_timestamp_ns} ns）发生在首个 H.264 关键帧之前，"
                "无法解码"
            )
        self._append_frames(frames)


# --------------------------------------------------------------------------
# Link7 批量转换（convert_mcap 命令行入口）
# --------------------------------------------------------------------------

# 输出目录中的固定文件名。
OUTPUT_ZARR_NAME = "dataset.zarr"
MANIFEST_NAME = "manifest.json"
EXTRINSIC_MIGRATION = (
    "--extrinsic 已移除：convert_mcap 现在输出 Link7 数据，请使用 --tracker-camera-extrinsic "
    "与 --end-camera-extrinsic（缺省为包内已绑定的默认外参），并提供 --urdf 与 --output-dir"
)
# 逐 episode 写入 Zarr 根属性的元数据键。
_ZARR_EPISODE_KEYS = (
    "record_id", "episode_index", "task_name", "session_id", "collection_uuid",
    "nominal_source_fps", "nominal_source_fps_source", "measured_source_fps",
    "fitted_frame_rate_hz", "selection_path", "frame_drop_count", "image_mode",
    "image_timestamp_source", "expected_tracker_serial", "observed_tracker_serial",
    "tracker_serial_source", "tracker_binding_sha256", "tracker_camera_extrinsic_sha256",
    "end_camera_extrinsic_sha256", "tracker_time_offset_ms", "tracker_time_offset_source_s",
    "time_offset_convention", "time_calibration_status", "time_calibration_timestamp_source",
    "time_calibration_conditions",
)


@dataclass(frozen=True)
class SourceRecord:
    """一条待转换来源：稳定标识、bag 路径及可选的采集记录快照。"""

    record_id: str
    bag_uri: Path
    record_dir: Optional[Path]
    session: Optional[Dict]


def _package_paths(*parts: str) -> list[Path]:
    """返回安装 share 目录与源码树中的候选路径。"""
    candidates = []
    try:
        candidates.append(Path(get_package_share_directory("fastumi_data")).joinpath(*parts))
    except PackageNotFoundError:
        pass
    candidates.append(Path(__file__).resolve().parents[1].joinpath(*parts))
    return candidates


def _first_existing(candidates: list[Path], label: str) -> Path:
    """返回第一个存在的候选文件。"""
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到{label}：{', '.join(map(str, candidates))}")


def default_processing_path() -> Path:
    """包默认 processing.yaml。"""
    return _first_existing(_package_paths("config", "processing.yaml"), "包默认处理配置")


def default_calibration_path(name: str) -> Path:
    """包内安装的默认外参；源码树回退到仓库 ``config/calibration``。"""
    candidates = _package_paths("config", "calibration", name)
    candidates.append(Path(__file__).resolve().parents[4] / "config" / "calibration" / name)
    return _first_existing(candidates, f"默认外参 {name}")


def _is_bag_dir(path: Path) -> bool:
    """目录是否为 rosbag2 MCAP 目录。"""
    return path.is_dir() and (
        (path / "metadata.yaml").is_file() or any(path.glob("*.mcap"))
    )


def _load_session(record_dir: Path) -> Dict:
    """读取采集记录的 session.yaml。"""
    try:
        session = yaml.safe_load((record_dir / "session.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取 {record_dir / 'session.yaml'}: {error}") from error
    if not isinstance(session, dict):
        raise ValueError(f"{record_dir / 'session.yaml'} 必须是 YAML 映射")
    return session


def _record_from_dir(record_dir: Path) -> SourceRecord:
    """从含 session.yaml 的采集记录目录构造来源。"""
    session = _load_session(record_dir)
    bag_uri = record_dir / str(session.get("bag_uri") or "raw/bag")
    if not _is_bag_dir(bag_uri):
        raise ValueError(f"记录 {record_dir} 的 bag {bag_uri} 不存在")
    task = str(session.get("task_name") or record_dir.parent.name).strip()
    name = str(session.get("dir_name") or record_dir.name).strip()
    return SourceRecord(f"{task}/{name}", bag_uri.resolve(), record_dir.resolve(), session)


def discover_sources(input_path: Path) -> list[SourceRecord]:
    """发现单个 bag、单条已保存记录或任务目录下的全部来源，按来源路径稳定排序。

    任务目录内跳过 ``.staging`` 和 ``.trash``；目录符号链接不跟随。
    """
    path = Path(input_path).expanduser().resolve()
    if path.is_file() and path.suffix == ".mcap":
        return [SourceRecord(path.stem, path, None, None)]
    if not path.is_dir():
        raise ValueError(f"输入 {path} 不是目录或 .mcap 文件")
    if (path / "session.yaml").is_file():
        return [_record_from_dir(path)]
    if _is_bag_dir(path):
        record_dir = path.parents[1] if len(path.parents) > 1 else None
        if record_dir is not None and (record_dir / "session.yaml").is_file():
            record = _record_from_dir(record_dir)
            if record.bag_uri == path:
                return [record]
        return [SourceRecord(path.name, path, None, None)]
    records: list[SourceRecord] = []
    bags: list[Path] = []
    for directory, dirs, files in os.walk(path, followlinks=False):
        current = Path(directory)
        dirs[:] = sorted(
            name for name in dirs
            if name not in (".staging", ".trash") and not (current / name).is_symlink()
        )
        if "session.yaml" in files:
            records.append(_record_from_dir(current))
            dirs.clear()
        elif _is_bag_dir(current):
            bags.append(current)
            dirs.clear()
    for bag in bags:
        records.append(SourceRecord(bag.relative_to(path).as_posix(), bag, None, None))
    records.sort(key=lambda record: str(record.record_dir or record.bag_uri))
    identifiers = [record.record_id for record in records]
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    if duplicates:
        raise ValueError(f"来源记录标识重复: {', '.join(duplicates)}")
    return records


def _snapshot_processing(record: SourceRecord) -> Optional[Path]:
    """返回记录快照中的 processing.yaml 路径。"""
    if record.record_dir is None or not record.session:
        return None
    entry = record.session.get("processing_config")
    relative = entry.get("path") if isinstance(entry, dict) else None
    candidate = record.record_dir / (relative or "calibration_snapshot/processing.yaml")
    return candidate if candidate.is_file() else None


def _sha256_file(path: Path) -> str:
    """返回文件原始字节的 SHA-256。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy_snapshot(source: Path, target_dir: Path, expected_sha: str) -> str:
    """复制快照并核对摘要，返回快照相对路径名。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(source).name
    shutil.copy2(source, target)
    if _sha256_file(target) != expected_sha:
        raise CalibrationError(f"快照 {target} 摘要与源文件不一致")
    return target.name


def _build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="convert_mcap",
        description=(
            "把 FastUMI MCAP 记录转换为 30 Hz Link7 HDF5，并合并导出 "
            "fastumi-rm75-link7-v1 Zarr"
        ),
    )
    parser.add_argument("input", help="单个 bag、单条已保存记录或任务目录")
    parser.add_argument("--output-dir", required=True, help="新输出目录，必须尚不存在")
    parser.add_argument("--urdf", required=True, help="实际部署使用的 URDF")
    parser.add_argument(
        "--tracker-camera-extrinsic", default="",
        help="T_cam_tracker 外参；缺省为包内已绑定的默认外参",
    )
    parser.add_argument(
        "--end-camera-extrinsic", default="",
        help="T_cam_end 外参；缺省为包内默认外参",
    )
    parser.add_argument(
        "--tracker-serial", default="",
        help="自定义外参缺少设备绑定时声明标定所用 Tracker 序列号；不能覆盖既有绑定",
    )
    parser.add_argument(
        "--tracker-time-offset-convention", default="",
        choices=["", *sorted(TIME_OFFSET_CONVENTIONS)],
        help="外参与绑定均未声明 time_offset 语义时显式指定",
    )
    parser.add_argument(
        "--source-camera-fps", type=float, default=None,
        help="标称源帧率覆盖值（如 30、29.97、90）；缺省读取各记录快照",
    )
    parser.add_argument("--config", default="", help="可选处理配置；缺省用记录快照或包默认配置")
    return parser


def _episode_zarr_metadata(metadata: Dict) -> Dict:
    """挑选写入 Zarr 逐 episode 元数据的字段。"""
    return {key: metadata.get(key) for key in _ZARR_EPISODE_KEYS}


def run_batch(argv: Optional[list[str]] = None) -> int:
    """执行批量转换，返回进程退出码（0 成功，1 失败）。"""
    arguments_list = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if any(item == "--extrinsic" or item.startswith("--extrinsic=") for item in arguments_list):
        parser.error(EXTRINSIC_MIGRATION)
    arguments = parser.parse_args(arguments_list)
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    if output_dir.exists():
        parser.error(f"--output-dir {output_dir} 已存在；请指定新的输出目录")
    urdf_path = Path(arguments.urdf).expanduser().resolve()
    if not urdf_path.is_file():
        parser.error(f"--urdf {urdf_path} 不存在")
    urdf_sha256 = _sha256_file(urdf_path)
    try:
        tracker_extrinsic = (
            Path(arguments.tracker_camera_extrinsic).expanduser()
            if arguments.tracker_camera_extrinsic
            else default_calibration_path("T_cam_tracker.yaml")
        )
        end_extrinsic = (
            Path(arguments.end_camera_extrinsic).expanduser()
            if arguments.end_camera_extrinsic
            else default_calibration_path("T_cam_end.yaml")
        )
        calibration = load_link7_calibration(
            str(tracker_extrinsic),
            str(end_extrinsic),
            arguments.tracker_serial,
            arguments.tracker_time_offset_convention,
        )
        sources = discover_sources(Path(arguments.input))
        if not sources:
            raise ValueError(f"{arguments.input} 下没有可转换的 MCAP 记录")
        explicit = Path(arguments.config).expanduser() if arguments.config else None
        default_config = default_processing_path()
        plans = []
        for record in sources:
            processing = resolve_processing(explicit, _snapshot_processing(record), default_config)
            fps, fps_source = resolve_source_fps(
                arguments.source_camera_fps, record.session, record.record_id
            )
            image = (record.session or {}).get("image") or {}
            try:
                note = check_timestamp_source(calibration, image.get("timestamp_source"))
            except CalibrationError as error:
                raise CalibrationError(f"{record.record_id}: {error}") from error
            if note:
                print(f"提示：{record.record_id} {note}")
            plans.append((record, processing, fps, fps_source))
        print(f"发现 {len(plans)} 条来源记录；预检 Tracker 设备身份 {calibration.tracker_serial}")
        identities = {}
        for record, processing, _, _ in plans:
            identities[record.record_id] = precheck_tracker_identity(
                record.bag_uri, processing.topics["tracker_status"], calibration.tracker_serial
            )
    except (CalibrationError, ConfigError, ValueError, OSError, RuntimeError) as error:
        print(f"转换前检查失败：{error}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True)
    manifest: Dict = {
        "format": ZARR_FORMAT,
        "hdf5_schema": LINK7_SCHEMA_VERSION,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(arguments.input).expanduser().resolve()),
        "output_dir": str(output_dir),
        "arguments": arguments_list,
        "calibration": calibration.metadata(),
        "urdf": {"path": str(urdf_path), "sha256": urdf_sha256},
        "sources": [],
        "zarr": None,
        "warnings": [],
    }
    manifest_path = output_dir / MANIFEST_NAME

    def save_manifest() -> None:
        write_json_report(str(manifest_path), manifest)

    accepted_sources: list[ZarrEpisodeSource] = []
    try:
        snapshots = output_dir / "snapshots"
        manifest["calibration"]["snapshots"] = {
            "tracker_camera": _copy_snapshot(
                Path(calibration.tracker_camera.path), snapshots / "calibration",
                calibration.tracker_camera.sha256,
            ),
            "end_camera": _copy_snapshot(
                Path(calibration.end_camera.path), snapshots / "calibration",
                calibration.end_camera.sha256,
            ),
        }
        if calibration.binding is not None:
            manifest["calibration"]["snapshots"]["binding"] = _copy_snapshot(
                Path(calibration.binding.path), snapshots / "calibration",
                calibration.binding.sha256,
            )
        manifest["urdf"]["snapshot"] = "snapshots/urdf/" + _copy_snapshot(
            urdf_path, snapshots / "urdf", urdf_sha256
        )
        save_manifest()
        for record, processing, fps, fps_source in plans:
            record_dir = output_dir / "records" / record.record_id
            work_dir = output_dir / ".work" / record.record_id
            session_image = (record.session or {}).get("image") or {}
            base_metadata = dict(calibration.metadata())
            base_metadata.update(identities[record.record_id])
            base_metadata.update(
                {
                    "record_id": record.record_id,
                    "source_bag": str(record.bag_uri),
                    "source_record_dir": str(record.record_dir or ""),
                    "collection_uuid": str((record.session or {}).get("collection_uuid", "")),
                    "nominal_source_fps": fps,
                    "nominal_source_fps_source": fps_source,
                    "selection_path": selection_path_for(fps),
                    "image_timestamp_source": str(
                        session_image.get("timestamp_source") or "unrecorded"
                    ),
                    "image_timestamp_source_normalized": normalize_timestamp_source(
                        session_image.get("timestamp_source")
                    ),
                    "tracker_status_topic": processing.topics["tracker_status"],
                    "processing": processing.report(),
                    "urdf_sha256": urdf_sha256,
                    "urdf_file": urdf_path.name,
                }
            )
            print(
                f"转换 {record.record_id}（标称 {fps:g} Hz，{fps_source}，"
                f"{selection_path_for(fps)}，配置 {processing.source}）"
            )
            for item in processing.adjustments:
                if item["original"] is not None:
                    print(f"  配置调整 {item['field']}: {item['original']} → {item['effective']}（{item['reason']}）")
            summary = Link7RecordConverter(
                record.record_id, record.bag_uri, record_dir, work_dir,
                processing, calibration, fps, base_metadata,
            ).convert()
            record_report = {
                "record_id": record.record_id,
                "source_bag": str(record.bag_uri),
                "nominal_source_fps": fps,
                "nominal_source_fps_source": fps_source,
                "processing": processing.report(),
                "identity": identities[record.record_id],
                "decode": summary.stats,
                "accepted": [
                    {key: item[key] for key in ("name", "hdf5", "report", "frames")}
                    for item in summary.accepted
                ],
                "rejected": summary.rejected,
            }
            write_json_report(str(record_dir / "record_report.json"), record_report)
            manifest["sources"].append(record_report)
            for item in summary.accepted:
                accepted_sources.append(
                    ZarrEpisodeSource(
                        item["name"], Path(item["hdf5"]),
                        _episode_zarr_metadata(item["metadata"]),
                    )
                )
            save_manifest()
        shutil.rmtree(output_dir / ".work", ignore_errors=True)
        if not accepted_sources:
            manifest["status"] = "failed"
            manifest["error"] = "没有任何有效 episode"
            save_manifest()
            print("转换失败：没有任何有效 episode", file=sys.stderr)
            return 1
        manifest["zarr"] = export_link7_zarr(
            accepted_sources,
            output_dir / OUTPUT_ZARR_NAME,
            urdf_sha256,
            {
                "source_manifest": MANIFEST_NAME,
                "hdf5_schema": LINK7_SCHEMA_VERSION,
                "urdf_file": manifest["urdf"]["snapshot"],
                "expected_tracker_serial": calibration.tracker_serial,
                "tracker_camera_extrinsic_sha256": calibration.tracker_camera.sha256,
                "end_camera_extrinsic_sha256": calibration.end_camera.sha256,
            },
        )
    except Exception as error:  # 硬错误：中止批次并保留诊断产物。
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        save_manifest()
        print(f"转换中止：{manifest['error']}", file=sys.stderr)
        return 1
    if len(accepted_sources) < 2:
        message = "只有 1 条有效 episode：当前训练默认验证划分至少需要两条有效 episode"
        manifest["warnings"].append(message)
        print(f"提示：{message}")
    manifest["status"] = "succeeded"
    save_manifest()
    rejected = sum(len(item["rejected"]) for item in manifest["sources"])
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "accepted_episodes": len(accepted_sources),
                "rejected_episodes": rejected,
                "frames": manifest["zarr"]["frames"],
                "dataset": manifest["zarr"]["path"],
                "time_calibration_status": manifest["zarr"]["time_calibration_status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main(argv: Optional[list[str]] = None) -> None:
    """``convert_mcap`` 命令行入口。"""
    sys.exit(run_batch(argv))


if __name__ == "__main__":
    main()
