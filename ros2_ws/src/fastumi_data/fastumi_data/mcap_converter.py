"""读取连续 ROS 2 MCAP 会话并生成同步的 FastUMI HDF5 episodes。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from fastumi_interfaces.msg import EpisodeEvent, TrackerStatus
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import yaml

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
from fastumi_data.synchronizer import synchronize_episode


def _stamp_to_ns(stamp) -> int:
    """把 builtin_interfaces/Time 转换为 Unix 纳秒整数。"""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _load_processing_document(path: str) -> tuple[ProcessingConfig, Dict]:
    """加载同步参数和话题映射。"""
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
        self._bridge = CvBridge()
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
        )
        report = build_quality_report(
            result.episode,
            result.warnings,
            self._extrinsic.calibration_hash,
            tracker_time_offset_ms=self._extrinsic.time_offset_ms,
            source_calibration_sha256=(
                self._extrinsic.source_calibration_sha256
            ),
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
        message_count = 0
        while reader.has_next():
            topic, serialized, _ = reader.read_next()
            if topic not in message_types:
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
            if topic == self._topics["image"]:
                timestamp_ns = _stamp_to_ns(message.header.stamp)
                image_rgb = self._bridge.imgmsg_to_cv2(
                    message, desired_encoding="rgb8"
                )
                self._active.images.append(
                    ImageSample(timestamp_ns, np.asarray(image_rgb).copy())
                )
            elif topic == self._topics["tracker_pose"]:
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
        if self._active is not None:
            self._write_rejection(
                self._active.start_event, ["MCAP 结束时 episode 尚未 STOP"]
            )
            self._active = None
        return {
            "messages": message_count,
            "converted": self._converted_count,
            "rejected": self._rejected_count,
        }


def main(argv: Optional[list[str]] = None) -> None:
    """解析参数并执行 MCAP 到 HDF5 转换。"""
    parser = argparse.ArgumentParser(
        description="把 FastUMI ROS2 MCAP 转换为 HDF5 episodes"
    )
    parser.add_argument("bag_uri", help="rosbag2 MCAP 目录")
    parser.add_argument("--extrinsic", required=True)
    parser.add_argument(
        "--config",
        default=str(
            Path(get_package_share_directory("fastumi_data"))
            / "config"
            / "processing.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="默认使用 <session>/ 目录",
    )
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)
    processing_config, topics = _load_processing_document(arguments.config)
    bag_path = Path(arguments.bag_uri).resolve()
    output_dir = (
        Path(arguments.output_dir).resolve()
        if arguments.output_dir
        else bag_path.parents[1]
    )
    rclpy.init()
    try:
        converter = McapEpisodeConverter(
            str(bag_path),
            str(output_dir),
            processing_config,
            topics,
            arguments.extrinsic,
            arguments.force,
        )
        summary = converter.convert()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
