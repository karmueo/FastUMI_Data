"""单遍读取双 ArUco 标定所需的 Tracker、状态和夹爪时间线。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from rclpy.serialization import deserialize_message

from fastumi_data.models import GripperSample, PoseSample, TrackerStatusSample
from fastumi_data.tracker_camera_bag import (
    _check_monotonic,
    _open_reader,
    _stamp_to_ns,
    _topic_message_types,
)


@dataclass(frozen=True)
class ArucoTcpTimeline:
    """保存按 header 时间递增的 Tracker、状态和 GripperState 样本。"""

    poses: tuple[PoseSample, ...]
    statuses: tuple[TrackerStatusSample, ...]
    grippers: tuple[GripperSample, ...]
    pose_timestamps_ns: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    status_timestamps_ns: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    gripper_timestamps_ns: np.ndarray = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """校验时间顺序并创建只读的时间戳索引。"""
        if len(self.poses) < 2:
            raise ValueError("Tracker pose 话题至少需要两个样本")
        if not self.statuses:
            raise ValueError("Tracker status 话题没有样本")
        if not self.grippers:
            raise ValueError("GripperState 话题没有样本")
        timestamps = {
            "Tracker pose": np.fromiter(
                (sample.timestamp_ns for sample in self.poses), dtype=np.int64
            ),
            "Tracker status": np.fromiter(
                (sample.timestamp_ns for sample in self.statuses), dtype=np.int64
            ),
            "GripperState": np.fromiter(
                (sample.timestamp_ns for sample in self.grippers), dtype=np.int64
            ),
        }
        for description, values in timestamps.items():
            if np.any(np.diff(values) <= 0):
                raise ValueError(f"{description} 时间戳必须严格递增")
            values.setflags(write=False)
        object.__setattr__(self, "pose_timestamps_ns", timestamps["Tracker pose"])
        object.__setattr__(self, "status_timestamps_ns", timestamps["Tracker status"])
        object.__setattr__(self, "gripper_timestamps_ns", timestamps["GripperState"])

    def interpolate_openness(
        self, timestamp_ns: int, maximum_gap_ms: float
    ) -> float | None:
        """按有效 raw_openness 在相邻样本间线性插值。"""
        if not np.isfinite(maximum_gap_ms) or maximum_gap_ms <= 0.0:
            raise ValueError("maximum_gap_ms 必须为有限正数")
        valid_samples = [
            sample
            for sample in self.grippers
            if sample.valid
            and np.isfinite(sample.raw_openness)
            and 0.0 <= sample.raw_openness <= 1.0
        ]
        if not valid_samples:
            return None
        valid_timestamps = np.asarray(
            [sample.timestamp_ns for sample in valid_samples], dtype=np.int64
        )
        target = int(timestamp_ns)
        insertion = int(np.searchsorted(valid_timestamps, target, side="left"))
        if insertion < len(valid_samples) and valid_timestamps[insertion] == target:
            return float(valid_samples[insertion].raw_openness)
        if insertion == 0 or insertion >= len(valid_samples):
            return None
        first = valid_samples[insertion - 1]
        second = valid_samples[insertion]
        gap_ns = second.timestamp_ns - first.timestamp_ns
        if gap_ns <= 0 or gap_ns > int(round(maximum_gap_ms * 1.0e6)):
            return None
        ratio = (target - first.timestamp_ns) / gap_ns
        value = (1.0 - ratio) * first.raw_openness + ratio * second.raw_openness
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            return None
        return float(value)


def read_aruco_tcp_timeline(
    bag_uri: str,
    tracker_topic: str = "/vive_tracker/pose",
    status_topic: str = "/vive_tracker/status",
    gripper_topic: str = "/gripper/state",
) -> ArucoTcpTimeline:
    """单遍读取三个话题，并严格使用消息 header 时间排序。"""
    reader = _open_reader(bag_uri)
    topics = (tracker_topic, status_topic, gripper_topic)
    message_types = _topic_message_types(reader, topics)
    poses: list[PoseSample] = []
    statuses: list[TrackerStatusSample] = []
    grippers: list[GripperSample] = []
    previous_timestamps: dict[str, int | None] = {
        topic: None for topic in topics
    }
    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        if topic not in message_types:
            continue
        message = deserialize_message(serialized, message_types[topic])
        timestamp_ns = _stamp_to_ns(message.header.stamp)
        _check_monotonic(previous_timestamps[topic], timestamp_ns, topic)
        previous_timestamps[topic] = timestamp_ns
        if topic == tracker_topic:
            pose = message.pose
            poses.append(
                PoseSample(
                    timestamp_ns=timestamp_ns,
                    position_m=np.asarray(
                        [pose.position.x, pose.position.y, pose.position.z],
                        dtype=np.float64,
                    ),
                    quaternion_xyzw=np.asarray(
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
        elif topic == status_topic:
            statuses.append(
                TrackerStatusSample(
                    timestamp_ns=timestamp_ns,
                    device_connected=bool(message.device_connected),
                    pose_valid=bool(message.pose_valid),
                    tracking_state=int(message.tracking_state),
                )
            )
        else:
            grippers.append(
                GripperSample(
                    timestamp_ns=timestamp_ns,
                    raw_openness=float(message.raw_openness),
                    filtered_openness=float(message.filtered_openness),
                    detected_marker_count=int(message.detected_marker_count),
                    valid=bool(message.valid),
                )
            )
    return ArucoTcpTimeline(tuple(poses), tuple(statuses), tuple(grippers))
