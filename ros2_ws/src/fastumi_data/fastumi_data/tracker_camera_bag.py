"""以两遍流式方式读取标定 MCAP，并提供 Tracker 时间插值。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from cv_bridge import CvBridge
import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message

from fastumi_data.models import PoseSample, TrackerStatusSample
from fastumi_data.pose_math import interpolate_pose, pose_to_matrix


@dataclass(frozen=True)
class ImageFrame:
    """保存图像 header 时间、bag 写入时间和 BGR 像素数组。"""

    timestamp_ns: int
    bag_timestamp_ns: int
    image: np.ndarray


@dataclass(frozen=True)
class TrackerTimeline:
    """保存按 header 时间递增排列的 Tracker 位姿和状态样本。"""

    poses: tuple[PoseSample, ...]
    statuses: tuple[TrackerStatusSample, ...]
    pose_timestamps_ns: np.ndarray = field(
        init=False, repr=False, compare=False
    )
    status_timestamps_ns: np.ndarray = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """一次性校验时间顺序并缓存只读时间戳数组。"""
        pose_timestamps = np.fromiter(
            (sample.timestamp_ns for sample in self.poses), dtype=np.int64
        )
        status_timestamps = np.fromiter(
            (sample.timestamp_ns for sample in self.statuses), dtype=np.int64
        )
        for name, timestamps in (
            ("Tracker pose", pose_timestamps),
            ("Tracker status", status_timestamps),
        ):
            if np.any(np.diff(timestamps) <= 0):
                raise ValueError(f"{name} 时间戳必须严格递增")
            timestamps.setflags(write=False)
        object.__setattr__(self, "pose_timestamps_ns", pose_timestamps)
        object.__setattr__(self, "status_timestamps_ns", status_timestamps)


def _stamp_to_ns(stamp: object) -> int:
    """把 ROS builtin_interfaces/Time 转换为纳秒整数。"""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _open_reader(bag_uri: str) -> rosbag2_py.SequentialReader:
    """按 MCAP 存储格式打开顺序读取器。"""
    bag_path = Path(bag_uri)
    if not bag_path.exists():
        raise ValueError(f"MCAP 路径不存在: {bag_path}")
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(
            rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""),
        )
    except RuntimeError as error:
        raise ValueError(f"无法打开 MCAP {bag_path}: {error}") from error
    return reader


def _topic_message_types(
    reader: rosbag2_py.SequentialReader,
    required_topics: Sequence[str],
    expected_types: Mapping[str, str] | None = None,
) -> dict[str, type]:
    """解析 bag 话题消息类，并检查必需话题及指定消息类型。"""
    topic_types = {
        metadata.name: metadata.type
        for metadata in reader.get_all_topics_and_types()
    }
    missing_topics = [
        topic for topic in required_topics if topic not in topic_types
    ]
    if missing_topics:
        raise ValueError(f"MCAP 缺少话题: {', '.join(missing_topics)}")
    for topic, expected_type in (expected_types or {}).items():
        actual_type = topic_types[topic]
        if actual_type != expected_type:
            raise ValueError(
                f"话题 {topic} 必须使用 {expected_type}，实际为 {actual_type}"
            )
    return {
        topic: get_message(topic_types[topic]) for topic in required_topics
    }


def _check_monotonic(
    previous_ns: int | None, current_ns: int, topic: str
) -> None:
    """要求单个话题的 header 时间严格递增。"""
    if previous_ns is not None and current_ns <= previous_ns:
        raise ValueError(f"话题 {topic} 的 header 时间戳不是严格递增")


def image_message_to_frame(
    message: object,
    bag_timestamp_ns: int,
    bridge: CvBridge | None = None,
) -> ImageFrame:
    """使用 Image.header.stamp 构造 BGR 图像帧并保留 bag 时间诊断。"""
    converter = bridge if bridge is not None else CvBridge()
    image = converter.imgmsg_to_cv2(message, desired_encoding="bgr8")
    pixels = np.asarray(image).copy()
    return ImageFrame(
        timestamp_ns=_stamp_to_ns(message.header.stamp),
        bag_timestamp_ns=int(bag_timestamp_ns),
        image=pixels,
    )


def read_tracker_timeline(
    bag_uri: str,
    tracker_topic: str = "/vive_tracker/odom",
    status_topic: str = "/vive_tracker/status",
) -> TrackerTimeline:
    """第一遍流式读取 Tracker Odometry/status 并返回 header 时间线。"""
    reader = _open_reader(bag_uri)
    message_types = _topic_message_types(
        reader,
        (tracker_topic, status_topic),
        {tracker_topic: "nav_msgs/msg/Odometry"},
    )
    poses = []
    statuses = []
    previous_timestamps: dict[str, int | None] = {
        tracker_topic: None,
        status_topic: None,
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
            pose = message.pose.pose
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
        else:
            statuses.append(
                TrackerStatusSample(
                    timestamp_ns=timestamp_ns,
                    device_connected=bool(message.device_connected),
                    pose_valid=bool(message.pose_valid),
                    tracking_state=int(message.tracking_state),
                )
            )
    if len(poses) < 2:
        raise ValueError("Tracker Odometry 话题至少需要两个样本")
    if not statuses:
        raise ValueError("Tracker status 话题没有样本")
    return TrackerTimeline(tuple(poses), tuple(statuses))


def iter_image_frames(
    bag_uri: str,
    image_topic: str,
    frame_stride: int = 1,
    bridge: CvBridge | None = None,
) -> Iterator[ImageFrame]:
    """第二遍重新打开 MCAP，并按步长流式解码目标图像。"""
    if frame_stride <= 0:
        raise ValueError("frame_stride 必须为正整数")
    reader = _open_reader(bag_uri)
    message_types = _topic_message_types(reader, (image_topic,))
    converter = bridge if bridge is not None else CvBridge()
    image_index = 0
    previous_timestamp_ns = None
    while reader.has_next():
        topic, serialized, bag_timestamp_ns = reader.read_next()
        if topic != image_topic:
            continue
        message = deserialize_message(serialized, message_types[image_topic])
        timestamp_ns = _stamp_to_ns(message.header.stamp)
        _check_monotonic(previous_timestamp_ns, timestamp_ns, image_topic)
        previous_timestamp_ns = timestamp_ns
        selected = image_index % frame_stride == 0
        image_index += 1
        if selected:
            yield image_message_to_frame(
                message, bag_timestamp_ns, bridge=converter
            )


def interpolate_world_from_tracker(
    samples: Sequence[PoseSample],
    target_ns: int,
    max_gap_ms: float,
    timestamps_ns: np.ndarray | None = None,
) -> tuple[np.ndarray, float] | None:
    """在相邻 Tracker 样本之间插值 ``^world T_tracker``。

    平移使用线性插值，姿态使用最短路径 SLERP。返回值中的间隔单位为毫秒。
    时间边界外、样本未递增或包围间隔超过门限时返回 ``None``。
    """
    if len(samples) < 2:
        return None
    if max_gap_ms <= 0.0:
        raise ValueError("max_gap_ms 必须为正数")
    if timestamps_ns is None:
        timestamps = np.fromiter(
            (sample.timestamp_ns for sample in samples), dtype=np.int64
        )
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Tracker pose 时间戳必须严格递增")
    else:
        timestamps = np.asarray(timestamps_ns, dtype=np.int64)
        if timestamps.shape != (len(samples),):
            raise ValueError("Tracker pose 时间戳缓存长度不匹配")
    insertion = int(np.searchsorted(timestamps, target_ns, side="left"))
    if insertion == 0 or insertion >= len(samples):
        return None
    first, second = samples[insertion - 1], samples[insertion]
    gap_ns = second.timestamp_ns - first.timestamp_ns
    if gap_ns > int(max_gap_ms * 1.0e6):
        return None
    ratio = (target_ns - first.timestamp_ns) / gap_ns
    position, quaternion = interpolate_pose(
        first.position_m,
        first.quaternion_xyzw,
        second.position_m,
        second.quaternion_xyzw,
        ratio,
    )
    return pose_to_matrix(position, quaternion), gap_ns / 1.0e6


def tracker_status_valid_at(
    samples: Sequence[TrackerStatusSample],
    target_ns: int,
    maximum_delta_ms: float,
    timestamps_ns: np.ndarray | None = None,
) -> bool:
    """返回最近 Tracker 状态是否足够接近且满足 6DoF 有效条件。"""
    if maximum_delta_ms < 0.0:
        raise ValueError("maximum_delta_ms 不能为负数")
    if not samples:
        return False
    if timestamps_ns is None:
        timestamps = np.fromiter(
            (sample.timestamp_ns for sample in samples), dtype=np.int64
        )
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Tracker status 时间戳必须严格递增")
    else:
        timestamps = np.asarray(timestamps_ns, dtype=np.int64)
        if timestamps.shape != (len(samples),):
            raise ValueError("Tracker status 时间戳缓存长度不匹配")
    insertion = int(np.searchsorted(timestamps, target_ns, side="left"))
    candidates = []
    if insertion < len(samples):
        candidates.append(insertion)
    if insertion > 0:
        candidates.append(insertion - 1)
    nearest_index = min(
        candidates, key=lambda index: abs(timestamps[index] - target_ns)
    )
    if abs(timestamps[nearest_index] - target_ns) > int(
        maximum_delta_ms * 1.0e6
    ):
        return False
    sample = samples[nearest_index]
    return (
        sample.device_connected
        and sample.pose_valid
        and sample.tracking_state == 3
    )


def tracker_status_valid_for_interval(
    samples: Sequence[TrackerStatusSample],
    start_ns: int,
    end_ns: int,
    maximum_delta_ms: float,
    timestamps_ns: np.ndarray | None = None,
) -> bool:
    """检查区间端点可配对，并要求区间内全部 Tracker 状态有效。"""
    if start_ns > end_ns:
        raise ValueError("Tracker 状态区间起点不能晚于终点")
    if not samples:
        return False
    if timestamps_ns is None:
        timestamps = np.fromiter(
            (sample.timestamp_ns for sample in samples), dtype=np.int64
        )
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Tracker status 时间戳必须严格递增")
    else:
        timestamps = np.asarray(timestamps_ns, dtype=np.int64)
        if timestamps.shape != (len(samples),):
            raise ValueError("Tracker status 时间戳缓存长度不匹配")
    for target_ns in (start_ns, end_ns):
        if not tracker_status_valid_at(
            samples,
            target_ns,
            maximum_delta_ms,
            timestamps,
        ):
            return False
    first = int(np.searchsorted(timestamps, start_ns, side="left"))
    last = int(np.searchsorted(timestamps, end_ns, side="right"))
    return all(
        sample.device_connected and sample.pose_valid
        and sample.tracking_state == 3
        for sample in samples[first:last]
    )
