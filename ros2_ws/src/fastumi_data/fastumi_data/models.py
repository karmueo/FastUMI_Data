"""定义 FastUMI 离线同步使用的纯 Python 数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass(frozen=True)
class ImageSample:
    """保存一帧已转换为 RGB 的鱼眼图像。"""

    timestamp_ns: int
    image_rgb: np.ndarray


@dataclass(frozen=True)
class PoseSample:
    """保存一个世界坐标系中的 Tracker 位姿。"""

    timestamp_ns: int
    position_m: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class GripperSample:
    """保存一个带源图像时间戳的夹爪估计状态。"""

    timestamp_ns: int
    raw_openness: float
    filtered_openness: float
    detected_marker_count: int
    valid: bool


@dataclass(frozen=True)
class TrackerStatusSample:
    """保存 Tracker 连接、位姿有效性和 OpenVR 跟踪状态。"""

    timestamp_ns: int
    device_connected: bool
    pose_valid: bool
    tracking_state: int


@dataclass(frozen=True)
class EpisodeEventRecord:
    """保存从 MCAP 读取的 episode 边界事件。"""

    timestamp_ns: int
    session_id: str
    task_name: str
    episode_index: int
    event_type: int


@dataclass
class EpisodeBuffer:
    """积累单条 episode 的原始图像、位姿和夹爪消息。"""

    start_event: EpisodeEventRecord
    images: List[ImageSample] = field(default_factory=list)
    poses: List[PoseSample] = field(default_factory=list)
    grippers: List[GripperSample] = field(default_factory=list)
    tracker_statuses: List[TrackerStatusSample] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class ProcessingConfig:
    """定义离线同步、插值和质量门限。"""

    sample_rate_hz: float = 20.0
    max_image_delta_s: float = 0.03
    max_pose_gap_s: float = 0.1
    max_gripper_gap_s: float = 0.2
    minimum_samples: int = 10
    require_tracker_status: bool = True


@dataclass(frozen=True)
class SynchronizedEpisode:
    """保存完成时间同步和 TCP 坐标转换的 episode。"""

    timestamp_ns: np.ndarray
    images_rgb: np.ndarray
    qpos: np.ndarray
    gripper_observed: np.ndarray
    tracker_tracking_ok: np.ndarray
    pose_gap_ms: np.ndarray
    gripper_gap_ms: np.ndarray


@dataclass(frozen=True)
class ProcessingResult:
    """保存单条 episode 的处理结果或明确拒绝原因。"""

    episode: SynchronizedEpisode | None
    rejection_reasons: List[str]
    warnings: List[str]
