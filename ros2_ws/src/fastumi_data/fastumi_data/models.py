"""定义 FastUMI 离线同步使用的纯 Python 数据模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
