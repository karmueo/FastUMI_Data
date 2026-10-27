"""按所选图像的实际时间同步 Tracker 位姿与夹爪，生成 episode 起始 UMI TCP 相对轨迹。

时间约定：图像 ``k`` 的实际消息时间为 ``t_image(k)``。Tracker 位姿与跟踪状态在
``t_image(k) + tracker_time_offset_ns`` 查询（偏移已按外参语义换算，见
:mod:`fastumi_data.link7_calibration`），位置线性插值、旋转 SLERP；夹爪在
``t_image(k)`` 插值。名义时间只用于标注，不参与插值。

几何约定：``T_W_TCP(t) = T_W_Tracker(t + offset) × T_Tracker_TCP``，再以
裁剪后首帧为参考 ``T_TCPstart_TCP(t) = inverse(T_W_TCP(t0)) × T_W_TCP(t)``。
``qpos`` 每行为 ``[x, y, z, qx, qy, qz, qw, gripper]``，位置单位米，夹爪 [0,1]。

首尾因缺少插值支撑可以裁剪；帧间隔异常、拟合帧率偏差、有效帧率不足、高帧率
匹配超限/复用，以及内部同步缺口或跟踪异常都拒绝整条 episode。
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fastumi_data.frame_selection import (
    FrameSelection,
    FrameSelectionConfig,
    TARGET_FREQUENCY_HZ,
    issue,
    select_frames,
    windows_crossing_drops,
)
from fastumi_data.models import GripperSample, PoseSample, TrackerStatusSample
from fastumi_data.pose_math import (
    interpolate_pose,
    pose_to_matrix,
    transform_series_to_episode_frame,
)


# 新格式最终至少保留的帧数（16 步动作窗口）。
MINIMUM_LINK7_FRAMES = 16
# 训练时一个样本覆盖的相邻行数：2 帧观测（向前 1 行）+ 16 步动作。
TRAINING_WINDOW_FRAMES = 17

SYNC_GAP = "SYNC_GAP"
TRACKING_LOST = "TRACKING_LOST"
TOO_FEW_FRAMES = "TOO_FEW_FRAMES"
MISSING_STREAM = "MISSING_STREAM"
NONFINITE_RESULT = "NONFINITE_RESULT"
HEAD_TAIL_TRIMMED = "HEAD_TAIL_TRIMMED"


@dataclass(frozen=True)
class Link7SyncConfig:
    """示教轨迹同步与质量门限。"""

    max_pose_gap_s: float = 0.10
    max_gripper_gap_s: float = 0.20
    selection: FrameSelectionConfig = field(default_factory=FrameSelectionConfig)


@dataclass
class Link7Episode:
    """完成同步和 TCP 相对变换的 episode；图像以位置索引引用源序列。"""

    selection: FrameSelection
    """裁剪后的选帧（名义时间、丢帧、匹配误差均已按首帧重算）。"""
    qpos: np.ndarray
    gripper_observed: np.ndarray
    tracker_tracking_ok: np.ndarray
    pose_gap_ms: np.ndarray
    gripper_gap_ms: np.ndarray
    world_from_start_tcp: np.ndarray
    """裁剪后首帧的 ``T_W_TCP``，用于追溯。"""

    def __len__(self) -> int:
        """输出帧数。"""
        return int(self.qpos.shape[0])


@dataclass
class Link7SyncResult:
    """单条 episode 的同步结果；``episode`` 为 None 表示整条拒绝。"""

    episode: Optional[Link7Episode]
    rejections: List[Dict]
    warnings: List[Dict]
    stats: Dict


def _nearest_index(timestamps: Sequence[int], target: int) -> int:
    """返回有序时间戳中距离目标最近的索引。"""
    insertion = bisect_left(timestamps, target)
    if insertion <= 0:
        return 0
    if insertion >= len(timestamps):
        return len(timestamps) - 1
    before = insertion - 1
    if target - timestamps[before] <= timestamps[insertion] - target:
        return before
    return insertion


def _interpolate_pose_sample(
    samples: Sequence[PoseSample],
    timestamps: Sequence[int],
    target_ns: int,
    maximum_gap_ns: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """在目标时间插值位姿并返回插值跨度毫秒数。"""
    insertion = bisect_left(timestamps, target_ns)
    if insertion < len(timestamps) and timestamps[insertion] == target_ns:
        sample = samples[insertion]
        return sample.position_m, sample.quaternion_xyzw, 0.0
    if insertion == 0 or insertion >= len(samples):
        return None
    first = samples[insertion - 1]
    second = samples[insertion]
    gap_ns = second.timestamp_ns - first.timestamp_ns
    if gap_ns <= 0 or gap_ns > maximum_gap_ns:
        return None
    ratio = (target_ns - first.timestamp_ns) / gap_ns
    position, quaternion = interpolate_pose(
        first.position_m,
        first.quaternion_xyzw,
        second.position_m,
        second.quaternion_xyzw,
        ratio,
    )
    return position, quaternion, gap_ns / 1.0e6


def _interpolate_gripper_sample(
    samples: Sequence[GripperSample],
    timestamps: Sequence[int],
    target_ns: int,
    maximum_gap_ns: int,
    field_name: str = "filtered_openness",
    interpolation: str = "linear",
) -> Optional[Tuple[float, bool, float]]:
    """在有效夹爪估计之间取开度并标记是否为原图直接观测。

    ``field_name`` 为 ``GripperSample`` 中的开度字段名。``interpolation="linear"`` 按时间线性插值；
    ``"nearest"`` 取前后有效样本中时间更近者（等距取较早），用于二值开度，不产生中间值。
    两种方式都要求目标两侧都有有效样本且间隔不超过 ``maximum_gap_ns``。
    """
    if interpolation not in ("linear", "nearest"):
        raise ValueError(f"未知的夹爪插值方式 {interpolation!r}")
    if not samples:
        return None
    nearest = _nearest_index(timestamps, target_ns)
    direct_tolerance_ns = 1_000_000
    if abs(timestamps[nearest] - target_ns) <= direct_tolerance_ns:
        sample = samples[nearest]
        value = getattr(sample, field_name)
        if sample.valid and np.isfinite(value):
            return float(value), True, 0.0

    valid_samples = [
        sample
        for sample in samples
        if sample.valid and np.isfinite(getattr(sample, field_name))
    ]
    if not valid_samples:
        return None
    valid_timestamps = [sample.timestamp_ns for sample in valid_samples]
    insertion = bisect_left(valid_timestamps, target_ns)
    if insertion == 0 or insertion >= len(valid_samples):
        return None
    first = valid_samples[insertion - 1]
    second = valid_samples[insertion]
    gap_ns = second.timestamp_ns - first.timestamp_ns
    if gap_ns <= 0 or gap_ns > maximum_gap_ns:
        return None
    if interpolation == "nearest":
        closer = first if target_ns - first.timestamp_ns <= second.timestamp_ns - target_ns else second
        return float(getattr(closer, field_name)), False, gap_ns / 1.0e6
    ratio = (target_ns - first.timestamp_ns) / gap_ns
    openness = (
        (1.0 - ratio) * getattr(first, field_name)
        + ratio * getattr(second, field_name)
    )
    return float(np.clip(openness, 0.0, 1.0)), False, gap_ns / 1.0e6


def _tracker_status_at(
    samples: Sequence[TrackerStatusSample],
    timestamps: Sequence[int],
    target_ns: int,
    maximum_delta_ns: int,
) -> Optional[bool]:
    """返回目标时刻最近 Tracker 状态是否可用于 6DoF 训练。"""
    if not samples:
        return None
    nearest = _nearest_index(timestamps, target_ns)
    if abs(timestamps[nearest] - target_ns) > maximum_delta_ns:
        return None
    sample = samples[nearest]
    # TrackerStatus.TRACKING_RUNNING_OK 的接口常量值为 3。
    return (
        sample.device_connected
        and sample.pose_valid
        and sample.tracking_state == 3
    )


def synchronize_link7_episode(
    image_timestamps_ns: Sequence[int],
    poses: Sequence[PoseSample],
    grippers: Sequence[GripperSample],
    tracker_statuses: Sequence[TrackerStatusSample],
    tracker_to_tcp: np.ndarray,
    nominal_source_fps: float,
    config: Link7SyncConfig = Link7SyncConfig(),
    tracker_time_offset_ns: int = 0,
    gripper_field: str = "filtered_openness",
    gripper_interpolation: str = "linear",
) -> Link7SyncResult:
    """同步一条 episode 并转换到起始 TCP 相对坐标系。

    Args:
        image_timestamps_ns: episode 范围内按解码顺序排列的源图像时间。
        poses: 世界系 Tracker 位姿。
        grippers: 夹爪估计（带源图像时间）。
        tracker_statuses: Tracker 跟踪状态，新格式必须存在。
        tracker_to_tcp: ``T_Tracker_TCP``。
        nominal_source_fps: 标称源帧率，决定选帧路径。
        config: 同步与质量门限。
        tracker_time_offset_ns: Tracker 查询时间相对图像时间的偏移。
        gripper_field: ``GripperSample`` 中的开度字段名。
        gripper_interpolation: ``linear`` 线性插值，``nearest`` 最近邻（二值开度）。
    """
    poses = sorted(poses, key=lambda sample: sample.timestamp_ns)
    grippers = sorted(grippers, key=lambda sample: sample.timestamp_ns)
    statuses = sorted(tracker_statuses, key=lambda sample: sample.timestamp_ns)
    missing = []
    if len(poses) < 2:
        missing.append("episode 内 Tracker 位姿不足两帧")
    if not grippers:
        missing.append("episode 内没有夹爪状态")
    if not statuses:
        missing.append("episode 内没有 Tracker 状态消息")
    if missing:
        return Link7SyncResult(
            None, [issue(MISSING_STREAM, text) for text in missing], [], {}
        )

    selection = select_frames(
        image_timestamps_ns, nominal_source_fps, config.selection
    )
    stats = dict(selection.stats)
    warnings = list(selection.warnings)
    if not selection.accepted:
        return Link7SyncResult(None, list(selection.rejections), warnings, stats)

    pose_times = [sample.timestamp_ns for sample in poses]
    gripper_times = [sample.timestamp_ns for sample in grippers]
    status_times = [sample.timestamp_ns for sample in statuses]
    maximum_pose_gap_ns = int(round(config.max_pose_gap_s * 1.0e9))
    maximum_gripper_gap_ns = int(round(config.max_gripper_gap_s * 1.0e9))
    records = []
    valid = np.zeros(len(selection), dtype=bool)
    for index, image_ns in enumerate(selection.image_timestamps_ns.tolist()):
        query_ns = int(image_ns) + int(tracker_time_offset_ns)
        pose = _interpolate_pose_sample(poses, pose_times, query_ns, maximum_pose_gap_ns)
        gripper = _interpolate_gripper_sample(
            grippers, gripper_times, int(image_ns), maximum_gripper_gap_ns,
            gripper_field, gripper_interpolation,
        )
        tracking = _tracker_status_at(statuses, status_times, query_ns, maximum_pose_gap_ns)
        records.append((pose, gripper, tracking))
        valid[index] = pose is not None and gripper is not None and tracking is True

    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        tracking_lost = any(record[2] is False for record in records)
        return Link7SyncResult(
            None,
            [
                issue(TRACKING_LOST, "Tracker 状态报告跟踪丢失或位姿无效")
                if tracking_lost
                else issue(SYNC_GAP, "所选图像时刻没有可同步的位姿和夹爪状态")
            ],
            warnings,
            stats,
        )
    first, last = int(valid_indices[0]), int(valid_indices[-1])
    inner = valid[first:last + 1]
    if not inner.all():
        rejections = []
        bad = np.flatnonzero(~inner) + first
        if any(records[index][2] is not True for index in bad):
            rejections.append(
                issue(
                    TRACKING_LOST,
                    "同步区间内部存在 Tracker 跟踪丢失或状态缺失",
                    first_position=int(bad[0]),
                )
            )
        if any(records[index][0] is None or records[index][1] is None for index in bad):
            rejections.append(
                issue(
                    SYNC_GAP,
                    "同步区间内部存在超过门限的 Tracker 或夹爪数据缺口",
                    first_position=int(bad[0]),
                )
            )
        return Link7SyncResult(None, rejections, warnings, stats)
    trimmed = first + (len(valid) - 1 - last)
    if trimmed:
        warnings.append(
            issue(
                HEAD_TAIL_TRIMMED,
                f"首尾缺少插值支撑，裁剪 {first} + {len(valid) - 1 - last} 帧",
                head=first,
                tail=int(len(valid) - 1 - last),
            )
        )
    kept = selection.subset(first, last + 1)
    kept_records = records[first:last + 1]
    if len(kept) < MINIMUM_LINK7_FRAMES:
        return Link7SyncResult(
            None,
            [issue(TOO_FEW_FRAMES, f"有效帧数 {len(kept)} 少于 {MINIMUM_LINK7_FRAMES}")],
            warnings,
            stats,
        )

    positions = np.asarray([record[0][0] for record in kept_records], dtype=np.float64)
    quaternions = np.asarray([record[0][1] for record in kept_records], dtype=np.float64)
    relative_positions, relative_quaternions = transform_series_to_episode_frame(
        positions, quaternions, tracker_to_tcp
    )
    openness = np.asarray([record[1][0] for record in kept_records], dtype=np.float64)
    qpos = np.concatenate(
        (relative_positions, relative_quaternions, openness.reshape(-1, 1)), axis=1
    )
    if not np.all(np.isfinite(qpos)):
        return Link7SyncResult(
            None, [issue(NONFINITE_RESULT, "同步结果包含 NaN 或 Inf")], warnings, stats
        )
    dropped = kept.dropped_frames_before
    actual_span_ns = int(kept.image_timestamps_ns[-1] - kept.image_timestamps_ns[0])
    nominal_span_ns = int(kept.nominal_timestamp_ns[-1])
    stats.update(
        {
            "output_frames": int(len(kept)),
            "trimmed_head": first,
            "trimmed_tail": int(len(valid) - 1 - last),
            "output_dropped_frame_total": int(dropped.sum()),
            "nominal_minus_actual_end_ms": (nominal_span_ns - actual_span_ns) / 1.0e6,
            "training_windows_crossing_drops": windows_crossing_drops(
                dropped, TRAINING_WINDOW_FRAMES
            ),
            "output_frequency_hz": TARGET_FREQUENCY_HZ,
        }
    )
    position_steps = np.linalg.norm(np.diff(relative_positions, axis=0), axis=1)
    if len(position_steps):
        stats["position_step_m"] = {
            "max": float(position_steps.max()),
            "mean": float(position_steps.mean()),
        }
    episode = Link7Episode(
        selection=kept,
        qpos=qpos,
        gripper_observed=np.asarray([record[1][1] for record in kept_records], dtype=bool),
        tracker_tracking_ok=np.ones(len(kept), dtype=bool),
        pose_gap_ms=np.asarray([record[0][2] for record in kept_records], dtype=np.float32),
        gripper_gap_ms=np.asarray([record[1][2] for record in kept_records], dtype=np.float32),
        world_from_start_tcp=pose_to_matrix(positions[0], quaternions[0]) @ tracker_to_tcp,
    )
    return Link7SyncResult(episode, [], warnings, stats)
