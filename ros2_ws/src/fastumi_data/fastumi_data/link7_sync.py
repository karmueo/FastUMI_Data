"""按所选图像的实际时间同步 Tracker 位姿与夹爪，生成 episode 起始 Link7 相对轨迹。

时间约定：图像 ``k`` 的实际消息时间为 ``t_image(k)``。Tracker 位姿与跟踪状态在
``t_image(k) + tracker_time_offset_ns`` 查询（偏移已按外参语义换算，见
:mod:`fastumi_data.link7_calibration`），位置线性插值、旋转 SLERP；夹爪在
``t_image(k)`` 插值。名义时间只用于标注，不参与插值。

几何约定：``T_W_Link7(t) = T_W_Tracker(t + offset) × T_Tracker_Link7``，再以
裁剪后首帧为参考 ``T_Link7_start_Link7(t) = inverse(T_W_Link7(t0)) × T_W_Link7(t)``。
``qpos`` 每行为 ``[x, y, z, qx, qy, qz, qw, gripper]``，位置单位米，夹爪 [0,1]。

首尾因缺少插值支撑可以裁剪；帧间隔异常、拟合帧率偏差、有效帧率不足、高帧率
匹配超限/复用，以及内部同步缺口或跟踪异常都拒绝整条 episode。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

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
from fastumi_data.pose_math import pose_to_matrix, transform_series_to_episode_frame
from fastumi_data.synchronizer import (
    _interpolate_gripper_sample,
    _interpolate_pose_sample,
    _tracker_status_at,
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
    """Link7 同步与质量门限。"""

    max_pose_gap_s: float = 0.10
    max_gripper_gap_s: float = 0.20
    minimum_samples: int = MINIMUM_LINK7_FRAMES
    selection: FrameSelectionConfig = field(default_factory=FrameSelectionConfig)


@dataclass
class Link7Episode:
    """完成同步和 Link7 相对变换的 episode；图像以位置索引引用源序列。"""

    selection: FrameSelection
    """裁剪后的选帧（名义时间、丢帧、匹配误差均已按首帧重算）。"""
    qpos: np.ndarray
    gripper_observed: np.ndarray
    tracker_tracking_ok: np.ndarray
    pose_gap_ms: np.ndarray
    gripper_gap_ms: np.ndarray
    world_from_start_link7: np.ndarray
    """裁剪后首帧的 ``T_W_Link7``，用于追溯。"""

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


def synchronize_link7_episode(
    image_timestamps_ns: Sequence[int],
    poses: Sequence[PoseSample],
    grippers: Sequence[GripperSample],
    tracker_statuses: Sequence[TrackerStatusSample],
    tracker_to_link7: np.ndarray,
    nominal_source_fps: float,
    config: Link7SyncConfig = Link7SyncConfig(),
    tracker_time_offset_ns: int = 0,
) -> Link7SyncResult:
    """同步一条 episode 并转换到起始 Link7 相对坐标系。

    Args:
        image_timestamps_ns: episode 范围内按解码顺序排列的源图像时间。
        poses: 世界系 Tracker 位姿。
        grippers: 夹爪估计（带源图像时间）。
        tracker_statuses: Tracker 跟踪状态，新格式必须存在。
        tracker_to_link7: ``T_Tracker_Link7``。
        nominal_source_fps: 标称源帧率，决定选帧路径。
        config: 同步与质量门限。
        tracker_time_offset_ns: Tracker 查询时间相对图像时间的偏移。
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
            grippers, gripper_times, int(image_ns), maximum_gripper_gap_ns
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
    minimum = max(int(config.minimum_samples), MINIMUM_LINK7_FRAMES)
    if len(kept) < minimum:
        return Link7SyncResult(
            None,
            [issue(TOO_FEW_FRAMES, f"有效帧数 {len(kept)} 少于 {minimum}")],
            warnings,
            stats,
        )

    positions = np.asarray([record[0][0] for record in kept_records], dtype=np.float64)
    quaternions = np.asarray([record[0][1] for record in kept_records], dtype=np.float64)
    relative_positions, relative_quaternions = transform_series_to_episode_frame(
        positions, quaternions, tracker_to_link7
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
        world_from_start_link7=pose_to_matrix(positions[0], quaternions[0]) @ tracker_to_link7,
    )
    return Link7SyncResult(episode, [], warnings, stats)
