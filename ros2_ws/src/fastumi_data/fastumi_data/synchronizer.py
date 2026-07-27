"""按鱼眼图像时钟同步 Tracker 位姿和夹爪状态并生成相对 TCP 轨迹。"""

from bisect import bisect_left
from typing import List, Optional, Sequence, Tuple

import numpy as np

from fastumi_data.models import (
    EpisodeBuffer,
    GripperSample,
    PoseSample,
    ProcessingConfig,
    ProcessingResult,
    SynchronizedEpisode,
    TrackerStatusSample,
)
from fastumi_data.pose_math import (
    interpolate_pose,
    transform_series_to_episode_frame,
)


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


def _select_image_indices(
    image_timestamps: Sequence[int],
    start_ns: int,
    stop_ns: int,
    config: ProcessingConfig,
) -> List[Optional[int]]:
    """在均匀采样网格上选择最近图像，并保留未匹配的网格槽位。"""
    period_ns = int(round(1.0e9 / config.sample_rate_hz))
    maximum_delta_ns = int(round(config.max_image_delta_s * 1.0e9))
    selected = []
    previous_index: Optional[int] = None
    for grid_time in range(start_ns, stop_ns, period_ns):
        image_index = _nearest_index(image_timestamps, grid_time)
        if abs(image_timestamps[image_index] - grid_time) > maximum_delta_ns:
            selected.append(None)
            previous_index = None
            continue
        if image_index == previous_index:
            selected.append(None)
            continue
        selected.append(image_index)
        previous_index = image_index
    return selected


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
) -> Optional[Tuple[float, bool, float]]:
    """在有效夹爪估计之间插值开度并标记是否为原图直接观测。"""
    if not samples:
        return None
    nearest = _nearest_index(timestamps, target_ns)
    direct_tolerance_ns = 1_000_000
    if abs(timestamps[nearest] - target_ns) <= direct_tolerance_ns:
        sample = samples[nearest]
        if sample.valid and np.isfinite(sample.filtered_openness):
            return float(sample.filtered_openness), True, 0.0

    valid_samples = [
        sample
        for sample in samples
        if sample.valid and np.isfinite(sample.filtered_openness)
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
    ratio = (target_ns - first.timestamp_ns) / gap_ns
    openness = (
        (1.0 - ratio) * first.filtered_openness
        + ratio * second.filtered_openness
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


def synchronize_episode(
    buffer: EpisodeBuffer,
    stop_timestamp_ns: int,
    tracker_to_tcp: np.ndarray,
    config: ProcessingConfig,
) -> ProcessingResult:
    """同步单条 episode 并转换到起始 TCP 相对坐标系。

    首尾缺失样本会被裁剪。同步区间内部出现超过门限的位姿或夹爪缺口时，
    整条 episode 会被拒绝，防止训练数据包含隐式长时间保持值。
    """
    rejection_reasons: List[str] = []
    warnings: List[str] = []
    images = sorted(buffer.images, key=lambda sample: sample.timestamp_ns)
    poses = sorted(buffer.poses, key=lambda sample: sample.timestamp_ns)
    grippers = sorted(buffer.grippers, key=lambda sample: sample.timestamp_ns)
    tracker_statuses = sorted(
        buffer.tracker_statuses, key=lambda sample: sample.timestamp_ns
    )
    if not images:
        return ProcessingResult(None, ["episode 内没有鱼眼图像"], warnings)
    if len(poses) < 2:
        return ProcessingResult(None, ["episode 内有效 Vive 位姿不足两帧"], warnings)
    if not grippers:
        return ProcessingResult(None, ["episode 内没有夹爪状态"], warnings)
    if config.require_tracker_status and not tracker_statuses:
        return ProcessingResult(
            None, ["episode 内没有 Vive TrackerStatus"], warnings
        )

    image_timestamps = [sample.timestamp_ns for sample in images]
    pose_timestamps = [sample.timestamp_ns for sample in poses]
    gripper_timestamps = [sample.timestamp_ns for sample in grippers]
    tracker_status_timestamps = [
        sample.timestamp_ns for sample in tracker_statuses
    ]
    selected_indices = _select_image_indices(
        image_timestamps,
        buffer.start_event.timestamp_ns,
        stop_timestamp_ns,
        config,
    )
    if not selected_indices:
        return ProcessingResult(None, ["20 Hz 时间网格未匹配到图像"], warnings)

    maximum_pose_gap_ns = int(round(config.max_pose_gap_s * 1.0e9))
    maximum_gripper_gap_ns = int(round(config.max_gripper_gap_s * 1.0e9))
    synchronized_records = []
    valid_mask = []
    for image_index in selected_indices:
        if image_index is None:
            valid_mask.append(False)
            synchronized_records.append((None, None, None, None))
            continue
        image_sample = images[image_index]
        pose_result = _interpolate_pose_sample(
            poses,
            pose_timestamps,
            image_sample.timestamp_ns,
            maximum_pose_gap_ns,
        )
        gripper_result = _interpolate_gripper_sample(
            grippers,
            gripper_timestamps,
            image_sample.timestamp_ns,
            maximum_gripper_gap_ns,
        )
        tracker_tracking_ok = _tracker_status_at(
            tracker_statuses,
            tracker_status_timestamps,
            image_sample.timestamp_ns,
            maximum_pose_gap_ns,
        )
        if not config.require_tracker_status and tracker_tracking_ok is None:
            tracker_tracking_ok = True
        valid = (
            pose_result is not None
            and gripper_result is not None
            and tracker_tracking_ok is True
        )
        valid_mask.append(valid)
        synchronized_records.append(
            (
                image_sample,
                pose_result,
                gripper_result,
                tracker_tracking_ok,
            )
        )

    valid_indices = [index for index, valid in enumerate(valid_mask) if valid]
    if not valid_indices:
        tracker_loss = any(
            record[3] is False for record in synchronized_records
        )
        return ProcessingResult(
            None,
            [
                (
                    "Vive 状态报告跟踪丢失或位姿无效"
                    if tracker_loss
                    else "图像时间戳没有可同步的位姿和夹爪状态"
                )
            ],
            warnings,
        )
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]
    if not all(valid_mask[first_valid:last_valid + 1]):
        if any(
            record[3] is not True
            for record in synchronized_records[
                first_valid:last_valid + 1
            ]
        ):
            rejection_reasons.append(
                "同步区间内部存在 Vive 跟踪丢失或状态缺失"
            )
        rejection_reasons.append(
            "同步区间内部存在超过门限的图像、Vive 或夹爪数据缺口"
        )
        return ProcessingResult(None, rejection_reasons, warnings)
    if first_valid > 0 or last_valid < len(valid_mask) - 1:
        warnings.append(
            f"裁剪首尾 {first_valid + len(valid_mask) - last_valid - 1} 帧"
        )
    records = synchronized_records[first_valid:last_valid + 1]
    if len(records) < config.minimum_samples:
        return ProcessingResult(
            None,
            [f"有效样本数 {len(records)} 少于 {config.minimum_samples}"],
            warnings,
        )

    positions = np.asarray(
        [record[1][0] for record in records], dtype=np.float64
    )
    quaternions = np.asarray(
        [record[1][1] for record in records], dtype=np.float64
    )
    relative_positions, relative_quaternions = (
        transform_series_to_episode_frame(
            positions, quaternions, tracker_to_tcp
        )
    )
    openness = np.asarray(
        [record[2][0] for record in records], dtype=np.float64
    )
    qpos = np.concatenate(
        (
            relative_positions,
            relative_quaternions,
            openness.reshape(-1, 1),
        ),
        axis=1,
    ).astype(np.float32)
    synchronized = SynchronizedEpisode(
        timestamp_ns=np.asarray(
            [record[0].timestamp_ns for record in records], dtype=np.int64
        ),
        images_rgb=np.stack(
            [record[0].image_rgb for record in records]
        ).astype(np.uint8),
        qpos=qpos,
        gripper_observed=np.asarray(
            [record[2][1] for record in records], dtype=np.bool_
        ),
        tracker_tracking_ok=np.asarray(
            [record[3] is True for record in records], dtype=np.bool_
        ),
        pose_gap_ms=np.asarray(
            [record[1][2] for record in records], dtype=np.float32
        ),
        gripper_gap_ms=np.asarray(
            [record[2][2] for record in records], dtype=np.float32
        ),
    )
    if not np.all(np.isfinite(synchronized.qpos)):
        return ProcessingResult(None, ["同步结果包含 NaN 或 Inf"], warnings)
    return ProcessingResult(synchronized, rejection_reasons, warnings)
