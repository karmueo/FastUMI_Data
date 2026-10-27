"""验证 Link7 同步的时间基准、偏移方向、裁剪、拒绝条件与相对位姿不变性。"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastumi_data.frame_selection import FRAME_DROP, FRAME_INTERVAL_ANOMALY
from fastumi_data.link7_sync import (
    HEAD_TAIL_TRIMMED,
    SYNC_GAP,
    TOO_FEW_FRAMES,
    TRACKING_LOST,
    Link7SyncConfig,
    synchronize_link7_episode,
)
from fastumi_data.models import GripperSample, PoseSample, TrackerStatusSample
from fastumi_data.pose_math import matrix_to_pose, pose_to_matrix


START = 1_000_000_000_000


def images(count=60, rate=30.0, start=START) -> np.ndarray:
    """生成图像时间戳。"""
    return start + np.round(np.arange(count) * 1e9 / rate).astype(np.int64)


def tracker_stream(position_fn, rotation_fn=lambda t: [0, 0, 0], start=START - 200_000_000,
                   stop=None, rate=100.0):
    """以 100 Hz 生成世界系 Tracker 位姿与有效状态。"""
    stop = stop or START + 3_000_000_000
    times = np.arange(start, stop, int(1e9 / rate), dtype=np.int64)
    poses, statuses = [], []
    for stamp in times:
        seconds = (stamp - START) / 1e9
        poses.append(PoseSample(int(stamp), np.asarray(position_fn(seconds), dtype=float),
                                Rotation.from_rotvec(rotation_fn(seconds)).as_quat()))
        statuses.append(TrackerStatusSample(int(stamp), True, True, 3))
    return poses, statuses


def gripper_stream(image_times, value_fn=lambda t: 0.5):
    """每张图像对应一个有效夹爪估计。"""
    return [GripperSample(int(t), 0.0, float(value_fn((t - START) / 1e9)), 2, True)
            for t in image_times]


def run(image_times, poses, grippers, statuses, extrinsic=np.eye(4), offset_ns=0, fps=30):
    """调用同步。"""
    return synchronize_link7_episode(
        image_times, poses, grippers, statuses, extrinsic, fps, Link7SyncConfig(),
        tracker_time_offset_ns=offset_ns,
    )


def test_constant_speed_gives_ten_millimetre_steps_at_ideal_thirty_hz() -> None:
    """0.3 m/s 匀速轨迹在理想 30 Hz 下动作步长为 10 mm。"""
    times = images()
    poses, statuses = tracker_stream(lambda t: [0.3 * t, 0, 0])
    result = run(times, poses, gripper_stream(times), statuses)
    assert result.episode is not None, result.rejections
    steps = np.linalg.norm(np.diff(result.episode.qpos[:, :3], axis=0), axis=1)
    np.testing.assert_allclose(steps, 0.010, atol=1e-9)


def test_jittered_times_give_steps_matching_actual_intervals() -> None:
    """抖动时步长与实际时间间隔一致，名义时间仍为 n/30。"""
    rng = np.random.default_rng(1)
    times = images() + rng.integers(-3_000_000, 3_000_000, size=60)
    poses, statuses = tracker_stream(lambda t: [0.3 * t, 0, 0])
    result = run(times, poses, gripper_stream(times), statuses)
    episode = result.episode
    assert episode is not None
    steps = np.diff(episode.qpos[:, 0])
    actual = np.diff(episode.selection.image_timestamps_ns) / 1e9
    np.testing.assert_allclose(steps, 0.3 * actual, atol=1e-9)
    np.testing.assert_allclose(np.diff(episode.selection.nominal_timestamp_ns), 1e9 / 30, atol=1)


def test_pose_uses_image_time_plus_offset_and_gripper_uses_image_time() -> None:
    """位姿在 t_image + offset 查询（offset = −32.364 ms），夹爪在 t_image 插值。"""
    times = images()
    poses, statuses = tracker_stream(lambda t: [t, 0, 0])
    grippers = [GripperSample(int(t), 0.0, float((t - START) / 3e9), 2, True)
                for t in times[::2]]
    offset_ns = -32_364_096
    result = run(times, poses, grippers, statuses, offset_ns=offset_ns)
    episode = result.episode
    assert episode is not None
    t0 = (episode.selection.image_timestamps_ns[0] - START) / 1e9
    expected_x = (episode.selection.image_timestamps_ns - START) / 1e9 - t0
    np.testing.assert_allclose(episode.qpos[:, 0], expected_x, atol=1e-9)
    first_world_x = episode.world_from_start_tcp[0, 3]
    assert first_world_x == pytest.approx(t0 + offset_ns / 1e9, abs=1e-9)
    expected_gripper = (episode.selection.image_timestamps_ns - START) / 3e9
    np.testing.assert_allclose(episode.qpos[:, 7], expected_gripper, atol=1e-6)
    assert episode.gripper_observed[0] and not episode.gripper_observed[1]


def test_global_rigid_transform_does_not_change_relative_trajectory() -> None:
    """任意世界系刚体变换不改变 episode 起始 Link7 相对位姿与相对动作。"""
    times = images()
    position = lambda t: [0.2 * t, 0.1 * np.sin(t), 0.05 * t * t]
    rotation = lambda t: [0.3 * t, -0.2 * t, 0.1]
    extrinsic = pose_to_matrix(np.array([0.01, 0.02, -0.03]),
                               Rotation.from_rotvec([0.2, 0.1, -0.3]).as_quat())
    poses, statuses = tracker_stream(position, rotation)
    base = run(times, poses, gripper_stream(times), statuses, extrinsic).episode
    world = pose_to_matrix(np.array([3.0, -1.0, 2.0]), Rotation.from_rotvec([1.0, -0.5, 0.7]).as_quat())
    moved = []
    for sample in poses:
        p, q = matrix_to_pose(world @ pose_to_matrix(sample.position_m, sample.quaternion_xyzw))
        moved.append(PoseSample(sample.timestamp_ns, p, q))
    shifted = run(times, moved, gripper_stream(times), statuses, extrinsic).episode
    np.testing.assert_allclose(base.qpos[:, :3], shifted.qpos[:, :3], atol=1e-9)
    dots = np.abs(np.sum(base.qpos[:, 3:7] * shifted.qpos[:, 3:7], axis=1))
    np.testing.assert_allclose(dots, 1.0, atol=1e-9)


def test_head_and_tail_without_support_are_trimmed() -> None:
    """首尾缺少位姿支撑的图像被裁剪并给出提示。"""
    times = images(90)
    poses, statuses = tracker_stream(lambda t: [t, 0, 0], start=START + 300_000_000,
                                     stop=START + 2_500_000_000)
    result = run(times, poses, gripper_stream(times), statuses)
    assert result.episode is not None
    assert HEAD_TAIL_TRIMMED in {item["code"] for item in result.warnings}
    assert result.episode.selection.nominal_timestamp_ns[0] == 0
    assert result.stats["trimmed_head"] > 0 and result.stats["trimmed_tail"] > 0


def test_internal_pose_gap_rejects_episode() -> None:
    """内部位姿缺口超过门限时拒绝，不靠裁剪隐藏。"""
    times = images()
    poses, statuses = tracker_stream(lambda t: [t, 0, 0])
    poses = [pose for pose in poses if not START + 800_000_000 < pose.timestamp_ns < START + 1_000_000_000]
    result = run(times, poses, gripper_stream(times), statuses)
    assert result.episode is None
    assert SYNC_GAP in {item["code"] for item in result.rejections}


def test_internal_tracking_loss_rejects_episode() -> None:
    """内部跟踪状态异常时拒绝整条 episode。"""
    times = images()
    poses, statuses = tracker_stream(lambda t: [t, 0, 0])
    statuses = [TrackerStatusSample(s.timestamp_ns, True, True,
                                    4 if START + 900_000_000 < s.timestamp_ns < START + 950_000_000 else 3)
                for s in statuses]
    result = run(times, poses, gripper_stream(times), statuses)
    assert result.episode is None
    assert TRACKING_LOST in {item["code"] for item in result.rejections}


def test_frame_drop_keeps_episode_and_marks_drop() -> None:
    """单次丢帧保留 episode，并在名义时间与丢帧计数中体现。"""
    times = np.delete(images(), [30])
    poses, statuses = tracker_stream(lambda t: [t, 0, 0])
    result = run(times, poses, gripper_stream(times), statuses)
    assert result.episode is not None
    assert FRAME_DROP in {item["code"] for item in result.warnings}
    assert int(result.episode.selection.dropped_frames_before.sum()) == 1
    assert result.stats["training_windows_crossing_drops"] > 0


def test_interval_anomaly_rejects_before_interpolation() -> None:
    """间隔异常直接拒绝。"""
    times = images()
    times[30:] += 15_000_000
    poses, statuses = tracker_stream(lambda t: [t, 0, 0])
    result = run(times, poses, gripper_stream(times), statuses)
    assert FRAME_INTERVAL_ANOMALY in {item["code"] for item in result.rejections}


def test_minimum_sixteen_frames_required() -> None:
    """最终少于 16 帧时拒绝。"""
    times = images(15)
    poses, statuses = tracker_stream(lambda t: [t, 0, 0])
    result = run(times, poses, gripper_stream(times), statuses)
    assert TOO_FEW_FRAMES in {item["code"] for item in result.rejections}


def test_missing_tracker_status_is_rejected() -> None:
    """新格式强制要求 Tracker 状态。"""
    times = images()
    poses, _ = tracker_stream(lambda t: [t, 0, 0])
    result = run(times, poses, gripper_stream(times), [])
    assert result.episode is None


def test_nearest_gripper_interpolation_keeps_binary_values() -> None:
    """二值开度用最近邻取值：无中间值，等距取较早样本。"""
    times = images()
    poses, statuses = tracker_stream(lambda t: [t, 0, 0])
    # 夹爪样本与图像错开半帧，且每隔一帧缺失，强制走插值分支。
    half = int(round(1e9 / 60))
    grippers = [GripperSample(int(t) + half, 0.0, 1.0 if k < 30 else 0.0, 2, True)
                for k, t in enumerate(times) if k % 2 == 0]
    linear = run(times, poses, grippers, statuses).episode
    nearest = synchronize_link7_episode(
        times, poses, grippers, statuses, np.eye(4), 30, Link7SyncConfig(),
        gripper_interpolation="nearest",
    ).episode
    assert set(np.unique(nearest.qpos[:, 7])) == {0.0, 1.0}
    assert not np.isin(linear.qpos[:, 7], [0.0, 1.0]).all()
    assert not nearest.gripper_observed.any()


def test_nearest_gripper_tie_picks_earlier_sample() -> None:
    """目标时刻恰在两样本正中时取较早的样本。"""
    from fastumi_data.link7_sync import _interpolate_gripper_sample

    samples = [GripperSample(0, 0.0, 1.0, 2, True), GripperSample(100_000_000, 0.0, 0.0, 2, True)]
    stamps = [sample.timestamp_ns for sample in samples]
    value, observed, _ = _interpolate_gripper_sample(
        samples, stamps, 50_000_000, 200_000_000, interpolation="nearest")
    assert value == 1.0 and not observed
    assert _interpolate_gripper_sample(
        samples, stamps, 60_000_000, 200_000_000, interpolation="nearest")[0] == 0.0
    assert _interpolate_gripper_sample(
        samples, stamps, 50_000_000, 90_000_000, interpolation="nearest") is None


def test_gripper_field_selects_raw_openness() -> None:
    """gripper_field 选择 raw_openness 时忽略 filtered_openness。"""
    times = images()
    poses, statuses = tracker_stream(lambda t: [t, 0, 0])
    grippers = [GripperSample(int(t), 0.25, 0.75, 2, True) for t in times]
    episode = synchronize_link7_episode(
        times, poses, grippers, statuses, np.eye(4), 30, Link7SyncConfig(),
        gripper_field="raw_openness",
    ).episode
    np.testing.assert_allclose(episode.qpos[:, 7], 0.25)
