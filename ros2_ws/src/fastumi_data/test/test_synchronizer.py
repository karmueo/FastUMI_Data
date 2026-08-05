"""测试图像主时钟同步、TCP 相对轨迹和长缺口拒绝。"""

import numpy as np
import pytest

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


def _make_buffer(
    remove_pose_gap: bool = False,
    remove_image_gap: bool = False,
) -> EpisodeBuffer:
    """生成一秒、30 Hz 的合成 episode。"""
    start = EpisodeEventRecord(0, "session", "task", 0, 1)
    buffer = EpisodeBuffer(start_event=start)
    for index in range(31):
        timestamp_ns = int(round(index / 30.0 * 1.0e9))
        in_removed_image_gap = 0.40e9 < timestamp_ns < 0.60e9
        if not remove_image_gap or not in_removed_image_gap:
            buffer.images.append(
                ImageSample(
                    timestamp_ns,
                    np.full((8, 8, 3), index, dtype=np.uint8),
                )
            )
        in_removed_gap = 0.40e9 < timestamp_ns < 0.60e9
        if not remove_pose_gap or not in_removed_gap:
            buffer.poses.append(
                PoseSample(
                    timestamp_ns,
                    np.asarray([index / 300.0, 0.0, 0.0]),
                    np.asarray([0.0, 0.0, 0.0, 1.0]),
                )
            )
        buffer.grippers.append(
            GripperSample(
                timestamp_ns,
                index / 30.0,
                index / 30.0,
                2,
                True,
            )
        )
        buffer.tracker_statuses.append(
            TrackerStatusSample(timestamp_ns, True, True, 3)
        )
    return buffer


def _make_offset_buffer() -> EpisodeBuffer:
    """生成 20 Hz 图像与夹爪、100 Hz 二次轨迹的合成 episode。"""
    start = EpisodeEventRecord(0, "session", "task", 0, 1)
    buffer = EpisodeBuffer(start_event=start)
    for index in range(12):
        timestamp_ns = index * 50_000_000
        buffer.images.append(
            ImageSample(
                timestamp_ns,
                np.full((8, 8, 3), index, dtype=np.uint8),
            )
        )
        buffer.grippers.append(
            GripperSample(
                timestamp_ns,
                index / 10.0,
                index / 10.0,
                2,
                True,
            )
        )
    for index in range(62):
        timestamp_ns = index * 10_000_000
        timestamp_s = timestamp_ns / 1.0e9
        buffer.poses.append(
            PoseSample(
                timestamp_ns,
                np.asarray([timestamp_s**2, 0.0, 0.0]),
                np.asarray([0.0, 0.0, 0.0, 1.0]),
            )
        )
        buffer.tracker_statuses.append(
            TrackerStatusSample(timestamp_ns, True, True, 3)
        )
    return buffer


def test_synchronizes_to_twenty_hz_and_relative_tcp() -> None:
    """验证合成 30 Hz 数据生成约 20 Hz 的相对 TCP HDF5 内容。"""
    result = synchronize_episode(
        _make_buffer(),
        int(1.0e9),
        np.eye(4),
        ProcessingConfig(),
    )

    assert result.episode is not None
    assert result.rejection_reasons == []
    assert 19 <= result.episode.qpos.shape[0] <= 20
    assert result.episode.qpos.shape[1] == 8
    assert np.allclose(result.episode.qpos[0, :3], 0.0)
    assert np.allclose(
        np.linalg.norm(result.episode.qpos[:, 3:7], axis=1), 1.0
    )
    assert np.all(result.episode.gripper_observed)


def test_rejects_interior_pose_gap() -> None:
    """验证超过 0.1 秒的 Vive 内部缺口会拒绝整条 episode。"""
    result = synchronize_episode(
        _make_buffer(remove_pose_gap=True),
        int(1.0e9),
        np.eye(4),
        ProcessingConfig(),
    )

    assert result.episode is None
    assert any("内部" in reason for reason in result.rejection_reasons)


def test_rejects_interior_image_gap() -> None:
    """验证相机中间缺帧不会被静默压缩为固定频率数据。"""
    result = synchronize_episode(
        _make_buffer(remove_image_gap=True),
        int(1.0e9),
        np.eye(4),
        ProcessingConfig(),
    )

    assert result.episode is None
    assert any("内部" in reason for reason in result.rejection_reasons)


def test_rejects_reported_tracker_loss() -> None:
    """验证状态话题报告的短时跟踪丢失不会被位姿插值掩盖。"""
    buffer = _make_buffer()
    buffer.tracker_statuses[15] = TrackerStatusSample(
        buffer.tracker_statuses[15].timestamp_ns,
        True,
        False,
        4,
    )
    result = synchronize_episode(
        buffer,
        int(1.0e9),
        np.eye(4),
        ProcessingConfig(),
    )

    assert result.episode is None
    assert any("跟踪丢失" in reason for reason in result.rejection_reasons)


def test_queries_tracker_pose_at_offset_without_shifting_gripper_or_output_clock() -> None:
    """验证偏移只作用于 Tracker 位姿，输出和夹爪保持图像时钟。"""
    buffer = _make_offset_buffer()
    result = synchronize_episode(
        buffer,
        600_000_000,
        np.eye(4),
        ProcessingConfig(),
        tracker_time_offset_ns=10_000_000,
    )

    assert result.episode is not None
    assert result.episode.timestamp_ns[0] == 0
    assert result.episode.qpos[1, 0] == pytest.approx(0.0035, abs=1.0e-6)
    assert result.episode.qpos[1, 7] == pytest.approx(0.1, abs=1.0e-6)


def test_rejects_tracker_status_at_offset_boundary() -> None:
    """验证偏移后的最近 TrackerStatus 无效时会拒绝该 episode。"""
    buffer = _make_offset_buffer()
    invalid_status_index = 26
    invalid_timestamp_ns = invalid_status_index * 10_000_000
    buffer.tracker_statuses[invalid_status_index] = TrackerStatusSample(
        invalid_timestamp_ns,
        True,
        False,
        4,
    )

    result = synchronize_episode(
        buffer,
        600_000_000,
        np.eye(4),
        ProcessingConfig(),
        tracker_time_offset_ns=10_000_000,
    )

    assert result.episode is None
    assert any("跟踪丢失" in reason for reason in result.rejection_reasons)
