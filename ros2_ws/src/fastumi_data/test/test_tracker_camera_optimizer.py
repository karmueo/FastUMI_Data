"""验证时间块划分、偏移扫描和原始鱼眼角点联合优化。"""

from dataclasses import dataclass

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastumi_data.models import PoseSample, TrackerStatusSample
from fastumi_data.pose_math import pose_to_matrix
from fastumi_data.tracker_camera_bag import (
    TrackerTimeline,
    interpolate_world_from_tracker,
)
from fastumi_data.tracker_camera_config import (
    AprilGridSpec,
    FisheyeCameraModel,
    tag_object_corners,
)
from fastumi_data.tracker_camera_optimizer import (
    CalibrationSample,
    OptimizationOptions,
    _samples_valid_for_full_search,
    optimize_spatiotemporal,
    split_temporal_blocks,
)
from fastumi_data.tracker_camera_pnp import project_fisheye_points


@dataclass(frozen=True)
class SpatiotemporalFixture:
    """保存联合优化合成真值和观测。"""

    samples: list[CalibrationSample]
    timeline: TrackerTimeline
    camera: FisheyeCameraModel
    handeye_seed: np.ndarray
    board_seed: np.ndarray
    tracker_from_camera: np.ndarray
    world_from_board: np.ndarray


def make_transform(
    translation_m: list[float] | np.ndarray,
    euler_xyz_deg: list[float],
) -> np.ndarray:
    """从米制平移和 xyz 欧拉角生成刚体变换。"""
    return pose_to_matrix(
        np.asarray(translation_m, dtype=np.float64),
        Rotation.from_euler(
            "xyz", euler_xyz_deg, degrees=True
        ).as_quat(),
    )


def analytic_tracker_pose(time_s: float) -> np.ndarray:
    """返回具有平移加速度和三轴转动的连续 Tracker 轨迹。"""
    translation = [
        0.25 * np.sin(1.7 * time_s) + 0.025 * time_s,
        0.18 * np.cos(1.2 * time_s),
        0.75 + 0.12 * np.sin(2.1 * time_s),
    ]
    angles = [
        24.0 * np.sin(1.3 * time_s),
        20.0 * np.cos(1.8 * time_s),
        35.0 * np.sin(0.9 * time_s) + 6.0 * time_s,
    ]
    return make_transform(translation, angles)


def make_spatiotemporal_fixture(
    time_offset_ms: float,
    pixel_noise_sigma: float,
    outlier_fraction: float = 0.0,
) -> SpatiotemporalFixture:
    """生成固定板、已知外参及可插值 Tracker 时间线。"""
    camera = FisheyeCameraModel(
        np.asarray(
            [[397.0, 0.0, 640.0], [0.0, 397.0, 640.0], [0.0, 0.0, 1.0]]
        ),
        np.asarray([0.08, -0.018, 0.004, -0.003]),
        (1280, 1280),
    )
    tracker_from_camera = make_transform(
        [0.07, -0.025, 0.035], [12.0, -6.0, 18.0]
    )
    world_from_board = make_transform(
        [0.42, -0.16, 1.12], [4.0, 2.0, -8.0]
    )
    timeline_poses = []
    for timestamp_ns in range(0, 4_000_000_001, 2_000_000):
        transform = analytic_tracker_pose(timestamp_ns / 1.0e9)
        timeline_poses.append(
            PoseSample(
                timestamp_ns,
                transform[:3, 3],
                Rotation.from_matrix(transform[:3, :3]).as_quat(),
            )
        )
    timeline = TrackerTimeline(tuple(timeline_poses), ())
    spec = AprilGridSpec(6, 6, 0.055, 0.3, "tag36h11")
    object_points = np.vstack(
        [tag_object_corners(spec, tag_id) for tag_id in range(6)]
    )
    generator = np.random.default_rng(20260804)
    samples = []
    for timestamp_ns in np.linspace(
        400_000_000, 3_600_000_000, 36, dtype=np.int64
    ):
        interpolation = interpolate_world_from_tracker(
            timeline.poses,
            int(timestamp_ns + round(time_offset_ms * 1.0e6)),
            20.0,
        )
        assert interpolation is not None
        camera_from_board = (
            np.linalg.inv(tracker_from_camera)
            @ np.linalg.inv(interpolation[0])
            @ world_from_board
        )
        image_points = project_fisheye_points(
            object_points, camera_from_board, camera
        )
        image_points += generator.normal(
            0.0, pixel_noise_sigma, image_points.shape
        )
        samples.append(
            CalibrationSample(
                int(timestamp_ns),
                object_points,
                image_points,
                camera_from_board,
                6,
            )
        )
    if outlier_fraction > 0.0:
        all_locations = [
            (sample_index, point_index)
            for sample_index, sample in enumerate(samples)
            for point_index in range(len(sample.image_points_px))
        ]
        outlier_count = int(round(len(all_locations) * outlier_fraction))
        chosen = generator.choice(
            len(all_locations), outlier_count, replace=False
        )
        for location_index in chosen:
            sample_index, point_index = all_locations[location_index]
            sample = samples[sample_index]
            corrupted = sample.image_points_px.copy()
            corrupted[point_index] += generator.normal(0.0, 8.0, 2)
            samples[sample_index] = CalibrationSample(
                sample.timestamp_ns,
                sample.object_points_m,
                corrupted,
                sample.camera_from_board,
                sample.tag_count,
            )
    handeye_seed = tracker_from_camera @ make_transform(
        [0.003, -0.002, 0.001], [0.4, -0.3, 0.2]
    )
    board_seed = world_from_board @ make_transform(
        [0.004, 0.002, -0.003], [-0.3, 0.2, 0.4]
    )
    return SpatiotemporalFixture(
        samples,
        timeline,
        camera,
        handeye_seed,
        board_seed,
        tracker_from_camera,
        world_from_board,
    )


def transform_error(
    expected: np.ndarray, actual: np.ndarray
) -> tuple[float, float]:
    """返回相对平移毫米和旋转角度。"""
    relative = np.linalg.inv(expected) @ actual
    return (
        float(np.linalg.norm(relative[:3, 3]) * 1000.0),
        float(
            np.rad2deg(
                np.linalg.norm(
                    Rotation.from_matrix(relative[:3, :3]).as_rotvec()
                )
            )
        ),
    )


def test_full_search_rejects_status_failure_inside_offset_window() -> None:
    """偏移搜索窗口内出现无效状态时，该图像不能进入联合优化。"""
    fixture = make_spatiotemporal_fixture(18.0, 0.0)
    statuses = []
    rejected_timestamp_ns = fixture.samples[10].timestamp_ns
    for pose in fixture.timeline.poses:
        is_valid = abs(pose.timestamp_ns - rejected_timestamp_ns) > 40_000_000
        statuses.append(
            TrackerStatusSample(pose.timestamp_ns, True, is_valid, 3)
        )
    timeline = TrackerTimeline(fixture.timeline.poses, tuple(statuses))
    _, indices = _samples_valid_for_full_search(
        fixture.samples,
        timeline,
        OptimizationOptions(-50.0, 50.0, 2.0, 20.0, 0.2),
    )
    assert 10 not in indices


def test_temporal_split_keeps_neighboring_frames_in_same_partition() -> None:
    """同一两秒时间块不能同时进入训练和验证。"""
    timestamps = [int(index * 0.1e9) for index in range(100)]
    train, validation = split_temporal_blocks(timestamps, 2.0, 0.2)
    train_blocks = {
        timestamps[index] // 2_000_000_000 for index in train
    }
    validation_blocks = {
        timestamps[index] // 2_000_000_000 for index in validation
    }
    assert train_blocks.isdisjoint(validation_blocks)
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(100))


def test_joint_optimizer_recovers_extrinsic_and_time_offset() -> None:
    """合成鱼眼角点应恢复外参和 18 ms 时间偏移。"""
    fixture = make_spatiotemporal_fixture(18.0, 0.15)
    result = optimize_spatiotemporal(
        fixture.samples,
        fixture.timeline,
        fixture.camera,
        fixture.handeye_seed,
        fixture.board_seed,
        OptimizationOptions(-50.0, 50.0, 2.0, 20.0, 0.2),
    )
    translation_mm, rotation_deg = transform_error(
        fixture.tracker_from_camera, result.tracker_from_camera
    )
    assert translation_mm < 2.0
    assert rotation_deg < 0.2
    assert result.time_offset_ms == pytest.approx(18.0, abs=1.0)
    np.testing.assert_allclose(
        result.tracker_from_camera @ result.camera_from_tracker,
        np.eye(4),
        atol=1.0e-10,
    )
    assert set(result.train_indices).isdisjoint(result.validation_indices)


def test_optimizer_is_robust_to_five_percent_outlier_corners() -> None:
    """soft_l1 应限制 5% 大像素离群点对外参的影响。"""
    fixture = make_spatiotemporal_fixture(18.0, 0.15, 0.05)
    result = optimize_spatiotemporal(
        fixture.samples,
        fixture.timeline,
        fixture.camera,
        fixture.handeye_seed,
        fixture.board_seed,
        OptimizationOptions(-50.0, 50.0, 2.0, 20.0, 0.2),
    )
    translation_mm, rotation_deg = transform_error(
        fixture.tracker_from_camera, result.tracker_from_camera
    )
    assert translation_mm < 5.0
    assert rotation_deg < 0.5


def test_time_offset_outside_search_is_marked_at_boundary() -> None:
    """真实偏移超出搜索区间时结果必须标记时间变量触边。"""
    fixture = make_spatiotemporal_fixture(35.0, 0.1)
    result = optimize_spatiotemporal(
        fixture.samples,
        fixture.timeline,
        fixture.camera,
        fixture.handeye_seed,
        fixture.board_seed,
        OptimizationOptions(-20.0, 20.0, 2.0, 20.0, 0.2),
    )
    assert result.time_offset_ms == pytest.approx(20.0, abs=0.1)
    assert result.metrics["time_offset_at_boundary"] is True
