"""验证双 ArUco 单帧质量门、稳健 SE(3) 聚合和变换方向。"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastumi_data.aruco_tcp_calibration import (
    CalibrationThresholds,
    calibrate_frames,
)
from fastumi_data.aruco_tcp_estimator import FrameTcpEstimate, TagPoseEstimate


def _tag(tag_id: int, center) -> TagPoseEstimate:
    """构造质量诊断所需的正深度标签位姿。"""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = center
    return TagPoseEstimate(tag_id, transform, 0.5)


def _frame(
    index: int,
    translation=(0.012, 0.0, 1.018),
    rotation=None,
    rmse=(0.5, 0.5),
    distance=0.126,
    candidate_difference=0.0,
) -> FrameTcpEstimate:
    """按指定相机→TCP 真值和质量指标构造单帧结果。"""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
    )
    transform[:3, 3] = translation
    pair = np.eye(4, dtype=np.float64)
    pair[:3, 3] = [0.0, 0.0, 1.0]
    return FrameTcpEstimate(
        timestamp_ns=index,
        camera_from_pair=pair,
        camera_from_tcp=transform,
        tcp_from_tag0_m=np.asarray(translation, dtype=np.float64),
        tcp_from_tag1_m=np.asarray(translation, dtype=np.float64),
        fused_tcp_position_m=np.asarray(translation, dtype=np.float64),
        measured_tag_distance_m=distance,
        expected_tag_distance_m=0.126,
        candidate_translation_difference_m=candidate_difference,
        tag0_reprojection_rmse_px=rmse[0],
        tag1_reprojection_rmse_px=rmse[1],
        tag0=_tag(0, [0.0, -0.063, 1.0]),
        tag1=_tag(1, [0.0, 0.063, 1.0]),
    )


def _rotation_error_deg(expected: np.ndarray, actual: np.ndarray) -> float:
    """计算两个刚体旋转之间的最短角度误差。"""
    relative = expected[:3, :3].T @ actual[:3, :3]
    return float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(relative).as_rotvec())))


def test_calibrate_frames_rejects_outliers_and_recovers_true_se3():
    """稳健聚合应剔除 30 mm/15° 离群并通过默认质量门。"""
    generator = np.random.default_rng(20260806)
    true_translation = np.asarray([0.012, -0.021, 0.318])
    true_rotation = Rotation.from_euler("xyz", [4.0, -7.0, 12.0], degrees=True).as_matrix()
    frames = []
    for index in range(37):
        translation = true_translation + generator.normal(0.0, 0.00025, 3)
        rotation = Rotation.from_rotvec(
            generator.normal(0.0, np.deg2rad(0.08), 3)
        ).as_matrix() @ true_rotation
        frames.append(_frame(index, translation, rotation))
    for index in range(37, 40):
        frames.append(
            _frame(
                index,
                true_translation + np.asarray([0.03, 0.0, 0.0]),
                Rotation.from_euler("z", 15.0, degrees=True).as_matrix()
                @ true_rotation,
            )
        )

    result = calibrate_frames(
        frames,
        np.eye(4),
        CalibrationThresholds(minimum_valid_frames=30),
    )

    assert result.accepted is True
    assert result.tracker_from_tcp is not None
    np.testing.assert_allclose(
        result.camera_from_tcp[:3, 3], true_translation, atol=0.001
    )
    assert _rotation_error_deg(
        np.block([[true_rotation, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]]),
        result.camera_from_tcp,
    ) < 1.0
    assert result.metrics["valid_frames"] >= 30
    assert result.metrics["rejected_frames"] >= 3
    assert result.metrics["translation_p95_mm"] <= 3.0
    assert result.metrics["rotation_p95_deg"] <= 2.0


def test_calibrate_frames_composes_tracker_from_camera_in_declared_direction():
    """最终外参必须严格按 Tracker←Camera←TCP 方向相乘。"""
    camera_from_tcp = _frame(0).camera_from_tcp
    source = np.eye(4, dtype=np.float64)
    source[:3, 3] = [0.4, -0.2, 0.1]
    source[:3, :3] = Rotation.from_euler("z", 20.0, degrees=True).as_matrix()
    frames = [_frame(index, camera_from_tcp[:3, 3], camera_from_tcp[:3, :3]) for index in range(30)]

    result = calibrate_frames(frames, source, CalibrationThresholds())

    assert result.accepted is True
    np.testing.assert_allclose(result.tracker_from_tcp, source @ camera_from_tcp)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("rmse", (1.6, 0.5), "重投影"),
        ("distance", 0.132, "距离"),
        ("candidate_difference", 0.006, "候选"),
    ],
)
def test_calibrate_frames_quality_gates_report_rejection(field, value, reason):
    """单帧 RMSE、模型距离和候选分歧超门限时应拒绝。"""
    frames = []
    for index in range(30):
        kwargs = {}
        if field == "rmse":
            kwargs["rmse"] = value
        elif field == "distance":
            kwargs["distance"] = value
        else:
            kwargs["candidate_difference"] = value
        frames.append(_frame(index, **kwargs))

    result = calibrate_frames(frames, np.eye(4), CalibrationThresholds())

    assert result.accepted is False
    assert any(reason in failure for failure in result.failures)
    assert result.tracker_from_tcp is None


def test_calibrate_frames_rejects_too_few_accepted_frames():
    """质量门通过的帧少于 30 帧时不能产生可转换外参。"""
    result = calibrate_frames(
        [_frame(index) for index in range(29)],
        np.eye(4),
        CalibrationThresholds(minimum_valid_frames=30),
    )

    assert result.accepted is False
    assert any("30" in failure for failure in result.failures)
