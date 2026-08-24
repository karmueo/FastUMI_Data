"""验证 AprilGrid 原始鱼眼角点投影和单帧 IPPE 位姿估计。"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastumi_data.pose_math import pose_to_matrix
from fastumi_data.tracker_camera_config import (
    AprilGridSpec,
    CheckerboardSpec,
    FisheyeCameraModel,
    checkerboard_object_points,
    tag_object_corners,
)
from fastumi_data.tracker_camera_detection import (
    AprilGridObservation,
    CheckerboardObservation,
)
from fastumi_data.tracker_camera_pnp import (
    PoseEstimationError,
    estimate_camera_from_board,
    fisheye_reprojection_errors,
    project_fisheye_points,
)


def make_project_camera() -> FisheyeCameraModel:
    """返回覆盖 6×6 标定板的合成鱼眼相机。"""
    return FisheyeCameraModel(
        k=np.asarray(
            [[397.0, 0.0, 640.0], [0.0, 397.0, 640.0], [0.0, 0.0, 1.0]]
        ),
        d=np.asarray([0.08, -0.018, 0.004, -0.003]),
        resolution=(1280, 1280),
    )


def make_fixture() -> tuple[
    FisheyeCameraModel, np.ndarray, np.ndarray, np.ndarray
]:
    """返回相机、板角点、真值变换和无噪声像素。"""
    camera = make_project_camera()
    spec = AprilGridSpec(6, 6, 0.055, 0.3, "tag36h11")
    object_points = np.vstack(
        [tag_object_corners(spec, tag_id) for tag_id in range(36)]
    )
    expected = pose_to_matrix(
        np.array([-0.18, -0.18, 0.65]),
        Rotation.from_euler(
            "xyz", [8.0, -12.0, 4.0], degrees=True
        ).as_quat(),
    )
    image_points = project_fisheye_points(object_points, expected, camera)
    return camera, object_points, expected, image_points


def transform_error(
    expected: np.ndarray, actual: np.ndarray
) -> tuple[float, float]:
    """返回两个变换之间的平移毫米和旋转角度误差。"""
    relative = np.linalg.inv(expected) @ actual
    translation_mm = float(np.linalg.norm(relative[:3, 3]) * 1000.0)
    rotation_deg = float(
        np.rad2deg(
            np.linalg.norm(
                Rotation.from_matrix(relative[:3, :3]).as_rotvec()
            )
        )
    )
    return translation_mm, rotation_deg


def make_observation(
    image_points: np.ndarray, object_points: np.ndarray
) -> AprilGridObservation:
    """把完整 6×6 角点组成单帧观测。"""
    return AprilGridObservation(
        timestamp_ns=0,
        image_points_px=image_points,
        object_points_m=object_points,
        tag_ids=tuple(range(36)),
        tag_count=36,
    )


def test_fisheye_ippe_recovers_known_board_pose() -> None:
    """无噪声鱼眼角点应恢复正确的 ``^camera T_board``。"""
    camera, object_points, expected, image_points = make_fixture()
    result = estimate_camera_from_board(
        make_observation(image_points, object_points), camera
    )
    translation_mm, rotation_deg = transform_error(
        expected, result.camera_from_board
    )
    assert translation_mm < 0.1
    assert rotation_deg < 0.05
    assert result.p95_error_px < 0.05
    assert result.positive_depth is True


def test_fisheye_pnp_remains_accurate_with_pixel_noise() -> None:
    """0.3 px 高斯噪声下应保持毫米级位置和亚度姿态精度。"""
    camera, object_points, expected, image_points = make_fixture()
    generator = np.random.default_rng(20260804)
    noisy_points = image_points + generator.normal(
        0.0, 0.3, image_points.shape
    )
    result = estimate_camera_from_board(
        make_observation(noisy_points, object_points), camera
    )
    translation_mm, rotation_deg = transform_error(
        expected, result.camera_from_board
    )
    assert translation_mm < 2.0
    assert rotation_deg < 0.3
    assert result.p95_error_px < 1.0


def test_fisheye_ippe_accepts_tags_from_one_grid_row() -> None:
    """同一行标签的四角仍张成平面，应产生有效 IPPE 位姿。"""
    camera = make_project_camera()
    spec = AprilGridSpec(6, 6, 0.055, 0.3, "tag36h11")
    tag_ids = tuple(range(4))
    object_points = np.vstack([
        tag_object_corners(spec, tag_id) for tag_id in tag_ids
    ])
    expected = pose_to_matrix(
        np.array([-0.12, -0.04, 0.65]),
        Rotation.from_euler(
            "xyz", [8.0, -12.0, 4.0], degrees=True
        ).as_quat(),
    )
    image_points = project_fisheye_points(object_points, expected, camera)
    observation = AprilGridObservation(
        0, image_points, object_points, tag_ids, len(tag_ids)
    )

    result = estimate_camera_from_board(observation, camera)

    translation_mm, rotation_deg = transform_error(
        expected, result.camera_from_board
    )
    assert translation_mm < 0.1
    assert rotation_deg < 0.05


def test_reprojection_errors_are_euclidean_pixel_distances() -> None:
    """重投影误差应逐角点返回二维像素欧氏距离。"""
    camera, object_points, expected, image_points = make_fixture()
    shifted = image_points.copy()
    shifted[:, 0] += 3.0
    shifted[:, 1] += 4.0
    errors = fisheye_reprojection_errors(
        object_points, shifted, expected, camera
    )
    np.testing.assert_allclose(errors, 5.0, atol=1.0e-10)


def test_pose_estimation_rejects_nonfinite_or_too_few_points() -> None:
    """非法角点或少于四个特征点不能进入整板 IPPE。"""
    camera, object_points, _, image_points = make_fixture()
    invalid = image_points.copy()
    invalid[0, 0] = np.nan
    invalid_observation = SimpleNamespace(
        image_points_px=invalid,
        object_points_m=object_points,
    )
    with pytest.raises(PoseEstimationError, match="有限"):
        estimate_camera_from_board(invalid_observation, camera)
    small_observation = SimpleNamespace(
        image_points_px=image_points[:3],
        object_points_m=object_points[:3],
    )
    with pytest.raises(PoseEstimationError, match="至少 4"):
        estimate_camera_from_board(small_observation, camera)


def test_pose_estimation_rejects_collinear_object_points() -> None:
    """IPPE 前应明确拒绝四个或更多共线目标点。"""
    camera = make_project_camera()
    object_points = np.asarray([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.2, 0.0, 0.0],
        [0.3, 0.0, 0.0],
    ])
    observation = SimpleNamespace(
        object_points_m=object_points,
        image_points_px=np.asarray([
            [500.0, 600.0],
            [550.0, 600.0],
            [600.0, 600.0],
            [650.0, 600.0],
        ]),
    )
    with pytest.raises(PoseEstimationError, match="不能共线"):
        estimate_camera_from_board(observation, camera)


def test_fisheye_ippe_accepts_checkerboard_observation() -> None:
    """通用鱼眼 IPPE 应直接接受 11×8 棋盘格的 Nx2/Nx3 观测。"""
    camera = make_project_camera()
    spec = CheckerboardSpec(11, 8, 0.03, 0.03)
    object_points = checkerboard_object_points(spec)
    expected = pose_to_matrix(
        np.array([-0.14, -0.12, 0.72]),
        Rotation.from_euler("xyz", [6.0, -9.0, 3.0], degrees=True).as_quat(),
    )
    image_points = project_fisheye_points(object_points, expected, camera)
    observation = CheckerboardObservation(0, image_points, object_points)
    result = estimate_camera_from_board(observation, camera)
    translation_mm, rotation_deg = transform_error(
        expected, result.camera_from_board
    )
    assert observation.feature_count == 88
    assert translation_mm < 0.1
    assert rotation_deg < 0.05
    assert result.p95_error_px < 0.05
