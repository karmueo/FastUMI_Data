"""验证 OpenCV 多算法 Hand-Eye 方向和固定板闭环评分。"""

import numpy as np
from scipy.spatial.transform import Rotation

from fastumi_data.pose_math import pose_to_matrix
from fastumi_data.tracker_camera_handeye import (
    board_closure_errors,
    board_transforms,
    select_handeye_seed,
    solve_handeye_candidates,
)


def make_transform(
    translation_m: list[float], euler_xyz_deg: list[float]
) -> np.ndarray:
    """从平移和 xyz 欧拉角生成测试刚体变换。"""
    return pose_to_matrix(
        np.asarray(translation_m, dtype=np.float64),
        Rotation.from_euler(
            "xyz", euler_xyz_deg, degrees=True
        ).as_quat(),
    )


def make_diverse_tracker_trajectory(count: int) -> list[np.ndarray]:
    """生成具有三轴转动和平移激励的 Tracker 轨迹。"""
    poses = []
    for index in range(count):
        phase = index / max(count - 1, 1)
        translation = [
            0.25 * np.sin(phase * 2.3 * np.pi),
            0.18 * np.cos(phase * 1.7 * np.pi),
            0.7 + 0.12 * np.sin(phase * 3.1 * np.pi),
        ]
        angles = [
            -35.0 + 80.0 * phase,
            28.0 * np.sin(phase * 2.1 * np.pi),
            -50.0 + 110.0 * phase,
        ]
        poses.append(make_transform(translation, angles))
    return poses


def transform_error(
    expected: np.ndarray, actual: np.ndarray
) -> tuple[float, float]:
    """返回两个变换的相对平移毫米和旋转角度。"""
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


def make_handeye_fixture() -> tuple[
    np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]
]:
    """生成满足固定板闭环方程的无噪声 Hand-Eye 数据。"""
    tracker_from_camera = make_transform(
        [0.07, -0.025, 0.035], [12.0, -6.0, 18.0]
    )
    world_from_board = make_transform(
        [0.4, -0.2, 1.1], [4.0, 2.0, -8.0]
    )
    world_from_tracker = make_diverse_tracker_trajectory(24)
    camera_from_board = [
        np.linalg.inv(tracker_from_camera)
        @ np.linalg.inv(pose)
        @ world_from_board
        for pose in world_from_tracker
    ]
    return (
        tracker_from_camera,
        world_from_board,
        world_from_tracker,
        camera_from_board,
    )


def test_handeye_candidates_recover_tracker_from_camera() -> None:
    """五算法候选应恢复 ``^tracker T_camera``，不能返回其逆。"""
    (
        tracker_from_camera,
        _,
        world_from_tracker,
        camera_from_board,
    ) = make_handeye_fixture()
    candidates = solve_handeye_candidates(
        world_from_tracker, camera_from_board
    )
    assert {item.method for item in candidates} >= {
        "TSAI",
        "PARK",
        "HORAUD",
    }
    best = select_handeye_seed(candidates)
    translation_mm, rotation_deg = transform_error(
        tracker_from_camera, best.tracker_from_camera
    )
    assert translation_mm < 0.5
    assert rotation_deg < 0.1
    inverse_error_mm, _ = transform_error(
        np.linalg.inv(tracker_from_camera), best.tracker_from_camera
    )
    assert inverse_error_mm > 20.0


def test_board_closure_is_zero_for_known_extrinsic() -> None:
    """真值外参计算的每帧 ``^world T_board`` 应保持固定。"""
    (
        tracker_from_camera,
        expected_board,
        world_from_tracker,
        camera_from_board,
    ) = make_handeye_fixture()
    boards = board_transforms(
        world_from_tracker, camera_from_board, tracker_from_camera
    )
    for board in boards:
        np.testing.assert_allclose(board, expected_board, atol=1.0e-10)
    translation_mm, rotation_deg, mean_board = board_closure_errors(boards)
    assert np.max(translation_mm) < 1.0e-9
    assert np.max(rotation_deg) < 1.0e-9
    np.testing.assert_allclose(mean_board, expected_board, atol=1.0e-10)


def test_noisy_candidates_remain_finite_and_consistent() -> None:
    """轻微位姿噪声下至少三种算法应给出有限且一致的候选。"""
    (
        tracker_from_camera,
        _,
        world_from_tracker,
        camera_from_board,
    ) = make_handeye_fixture()
    generator = np.random.default_rng(20260804)
    noisy_camera_from_board = []
    for pose in camera_from_board:
        perturbation = pose_to_matrix(
            generator.normal(0.0, 0.0002, 3),
            Rotation.from_rotvec(
                generator.normal(0.0, np.deg2rad(0.03), 3)
            ).as_quat(),
        )
        noisy_camera_from_board.append(perturbation @ pose)
    candidates = solve_handeye_candidates(
        world_from_tracker, noisy_camera_from_board
    )
    assert len(candidates) >= 3
    assert all(
        np.isfinite(candidate.translation_rmse_mm)
        and np.isfinite(candidate.rotation_rmse_deg)
        for candidate in candidates
    )
    best = select_handeye_seed(candidates)
    translation_mm, rotation_deg = transform_error(
        tracker_from_camera, best.tracker_from_camera
    )
    assert translation_mm < 2.0
    assert rotation_deg < 0.2
