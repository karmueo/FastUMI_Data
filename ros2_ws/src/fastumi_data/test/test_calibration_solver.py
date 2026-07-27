"""测试 robot-world/hand-eye 和 pivot 外参求解。"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastumi_data.calibration_solver import (
    solve_pivot_translation,
    solve_robot_world_hand_eye,
)
from fastumi_data.pose_math import pose_to_matrix


def _random_transform(generator: np.random.Generator) -> np.ndarray:
    """生成覆盖平移和三维旋转的确定性随机变换。"""
    return pose_to_matrix(
        generator.uniform(-0.5, 0.5, size=3),
        Rotation.from_rotvec(
            generator.uniform(-1.2, 1.2, size=3)
        ).as_quat(),
    )


def test_recovers_tracker_to_tcp_from_paired_poses() -> None:
    """验证无噪声配对姿态恢复毫米级和角度级外参。"""
    generator = np.random.default_rng(7)
    base_to_vive = _random_transform(generator)
    tracker_to_tcp = _random_transform(generator)
    tracker_poses = [_random_transform(generator) for _ in range(24)]
    base_tcp_poses = [
        base_to_vive @ tracker @ tracker_to_tcp
        for tracker in tracker_poses
    ]

    result = solve_robot_world_hand_eye(
        base_tcp_poses,
        tracker_poses,
        initial_tracker_to_tcp=tracker_to_tcp,
    )

    assert result.translation_rmse_mm < 1.0e-5
    assert result.rotation_rmse_deg < 1.0e-5
    assert result.tracker_to_tcp == pytest.approx(
        tracker_to_tcp, abs=1.0e-7
    )


def test_pivot_recovers_tcp_translation() -> None:
    """验证固定点多方向姿态恢复 Tracker 坐标系中的 TCP 平移。"""
    generator = np.random.default_rng(11)
    tracker_to_tcp = np.asarray([0.04, -0.02, 0.12])
    fixed_world_point = np.asarray([0.5, -0.1, 0.8])
    poses = []
    for _ in range(20):
        rotation = Rotation.from_rotvec(
            generator.uniform(-1.5, 1.5, size=3)
        ).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = fixed_world_point - rotation @ tracker_to_tcp
        poses.append(transform)

    recovered, rmse_mm = solve_pivot_translation(poses)

    assert recovered == pytest.approx(tracker_to_tcp, abs=1.0e-9)
    assert rmse_mm < 1.0e-6
