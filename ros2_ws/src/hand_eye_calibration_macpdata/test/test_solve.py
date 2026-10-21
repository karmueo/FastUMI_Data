"""手眼联合优化：无图像的合成观测（带像素噪声与一个坏样本）。"""

from pathlib import Path

import numpy as np
import pytest

from fastumi_bringup.initial_pose import INITIAL_JOINT_POSITIONS
from fastumi_bringup.kinematics import ArmKinematics
from fastumi_bringup.transforms import (
    axis_angle_matrix, invert, make_transform, rotation_angle,
)
from hand_eye_calibration_macpdata.board import AprilGridSpec, load_fisheye_camera
from hand_eye_calibration_macpdata.estimate import load_prior
from hand_eye_calibration_macpdata.solve import View, solve_hand_eye


SRC = Path(__file__).resolve().parents[2]
SPEC = AprilGridSpec(6, 6, 0.055, 0.3)


def _views(rng, count, noise_px):
    arm = ArmKinematics.from_urdf(SRC / "fastumi_rm75/assets/rm_75_kinematic.urdf")
    camera = load_fisheye_camera(SRC / "fastumi_usb_camera/config/EndEffectorCalib.yaml")
    flange_from_camera = load_prior(SRC.parents[1] / "config/calibration/T_cam_end.yaml")
    reference = arm.forward(np.array(INITIAL_JOINT_POSITIONS)) @ flange_from_camera
    rotation = reference[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    center = reference[:3, 3] + 0.35 * reference[:3, 2]
    base_from_board = make_transform(rotation, center - rotation @ SPEC.center_point())
    ids = np.arange(SPEC.corner_count)
    views = []
    while len(views) < count:
        # 绕板中心随机转动相机（±25°），保证板在视野内。
        axis = rng.normal(size=3)
        turn = axis_angle_matrix(axis / np.linalg.norm(axis), np.radians(rng.uniform(5, 25)))
        base_from_camera = reference.copy()
        base_from_camera[:3, :3] = turn @ reference[:3, :3]
        base_from_camera[:3, 3] = center + turn @ (reference[:3, 3] - center)
        camera_from_board = invert(base_from_camera) @ base_from_board
        points = SPEC.object_points(ids)
        pixels = camera.project(points @ camera_from_board[:3, :3].T + camera_from_board[:3, 3])
        if not np.all(np.isfinite(pixels)):
            continue
        views.append(View(base_from_flange=base_from_camera @ invert(flange_from_camera),
                          camera_from_board=camera_from_board, object_points=points,
                          pixels=pixels + rng.normal(scale=noise_px, size=pixels.shape)))
    return camera, views, flange_from_camera, base_from_board


def test_refinement_recovers_truth_and_rejects_bad_sample():
    rng = np.random.default_rng(0)
    camera, views, truth, base_from_board = _views(rng, 16, noise_px=0.3)
    # 第 5 个样本的法兰位姿偏 8 mm（如图像与关节不同步）。
    views[5].base_from_flange = views[5].base_from_flange @ make_transform(
        np.eye(3), [0.008, 0.0, 0.0])
    result = solve_hand_eye(camera, views)
    assert 5 not in result.inliers and len(result.inliers) == 15
    assert np.linalg.norm(result.flange_from_camera[:3, 3] - truth[:3, 3]) < 0.0005
    assert rotation_angle(result.flange_from_camera[:3, :3], truth[:3, :3]) < np.radians(0.05)
    assert result.base_from_board == pytest.approx(base_from_board, abs=1e-3)
    # 每轴 σ=0.3 px，点误差范数 RMS 为 0.3·√2。
    assert result.rms_px == pytest.approx(0.3 * np.sqrt(2.0), abs=0.05)
    assert set(result.methods) >= {"park", "tsai"}


def test_warns_on_small_or_poorly_excited_sets():
    rng = np.random.default_rng(1)
    camera, views, _, _ = _views(rng, 5, noise_px=0.1)
    result = solve_hand_eye(camera, views)
    assert any("少于" in warning for warning in result.warnings)
    with pytest.raises(ValueError):
        solve_hand_eye(camera, views[:2])
