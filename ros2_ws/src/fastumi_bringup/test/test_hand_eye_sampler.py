"""位姿采样器、粗手眼估计与多样性统计。"""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from fastumi_bringup.hand_eye.board import CheckerboardSpec, load_fisheye_camera
from fastumi_bringup.hand_eye.estimate import (
    diversity_stats, estimate_board_in_base, estimate_flange_from_camera, load_prior,
)
from fastumi_bringup.hand_eye.kinematics import ArmKinematics
from fastumi_bringup.hand_eye.occlusion import check_board_view
from fastumi_bringup.hand_eye.sampler import (
    PlanContext, SamplerSettings, aim_for_mask, bootstrap_targets, plan_next,
    sample_camera_pose,
)
from fastumi_bringup.hand_eye.transforms import (
    invert, make_transform, rotation_angle,
)
from fastumi_bringup.initial_pose import INITIAL_JOINT_POSITIONS


SRC = Path(__file__).resolve().parents[2]
PRIOR = SRC.parents[1] / "config/calibration/T_cam_end.yaml"
# 与相机同向（相机在板 -z 侧）的 OpenCV 约定板，外框 0.32 x 0.20 m。
SPEC = CheckerboardSpec(7, 4, 0.04, 0.04)


@pytest.fixture(scope="module")
def scene():
    arm = ArmKinematics.from_urdf(SRC / "fastumi_rm75/assets/rm_75_kinematic.urdf")
    camera = load_fisheye_camera(SRC / "fastumi_usb_camera/config/EndEffectorCalib.yaml")
    flange_from_camera = load_prior(PRIOR)
    joints = np.array(INITIAL_JOINT_POSITIONS)
    base_from_camera = arm.forward(joints) @ flange_from_camera
    center = base_from_camera[:3, 3] + 0.35 * base_from_camera[:3, 2]
    rotation = base_from_camera[:3, :3]
    board = make_transform(rotation, center - rotation @ SPEC.center_point())
    return arm, camera, flange_from_camera, joints, base_from_camera, board


def test_prior_is_inverted_cam_from_hand():
    prior = load_prior(PRIOR)
    raw = invert(prior)
    assert raw[1, 3] == pytest.approx(0.0930904727)
    assert load_prior(PRIOR, "flange_from_cam") == pytest.approx(raw)
    with pytest.raises(ValueError):
        load_prior(PRIOR, "unknown")


def test_plan_sequence_is_safe_and_diverse(scene):
    arm, camera, flange_from_camera, joints, base_from_camera, board = scene
    settings = SamplerSettings(candidate_count=6)
    mask = np.zeros((camera.height, camera.width), np.uint8)
    mask[int(camera.height * 0.75):, int(camera.width * 0.3):int(camera.width * 0.7)] = 255
    context = PlanContext(board, flange_from_camera, base_from_camera)
    rng = np.random.default_rng(2)
    current = joints
    for _ in range(12):
        candidate, reasons = plan_next(arm, SPEC, camera, mask, context, current, rng, settings)
        assert candidate is not None, reasons
        assert np.max(np.abs(candidate.joints - current)) <= settings.max_joint_step_rad
        assert arm.within_limits(candidate.joints, settings.joint_limit_margin_rad)
        flange = arm.forward(candidate.joints)
        assert np.all(flange[:3, 3] >= settings.workspace_min)
        assert np.all(flange[:3, 3] <= settings.workspace_max)
        view = check_board_view(SPEC, camera, invert(flange @ flange_from_camera) @ board, mask)
        assert view.ok, view.reason
        context.sampled_base_from_flange.append(flange)
        current = candidate.joints
    stats = diversity_stats(context.sampled_base_from_flange)
    # hand_eye_calibration 默认门限：旋转跨度 ≥ 35°，轴分离 ≥ 10°。
    assert stats.rotation_span_deg >= 35.0
    assert stats.axis_separation_deg >= 10.0


def _finger_mask(camera):
    """类似实机的夹爪掩膜：底部机身加两根伸到画面 57% 高度的指尖。"""
    mask = np.zeros((camera.height, camera.width), np.uint8)
    width, height = camera.width, camera.height
    mask[int(height * 0.8):, int(width * 0.1):int(width * 0.9)] = 255
    mask[int(height * 0.57):, int(width * 0.3):int(width * 0.45)] = 255
    mask[int(height * 0.57):, int(width * 0.55):int(width * 0.7)] = 255
    return mask


def test_aim_avoids_gripper_fingers(scene):
    """指尖遮住画面中下部时，板中心瞄准掩膜上方居中的位置，规划仍能连续成功。"""
    arm, camera, flange_from_camera, joints, base_from_camera, board = scene
    mask = _finger_mask(camera)
    assert aim_for_mask(camera, None)[0] == pytest.approx((camera.width / 2, camera.height / 2))
    pixel, direction = aim_for_mask(camera, mask)
    assert pixel[1] < 0.45 * camera.height
    assert abs(pixel[0] - camera.width / 2) < 0.1 * camera.width

    settings = SamplerSettings(aim_direction=tuple(direction), target_jitter_m=0.0, roll_deg=0.0)
    context = PlanContext(board, flange_from_camera, base_from_camera)
    rng = np.random.default_rng(0)
    center = board @ np.append(SPEC.center_point(), 1.0)
    for _ in range(5):
        camera_pose = sample_camera_pose(context, SPEC, rng, settings)
        projected = camera.project((invert(camera_pose) @ center)[None, :3])[0]
        assert projected == pytest.approx(pixel, abs=1.0)

    current = joints
    for _ in range(6):
        candidate, reasons = plan_next(arm, SPEC, camera, mask, context, current, rng, settings)
        assert candidate is not None, reasons
        context.sampled_base_from_flange.append(candidate.base_from_flange)
        current = candidate.joints


def test_bootstrap_targets_rotate_about_pivot():
    reference = make_transform(np.eye(3), [0.3, 0.0, 0.4])
    pivot = np.array([0.0, 0.05, 0.3])
    targets = bootstrap_targets(reference, pivot)
    assert len(targets) == 8
    pivot_base = reference @ np.append(pivot, 1.0)
    for target in targets:
        assert target @ np.append(pivot, 1.0) == pytest.approx(pivot_base)
        assert 0.1 < math.degrees(rotation_angle(reference[:3, :3], target[:3, :3])) <= 15.01


def test_hand_eye_estimate_recovers_truth(scene):
    arm, _, flange_from_camera, joints, _, board = scene
    rng = np.random.default_rng(5)
    flanges, boards = [], []
    for _ in range(10):
        target = arm.forward(joints + rng.uniform(-0.25, 0.25, 7))
        flanges.append(target)
        boards.append(invert(target @ flange_from_camera) @ board)
    estimate = estimate_flange_from_camera(flanges, boards, cv2.CALIB_HAND_EYE_PARK)
    assert estimate == pytest.approx(flange_from_camera, abs=1e-6)
    board_estimate = estimate_board_in_base(flanges, boards, estimate)
    assert board_estimate.base_from_board == pytest.approx(board, abs=1e-6)
    assert board_estimate.spread_mm < 1e-3
    assert estimate_flange_from_camera(flanges[:2], boards[:2]) is None
