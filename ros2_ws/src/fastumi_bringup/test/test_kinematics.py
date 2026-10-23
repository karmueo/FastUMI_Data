"""RM75 URDF 正逆运动学与变换工具。"""

import math
from pathlib import Path

import numpy as np
import pytest

from fastumi_bringup.kinematics import ArmKinematics
from fastumi_bringup.transforms import (
    average_transforms, axis_angle_matrix, invert, make_transform, matrix_from_quaternion,
    quaternion_from_matrix, rotation_angle, rotation_log,
)
from fastumi_bringup.initial_pose import INITIAL_JOINT_POSITIONS


URDF = Path(__file__).resolve().parents[2] / "fastumi_rm75/assets/rm_75_kinematic.urdf"


@pytest.fixture(scope="module")
def arm():
    return ArmKinematics.from_urdf(URDF)


def test_zero_pose_is_vertical_stack(arm):
    # 0.2405 + 0.256 + 0.21 + 0.144 = 0.8505 m，各关节零位时链条竖直。
    pose = arm.forward(np.zeros(7))
    assert pose[:3, 3] == pytest.approx([0.0, 0.0, 0.8505], abs=1e-6)
    assert pose[:3, :3] == pytest.approx(np.eye(3), abs=1e-6)


def test_configured_initial_pose_is_valid_and_reachable(arm):
    """起点可由 YAML 调整，须在关节限位内且可由附近种子求解。"""
    assert arm.within_limits(INITIAL_JOINT_POSITIONS)
    pose = arm.forward(INITIAL_JOINT_POSITIONS)
    joints = arm.solve_ik(pose, np.asarray(INITIAL_JOINT_POSITIONS) + 0.01)
    assert joints is not None
    reached = arm.forward(joints)
    assert np.linalg.norm(reached[:3, 3] - pose[:3, 3]) < 5e-4
    assert rotation_angle(reached[:3, :3], pose[:3, :3]) < math.radians(0.1)


def test_jacobian_matches_finite_difference(arm):
    q = np.radians([10, 30, -20, 60, 15, 70, 30])
    base = arm.forward(q)
    numeric = np.zeros((6, 7))
    for index in range(7):
        delta = np.zeros(7)
        delta[index] = 1e-6
        moved = arm.forward(q + delta)
        numeric[:3, index] = (moved[:3, 3] - base[:3, 3]) / 1e-6
        numeric[3:, index] = base[:3, :3] @ rotation_log(base[:3, :3].T @ moved[:3, :3]) / 1e-6
    assert arm.jacobian(q) == pytest.approx(numeric, abs=1e-5)


def test_ik_round_trip_near_seed(arm):
    rng = np.random.default_rng(0)
    seed = np.array(INITIAL_JOINT_POSITIONS)
    start = arm.forward(seed)
    solved = 0
    for _ in range(20):
        target = start.copy()
        target[:3, :3] = start[:3, :3] @ axis_angle_matrix(rng.normal(size=3), math.radians(20))
        target[:3, 3] += rng.uniform(-0.05, 0.05, 3)
        joints = arm.solve_ik(target, seed)
        if joints is None:
            continue
        solved += 1
        reached = arm.forward(joints)
        assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) < 5e-4
        assert rotation_angle(reached[:3, :3], target[:3, :3]) < math.radians(0.1)
        assert arm.within_limits(joints)
    assert solved >= 15


def test_ik_rejects_unreachable_target(arm):
    target = make_transform(np.eye(3), [2.0, 0.0, 0.5])
    assert arm.solve_ik(target, np.array(INITIAL_JOINT_POSITIONS)) is None


def test_quaternion_round_trip_and_average():
    rotation = axis_angle_matrix([0.3, -0.5, 0.8], 2.9)
    quaternion = quaternion_from_matrix(rotation)
    assert quaternion[3] >= 0.0
    assert matrix_from_quaternion(quaternion) == pytest.approx(rotation, abs=1e-9)
    transform = make_transform(rotation, [0.1, 0.2, 0.3])
    assert invert(transform) @ transform == pytest.approx(np.eye(4), abs=1e-12)
    jittered = [transform @ make_transform(axis_angle_matrix([1, 0, 0], angle), [0, 0, 0])
                for angle in (-0.01, 0.01)]
    assert average_transforms(jittered) == pytest.approx(transform, abs=1e-4)
