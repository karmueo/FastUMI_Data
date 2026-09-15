"""验证实机执行器的时间选择、插值、限速和姿态误差计算。"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("rclpy")
pytest.importorskip("fastumi_interfaces.msg")

from vr_umi_ros.real_executor import (
    ExecutionPlan,
    interpolate_pose,
    pose_error,
    rate_limit_pose,
    sample_plan,
)


def plan():
    """构造从零到两厘米的确定性两点轨迹。"""
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    return ExecutionPlan(
        times_ns=np.array([200, 300], dtype=np.int64),
        positions=np.array([[0.01, 0, 0], [0.02, 0, 0]], dtype=np.float64),
        quaternions=np.stack([identity, identity]),
        grippers=np.array([0.6, 0.8]),
        anchor_time_ns=100,
        anchor_position=np.zeros(3),
        anchor_quaternion=identity,
        anchor_gripper=0.4,
        source="test",
    )


def test_sample_plan_interpolates_anchor_and_future_points():
    """当前时间落在首点前或两点间时均使用绝对时轴插值。"""
    position, quaternion, gripper = sample_plan(plan(), 150)
    np.testing.assert_allclose(position, [0.005, 0, 0])
    np.testing.assert_allclose(quaternion, [0, 0, 0, 1])
    assert gripper == pytest.approx(0.5)
    position, _, gripper = sample_plan(plan(), 250)
    np.testing.assert_allclose(position, [0.015, 0, 0])
    assert gripper == pytest.approx(0.7)


def test_pose_interpolation_and_rate_limits_take_shortest_rotation():
    """位置和旋转每周期不能越过配置上限。"""
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    target_quaternion = Rotation.from_euler("z", 0.4).as_quat()
    position, quaternion = rate_limit_pose(
        np.zeros(3), identity, np.array([0.1, 0, 0]), target_quaternion,
        max_translation=0.002, max_rotation=0.01)
    np.testing.assert_allclose(position, [0.002, 0, 0])
    translation, rotation = pose_error(np.zeros(3), identity, position, quaternion)
    assert translation == pytest.approx(0.002)
    assert rotation == pytest.approx(0.01)
    midpoint, mid_quaternion = interpolate_pose(
        np.zeros(3), identity, np.array([0.1, 0, 0]), target_quaternion, 0.5)
    np.testing.assert_allclose(midpoint, [0.05, 0, 0])
    assert Rotation.from_quat(mid_quaternion).magnitude() == pytest.approx(0.2)
