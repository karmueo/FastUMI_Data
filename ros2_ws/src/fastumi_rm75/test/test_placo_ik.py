"""使用部署 URDF 验证 Placo IK、速度限制和关节限制。"""

from pathlib import Path

import numpy as np
import placo
import pytest

from fastumi_bringup.initial_pose import load_initial_joint_positions
from fastumi_rm75.placo_control import JOINT_NAMES, urdf_limits


URDF = Path(__file__).parents[1] / "assets" / "rm_75_kinematic.urdf"


@pytest.mark.parametrize("rate_hz", [50, 100, 200])
def test_small_link7_target_respects_placo_limits(rate_hz):
    """不同频率下运行一秒，厘米级目标收敛且关节速度仍受 25% 上限约束。"""
    lower, upper, velocity = urdf_limits(URDF)
    robot = placo.RobotWrapper(str(URDF), placo.Flags.ignore_collisions)
    initial = np.asarray(load_initial_joint_positions())
    for name, position, limit in zip(JOINT_NAMES, initial, velocity):
        robot.set_joint(name, float(position))
        robot.set_velocity_limit(name, float(limit * 0.25))
    robot.update_kinematics()
    target = robot.get_T_world_frame("Link7").copy()
    target[2, 3] += 0.01
    solver = placo.KinematicsSolver(robot)
    solver.dt = 1.0 / rate_hz
    solver.mask_fbase(True)
    solver.enable_joint_limits(True)
    solver.enable_velocity_limits(True)
    task = solver.add_frame_task("Link7", target)
    task.configure("test_target", "soft", 1.0)
    largest_step = np.zeros(7)
    for _ in range(rate_hz):
        before = np.array([robot.get_joint(name) for name in JOINT_NAMES])
        solver.solve(True)
        robot.update_kinematics()
        after = np.array([robot.get_joint(name) for name in JOINT_NAMES])
        largest_step = np.maximum(largest_step, np.abs(after - before))
    final = np.array([robot.get_joint(name) for name in JOINT_NAMES])
    assert np.linalg.norm(
        robot.get_T_world_frame("Link7")[:3, 3] - target[:3, 3]
    ) < 1.0e-4
    assert np.all(largest_step <= velocity * 0.25 / rate_hz + 1.0e-8)
    assert np.all(final >= lower) and np.all(final <= upper)
