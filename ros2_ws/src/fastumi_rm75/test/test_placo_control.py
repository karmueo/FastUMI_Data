"""测试 Placo 控制器的序列校验、采样和 RM75 消息编码。"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastumi_rm75.placo_control import (
    TargetTrajectory,
    decode_trajectory,
    fill_joint_command,
    gate_gripper_target,
    ordered_joint_positions,
    sample_trajectory,
    sequence_is_new,
    validate_gripper_targets,
    validate_start_envelope,
    validate_workspace,
)


def _stamp(nanoseconds):
    return SimpleNamespace(
        sec=nanoseconds // 1_000_000_000,
        nanosec=nanoseconds % 1_000_000_000,
    )


def _pose(x, quaternion=(0.0, 0.0, 0.0, 1.0)):
    return SimpleNamespace(
        position=SimpleNamespace(x=x, y=0.0, z=0.3),
        orientation=SimpleNamespace(
            x=quaternion[0],
            y=quaternion[1],
            z=quaternion[2],
            w=quaternion[3],
        ),
    )


def _message(*, episode=1, sequence=2):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=_stamp(1_000_000_000), frame_id="base_link"),
        end_frame="Link7",
        episode_id=episode,
        sequence_id=sequence,
        time_from_start=[_stamp(0), _stamp(100_000_000), _stamp(200_000_000)],
        poses=[_pose(0.30), _pose(0.31), _pose(0.32)],
        gripper_openness=[0.4, 0.5, 0.6],
    )


def test_joint_feedback_is_reordered_and_rejects_duplicates():
    """额外关节和乱序可接受，重复名称会被拒绝。"""
    names = ["extra", "joint3", "joint1", "joint7", "joint2", "joint6", "joint5", "joint4"]
    positions = [9.0, 3.0, 1.0, 7.0, 2.0, 6.0, 5.0, 4.0]
    np.testing.assert_allclose(
        ordered_joint_positions(names, positions), np.arange(1.0, 8.0)
    )
    with pytest.raises(ValueError, match="unique"):
        ordered_joint_positions(["joint1", "joint1"], [0.0, 0.0])


def test_sequence_order_handles_duplicates_and_episode_changes():
    """同 episode 只接受递增编号，并拒绝旧 episode 的在途结果。"""
    assert sequence_is_new(None, None, 0, 1)
    assert sequence_is_new(2, 9, 2, 10)
    assert not sequence_is_new(2, 9, 2, 9)
    assert not sequence_is_new(2, 9, 1, 100)
    assert sequence_is_new(2, 9, 3, 1)


def test_decode_skips_expired_points_and_anchors_at_receive_time():
    """接收后仅保留未来目标，首段从给定 FK 锚点开始。"""
    anchor = np.eye(4)
    anchor[:3, 3] = [0.29, 0.0, 0.3]
    trajectory = decode_trajectory(
        _message(), 1_050_000_000, anchor, anchor_gripper=None
    )
    np.testing.assert_array_equal(
        trajectory.times_ns, [1_100_000_000, 1_200_000_000]
    )
    np.testing.assert_allclose(trajectory.anchor_position, [0.29, 0.0, 0.3])
    assert trajectory.anchor_gripper == pytest.approx(0.5)

    position, quaternion, gripper = sample_trajectory(trajectory, 1_075_000_000)
    np.testing.assert_allclose(position, [0.30, 0.0, 0.3])
    np.testing.assert_allclose(quaternion, [0.0, 0.0, 0.0, 1.0])
    assert gripper == pytest.approx(0.5)


def test_sampling_uses_shortest_rotation_and_interpolates_gripper():
    """相邻目标之间按绝对时间插值姿态和夹爪。"""
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    rotated = Rotation.from_euler("z", 0.4).as_quat()
    trajectory = TargetTrajectory(
        times_ns=np.array([200, 300]),
        positions=np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]),
        quaternions=np.stack([identity, -rotated]),
        grippers=np.array([0.4, 0.8]),
        anchor_time_ns=100,
        anchor_position=np.zeros(3),
        anchor_quaternion=identity,
        anchor_gripper=0.4,
        episode_id=0,
        sequence_id=1,
    )
    position, quaternion, gripper = sample_trajectory(trajectory, 250)
    np.testing.assert_allclose(position, [0.01, 0.0, 0.0])
    assert Rotation.from_quat(quaternion).magnitude() == pytest.approx(0.2)
    assert gripper == pytest.approx(0.6)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda message: setattr(message, "end_frame", "tool0"), "frame"),
        (lambda message: message.time_from_start.pop(), "equal"),
        (lambda message: message.time_from_start.__setitem__(1, _stamp(0)), "increasing"),
        (lambda message: message.gripper_openness.__setitem__(0, 1.1), "gripper"),
        (lambda message: setattr(message.poses[0].orientation, "w", 0.0), "quaternion"),
    ],
)
def test_decode_rejects_invalid_policy_messages(mutation, match):
    """坐标系、数组、时间、夹爪或四元数非法时拒绝整段消息。"""
    message = _message()
    mutation(message)
    with pytest.raises(ValueError, match=match):
        decode_trajectory(message, 1_050_000_000, np.eye(4))


def test_fully_expired_sequence_is_rejected():
    with pytest.raises(ValueError, match="fully expired"):
        decode_trajectory(_message(), 1_300_000_000, np.eye(4))


def test_joint_command_uses_radians_and_low_follow_fields():
    """Jointpos 保持输入弧度并设置七轴低跟随协议字段。"""
    message = SimpleNamespace()
    positions = np.linspace(-0.3, 0.3, 7)
    fill_joint_command(message, positions)
    np.testing.assert_allclose(message.joint, positions, rtol=1.0e-6)
    assert message.follow is False
    assert message.expand == 0.0
    assert message.dof == 7


def test_workspace_rejects_outside_link7_target():
    """任务工作区同时约束当前锚点和将要执行的 Link7 目标。"""
    lower = [0.10, -0.22, 0.16]
    upper = [0.54, 0.28, 0.50]
    validate_workspace([[0.30, 0.0, 0.34], [0.45, -0.10, 0.42]], lower, upper)
    with pytest.raises(ValueError, match="workspace"):
        validate_workspace([[0.30, 0.0, 0.34], [0.09, 0.0, 0.34]], lower, upper)
    with pytest.raises(ValueError, match="workspace"):
        validate_workspace([0.30, 0.0, 0.61], lower, upper)


def test_gripper_safety_floor_rejects_spurious_full_close():
    """可选的非零夹爪命令下限仍可用于其他任务。"""
    validate_gripper_targets([1.0, 0.5, 0.35], 0.34)
    with pytest.raises(ValueError, match="configured minimum"):
        validate_gripper_targets([1.0, 0.0], 0.34)


def test_trial_envelope_rejects_ik_command_before_hardware_boundary():
    """控制器必须在发布关节目标前拒绝超出起点距离或上升限制的 Link7 位姿。"""
    start = [0.30, 0.0, 0.34]
    validate_start_envelope([0.27, -0.03, 0.35], start, 0.05, 0.015)
    with pytest.raises(ValueError, match="start displacement"):
        validate_start_envelope([0.24, 0.0, 0.34], start, 0.05, 0.015)
    with pytest.raises(ValueError, match="start rise"):
        validate_start_envelope([0.30, 0.0, 0.36], start, 0.05, 0.015)
    with pytest.raises(ValueError, match="finite"):
        validate_start_envelope([np.nan, 0.0, 0.34], start, 0.05, 0.015)


def test_gripper_closure_waits_for_visual_approval_but_can_hold_or_reopen():
    """训练需要 0 闭合命令，视觉许可缺失时保持开度并允许张开。"""
    assert gate_gripper_target(0.0, None, 0.99, False) == pytest.approx(0.99)
    assert gate_gripper_target(0.0, 0.60, 0.70, False) == pytest.approx(0.60)
    assert gate_gripper_target(0.80, 0.60, 0.70, False) == pytest.approx(0.80)
    assert gate_gripper_target(0.0, 0.60, 0.70, True) == pytest.approx(0.0)
    assert gate_gripper_target(0.0, None, None, False) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="gripper target"):
        gate_gripper_target(float("nan"), None, 1.0, True)
