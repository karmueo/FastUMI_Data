"""按观测绝对时间采样 RM75 七关节与夹爪序列，不执行硬件操作。"""

from dataclasses import dataclass

import numpy as np

from fastumi_rm75.placo_control import JOINT_NAMES, stamp_to_ns


@dataclass(frozen=True)
class JointTrajectory:
    """时间为 ROS 纳秒，关节为弧度，夹爪为 [0,1]。"""

    times_ns: np.ndarray
    positions: np.ndarray
    grippers: np.ndarray
    anchor_ns: int
    anchor_positions: np.ndarray
    anchor_gripper: float


def validate_joints(values, lower, upper):
    """拒绝非有限值、错误维度和超出训练/硬件 URDF 的关节值。"""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim not in (1, 2) or array.shape[-1] != 7:
        raise ValueError("joint positions must have shape (7,) or (N,7)")
    if not np.isfinite(array).all() or np.any(array < lower) or np.any(array > upper):
        raise ValueError("joint positions are non-finite or outside URDF limits")
    return array


def decode_joint_trajectory(message, now_ns, anchor, anchor_gripper, lower, upper):
    """校验整个序列，丢弃已过期点，不改写观测锚点时间。"""
    if message.header.frame_id != "base_link" or tuple(message.joint_names) != JOINT_NAMES:
        raise ValueError("joint sequence must use base_link and joint1..joint7")
    count = len(message.points)
    if not count or len(message.gripper_openness) != count:
        raise ValueError("joint points and grippers must have equal nonzero length")
    offsets = np.asarray([stamp_to_ns(p.time_from_start) for p in message.points], dtype=np.int64)
    origin = stamp_to_ns(message.header.stamp)
    if origin <= 0 or origin > now_ns or offsets[0] < 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("invalid observation stamp or non-increasing offsets")
    if any(p.velocities or p.accelerations or p.effort for p in message.points):
        raise ValueError("only positions and time_from_start are supported")
    positions = validate_joints([p.positions for p in message.points], lower, upper)
    grippers = np.asarray(message.gripper_openness, dtype=np.float64)
    if not np.isfinite(grippers).all() or np.any((grippers < 0) | (grippers > 1)):
        raise ValueError("gripper values must be finite and in [0,1]")
    times = origin + offsets
    if np.any(times < origin):
        raise ValueError("timestamp overflow")
    future = times > now_ns
    if not future.any():
        raise ValueError("joint sequence is fully expired")
    anchor = validate_joints(anchor, lower, upper)
    if not np.isfinite(anchor_gripper) or not 0 <= anchor_gripper <= 1:
        raise ValueError("invalid anchor gripper")
    return JointTrajectory(times[future], positions[future], grippers[future], int(now_ns),
                           anchor.copy(), float(anchor_gripper))


def sample_joint_trajectory(trajectory, now_ns):
    """在实测锚点及未来预测之间逐关节/夹爪线性插值。"""
    if now_ns < trajectory.anchor_ns:
        raise ValueError("trajectory clock regressed")
    if now_ns >= trajectory.times_ns[-1]:
        raise ValueError("trajectory horizon exhausted")
    index = int(np.searchsorted(trajectory.times_ns, now_ns, side="right"))
    if index == 0:
        begin, q, g = trajectory.anchor_ns, trajectory.anchor_positions, trajectory.anchor_gripper
    else:
        begin, q, g = trajectory.times_ns[index - 1], trajectory.positions[index - 1], trajectory.grippers[index - 1]
    ratio = (now_ns - begin) / (trajectory.times_ns[index] - begin)
    return (q + ratio * (trajectory.positions[index] - q),
            float(g + ratio * (trajectory.grippers[index] - g)))


def limit_joint_step(target, previous, velocities, dt, period):
    """按上一发送值限速；延迟 tick 不得扩大正常周期允许的位移。"""
    if not np.isfinite(dt) or dt <= 0 or period <= 0:
        raise ValueError("nonpositive controller interval")
    maximum = np.asarray(velocities) * min(dt, period)
    return np.asarray(previous) + np.clip(np.asarray(target) - previous, -maximum, maximum)
