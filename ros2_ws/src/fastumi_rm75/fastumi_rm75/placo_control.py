"""RM75 策略序列校验、绝对时间采样和关节消息辅助函数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))


def urdf_limits(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 joint1..joint7 的位置和速度限制，单位为弧度与弧度每秒。"""
    root = ET.parse(path).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    lower, upper, velocity = [], [], []
    for name in JOINT_NAMES:
        joint = joints.get(name)
        limit = None if joint is None else joint.find("limit")
        if limit is None or not all(
            key in limit.attrib for key in ("lower", "upper", "velocity")
        ):
            raise ValueError(f"URDF is missing limits for {name}")
        lower.append(float(limit.attrib["lower"]))
        upper.append(float(limit.attrib["upper"]))
        velocity.append(float(limit.attrib["velocity"]))
    arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (lower, upper, velocity)
    )
    if not all(np.isfinite(values).all() for values in arrays) or np.any(
        arrays[2] <= 0.0
    ):
        raise ValueError("URDF contains invalid RM75 joint limits")
    return arrays


@dataclass(frozen=True)
class TargetTrajectory:
    """从接收时刻锚定、按 ROS 绝对时钟执行的 Link7 目标。"""

    times_ns: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray
    grippers: np.ndarray
    anchor_time_ns: int
    anchor_position: np.ndarray
    anchor_quaternion: np.ndarray
    anchor_gripper: float
    episode_id: int
    sequence_id: int


def stamp_to_ns(stamp) -> int:
    """将 ROS sec/nanosec 时间转换成整数纳秒。"""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ordered_joint_positions(names, positions) -> np.ndarray:
    """按 joint1..joint7 返回有限弧度值，并拒绝歧义反馈。"""
    if len(names) != len(positions) or len(set(names)) != len(names):
        raise ValueError("joint state names must be unique and match positions")
    by_name = dict(zip(names, positions))
    missing = [name for name in JOINT_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"joint state is missing: {', '.join(missing)}")
    result = np.asarray([by_name[name] for name in JOINT_NAMES], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("joint state contains a non-finite value")
    return result


def sequence_is_new(
    latest_episode: int | None,
    latest_sequence: int | None,
    episode_id: int,
    sequence_id: int,
) -> bool:
    """判断策略序列是否来自更新的 episode 或含有递增编号。"""
    if latest_episode is None:
        return True
    if episode_id < latest_episode:
        return False
    if episode_id == latest_episode:
        return latest_sequence is not None and sequence_id > latest_sequence
    return True


def pose_arrays(pose) -> tuple[np.ndarray, np.ndarray]:
    """读取 geometry Pose，返回米和单位 xyzw 四元数。"""
    position = np.asarray(
        [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
    )
    quaternion = np.asarray(
        [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("policy pose contains a non-finite value")
    if norm < 1.0e-8:
        raise ValueError("policy pose contains an invalid quaternion")
    return position, quaternion / norm


def matrix_pose(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将 4×4 base_link→Link7 变换转换成位置和 xyzw 四元数。"""
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return transform[:3, 3].copy(), Rotation.from_matrix(transform[:3, :3]).as_quat()


def pose_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """从位置和 xyzw 四元数生成 4×4 变换。"""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    transform[:3, 3] = position
    return transform


def interpolate_pose(
    start_position: np.ndarray,
    start_quaternion: np.ndarray,
    goal_position: np.ndarray,
    goal_quaternion: np.ndarray,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """线性插值位置，并沿最短旋转路径插值姿态。"""
    ratio = float(np.clip(ratio, 0.0, 1.0))
    position = start_position + ratio * (goal_position - start_position)
    start = Rotation.from_quat(start_quaternion)
    delta = start.inv() * Rotation.from_quat(goal_quaternion)
    quaternion = (start * Rotation.from_rotvec(delta.as_rotvec() * ratio)).as_quat()
    return position, quaternion


def decode_trajectory(
    message,
    now_ns: int,
    anchor_transform: np.ndarray,
    anchor_gripper: float | None = None,
    *,
    base_frame: str = "base_link",
    end_frame: str = "Link7",
) -> TargetTrajectory:
    """校验策略消息、丢弃过期点并建立接收时刻锚点。"""
    if message.header.frame_id != base_frame or message.end_frame != end_frame:
        raise ValueError(
            f"policy frame must be {base_frame} -> {end_frame}"
        )
    count = len(message.poses)
    if count == 0 or len(message.time_from_start) != count or len(
        message.gripper_openness
    ) != count:
        raise ValueError("policy sequence arrays must have one equal non-zero length")
    base_ns = stamp_to_ns(message.header.stamp)
    offsets = np.asarray(
        [stamp_to_ns(value) for value in message.time_from_start], dtype=np.int64
    )
    if base_ns <= 0 or offsets[0] < 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError("policy timestamp must be positive and offsets strictly increasing")
    times_ns = base_ns + offsets
    if np.any(times_ns < base_ns):
        raise ValueError("policy target timestamp overflowed")
    decoded = [pose_arrays(pose) for pose in message.poses]
    positions = np.stack([item[0] for item in decoded])
    quaternions = np.stack([item[1] for item in decoded])
    grippers = np.asarray(message.gripper_openness, dtype=np.float64)
    if not np.isfinite(grippers).all() or np.any((grippers < 0.0) | (grippers > 1.0)):
        raise ValueError("policy gripper targets must be finite and in [0, 1]")
    future = times_ns > int(now_ns)
    if not np.any(future):
        raise ValueError("policy sequence is fully expired")
    positions = positions[future]
    quaternions = quaternions[future]
    grippers = grippers[future]
    anchor_position, anchor_quaternion = matrix_pose(anchor_transform)
    if anchor_gripper is None:
        anchor_gripper = float(grippers[0])
    if not np.isfinite(anchor_gripper) or not 0.0 <= anchor_gripper <= 1.0:
        raise ValueError("anchor gripper must be finite and in [0, 1]")
    return TargetTrajectory(
        times_ns=times_ns[future],
        positions=positions,
        quaternions=quaternions,
        grippers=grippers,
        anchor_time_ns=int(now_ns),
        anchor_position=anchor_position,
        anchor_quaternion=anchor_quaternion,
        anchor_gripper=float(anchor_gripper),
        episode_id=int(message.episode_id),
        sequence_id=int(message.sequence_id),
    )


def sample_trajectory(
    trajectory: TargetTrajectory, now_ns: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """在锚点、首个未来点及后续目标间按绝对时间插值。"""
    index = int(np.searchsorted(trajectory.times_ns, now_ns, side="right"))
    if index == 0:
        start_ns = trajectory.anchor_time_ns
        end_ns = int(trajectory.times_ns[0])
        ratio = (now_ns - start_ns) / max(1, end_ns - start_ns)
        position, quaternion = interpolate_pose(
            trajectory.anchor_position,
            trajectory.anchor_quaternion,
            trajectory.positions[0],
            trajectory.quaternions[0],
            ratio,
        )
        gripper = np.interp(
            np.clip(ratio, 0.0, 1.0),
            [0.0, 1.0],
            [trajectory.anchor_gripper, trajectory.grippers[0]],
        )
        return position, quaternion, float(gripper)
    if index >= len(trajectory.times_ns):
        return (
            trajectory.positions[-1].copy(),
            trajectory.quaternions[-1].copy(),
            float(trajectory.grippers[-1]),
        )
    start = index - 1
    denominator = max(
        1, int(trajectory.times_ns[index] - trajectory.times_ns[start])
    )
    ratio = (now_ns - int(trajectory.times_ns[start])) / denominator
    position, quaternion = interpolate_pose(
        trajectory.positions[start],
        trajectory.quaternions[start],
        trajectory.positions[index],
        trajectory.quaternions[index],
        ratio,
    )
    gripper = np.interp(
        ratio,
        [0.0, 1.0],
        [trajectory.grippers[start], trajectory.grippers[index]],
    )
    return position, quaternion, float(gripper)


def fill_joint_command(message, positions: np.ndarray) -> None:
    """将七轴弧度目标写入 RealMan 低跟随 CANFD 消息。"""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    if positions.shape != (7,) or not np.isfinite(positions).all():
        raise ValueError("RM75 command must contain seven finite joint positions")
    message.joint = positions.astype(np.float32).tolist()
    message.follow = False
    message.expand = 0.0
    message.dof = 7


def validate_workspace(positions, lower, upper) -> None:
    """确认 base_link 下的 Link7 位置均在配置工作区内，单位米。"""
    values = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    minimum = np.asarray(lower, dtype=np.float64).reshape(3)
    maximum = np.asarray(upper, dtype=np.float64).reshape(3)
    if not np.isfinite(values).all():
        raise ValueError("Link7 position is non-finite")
    outside = np.any((values < minimum) | (values > maximum), axis=1)
    if np.any(outside):
        raise ValueError(
            "Link7 position is outside the configured workspace: "
            f"{values[np.flatnonzero(outside)[0]].round(4).tolist()}")


def validate_start_envelope(position, start, max_displacement_m, max_rise_m) -> None:
    """在发布 CANFD 指令前限制 base_link 下 Link7 相对起始位置的米制位移。"""
    position = np.asarray(position, dtype=np.float64).reshape(3)
    start = np.asarray(start, dtype=np.float64).reshape(3)
    if not np.isfinite(position).all() or not np.isfinite(start).all():
        raise ValueError("trial Link7 position must be finite")
    if (not np.isfinite(max_displacement_m) or max_displacement_m < 0
            or not np.isfinite(max_rise_m) or max_rise_m < 0):
        raise ValueError("trial displacement and rise limits must be nonnegative")
    delta = position - start
    if max_displacement_m and np.linalg.norm(delta) > max_displacement_m:
        raise ValueError("Link7 command exceeds the configured start displacement")
    if max_rise_m and delta[2] > max_rise_m:
        raise ValueError("Link7 command exceeds the configured start rise")


def validate_gripper_targets(openness, minimum) -> None:
    """拒绝低于当前任务允许开度的未来夹爪目标；0 闭合，1 张开。"""
    values = np.asarray(openness, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("gripper targets must be a non-empty finite sequence")
    if np.any(values < minimum):
        raise ValueError(
            "gripper target is below the configured minimum: "
            f"{float(values.min()):.3f} < {minimum:.3f}"
        )


def gate_gripper_target(target, previous_command, feedback, close_allowed) -> float:
    """视觉门控不通过时保持当前夹爪命令；允许正常张开但不新增闭合动作。"""
    target = float(target)
    if not np.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("gripper target must be finite and in [0, 1]")
    hold = previous_command if previous_command is not None else feedback
    if hold is None or not np.isfinite(hold) or not 0.0 <= hold <= 1.0:
        hold = 1.0
    return target if close_allowed or target >= hold else float(hold)
