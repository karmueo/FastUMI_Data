"""实现 RM75 策略目标发送前的工作空间、步长和时效安全检查。"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class SafetyLimits:
    """定义策略桥接器的笛卡尔、关节和时效安全门限。"""

    workspace_min_m: np.ndarray
    workspace_max_m: np.ndarray
    joint_min_rad: np.ndarray
    joint_max_rad: np.ndarray
    maximum_step_translation_m: float
    maximum_step_rotation_rad: float
    state_timeout_s: float
    target_timeout_s: float


def validate_limits(limits: SafetyLimits) -> None:
    """校验安全门限本身，阻止无效配置启动。"""
    minimum = np.asarray(limits.workspace_min_m, dtype=np.float64)
    maximum = np.asarray(limits.workspace_max_m, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("工作空间边界必须各包含三个数值")
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError("工作空间边界必须为有限数值")
    if np.any(minimum >= maximum):
        raise ValueError("工作空间最小边界必须小于最大边界")
    joint_minimum = np.asarray(limits.joint_min_rad, dtype=np.float64)
    joint_maximum = np.asarray(limits.joint_max_rad, dtype=np.float64)
    if joint_minimum.shape != (7,) or joint_maximum.shape != (7,):
        raise ValueError("RM75 关节边界必须各包含七个数值")
    if not np.all(np.isfinite(joint_minimum)) or not np.all(
        np.isfinite(joint_maximum)
    ):
        raise ValueError("RM75 关节边界必须为有限数值")
    if np.any(joint_minimum >= joint_maximum):
        raise ValueError("RM75 关节最小边界必须小于最大边界")
    positive_values = (
        limits.maximum_step_translation_m,
        limits.maximum_step_rotation_rad,
        limits.state_timeout_s,
        limits.target_timeout_s,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in positive_values):
        raise ValueError("步长和时效门限必须是正的有限数值")


class SafetyGate:
    """检查目标是否位于工作空间内且相邻目标变化受限。"""

    def __init__(self, limits: SafetyLimits) -> None:
        """保存经过验证的安全门限。"""
        validate_limits(limits)
        self.limits = limits

    def validate_target(
        self, target: np.ndarray, previous: Optional[np.ndarray]
    ) -> Optional[str]:
        """返回目标不安全的原因；目标安全时返回 ``None``。"""
        transform = np.asarray(target, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            return "目标变换形状无效或包含非有限数值"
        position = transform[:3, 3]
        if np.any(position < self.limits.workspace_min_m) or np.any(
            position > self.limits.workspace_max_m
        ):
            return "目标超出配置的 RM75 工作空间"
        if previous is None:
            return None
        delta = np.linalg.inv(previous) @ transform
        translation_step = float(np.linalg.norm(delta[:3, 3]))
        rotation_step = float(
            np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())
        )
        if translation_step > self.limits.maximum_step_translation_m:
            return (
                f"目标平移步长 {translation_step:.4f} m 超过 "
                f"{self.limits.maximum_step_translation_m:.4f} m"
            )
        if rotation_step > self.limits.maximum_step_rotation_rad:
            return (
                f"目标旋转步长 {rotation_step:.4f} rad 超过 "
                f"{self.limits.maximum_step_rotation_rad:.4f} rad"
            )
        return None

    def validate_joint_positions(
        self, positions_rad: np.ndarray
    ) -> Optional[str]:
        """返回关节状态不安全的原因；状态安全时返回 ``None``。"""
        positions = np.asarray(positions_rad, dtype=np.float64)
        if positions.shape != (7,) or not np.all(np.isfinite(positions)):
            return "RM75 关节状态必须包含七个有限弧度值"
        outside = np.logical_or(
            positions < self.limits.joint_min_rad,
            positions > self.limits.joint_max_rad,
        )
        if np.any(outside):
            joint_index = int(np.flatnonzero(outside)[0])
            return (
                f"RM75 关节 {joint_index + 1} 超出配置限位: "
                f"{positions[joint_index]:.4f} rad"
            )
        return None

    def stale_reason(
        self,
        now_ns: int,
        state_timestamp_ns: Optional[int],
        target_timestamp_ns: Optional[int],
    ) -> Optional[str]:
        """检查机器人状态和策略目标是否超时。"""
        if state_timestamp_ns is None:
            return "尚未收到 RM75 joint_states"
        state_age_s = (now_ns - state_timestamp_ns) / 1.0e9
        if state_age_s < -self.limits.state_timeout_s:
            return f"RM75 状态时间戳超前 {-state_age_s:.3f} s"
        if state_age_s > self.limits.state_timeout_s:
            return f"RM75 状态已过期 {state_age_s:.3f} s"
        if target_timestamp_ns is None:
            return "尚未收到策略目标"
        target_age_s = (now_ns - target_timestamp_ns) / 1.0e9
        if target_age_s < -self.limits.target_timeout_s:
            return f"策略目标时间戳超前 {-target_age_s:.3f} s"
        if target_age_s > self.limits.target_timeout_s:
            return f"策略目标已过期 {target_age_s:.3f} s"
        return None
