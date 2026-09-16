"""提供遥操位姿映射、输入校验和关节指令平滑的纯计算逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


def parse_home_joint_positions(
    values: Optional[list[float]],
) -> Optional[np.ndarray]:
    """解析可选的回位关节配置，不修改输入。

    Args:
        values: 按 joint1 至 joint7 排列的关节角，单位为弧度。
            空列表或 ROS 未设置值 None 表示使用启动后的首帧完整反馈。

    Returns:
        配置的七轴关节角副本；未配置时返回 None。

    Raises:
        ValueError: 非空配置不包含七个有限数值时抛出。
    """
    if values is None:
        return None
    positions = np.asarray(values)
    if positions.shape == (0,):
        return None
    if (
        positions.shape != (7,)
        or positions.dtype.kind not in "iuf"
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(
            "home_joint_positions_rad 必须为空列表或七个有限数值"
            "（joint1 至 joint7，单位为弧度）"
        )
    return positions.astype(np.float64, copy=True)


def validate_scale(value: float, name: str) -> float:
    """校验遥操灵敏度并返回浮点值。

    Args:
        value: 待校验的灵敏度。
        name: 用于异常信息的参数名称。

    Returns:
        位于开区间下界、闭区间上界 ``(0, 1]`` 的灵敏度。

    Raises:
        ValueError: 数值非有限或超出范围时抛出。
    """
    scale = float(value)
    if not np.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError(f"{name} 必须位于 (0, 1] 范围")
    return scale


def validate_transform(transform: np.ndarray, name: str = "位姿") -> np.ndarray:
    """校验并复制一个有限、右手正交的 4×4 齐次变换。"""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} 必须是有限的 4x4 矩阵")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise ValueError(f"{name} 的齐次矩阵末行无效")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError(f"{name} 的旋转矩阵不正交")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise ValueError(f"{name} 的旋转矩阵必须为右手系")
    return matrix.copy()


def pose_to_matrix(
    position: np.ndarray, quaternion_xyzw: np.ndarray
) -> np.ndarray:
    """将米制位置和 xyzw 四元数组合为齐次变换。"""
    translation = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("位置必须包含三个有限数值")
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("四元数必须包含四个有限数值")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-8:
        raise ValueError("四元数范数过小")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    transform[:3, 3] = translation
    return transform


def axis_mapping_from_rpy(rpy_deg: np.ndarray) -> np.ndarray:
    """按 ``Rz(yaw) Ry(pitch) Rx(roll)`` 生成 Tracker 安装轴向矩阵。"""
    angles = np.asarray(rpy_deg, dtype=np.float64)
    if angles.shape != (3,) or not np.all(np.isfinite(angles)):
        raise ValueError("axis_mapping_rpy_deg 必须包含三个有限数值")
    return Rotation.from_euler("xyz", angles, degrees=True).as_matrix()


def mapping_basis_from_reference(
    tracker_pose: np.ndarray,
    eef_pose: np.ndarray,
    axis_mapping: np.ndarray,
) -> np.ndarray:
    """根据初始化时的 Tracker 和末端姿态建立固定映射轴向。"""
    tracker_reference = validate_transform(
        tracker_pose, "Tracker 初始化位姿"
    )
    eef_reference = validate_transform(eef_pose, "末端初始化位姿")
    mapping = np.asarray(axis_mapping, dtype=np.float64)
    if mapping.shape != (3, 3) or not np.all(np.isfinite(mapping)):
        raise ValueError("安装轴向必须是有限的 3x3 矩阵")
    if not np.allclose(mapping.T @ mapping, np.eye(3), atol=1.0e-6):
        raise ValueError("安装轴向必须是正交矩阵")
    if not np.isclose(np.linalg.det(mapping), 1.0, atol=1.0e-6):
        raise ValueError("安装轴向必须为右手旋转")
    return (
        eef_reference[:3, :3]
        @ mapping
        @ tracker_reference[:3, :3].T
    )


def workspace_mapping_from_samples(
    start_pose: np.ndarray,
    up_pose: np.ndarray,
    forward_pose: np.ndarray,
    minimum_distance_m: float = 0.05,
    minimum_angle_deg: float = 60.0,
) -> np.ndarray:
    """根据起点、向上终点和向前终点建立 odom 到 Base 的轴向映射。

    三个输入均为 Tracker 在同一 odom 坐标系中的位姿。返回矩阵的三行依次
    表示 Base 的 ``+X`` 前方、``+Y`` 左方和 ``+Z`` 上方在 odom 中的单位轴。

    Args:
        start_pose: 标定起点位姿。
        up_pose: 从起点向上平移后的位姿。
        forward_pose: 从上方位置继续向前平移后的位姿。
        minimum_distance_m: 每段平移允许的最小距离，单位为米。
        minimum_angle_deg: 两段方向与同向、反向共线方向的最小夹角，单位为度。

    Returns:
        将 odom 向量映射到 Base ``+X/+Y/+Z`` 分量的右手正交矩阵。

    Raises:
        ValueError: 位姿、阈值或采样几何关系无效时抛出。
    """
    poses = (
        validate_transform(start_pose, "工作空间标定起点"),
        validate_transform(up_pose, "工作空间标定上方点"),
        validate_transform(forward_pose, "工作空间标定前方点"),
    )
    if not np.isfinite(minimum_distance_m) or minimum_distance_m <= 0.0:
        raise ValueError("工作空间标定最小位移必须为正有限数")
    if not np.isfinite(minimum_angle_deg) or not 0.0 < minimum_angle_deg < 90.0:
        raise ValueError("工作空间标定最小夹角必须位于 (0, 90) 度")

    up_displacement = poses[1][:3, 3] - poses[0][:3, 3]
    forward_displacement = poses[2][:3, 3] - poses[1][:3, 3]
    up_distance = float(np.linalg.norm(up_displacement))
    forward_distance = float(np.linalg.norm(forward_displacement))
    if up_distance < minimum_distance_m:
        raise ValueError(
            f"向上位移至少需要 {minimum_distance_m:.3f} m，"
            f"当前为 {up_distance:.3f} m"
        )
    if forward_distance < minimum_distance_m:
        raise ValueError(
            f"向前位移至少需要 {minimum_distance_m:.3f} m，"
            f"当前为 {forward_distance:.3f} m"
        )

    up_axis = up_displacement / up_distance
    forward_direction = forward_displacement / forward_distance
    direction_sum = up_axis + forward_direction
    direction_difference = up_axis - forward_direction
    sum_norm = float(np.linalg.norm(direction_sum))
    difference_norm = float(np.linalg.norm(direction_difference))
    if sum_norm < 1.0e-6 or difference_norm < 1.0e-6:
        raise ValueError("向上和向前位移方向共线，无法确定坐标系")
    displacement_angle_deg = float(
        np.rad2deg(
            np.arccos(np.clip(np.dot(up_axis, forward_direction), -1.0, 1.0))
        )
    )
    angle_tolerance_deg = 1.0e-9  # 消除浮点舍入导致的边界误判。
    if (
        displacement_angle_deg < minimum_angle_deg - angle_tolerance_deg
        or displacement_angle_deg > 180.0 - minimum_angle_deg + angle_tolerance_deg
    ):
        raise ValueError(
            "向上和向前位移夹角必须位于 "
            f"[{minimum_angle_deg:.1f}, {180.0 - minimum_angle_deg:.1f}] 度，"
            f"当前为 {displacement_angle_deg:.1f} 度"
        )

    direction_bisector = direction_sum / sum_norm
    direction_difference /= difference_norm
    normalization_scale = float(np.sqrt(2.0))
    up_axis = (
        direction_bisector + direction_difference
    ) / normalization_scale
    forward_axis = (
        direction_bisector - direction_difference
    ) / normalization_scale
    left_axis = np.cross(up_axis, forward_axis)
    left_axis /= np.linalg.norm(left_axis)
    mapping = np.vstack((forward_axis, left_axis, up_axis))
    if not np.allclose(mapping @ mapping.T, np.eye(3), atol=1.0e-6):
        raise ValueError("工作空间标定未能生成正交轴向")
    if not np.isclose(np.linalg.det(mapping), 1.0, atol=1.0e-6):
        raise ValueError("工作空间标定未能生成右手轴向")
    return mapping


def map_tracker_target(
    reference_tracker: np.ndarray,
    current_tracker: np.ndarray,
    reference_eef: np.ndarray,
    mapping_basis: np.ndarray,
    translation_scale: float,
    rotation_scale: float,
) -> np.ndarray:
    """使用初始化后固定的轴向把 Tracker 相对运动映射到末端目标。"""
    tracker_zero = validate_transform(reference_tracker, "Tracker 参考位姿")
    tracker_now = validate_transform(current_tracker, "Tracker 当前位姿")
    eef_zero = validate_transform(reference_eef, "末端参考位姿")
    basis = np.asarray(mapping_basis, dtype=np.float64)
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
        raise ValueError("固定映射轴向必须是有限的 3x3 矩阵")
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1.0e-6):
        raise ValueError("固定映射轴向必须是正交矩阵")
    if not np.isclose(np.linalg.det(basis), 1.0, atol=1.0e-6):
        raise ValueError("固定映射轴向必须为右手旋转")
    translation_gain = validate_scale(translation_scale, "translation_scale")
    rotation_gain = validate_scale(rotation_scale, "rotation_scale")

    target = np.eye(4, dtype=np.float64)
    tracker_translation = tracker_now[:3, 3] - tracker_zero[:3, 3]
    target[:3, 3] = (
        eef_zero[:3, 3]
        + basis @ (translation_gain * tracker_translation)
    )
    tracker_rotation_delta = (
        basis
        @ tracker_now[:3, :3]
        @ tracker_zero[:3, :3].T
        @ basis.T
    )
    rotation_vector = Rotation.from_matrix(tracker_rotation_delta).as_rotvec()
    target[:3, :3] = Rotation.from_rotvec(
        rotation_gain * rotation_vector
    ).as_matrix() @ eef_zero[:3, :3]
    return target


def low_pass_pose(
    previous: np.ndarray,
    current: np.ndarray,
    cutoff_hz: float,
    dt: float,
) -> np.ndarray:
    """对位姿执行一阶低通，其中姿态采用最短旋转向量插值。"""
    old_pose = validate_transform(previous, "上一 Tracker 位姿")
    new_pose = validate_transform(current, "当前 Tracker 位姿")
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
        raise ValueError("滤波截止频率必须为正有限数")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("滤波周期必须为正有限数")
    alpha = float(1.0 - np.exp(-2.0 * np.pi * cutoff_hz * dt))
    filtered = np.eye(4, dtype=np.float64)
    filtered[:3, 3] = (
        old_pose[:3, 3]
        + alpha * (new_pose[:3, 3] - old_pose[:3, 3])
    )
    delta = new_pose[:3, :3] @ old_pose[:3, :3].T
    filtered[:3, :3] = Rotation.from_rotvec(
        alpha * Rotation.from_matrix(delta).as_rotvec()
    ).as_matrix() @ old_pose[:3, :3]
    return filtered


def advance_joint_command(
    current: np.ndarray,
    velocity: np.ndarray,
    target: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limit: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """在速度和加速度约束下推进一次七轴关节目标。"""
    current_position = np.asarray(current, dtype=np.float64).reshape(7)
    current_velocity = np.asarray(velocity, dtype=np.float64).reshape(7)
    target_position = np.asarray(target, dtype=np.float64).reshape(7)
    maximum_velocity = np.asarray(velocity_limits, dtype=np.float64).reshape(7)
    arrays = (
        current_position,
        current_velocity,
        target_position,
        maximum_velocity,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("关节平滑输入必须为有限数值")
    if np.any(maximum_velocity <= 0.0):
        raise ValueError("关节速度上限必须为正数")
    if not np.isfinite(acceleration_limit) or acceleration_limit <= 0.0:
        raise ValueError("关节加速度上限必须为正有限数")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("控制周期必须为正有限数")

    error = target_position - current_position
    braking_speed = np.sqrt(2.0 * acceleration_limit * np.abs(error))
    desired_velocity = np.sign(error) * np.minimum(
        maximum_velocity, braking_speed
    )
    maximum_change = acceleration_limit * dt
    next_velocity = np.clip(
        desired_velocity,
        current_velocity - maximum_change,
        current_velocity + maximum_change,
    )
    next_position = current_position + next_velocity * dt
    crossed = error * (target_position - next_position) <= 0.0
    can_stop = np.abs(current_velocity) <= maximum_change
    settled = crossed & can_stop
    next_position[settled] = target_position[settled]
    next_velocity[settled] = 0.0
    return next_position, next_velocity


@dataclass(frozen=True)
class PoseValidation:
    """描述一次 Tracker 位姿连续性检查的结果。"""

    accepted: bool
    ready: bool
    relocalized: bool
    reason: str = ""


class PoseStreamValidator:
    """拒绝单帧跳变，并在连续稳定帧后采用新的跟踪原点。"""

    def __init__(
        self,
        position_jump_m: float = 0.3,
        rotation_jump_deg: float = 45.0,
        recovery_samples: int = 3,
    ) -> None:
        """保存连续性阈值并初始化输入状态。"""
        if position_jump_m <= 0.0 or not np.isfinite(position_jump_m):
            raise ValueError("位置跳变阈值必须为正有限数")
        if rotation_jump_deg <= 0.0 or not np.isfinite(rotation_jump_deg):
            raise ValueError("姿态跳变阈值必须为正有限数")
        if recovery_samples < 1:
            raise ValueError("恢复帧数必须大于零")
        self.position_jump_m = float(position_jump_m)
        self.rotation_jump_rad = float(np.deg2rad(rotation_jump_deg))
        self.recovery_samples = int(recovery_samples)
        self.last_pose: Optional[np.ndarray] = None
        self.candidate_pose: Optional[np.ndarray] = None
        self.stable_samples = 0

    def invalidate(self) -> None:
        """清除输入就绪状态，并保留最近有效位置用于检测恢复跳变。"""
        self.candidate_pose = None
        self.stable_samples = 0

    def _pose_step(
        self, first: np.ndarray, second: np.ndarray
    ) -> tuple[float, float]:
        """返回同一 odom 坐标系中两帧的位置差（米）和旋转差（弧度）。"""
        position_step = float(
            np.linalg.norm(second[:3, 3] - first[:3, 3])
        )
        rotation_step = float(
            np.linalg.norm(
                Rotation.from_matrix(
                    second[:3, :3] @ first[:3, :3].T
                ).as_rotvec()
            )
        )
        return position_step, rotation_step

    def _within_threshold(self, first: np.ndarray, second: np.ndarray) -> bool:
        """判断两帧之间的位置和姿态变化是否均未越限。"""
        position_step, rotation_step = self._pose_step(first, second)
        return (
            position_step <= self.position_jump_m
            and rotation_step <= self.rotation_jump_rad
        )

    def update(self, pose: np.ndarray) -> PoseValidation:
        """校验一帧位姿，并在必要时确认新的稳定跟踪位置。"""
        current = validate_transform(pose, "Tracker 位姿")
        if self.last_pose is None:
            self.last_pose = current
            self.stable_samples = 1
            return PoseValidation(
                True, self.stable_samples >= self.recovery_samples, False
            )

        if self.candidate_pose is None and self._within_threshold(
            self.last_pose, current
        ):
            self.last_pose = current
            self.stable_samples = min(
                self.recovery_samples, self.stable_samples + 1
            )
            return PoseValidation(
                True, self.stable_samples >= self.recovery_samples, False
            )

        if self.candidate_pose is None or not self._within_threshold(
            self.candidate_pose, current
        ):
            # 保留相对上一有效位姿的实际跳变量，便于现场区分平移与旋转异常。
            position_step, rotation_step = self._pose_step(
                self.last_pose, current
            )
            self.candidate_pose = current
            self.stable_samples = 1
            return PoseValidation(
                False,
                False,
                False,
                "Tracker 位姿发生跳变："
                f"位置差={position_step:.3f} m（阈值 {self.position_jump_m:.3f} m），"
                f"旋转差={np.rad2deg(rotation_step):.1f}°"
                f"（阈值 {np.rad2deg(self.rotation_jump_rad):.1f}°）",
            )

        self.candidate_pose = current
        self.stable_samples += 1
        if self.stable_samples < self.recovery_samples:
            return PoseValidation(False, False, False, "等待 Tracker 跟踪稳定")

        # 坏帧后回到原有轨迹时保留标定；只有稳定的新位置才视为重定位。
        relocalized = not self._within_threshold(self.last_pose, current)
        self.last_pose = current
        self.candidate_pose = None
        self.stable_samples = self.recovery_samples
        reason = (
            "Tracker 已采用新的稳定位置"
            if relocalized
            else "Tracker 已恢复原有轨迹"
        )
        return PoseValidation(True, True, relocalized, reason)
