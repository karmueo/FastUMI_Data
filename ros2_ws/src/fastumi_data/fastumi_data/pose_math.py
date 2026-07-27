"""提供采集转换和 RM75 部署共享的 SE(3) 位姿运算。"""

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def normalize_quaternion(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """归一化 xyzw 四元数并拒绝退化输入。

    Args:
        quaternion_xyzw: 四元数数组。

    Returns:
        单位四元数。

    Raises:
        ValueError: 输入不是四个有限数值或范数过小时抛出。
    """
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("四元数必须包含四个有限数值")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise ValueError("四元数范数过小")
    return quaternion / norm


def pose_to_matrix(
    position_m: np.ndarray, quaternion_xyzw: np.ndarray
) -> np.ndarray:
    """将米和 xyzw 四元数组合为 4×4 齐次变换。"""
    position = np.asarray(position_m, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("位置必须包含三个有限数值")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(
        normalize_quaternion(quaternion_xyzw)
    ).as_matrix()
    transform[:3, 3] = position
    return transform


def matrix_to_pose(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将 4×4 齐次变换拆分为位置和 xyzw 单位四元数。"""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("齐次变换必须是包含有限数值的 4x4 矩阵")
    position = matrix[:3, 3].copy()
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return position, normalize_quaternion(quaternion)


def relative_transform(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    """计算当前位姿在参考位姿坐标系中的相对变换。"""
    return np.linalg.inv(reference) @ current


def interpolate_pose(
    first_position: np.ndarray,
    first_quaternion: np.ndarray,
    second_position: np.ndarray,
    second_quaternion: np.ndarray,
    ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """对两个位姿执行线性位置插值和最短路径姿态 SLERP。"""
    interpolation_ratio = float(np.clip(ratio, 0.0, 1.0))
    position = (
        (1.0 - interpolation_ratio) * np.asarray(first_position)
        + interpolation_ratio * np.asarray(second_position)
    )
    first = normalize_quaternion(first_quaternion)
    second = normalize_quaternion(second_quaternion)
    if float(np.dot(first, second)) < 0.0:
        second = -second
    rotations = Rotation.from_quat(np.vstack((first, second)))
    slerp = Slerp([0.0, 1.0], rotations)
    quaternion = slerp([interpolation_ratio]).as_quat()[0]
    return position.astype(np.float64), normalize_quaternion(quaternion)


def transform_series_to_episode_frame(
    positions_m: np.ndarray,
    quaternions_xyzw: np.ndarray,
    tracker_to_tcp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """把世界下 Tracker 位姿转换为 episode 起始 TCP 相对位姿序列。"""
    tcp_transforms = []
    for position, quaternion in zip(positions_m, quaternions_xyzw):
        world_to_tracker = pose_to_matrix(position, quaternion)
        tcp_transforms.append(world_to_tracker @ tracker_to_tcp)
    reference = tcp_transforms[0]
    relative_positions = []
    relative_quaternions = []
    for transform in tcp_transforms:
        position, quaternion = matrix_to_pose(
            relative_transform(reference, transform)
        )
        relative_positions.append(position)
        relative_quaternions.append(quaternion)
    return (
        np.asarray(relative_positions, dtype=np.float64),
        np.asarray(relative_quaternions, dtype=np.float64),
    )
