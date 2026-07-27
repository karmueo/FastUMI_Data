"""求解 RM75 基座、Vive 世界和 Tracker 到 UMI TCP 的 6D 外参。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from fastumi_data.pose_math import matrix_to_pose, pose_to_matrix


@dataclass(frozen=True)
class HandEyeResult:
    """保存 robot-world/hand-eye 标定结果和留出集误差。"""

    base_to_vive: np.ndarray
    tracker_to_tcp: np.ndarray
    translation_rmse_mm: float
    rotation_rmse_deg: float
    training_sample_count: int
    validation_sample_count: int


def _vector_to_transform(parameters: np.ndarray) -> np.ndarray:
    """把旋转向量和平移组成齐次变换。"""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    transform[:3, 3] = parameters[3:6]
    return transform


def _transform_to_vector(transform: np.ndarray) -> np.ndarray:
    """把齐次变换转换为旋转向量和平移。"""
    return np.concatenate(
        (
            Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
            transform[:3, 3],
        )
    )


def _pose_errors(
    expected: Sequence[np.ndarray],
    vive_tracker: Sequence[np.ndarray],
    base_to_vive: np.ndarray,
    tracker_to_tcp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """计算一组配对姿态的平移和旋转绝对误差。"""
    translation_errors = []
    rotation_errors = []
    for expected_pose, tracker_pose in zip(expected, vive_tracker):
        predicted = base_to_vive @ tracker_pose @ tracker_to_tcp
        error = np.linalg.inv(expected_pose) @ predicted
        translation_errors.append(np.linalg.norm(error[:3, 3]))
        rotation_errors.append(
            np.linalg.norm(Rotation.from_matrix(error[:3, :3]).as_rotvec())
        )
    return np.asarray(translation_errors), np.asarray(rotation_errors)


def solve_robot_world_hand_eye(
    base_tcp_poses: Sequence[np.ndarray],
    vive_tracker_poses: Sequence[np.ndarray],
    initial_tracker_to_tcp: np.ndarray | None = None,
    validation_fraction: float = 0.2,
) -> HandEyeResult:
    """联合求解 ``T_base_vive`` 和 ``T_tracker_tcp``。

    Args:
        base_tcp_poses: RM75 基座下的公共 TCP 位姿。
        vive_tracker_poses: 相同时刻 Vive 世界下的 Tracker 位姿。
        initial_tracker_to_tcp: 可选 CAD 初值。
        validation_fraction: 按输入尾部留作验证的比例。

    Returns:
        标定变换与留出集 RMSE。
    """
    base_poses = [np.asarray(pose, dtype=np.float64) for pose in base_tcp_poses]
    tracker_poses = [
        np.asarray(pose, dtype=np.float64) for pose in vive_tracker_poses
    ]
    if len(base_poses) != len(tracker_poses) or len(base_poses) < 8:
        raise ValueError("配对姿态数量必须一致且至少为 8")
    validation_count = max(2, int(round(len(base_poses) * validation_fraction)))
    training_count = len(base_poses) - validation_count
    if training_count < 6:
        raise ValueError("训练姿态至少需要 6 组")
    tracker_to_tcp_initial = (
        np.eye(4, dtype=np.float64)
        if initial_tracker_to_tcp is None
        else np.asarray(initial_tracker_to_tcp, dtype=np.float64)
    )
    base_to_vive_initial = (
        base_poses[0]
        @ np.linalg.inv(tracker_poses[0] @ tracker_to_tcp_initial)
    )
    initial_parameters = np.concatenate(
        (
            _transform_to_vector(base_to_vive_initial),
            _transform_to_vector(tracker_to_tcp_initial),
        )
    )
    translation_scale_m = 0.01
    rotation_scale_rad = np.deg2rad(5.0)

    def residual(parameters: np.ndarray) -> np.ndarray:
        """计算训练集上的缩放 SE(3) 残差。"""
        base_to_vive = _vector_to_transform(parameters[:6])
        tracker_to_tcp = _vector_to_transform(parameters[6:])
        residuals = []
        for expected_pose, tracker_pose in zip(
            base_poses[:training_count], tracker_poses[:training_count]
        ):
            predicted = base_to_vive @ tracker_pose @ tracker_to_tcp
            error = np.linalg.inv(expected_pose) @ predicted
            residuals.extend(error[:3, 3] / translation_scale_m)
            residuals.extend(
                Rotation.from_matrix(error[:3, :3]).as_rotvec()
                / rotation_scale_rad
            )
        return np.asarray(residuals, dtype=np.float64)

    optimized = least_squares(
        residual,
        initial_parameters,
        method="trf",
        loss="soft_l1",
        max_nfev=5000,
    )
    if not optimized.success:
        raise RuntimeError(f"hand-eye 优化失败: {optimized.message}")
    base_to_vive = _vector_to_transform(optimized.x[:6])
    tracker_to_tcp = _vector_to_transform(optimized.x[6:])
    validation_translation, validation_rotation = _pose_errors(
        base_poses[training_count:],
        tracker_poses[training_count:],
        base_to_vive,
        tracker_to_tcp,
    )
    return HandEyeResult(
        base_to_vive=base_to_vive,
        tracker_to_tcp=tracker_to_tcp,
        translation_rmse_mm=float(
            np.sqrt(np.mean(np.square(validation_translation))) * 1000.0
        ),
        rotation_rmse_deg=float(
            np.rad2deg(np.sqrt(np.mean(np.square(validation_rotation))))
        ),
        training_sample_count=training_count,
        validation_sample_count=validation_count,
    )


def solve_pivot_translation(
    vive_tracker_poses: Sequence[np.ndarray],
) -> Tuple[np.ndarray, float]:
    """通过固定 TCP 点、多方向旋转求解 Tracker 到 TCP 的平移。

    Returns:
        Tracker 坐标系中的 TCP 平移和固定点拟合 RMSE（毫米）。
    """
    poses = [np.asarray(pose, dtype=np.float64) for pose in vive_tracker_poses]
    if len(poses) < 8:
        raise ValueError("pivot 标定至少需要 8 个不同姿态")
    system_rows = []
    right_hand = []
    for pose in poses:
        system_rows.append(np.hstack((pose[:3, :3], -np.eye(3))))
        right_hand.append(-pose[:3, 3])
    solution, _, rank, _ = np.linalg.lstsq(
        np.vstack(system_rows), np.concatenate(right_hand), rcond=None
    )
    if rank < 6:
        raise ValueError("pivot 姿态变化不足，无法唯一求解")
    tracker_to_tcp_translation = solution[:3]
    fixed_point = solution[3:]
    errors = [
        np.linalg.norm(
            pose[:3, 3]
            + pose[:3, :3] @ tracker_to_tcp_translation
            - fixed_point
        )
        for pose in poses
    ]
    rmse_mm = float(np.sqrt(np.mean(np.square(errors))) * 1000.0)
    return tracker_to_tcp_translation, rmse_mm


def transform_to_document(transform: np.ndarray) -> dict:
    """把变换转换为 YAML 可序列化的位置和四元数映射。"""
    position, quaternion = matrix_to_pose(transform)
    return {
        "translation_m": [float(value) for value in position],
        "quaternion_xyzw": [float(value) for value in quaternion],
    }


def document_to_transform(document: dict) -> np.ndarray:
    """从 YAML 位姿映射构造齐次变换。"""
    return pose_to_matrix(
        np.asarray(document["translation_m"], dtype=np.float64),
        np.asarray(document["quaternion_xyzw"], dtype=np.float64),
    )
