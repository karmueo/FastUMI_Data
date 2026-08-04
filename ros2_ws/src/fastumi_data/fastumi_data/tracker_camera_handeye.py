"""运行 OpenCV 多算法 Hand-Eye 初值求解并评估固定板闭环。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


class HandEyeEstimationError(RuntimeError):
    """表示所有 Hand-Eye 算法均未生成有效候选。"""


@dataclass(frozen=True)
class HandEyeCandidate:
    """保存一个算法的 ``^tracker T_camera`` 与固定板闭环指标。"""

    method: str
    tracker_from_camera: np.ndarray
    world_from_board: np.ndarray
    translation_rmse_mm: float
    rotation_rmse_deg: float


def _checked_transform(transform: np.ndarray, description: str) -> np.ndarray:
    """把输入校验为有限、右手且末行合法的 4×4 刚体变换。"""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{description} 必须是有限 4x4 矩阵")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise ValueError(f"{description} 的齐次末行无效")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError(f"{description} 的旋转矩阵不正交")
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError(f"{description} 的旋转矩阵不是右手系")
    return matrix


def board_transforms(
    world_from_tracker: Sequence[np.ndarray],
    camera_from_board: Sequence[np.ndarray],
    tracker_from_camera: np.ndarray,
) -> list[np.ndarray]:
    """按固定板方程计算各帧 ``^world T_board``。"""
    if len(world_from_tracker) != len(camera_from_board):
        raise ValueError("Tracker 与标定板位姿数量必须一致")
    extrinsic = _checked_transform(
        tracker_from_camera, "tracker_from_camera"
    )
    return [
        _checked_transform(world_tracker, "world_from_tracker")
        @ extrinsic
        @ _checked_transform(camera_board, "camera_from_board")
        for world_tracker, camera_board in zip(
            world_from_tracker, camera_from_board
        )
    ]


def board_closure_errors(
    transforms: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回固定板相对均值的逐帧平移毫米、旋转角度和均值变换。"""
    if not transforms:
        raise ValueError("固定板闭环至少需要一个变换")
    matrices = [
        _checked_transform(transform, "world_from_board")
        for transform in transforms
    ]
    mean_transform = np.eye(4, dtype=np.float64)
    mean_transform[:3, :3] = Rotation.from_matrix(
        np.stack([matrix[:3, :3] for matrix in matrices])
    ).mean().as_matrix()
    mean_transform[:3, 3] = np.mean(
        [matrix[:3, 3] for matrix in matrices], axis=0
    )
    translation_errors = []
    rotation_errors = []
    for transform in matrices:
        relative = np.linalg.inv(mean_transform) @ transform
        translation_errors.append(np.linalg.norm(relative[:3, 3]) * 1000.0)
        rotation_errors.append(
            np.rad2deg(
                np.linalg.norm(
                    Rotation.from_matrix(relative[:3, :3]).as_rotvec()
                )
            )
        )
    return (
        np.asarray(translation_errors, dtype=np.float64),
        np.asarray(rotation_errors, dtype=np.float64),
        mean_transform,
    )


def solve_handeye_candidates(
    world_from_tracker: Sequence[np.ndarray],
    camera_from_board: Sequence[np.ndarray],
) -> list[HandEyeCandidate]:
    """运行五种 OpenCV Hand-Eye 算法并返回所有有限右手候选。

    OpenCV 输入分别为 ``^world T_tracker`` 和 ``^camera T_board``，
    返回的 camera-to-gripper 直接对应 ``^tracker T_camera``。
    """
    if len(world_from_tracker) != len(camera_from_board):
        raise ValueError("Tracker 与标定板位姿数量必须一致")
    if len(world_from_tracker) < 4:
        raise ValueError("Hand-Eye 求解至少需要 4 组配对位姿")
    tracker_poses = [
        _checked_transform(transform, "world_from_tracker")
        for transform in world_from_tracker
    ]
    board_poses = [
        _checked_transform(transform, "camera_from_board")
        for transform in camera_from_board
    ]
    tracker_rotations = [pose[:3, :3] for pose in tracker_poses]
    tracker_translations = [pose[:3, 3].reshape(3, 1) for pose in tracker_poses]
    board_rotations = [pose[:3, :3] for pose in board_poses]
    board_translations = [pose[:3, 3].reshape(3, 1) for pose in board_poses]
    candidates = []
    failures = []
    for method_name, method_id in HAND_EYE_METHODS.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                tracker_rotations,
                tracker_translations,
                board_rotations,
                board_translations,
                method=method_id,
            )
            tracker_from_camera = np.eye(4, dtype=np.float64)
            tracker_from_camera[:3, :3] = np.asarray(
                rotation, dtype=np.float64
            ).reshape(3, 3)
            tracker_from_camera[:3, 3] = np.asarray(
                translation, dtype=np.float64
            ).reshape(3)
            tracker_from_camera = _checked_transform(
                tracker_from_camera,
                f"{method_name} tracker_from_camera",
            )
            closures = board_transforms(
                tracker_poses, board_poses, tracker_from_camera
            )
            translation_errors, rotation_errors, mean_board = (
                board_closure_errors(closures)
            )
            candidates.append(
                HandEyeCandidate(
                    method=method_name,
                    tracker_from_camera=tracker_from_camera,
                    world_from_board=mean_board,
                    translation_rmse_mm=float(
                        np.sqrt(np.mean(np.square(translation_errors)))
                    ),
                    rotation_rmse_deg=float(
                        np.sqrt(np.mean(np.square(rotation_errors)))
                    ),
                )
            )
        except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
            failures.append(f"{method_name}: {error}")
    if not candidates:
        detail = "; ".join(failures)
        raise HandEyeEstimationError(
            f"所有 Hand-Eye 算法均失败: {detail}"
        )
    return candidates


def select_handeye_seed(
    candidates: Sequence[HandEyeCandidate],
) -> HandEyeCandidate:
    """按固定板平移 RMSE、旋转 RMSE 依次选择最佳初值。"""
    if not candidates:
        raise HandEyeEstimationError("没有可选择的 Hand-Eye 候选")
    finite_candidates = [
        candidate
        for candidate in candidates
        if np.isfinite(candidate.translation_rmse_mm)
        and np.isfinite(candidate.rotation_rmse_deg)
    ]
    if not finite_candidates:
        raise HandEyeEstimationError("Hand-Eye 候选闭环指标均为非有限数")
    return min(
        finite_candidates,
        key=lambda candidate: (
            candidate.translation_rmse_mm,
            candidate.rotation_rmse_deg,
            candidate.method,
        ),
    )
