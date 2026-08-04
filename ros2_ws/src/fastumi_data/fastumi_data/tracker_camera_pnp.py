"""使用原始鱼眼 AprilGrid 角点估计单帧 ``^camera T_board``。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from fastumi_data.tracker_camera_config import FisheyeCameraModel
from fastumi_data.tracker_camera_detection import AprilGridObservation


class PoseEstimationError(RuntimeError):
    """表示当前 AprilGrid 观测无法产生有效正深度位姿。"""


@dataclass(frozen=True)
class BoardPoseEstimate:
    """保存 ``^camera T_board`` 和原始鱼眼像素质量指标。"""

    camera_from_board: np.ndarray
    median_error_px: float
    p95_error_px: float
    positive_depth: bool


def _checked_points(
    object_points_m: np.ndarray, image_points_px: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """校验二维三维点数量、形状和有限性。"""
    object_points = np.asarray(object_points_m, dtype=np.float64)
    image_points = np.asarray(image_points_px, dtype=np.float64)
    if object_points.ndim != 2 or object_points.shape[1:] != (3,):
        raise PoseEstimationError("目标点必须是 Nx3 数组")
    if image_points.ndim != 2 or image_points.shape[1:] != (2,):
        raise PoseEstimationError("图像点必须是 Nx2 数组")
    if len(object_points) != len(image_points):
        raise PoseEstimationError("二维三维点数量必须一致")
    if not np.all(np.isfinite(object_points)) or not np.all(
        np.isfinite(image_points)
    ):
        raise PoseEstimationError("二维三维角点必须是有限数值")
    return object_points, image_points


def _transform_from_vectors(
    rotation_vector: np.ndarray, translation_vector: np.ndarray
) -> np.ndarray:
    """把 OpenCV 旋转向量和平移组成刚体变换。"""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(
        np.asarray(rotation_vector, dtype=np.float64).reshape(3)
    ).as_matrix()
    transform[:3, 3] = np.asarray(
        translation_vector, dtype=np.float64
    ).reshape(3)
    return transform


def _all_points_have_positive_depth(
    object_points_m: np.ndarray, camera_from_object: np.ndarray
) -> bool:
    """检查变换后的全部目标点是否位于相机前方。"""
    camera_points = (
        camera_from_object[:3, :3] @ object_points_m.T
    ).T + camera_from_object[:3, 3]
    return bool(np.all(camera_points[:, 2] > 0.0))


def project_fisheye_points(
    object_points_m: np.ndarray,
    camera_from_object: np.ndarray,
    camera: FisheyeCameraModel,
) -> np.ndarray:
    """使用 equidistant 模型把三维点投影到原始鱼眼像素。

    Args:
        object_points_m: 目标坐标系下形状为 ``(N, 3)`` 的米制点。
        camera_from_object: ``^camera T_object`` 4×4 刚体变换。
        camera: pinhole+equidistant 相机模型。

    Returns:
        原始鱼眼图像中形状为 ``(N, 2)`` 的像素坐标。
    """
    object_points = np.asarray(object_points_m, dtype=np.float64)
    transform = np.asarray(camera_from_object, dtype=np.float64)
    if object_points.ndim != 2 or object_points.shape[1:] != (3,):
        raise ValueError("目标点必须是 Nx3 数组")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("相机变换必须是有限 4x4 矩阵")
    if not np.all(np.isfinite(object_points)):
        raise ValueError("目标点必须是有限数值")
    rotation_vector = np.ascontiguousarray(
        Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
        dtype=np.float64,
    ).reshape(3, 1)
    translation_vector = np.ascontiguousarray(
        transform[:3, 3], dtype=np.float64
    ).reshape(3, 1)
    projected, _ = cv2.fisheye.projectPoints(
        object_points.reshape(-1, 1, 3),
        rotation_vector,
        translation_vector,
        camera.k,
        camera.d.reshape(4, 1),
    )
    return projected.reshape(-1, 2)


def fisheye_reprojection_errors(
    object_points_m: np.ndarray,
    image_points_px: np.ndarray,
    camera_from_object: np.ndarray,
    camera: FisheyeCameraModel,
) -> np.ndarray:
    """返回各三维点在原始鱼眼图像中的像素欧氏误差。"""
    object_points, image_points = _checked_points(
        object_points_m, image_points_px
    )
    projected = project_fisheye_points(
        object_points, camera_from_object, camera
    )
    return np.linalg.norm(projected - image_points, axis=1)


def estimate_camera_from_board(
    observation: AprilGridObservation,
    camera: FisheyeCameraModel,
) -> BoardPoseEstimate:
    """从整板原始鱼眼角点估计 ``^camera T_board``。

    角点先去畸变至以 ``K`` 表示的虚拟针孔像素平面，再通过 IPPE 生成
    共面候选；候选以全点正深度和原始鱼眼 P95 误差筛选，最后使用 LM 精化。
    """
    object_points, image_points = _checked_points(
        observation.object_points_m, observation.image_points_px
    )
    if len(object_points) < 8:
        raise PoseEstimationError("整板 PnP 至少 8 个角点")
    undistorted = cv2.fisheye.undistortPoints(
        image_points.reshape(-1, 1, 2),
        camera.k,
        camera.d.reshape(4, 1),
        P=camera.k,
    )
    try:
        solved, rotation_vectors, translation_vectors, _ = (
            cv2.solvePnPGeneric(
                object_points.reshape(-1, 1, 3),
                undistorted,
                camera.k,
                None,
                flags=cv2.SOLVEPNP_IPPE,
            )
        )
    except cv2.error as error:
        raise PoseEstimationError(f"IPPE 求解失败: {error}") from error
    if not solved:
        raise PoseEstimationError("IPPE 没有返回候选位姿")
    candidates = []
    for rotation_vector, translation_vector in zip(
        rotation_vectors, translation_vectors
    ):
        transform = _transform_from_vectors(
            rotation_vector, translation_vector
        )
        if not _all_points_have_positive_depth(object_points, transform):
            continue
        errors = fisheye_reprojection_errors(
            object_points, image_points, transform, camera
        )
        candidates.append(
            (
                float(np.percentile(errors, 95.0)),
                rotation_vector,
                translation_vector,
            )
        )
    if not candidates:
        raise PoseEstimationError("IPPE 没有返回全部角点正深度的候选解")
    _, seed_rotation, seed_translation = min(
        candidates, key=lambda candidate: candidate[0]
    )
    try:
        refined_rotation, refined_translation = cv2.solvePnPRefineLM(
            object_points.reshape(-1, 1, 3),
            undistorted,
            camera.k,
            None,
            np.asarray(seed_rotation, dtype=np.float64).reshape(3, 1),
            np.asarray(seed_translation, dtype=np.float64).reshape(3, 1),
        )
    except cv2.error as error:
        raise PoseEstimationError(f"PnP LM 精化失败: {error}") from error
    camera_from_board = _transform_from_vectors(
        refined_rotation, refined_translation
    )
    positive_depth = _all_points_have_positive_depth(
        object_points, camera_from_board
    )
    if not positive_depth:
        raise PoseEstimationError("PnP 精化结果包含非正深度角点")
    errors = fisheye_reprojection_errors(
        object_points, image_points, camera_from_board, camera
    )
    return BoardPoseEstimate(
        camera_from_board=camera_from_board,
        median_error_px=float(np.median(errors)),
        p95_error_px=float(np.percentile(errors, 95.0)),
        positive_depth=positive_depth,
    )
