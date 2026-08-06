"""筛选双 ArUco 单帧结果并稳健聚合固定 Tracker→TCP 外参。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from fastumi_data.aruco_tcp_estimator import FrameTcpEstimate


@dataclass(frozen=True)
class CalibrationThresholds:
    """保存双 ArUco 单帧和跨帧聚合质量门限。"""

    minimum_valid_frames: int = 30
    max_reprojection_rmse_px: float = 1.5
    max_distance_error_m: float = 0.005
    max_candidate_difference_m: float = 0.005
    max_translation_p95_m: float = 0.003
    max_rotation_p95_deg: float = 2.0

    def __post_init__(self) -> None:
        """校验帧数和全部数值门限。"""
        if self.minimum_valid_frames <= 0:
            raise ValueError("minimum_valid_frames 必须为正整数")
        values = (
            self.max_reprojection_rmse_px,
            self.max_distance_error_m,
            self.max_candidate_difference_m,
            self.max_translation_p95_m,
            self.max_rotation_p95_deg,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("标定质量门限必须是有限正数")

    @property
    def minimum_frames(self) -> int:
        """返回 minimum_valid_frames 的兼容别名。"""
        return self.minimum_valid_frames

    @property
    def max_rmse_px(self) -> float:
        """返回最大单标签重投影 RMSE。"""
        return self.max_reprojection_rmse_px


@dataclass(frozen=True)
class ArucoTcpCalibrationResult:
    """保存固定相机→TCP、Tracker→TCP 和完整质量诊断。"""

    accepted: bool
    camera_from_tcp: np.ndarray | None
    tracker_from_tcp: np.ndarray | None
    valid_frames: tuple[FrameTcpEstimate, ...]
    metrics: Mapping[str, float | int | bool]
    failures: tuple[str, ...]
    rejection_histogram: Mapping[str, int]

    @property
    def accepted_frames(self) -> tuple[FrameTcpEstimate, ...]:
        """返回通过单帧和稳健离群门的帧。"""
        return self.valid_frames


def _validated_transform(transform: np.ndarray, description: str) -> np.ndarray:
    """校验有限右手 4×4 刚体变换。"""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{description} 必须是有限 4x4 矩阵")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise ValueError(f"{description} 齐次末行无效")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise ValueError(f"{description} 旋转不正交")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise ValueError(f"{description} 旋转不是右手系")
    return np.array(matrix, dtype=np.float64, copy=True)


def _quaternion_average(rotations: Sequence[np.ndarray]) -> np.ndarray:
    """用符号一致的最大特征向量计算旋转均值。"""
    if not rotations:
        raise ValueError("旋转样本不能为空")
    quaternions = [Rotation.from_matrix(matrix).as_quat() for matrix in rotations]
    reference = quaternions[0]
    aligned = [
        quaternion if float(np.dot(quaternion, reference)) >= 0.0 else -quaternion
        for quaternion in quaternions
    ]
    scatter = np.zeros((4, 4), dtype=np.float64)
    for quaternion in aligned:
        scatter += np.outer(quaternion, quaternion)
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    quaternion = eigenvectors[:, int(np.argmax(eigenvalues))]
    quaternion /= np.linalg.norm(quaternion)
    if float(np.dot(quaternion, reference)) < 0.0:
        quaternion = -quaternion
    return Rotation.from_quat(quaternion).as_matrix()


def _huber_location(points: np.ndarray) -> np.ndarray:
    """以逐轴中位数为初值执行有限次 Huber 加权位置估计。"""
    location = np.median(points, axis=0)
    for _ in range(8):
        residuals = np.linalg.norm(points - location, axis=1)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        delta = max(0.0005, median + 1.4826 * max(mad, 1.0e-12))
        weights = np.ones(len(points), dtype=np.float64)
        nonzero = residuals > delta
        weights[nonzero] = delta / residuals[nonzero]
        updated = np.average(points, axis=0, weights=weights)
        if np.linalg.norm(updated - location) < 1.0e-10:
            break
        location = updated
    return location


def _robust_inlier_mask(
    translations: np.ndarray,
    rotations: Sequence[np.ndarray],
    thresholds: CalibrationThresholds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """基于 MAD 和固定质量门迭代剔除平移/旋转离群。"""
    translation_center = _huber_location(translations)
    rotation_center = _quaternion_average(rotations)
    mask = np.ones(len(translations), dtype=bool)
    for _ in range(5):
        translation_residuals = np.linalg.norm(
            translations - translation_center, axis=1
        )
        rotation_residuals = np.asarray(
            [
                np.rad2deg(
                    np.linalg.norm(
                        Rotation.from_matrix(
                            rotation_center.T @ rotation
                        ).as_rotvec()
                    )
                )
                for rotation in rotations
            ],
            dtype=np.float64,
        )
        translation_median = float(np.median(translation_residuals))
        translation_mad = float(
            np.median(np.abs(translation_residuals - translation_median))
        )
        rotation_median = float(np.median(rotation_residuals))
        rotation_mad = float(
            np.median(np.abs(rotation_residuals - rotation_median))
        )
        translation_cut = max(
            thresholds.max_translation_p95_m,
            translation_median + 6.0 * 1.4826 * max(translation_mad, 1.0e-12),
        )
        rotation_cut = max(
            thresholds.max_rotation_p95_deg,
            rotation_median + 6.0 * 1.4826 * max(rotation_mad, 1.0e-12),
        )
        updated_mask = (translation_residuals <= translation_cut) & (
            rotation_residuals <= rotation_cut
        )
        if not np.any(updated_mask):
            break
        translation_center = _huber_location(translations[updated_mask])
        rotation_center = _quaternion_average(
            [rotations[index] for index in np.flatnonzero(updated_mask)]
        )
        if np.array_equal(mask, updated_mask):
            mask = updated_mask
            break
        mask = updated_mask
    return mask, translation_center, rotation_center


def calibrate_frames(
    frames: Sequence[FrameTcpEstimate],
    tracker_from_camera: np.ndarray,
    thresholds: CalibrationThresholds,
) -> ArucoTcpCalibrationResult:
    """筛选单帧结果、稳健聚合并组合 ``^tracker T_camera``。"""
    source_transform = _validated_transform(
        tracker_from_camera, "tracker_from_camera"
    )
    rejection_histogram: dict[str, int] = {}
    accepted_candidates: list[FrameTcpEstimate] = []
    for frame in frames:
        reason = None
        rmse_values = (
            frame.tag0_reprojection_rmse_px,
            frame.tag1_reprojection_rmse_px,
        )
        if not all(
            np.isfinite(value) and value >= 0.0 for value in rmse_values
        ) or max(rmse_values) > thresholds.max_reprojection_rmse_px:
            reason = "重投影 RMSE 超过门限"
        elif (
            not np.isfinite(frame.measured_tag_distance_m)
            or not np.isfinite(frame.expected_tag_distance_m)
            or abs(
                frame.measured_tag_distance_m - frame.expected_tag_distance_m
            )
            > thresholds.max_distance_error_m
        ):
            reason = "tag 中心距离模型误差超过门限"
        elif (
            not np.isfinite(frame.candidate_translation_difference_m)
            or frame.candidate_translation_difference_m
            > thresholds.max_candidate_difference_m
        ):
            reason = "TCP 候选平移差超过门限"
        else:
            try:
                _validated_transform(frame.camera_from_pair, "camera_from_pair")
                _validated_transform(frame.camera_from_tcp, "camera_from_tcp")
            except ValueError as error:
                reason = str(error)
        if reason is None:
            accepted_candidates.append(frame)
        else:
            rejection_histogram[reason] = rejection_histogram.get(reason, 0) + 1

    failures: list[str] = []
    if not accepted_candidates:
        failures.extend(sorted(rejection_histogram))
        failures.append("没有通过单帧质量门的 ArUco 帧")
        metrics = {
            "candidate_frames": len(frames),
            "valid_frames": 0,
            "rejected_frames": len(frames),
            "translation_p95_mm": float("inf"),
            "rotation_p95_deg": float("inf"),
        }
        return ArucoTcpCalibrationResult(
            False, None, None, (), metrics, tuple(failures), rejection_histogram
        )

    translations = np.asarray(
        [frame.camera_from_tcp[:3, 3] for frame in accepted_candidates],
        dtype=np.float64,
    )
    rotations = [
        np.array(frame.camera_from_tcp[:3, :3], dtype=np.float64, copy=True)
        for frame in accepted_candidates
    ]
    inlier_mask, translation_center, rotation_center = _robust_inlier_mask(
        translations, rotations, thresholds
    )
    inlier_frames = tuple(
        frame
        for index, frame in enumerate(accepted_candidates)
        if bool(inlier_mask[index])
    )
    outlier_count = len(accepted_candidates) - len(inlier_frames)
    if outlier_count:
        rejection_histogram["稳健聚合离群"] = outlier_count
    translation_residuals = np.linalg.norm(
        translations[inlier_mask] - translation_center, axis=1
    )
    rotation_residuals = np.asarray(
        [
            np.rad2deg(
                np.linalg.norm(
                    Rotation.from_matrix(
                        rotation_center.T @ rotations[index]
                    ).as_rotvec()
                )
            )
            for index in np.flatnonzero(inlier_mask)
        ],
        dtype=np.float64,
    )
    translation_p95_m = float(np.percentile(translation_residuals, 95.0))
    rotation_p95_deg = float(np.percentile(rotation_residuals, 95.0))
    metrics: dict[str, float | int | bool] = {
        "candidate_frames": len(frames),
        "single_frame_valid_frames": len(accepted_candidates),
        "valid_frames": len(inlier_frames),
        "rejected_frames": len(frames) - len(inlier_frames),
        "translation_p95_m": translation_p95_m,
        "translation_p95_mm": translation_p95_m * 1000.0,
        "rotation_p95_deg": rotation_p95_deg,
        "translation_rmse_mm": float(
            np.sqrt(np.mean(translation_residuals**2)) * 1000.0
        ),
        "rotation_rmse_deg": float(
            np.sqrt(np.mean(rotation_residuals**2))
        ),
        "max_reprojection_rmse_px": float(
            max(
                max(frame.tag0_reprojection_rmse_px, frame.tag1_reprojection_rmse_px)
                for frame in inlier_frames
            )
        )
        if inlier_frames
        else float("inf"),
    }
    if len(inlier_frames) < thresholds.minimum_valid_frames:
        failures.append(
            f"有效帧 {len(inlier_frames)} 少于 {thresholds.minimum_valid_frames}"
        )
    if translation_p95_m > thresholds.max_translation_p95_m:
        failures.append(
            "平移 P95 "
            f"{translation_p95_m * 1000.0:.3f} mm 超过 "
            f"{thresholds.max_translation_p95_m * 1000.0:.3f} mm"
        )
    if rotation_p95_deg > thresholds.max_rotation_p95_deg:
        failures.append(
            f"旋转 P95 {rotation_p95_deg:.3f}° 超过 "
            f"{thresholds.max_rotation_p95_deg:.3f}°"
        )
    camera_from_tcp = np.eye(4, dtype=np.float64)
    camera_from_tcp[:3, :3] = rotation_center
    camera_from_tcp[:3, 3] = translation_center
    tracker_from_tcp = source_transform @ camera_from_tcp
    accepted = not failures
    if not accepted:
        tracker_from_tcp = None
    return ArucoTcpCalibrationResult(
        accepted=accepted,
        camera_from_tcp=camera_from_tcp,
        tracker_from_tcp=tracker_from_tcp,
        valid_frames=inlier_frames,
        metrics=metrics,
        failures=tuple(failures),
        rejection_histogram=rejection_histogram,
    )
