"""在 Kalibr 去畸变图像上检测双 ArUco 并融合夹爪中心 TCP。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from fastumi_data.aruco_tcp_config import ArucoTcpConfig
from fastumi_data.tracker_camera_config import FisheyeCameraModel


class FrameEstimationError(ValueError):
    """表示当前帧无法形成有效的双 ArUco TCP 估计。"""


def _readonly_array(value: np.ndarray, shape: tuple[int, ...], description: str):
    """复制、校验并冻结指定形状的有限数组。"""
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError(f"{description} 必须是 {shape} 数组")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{description} 必须是有限数值")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TagPoseEstimate:
    """保存单枚标签的 ``^camera T_tag`` 和像素重投影质量。"""

    tag_id: int
    camera_from_tag: np.ndarray
    reprojection_rmse_px: float
    corners_px: np.ndarray | None = None

    def __post_init__(self) -> None:
        """复制变换和角点，保留几何检查到帧融合阶段执行。"""
        transform = _readonly_array(
            self.camera_from_tag, (4, 4), "camera_from_tag"
        )
        if self.corners_px is None:
            corners = None
        else:
            corners = _readonly_array(self.corners_px, (4, 2), "corners_px")
        object.__setattr__(self, "camera_from_tag", transform)
        object.__setattr__(self, "corners_px", corners)

    @property
    def positive_depth(self) -> bool:
        """返回标签中心是否位于相机前方。"""
        return bool(np.isfinite(self.camera_from_tag[2, 3]) and self.camera_from_tag[2, 3] > 0.0)

    @property
    def center_camera_m(self) -> np.ndarray:
        """返回标签中心在相机坐标系中的米制位置。"""
        return self.camera_from_tag[:3, 3].copy()

    @property
    def normal_camera(self) -> np.ndarray:
        """返回标签局部 +Z 法向在相机坐标系中的方向。"""
        return self.camera_from_tag[:3, :3] @ np.asarray([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class FrameTcpEstimate:
    """保存单帧 pair、TCP 候选、融合结果和质量诊断。"""

    timestamp_ns: int
    camera_from_pair: np.ndarray
    camera_from_tcp: np.ndarray
    tcp_from_tag0_m: np.ndarray
    tcp_from_tag1_m: np.ndarray
    fused_tcp_position_m: np.ndarray
    measured_tag_distance_m: float
    expected_tag_distance_m: float
    candidate_translation_difference_m: float
    tag0_reprojection_rmse_px: float
    tag1_reprojection_rmse_px: float
    tag0: TagPoseEstimate
    tag1: TagPoseEstimate

    def __post_init__(self) -> None:
        """复制并冻结单帧输出中的全部数组。"""
        object.__setattr__(
            self,
            "camera_from_pair",
            _readonly_array(self.camera_from_pair, (4, 4), "camera_from_pair"),
        )
        object.__setattr__(
            self,
            "camera_from_tcp",
            _readonly_array(self.camera_from_tcp, (4, 4), "camera_from_tcp"),
        )
        for name in (
            "tcp_from_tag0_m",
            "tcp_from_tag1_m",
            "fused_tcp_position_m",
        ):
            object.__setattr__(
                self,
                name,
                _readonly_array(getattr(self, name), (3,), name),
            )

    @property
    def tcp_candidate_from_tag0_m(self) -> np.ndarray:
        """返回 ID 0 推导的 TCP 位置候选。"""
        return self.tcp_from_tag0_m

    @property
    def tcp_candidate_from_tag1_m(self) -> np.ndarray:
        """返回 ID 1 推导的 TCP 位置候选。"""
        return self.tcp_from_tag1_m

    @property
    def candidate_difference_m(self) -> float:
        """返回两路 TCP 候选的平移差。"""
        return self.candidate_translation_difference_m

    @property
    def measured_distance_m(self) -> float:
        """返回实测双标签中心距离。"""
        return self.measured_tag_distance_m

    @property
    def expected_distance_m(self) -> float:
        """返回按 openness 模型得到的期望双标签中心距离。"""
        return self.expected_tag_distance_m

    @property
    def reprojection_rmse_px(self) -> tuple[float, float]:
        """返回 ID 0 和 ID 1 的重投影 RMSE。"""
        return self.tag0_reprojection_rmse_px, self.tag1_reprojection_rmse_px


def _validate_rigid_transform(
    transform: np.ndarray, description: str
) -> np.ndarray:
    """校验有限的右手刚体变换并返回副本。"""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise FrameEstimationError(f"{description} 必须是有限 4x4 变换")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-8):
        raise FrameEstimationError(f"{description} 齐次末行无效")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6):
        raise FrameEstimationError(f"{description} 旋转不正交")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-6):
        raise FrameEstimationError(f"{description} 旋转不是右手系")
    return matrix


def estimate_tcp_from_tag_poses(
    tag0: TagPoseEstimate,
    tag1: TagPoseEstimate,
    openness: float,
    config: ArucoTcpConfig,
    timestamp_ns: int = 0,
) -> FrameTcpEstimate:
    """从双标签位姿、开度和几何配置计算单帧相机→TCP。"""
    if tag0.tag_id != config.aruco.tag0_id or tag1.tag_id != config.aruco.tag1_id:
        raise FrameEstimationError("双 ArUco ID 与配置不一致")
    try:
        expected_half_distance = config.expected_half_distance_m(openness)
    except ValueError as error:
        raise FrameEstimationError(str(error)) from error
    tag0_transform = _validate_rigid_transform(
        tag0.camera_from_tag, "ID 0 位姿"
    )
    tag1_transform = _validate_rigid_transform(
        tag1.camera_from_tag, "ID 1 位姿"
    )
    if not tag0.positive_depth or not tag1.positive_depth:
        raise FrameEstimationError("双 ArUco 位姿必须具有正深度")
    for tag in (tag0, tag1):
        if not np.isfinite(tag.reprojection_rmse_px) or tag.reprojection_rmse_px < 0.0:
            raise FrameEstimationError("ArUco 重投影误差必须是有限非负数")

    center0 = tag0_transform[:3, 3]
    center1 = tag1_transform[:3, 3]
    center_delta = center1 - center0
    measured_distance = float(np.linalg.norm(center_delta))
    if not np.isfinite(measured_distance) or measured_distance <= 1.0e-9:
        raise FrameEstimationError("双 ArUco 中心不能重合")
    y_axis = center_delta / measured_distance

    normal0 = tag0_transform[:3, :3] @ np.asarray([0.0, 0.0, 1.0])
    normal1 = tag1_transform[:3, :3] @ np.asarray([0.0, 0.0, 1.0])
    if not np.all(np.isfinite(normal0)) or not np.all(np.isfinite(normal1)):
        raise FrameEstimationError("ArUco 法向必须是有限数值")
    if np.linalg.norm(normal0) <= 1.0e-9 or np.linalg.norm(normal1) <= 1.0e-9:
        raise FrameEstimationError("ArUco 法向退化")
    normal0 /= np.linalg.norm(normal0)
    normal1 /= np.linalg.norm(normal1)
    if float(np.dot(normal0, normal1)) < 0.0:
        normal1 = -normal1
    z_axis = float(config.pair_frame.marker_normal_sign) * (normal0 + normal1)
    z_axis -= float(np.dot(z_axis, y_axis)) * y_axis
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm <= 1.0e-9:
        raise FrameEstimationError("双 ArUco 法向与 Y 轴平行，无法构造 pair")
    z_axis /= z_norm
    x_axis = np.cross(y_axis, z_axis)
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm <= 1.0e-9:
        raise FrameEstimationError("双 ArUco pair X 轴退化")
    x_axis /= x_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    pair_rotation = np.column_stack((x_axis, y_axis, z_axis))
    if not np.isclose(np.linalg.det(pair_rotation), 1.0, atol=1.0e-6):
        raise FrameEstimationError("pair 旋转不是右手系")

    camera_from_pair = np.eye(4, dtype=np.float64)
    camera_from_pair[:3, :3] = pair_rotation
    camera_from_pair[:3, 3] = (center0 + center1) / 2.0

    pair_tcp_translation = config.pair_from_tcp[:3, 3]
    tag0_offset = pair_tcp_translation + np.asarray(
        [0.0, expected_half_distance, 0.0]
    )
    tag1_offset = pair_tcp_translation + np.asarray(
        [0.0, -expected_half_distance, 0.0]
    )
    tcp_from_tag0 = center0 + pair_rotation @ tag0_offset
    tcp_from_tag1 = center1 + pair_rotation @ tag1_offset
    candidate_difference = float(
        np.linalg.norm(tcp_from_tag0 - tcp_from_tag1)
    )
    weights = np.asarray(
        [
            1.0 / max(float(tag0.reprojection_rmse_px), 0.05) ** 2,
            1.0 / max(float(tag1.reprojection_rmse_px), 0.05) ** 2,
        ],
        dtype=np.float64,
    )
    fused_position = (weights[0] * tcp_from_tag0 + weights[1] * tcp_from_tag1) / np.sum(weights)
    camera_from_tcp = np.eye(4, dtype=np.float64)
    camera_from_tcp[:3, :3] = pair_rotation @ config.pair_from_tcp[:3, :3]
    camera_from_tcp[:3, 3] = fused_position
    return FrameTcpEstimate(
        timestamp_ns=int(timestamp_ns),
        camera_from_pair=camera_from_pair,
        camera_from_tcp=camera_from_tcp,
        tcp_from_tag0_m=tcp_from_tag0,
        tcp_from_tag1_m=tcp_from_tag1,
        fused_tcp_position_m=fused_position,
        measured_tag_distance_m=measured_distance,
        expected_tag_distance_m=2.0 * expected_half_distance,
        candidate_translation_difference_m=candidate_difference,
        tag0_reprojection_rmse_px=float(tag0.reprojection_rmse_px),
        tag1_reprojection_rmse_px=float(tag1.reprojection_rmse_px),
        tag0=tag0,
        tag1=tag1,
    )


def _tag_object_points(marker_size_m: float) -> np.ndarray:
    """按 IPPE_SQUARE 要求返回图像角点对应的米制角点。

    OpenCV 的方形 IPPE 约定平面坐标的 ``+Y`` 指向图像上方，因此图像
    左上、右上、右下、左下对应 ``(-x,+y)、(+x,+y)、(+x,-y)、(-x,-y)``。
    """
    half_size = marker_size_m / 2.0
    return np.asarray(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float64,
    )


def _transform_from_pnp(
    rotation_vector: np.ndarray, translation_vector: np.ndarray
) -> np.ndarray:
    """把 OpenCV PnP 输出转换为 ``^camera T_tag``。"""
    rotation_matrix, _ = cv2.Rodrigues(np.asarray(rotation_vector).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = np.asarray(translation_vector, dtype=np.float64).reshape(3)
    return transform


def _positive_depth(object_points: np.ndarray, transform: np.ndarray) -> bool:
    """检查方形标签四角是否都位于相机前方。"""
    camera_points = (transform[:3, :3] @ object_points.T).T + transform[:3, 3]
    return bool(np.all(np.isfinite(camera_points)) and np.all(camera_points[:, 2] > 0.0))


def _pinhole_rmse(
    object_points: np.ndarray,
    corners_px: np.ndarray,
    rotation_vector: np.ndarray,
    translation_vector: np.ndarray,
    camera_k: np.ndarray,
) -> float:
    """计算 rectified K 平面上的四角针孔重投影 RMSE。"""
    projected, _ = cv2.projectPoints(
        object_points.reshape(-1, 1, 3),
        np.asarray(rotation_vector, dtype=np.float64).reshape(3, 1),
        np.asarray(translation_vector, dtype=np.float64).reshape(3, 1),
        camera_k,
        None,
    )
    errors = projected.reshape(4, 2) - corners_px
    return float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))


class DualArucoTcpEstimator:
    """在整幅 Kalibr 去畸变图像上检测双 ArUco 并估计 TCP。"""

    def __init__(
        self,
        camera: FisheyeCameraModel,
        config: ArucoTcpConfig,
    ) -> None:
        """一次初始化复用 Kalibr K 的去畸变映射和 ArUco 检测器。"""
        self._camera = camera
        self._config = config
        dictionary_constant = getattr(cv2.aruco, config.aruco.dictionary_name)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_constant)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
            camera.k,
            camera.d.reshape(4, 1),
            np.eye(3, dtype=np.float64),
            camera.k,
            camera.resolution,
            cv2.CV_16SC2,
        )

    @property
    def rectification_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """返回初始化后的整幅图像去畸变映射。"""
        return self._map1, self._map2

    def _solve_tag(self, tag_id: int, corners_px: np.ndarray) -> TagPoseEstimate:
        """使用 IPPE_SQUARE 从一枚标签四角估计正深度位姿。"""
        object_points = _tag_object_points(self._config.aruco.marker_size_m)
        corners = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
        try:
            result = cv2.solvePnPGeneric(
                object_points.reshape(-1, 1, 3),
                corners.reshape(-1, 1, 2),
                self._camera.k,
                None,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except cv2.error as error:
            raise FrameEstimationError(
                f"ID {tag_id} IPPE 求解失败: {error}"
            ) from error
        if len(result) < 3:
            raise FrameEstimationError(f"ID {tag_id} IPPE 没有返回候选")
        solved, rotation_vectors, translation_vectors = result[:3]
        if not solved or not rotation_vectors or not translation_vectors:
            raise FrameEstimationError(f"ID {tag_id} IPPE 没有返回候选")
        candidates = []
        for rotation_vector, translation_vector in zip(
            rotation_vectors, translation_vectors
        ):
            transform = _transform_from_pnp(rotation_vector, translation_vector)
            if not _positive_depth(object_points, transform):
                continue
            rmse = _pinhole_rmse(
                object_points,
                corners,
                rotation_vector,
                translation_vector,
                self._camera.k,
            )
            if np.isfinite(rmse):
                candidates.append((rmse, transform))
        if not candidates:
            raise FrameEstimationError(f"ID {tag_id} IPPE 候选包含非正深度")
        rmse, transform = min(candidates, key=lambda item: item[0])
        return TagPoseEstimate(
            tag_id=tag_id,
            camera_from_tag=transform,
            reprojection_rmse_px=rmse,
            corners_px=corners,
        )

    def estimate(
        self, image_bgr: np.ndarray, openness: float, timestamp_ns: int
    ) -> FrameTcpEstimate:
        """先整幅去畸变，再检测 ID 0/1、PnP 并融合单帧 TCP。"""
        image = np.asarray(image_bgr)
        expected_width, expected_height = self._camera.resolution
        if image.ndim not in (2, 3):
            raise FrameEstimationError("输入图像必须是灰度或 BGR 数组")
        if image.shape[0] != expected_height or image.shape[1] != expected_width:
            raise FrameEstimationError(
                f"图像分辨率 {image.shape[1]}x{image.shape[0]} 与相机配置"
                f" {expected_width}x{expected_height} 不符"
            )
        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise FrameEstimationError("输入图像必须是 BGR 或 BGRA 数组")
        try:
            rectified = cv2.remap(
                image,
                self._map1,
                self._map2,
                interpolation=cv2.INTER_LINEAR,
            )
            if rectified.ndim == 3 and rectified.shape[2] == 4:
                grayscale = cv2.cvtColor(rectified, cv2.COLOR_BGRA2GRAY)
            elif rectified.ndim == 3:
                grayscale = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
            else:
                grayscale = rectified
            corners, identifiers, _ = self._detector.detectMarkers(grayscale)
        except cv2.error as error:
            raise FrameEstimationError(f"ArUco 检测失败: {error}") from error
        if identifiers is None or len(identifiers) == 0:
            raise FrameEstimationError("当前帧缺少 ID 0/1")
        ids = [int(identifier) for identifier in np.asarray(identifiers).reshape(-1)]
        if len(set(ids)) != len(ids):
            raise FrameEstimationError("同帧检测到重复 ArUco ID")
        detections = {
            tag_id: np.asarray(corner, dtype=np.float64).reshape(4, 2)
            for tag_id, corner in zip(ids, corners)
        }
        required_ids = (self._config.aruco.tag0_id, self._config.aruco.tag1_id)
        missing_ids = [tag_id for tag_id in required_ids if tag_id not in detections]
        if missing_ids:
            raise FrameEstimationError(f"当前帧缺少 ArUco ID: {missing_ids}")
        tag0 = self._solve_tag(required_ids[0], detections[required_ids[0]])
        tag1 = self._solve_tag(required_ids[1], detections[required_ids[1]])
        return estimate_tcp_from_tag_poses(
            tag0,
            tag1,
            openness,
            self._config,
            timestamp_ns=timestamp_ns,
        )
