"""提供与 ROS 解耦的 ArUco 检测、鱼眼校正和三维位姿计算。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

import cv2
from gripper_openness.calibration import CameraCalibration
import numpy as np


def normalize_distance(distance_mm: float, minimum_mm: float, maximum_mm: float) -> float:
    """将三维毫米距离线性映射并裁剪到 `[0, 1]`。"""
    distance = float(distance_mm)
    minimum = float(minimum_mm)
    maximum = float(maximum_mm)
    if not all(math.isfinite(value) for value in (distance, minimum, maximum)):
        raise ValueError("距离归一化输入必须为有限数")
    if minimum < 0.0 or minimum >= maximum:
        raise ValueError("距离范围必须满足 0 <= min < max")
    return float(np.clip((distance - minimum) / (maximum - minimum), 0.0, 1.0))


@dataclass(frozen=True)
class MarkerPose:
    """保存一个标记相对于相机的 Rodrigues 位姿。"""

    rotation_vector: Tuple[float, float, float]
    translation_mm: Tuple[float, float, float]


@dataclass(frozen=True)
class VisionResult:
    """保存单帧检测结果，允许调用方处理无效双码帧。"""

    detected_marker_count: int
    left_corners: Optional[np.ndarray]
    right_corners: Optional[np.ndarray]
    left_pose: Optional[MarkerPose]
    right_pose: Optional[MarkerPose]
    distance_mm: Optional[float]
    roi: Tuple[int, int, int, int]

    @property
    def valid(self) -> bool:
        """返回两枚目标标记是否均完成有效位姿计算。"""
        return self.distance_mm is not None


class GripperVision:
    """检测双 ArUco 标记并计算中心之间的三维毫米距离。"""

    def __init__(
        self,
        camera_calibration: CameraCalibration,
        marker_size_mm: float = 16.0,
        dictionary_name: str = "DICT_4X4_50",
        left_marker_id: int = 0,
        right_marker_id: int = 1,
    ) -> None:
        """初始化相机、ArUco 字典和方形标记物点。"""
        if not math.isfinite(float(marker_size_mm)) or float(marker_size_mm) <= 0.0:
            raise ValueError("marker_size_mm 必须为正的有限数")
        if (
            left_marker_id < 0
            or right_marker_id < 0
            or left_marker_id == right_marker_id
        ):
            raise ValueError("左右 ArUco ID 必须为不同的非负整数")
        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f"未知 ArUco 字典: {dictionary_name}")
        self.camera_calibration = camera_calibration
        self.marker_size_mm = float(marker_size_mm)
        self.dictionary_name = dictionary_name
        self.left_marker_id = int(left_marker_id)
        self.right_marker_id = int(right_marker_id)
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._detector_parameters = self._make_detector_parameters()
        detector_class = getattr(cv2.aruco, "ArucoDetector", None)
        self._detector = (
            detector_class(self._dictionary, self._detector_parameters)
            if callable(detector_class)
            else None
        )
        half = self.marker_size_mm * 0.5
        self._object_points = np.asarray(
            [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
            dtype=np.float64,
        )
        self._zero_distortion = np.zeros(
            (max(4, camera_calibration.distortion_coefficients.size), 1),
            dtype=np.float64,
        )
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def detect(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> VisionResult:
        """在图像或指定 ROI 内检测标记并计算双码距离。

        Args:
            image: BGR、BGRA 或灰度图像。
            roi: 原图像素坐标 ``(x_min, y_min, x_max, y_max)``；为空时使用全图。

        Returns:
            包含检测数量、角点、位姿和距离的结果对象。

        Raises:
            ValueError: 输入图像为空、分辨率不符或 ROI 越界时抛出。
        """
        gray = self._to_gray(image)
        height, width = gray.shape
        expected_width, expected_height = self.camera_calibration.resolution
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"输入图像分辨率 {width}x{height} 与相机标定分辨率 "
                f"{expected_width}x{expected_height} 不一致"
            )
        bounds = self._validate_roi(roi, width, height)
        x_min, y_min, x_max, y_max = bounds
        roi_image = gray[y_min:y_max, x_min:x_max]
        blurred = cv2.GaussianBlur(roi_image, (5, 5), 0)
        enhanced = self._clahe.apply(blurred)
        try:
            if self._detector is not None:
                corners, ids, _ = self._detector.detectMarkers(enhanced)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    enhanced, self._dictionary, parameters=self._detector_parameters
                )
        except cv2.error:
            corners, ids = [], None

        found: Dict[int, np.ndarray] = {}
        if ids is not None:
            offset = np.asarray([x_min, y_min], dtype=np.float64)
            for detected_corners, marker_id in zip(corners, ids.flatten()):
                marker_number = int(marker_id)
                if marker_number in (self.left_marker_id, self.right_marker_id):
                    found[marker_number] = (
                        np.asarray(detected_corners, dtype=np.float64).reshape(4, 2)
                        + offset
                    )
        left_corners = found.get(self.left_marker_id)
        right_corners = found.get(self.right_marker_id)
        left_pose = self.estimate_pose(left_corners) if left_corners is not None else None
        right_pose = self.estimate_pose(right_corners) if right_corners is not None else None
        distance = None
        if left_pose is not None and right_pose is not None:
            left_position = np.asarray(left_pose.translation_mm, dtype=np.float64)
            right_position = np.asarray(right_pose.translation_mm, dtype=np.float64)
            candidate = float(np.linalg.norm(left_position - right_position))
            if math.isfinite(candidate):
                distance = candidate
        return VisionResult(
            detected_marker_count=len(found),
            left_corners=left_corners,
            right_corners=right_corners,
            left_pose=left_pose,
            right_pose=right_pose,
            distance_mm=distance,
            roi=bounds,
        )

    def estimate_pose(self, corners: np.ndarray) -> Optional[MarkerPose]:
        """从原始图像角点计算标记相对于相机的毫米位姿。"""
        if corners is None:
            return None
        points = np.asarray(corners, dtype=np.float64).reshape(4, 1, 2)
        if not np.all(np.isfinite(points)):
            return None
        calibration = self.camera_calibration
        if calibration.is_fisheye:
            corrected = cv2.fisheye.undistortPoints(
                points,
                calibration.camera_matrix,
                calibration.distortion_coefficients.reshape(4, 1),
                P=calibration.camera_matrix,
            ).reshape(4, 2)
            distortion = np.zeros((4, 1), dtype=np.float64)
        else:
            corrected = cv2.undistortPoints(
                points,
                calibration.camera_matrix,
                calibration.distortion_coefficients,
                P=calibration.camera_matrix,
            ).reshape(4, 2)
            distortion = np.zeros(
                (max(4, calibration.distortion_coefficients.size), 1), dtype=np.float64
            )
        try:
            success, rotation, translation = cv2.solvePnP(
                self._object_points,
                corrected,
                calibration.camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not success:
                success, rotation, translation = cv2.solvePnP(
                    self._object_points,
                    corrected,
                    calibration.camera_matrix,
                    distortion,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
        except cv2.error:
            return None
        rotation_values = np.asarray(rotation, dtype=np.float64).reshape(3)
        translation_values = np.asarray(translation, dtype=np.float64).reshape(3)
        if (
            not np.all(np.isfinite(rotation_values))
            or not np.all(np.isfinite(translation_values))
            or translation_values[2] <= 0.0
        ):
            return None
        return MarkerPose(
            rotation_vector=tuple(float(value) for value in rotation_values),
            translation_mm=tuple(float(value) for value in translation_values),
        )

    @staticmethod
    def _make_detector_parameters():
        """创建兼容 OpenCV 4.6 至 4.11 的检测参数。"""
        factory = getattr(cv2.aruco, "DetectorParameters", None)
        parameters = factory() if callable(factory) else cv2.aruco.DetectorParameters_create()
        parameters.adaptiveThreshConstant = 7
        parameters.minMarkerPerimeterRate = 0.005
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return parameters

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """将 BGR/BGRA/灰度图像转换为灰度数组。"""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("输入图像不能为空")
        if image.ndim == 2:
            return image
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("输入图像必须是灰度、BGR 或 BGRA")
        code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(image, code)

    @staticmethod
    def _validate_roi(
        roi: Optional[Tuple[int, int, int, int]], width: int, height: int
    ) -> Tuple[int, int, int, int]:
        """校验并裁剪像素 ROI。"""
        if roi is None:
            return 0, 0, width, height
        if len(roi) != 4:
            raise ValueError("ROI 必须包含四个边界")
        x_min, y_min, x_max, y_max = (int(value) for value in roi)
        if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
            raise ValueError("ROI 必须位于图像边界内且具有正面积")
        return x_min, y_min, x_max, y_max
