"""提供与 ROS 解耦的双 ArUco 标记夹爪开合度估计算法。"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class OpennessEstimate:
    """保存单帧夹爪开合度估计结果。

    Attributes:
        openness: 平滑后的归一化开合度，范围为 0 到 1。
        raw_openness: 当前帧未经平滑的归一化开合度。
        distance_px: 两枚 ArUco 标记中心之间的像素距离。
        left_center: ID 0 标记中心在完整图像中的像素坐标。
        right_center: ID 1 标记中心在完整图像中的像素坐标。
        roi: 检测区域，顺序为 x_min、y_min、x_max、y_max。
    """

    openness: float
    raw_openness: float
    distance_px: float
    left_center: Tuple[float, float]
    right_center: Tuple[float, float]
    roi: Tuple[int, int, int, int]


class GripperOpennessEstimator:
    """通过两枚 ArUco 标记的像素间距估计夹爪开合度。

    参数中的 ROI 使用相对于图像宽高的比例，便于兼容不同分辨率。
    估计器有指数平滑状态；只有两枚目标标记同时有效时才会更新该状态。

    Args:
        closed_distance_px: 夹爪完全闭合时的标记中心距离。
        open_distance_px: 夹爪完全张开时的标记中心距离。
        smoothing_alpha: 当前帧在指数平滑中的权重，范围为 (0, 1]。
        dictionary_name: OpenCV ArUco 预定义字典名称。
        left_marker_id: 左侧标记 ID。
        right_marker_id: 右侧标记 ID。
        roi_ratios: ROI 比例，顺序为 x_min、y_min、x_max、y_max。

    Raises:
        ValueError: 标定距离、平滑系数、字典或 ROI 参数无效时抛出。
    """

    def __init__(
        self,
        closed_distance_px: float,
        open_distance_px: float,
        smoothing_alpha: float = 0.35,
        dictionary_name: str = "DICT_4X4_50",
        left_marker_id: int = 0,
        right_marker_id: int = 1,
        roi_ratios: Tuple[float, float, float, float] = (0.15, 0.58, 0.85, 0.82),
    ) -> None:
        """初始化 ArUco 检测器和归一化参数。"""
        if closed_distance_px >= open_distance_px:
            raise ValueError("closed_distance_px 必须小于 open_distance_px")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha 必须在 (0, 1] 范围内")
        if left_marker_id == right_marker_id:
            raise ValueError("left_marker_id 与 right_marker_id 必须不同")
        self._validate_roi(roi_ratios)

        # 闭合和张开标定距离，用于线性归一化。
        self.closed_distance_px = float(closed_distance_px)
        self.open_distance_px = float(open_distance_px)
        # 指数平滑权重，较大值带来更快响应。
        self.smoothing_alpha = float(smoothing_alpha)
        # 两侧目标标记 ID。
        self.left_marker_id = int(left_marker_id)
        self.right_marker_id = int(right_marker_id)
        # 相对于输入图像尺寸的检测区域。
        self.roi_ratios = tuple(float(value) for value in roi_ratios)
        # 上一帧有效的平滑结果；首次估计前为空。
        self._smoothed_openness: Optional[float] = None

        if not hasattr(cv2.aruco, dictionary_name):
            raise ValueError(f"未知 ArUco 字典: {dictionary_name}")
        # OpenCV 预定义字典编号。
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        # ArUco 字典在节点生命周期内复用，兼容 OpenCV 4.6 的过程式 API。
        self._dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        # OpenCV 4.7 起可复用面向对象检测器，旧版本在检测时回退到模块函数。
        detector_class = getattr(cv2.aruco, "ArucoDetector", None)
        if callable(detector_class):
            self._detector = detector_class(self._dictionary)
        else:
            self._detector = None

    @staticmethod
    def _validate_roi(roi_ratios: Tuple[float, float, float, float]) -> None:
        """验证 ROI 比例参数。

        Args:
            roi_ratios: 顺序为 x_min、y_min、x_max、y_max 的比例。

        Raises:
            ValueError: 比例数量、范围或边界顺序无效时抛出。
        """
        if len(roi_ratios) != 4:
            raise ValueError("roi_ratios 必须包含四个值")
        x_min, y_min, x_max, y_max = roi_ratios
        if not all(0.0 <= value <= 1.0 for value in roi_ratios):
            raise ValueError("ROI 比例必须在 [0, 1] 范围内")
        if x_min >= x_max or y_min >= y_max:
            raise ValueError("ROI 最小边界必须小于最大边界")

    def normalize_distance(self, distance_px: float) -> float:
        """将标记像素距离线性映射并裁剪到 0 到 1。

        Args:
            distance_px: 两枚标记中心的像素距离。

        Returns:
            归一化开合度；0 表示闭合，1 表示完全张开。
        """
        calibrated_range = self.open_distance_px - self.closed_distance_px
        normalized = (float(distance_px) - self.closed_distance_px) / calibrated_range
        return float(np.clip(normalized, 0.0, 1.0))

    def reset(self) -> None:
        """清除平滑状态，使下一次有效估计直接采用当前帧结果。"""
        self._smoothed_openness = None

    def estimate(self, image: np.ndarray) -> Optional[OpennessEstimate]:
        """估计单帧图像中的夹爪开合度。

        Args:
            image: 灰度、BGR 或 BGRA 图像数组。

        Returns:
            两枚目标标记均被检测到时返回结果，否则返回 ``None``。

        Raises:
            ValueError: 输入图像为空或通道格式不受支持时抛出。
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像不能为空")
        gray_image = self._to_grayscale(image)
        # 输入图像尺寸，用于将比例 ROI 转换为像素边界。
        image_height, image_width = gray_image.shape
        x_min_ratio, y_min_ratio, x_max_ratio, y_max_ratio = self.roi_ratios
        # 像素 ROI 边界。
        roi = (
            int(round(image_width * x_min_ratio)),
            int(round(image_height * y_min_ratio)),
            int(round(image_width * x_max_ratio)),
            int(round(image_height * y_max_ratio)),
        )
        x_min, y_min, x_max, y_max = roi
        # 限定在夹爪区域内检测，以降低计算量和背景误检概率。
        roi_image = gray_image[y_min:y_max, x_min:x_max]
        if self._detector is not None:
            corners, marker_ids, _ = self._detector.detectMarkers(roi_image)
        else:
            corners, marker_ids, _ = cv2.aruco.detectMarkers(
                roi_image, self._dictionary
            )
        if marker_ids is None:
            return None

        # 将每个标记中心从 ROI 坐标转换到完整图像坐标。
        centers = {
            int(marker_id): marker_corners.reshape(4, 2).mean(axis=0)
            + np.asarray([x_min, y_min], dtype=np.float32)
            for marker_corners, marker_id in zip(corners, marker_ids.flatten())
        }
        if self.left_marker_id not in centers or self.right_marker_id not in centers:
            return None

        # 两侧标记中心及欧氏像素距离。
        left_center_array = centers[self.left_marker_id]
        right_center_array = centers[self.right_marker_id]
        distance_px = float(np.linalg.norm(left_center_array - right_center_array))
        raw_openness = self.normalize_distance(distance_px)
        if self._smoothed_openness is None:
            smoothed_openness = raw_openness
        else:
            smoothed_openness = (
                self.smoothing_alpha * raw_openness
                + (1.0 - self.smoothing_alpha) * self._smoothed_openness
            )
        self._smoothed_openness = smoothed_openness

        # 不可变坐标元组，供 ROS 节点绘制调试信息。
        left_center = tuple(float(value) for value in left_center_array)
        right_center = tuple(float(value) for value in right_center_array)
        return OpennessEstimate(
            openness=float(smoothed_openness),
            raw_openness=raw_openness,
            distance_px=distance_px,
            left_center=left_center,
            right_center=right_center,
            roi=roi,
        )

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """把支持的输入图像格式转换为单通道灰度图。

        Args:
            image: 灰度、BGR 或 BGRA 图像数组。

        Returns:
            单通道灰度图。

        Raises:
            ValueError: 图像维度或通道数量不受支持时抛出。
        """
        if image.ndim == 2:
            return image
        if image.ndim != 3:
            raise ValueError("输入图像必须是二维灰度图或三维彩色图")
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        raise ValueError("仅支持单通道、BGR 或 BGRA 图像")
