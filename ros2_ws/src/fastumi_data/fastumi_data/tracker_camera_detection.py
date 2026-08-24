"""提供 AprilGrid/棋盘格检测后端并组装通用整板观测。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import cv2
import numpy as np

from fastumi_data.tracker_camera_config import (
    AprilGridSpec,
    CheckerboardSpec,
    checkerboard_object_points,
    tag_object_corners,
)


class DetectionRejected(ValueError):
    """表示当前图像检测结果不满足标定输入约束。"""


@dataclass(frozen=True)
class RawTagDetection:
    """保存检测后端返回的单个标签 ID、像素角点和诊断量。

    Attributes:
        tag_id: 标签字典中的整数 ID。
        corners_px: Kalibr 左下、右下、右上、左上顺序的 ``(4, 2)`` 像素角点。
        decision_margin: 检测后端可选的判决裕量。
        hamming: 检测后端可选的汉明纠错位数。
    """

    tag_id: int
    corners_px: np.ndarray
    decision_margin: float | None
    hamming: int | None


@dataclass(frozen=True)
class AprilGridObservation:
    """保存一帧按标签 ID 排序的 AprilGrid 二维三维对应。

    ``image_points_px`` 形状为 ``(4N, 2)``；``object_points_m`` 是板坐标系
    下形状为 ``(4N, 3)`` 的米制点。
    """

    timestamp_ns: int
    image_points_px: np.ndarray
    object_points_m: np.ndarray
    tag_ids: tuple[int, ...]
    tag_count: int
    ignored_tag_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """复制数组并保护角点对应关系不被调用方意外修改。"""
        image_points = np.array(
            self.image_points_px, dtype=np.float64, copy=True
        )
        object_points = np.array(
            self.object_points_m, dtype=np.float64, copy=True
        )
        if image_points.shape != (4 * self.tag_count, 2):
            raise ValueError("图像角点数量必须等于 tag_count 的四倍")
        if object_points.shape != (4 * self.tag_count, 3):
            raise ValueError("目标角点数量必须等于 tag_count 的四倍")
        if len(self.tag_ids) != self.tag_count:
            raise ValueError("tag_ids 数量必须等于 tag_count")
        image_points.setflags(write=False)
        object_points.setflags(write=False)
        object.__setattr__(self, "image_points_px", image_points)
        object.__setattr__(self, "object_points_m", object_points)

    @property
    def feature_count(self) -> int:
        """返回通用角点数量视图，单位为点。"""
        return len(self.image_points_px)


@dataclass(frozen=True)
class CheckerboardObservation:
    """保存一帧完整棋盘格内部角点的二维三维对应。"""

    timestamp_ns: int
    image_points_px: np.ndarray
    object_points_m: np.ndarray
    feature_count: int | None = None

    def __post_init__(self) -> None:
        """复制并校验棋盘格二维三维点的一一对应关系。"""
        image_points = np.array(
            self.image_points_px, dtype=np.float64, copy=True
        )
        object_points = np.array(
            self.object_points_m, dtype=np.float64, copy=True
        )
        if image_points.ndim != 2 or image_points.shape[1:] != (2,):
            raise ValueError("棋盘格图像点必须是 Nx2 数组")
        if object_points.shape != (len(image_points), 3):
            raise ValueError("棋盘格目标点必须是与图像点等长的 Nx3 数组")
        if len(image_points) == 0:
            raise ValueError("棋盘格观测至少需要一个角点")
        if not np.all(np.isfinite(image_points)) or not np.all(
            np.isfinite(object_points)
        ):
            raise ValueError("棋盘格观测点必须是有限数值")
        feature_count = (
            len(image_points)
            if self.feature_count is None
            else int(self.feature_count)
        )
        if feature_count != len(image_points) or feature_count <= 0:
            raise ValueError("feature_count 必须等于棋盘格角点数量且为正数")
        image_points.setflags(write=False)
        object_points.setflags(write=False)
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "image_points_px", image_points)
        object.__setattr__(self, "object_points_m", object_points)
        object.__setattr__(self, "feature_count", feature_count)


class CalibrationObservation(Protocol):
    """约束 AprilGrid/棋盘格观测向 PnP 提供通用点接口。"""

    timestamp_ns: int
    image_points_px: np.ndarray
    object_points_m: np.ndarray
    feature_count: int


class TagDetector(Protocol):
    """约束 AprilTag 后端返回统一检测结构。"""

    def detect(self, gray_image: np.ndarray) -> Sequence[RawTagDetection]:
        """检测图像并返回标签 ID 与四角。"""


class OpenCvAprilTagDetector:
    """使用 OpenCV ArUco 模块的 AprilTag 字典检测标签。"""

    KALIBR_CORNER_INDICES = np.asarray([1, 0, 3, 2])
    FAMILY_DICTIONARIES = {
        "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }

    def __init__(self, tag_family: str) -> None:
        """创建指定 AprilTag family 的 OpenCV 检测器。"""
        dictionary_id = self.FAMILY_DICTIONARIES.get(tag_family)
        if dictionary_id is None:
            raise ValueError(f"OpenCV 后端不支持 Tag family: {tag_family}")
        parameters = cv2.aruco.DetectorParameters()
        parameters.errorCorrectionRate = 1.0
        parameters.perspectiveRemovePixelPerCell = 16
        parameters.perspectiveRemoveIgnoredMarginPerCell = 0.25
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        dictionary.maxCorrectionBits = 3
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    @property
    def settings(self) -> dict[str, object]:
        """返回影响 AprilTag 解码结果的只读参数快照。"""
        return {
            "max_correction_bits": 3,
            "error_correction_rate": 1.0,
            "perspective_remove_pixel_per_cell": 16,
            "perspective_remove_ignored_margin_per_cell": 0.25,
            "corner_refinement": "contour",
        }

    def detect(self, gray_image: np.ndarray) -> Sequence[RawTagDetection]:
        """统一 mono8/BGR/BGRA 图像并返回按 ID 排序的检测结果。"""
        image = np.asarray(gray_image)
        if image.ndim == 2:
            grayscale = image
        elif image.ndim == 3 and image.shape[2] == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("检测图像必须是 mono8、BGR 或 BGRA 格式")
        if grayscale.dtype != np.uint8:
            raise ValueError("检测图像必须使用 uint8 像素")
        corners, identifiers, _ = self._detector.detectMarkers(grayscale)
        if identifiers is None:
            return []
        detections = []
        for identifier, detected_corners in zip(
            identifiers.reshape(-1), corners
        ):
            detections.append(
                RawTagDetection(
                    tag_id=int(identifier),
                    corners_px=np.asarray(
                        detected_corners, dtype=np.float64
                    ).reshape(4, 2)[self.KALIBR_CORNER_INDICES],
                    decision_margin=None,
                    hamming=None,
                )
            )
        return sorted(detections, key=lambda item: item.tag_id)


class OpenCvCheckerboardDetector:
    """使用 OpenCV 经典棋盘格算法检测并精化完整内部角点。"""

    FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    SUBPIX_WINDOW = (11, 11)
    SUBPIX_CRITERIA = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.01,
    )

    def __init__(self, spec: CheckerboardSpec) -> None:
        """创建指定内部角点行列的棋盘格检测器。"""
        if not isinstance(spec, CheckerboardSpec):
            raise TypeError("棋盘格检测器需要 CheckerboardSpec")
        self.spec = spec
        self._pattern_size = (spec.target_cols, spec.target_rows)

    @property
    def settings(self) -> dict[str, object]:
        """返回经典棋盘格检测和亚像素精化参数。"""
        return {
            "pattern_size": list(self._pattern_size),
            "flags": int(self.FIND_FLAGS),
            "find_flags": int(self.FIND_FLAGS),
            "find_flags_names": ["adaptive_thresh", "normalize_image"],
            "corner_subpix_window": list(self.SUBPIX_WINDOW),
            "corner_subpix_criteria": {
                "type": "EPS|MAX_ITER",
                "max_iterations": 30,
                "epsilon": 0.01,
            },
        }

    @staticmethod
    def _grayscale(image: np.ndarray) -> np.ndarray:
        """把 mono8/BGR/BGRA 输入统一成 uint8 灰度图。"""
        image = np.asarray(image)
        if image.ndim == 2:
            grayscale = image
        elif image.ndim == 3 and image.shape[2] == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("检测图像必须是 mono8、BGR 或 BGRA 格式")
        if grayscale.dtype != np.uint8:
            raise ValueError("检测图像必须使用 uint8 像素")
        return grayscale

    def detect(self, image: np.ndarray) -> np.ndarray:
        """检测并返回完整精化角点，失败时返回 ``(0, 2)`` 数组。"""
        grayscale = self._grayscale(image)
        found, corners = cv2.findChessboardCorners(
            grayscale, self._pattern_size, flags=self.FIND_FLAGS
        )
        if not found or corners is None:
            return np.empty((0, 2), dtype=np.float64)
        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        if len(corners) != self.spec.corner_count:
            return np.empty((0, 2), dtype=np.float64)
        refined = cv2.cornerSubPix(
            grayscale,
            corners,
            self.SUBPIX_WINDOW,
            (-1, -1),
            self.SUBPIX_CRITERIA,
        )
        refined = np.asarray(refined, dtype=np.float64).reshape(-1, 2)
        if refined.shape != (self.spec.corner_count, 2) or not np.all(
            np.isfinite(refined)
        ):
            return np.empty((0, 2), dtype=np.float64)
        return refined


def build_aprilgrid_observation(
    timestamp_ns: int,
    detections: Sequence[RawTagDetection],
    spec: AprilGridSpec,
    min_tags: int,
) -> AprilGridObservation:
    """过滤并排序原始标签，组装一帧唯一二维三维角点对应。

    Args:
        timestamp_ns: 图像消息 header 时间，单位为纳秒。
        detections: 检测后端产出的标签序列。
        spec: 目标 AprilGrid 几何规格。
        min_tags: 一帧进入 PnP 所需的最少有效标签数。

    Returns:
        按标签 ID 排序的整板观测。

    Raises:
        DetectionRejected: 重复 ID、非法角点或有效标签不足时抛出。
    """
    if min_tags <= 0:
        raise ValueError("min_tags 必须为正整数")
    unique_ids: set[int] = set()
    valid_detections = []
    ignored_ids = []
    maximum_id = spec.tag_cols * spec.tag_rows
    for detection in detections:
        tag_id = int(detection.tag_id)
        if tag_id in unique_ids:
            raise DetectionRejected(f"同帧检测到重复 Tag ID {tag_id}")
        unique_ids.add(tag_id)
        corners = np.asarray(detection.corners_px, dtype=np.float64)
        if corners.shape != (4, 2):
            raise DetectionRejected(f"Tag ID {tag_id} 角点必须是 4x2 数组")
        if not np.all(np.isfinite(corners)):
            raise DetectionRejected(f"Tag ID {tag_id} 角点必须是有限数值")
        if tag_id < 0 or tag_id >= maximum_id:
            ignored_ids.append(tag_id)
            continue
        valid_detections.append((tag_id, corners))
    if len(valid_detections) < min_tags:
        raise DetectionRejected(
            f"有效标签 {len(valid_detections)} 个，至少 {min_tags} 个"
        )
    valid_detections.sort(key=lambda item: item[0])
    tag_ids = tuple(item[0] for item in valid_detections)
    image_points = np.vstack([item[1] for item in valid_detections])
    object_points = np.vstack(
        [tag_object_corners(spec, tag_id) for tag_id in tag_ids]
    )
    return AprilGridObservation(
        timestamp_ns=int(timestamp_ns),
        image_points_px=image_points,
        object_points_m=object_points,
        tag_ids=tag_ids,
        tag_count=len(tag_ids),
        ignored_tag_ids=tuple(sorted(ignored_ids)),
    )


def build_checkerboard_observation(
    timestamp_ns: int,
    image_points_px: np.ndarray,
    spec: CheckerboardSpec,
) -> CheckerboardObservation:
    """校验完整棋盘格角点并组装通用二维三维观测。"""
    corners = np.asarray(image_points_px, dtype=np.float64)
    expected = spec.corner_count
    if corners.shape != (expected, 2):
        raise DetectionRejected(
            f"棋盘格检测得到 {len(corners) if corners.ndim else 0} 个角点，"
            f"需要完整 {expected} 个"
        )
    if not np.all(np.isfinite(corners)):
        raise DetectionRejected("棋盘格角点必须是有限数值")
    return CheckerboardObservation(
        timestamp_ns=int(timestamp_ns),
        image_points_px=corners,
        object_points_m=checkerboard_object_points(spec),
        feature_count=expected,
    )


def validate_checkerboard_observations(
    observations: Sequence[CheckerboardObservation],
    spec: CheckerboardSpec,
    minimum_probe_frames: int = 5,
) -> None:
    """确认棋盘格预检帧数足够且每帧都包含完整内部角点。"""
    if minimum_probe_frames <= 0:
        raise ValueError("minimum_probe_frames 必须为正整数")
    if len(observations) < minimum_probe_frames:
        raise DetectionRejected(
            f"棋盘格预检至少 {minimum_probe_frames} 帧有效观测"
        )
    incomplete = sorted(
        {
            int(observation.feature_count)
            for observation in observations
            if observation.feature_count != spec.corner_count
        }
    )
    if incomplete:
        raise DetectionRejected(
            f"棋盘格观测角点数量 {incomplete}，需要完整 {spec.corner_count} 个"
        )


def validate_tag_family(
    observations: Sequence[AprilGridObservation],
    spec: AprilGridSpec,
    minimum_probe_frames: int = 5,
) -> None:
    """确认预检帧数充足且解码 ID 均落在目标板范围内。

    此检查结合检测器构造时的字典校验，用于在正式求解前阻止缺少稳定观测或
    ID 范围异常的数据进入流水线。
    """
    if minimum_probe_frames <= 0:
        raise ValueError("minimum_probe_frames 必须为正整数")
    if len(observations) < minimum_probe_frames:
        raise DetectionRejected(
            f"Tag family 预检至少 {minimum_probe_frames} 帧有效观测"
        )
    maximum_id = spec.tag_cols * spec.tag_rows
    invalid_ids = sorted(
        {
            tag_id
            for observation in observations
            for tag_id in observation.tag_ids
            if tag_id < 0 or tag_id >= maximum_id
        }
    )
    if invalid_ids:
        raise DetectionRejected(
            f"Tag family 预检发现目标范围外 ID: {invalid_ids}"
        )
