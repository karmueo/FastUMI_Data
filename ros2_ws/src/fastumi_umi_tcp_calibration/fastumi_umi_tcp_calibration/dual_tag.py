"""夹爪双 ArUco（ID 0 / ID 1）在UMI鱼眼图像中的检测与 u_tag 坐标系求解（无 ROS 依赖）。

u_tag 坐标系与 config/calibration/T_u_tag_tcp.yaml 一致：
    原点：两枚 tag 中心连线的中点；
    +X：沿两 tag 中心连线，由 ID 0 指向 ID 1；
    +Z：两枚 tag 平均法向（ArUco 正面法向，指向相机一侧），已对 X 正交化；
    +Y：Z × X（右手系），按安装约定指向 TCP。

检测在原始鱼眼图上进行（不整幅去畸变，以免裁掉画面边缘的夹爪 tag），角点经
cv2.fisheye.undistortPoints 转为归一化坐标后用 IPPE_SQUARE 求两个候选位姿。16 mm
小标签的 IPPE 翻转歧义用“两枚 tag 法向一致、共面”在 2×2 组合中消除。
所有变换为 4x4 齐次矩阵，^a T_b 表示 b 系在 a 系中的位姿，平移单位米。
"""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from fastumi_umi_tcp_calibration.transforms import (
    average_transforms,
    invert,
    make_transform,
    rotation_angle,
)


class DualTagError(ValueError):
    """当前帧无法得到有效的双 tag 观测；消息为可读的拒绝原因。"""


@dataclass(frozen=True)
class TagPose:
    """单枚 tag 的一个 IPPE 候选：^cam T_tag、鱼眼像素重投影 RMS 和原始角点 (4,2)。"""

    tag_id: int
    camera_from_tag: np.ndarray
    reprojection_rms_px: float
    corners_px: np.ndarray

    @property
    def center(self):
        """tag 中心在相机系中的位置（米）。"""
        return self.camera_from_tag[:3, 3]

    @property
    def normal(self):
        """tag 局部 +Z（正面法向）在相机系中的单位方向。"""
        return self.camera_from_tag[:3, 2]


@dataclass(frozen=True)
class DualTagObservation:
    """单帧双 tag 结果：^cam T_u_tag 及质量指标。"""

    tag0: TagPose
    tag1: TagPose
    camera_from_utag: np.ndarray
    normal_angle_deg: float  # 两枚 tag 法向夹角
    coplanarity_deg: float  # 中心连线偏离平均 tag 平面的角度
    tag_distance_m: float  # 两 tag 中心距离

    @property
    def reprojection_rms_px(self):
        """两枚 tag 重投影 RMS 的较大值。"""
        return max(self.tag0.reprojection_rms_px, self.tag1.reprojection_rms_px)


def tag_object_points(marker_size_m):
    """IPPE_SQUARE 约定的 tag 四角（左上、右上、右下、左下），局部 +Y 向上、+Z 指向观察者。"""
    half = marker_size_m / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def make_aruco_detector(dictionary_name):
    """按名称（如 DICT_4X4_50）创建亚像素角点细化的 ArUco 检测器。"""
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"未知 ArUco 字典 {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def detect_dual_tags(image, detector, tag_ids):
    """在原始鱼眼图上检测 tag_ids 中的两枚 tag，返回 {id: 角点 (4,2)}；缺失或重复时抛 DualTagError。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, identifiers, _ = detector.detectMarkers(gray)
    found = {}
    if identifiers is not None:
        for corner, identifier in zip(corners, np.asarray(identifiers).reshape(-1)):
            identifier = int(identifier)
            if identifier not in tag_ids:
                continue
            if identifier in found:
                raise DualTagError(f"同帧检测到重复 ID {identifier}")
            found[identifier] = np.asarray(corner, dtype=np.float64).reshape(4, 2)
    missing = [tag_id for tag_id in tag_ids if tag_id not in found]
    if missing:
        raise DualTagError(f"缺少 tag {missing}")
    return found


def solve_tag_candidates(tag_id, corners_px, camera, marker_size_m):
    """鱼眼角点 → 归一化坐标 → IPPE_SQUARE，返回正深度的候选 TagPose 列表（按 RMS 升序）。"""
    object_points = tag_object_points(marker_size_m)
    corners = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    normalized = cv2.fisheye.undistortPoints(corners.reshape(-1, 1, 2), camera.K, camera.D).reshape(
        -1, 1, 2
    )
    try:
        result = cv2.solvePnPGeneric(
            object_points.reshape(-1, 1, 3),
            normalized,
            np.eye(3),
            None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error as error:
        raise DualTagError(f"ID {tag_id} IPPE 求解失败") from error
    solved, rotation_vectors, translation_vectors = result[:3]
    candidates = []
    if solved:
        for rotation_vector, translation_vector in zip(rotation_vectors, translation_vectors):
            rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector).reshape(3, 1))
            transform = make_transform(rotation, np.asarray(translation_vector).reshape(3))
            if not np.all(np.isfinite(transform)):
                continue
            points = object_points @ rotation.T + transform[:3, 3]
            if not np.all(points[:, 2] > 0.0):
                continue
            projected = camera.project(points)
            if not np.all(np.isfinite(projected)):
                continue
            rms = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))
            candidates.append(TagPose(tag_id, transform, rms, corners))
    if not candidates:
        raise DualTagError(f"ID {tag_id} 没有正深度的 IPPE 候选")
    return sorted(candidates, key=lambda candidate: candidate.reprojection_rms_px)


def _unit(vector):
    """归一化有限非零三维向量，退化时拒绝当前帧。"""
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise DualTagError("向量退化")
    return np.asarray(vector, dtype=np.float64) / norm


def _pair_geometry(tag0, tag1):
    """返回 (法向夹角°, 共面偏差°, 平均法向)；平均法向已指向相机一侧。"""
    normal0, normal1 = _unit(tag0.normal), _unit(tag1.normal)
    normal_angle = math.degrees(math.acos(float(np.clip(np.dot(normal0, normal1), -1.0, 1.0))))
    normal = normal0 + normal1
    if np.linalg.norm(normal) <= 1e-9:
        return normal_angle, 90.0, None
    normal = _unit(normal)
    midpoint = (tag0.center + tag1.center) / 2.0
    if float(np.dot(normal, midpoint)) > 0.0:
        normal = -normal
    line = _unit(tag1.center - tag0.center)
    coplanarity = math.degrees(math.asin(min(1.0, abs(float(np.dot(line, normal))))))
    return normal_angle, coplanarity, normal


def select_consistent_pair(candidates0, candidates1, rms_weight_deg_per_px=1.0):
    """在两枚 tag 的 IPPE 候选组合中选法向最一致、最共面的一组，返回 (tag0, tag1)。

    评分 = 法向夹角° + 共面偏差° + rms_weight × (RMS0 + RMS1)。正确组合的法向近似平行、
    中心连线位于 tag 平面内；翻转候选的法向偏离视线方向，通常相差十几度以上。
    """
    best = None
    for tag0 in candidates0:
        for tag1 in candidates1:
            normal_angle, coplanarity, _ = _pair_geometry(tag0, tag1)
            score = (
                normal_angle
                + coplanarity
                + rms_weight_deg_per_px * (tag0.reprojection_rms_px + tag1.reprojection_rms_px)
            )
            if best is None or score < best[0]:
                best = (score, tag0, tag1)
    if best is None:
        raise DualTagError("没有可用的双 TAG 候选")
    return best[1], best[2]


def camera_from_utag(tag0, tag1):
    """按 u_tag 定义由两枚 tag 位姿构造 DualTagObservation。"""
    for tag in (tag0, tag1):
        transform = np.asarray(tag.camera_from_tag)
        if (
            transform.shape != (4, 4)
            or not np.all(np.isfinite(transform))
            or transform[2, 3] <= 0
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3))
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0)
        ):
            raise DualTagError("TAG 位姿不是有限正深度刚体变换")
    delta = tag1.center - tag0.center
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-6:
        raise DualTagError("两枚 tag 中心重合")
    normal_angle, coplanarity, normal = _pair_geometry(tag0, tag1)
    if normal is None:
        raise DualTagError("两枚 tag 法向相反")
    x_axis = delta / distance
    z_axis = normal - float(np.dot(normal, x_axis)) * x_axis
    if np.linalg.norm(z_axis) <= 1e-6:
        raise DualTagError("tag 法向与中心连线平行")
    z_axis = _unit(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    transform = make_transform(
        np.column_stack([x_axis, y_axis, z_axis]), (tag0.center + tag1.center) / 2.0
    )
    return DualTagObservation(tag0, tag1, transform, normal_angle, coplanarity, distance)


class DualTagEstimator:
    """UMI鱼眼图像 → ^cam T_u_tag。camera 为 config.FisheyeCamera。"""

    def __init__(self, camera, dictionary_name="DICT_4X4_50", marker_size_m=0.016, tag_ids=(0, 1)):
        """创建双码检测器；尺寸为米，ID 必须位于字典范围内。"""
        if not np.isfinite(marker_size_m) or marker_size_m <= 0.0:
            raise ValueError("marker_size_m 必须为正")
        if len(tag_ids) != 2 or tag_ids[0] == tag_ids[1]:
            raise ValueError("tag_ids 必须是两个不同的 ID")
        self.camera = camera
        self.marker_size_m = float(marker_size_m)
        self.tag_ids = tuple(int(tag_id) for tag_id in tag_ids)
        self.detector = make_aruco_detector(dictionary_name)
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
        if any(i < 0 or i >= len(dictionary.bytesList) or int(i) != i for i in tag_ids):
            raise ValueError("tag_ids 必须是字典范围内的整数")

    def estimate(self, image):
        """检测并求解一帧；失败抛 DualTagError。"""
        if image.shape[1] != self.camera.width or image.shape[0] != self.camera.height:
            raise ValueError(
                f"图像尺寸 {image.shape[1]}x{image.shape[0]} 与内参 "
                f"{self.camera.width}x{self.camera.height} 不符"
            )
        corners = detect_dual_tags(image, self.detector, self.tag_ids)
        candidates = [
            solve_tag_candidates(tag_id, corners[tag_id], self.camera, self.marker_size_m)
            for tag_id in self.tag_ids
        ]
        return camera_from_utag(*select_consistent_pair(*candidates))


@dataclass(frozen=True)
class RobustAverage:
    """鲁棒平均结果：均值、内点掩码，以及内点相对均值的 RMS 平移（米）/旋转（度）偏差。"""

    transform: np.ndarray
    inliers: np.ndarray
    translation_rms_m: float
    rotation_rms_deg: float


def robust_average(transforms, k=3.0, translation_floor_m=1e-3, rotation_floor_deg=0.5):
    """MAD 剔除离群后求刚体变换平均。

    偏差按相对平移中位数的距离、相对全体弦平均的旋转角计算；阈值为
    max(k × 1.4826 × MAD + 中位偏差, floor)，少于 3 个样本时全部保留。
    """
    transforms = [np.asarray(transform, dtype=np.float64) for transform in transforms]
    if not transforms:
        raise ValueError("没有可平均的变换")
    inliers = np.ones(len(transforms), dtype=bool)
    if len(transforms) >= 3:
        translations = np.stack([transform[:3, 3] for transform in transforms])
        reference = average_transforms(transforms)
        translation_dev = np.linalg.norm(translations - np.median(translations, axis=0), axis=1)
        rotation_dev = np.degrees(
            [rotation_angle(reference[:3, :3], transform[:3, :3]) for transform in transforms]
        )

        def limit(deviation, floor):
            """用 MAD 与最小噪声尺度确定离群门限。"""
            median = float(np.median(deviation))
            mad = float(np.median(np.abs(deviation - median)))
            return max(median + k * 1.4826 * mad, floor)

        inliers = (translation_dev <= limit(translation_dev, translation_floor_m)) & (
            rotation_dev <= limit(rotation_dev, rotation_floor_deg)
        )
    kept = [transform for transform, keep in zip(transforms, inliers) if keep]
    if not kept:
        raise ValueError("MAD 筛选后没有内点")
    mean = average_transforms(kept)
    translation_rms = float(
        np.sqrt(np.mean([np.sum((transform[:3, 3] - mean[:3, 3]) ** 2) for transform in kept]))
    )
    rotation_rms = float(
        np.sqrt(
            np.mean(
                [
                    math.degrees(rotation_angle(mean[:3, :3], transform[:3, :3])) ** 2
                    for transform in kept
                ]
            )
        )
    )
    return RobustAverage(mean, inliers, translation_rms, rotation_rms)


def transform_difference(reference, other):
    """inv(reference) @ other 的平移（毫米，reference 系下）与旋转角（度）。"""
    delta = invert(reference) @ other
    return delta[:3, 3] * 1000.0, math.degrees(rotation_angle(np.eye(3), delta[:3, :3]))


def load_matrix_yaml(path, key):
    """读取 YAML 中的 4x4 刚体变换；key 可用点号表示嵌套（如 cam0.T_cam_hand）。"""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    value = data
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"{path}: 缺少键 {key}")
        value = value[part]
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {key} 必须是数值矩阵") from error
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{path}: {key} 必须是 4x4 有限数值矩阵")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-3)
    ):
        raise ValueError(f"{path}: {key} 不是刚体变换")
    return matrix


def draw_observation(image, camera, observation, utag_from_tcp=None, axis_length_m=0.03):
    """在图像副本上画 tag 角点、u_tag 轴（X 红、Y 绿、Z 蓝）和 TCP 投影点（洋红）。"""
    canvas = image.copy()
    for tag in (observation.tag0, observation.tag1):
        cv2.polylines(canvas, [np.round(tag.corners_px).astype(np.int32)], True, (0, 255, 255), 2)
        corner = tuple(int(round(v)) for v in tag.corners_px[0])
        cv2.putText(
            canvas, str(tag.tag_id), corner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )
    transform = observation.camera_from_utag
    points = np.array(
        [[0, 0, 0], [axis_length_m, 0, 0], [0, axis_length_m, 0], [0, 0, axis_length_m]],
        dtype=np.float64,
    )
    pixels = camera.project(points @ transform[:3, :3].T + transform[:3, 3])
    if np.all(np.isfinite(pixels)):
        origin = tuple(int(round(v)) for v in pixels[0])
        for pixel, color in zip(pixels[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            cv2.arrowedLine(canvas, origin, tuple(int(round(v)) for v in pixel), color, 2)
    if utag_from_tcp is not None:
        tcp = (transform @ utag_from_tcp)[:3, 3]
        pixel = camera.project(tcp.reshape(1, 3))[0]
        if np.all(np.isfinite(pixel)):
            cv2.drawMarker(
                canvas, tuple(int(round(v)) for v in pixel), (255, 0, 255), cv2.MARKER_CROSS, 30, 3
            )
    # OpenCV putText 不支持中文，图上只写 ASCII 摘要。
    cv2.putText(
        canvas,
        f"rms {observation.reprojection_rms_px:.2f}px normal "
        f"{observation.normal_angle_deg:.1f}deg dist "
        f"{observation.tag_distance_m * 1000:.1f}mm",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )
    return canvas
