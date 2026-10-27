"""解析 UMI 鱼眼内参并校验离线标定参数；长度采用米，图像尺寸采用像素。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class FisheyeCamera:
    """OpenCV 鱼眼相机模型：K 为 (3,3)，D 为 (4,1)，图像为 width×height。"""

    K: np.ndarray
    D: np.ndarray
    width: int
    height: int

    def project(self, points_camera):
        """将相机系米制 (N,3) 点投影成 (N,2) 像素；相机后方点返回 NaN。"""
        points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
        pixels = np.full((len(points), 2), np.nan)
        valid = np.all(np.isfinite(points), axis=1) & (points[:, 2] > 1e-6)
        if np.any(valid):
            projected, _ = cv2.fisheye.projectPoints(
                points[valid].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.D
            )
            pixels[valid] = projected.reshape(-1, 2)
        return pixels


def load_fisheye_camera(path):
    """读取 cam0 或顶层 Kalibr 内参；非法模型、维度或数值抛出 ValueError。"""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("相机 YAML 顶层必须为映射")
    camera = data.get("cam0", data)
    if not isinstance(camera, dict):
        raise ValueError("cam0 必须为映射")
    if camera.get("camera_model", "pinhole") != "pinhole":
        raise ValueError("仅支持 pinhole 投影模型")
    if camera.get("distortion_model") not in ("fisheye", "equidistant"):
        raise ValueError("仅支持 fisheye/equidistant 畸变模型")
    try:
        intrinsics = np.asarray(camera["intrinsics"], dtype=float)
        distortion = np.asarray(camera["distortion_coeffs"], dtype=float)
        resolution = np.asarray(camera["resolution"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("相机内参缺失或包含非法数值") from error
    if (
        intrinsics.shape != (4,)
        or distortion.shape != (4,)
        or resolution.shape != (2,)
        or not all(np.all(np.isfinite(x)) for x in (intrinsics, distortion, resolution))
        or np.any(intrinsics[:2] <= 0)
        or np.any(resolution <= 0)
        or np.any(resolution != np.floor(resolution))
    ):
        raise ValueError("相机内参需为有限数值，焦距为正，分辨率为正整数，畸变系数为四项")
    fx, fy, cx, cy = intrinsics
    return FisheyeCamera(
        K=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
        D=distortion.reshape(4, 1),
        width=int(resolution[0]),
        height=int(resolution[1]),
    )


@dataclass(frozen=True)
class QualitySettings:
    """单帧质量门限及鲁棒聚合参数；误差单位由字段名指定。"""

    max_reprojection_px: float = 1.5
    max_normal_angle_deg: float = 10.0
    max_coplanarity_deg: float = 10.0
    min_inliers: int = 30
    mad_k: float = 3.0
    translation_floor_m: float = 0.001
    rotation_floor_deg: float = 0.5

    def __post_init__(self):
        """拒绝非有限、非正门限及非法最小内点数量。"""
        for name, value in vars(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为正的有限数")
        if int(self.min_inliers) != self.min_inliers:
            raise ValueError("min_inliers 必须为整数")
        if self.max_normal_angle_deg > 180 or self.max_coplanarity_deg > 90:
            raise ValueError("法向夹角不得超过 180°，共面偏差不得超过 90°")

    def rejection_reason(self, observation):
        """返回未通过门限的原因；合格观测返回空字符串。"""
        for name, value, limit in (
            ("重投影误差", observation.reprojection_rms_px, self.max_reprojection_px),
            ("法向夹角", observation.normal_angle_deg, self.max_normal_angle_deg),
            ("共面偏差", observation.coplanarity_deg, self.max_coplanarity_deg),
        ):
            if not np.isfinite(value) or value < 0 or value > limit:
                return f"{name}超过门限或无效"
        return ""
