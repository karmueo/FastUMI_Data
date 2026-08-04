"""解析 Tracker–鱼眼相机标定配置并生成 AprilGrid 米制几何。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


def _checked_mapping(value: Any, description: str) -> Mapping[str, Any]:
    """把 YAML 节点校验为映射并返回。"""
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} 必须是 YAML 映射")
    return value


def _load_yaml_mapping(path: str) -> Mapping[str, Any]:
    """读取 YAML 文件并要求顶层为映射。"""
    config_path = Path(path)
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取 YAML 配置 {config_path}: {error}") from error
    return _checked_mapping(document, f"配置 {config_path}")


@dataclass(frozen=True)
class FisheyeCameraModel:
    """保存 Kalibr pinhole+equidistant 相机内参与图像分辨率。

    Attributes:
        k: 针孔内参矩阵，形状为 ``(3, 3)``。
        d: equidistant 畸变参数，形状为 ``(4,)``。
        resolution: 图像宽度和高度，单位为像素。
        camera_model: Kalibr 投影模型，固定为 ``pinhole``。
        distortion_model: Kalibr 畸变模型，固定为 ``equidistant``。
    """

    k: np.ndarray
    d: np.ndarray
    resolution: tuple[int, int]
    camera_model: str = "pinhole"
    distortion_model: str = "equidistant"

    def __post_init__(self) -> None:
        """归一化数组并拒绝不受支持或退化的相机参数。"""
        intrinsic = np.array(self.k, dtype=np.float64, copy=True)
        distortion = np.array(self.d, dtype=np.float64, copy=True)
        if intrinsic.shape != (3, 3):
            raise ValueError("相机内参 K 必须是 3x3 矩阵")
        if distortion.shape != (4,):
            raise ValueError("鱼眼畸变参数 D 必须包含 4 个数值")
        if not np.all(np.isfinite(intrinsic)) or not np.all(
            np.isfinite(distortion)
        ):
            raise ValueError("相机内参 K 和畸变参数 D 必须是有限数值")
        if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
            raise ValueError("相机焦距必须为正数")
        if self.camera_model != "pinhole":
            raise ValueError("相机模型必须是 pinhole")
        if self.distortion_model != "equidistant":
            raise ValueError("畸变模型必须是 equidistant")
        if (
            len(self.resolution) != 2
            or any(int(value) != value or value <= 0 for value in self.resolution)
        ):
            raise ValueError("相机分辨率必须包含两个正整数")
        intrinsic.setflags(write=False)
        distortion.setflags(write=False)
        object.__setattr__(self, "k", intrinsic)
        object.__setattr__(self, "d", distortion)
        object.__setattr__(
            self,
            "resolution",
            (int(self.resolution[0]), int(self.resolution[1])),
        )


@dataclass(frozen=True)
class AprilGridSpec:
    """保存 Kalibr AprilGrid 的行列、米制尺寸和标签族。

    板坐标系以 Tag 0 左下检测角为原点，x 向右、y 向上。
    """

    tag_cols: int
    tag_rows: int
    tag_size_m: float
    tag_spacing: float
    tag_family: str

    def __post_init__(self) -> None:
        """校验网格尺寸、间距和标签族。"""
        if self.tag_cols <= 0 or self.tag_rows <= 0:
            raise ValueError("AprilGrid 行列数必须为正整数")
        if not np.isfinite(self.tag_size_m) or self.tag_size_m <= 0.0:
            raise ValueError("AprilGrid tagSize 必须为正数")
        if not np.isfinite(self.tag_spacing) or self.tag_spacing < 0.0:
            raise ValueError("AprilGrid tagSpacing 不能为负数")
        if not self.tag_family:
            raise ValueError("AprilGrid Tag family 不能为空")

    @property
    def board_extent_m(self) -> tuple[float, float]:
        """返回最外侧标签检测角之间的宽度和高度，单位为米。"""
        pitch = self.tag_size_m * (1.0 + self.tag_spacing)
        return (
            self.tag_size_m + (self.tag_cols - 1) * pitch,
            self.tag_size_m + (self.tag_rows - 1) * pitch,
        )


@dataclass(frozen=True)
class CalibrationSettings:
    """保存标定话题、筛选、时间搜索和默认质量门。"""

    image_topic: str = "/xv_sdk/SN250801DR48FB26001253/rgb/image"
    tracker_topic: str = "/vive_tracker/pose"
    status_topic: str = "/vive_tracker/status"
    tag_family: str = "tag36h11"
    frame_stride: int = 2
    min_tags: int = 6
    max_pose_gap_ms: float = 50.0
    time_offset_min_ms: float = -100.0
    time_offset_max_ms: float = 100.0
    time_offset_step_ms: float = 2.0
    minimum_valid_frames: int = 30
    validation_median_max_px: float = 1.0
    validation_p95_max_px: float = 2.0
    closure_translation_max_mm: float = 5.0
    closure_rotation_max_deg: float = 1.0

    def __post_init__(self) -> None:
        """拒绝会导致空采样或无效搜索区间的运行参数。"""
        for name in ("image_topic", "tracker_topic", "status_topic"):
            if not getattr(self, name):
                raise ValueError(f"{name} 不能为空")
        if self.frame_stride <= 0:
            raise ValueError("frame_stride 必须为正整数")
        if self.min_tags <= 0:
            raise ValueError("min_tags 必须为正整数")
        if self.max_pose_gap_ms <= 0.0:
            raise ValueError("max_pose_gap_ms 必须为正数")
        if self.time_offset_min_ms > self.time_offset_max_ms:
            raise ValueError("时间偏移最小值不能大于最大值")
        if self.time_offset_step_ms <= 0.0:
            raise ValueError("时间偏移步长必须为正数")
        if self.minimum_valid_frames <= 0:
            raise ValueError("有效帧门限必须为正整数")
        quality_limits = (
            self.validation_median_max_px,
            self.validation_p95_max_px,
            self.closure_translation_max_mm,
            self.closure_rotation_max_deg,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in quality_limits):
            raise ValueError("质量门限必须为有限正数")


def load_kalibr_camera(
    path: str, camera_key: str = "cam0"
) -> FisheyeCameraModel:
    """从 Kalibr camchain YAML 加载 pinhole+equidistant 相机模型。

    Args:
        path: Kalibr 相机配置路径。
        camera_key: 需要读取的相机节点名。

    Returns:
        经过严格校验的鱼眼相机模型。
    """
    document = _load_yaml_mapping(path)
    if camera_key not in document:
        raise ValueError(f"相机配置缺少节点 {camera_key}")
    camera = _checked_mapping(document[camera_key], f"相机节点 {camera_key}")
    camera_model = str(camera.get("camera_model", ""))
    if camera_model != "pinhole":
        raise ValueError("Kalibr camera_model 必须是 pinhole")
    distortion_model = str(camera.get("distortion_model", ""))
    if distortion_model != "equidistant":
        raise ValueError("Kalibr distortion_model 必须是 equidistant")
    try:
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
        distortion = np.asarray(
            camera["distortion_coeffs"], dtype=np.float64
        )
        resolution_values = camera["resolution"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Kalibr 相机配置缺少有效内参、畸变或分辨率") from error
    if intrinsics.shape != (4,):
        raise ValueError("Kalibr intrinsics 必须按 fx, fy, cx, cy 提供")
    intrinsic = np.asarray(
        [
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    try:
        resolution = (int(resolution_values[0]), int(resolution_values[1]))
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError("Kalibr resolution 必须包含宽度和高度") from error
    return FisheyeCameraModel(
        k=intrinsic,
        d=distortion,
        resolution=resolution,
        camera_model=camera_model,
        distortion_model=distortion_model,
    )


def load_aprilgrid(
    path: str, tag_family: str = "tag36h11"
) -> AprilGridSpec:
    """从 Kalibr target YAML 加载 AprilGrid 规格并绑定标签族。"""
    document = _load_yaml_mapping(path)
    if document.get("target_type") != "aprilgrid":
        raise ValueError("标定目标 target_type 必须是 aprilgrid")
    try:
        return AprilGridSpec(
            tag_cols=int(document["tagCols"]),
            tag_rows=int(document["tagRows"]),
            tag_size_m=float(document["tagSize"]),
            tag_spacing=float(document["tagSpacing"]),
            tag_family=tag_family,
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and not isinstance(error, KeyError):
            raise
        raise ValueError("AprilGrid 配置缺少有效行列、尺寸或间距") from error


def tag_object_corners(spec: AprilGridSpec, tag_id: int) -> np.ndarray:
    """按行优先 ID 生成单个标签的四个米制检测角。

    板坐标系以 Tag 0 左下检测角为原点，x 向右、y 向上。返回形状为
    ``(4, 3)`` 的 ``^board p``，顺序为左下、右下、右上、左上，坐标
    单位为米。
    """
    if tag_id < 0 or tag_id >= spec.tag_cols * spec.tag_rows:
        raise ValueError(f"Tag ID {tag_id} 超出目标板范围")
    row, column = divmod(tag_id, spec.tag_cols)
    pitch = spec.tag_size_m * (1.0 + spec.tag_spacing)
    x0, y0 = column * pitch, row * pitch
    size = spec.tag_size_m
    return np.asarray(
        [
            [x0, y0, 0.0],
            [x0 + size, y0, 0.0],
            [x0 + size, y0 + size, 0.0],
            [x0, y0 + size, 0.0],
        ],
        dtype=np.float64,
    )
