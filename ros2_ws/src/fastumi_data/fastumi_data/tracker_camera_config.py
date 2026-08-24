"""解析 Tracker–鱼眼相机标定配置并生成目标板米制几何。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Union

from ament_index_python.packages import get_package_share_directory
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

    @property
    def target_type(self) -> str:
        """返回目标类型标识，供通用标定流水线分派。"""
        return "aprilgrid"


@dataclass(frozen=True)
class CheckerboardSpec:
    """保存棋盘格内部角点的行列和米制横纵向间距。

    ``target_cols`` 与 ``target_rows`` 表示内部角点数量。板坐标系原点位于
    左上角内部角点，x 沿列方向、y 沿行方向，坐标顺序与 OpenCV 棋盘格检测
    保持一致。
    """

    target_cols: int
    target_rows: int
    row_spacing_m: float
    col_spacing_m: float
    target_type: str = "checkerboard"

    def __post_init__(self) -> None:
        """校验内部角点尺寸、间距和固定目标类型。"""
        for name in ("target_cols", "target_rows"):
            value = getattr(self, name)
            try:
                integral = int(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"棋盘格 {name} 必须是正整数") from error
            if isinstance(value, bool) or integral != value or integral < 3:
                raise ValueError(f"棋盘格 {name} 必须是至少为 3 的正整数")
            object.__setattr__(self, name, integral)
        for name in ("row_spacing_m", "col_spacing_m"):
            value = getattr(self, name)
            try:
                spacing = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"棋盘格 {name} 必须是有限正数") from error
            if not np.isfinite(spacing) or spacing <= 0.0:
                raise ValueError(f"棋盘格 {name} 必须是有限正数")
            object.__setattr__(self, name, spacing)
        if self.target_type != "checkerboard":
            raise ValueError("棋盘格 target_type 必须是 checkerboard")

    @property
    def corner_count(self) -> int:
        """返回整板内部角点总数。"""
        return self.target_cols * self.target_rows

    @property
    def board_extent_m(self) -> tuple[float, float]:
        """返回最外侧内部角点之间的宽度和高度，单位为米。"""
        return (
            (self.target_cols - 1) * self.col_spacing_m,
            (self.target_rows - 1) * self.row_spacing_m,
        )


CalibrationTargetSpec = Union[AprilGridSpec, CheckerboardSpec]


@dataclass(frozen=True)
class CalibrationSettings:
    """保存标定话题、筛选、时间搜索和默认质量门。"""

    image_topic: str = "/tof_stereo_camera/rgb/image_raw"
    tracker_topic: str = "/vive_tracker/odom"
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


def default_camera_config_path() -> str:
    """返回 ToF 双目相机包内 RGB 标定文件的安装路径。"""
    # ToF 双目相机包共享目录，用于定位唯一默认内参源。
    package_share = Path(get_package_share_directory("tof_stereo_camera"))
    return str(package_share / "config" / "calibration.yaml")


def load_kalibr_camera(
    path: str, camera_key: str = "cam0"
) -> FisheyeCameraModel:
    """从 ToF RGB 或 Kalibr YAML 加载统一鱼眼相机模型。

    Args:
        path: ToF 或 Kalibr 相机配置路径。
        camera_key: 显式 Kalibr 配置需要读取的相机节点名。

    Returns:
        经过严格校验的鱼眼相机模型。
    """
    document = _load_yaml_mapping(path)
    has_rgb = "rgb" in document
    has_kalibr = camera_key in document
    if has_rgb and has_kalibr:
        raise ValueError(
            "相机配置同时包含 rgb 和 "
            f"{camera_key}，无法确定标定来源"
        )
    if not has_rgb and not has_kalibr:
        raise ValueError(f"相机配置必须包含 rgb 或 {camera_key}")

    selected_key = "rgb" if has_rgb else camera_key
    camera = _checked_mapping(
        document[selected_key], f"相机节点 {selected_key}"
    )
    if selected_key == "rgb":
        source_distortion_model = str(camera.get("distortion_model", ""))
        if source_distortion_model != "fisheye":
            raise ValueError("ToF rgb distortion_model 必须是 fisheye")
    else:
        camera_model = str(camera.get("camera_model", ""))
        if camera_model != "pinhole":
            raise ValueError("Kalibr camera_model 必须是 pinhole")
        source_distortion_model = str(camera.get("distortion_model", ""))
        if source_distortion_model != "equidistant":
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
        camera_model="pinhole",
        distortion_model="equidistant",
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


def load_checkerboard(path: str) -> CheckerboardSpec:
    """从目标 YAML 加载棋盘格内部角点规格。"""
    document = _load_yaml_mapping(path)
    if document.get("target_type") != "checkerboard":
        raise ValueError("标定目标 target_type 必须是 checkerboard")
    try:
        return CheckerboardSpec(
            target_cols=document["targetCols"],
            target_rows=document["targetRows"],
            row_spacing_m=float(document["rowSpacingMeters"]),
            col_spacing_m=float(document["colSpacingMeters"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ValueError) and not isinstance(error, KeyError):
            raise
        raise ValueError(
            "棋盘格配置缺少有效内部角点行列或间距"
        ) from error


def load_calibration_target(
    path: str, tag_family: str = "tag36h11"
) -> CalibrationTargetSpec:
    """按 YAML ``target_type`` 加载 AprilGrid 或棋盘格目标。"""
    document = _load_yaml_mapping(path)
    target_type = document.get("target_type")
    if target_type == "aprilgrid":
        return load_aprilgrid(path, tag_family)
    if target_type == "checkerboard":
        return load_checkerboard(path)
    raise ValueError(
        "标定目标 target_type 必须是 aprilgrid 或 checkerboard"
    )


def checkerboard_object_points(spec: CheckerboardSpec) -> np.ndarray:
    """按 OpenCV 行优先顺序生成整板内部角点的米制三维坐标。

    返回形状为 ``(target_rows * target_cols, 3)`` 的数组，单点坐标为
    ``[column * col_spacing_m, row * row_spacing_m, 0]``。
    """
    rows, columns = np.indices(
        (spec.target_rows, spec.target_cols), dtype=np.float64
    )
    points = np.stack(
        (
            columns.reshape(-1) * spec.col_spacing_m,
            rows.reshape(-1) * spec.row_spacing_m,
            np.zeros(spec.corner_count, dtype=np.float64),
        ),
        axis=1,
    )
    return points


def checkerboard_object_corners(spec: CheckerboardSpec) -> np.ndarray:
    """兼容性别名：返回棋盘格内部角点的米制三维坐标。"""
    return checkerboard_object_points(spec)


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
