"""加载相机标定，保存夹爪范围标定，并提供稳健的采样统计。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraCalibration:
    """保存用于 ArUco PnP 的相机内参和畸变模型。"""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    resolution: Tuple[int, int]
    model: str = "fisheye"

    @property
    def is_fisheye(self) -> bool:
        """返回是否使用 OpenCV 鱼眼/equidistant 模型。"""
        return self.model in ("fisheye", "equidistant")


@dataclass(frozen=True)
class GripperCalibration:
    """保存夹爪标记距离范围和检测区域。"""

    marker_size_mm: float
    dictionary_name: str
    left_finger_tag_id: int
    right_finger_tag_id: int
    min_marker_dist_mm: float
    max_marker_dist_mm: float
    resolution: Tuple[int, int]
    crop_reference: Dict[str, int]
    total_valid_frames: int = 0
    # 标定过程统计（门限剔除数、端点平台、原始极值等），仅供追溯，不参与预测。
    statistics: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class CalibrationQualityConfig:
    """单帧质量门限和端点平台检测参数。

    Attributes:
        max_reprojection_error_px: 任一标记在去畸变像素平面上的 RMS 重投影误差上限。
        min_marker_side_px: 两码平均像素边长中较小者的下限，过小的码深度误差大。
        max_relative_rotation_deviation_deg: 两码相对旋转角偏离会话中位数的上限，
            用于剔除 IPPE 翻转解和局部遮挡造成的错误位姿。
        plateau_window_frames: 判定“停留”所需的连续有效帧数。
        plateau_band_mm: 停留窗口内距离极差上限，也是端点平台的宽度。
        min_endpoint_frames: 每个端点平台至少包含的帧数。
        min_range_mm: 闭合与全开端点之间的最小距离差，低于它视为未覆盖行程。
    """

    max_reprojection_error_px: float = 2.0
    min_marker_side_px: float = 20.0
    max_relative_rotation_deviation_deg: float = 10.0
    plateau_window_frames: int = 5
    plateau_band_mm: float = 1.5
    min_endpoint_frames: int = 5
    min_range_mm: float = 5.0

    def __post_init__(self) -> None:
        """拒绝非正或非有限的门限。"""
        for name in (
            "max_reprojection_error_px",
            "min_marker_side_px",
            "max_relative_rotation_deviation_deg",
            "plateau_band_mm",
            "min_range_mm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须为正的有限数")
        if int(self.plateau_window_frames) < 2:
            raise ValueError("plateau_window_frames 至少为 2")
        if int(self.min_endpoint_frames) < 1:
            raise ValueError("min_endpoint_frames 至少为 1")


def load_camera_calibration(path: str) -> CameraCalibration:
    """读取并严格校验 ``rgb`` 或 ``cam0`` 相机标定 YAML。

    支持项目 ToF 的 ``rgb/fisheye``、当前 USB 的 ``cam0/fisheye``、
    Kalibr 的 ``cam0/equidistant``，以及常见的 ``radtan``、
    ``plumb_bob`` 和 ``brown_conrady`` 针孔标定。

    Args:
        path: 相机标定 YAML 路径。

    Returns:
        已转换为 OpenCV 矩阵形式的标定对象。

    Raises:
        ValueError: 文件不存在、结构不完整或数值不合法时抛出。
    """
    calibration_path = _required_file(path, "相机标定")
    try:
        with calibration_path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"读取相机标定 YAML 失败: {error}") from error

    if not isinstance(document, Mapping):
        raise ValueError("相机标定 YAML 顶层必须是映射")
    blocks = [key for key in ("rgb", "cam0") if key in document]
    if len(blocks) != 1:
        raise ValueError("相机标定 YAML 必须只包含一个 rgb 或 cam0 相机块")
    camera_key = blocks[0]
    camera_data = document[camera_key]
    if not isinstance(camera_data, Mapping):
        raise ValueError(f"{camera_key} 必须是相机参数映射")

    try:
        raw_model = str(camera_data["distortion_model"]).lower()
    except KeyError as error:
        raise ValueError(f"{camera_key} 缺少 distortion_model 字段") from error
    model = _normalize_model(raw_model)
    intrinsics = _finite_array(camera_data, "intrinsics", camera_key, 4)
    if model in ("fisheye", "equidistant"):
        distortion = _finite_array(camera_data, "distortion_coeffs", camera_key, 4)
    else:
        distortion = _finite_array(
            camera_data, "distortion_coeffs", camera_key, minimum=4
        )
    resolution = _resolution(camera_data, camera_key)
    fx, fy, cx, cy = (float(value) for value in intrinsics)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("相机焦距必须为正数")
    matrix = np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return CameraCalibration(
        camera_matrix=matrix,
        distortion_coefficients=distortion.reshape(-1, 1),
        resolution=resolution,
        model=model,
    )


def load_gripper_calibration(path: str) -> GripperCalibration:
    """读取标定节点生成的夹爪范围 YAML。

    同时接受 ``gripper_calibration`` 根映射和 ROS 参数文件中的
    ``gripper_calibration.ros__parameters`` 结构，便于用户手工维护配置。
    """
    calibration_path = _required_file(path, "夹爪范围标定")
    try:
        with calibration_path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"读取夹爪范围标定 YAML 失败: {error}") from error
    if not isinstance(document, Mapping):
        raise ValueError("夹爪范围标定 YAML 顶层必须是映射")
    data: Mapping[str, Any] = document
    if isinstance(document.get("gripper_calibration"), Mapping):
        data = document["gripper_calibration"]
    if isinstance(data.get("ros__parameters"), Mapping):
        data = data["ros__parameters"]
    return _parse_gripper_calibration(data)


def write_gripper_calibration(
    path: str, calibration: GripperCalibration, overwrite: bool = False
) -> None:
    """以原子方式写入夹爪范围标定 YAML。

    Args:
        path: 输出路径。
        calibration: 待写入的标定对象。
        overwrite: 是否允许替换已有文件。

    Raises:
        FileExistsError: 输出存在且未开启覆盖时抛出。
        OSError: 父目录不存在或原子替换失败时抛出。
    """
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    if not isinstance(path, str) or not path.strip():
        raise ValueError("夹爪范围标定输出路径不能为空")
    target = Path(path).expanduser()
    if target.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在，设置 overwrite=true 才能覆盖: {target}")
    if not target.parent.exists():
        raise OSError(f"输出目录不存在: {target.parent}")
    document = {
        "gripper_calibration": {
            "marker_size_mm": float(calibration.marker_size_mm),
            "dictionary_name": calibration.dictionary_name,
            "left_finger_tag_id": int(calibration.left_finger_tag_id),
            "right_finger_tag_id": int(calibration.right_finger_tag_id),
            "min_marker_dist_mm": float(calibration.min_marker_dist_mm),
            "max_marker_dist_mm": float(calibration.max_marker_dist_mm),
            "resolution": [int(value) for value in calibration.resolution],
            "crop_reference": {
                key: int(value) for key, value in calibration.crop_reference.items()
            },
            "total_valid_frames": int(calibration.total_valid_frames),
        }
    }
    if calibration.statistics:
        document["gripper_calibration"]["calibration_statistics"] = calibration.statistics
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass


class RangeAccumulator:
    """累积双 ArUco 有效帧，经质量门限后用端点停留平台的中位数生成范围。

    原始极值会被单帧 PnP 噪声和错误解向外拉偏，因此范围端点取
    “闭合/全开停留段”内距离的中位数；ROI 仍按通过门限帧的角点分位数生成。
    """

    def __init__(self, quality: Optional[CalibrationQualityConfig] = None) -> None:
        """初始化空采样器和质量门限。"""
        self.quality = quality if quality is not None else CalibrationQualityConfig()
        self.reset()

    def reset(self) -> None:
        """清空距离、角点、质量指标和图像尺寸采样。"""
        self.distances_mm: List[float] = []
        self.top_y: List[float] = []
        self.bottom_y: List[float] = []
        self.left_x: List[float] = []
        self.right_x: List[float] = []
        self.reprojection_errors_px: List[float] = []
        self.marker_sides_px: List[float] = []
        self.relative_rotations_deg: List[float] = []
        self.frame_indices: List[int] = []
        self.resolution: Optional[Tuple[int, int]] = None

    @property
    def count(self) -> int:
        """返回门限前的有效双码帧数量。"""
        return len(self.distances_mm)

    def add(
        self,
        distance_mm: float,
        left_corners: np.ndarray,
        right_corners: np.ndarray,
        resolution: Tuple[int, int],
        *,
        reprojection_error_px: float = math.nan,
        relative_rotation_deg: float = math.nan,
        frame_index: Optional[int] = None,
    ) -> None:
        """加入一帧有效双码检测及其四角点和质量指标。

        Args:
            distance_mm: 两码中心三维距离，毫米。
            left_corners: 左码原图像素角点，形状 ``(4, 2)``。
            right_corners: 右码原图像素角点，形状 ``(4, 2)``。
            resolution: 图像宽高。
            reprojection_error_px: 两码中较大的重投影误差；NaN 表示不做该门限。
            relative_rotation_deg: 两码相对旋转角；NaN 表示不做该门限。
            frame_index: 源帧序号，用于判断停留窗口是否连续；缺省按加入顺序。
        """
        distance = float(distance_mm)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("标记距离必须为非负有限数")
        left = _corners(left_corners)
        right = _corners(right_corners)
        if self.resolution is None:
            self.resolution = _valid_resolution(resolution)
        elif tuple(resolution) != self.resolution:
            raise ValueError("采样帧分辨率不一致")
        if frame_index is None:
            frame_index = self.frame_indices[-1] + 1 if self.frame_indices else 0
        self.distances_mm.append(distance)
        all_corners = np.vstack((left, right))
        self.top_y.append(float(np.min(all_corners[:, 1])))
        self.bottom_y.append(float(np.max(all_corners[:, 1])))
        self.left_x.append(float(np.min(left[:, 0])))
        self.right_x.append(float(np.max(right[:, 0])))
        self.reprojection_errors_px.append(float(reprojection_error_px))
        self.marker_sides_px.append(min(_side_length(left), _side_length(right)))
        self.relative_rotations_deg.append(float(relative_rotation_deg))
        self.frame_indices.append(int(frame_index))

    def build(
        self,
        marker_size_mm: float,
        dictionary_name: str,
        left_finger_tag_id: int,
        right_finger_tag_id: int,
    ) -> GripperCalibration:
        """根据采样结果生成可写入的标定对象。

        Raises:
            ValueError: 无样本、全部被门限剔除、缺少端点停留段、范围退化或
                ROI 无效时抛出。
        """
        if self.count == 0 or self.resolution is None:
            raise ValueError("没有有效的双 ArUco 标定样本")
        quality = self.quality
        accepted, rejected, rotation_median = self._quality_mask()
        accepted_count = int(np.count_nonzero(accepted))
        if accepted_count == 0:
            raise ValueError(
                f"{self.count} 帧双码样本全部未通过质量门限："
                f"重投影误差 {rejected['reprojection_error']}，"
                f"像素边长 {rejected['marker_side']}，"
                f"两码相对姿态 {rejected['relative_rotation']}"
            )
        order = np.argsort(np.asarray(self.frame_indices)[accepted], kind="stable")
        distances = np.asarray(self.distances_mm, dtype=np.float64)[accepted][order]
        indices = np.asarray(self.frame_indices, dtype=np.int64)[accepted][order]
        stable = _stable_mask(
            distances, indices, int(quality.plateau_window_frames), float(quality.plateau_band_mm)
        )
        stable_distances = distances[stable]
        if stable_distances.size == 0:
            raise ValueError(
                f"{accepted_count} 帧通过门限，但没有连续 {quality.plateau_window_frames} 帧"
                f"距离波动不超过 {quality.plateau_band_mm} mm 的停留段；"
                "请在机械闭合和全开位置各停留约 1 秒"
            )
        band = float(quality.plateau_band_mm)
        low_anchor = float(np.min(stable_distances))
        high_anchor = float(np.max(stable_distances))
        # 两端平台各宽 band，锚点间距不超过 2 * band 时平台会重叠，等同于只停在一处。
        if high_anchor - low_anchor <= 2.0 * band:
            raise ValueError(
                "标定距离范围必须包含两个不同的停留端点，"
                f"当前停留段仅覆盖 {low_anchor:.3f}..{high_anchor:.3f} mm"
            )
        low_plateau = stable_distances[stable_distances <= low_anchor + band]
        high_plateau = stable_distances[stable_distances >= high_anchor - band]
        minimum_frames = int(quality.min_endpoint_frames)
        for name, plateau in (("闭合", low_plateau), ("全开", high_plateau)):
            if plateau.size < minimum_frames:
                raise ValueError(
                    f"{name}端停留帧 {plateau.size} 少于 {minimum_frames}，请在该位置停留更久"
                )
        minimum = float(np.median(low_plateau))
        maximum = float(np.median(high_plateau))
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
            raise ValueError("标定距离范围必须包含两个不同的有限端点")
        if maximum - minimum < float(quality.min_range_mm):
            raise ValueError(
                f"闭合与全开端点仅相差 {maximum - minimum:.3f} mm，小于 "
                f"min_range_mm={quality.min_range_mm}，请覆盖完整开闭行程"
            )

        crop = {
            "top_y": int(np.percentile(np.asarray(self.top_y)[accepted], 1.0)),
            "bottom_y": int(np.percentile(np.asarray(self.bottom_y)[accepted], 99.0)),
            "left_x": int(np.percentile(np.asarray(self.left_x)[accepted], 1.0)),
            "right_x": int(np.percentile(np.asarray(self.right_x)[accepted], 99.0)),
        }
        width, height = self.resolution
        if not (0 <= crop["left_x"] < crop["right_x"] <= width):
            raise ValueError("标定 ROI 的左右边界无效")
        if not (0 <= crop["top_y"] < crop["bottom_y"] <= height):
            raise ValueError("标定 ROI 的上下边界无效")

        low_percentile = float(np.percentile(distances, 1.0))
        high_percentile = float(np.percentile(distances, 99.0))
        warnings = []
        # 通过门限的帧明显越过平台，说明端点可能只是路过而没有停留。
        if low_percentile < minimum - 2.0 * band:
            warnings.append(
                f"1% 分位距离 {low_percentile:.3f} mm 明显小于闭合端 {minimum:.3f} mm，"
                "闭合端可能未停留"
            )
        if high_percentile > maximum + 2.0 * band:
            warnings.append(
                f"99% 分位距离 {high_percentile:.3f} mm 明显大于全开端 {maximum:.3f} mm，"
                "全开端可能未停留"
            )
        statistics = {
            "method": "plateau_median",
            "sampled_frames": self.count,
            "accepted_frames": accepted_count,
            "rejected_frames": rejected,
            "relative_rotation_median_deg": _round(rotation_median),
            "stable_frames": int(stable_distances.size),
            "raw_min_mm": _round(float(np.min(self.distances_mm))),
            "raw_max_mm": _round(float(np.max(self.distances_mm))),
            "accepted_p1_mm": _round(low_percentile),
            "accepted_p99_mm": _round(high_percentile),
            "min_endpoint": _plateau_statistics(low_plateau),
            "max_endpoint": _plateau_statistics(high_plateau),
            "thresholds": {
                key: (int(value) if isinstance(value, int) else float(value))
                for key, value in asdict(quality).items()
            },
            "warnings": warnings,
        }
        return GripperCalibration(
            marker_size_mm=float(marker_size_mm),
            dictionary_name=str(dictionary_name),
            left_finger_tag_id=int(left_finger_tag_id),
            right_finger_tag_id=int(right_finger_tag_id),
            min_marker_dist_mm=minimum,
            max_marker_dist_mm=maximum,
            resolution=self.resolution,
            crop_reference=crop,
            total_valid_frames=accepted_count,
            statistics=statistics,
        )

    def _quality_mask(self) -> Tuple[np.ndarray, Dict[str, int], float]:
        """按重投影误差、像素边长、两码相对姿态依次过滤，每帧只计入首个失败项。"""
        quality = self.quality
        errors = np.asarray(self.reprojection_errors_px, dtype=np.float64)
        sides = np.asarray(self.marker_sides_px, dtype=np.float64)
        rotations = np.asarray(self.relative_rotations_deg, dtype=np.float64)
        # NaN 表示调用方未提供该指标，不据此剔除。
        error_ok = ~(errors > float(quality.max_reprojection_error_px))
        side_ok = sides >= float(quality.min_marker_side_px)
        prior = error_ok & side_ok
        rotation_ok = np.ones_like(prior)
        rotation_median = math.nan
        reference = rotations[prior & np.isfinite(rotations)]
        if reference.size:
            rotation_median = float(np.median(reference))
            deviation = np.abs(rotations - rotation_median)
            rotation_ok = ~(deviation > float(quality.max_relative_rotation_deviation_deg))
        rejected = {
            "reprojection_error": int(np.count_nonzero(~error_ok)),
            "marker_side": int(np.count_nonzero(error_ok & ~side_ok)),
            "relative_rotation": int(np.count_nonzero(prior & ~rotation_ok)),
        }
        return prior & rotation_ok, rejected, rotation_median


def _stable_mask(
    distances: np.ndarray, indices: np.ndarray, window: int, band_mm: float
) -> np.ndarray:
    """标记位于任一“连续 window 帧、极差不超过 band_mm”窗口内的样本。

    源帧序号跨度超过 ``2 * (window - 1)`` 的窗口视为中间丢帧过多，不算停留。
    """
    stable = np.zeros(distances.size, dtype=bool)
    max_span = 2 * (window - 1)
    for start in range(0, distances.size - window + 1):
        stop = start + window
        if indices[stop - 1] - indices[start] > max_span:
            continue
        values = distances[start:stop]
        if float(np.max(values) - np.min(values)) <= band_mm:
            stable[start:stop] = True
    return stable


def _plateau_statistics(values: np.ndarray) -> Dict[str, Any]:
    """返回端点平台的帧数、中位数和 MAD。"""
    median = float(np.median(values))
    return {
        "frames": int(values.size),
        "median_mm": _round(median),
        "mad_mm": _round(float(np.median(np.abs(values - median)))),
    }


def _round(value: float) -> Optional[float]:
    """统计值保留 4 位小数，NaN 写为 null。"""
    return round(float(value), 4) if math.isfinite(value) else None


def _side_length(corners: np.ndarray) -> float:
    """返回四角点构成四边形的平均边长（像素）。"""
    return float(np.mean(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)))


def _parse_gripper_calibration(data: Mapping[str, Any]) -> GripperCalibration:
    """解析并验证夹爪范围字段。"""
    try:
        marker_size = float(data["marker_size_mm"])
        dictionary_name = str(data.get("dictionary_name", "DICT_4X4_50"))
        left_id = _integer_value(
            data.get("left_finger_tag_id", data.get("left_marker_id", 0)),
            "left_finger_tag_id",
        )
        right_id = _integer_value(
            data.get("right_finger_tag_id", data.get("right_marker_id", 1)),
            "right_finger_tag_id",
        )
        minimum = float(data["min_marker_dist_mm"])
        maximum = float(data["max_marker_dist_mm"])
        resolution = _valid_resolution(data["resolution"])
        raw_crop = data["crop_reference"]
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"夹爪范围标定字段不完整或类型错误: {error}") from error
    if not math.isfinite(marker_size) or marker_size <= 0.0:
        raise ValueError("marker_size_mm 必须为正的有限数")
    if not dictionary_name:
        raise ValueError("dictionary_name 不能为空")
    if left_id < 0 or right_id < 0 or left_id == right_id:
        raise ValueError("左右 ArUco ID 必须为不同的非负整数")
    if (
        not math.isfinite(minimum)
        or not math.isfinite(maximum)
        or minimum < 0.0
        or minimum >= maximum
    ):
        raise ValueError("标记距离范围必须满足 0 <= min < max")
    if not isinstance(raw_crop, Mapping):
        raise ValueError("crop_reference 必须是映射")
    try:
        crop = {
            key: _integer_value(raw_crop[key], f"crop_reference.{key}")
            for key in ("top_y", "bottom_y", "left_x", "right_x")
        }
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"crop_reference 字段不完整: {error}") from error
    width, height = resolution
    if not (0 <= crop["left_x"] < crop["right_x"] <= width):
        raise ValueError("crop_reference 左右边界无效")
    if not (0 <= crop["top_y"] < crop["bottom_y"] <= height):
        raise ValueError("crop_reference 上下边界无效")
    total = _integer_value(data.get("total_valid_frames", 0), "total_valid_frames")
    if total < 0:
        raise ValueError("total_valid_frames 不能为负数")
    raw_statistics = data.get("calibration_statistics")
    statistics = dict(raw_statistics) if isinstance(raw_statistics, Mapping) else None
    return GripperCalibration(
        marker_size_mm=marker_size,
        dictionary_name=dictionary_name,
        left_finger_tag_id=left_id,
        right_finger_tag_id=right_id,
        min_marker_dist_mm=minimum,
        max_marker_dist_mm=maximum,
        resolution=resolution,
        crop_reference=crop,
        total_valid_frames=total,
        statistics=statistics,
    )


def _normalize_model(model: str) -> str:
    """规范化相机畸变模型名称。"""
    aliases = {
        "fisheye": "fisheye",
        "equidistant": "equidistant",
        "radtan": "radtan",
        "plumb_bob": "plumb_bob",
        "brown_conrady": "brown_conrady",
        "brown-conrady": "brown_conrady",
    }
    if model not in aliases:
        raise ValueError(f"不支持的 distortion_model: {model}")
    return aliases[model]


def _finite_array(
    data: Mapping[str, Any],
    field: str,
    key: str,
    count: Optional[int] = None,
    minimum: Optional[int] = None,
) -> np.ndarray:
    """读取有限浮点数组并检查长度。"""
    try:
        values = np.asarray(data[field], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{key}.{field} 必须是有限数值数组: {error}") from error
    if count is not None and values.size != count:
        raise ValueError(f"{key}.{field} 必须包含 {count} 个数值")
    if minimum is not None and values.size < minimum:
        raise ValueError(f"{key}.{field} 至少需要 {minimum} 个数值")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key}.{field} 必须全部为有限数值")
    return values


def _resolution(data: Mapping[str, Any], key: str) -> Tuple[int, int]:
    """读取相机图像宽高。"""
    try:
        return _valid_resolution(data["resolution"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{key}.resolution 必须是两个正整数: {error}") from error


def _valid_resolution(value: Any) -> Tuple[int, int]:
    """验证宽高为正整数，拒绝隐式截断的小数。"""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("分辨率必须包含宽和高")
    result = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError("分辨率不能使用布尔值")
        number = float(item)
        if not math.isfinite(number) or number <= 0.0 or not number.is_integer():
            raise ValueError("分辨率必须为正整数")
        result.append(int(number))
    return result[0], result[1]


def _integer_value(value: Any, field: str) -> int:
    """读取不带小数部分的有限整数配置。"""
    if isinstance(value, bool):
        raise ValueError(f"{field} 不能是布尔值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} 必须为整数: {error}") from error
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{field} 必须为整数")
    return int(number)


def _corners(value: Any) -> np.ndarray:
    """验证并规范化 ArUco 四角点。"""
    corners = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise ValueError("ArUco 角点必须为四个有限二维坐标")
    return corners


def _required_file(path: str, description: str) -> Path:
    """验证输入文件路径。"""
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    if not isinstance(path, str) or not path.strip():
        raise ValueError(f"{description}路径不能为空")
    result = Path(path).expanduser()
    if not result.is_file():
        raise ValueError(f"{description}文件不存在: {result}")
    return result
