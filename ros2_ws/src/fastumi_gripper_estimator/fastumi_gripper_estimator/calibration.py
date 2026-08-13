"""加载相机标定文件并验证夹爪三维估计使用的毫米范围参数。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraCalibration:
    """保存 Kalibr 鱼眼相机标定结果。

    Attributes:
        camera_matrix: 形状为 ``(3, 3)`` 的针孔投影矩阵。
        distortion_coefficients: 四个 fisheye 兼容畸变系数。
        resolution: 标定图像的宽、高，单位为像素。
    """

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    resolution: Tuple[int, int]


@dataclass(frozen=True)
class GripperDistanceRange:
    """保存夹爪闭合和张开时的标签中心距离。

    Attributes:
        min_distance_mm: 夹爪完全闭合距离，单位为毫米。
        max_distance_mm: 夹爪完全张开距离，单位为毫米。
    """

    min_distance_mm: float
    max_distance_mm: float


def load_camera_calibration(path: str) -> CameraCalibration:
    """从 ToF 或 Kalibr YAML 加载鱼眼标定参数。

    Args:
        path: 标定 YAML 文件路径。

    Returns:
        经过结构和数值验证的相机标定对象。

    Raises:
        ValueError: 路径为空、文件不可读或标定内容无效时抛出。
    """
    calibration_path = _validate_file_path(path, "相机标定")
    try:
        with calibration_path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"读取相机标定 YAML 失败: {error}") from error

    camera_key, camera_data = _select_camera_block(document)
    _validate_camera_model(camera_key, camera_data)
    intrinsics = _load_finite_values(camera_data, "intrinsics", camera_key)
    distortion = _load_finite_values(
        camera_data, "distortion_coeffs", camera_key
    )
    resolution_values = _load_resolution(camera_data, camera_key)

    focal_x, focal_y, center_x, center_y = intrinsics
    if focal_x <= 0.0 or focal_y <= 0.0:
        raise ValueError("相机焦距必须为正数")
    # 针孔投影矩阵，用于鱼眼角点校正后的 PnP。
    camera_matrix = np.asarray(
        [
            [focal_x, 0.0, center_x],
            [0.0, focal_y, center_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return CameraCalibration(
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion.reshape(4, 1),
        resolution=(resolution_values[0], resolution_values[1]),
    )


def _select_camera_block(document) -> Tuple[str, dict]:
    """选择唯一支持的 ToF ``rgb`` 或 Kalibr ``cam0`` 相机块。"""
    if not isinstance(document, dict):
        raise ValueError("相机标定 YAML 顶层必须是映射")
    has_rgb = "rgb" in document
    has_cam0 = "cam0" in document
    if has_rgb and has_cam0:
        raise ValueError("相机标定 YAML 同时包含 rgb 和 cam0，无法确定标定来源")
    if not has_rgb and not has_cam0:
        raise ValueError("相机标定 YAML 必须包含 rgb 或 cam0")

    camera_key = "rgb" if has_rgb else "cam0"
    camera_data = document[camera_key]
    if not isinstance(camera_data, dict):
        raise ValueError(f"{camera_key} 必须是相机参数映射")
    return camera_key, camera_data


def _validate_camera_model(camera_key: str, camera_data: dict) -> None:
    """验证不同来源标定文件的相机和畸变模型约束。"""
    try:
        distortion_model = str(camera_data["distortion_model"])
    except KeyError as error:
        raise ValueError(
            f"{camera_key} 缺少 distortion_model 字段"
        ) from error

    if camera_key == "rgb":
        if distortion_model != "fisheye":
            raise ValueError(
                "仅支持 ToF rgb.fisheye 畸变模型，"
                f"当前为 {distortion_model}"
            )
        return

    try:
        camera_model = str(camera_data["camera_model"])
    except KeyError as error:
        raise ValueError("cam0 缺少 camera_model 字段") from error
    if camera_model != "pinhole":
        raise ValueError(f"仅支持 cam0.pinhole 相机模型，当前为 {camera_model}")
    if distortion_model != "equidistant":
        raise ValueError(
            "仅支持 cam0.equidistant 鱼眼畸变模型，"
            f"当前为 {distortion_model}"
        )


def _load_finite_values(
    camera_data: dict,
    field: str,
    camera_key: str,
) -> np.ndarray:
    """读取恰含四个有限数值的相机参数数组。"""
    try:
        values = np.asarray(camera_data[field], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{camera_key}.{field} 必须包含四个有限数值: {error}"
        ) from error
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{camera_key}.{field} 必须包含四个有限数值")
    return values


def _load_resolution(camera_data: dict, camera_key: str) -> Tuple[int, int]:
    """读取两个正整数像素尺寸，拒绝隐式截断的小数值。"""
    try:
        raw_resolution = camera_data["resolution"]
        if not isinstance(raw_resolution, (list, tuple)):
            raise ValueError("必须是 YAML 序列")
        if len(raw_resolution) != 2:
            raise ValueError("长度必须为 2")
        resolution_values = tuple(
            _validate_resolution_value(value) for value in raw_resolution
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{camera_key}.resolution 必须包含两个正整数像素值: {error}"
        ) from error
    return resolution_values


def _validate_resolution_value(value) -> int:
    """验证单个分辨率值为有限且无小数部分的正数。"""
    if isinstance(value, bool):
        raise ValueError("布尔值不能作为分辨率")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"值必须为数值: {error}") from error
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError("值必须为正有限数")
    if not numeric_value.is_integer():
        raise ValueError("值必须为整数")
    return int(numeric_value)


def validate_gripper_distance_range(
    min_distance_mm: float,
    max_distance_mm: float,
) -> GripperDistanceRange:
    """验证 ROS 参数提供的夹爪毫米距离范围。

    Args:
        min_distance_mm: 夹爪完全闭合时的标签中心距离，单位为毫米。
        max_distance_mm: 夹爪完全张开时的标签中心距离，单位为毫米。

    Returns:
        经过验证的闭合和张开距离。

    Raises:
        ValueError: 参数无法转换为浮点数或距离范围无效时抛出。
    """
    try:
        validated_min_distance_mm = float(min_distance_mm)
        validated_max_distance_mm = float(max_distance_mm)
    except (TypeError, ValueError) as error:
        raise ValueError(f"夹爪毫米范围必须为数值: {error}") from error

    if not np.isfinite(validated_min_distance_mm) or not np.isfinite(
        validated_max_distance_mm
    ):
        raise ValueError("夹爪毫米范围必须为有限数值")
    if validated_min_distance_mm < 0.0:
        raise ValueError("min_marker_dist_mm 不能为负数")
    if validated_min_distance_mm >= validated_max_distance_mm:
        raise ValueError("min_marker_dist_mm 必须小于 max_marker_dist_mm")
    return GripperDistanceRange(
        min_distance_mm=validated_min_distance_mm,
        max_distance_mm=validated_max_distance_mm,
    )


def _validate_file_path(path: str, description: str) -> Path:
    """验证必需配置文件路径。

    Args:
        path: 待验证的文件路径。
        description: 用于错误信息的文件说明。

    Returns:
        存在且为普通文件的路径对象。

    Raises:
        ValueError: 路径为空、不存在或不是普通文件时抛出。
    """
    if not path or not path.strip():
        raise ValueError(f"{description}文件路径不能为空")
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise ValueError(f"{description}文件不存在: {file_path}")
    return file_path
