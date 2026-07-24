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
        distortion_coefficients: 四个 equidistant 鱼眼畸变系数。
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
    """从 Kalibr YAML 加载 equidistant 鱼眼标定参数。

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

    try:
        camera_data = document["cam0"]
        camera_model = str(camera_data["camera_model"])
        distortion_model = str(camera_data["distortion_model"])
        intrinsics = np.asarray(camera_data["intrinsics"], dtype=np.float64)
        distortion = np.asarray(
            camera_data["distortion_coeffs"], dtype=np.float64
        )
        resolution_values = tuple(
            int(value) for value in camera_data["resolution"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"相机标定 YAML 缺少有效的 cam0 字段: {error}") from error

    if camera_model != "pinhole":
        raise ValueError(f"仅支持 pinhole 相机模型，当前为 {camera_model}")
    if distortion_model != "equidistant":
        raise ValueError(
            f"仅支持 equidistant 鱼眼畸变模型，当前为 {distortion_model}"
        )
    if intrinsics.shape != (4,) or not np.all(np.isfinite(intrinsics)):
        raise ValueError("cam0.intrinsics 必须包含四个有限数值")
    if distortion.shape != (4,) or not np.all(np.isfinite(distortion)):
        raise ValueError("cam0.distortion_coeffs 必须包含四个有限数值")
    if len(resolution_values) != 2 or any(
        value <= 0 for value in resolution_values
    ):
        raise ValueError("cam0.resolution 必须包含正数宽度和高度")

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
