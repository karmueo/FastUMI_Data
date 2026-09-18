"""加载相机标定，保存夹爪范围标定，并提供稳健的采样统计。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple

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
    """累积双 ArUco 有效帧并按参考脚本生成范围和 ROI。"""

    def __init__(self) -> None:
        """初始化空采样器。"""
        self.reset()

    def reset(self) -> None:
        """清空距离、角点和图像尺寸采样。"""
        self.distances_mm = []
        self.top_y = []
        self.bottom_y = []
        self.left_x = []
        self.right_x = []
        self.resolution: Optional[Tuple[int, int]] = None

    @property
    def count(self) -> int:
        """返回有效双码帧数量。"""
        return len(self.distances_mm)

    def add(
        self,
        distance_mm: float,
        left_corners: np.ndarray,
        right_corners: np.ndarray,
        resolution: Tuple[int, int],
    ) -> None:
        """加入一帧有效双码检测及其四角点。"""
        distance = float(distance_mm)
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError("标记距离必须为非负有限数")
        left = _corners(left_corners)
        right = _corners(right_corners)
        if self.resolution is None:
            self.resolution = _valid_resolution(resolution)
        elif tuple(resolution) != self.resolution:
            raise ValueError("采样帧分辨率不一致")
        self.distances_mm.append(distance)
        all_corners = np.vstack((left, right))
        self.top_y.append(float(np.min(all_corners[:, 1])))
        self.bottom_y.append(float(np.max(all_corners[:, 1])))
        self.left_x.append(float(np.min(left[:, 0])))
        self.right_x.append(float(np.max(right[:, 0])))

    def build(
        self,
        marker_size_mm: float,
        dictionary_name: str,
        left_finger_tag_id: int,
        right_finger_tag_id: int,
    ) -> GripperCalibration:
        """根据采样结果生成可写入的标定对象。"""
        if self.count == 0 or self.resolution is None:
            raise ValueError("没有有效的双 ArUco 标定样本")
        if not self.left_x or not self.right_x:
            raise ValueError("无法从标记角点生成有效 ROI")
        minimum = float(np.min(self.distances_mm))
        maximum = float(np.max(self.distances_mm))
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum >= maximum
        ):
            raise ValueError("标定距离范围必须包含两个不同的有限端点")
        crop = {
            "top_y": int(np.percentile(self.top_y, 1.0)),
            "bottom_y": int(np.percentile(self.bottom_y, 99.0)),
            "left_x": int(np.percentile(self.left_x, 1.0)),
            "right_x": int(np.percentile(self.right_x, 99.0)),
        }
        width, height = self.resolution
        if not (0 <= crop["left_x"] < crop["right_x"] <= width):
            raise ValueError("标定 ROI 的左右边界无效")
        if not (0 <= crop["top_y"] < crop["bottom_y"] <= height):
            raise ValueError("标定 ROI 的上下边界无效")
        return GripperCalibration(
            marker_size_mm=float(marker_size_mm),
            dictionary_name=str(dictionary_name),
            left_finger_tag_id=int(left_finger_tag_id),
            right_finger_tag_id=int(right_finger_tag_id),
            min_marker_dist_mm=minimum,
            max_marker_dist_mm=maximum,
            resolution=self.resolution,
            crop_reference=crop,
            total_valid_frames=self.count,
        )


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
