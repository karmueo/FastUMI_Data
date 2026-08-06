"""解析并严格校验双 ArUco 到夹爪中心 TCP 的标定配置。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    """把 YAML 节点校验为映射。"""
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} 必须是 YAML 映射")
    return value


def _finite_float(value: Any, description: str) -> float:
    """读取一个有限浮点数。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} 必须是有限数值") from error
    if not math.isfinite(number):
        raise ValueError(f"{description} 必须是有限数值")
    return number


def _positive_float(value: Any, description: str) -> float:
    """读取一个有限正浮点数。"""
    number = _finite_float(value, description)
    if number <= 0.0:
        raise ValueError(f"{description} 必须为正数")
    return number


def _integer(value: Any, description: str) -> int:
    """读取一个不带小数的整数。"""
    if isinstance(value, bool):
        raise ValueError(f"{description} 必须是整数")
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} 必须是整数") from error
    if number != value:
        raise ValueError(f"{description} 必须是整数")
    return number


def _read_vector(value: Any, size: int, description: str) -> np.ndarray:
    """读取指定长度的有限浮点向量。"""
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} 必须包含 {size} 个数值") from error
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{description} 必须包含 {size} 个有限数值")
    return vector


@dataclass(frozen=True)
class ArucoSpec:
    """保存 ArUco 字典、标签尺寸和双标签 ID。"""

    dictionary_name: str
    marker_size_m: float
    tag0_id: int
    tag1_id: int

    def __post_init__(self) -> None:
        """校验字典、标签尺寸和标签顺序。"""
        if not self.dictionary_name:
            raise ValueError("ArUco 字典不能为空")
        dictionary_constant = getattr(cv2.aruco, self.dictionary_name, None)
        if dictionary_constant is None:
            raise ValueError(f"ArUco 字典不存在: {self.dictionary_name}")
        try:
            cv2.aruco.getPredefinedDictionary(dictionary_constant)
        except (AttributeError, cv2.error) as error:
            raise ValueError(
                f"ArUco 字典不存在: {self.dictionary_name}"
            ) from error
        if not math.isfinite(self.marker_size_m) or self.marker_size_m <= 0.0:
            raise ValueError("ArUco marker_size_m 必须为正数")
        if self.tag0_id == self.tag1_id:
            raise ValueError("ArUco ID 必须不同")
        if self.tag0_id < 0 or self.tag1_id < 0:
            raise ValueError("ArUco ID 不能为负数")


@dataclass(frozen=True)
class PairFrameSpec:
    """保存双标签 pair 坐标系的轴向约定。"""

    y_axis_from_tag_id: int
    y_axis_to_tag_id: int
    z_axis_from_marker_normals: bool
    marker_normal_sign: int

    def __post_init__(self) -> None:
        """校验轴向来源、法向开关和法向符号。"""
        if self.y_axis_from_tag_id == self.y_axis_to_tag_id:
            raise ValueError("pair 的 Y 轴起止 ID 必须不同")
        if not self.z_axis_from_marker_normals:
            raise ValueError("pair 必须使用 ArUco 法向构造 Z 轴")
        if self.marker_normal_sign not in (-1, 1):
            raise ValueError("marker_normal_sign 符号只能为 ±1")


@dataclass(frozen=True)
class MotionModel:
    """保存夹爪开合到双标签中心距离的运动模型。"""

    type: str
    openness_definition: str
    closed_tag_center_distance_m: float
    open_tag_center_distance_m: float

    def __post_init__(self) -> None:
        """校验当前支持的对称平行线性模型和距离范围。"""
        if self.type != "symmetric_parallel_linear":
            raise ValueError("motion_model.type 只支持 symmetric_parallel_linear")
        if self.openness_definition != "0_closed_1_open":
            raise ValueError(
                "motion_model.openness_definition 只支持 0_closed_1_open"
            )
        closed = self.closed_tag_center_distance_m
        opened = self.open_tag_center_distance_m
        if not math.isfinite(closed) or not math.isfinite(opened):
            raise ValueError("开合 tag 中心距离必须是有限数")
        if closed <= 0.0 or opened <= 0.0:
            raise ValueError("开合 tag 中心距离必须为正数")
        if closed >= opened:
            raise ValueError("闭合 tag 中心距离必须小于全开距离")


@dataclass(frozen=True)
class ArucoTcpConfig:
    """保存双 ArUco→TCP 配置和源文件溯源信息。"""

    schema_version: int
    fixture_version: str
    aruco: ArucoSpec
    rectification_projection: str
    pair_frame: PairFrameSpec
    full_open_tag_center_distance_m: float
    pair_from_tcp: np.ndarray
    motion_model: MotionModel
    source_path: str
    source_sha256: str

    def __post_init__(self) -> None:
        """校验配置组合关系并冻结齐次变换数组。"""
        if self.schema_version != 2:
            raise ValueError("schema_version 必须为 2")
        if not self.fixture_version:
            raise ValueError("fixture_version 不能为空")
        if self.rectification_projection != "reuse_kalibr_intrinsics":
            raise ValueError(
                "rectification 投影只支持 reuse_kalibr_intrinsics"
            )
        if not math.isfinite(self.full_open_tag_center_distance_m):
            raise ValueError("全开 tag 中心距离必须是有限数")
        if not np.isclose(
            self.full_open_tag_center_distance_m,
            self.motion_model.open_tag_center_distance_m,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("full-open 距离必须与运动模型全开距离一致")
        matrix = np.array(self.pair_from_tcp, dtype=np.float64, copy=True)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("pair_from_tcp 必须是有限的 4x4 变换")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError("pair_from_tcp 齐次矩阵末行必须为 [0,0,0,1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8):
            raise ValueError("pair_from_tcp 旋转必须正交")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-8):
            raise ValueError("pair_from_tcp 旋转必须右手")
        matrix.setflags(write=False)
        object.__setattr__(self, "pair_from_tcp", matrix)

    def expected_half_distance_m(self, openness: float) -> float:
        """根据开度计算单枚标签到 pair 中心的期望距离。"""
        value = _finite_float(openness, "openness")
        if value < 0.0 or value > 1.0:
            raise ValueError("openness 必须位于 [0, 1]")
        distance = (
            self.motion_model.closed_tag_center_distance_m
            + value
            * (
                self.motion_model.open_tag_center_distance_m
                - self.motion_model.closed_tag_center_distance_m
            )
        )
        return distance / 2.0


def load_aruco_tcp_config(path: str) -> ArucoTcpConfig:
    """加载双 ArUco→TCP YAML 并执行全部安全与几何校验。"""
    config_path = Path(path)
    try:
        file_bytes = config_path.read_bytes()
        document = yaml.safe_load(file_bytes)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取 ArUco TCP 配置 {path}: {error}") from error
    root = _mapping(document, "ArUco TCP 配置")
    legacy_fields = {"verified", "calibration_verified"}.intersection(root)
    if legacy_fields:
        names = ", ".join(sorted(legacy_fields))
        raise ValueError(f"文件包含已删除字段: {names}")
    schema_version = root.get("schema_version")
    if type(schema_version) is not int or schema_version != 2:
        raise ValueError("文件 schema_version 必须为 2；旧 v1 不再兼容")
    fixture_version = str(root.get("fixture_version", "")).strip()
    if not fixture_version:
        raise ValueError("fixture_version 不能为空")

    aruco_document = _mapping(root.get("aruco"), "aruco")
    aruco = ArucoSpec(
        dictionary_name=str(aruco_document.get("dictionary_name", "")).strip(),
        marker_size_m=_positive_float(
            aruco_document.get("marker_size_m"), "marker_size_m"
        ),
        tag0_id=_integer(aruco_document.get("tag0_id"), "tag0_id"),
        tag1_id=_integer(aruco_document.get("tag1_id"), "tag1_id"),
    )

    rectification = _mapping(root.get("rectification"), "rectification")
    projection = str(rectification.get("projection", "")).strip()

    pair_document = _mapping(root.get("pair_frame"), "pair_frame")
    pair_frame = PairFrameSpec(
        y_axis_from_tag_id=_integer(
            pair_document.get("y_axis_from_tag_id"),
            "y_axis_from_tag_id",
        ),
        y_axis_to_tag_id=_integer(
            pair_document.get("y_axis_to_tag_id"), "y_axis_to_tag_id"
        ),
        z_axis_from_marker_normals=pair_document.get(
            "z_axis_from_marker_normals"
        ),
        marker_normal_sign=_integer(
            pair_document.get("marker_normal_sign"), "marker_normal_sign"
        ),
    )
    expected_ids = {aruco.tag0_id, aruco.tag1_id}
    if {
        pair_frame.y_axis_from_tag_id,
        pair_frame.y_axis_to_tag_id,
    } != expected_ids:
        raise ValueError("pair Y 轴必须使用 ArUco tag0_id 和 tag1_id")
    if (
        pair_frame.y_axis_from_tag_id != aruco.tag0_id
        or pair_frame.y_axis_to_tag_id != aruco.tag1_id
    ):
        raise ValueError("pair Y 轴必须按 ID 0 -> ID 1 的顺序配置")

    full_open = _mapping(root.get("full_open_geometry"), "full_open_geometry")
    full_open_distance = _positive_float(
        full_open.get("tag_center_distance_m"), "全开 tag 中心距离"
    )
    pair_from_tcp_document = _mapping(
        full_open.get("pair_from_tcp"), "pair_from_tcp"
    )
    translation = _read_vector(
        pair_from_tcp_document.get("translation_m"), 3, "pair_from_tcp.translation_m"
    )
    quaternion = _read_vector(
        pair_from_tcp_document.get("quaternion_xyzw"),
        4,
        "pair_from_tcp.quaternion_xyzw",
    )
    quaternion_norm = float(np.linalg.norm(quaternion))
    if not np.isclose(quaternion_norm, 1.0, rtol=0.0, atol=1.0e-6):
        raise ValueError("pair_from_tcp 必须使用单位四元数")
    pair_from_tcp = np.eye(4, dtype=np.float64)
    pair_from_tcp[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    pair_from_tcp[:3, 3] = translation

    motion_document = _mapping(root.get("motion_model"), "motion_model")
    motion_model = MotionModel(
        type=str(motion_document.get("type", "")).strip(),
        openness_definition=str(
            motion_document.get("openness_definition", "")
        ).strip(),
        closed_tag_center_distance_m=_positive_float(
            motion_document.get("closed_tag_center_distance_m"),
            "闭合 tag 中心距离",
        ),
        open_tag_center_distance_m=_positive_float(
            motion_document.get("open_tag_center_distance_m"),
            "全开 tag 中心距离",
        ),
    )
    return ArucoTcpConfig(
        schema_version=schema_version,
        fixture_version=fixture_version,
        aruco=aruco,
        rectification_projection=projection,
        pair_frame=pair_frame,
        full_open_tag_center_distance_m=full_open_distance,
        pair_from_tcp=pair_from_tcp,
        motion_model=motion_model,
        source_path=str(config_path.resolve()),
        source_sha256=hashlib.sha256(file_bytes).hexdigest(),
    )
