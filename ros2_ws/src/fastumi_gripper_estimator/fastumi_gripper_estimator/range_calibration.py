"""提供夹爪开合端点距离的纯统计、配置加载和安全 YAML 输出功能。"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

from fastumi_gripper_estimator.calibration import load_camera_calibration
from fastumi_gripper_estimator.estimator import GripperOpennessEstimator
import numpy as np
import yaml

# 以 MAD 估算正态分布标准差的常用比例系数。
MAD_TO_ROBUST_SIGMA = 1.4826
# 每轮保留距离与中位数的最大稳健标准差倍数。
MAD_REJECTION_SIGMA = 3.5


class CalibrationQualityError(ValueError):
    """表示采样统计未满足可写入标定参数的质量阈值。"""


@dataclass(frozen=True)
class EndpointStatistics:
    """保存单个夹爪端点的稳健距离统计。

    Attributes:
        estimate_mm: 保留样本的中位数距离，单位为毫米。
        raw_count: 分配到该端点的有效原始样本数。
        retained_count: MAD 迭代剔除后保留的样本数。
        mad_mm: 保留样本相对于中位数的中位绝对偏差，单位为毫米。
        robust_sigma_mm: 由 MAD 换算的稳健标准差，单位为毫米。
    """

    estimate_mm: float
    raw_count: int
    retained_count: int
    mad_mm: float
    robust_sigma_mm: float


@dataclass(frozen=True)
class RangeCalibrationStatistics:
    """保存两端夹爪距离的完整统计与数据质量信息。

    Attributes:
        closed: 闭合端点统计。
        opened: 张开端点统计。
        total_count: 尝试处理的全部图像帧数。
        valid_count: 检测和三维距离均有效且有限的帧数。
        validity_rate: 有效帧占全部处理帧的比例。
    """

    closed: EndpointStatistics
    opened: EndpointStatistics
    total_count: int
    valid_count: int
    validity_rate: float


def calculate_bag_statistics(
    distances_mm: Iterable[Optional[float]], total_count: int
) -> RangeCalibrationStatistics:
    """按确定性二均值和 MAD 剔除计算离线 bag 的两个端点。"""
    finite_values = _finite_distances(distances_mm)
    _validate_total_count(total_count, len(finite_values))
    if len(finite_values) < 2:
        raise CalibrationQualityError("有效距离不足，无法划分闭合和张开端点")
    closed_values, open_values = _kmeans_two_clusters(finite_values)
    return RangeCalibrationStatistics(
        closed=_endpoint_statistics(closed_values),
        opened=_endpoint_statistics(open_values),
        total_count=total_count,
        valid_count=len(finite_values),
        validity_rate=len(finite_values) / total_count,
    )


def calculate_live_statistics(
    closed_distances_mm: Iterable[Optional[float]],
    closed_total_count: int,
    open_distances_mm: Iterable[Optional[float]],
    open_total_count: int,
) -> RangeCalibrationStatistics:
    """计算按操作员顺序采集的闭合和张开端点统计。"""
    closed_values = _finite_distances(closed_distances_mm)
    open_values = _finite_distances(open_distances_mm)
    _validate_total_count(closed_total_count, len(closed_values))
    _validate_total_count(open_total_count, len(open_values))
    total_count = closed_total_count + open_total_count
    valid_count = len(closed_values) + len(open_values)
    if total_count == 0:
        raise CalibrationQualityError("实时采样未收到任何图像帧")
    return RangeCalibrationStatistics(
        closed=_endpoint_statistics(closed_values),
        opened=_endpoint_statistics(open_values),
        total_count=total_count,
        valid_count=valid_count,
        validity_rate=valid_count / total_count,
    )


def validate_calibration_quality(
    statistics: RangeCalibrationStatistics, is_bag: bool
) -> None:
    """验证标定统计满足生产写入 YAML 前的默认质量阈值。"""
    if statistics.validity_rate < 0.50:
        raise CalibrationQualityError(
            "有效帧比例不足：valid/total 必须至少为 0.50"
        )
    for name, endpoint in (("闭合端点", statistics.closed),
                           ("张开端点", statistics.opened)):
        if endpoint.retained_count < 20:
            raise CalibrationQualityError(
                f"{name}保留样本不足：retained 必须至少为 20"
            )
        if endpoint.robust_sigma_mm > 2.0:
            raise CalibrationQualityError(
                f"{name}离散度过大：robust sigma 必须不超过 2 mm"
            )
        if is_bag and endpoint.raw_count < statistics.valid_count * 0.10:
            raise CalibrationQualityError(
                f"{name}原始聚类占比不足：每个 raw bag cluster "
                "必须至少占有效帧的 10%"
            )
    gap_mm = statistics.opened.estimate_mm - statistics.closed.estimate_mm
    if gap_mm < 20.0:
        raise CalibrationQualityError("端点间距不足：endpoint gap 必须至少为 20 mm")


def load_gripper_parameters(path: str) -> Tuple[dict, MutableMapping]:
    """加载 ROS 参数 YAML，并返回完整文档和可修改参数映射。"""
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ValueError(f"夹爪配置文件不存在: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"读取夹爪配置 YAML 失败: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("夹爪配置 YAML 顶层必须是映射")
    parameters = _find_ros_parameters(document)
    _validate_gripper_parameters(parameters)
    return document, parameters


def create_estimator(
    parameters: Mapping, camera_calibration_path: str
) -> GripperOpennessEstimator:
    """由 ROS 配置和相机标定创建不影响采样的夹爪距离估计器。"""
    _validate_gripper_parameters(parameters)
    calibration = load_camera_calibration(camera_calibration_path)
    gripper_range = parameters["gripper_range"]
    return GripperOpennessEstimator(
        camera_matrix=calibration.camera_matrix,
        distortion_coefficients=calibration.distortion_coefficients,
        closed_distance_mm=float(gripper_range["min_marker_dist_mm"]),
        open_distance_mm=float(gripper_range["max_marker_dist_mm"]),
        marker_size_mm=float(parameters["marker_size_mm"]),
        # 标定只读取 distance_mm，1.0 使平滑状态不影响采样结果。
        smoothing_alpha=1.0,
        dictionary_name=str(parameters["dictionary_name"]),
        left_marker_id=int(gripper_range["left_finger_tag_id"]),
        right_marker_id=int(gripper_range["right_finger_tag_id"]),
        roi_ratios=tuple(float(value) for value in parameters["roi_ratios"]),
        image_resolution=calibration.resolution,
    )


def write_calibrated_config(
    input_path: str,
    output_path: str,
    statistics: RangeCalibrationStatistics,
    force: bool = False,
) -> Path:
    """原子写入完整 ROS YAML 副本，仅更新两个标定距离。"""
    source, target = preflight_output_path(input_path, output_path, force)
    document, parameters = load_gripper_parameters(str(source))
    gripper_range = parameters["gripper_range"]
    gripper_range["min_marker_dist_mm"] = round(
        statistics.closed.estimate_mm, 3
    )
    gripper_range["max_marker_dist_mm"] = round(statistics.opened.estimate_mm, 3)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(target.parent),
            prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_name = stream.name
            yaml.safe_dump(
                document, stream, allow_unicode=True, sort_keys=False,
                default_flow_style=False
            )
            stream.flush()
            os.fsync(stream.fileno())
        _commit_output_file(temporary_name, target, force)
        temporary_name = None
    except (OSError, yaml.YAMLError) as error:
        _unlink_temporary_file(temporary_name)
        if isinstance(error, FileExistsError):
            raise ValueError(
                f"输出文件已存在，请使用 --force 覆盖: {target}"
            ) from error
        raise ValueError(f"原子写入标定配置失败: {error}") from error
    finally:
        _unlink_temporary_file(temporary_name)
    return target


def preflight_output_path(
    input_path: str, output_path: str, force: bool = False
) -> Tuple[Path, Path]:
    """在采样前验证输出路径绝不会覆盖输入且符合覆盖策略。"""
    source = Path(input_path).expanduser().resolve()
    requested_target = Path(output_path).expanduser()
    if not requested_target.name:
        raise ValueError("输出路径必须包含文件名")
    target_parent = requested_target.parent.resolve()
    target = target_parent / requested_target.name
    if source == target:
        raise ValueError("输出路径不能与输入配置路径相同")
    if target.is_symlink():
        raise ValueError(f"输出路径不能是符号链接: {target}")
    if os.path.lexists(target) and not force:
        raise ValueError(f"输出文件已存在，请使用 --force 覆盖: {target}")
    return source, target


def _commit_output_file(temporary_name: str, target: Path, force: bool) -> None:
    """提交同目录临时文件，并在无覆盖模式下保持 no-clobber 语义。"""
    if target.is_symlink():
        raise ValueError(f"输出路径不能是符号链接: {target}")
    if force:
        os.replace(temporary_name, target)
        return
    try:
        os.link(temporary_name, target)
    except BaseException:
        if not _is_committed_hard_link(temporary_name, target):
            raise
    try:
        os.unlink(temporary_name)
    except (OSError, KeyboardInterrupt):
        pass


def _is_committed_hard_link(temporary_name: str, target: Path) -> bool:
    """确认硬链接提交已生效，供提交点紧邻异常时判定成功。"""
    try:
        return os.path.samefile(temporary_name, target)
    except OSError:
        return False


def _unlink_temporary_file(temporary_name: Optional[str]) -> None:
    """尽力删除尚未提交的临时文件，不屏蔽原始写入错误。"""
    if temporary_name is None:
        return
    try:
        Path(temporary_name).unlink(missing_ok=True)
    except OSError:
        pass


def _finite_distances(values: Iterable[Optional[float]]) -> np.ndarray:
    """返回输入中可转换为有限浮点数的距离数组。"""
    finite_values = []
    for value in values:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(numeric_value):
            finite_values.append(numeric_value)
    return np.asarray(finite_values, dtype=np.float64)


def _validate_total_count(total_count: int, valid_count: int) -> None:
    """验证采样总数能表示提供的有效距离数。"""
    if isinstance(total_count, bool) or not isinstance(total_count, int):
        raise CalibrationQualityError("total_count 必须是非负整数")
    if total_count < valid_count:
        raise CalibrationQualityError("total_count 不能小于有效距离数")


def _kmeans_two_clusters(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """以最小和最大值初始化，确定性运行一维二均值并按中心排序。"""
    if float(np.min(values)) == float(np.max(values)):
        raise CalibrationQualityError("有效距离没有两个可区分的端点")
    centers = np.asarray([np.min(values), np.max(values)], dtype=np.float64)
    labels = np.zeros(values.size, dtype=np.int8)
    for _ in range(100):
        new_labels = (np.abs(values - centers[1]) <
                      np.abs(values - centers[0])).astype(np.int8)
        if not np.any(new_labels == 0) or not np.any(new_labels == 1):
            raise CalibrationQualityError("二均值聚类产生空端点")
        new_centers = np.asarray(
            [np.mean(values[new_labels == index]) for index in (0, 1)],
            dtype=np.float64,
        )
        if np.array_equal(new_labels, labels) and np.allclose(
                new_centers, centers, rtol=0.0, atol=1e-12):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers
    clusters = (values[labels == 0], values[labels == 1])
    if centers[0] <= centers[1]:
        return clusters
    return clusters[1], clusters[0]


def _endpoint_statistics(values: Sequence[float]) -> EndpointStatistics:
    """对一个端点执行迭代中位数/MAD 剔除并生成诊断信息。"""
    raw_values = np.asarray(values, dtype=np.float64)
    if raw_values.size == 0:
        raise CalibrationQualityError("端点没有有效原始样本")
    retained = raw_values
    for _ in range(100):
        median = float(np.median(retained))
        mad = float(np.median(np.abs(retained - median)))
        if mad == 0.0:
            keep = np.isclose(retained, median, rtol=0.0, atol=1e-12)
        else:
            robust_sigma = MAD_TO_ROBUST_SIGMA * mad
            keep = np.abs(retained - median) <= MAD_REJECTION_SIGMA * robust_sigma
        updated = retained[keep]
        if updated.size == 0:
            raise CalibrationQualityError("MAD 异常值剔除后端点没有保留样本")
        if updated.size == retained.size:
            break
        retained = updated
    median = float(np.median(retained))
    mad = float(np.median(np.abs(retained - median)))
    return EndpointStatistics(
        estimate_mm=median,
        raw_count=int(raw_values.size),
        retained_count=int(retained.size),
        mad_mm=mad,
        robust_sigma_mm=MAD_TO_ROBUST_SIGMA * mad,
    )


def _find_ros_parameters(document: MutableMapping) -> MutableMapping:
    """定位标准 ROS 2 YAML 中唯一可用的 ``ros__parameters`` 映射。"""
    direct_parameters = document.get("ros__parameters")
    if isinstance(direct_parameters, dict):
        return direct_parameters
    candidates = []
    for node_data in document.values():
        if isinstance(node_data, dict) and isinstance(
                node_data.get("ros__parameters"), dict):
            candidates.append(node_data["ros__parameters"])
    if len(candidates) != 1:
        raise ValueError("配置必须包含唯一的 ros__parameters 参数映射")
    return candidates[0]


def _validate_gripper_parameters(parameters: Mapping) -> None:
    """验证创建标定估计器所需的既有 ROS 参数。"""
    required = ("image_topic", "marker_size_mm", "dictionary_name",
                "roi_ratios", "gripper_range")
    missing = [key for key in required if key not in parameters]
    if missing:
        raise ValueError(f"夹爪配置缺少必需参数: {', '.join(missing)}")
    gripper_range = parameters["gripper_range"]
    if not isinstance(gripper_range, Mapping):
        raise ValueError("gripper_range 必须是映射")
    range_fields = ("left_finger_tag_id", "right_finger_tag_id",
                    "min_marker_dist_mm", "max_marker_dist_mm")
    missing_range = [key for key in range_fields if key not in gripper_range]
    if missing_range:
        raise ValueError("gripper_range 缺少必需参数: " +
                         ", ".join(missing_range))
