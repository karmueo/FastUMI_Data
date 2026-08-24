"""扫描 Tracker 时间偏移并联合优化目标板外参、位姿和时间。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from fastumi_data.tracker_camera_bag import (
    TrackerTimeline,
    interpolate_world_from_tracker,
    tracker_status_valid_for_interval,
)
from fastumi_data.tracker_camera_config import FisheyeCameraModel
from fastumi_data.tracker_camera_handeye import (
    HandEyeCandidate,
    HandEyeEstimationError,
    board_closure_errors,
    board_transforms,
    select_handeye_seed,
    solve_handeye_candidates,
)
from fastumi_data.tracker_camera_pnp import project_fisheye_points


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    **details: object,
) -> None:
    """在调用方提供回调时发出结构化进度事件。"""
    if callback is not None:
        callback(stage, details)


@dataclass(frozen=True)
class CalibrationSample:
    """保存一帧联合优化所需的原始角点和 PnP 初值。

    ``object_points_m`` 为板坐标系下 ``(N, 3)`` 米制点，
    ``image_points_px`` 为对应的原始鱼眼 ``(N, 2)`` 像素。``tag_count``
    保留 AprilGrid 的标签计数语义；棋盘格使用 ``target_type`` 和
    ``feature_count`` 表示完整角点观测。
    """

    timestamp_ns: int
    object_points_m: np.ndarray
    image_points_px: np.ndarray
    camera_from_board: np.ndarray
    tag_count: int | None = None
    target_type: str = "aprilgrid"
    feature_count: int | None = None

    def __post_init__(self) -> None:
        """复制并校验观测数组，防止优化期间输入被修改。"""
        object_points = np.array(
            self.object_points_m, dtype=np.float64, copy=True
        )
        image_points = np.array(
            self.image_points_px, dtype=np.float64, copy=True
        )
        camera_from_board = np.array(
            self.camera_from_board, dtype=np.float64, copy=True
        )
        if object_points.ndim != 2 or object_points.shape[1:] != (3,):
            raise ValueError("目标点必须是 Nx3 数组")
        if image_points.shape != (len(object_points), 2):
            raise ValueError("图像点必须是与目标点等长的 Nx2 数组")
        if camera_from_board.shape != (4, 4):
            raise ValueError("camera_from_board 必须是 4x4 矩阵")
        if not all(
            np.all(np.isfinite(array))
            for array in (object_points, image_points, camera_from_board)
        ):
            raise ValueError("标定样本必须只包含有限数值")
        if self.target_type not in {"aprilgrid", "checkerboard"}:
            raise ValueError("target_type 必须是 aprilgrid 或 checkerboard")
        if self.target_type == "aprilgrid":
            if self.tag_count is None or int(self.tag_count) <= 0:
                raise ValueError("AprilGrid tag_count 必须为正整数")
            tag_count = int(self.tag_count)
        else:
            if self.tag_count not in (None, 0):
                raise ValueError("棋盘格 tag_count 必须为空")
            tag_count = None
        feature_count = (
            len(object_points)
            if self.feature_count is None
            else int(self.feature_count)
        )
        if feature_count != len(object_points) or feature_count <= 0:
            raise ValueError("feature_count 必须等于观测点数量且为正数")
        object_points.setflags(write=False)
        image_points.setflags(write=False)
        camera_from_board.setflags(write=False)
        object.__setattr__(self, "object_points_m", object_points)
        object.__setattr__(self, "image_points_px", image_points)
        object.__setattr__(self, "camera_from_board", camera_from_board)
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "tag_count", tag_count)
        object.__setattr__(self, "target_type", self.target_type)
        object.__setattr__(self, "feature_count", feature_count)


@dataclass(frozen=True)
class OptimizationOptions:
    """保存时间搜索、插值和训练验证划分参数。"""

    time_offset_min_ms: float = -100.0
    time_offset_max_ms: float = 100.0
    coarse_step_ms: float = 2.0
    max_pose_gap_ms: float = 50.0
    validation_fraction: float = 0.2
    temporal_block_seconds: float = 2.0

    def __post_init__(self) -> None:
        """校验搜索范围、步长和验证比例。"""
        if self.time_offset_min_ms > self.time_offset_max_ms:
            raise ValueError("时间偏移最小值不能大于最大值")
        if self.coarse_step_ms <= 0.0:
            raise ValueError("时间粗扫描步长必须为正数")
        if self.max_pose_gap_ms <= 0.0:
            raise ValueError("Tracker 最大插值间隔必须为正数")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction 必须在 0 和 1 之间")
        if self.temporal_block_seconds <= 0.0:
            raise ValueError("时间块长度必须为正数")


@dataclass(frozen=True)
class TimeOffsetScanPoint:
    """保存一个粗扫描时间点的闭环分数和最佳算法。"""

    time_offset_ms: float
    translation_rmse_mm: float
    rotation_rmse_deg: float
    method: str
    valid: bool


@dataclass(frozen=True)
class OptimizationResult:
    """保存联合优化结果、分区索引、指标和时间扫描曲线。"""

    tracker_from_camera: np.ndarray
    camera_from_tracker: np.ndarray
    world_from_board: np.ndarray
    time_offset_ms: float
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    metrics: dict[str, Any]
    scan_points: tuple[TimeOffsetScanPoint, ...]


@dataclass(frozen=True)
class ResidualContext:
    """保存只含训练帧的固定维度鱼眼残差上下文。"""

    samples: tuple[CalibrationSample, ...]
    timeline: TrackerTimeline
    camera: FisheyeCameraModel
    options: OptimizationOptions


def vector_to_transform(parameters: np.ndarray) -> np.ndarray:
    """把旋转向量和平移六参数转换为 4×4 刚体变换。"""
    vector = np.asarray(parameters, dtype=np.float64)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError("刚体参数必须是 6 个有限数值")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    transform[:3, 3] = vector[3:]
    return transform


def transform_to_vector(transform: np.ndarray) -> np.ndarray:
    """把 4×4 刚体变换转换为旋转向量和平移六参数。"""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("刚体变换必须是有限 4x4 矩阵")
    return np.concatenate(
        (
            Rotation.from_matrix(matrix[:3, :3]).as_rotvec(),
            matrix[:3, 3],
        )
    )


def split_temporal_blocks(
    timestamps_ns: Sequence[int],
    block_seconds: float,
    validation_fraction: float,
) -> tuple[list[int], list[int]]:
    """按完整时间块划分训练与验证，避免相邻帧泄漏。"""
    if len(timestamps_ns) < 2:
        raise ValueError("时间块划分至少需要两个样本")
    if block_seconds <= 0.0:
        raise ValueError("block_seconds 必须为正数")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction 必须在 0 和 1 之间")
    timestamps = [int(timestamp) for timestamp in timestamps_ns]
    if any(
        second <= first
        for first, second in zip(timestamps[:-1], timestamps[1:])
    ):
        raise ValueError("标定样本时间戳必须严格递增")
    block_ns = int(round(block_seconds * 1.0e9))
    origin_ns = timestamps[0]
    block_ids = [(timestamp - origin_ns) // block_ns for timestamp in timestamps]
    unique_blocks = list(dict.fromkeys(block_ids))
    if len(unique_blocks) < 2:
        raise ValueError("时间跨度不足两个完整分区块")
    validation_count = max(
        1, int(round(len(unique_blocks) * validation_fraction))
    )
    validation_count = min(validation_count, len(unique_blocks) - 1)
    validation_positions = np.linspace(
        0, len(unique_blocks) - 1, validation_count + 2, dtype=int
    )[1:-1]
    validation_blocks = {
        unique_blocks[position] for position in validation_positions
    }
    train_indices = [
        index
        for index, block_id in enumerate(block_ids)
        if block_id not in validation_blocks
    ]
    validation_indices = [
        index
        for index, block_id in enumerate(block_ids)
        if block_id in validation_blocks
    ]
    return train_indices, validation_indices


def predict_camera_from_board(
    world_from_tracker: np.ndarray,
    tracker_from_camera: np.ndarray,
    world_from_board: np.ndarray,
) -> np.ndarray:
    """由 Tracker、外参和固定板位姿预测 ``^camera T_board``。"""
    return (
        np.linalg.inv(tracker_from_camera)
        @ np.linalg.inv(world_from_tracker)
        @ world_from_board
    )


def parameter_residuals(
    parameters: np.ndarray, context: ResidualContext
) -> np.ndarray:
    """返回全部训练角点的原始鱼眼二维残差。"""
    tracker_from_camera = vector_to_transform(parameters[:6])
    world_from_board = vector_to_transform(parameters[6:12])
    time_offset_ns = int(round(parameters[12] * 1.0e9))
    residuals = []
    for sample in context.samples:
        interpolation = interpolate_world_from_tracker(
            context.timeline.poses,
            sample.timestamp_ns + time_offset_ns,
            context.options.max_pose_gap_ms,
            context.timeline.pose_timestamps_ns,
        )
        if interpolation is None:
            raise RuntimeError(
                "优化样本超出预先验证的 Tracker 时间边界"
            )
        camera_from_board = predict_camera_from_board(
            interpolation[0], tracker_from_camera, world_from_board
        )
        projected = project_fisheye_points(
            sample.object_points_m, camera_from_board, context.camera
        )
        residuals.extend(
            (projected - sample.image_points_px).reshape(-1)
        )
    return np.asarray(residuals, dtype=np.float64)


def _offset_values(options: OptimizationOptions) -> np.ndarray:
    """生成包含搜索上下界的确定性粗扫描毫秒序列。"""
    values = np.arange(
        options.time_offset_min_ms,
        options.time_offset_max_ms + options.coarse_step_ms * 0.5,
        options.coarse_step_ms,
        dtype=np.float64,
    )
    values = values[values <= options.time_offset_max_ms + 1.0e-9]
    if len(values) == 0 or not np.isclose(
        values[-1], options.time_offset_max_ms
    ):
        values = np.append(values, options.time_offset_max_ms)
    return values


def scan_time_offset(
    samples: Sequence[CalibrationSample],
    timeline: TrackerTimeline,
    options: OptimizationOptions,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[TimeOffsetScanPoint], HandEyeCandidate, float]:
    """逐粗网格偏移重插值并以固定板闭环选择 Hand-Eye 初值。"""
    scan_points = []
    successful: list[tuple[TimeOffsetScanPoint, HandEyeCandidate]] = []
    offset_values = _offset_values(options)
    for completed, offset_ms in enumerate(offset_values, start=1):
        world_from_tracker = []
        camera_from_board = []
        offset_ns = int(round(offset_ms * 1.0e6))
        for sample in samples:
            interpolation = interpolate_world_from_tracker(
                timeline.poses,
                sample.timestamp_ns + offset_ns,
                options.max_pose_gap_ms,
                timeline.pose_timestamps_ns,
            )
            if interpolation is None:
                break
            world_from_tracker.append(interpolation[0])
            camera_from_board.append(sample.camera_from_board)
        if len(world_from_tracker) != len(samples):
            point = TimeOffsetScanPoint(
                float(offset_ms), float("inf"), float("inf"), "", False
            )
            scan_points.append(point)
            _emit_progress(
                progress_callback,
                "time_offset_scan",
                completed=completed,
                total=len(offset_values),
                offset_ms=float(offset_ms),
                valid=False,
                best_offset_ms=(
                    min(successful, key=lambda item: item[0].translation_rmse_mm)[
                        0
                    ].time_offset_ms
                    if successful
                    else None
                ),
            )
            continue
        try:
            candidate = select_handeye_seed(
                solve_handeye_candidates(
                    world_from_tracker, camera_from_board
                )
            )
        except (HandEyeEstimationError, ValueError):
            point = TimeOffsetScanPoint(
                float(offset_ms), float("inf"), float("inf"), "", False
            )
            scan_points.append(point)
            _emit_progress(
                progress_callback,
                "time_offset_scan",
                completed=completed,
                total=len(offset_values),
                offset_ms=float(offset_ms),
                valid=False,
                best_offset_ms=(
                    min(successful, key=lambda item: item[0].translation_rmse_mm)[
                        0
                    ].time_offset_ms
                    if successful
                    else None
                ),
            )
            continue
        point = TimeOffsetScanPoint(
            time_offset_ms=float(offset_ms),
            translation_rmse_mm=candidate.translation_rmse_mm,
            rotation_rmse_deg=candidate.rotation_rmse_deg,
            method=candidate.method,
            valid=True,
        )
        scan_points.append(point)
        successful.append((point, candidate))
        current_best = min(
            successful,
            key=lambda item: (
                item[0].translation_rmse_mm,
                item[0].rotation_rmse_deg,
                abs(item[0].time_offset_ms),
            ),
        )[0]
        _emit_progress(
            progress_callback,
            "time_offset_scan",
            completed=completed,
            total=len(offset_values),
            offset_ms=float(offset_ms),
            valid=True,
            best_offset_ms=current_best.time_offset_ms,
        )
    if not successful:
        raise HandEyeEstimationError("时间粗扫描没有产生有效 Hand-Eye 候选")
    best_point, best_candidate = min(
        successful,
        key=lambda item: (
            item[0].translation_rmse_mm,
            item[0].rotation_rmse_deg,
            abs(item[0].time_offset_ms),
        ),
    )
    return scan_points, best_candidate, best_point.time_offset_ms


def _tracker_status_valid_for_interval(
    timeline: TrackerTimeline,
    start_ns: int,
    end_ns: int,
    maximum_delta_ms: float,
) -> bool:
    """检查偏移区间端点和区间内全部 Tracker 状态。"""
    if not timeline.statuses:
        return True
    return tracker_status_valid_for_interval(
        timeline.statuses,
        start_ns,
        end_ns,
        maximum_delta_ms,
        timeline.status_timestamps_ns,
    )


def _samples_valid_for_full_search(
    samples: Sequence[CalibrationSample],
    timeline: TrackerTimeline,
    options: OptimizationOptions,
) -> tuple[list[CalibrationSample], list[int]]:
    """裁掉无法覆盖完整偏移搜索区间的帧并保留原索引。"""
    valid_samples = []
    original_indices = []
    boundary_offsets = (
        int(round(options.time_offset_min_ms * 1.0e6)),
        int(round(options.time_offset_max_ms * 1.0e6)),
    )
    for index, sample in enumerate(samples):
        start_ns = sample.timestamp_ns + boundary_offsets[0]
        end_ns = sample.timestamp_ns + boundary_offsets[1]
        pose_valid = all(
            interpolate_world_from_tracker(
                timeline.poses,
                sample.timestamp_ns + offset_ns,
                options.max_pose_gap_ms,
                timeline.pose_timestamps_ns,
            )
            is not None
            for offset_ns in boundary_offsets
        )
        status_valid = _tracker_status_valid_for_interval(
            timeline,
            start_ns,
            end_ns,
            options.max_pose_gap_ms,
        )
        if pose_valid and status_valid:
            valid_samples.append(sample)
            original_indices.append(index)
    return valid_samples, original_indices


def _pixel_errors(
    samples: Sequence[CalibrationSample],
    timeline: TrackerTimeline,
    camera: FisheyeCameraModel,
    options: OptimizationOptions,
    tracker_from_camera: np.ndarray,
    world_from_board: np.ndarray,
    time_offset_ms: float,
) -> np.ndarray:
    """计算给定样本集合的逐角点原始鱼眼像素误差。"""
    errors = []
    offset_ns = int(round(time_offset_ms * 1.0e6))
    for sample in samples:
        interpolation = interpolate_world_from_tracker(
            timeline.poses,
            sample.timestamp_ns + offset_ns,
            options.max_pose_gap_ms,
            timeline.pose_timestamps_ns,
        )
        if interpolation is None:
            raise RuntimeError("结果评估时 Tracker 插值失败")
        predicted = predict_camera_from_board(
            interpolation[0], tracker_from_camera, world_from_board
        )
        projected = project_fisheye_points(
            sample.object_points_m, predicted, camera
        )
        errors.extend(
            np.linalg.norm(
                projected - sample.image_points_px, axis=1
            ).tolist()
        )
    return np.asarray(errors, dtype=np.float64)


def optimize_spatiotemporal(
    samples: Sequence[CalibrationSample],
    timeline: TrackerTimeline,
    camera: FisheyeCameraModel,
    handeye_seed: np.ndarray,
    board_seed: np.ndarray,
    options: OptimizationOptions,
    progress_callback: ProgressCallback | None = None,
) -> OptimizationResult:
    """粗扫时间后联合优化 ``^tracker T_camera``、固定板和时间偏移。"""
    if len(samples) < 8:
        raise ValueError("联合优化至少需要 8 帧标定样本")
    valid_samples, original_indices = _samples_valid_for_full_search(
        samples, timeline, options
    )
    if len(valid_samples) < 8:
        raise ValueError("完整时间搜索范围内可插值的标定帧不足 8")
    train_local, validation_local = split_temporal_blocks(
        [sample.timestamp_ns for sample in valid_samples],
        options.temporal_block_seconds,
        options.validation_fraction,
    )
    training_samples = [valid_samples[index] for index in train_local]
    validation_samples = [valid_samples[index] for index in validation_local]
    if len(training_samples) < 4 or not validation_samples:
        raise ValueError("时间块划分后训练或验证样本不足")
    scan_points, scan_seed, scan_time_ms = scan_time_offset(
        training_samples, timeline, options, progress_callback
    )
    initial_handeye = scan_seed.tracker_from_camera
    initial_board = scan_seed.world_from_board
    if not np.all(np.isfinite(initial_handeye)):
        initial_handeye = np.asarray(handeye_seed, dtype=np.float64)
    if not np.all(np.isfinite(initial_board)):
        initial_board = np.asarray(board_seed, dtype=np.float64)
    initial = np.concatenate(
        (
            transform_to_vector(initial_handeye),
            transform_to_vector(initial_board),
            [scan_time_ms / 1000.0],
        )
    )
    lower_bounds = np.full(13, -np.inf, dtype=np.float64)
    upper_bounds = np.full(13, np.inf, dtype=np.float64)
    lower_bounds[12] = options.time_offset_min_ms / 1000.0
    upper_bounds[12] = options.time_offset_max_ms / 1000.0
    context = ResidualContext(
        tuple(training_samples), timeline, camera, options
    )
    max_nfev = 2000
    residual_calls = 0

    def tracked_residuals(
        parameters: np.ndarray, residual_context: ResidualContext
    ) -> np.ndarray:
        """计算残差并按数值差分规模发出近似评估进度。"""
        nonlocal residual_calls
        residuals = parameter_residuals(parameters, residual_context)
        residual_calls += 1
        if residual_calls == 1 or residual_calls % 14 == 0:
            _emit_progress(
                progress_callback,
                "joint_optimization_evaluation",
                residual_calls=residual_calls,
                approx_nfev=(residual_calls + 13) // 14,
                max_nfev=max_nfev,
                residual_rms_px=float(
                    np.sqrt(np.mean(np.square(residuals)))
                ),
            )
        return residuals

    _emit_progress(
        progress_callback,
        "joint_optimization_start",
        training_samples=len(training_samples),
        validation_samples=len(validation_samples),
        max_nfev=max_nfev,
    )
    optimized = least_squares(
        tracked_residuals,
        initial,
        args=(context,),
        method="trf",
        loss="soft_l1",
        f_scale=1.0,
        bounds=(lower_bounds, upper_bounds),
        x_scale="jac",
        max_nfev=max_nfev,
    )
    _emit_progress(
        progress_callback,
        "joint_optimization_complete",
        residual_calls=residual_calls,
        nfev=int(optimized.nfev),
        status=int(optimized.status),
        message=str(optimized.message),
        cost=float(optimized.cost),
    )
    if not optimized.success:
        raise RuntimeError(f"时空联合优化失败: {optimized.message}")
    tracker_from_camera = vector_to_transform(optimized.x[:6])
    world_from_board = vector_to_transform(optimized.x[6:12])
    time_offset_ms = float(optimized.x[12] * 1000.0)
    train_errors = _pixel_errors(
        training_samples,
        timeline,
        camera,
        options,
        tracker_from_camera,
        world_from_board,
        time_offset_ms,
    )
    validation_errors = _pixel_errors(
        validation_samples,
        timeline,
        camera,
        options,
        tracker_from_camera,
        world_from_board,
        time_offset_ms,
    )
    closure_world_tracker = []
    closure_camera_board = []
    offset_ns = int(round(time_offset_ms * 1.0e6))
    for sample in valid_samples:
        interpolation = interpolate_world_from_tracker(
            timeline.poses,
            sample.timestamp_ns + offset_ns,
            options.max_pose_gap_ms,
            timeline.pose_timestamps_ns,
        )
        if interpolation is None:
            raise RuntimeError("闭环评估时 Tracker 插值失败")
        closure_world_tracker.append(interpolation[0])
        closure_camera_board.append(sample.camera_from_board)
    closure_translation, closure_rotation, _ = board_closure_errors(
        board_transforms(
            closure_world_tracker,
            closure_camera_board,
            tracker_from_camera,
        )
    )
    boundary_tolerance_ms = max(0.05, options.coarse_step_ms * 0.05)
    at_boundary = (
        abs(time_offset_ms - options.time_offset_min_ms)
        <= boundary_tolerance_ms
        or abs(time_offset_ms - options.time_offset_max_ms)
        <= boundary_tolerance_ms
    )
    metrics = {
        "valid_frames": len(valid_samples),
        "training_median_px": float(np.median(train_errors)),
        "training_p95_px": float(np.percentile(train_errors, 95.0)),
        "validation_median_px": float(np.median(validation_errors)),
        "validation_p95_px": float(
            np.percentile(validation_errors, 95.0)
        ),
        "closure_translation_rmse_mm": float(
            np.sqrt(np.mean(np.square(closure_translation)))
        ),
        "closure_rotation_rmse_deg": float(
            np.sqrt(np.mean(np.square(closure_rotation)))
        ),
        "time_offset_at_boundary": bool(at_boundary),
        "optimizer_cost": float(optimized.cost),
        "optimizer_optimality": float(optimized.optimality),
    }
    return OptimizationResult(
        tracker_from_camera=tracker_from_camera,
        camera_from_tracker=np.linalg.inv(tracker_from_camera),
        world_from_board=world_from_board,
        time_offset_ms=time_offset_ms,
        train_indices=tuple(original_indices[index] for index in train_local),
        validation_indices=tuple(
            original_indices[index] for index in validation_local
        ),
        metrics=metrics,
        scan_points=tuple(scan_points),
    )
