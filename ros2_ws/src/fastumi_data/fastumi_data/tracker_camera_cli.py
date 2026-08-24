"""编排 Tracker–鱼眼相机离线 MCAP 目标板标定流水线和检测预检。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from fastumi_data.tracker_camera_bag import (
    ImageFrame,
    interpolate_world_from_tracker,
    iter_image_frames,
    read_tracker_timeline,
    tracker_status_valid_at,
    tracker_status_valid_for_interval,
)
from fastumi_data.tracker_camera_config import (
    CalibrationSettings,
    load_calibration_target,
    load_aprilgrid,
    load_kalibr_camera,
)
from fastumi_data.tracker_camera_detection import (
    CalibrationObservation,
    DetectionRejected,
    OpenCvAprilTagDetector,
    OpenCvCheckerboardDetector,
    build_aprilgrid_observation,
    build_checkerboard_observation,
    validate_checkerboard_observations,
    validate_tag_family,
)
from fastumi_data.tracker_camera_handeye import (
    HandEyeEstimationError,
    select_handeye_seed,
    solve_handeye_candidates,
)
from fastumi_data.tracker_camera_optimizer import (
    CalibrationSample,
    OptimizationOptions,
    optimize_spatiotemporal,
    predict_camera_from_board,
)
from fastumi_data.tracker_camera_pnp import (
    PoseEstimationError,
    estimate_camera_from_board,
    project_fisheye_points,
)
from fastumi_data.tracker_camera_progress import CalibrationProgressLogger
from fastumi_data.tracker_camera_report import (
    OverlaySample,
    QualityThresholds,
    ReportContext,
    evaluate_quality,
    matplotlib_available,
    write_calibration_report,
)


@dataclass(frozen=True)
class PipelineOutcome:
    """保存流水线质量判定和已生成输出路径。"""

    accepted: bool
    output_paths: dict[str, Path]
    fatal: bool = False


class _OptimizationProgressAdapter:
    """把优化器结构化事件转换为面向用户的阶段日志。"""

    def __init__(self, progress: CalibrationProgressLogger) -> None:
        """保存目标进度记录器并初始化阶段状态。"""
        self._progress = progress
        self._stage = ""

    def __call__(self, event: str, details: Mapping[str, object]) -> None:
        """处理一次粗扫描或联合优化进度事件。"""
        if event == "time_offset_scan":
            if self._stage != "time_offset_scan":
                self._progress.start_stage(
                    "time_offset_scan", "开始时间偏移粗扫描"
                )
                self._stage = "time_offset_scan"
            completed = int(details["completed"])
            total = int(details["total"])
            best_offset = details.get("best_offset_ms")
            best_text = (
                "未知" if best_offset is None else f"{float(best_offset):.3f}"
            )
            self._progress.update(
                "时间偏移粗扫描进行中",
                completed=completed,
                total=total,
                force=completed >= total,
                extra=(
                    f"offset_ms={float(details['offset_ms']):.3f} | "
                    f"valid={bool(details['valid'])} | "
                    f"best_offset_ms={best_text}"
                ),
            )
            if completed >= total:
                self._progress.finish_stage(
                    f"时间偏移粗扫描完成，best_offset_ms={best_text}"
                )
            return
        if event == "joint_optimization_start":
            self._progress.start_stage(
                "joint_optimization",
                "开始时空联合优化: "
                f"training={int(details['training_samples'])}, "
                f"validation={int(details['validation_samples'])}, "
                f"max_nfev={int(details['max_nfev'])}",
            )
            self._stage = "joint_optimization"
            return
        if event == "joint_optimization_evaluation":
            self._progress.update(
                "联合优化近似进度",
                completed=int(details["approx_nfev"]),
                total=int(details["max_nfev"]),
                extra=(
                    "保守上限 ETA 基于 max_nfev | "
                    f"residual_calls={int(details['residual_calls'])} | "
                    f"residual_rms_px={float(details['residual_rms_px']):.6f}"
                ),
            )
            return
        if event == "joint_optimization_complete":
            self._progress.update(
                "联合优化器已返回",
                force=True,
                extra=(
                    f"nfev={int(details['nfev'])} | "
                    f"residual_calls={int(details['residual_calls'])} | "
                    f"status={int(details['status'])} | "
                    f"cost={float(details['cost']):.6f}"
                ),
            )
            self._progress.finish_stage(
                f"联合优化完成: {details['message']}"
            )


def _write_failure_summary(
    output_dir: Path | str,
    stage: str,
    failure: str,
    counters: Mapping[str, int] | None = None,
    fatal: bool = False,
) -> PipelineOutcome:
    """保存流水线中止阶段、原因和计数，供补采与自动化诊断。"""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = {
        "accepted": False,
        "fatal": fatal,
        "stage": stage,
        "failure": failure,
        "counters": dict(counters or {}),
        "suggestion": "检查采集覆盖、Tracker 状态和 AprilGrid 可见性后补采",
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return PipelineOutcome(False, {"summary": summary_path}, fatal=fatal)


def _attach_progress_log(
    outcome: PipelineOutcome, progress: CalibrationProgressLogger
) -> PipelineOutcome:
    """把持久进度日志加入流水线输出路径。"""
    return PipelineOutcome(
        outcome.accepted,
        {**outcome.output_paths, "calibration_log": progress.log_path},
        fatal=outcome.fatal,
    )


def _logged_failure(
    progress: CalibrationProgressLogger,
    output_dir: Path | str,
    stage: str,
    failure: str,
    counters: Mapping[str, int] | None = None,
    fatal: bool = False,
) -> PipelineOutcome:
    """记录失败阶段并生成带日志路径的结构化失败结果。"""
    progress.start_stage(stage, f"开始检查阶段 {stage}")
    progress.fail(failure)
    return _attach_progress_log(
        _write_failure_summary(output_dir, stage, failure, counters, fatal),
        progress,
    )


SETTING_GROUPS = {
    "topics": {
        "image": "image_topic",
        "tracker_pose": "tracker_topic",
        "tracker_status": "status_topic",
    },
    "target": {"tag_family": "tag_family"},
    "filtering": {
        "frame_stride": "frame_stride",
        "min_tags": "min_tags",
        "max_pose_gap_ms": "max_pose_gap_ms",
        "sample_start_offset_s": "sample_start_offset_s",
        "sample_end_offset_s": "sample_end_offset_s",
    },
    "optimization": {
        "time_offset_min_ms": "time_offset_min_ms",
        "time_offset_max_ms": "time_offset_max_ms",
        "time_offset_step_ms": "time_offset_step_ms",
    },
    "quality": {
        "minimum_valid_frames": "minimum_valid_frames",
        "validation_median_max_px": "validation_median_max_px",
        "validation_p95_max_px": "validation_p95_max_px",
        "closure_translation_max_mm": "closure_translation_max_mm",
        "closure_rotation_max_deg": "closure_rotation_max_deg",
    },
}


def _finite_positive_float(value: str) -> float:
    """解析有限正浮点数，供进度心跳参数使用。"""
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是浮点数") from error
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("必须是有限正数")
    return parsed


def _finite_nonnegative_float(value: str) -> float:
    """解析有限非负浮点数，供样本时间窗口参数使用。"""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("必须是浮点数") from error
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("必须是有限非负数")
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    """创建标定命令行解析器并写入已确认的 bag 默认话题。"""
    defaults = CalibrationSettings()
    parser = argparse.ArgumentParser(
        description="从 Vive Tracker 和鱼眼目标板 MCAP 标定刚性外参"
    )
    parser.add_argument("--bag", required=True)
    parser.add_argument(
        "--camera-config",
        default=None,
        help="完整标定必填的 Kalibr 或 ToF 鱼眼相机标定 YAML",
    )
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--settings-config")
    parser.add_argument("--image-topic", default=defaults.image_topic)
    parser.add_argument("--tracker-topic", default=defaults.tracker_topic)
    parser.add_argument("--status-topic", default=defaults.status_topic)
    parser.add_argument(
        "--tag-family",
        default=defaults.tag_family,
        help="AprilGrid-only 标签族；checkerboard 目标会忽略此参数",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=defaults.frame_stride
    )
    parser.add_argument(
        "--min-tags",
        type=int,
        default=defaults.min_tags,
        help="AprilGrid-only 最少标签数；checkerboard 由完整角点检测门控",
    )
    parser.add_argument(
        "--max-pose-gap-ms", type=float, default=defaults.max_pose_gap_ms
    )
    parser.add_argument(
        "--sample-start-offset-s",
        type=_finite_nonnegative_float,
        help=(
            "跳过首个抽帧图像 header 时间之后指定秒数之前的样本；"
            "默认从记录开头处理"
        ),
    )
    parser.add_argument(
        "--sample-end-offset-s",
        type=_finite_nonnegative_float,
        help=(
            "仅使用首个抽帧图像 header 时间之后指定秒数内的样本；"
            "默认处理完整记录"
        ),
    )
    parser.add_argument(
        "--time-offset-min-ms", type=float,
        default=defaults.time_offset_min_ms,
    )
    parser.add_argument(
        "--time-offset-max-ms", type=float,
        default=defaults.time_offset_max_ms,
    )
    parser.add_argument(
        "--time-offset-step-ms", type=float,
        default=defaults.time_offset_step_ms,
    )
    parser.add_argument(
        "--minimum-valid-frames", type=int,
        default=defaults.minimum_valid_frames,
    )
    parser.add_argument(
        "--validation-median-max-px", type=float,
        default=defaults.validation_median_max_px,
    )
    parser.add_argument(
        "--validation-p95-max-px", type=float,
        default=defaults.validation_p95_max_px,
    )
    parser.add_argument(
        "--closure-translation-max-mm", type=float,
        default=defaults.closure_translation_max_mm,
    )
    parser.add_argument(
        "--closure-rotation-max-deg", type=float,
        default=defaults.closure_rotation_max_deg,
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=_finite_positive_float,
        default=30.0,
    )
    parser.add_argument("--detect-only", action="store_true")
    parser.add_argument("--allow-high-residual", action="store_true")
    return parser


def apply_settings_file(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """以 YAML 覆盖解析器默认值，同时保留显式 CLI 参数优先级。"""
    if not arguments.settings_config:
        return arguments
    path = Path(arguments.settings_config)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取设置文件 {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise ValueError("标定设置文件顶层必须是映射")
    unknown_groups = sorted(set(document) - set(SETTING_GROUPS))
    if unknown_groups:
        raise ValueError(f"未知设置分组: {', '.join(unknown_groups)}")
    for group_name, values in document.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"设置分组 {group_name} 必须是映射")
        mappings = SETTING_GROUPS[group_name]
        unknown_keys = sorted(set(values) - set(mappings))
        if unknown_keys:
            raise ValueError(
                f"未知设置 {group_name}: {', '.join(unknown_keys)}"
            )
        for key, value in values.items():
            attribute = mappings[key]
            if getattr(arguments, attribute) == parser.get_default(attribute):
                if (
                    attribute in {
                        "sample_start_offset_s", "sample_end_offset_s"
                    }
                    and value is not None
                ):
                    try:
                        value = _finite_nonnegative_float(value)
                    except argparse.ArgumentTypeError as error:
                        raise ValueError(
                            f"设置 filtering.{key} {error}"
                        ) from error
                setattr(arguments, attribute, value)
    return arguments


def validate_arguments(
    arguments: argparse.Namespace, parser: argparse.ArgumentParser
) -> argparse.Namespace:
    """检查依赖运行模式的参数约束并返回原参数对象。

    Args:
        arguments: 已应用设置文件覆盖的命令行参数。
        parser: 用于输出统一用法和错误信息的解析器。

    Returns:
        通过校验的原参数对象。

    Raises:
        SystemExit: 完整外参标定未提供相机内参配置时由解析器抛出。
    """
    if not arguments.detect_only and not arguments.camera_config:
        parser.error("完整标定必须显式指定 --camera-config")
    start_offset_s = arguments.sample_start_offset_s
    end_offset_s = arguments.sample_end_offset_s
    if (
        start_offset_s is not None
        and end_offset_s is not None
        and start_offset_s > end_offset_s
    ):
        parser.error("--sample-start-offset-s 不能大于 --sample-end-offset-s")
    return arguments


def _sha256_path(path: Path | str) -> str:
    """计算文件内容 SHA-256，记录显式相机配置来源。"""
    # 摘要对象用于以流式方式处理可能较大的配置文件。
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _thresholds(arguments: argparse.Namespace) -> QualityThresholds:
    """从参数构造报告质量门。"""
    return QualityThresholds(
        int(arguments.minimum_valid_frames),
        float(arguments.validation_median_max_px),
        float(arguments.validation_p95_max_px),
        float(arguments.closure_translation_max_mm),
        float(arguments.closure_rotation_max_deg),
    )


def _target_type(target: Any) -> str:
    """返回配置目标类型，旧 AprilGrid 对象默认按 AprilGrid 处理。"""
    return str(getattr(target, "target_type", "aprilgrid"))


def _is_checkerboard(target: Any) -> bool:
    """返回目标是否使用棋盘格检测和完整角点门控。"""
    return _target_type(target) == "checkerboard"


def _load_target_config(path: str, tag_family: str) -> Any:
    """按 target_type 加载目标，并兼容旧测试对 load_aprilgrid 的替换。"""
    if not Path(path).exists():
        return load_aprilgrid(path, tag_family)
    return load_calibration_target(path, tag_family)


def _make_target_detector(target: Any, tag_family: str) -> Any:
    """为目标规格创建对应 OpenCV 检测器。"""
    if _is_checkerboard(target):
        return OpenCvCheckerboardDetector(target)
    return OpenCvAprilTagDetector(tag_family)


def _build_target_observation(
    timestamp_ns: int,
    image: np.ndarray,
    detector: Any,
    target: Any,
    min_tags: int,
) -> CalibrationObservation:
    """用统一入口构造 AprilGrid 或棋盘格观测。"""
    if _is_checkerboard(target):
        return build_checkerboard_observation(
            timestamp_ns, detector.detect(image), target
        )
    return build_aprilgrid_observation(
        timestamp_ns, detector.detect(image), target, min_tags
    )


def _validate_target_observations(
    observations: Sequence[CalibrationObservation],
    target: Any,
    minimum_probe_frames: int = 5,
) -> None:
    """按目标类型执行预检帧数和完整特征门控。"""
    if _is_checkerboard(target):
        validate_checkerboard_observations(
            observations, target, minimum_probe_frames
        )
    else:
        validate_tag_family(observations, target, minimum_probe_frames)


def _target_spec_snapshot(target: Any) -> dict[str, object]:
    """把目标规格转换为报告和摘要中的稳定映射。"""
    if _is_checkerboard(target):
        return {
            "target_type": "checkerboard",
            "target_cols": int(target.target_cols),
            "target_rows": int(target.target_rows),
            "row_spacing_m": float(target.row_spacing_m),
            "col_spacing_m": float(target.col_spacing_m),
            "corner_count": int(target.corner_count),
            "board_extent_m": [
                float(value) for value in target.board_extent_m
            ],
        }
    return {
        "target_type": "aprilgrid",
        "tag_cols": int(target.tag_cols),
        "tag_rows": int(target.tag_rows),
        "tag_size_m": float(target.tag_size_m),
        "tag_spacing": float(target.tag_spacing),
        "board_extent_m": [float(value) for value in target.board_extent_m],
    }


def _reservoir_add(
    reservoir: list[Any], item: Any, seen: int,
    maximum: int, generator: np.random.Generator,
) -> None:
    """以固定种子 reservoir 保留跨完整时段的少量对象。"""
    if len(reservoir) < maximum:
        reservoir.append(item)
        return
    replacement = int(generator.integers(0, seen))
    if replacement < maximum:
        reservoir[replacement] = item


def _write_detection_overlay(
    path: Path, frame: ImageFrame, observation: Any, target: Any = None
) -> None:
    """把目标检测角点绘制到 detect-only 抽样图像。"""
    canvas = np.asarray(frame.image).copy()
    if _is_checkerboard(target):
        corners = np.asarray(
            observation.image_points_px, dtype=np.float32
        ).reshape(-1, 1, 2)
        cv2.drawChessboardCorners(
            canvas, (target.target_cols, target.target_rows), corners, True
        )
    else:
        for index, tag_id in enumerate(observation.tag_ids):
            corners = observation.image_points_px[
                index * 4:(index + 1) * 4
            ]
            polygon = np.rint(corners).astype(np.int32)
            cv2.polylines(canvas, [polygon], True, (0, 255, 0), 2)
            cv2.putText(
                canvas, str(tag_id), tuple(polygon[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                cv2.LINE_AA,
            )
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"无法写入检测叠加图 {path}")


def _run_detection_only(
    arguments: argparse.Namespace,
    detector: Any,
    target: Any,
    progress: CalibrationProgressLogger | None = None,
) -> PipelineOutcome:
    """全 bag 目标检测并用 reservoir 输出 20 帧跨时段预检。"""
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(20260804)
    reservoir = []
    observations: list[CalibrationObservation] = []
    decoded = 0
    rejected = 0
    checkerboard = _is_checkerboard(target)
    target_name = "棋盘格" if checkerboard else "AprilGrid"
    if progress is not None:
        progress.start_stage(
            "detection_only", f"开始全 bag {target_name} 检测预检"
        )
    for frame in _iter_sample_frames(arguments):
        decoded += 1
        if progress is not None:
            progress.update(
                f"{target_name} 检测预检进行中",
                extra=(
                    f"decoded={decoded} | valid={len(observations)} | "
                    f"rejected={rejected}"
                ),
            )
        try:
            observation = _build_target_observation(
                frame.timestamp_ns,
                frame.image,
                detector,
                target,
                int(getattr(arguments, "min_tags", 1)),
            )
        except DetectionRejected:
            rejected += 1
            continue
        observations.append(observation)
        _reservoir_add(
            reservoir, (frame, observation), len(observations), 20, generator
        )
    failures = []
    try:
        _validate_target_observations(observations, target, 5)
    except DetectionRejected as error:
        failures.append(str(error))
    if len(reservoir) < 20:
        failures.append(
            f"检测预检只得到 {len(reservoir)} 帧，至少需要 20 帧"
        )
    paths = {}
    for index, (frame, observation) in enumerate(
        sorted(reservoir, key=lambda item: item[0].timestamp_ns)
    ):
        path = output_dir / f"detection_overlay_{index:03d}.png"
        _write_detection_overlay(path, frame, observation, target)
        paths[f"overlay_{index:03d}"] = path
    summary = {
        "accepted": not failures,
        "failures": failures,
        "decoded_frames": decoded,
        "valid_frames": len(observations),
        "rejected_frames": rejected,
        "probe_frames": len(reservoir),
        "target_type": _target_type(target),
        "detector_settings": dict(detector.settings),
    }
    if checkerboard:
        corner_histogram = Counter(
            observation.feature_count for observation in observations
        )
        summary.update({
            "target_cols": int(target.target_cols),
            "target_rows": int(target.target_rows),
            "expected_corner_count": int(target.corner_count),
            "corner_count_histogram": {
                str(count): corner_histogram[count]
                for count in sorted(corner_histogram)
            },
        })
    else:
        tag_ids = [tag for item in observations for tag in item.tag_ids]
        tag_count_histogram = Counter(
            observation.tag_count for observation in observations
        )
        tag_id_frame_counts = Counter(tag_ids)
        summary.update({
            "tag_family": target.tag_family,
            "tag_id_min": min(tag_ids) if tag_ids else None,
            "tag_id_max": max(tag_ids) if tag_ids else None,
            "tag_count_histogram": {
                str(count): tag_count_histogram[count]
                for count in sorted(tag_count_histogram)
            },
            "tag_id_frame_counts": {
                str(tag_id): tag_id_frame_counts[tag_id]
                for tag_id in sorted(tag_id_frame_counts)
            },
        })
    summary_path = output_dir / "detection_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["detection_summary"] = summary_path
    if progress is not None:
        progress.update(
            "检测预检计数已汇总",
            force=True,
            extra=(
                f"decoded={decoded} | valid={len(observations)} | "
                f"rejected={rejected}"
            ),
        )
        progress.finish_stage(f"全 bag {target_name} 检测预检完成")
    print(json.dumps(summary, ensure_ascii=False))
    return PipelineOutcome(not failures, paths)


def _motion_diverse_indices(transforms: Sequence[np.ndarray]) -> list[int]:
    """保留相对上一选中帧平移 2 mm 或旋转 1° 以上的索引。"""
    if not transforms:
        return []
    selected = [0]
    reference = transforms[0]
    for index, transform in enumerate(transforms[1:], start=1):
        relative = np.linalg.inv(reference) @ transform
        translation_m = np.linalg.norm(relative[:3, 3])
        rotation_deg = np.rad2deg(
            np.linalg.norm(
                Rotation.from_matrix(relative[:3, :3]).as_rotvec()
            )
        )
        if translation_m >= 0.002 or rotation_deg >= 1.0:
            selected.append(index)
            reference = transform
    return selected


def _iter_sample_frames(arguments: argparse.Namespace):
    """按首个抽帧图像的 header 时间应用可选闭区间样本窗口。"""
    first_timestamp_ns = None
    start_offset_s = getattr(arguments, "sample_start_offset_s", None)
    end_offset_s = getattr(arguments, "sample_end_offset_s", None)
    for frame in iter_image_frames(
        arguments.bag, arguments.image_topic, arguments.frame_stride
    ):
        if first_timestamp_ns is None:
            first_timestamp_ns = frame.timestamp_ns
        elapsed_s = (frame.timestamp_ns - first_timestamp_ns) * 1.0e-9
        if start_offset_s is not None and elapsed_s < start_offset_s:
            continue
        if end_offset_s is not None and elapsed_s > end_offset_s:
            break
        yield frame


def _collect_samples(
    arguments: argparse.Namespace,
    detector: Any,
    target: Any,
    camera: Any,
    timeline: Any,
    progress: CalibrationProgressLogger | None = None,
) -> tuple:
    """第二遍流式执行状态过滤、目标检测、PnP 和零偏移插值。"""
    samples = []
    estimates = []
    tracker_poses = []
    overlay_frames = {}
    counters = {name: 0 for name in (
        "decoded", "status_rejected", "detection_rejected",
        "pnp_rejected", "interpolation_rejected",
    )}
    observations = []
    if progress is not None:
        progress.start_stage(
            "sample_collection", "开始解码图像、检测目标并估计 PnP"
        )
    for frame in _iter_sample_frames(arguments):
        counters["decoded"] += 1
        if progress is not None:
            progress.update(
                "样本采集进行中",
                extra=(
                    f"decoded={counters['decoded']} | valid={len(samples)} | "
                    f"rejected={sum(counters.values()) - counters['decoded']}"
                ),
            )
        status_start_ns = frame.timestamp_ns + int(
            round(arguments.time_offset_min_ms * 1.0e6)
        )
        status_end_ns = frame.timestamp_ns + int(
            round(arguments.time_offset_max_ms * 1.0e6)
        )
        if not tracker_status_valid_for_interval(
            timeline.statuses, status_start_ns, status_end_ns,
            arguments.max_pose_gap_ms,
            timeline.status_timestamps_ns,
        ):
            counters["status_rejected"] += 1
            continue
        try:
            observation = _build_target_observation(
                frame.timestamp_ns, frame.image, detector, target,
                int(getattr(arguments, "min_tags", 1)),
            )
        except DetectionRejected:
            counters["detection_rejected"] += 1
            continue
        try:
            estimate = estimate_camera_from_board(observation, camera)
        except PoseEstimationError:
            counters["pnp_rejected"] += 1
            continue
        interpolation = interpolate_world_from_tracker(
            timeline.poses,
            frame.timestamp_ns,
            arguments.max_pose_gap_ms,
            timeline.pose_timestamps_ns,
        )
        if interpolation is None:
            counters["interpolation_rejected"] += 1
            continue
        index = len(samples)
        samples.append(CalibrationSample(
            frame.timestamp_ns, observation.object_points_m,
            observation.image_points_px, estimate.camera_from_board,
            getattr(observation, "tag_count", None),
            target_type=_target_type(target),
            feature_count=getattr(
                observation, "feature_count", len(observation.object_points_m)
            ),
        ))
        estimates.append(estimate)
        tracker_poses.append(interpolation[0])
        observations.append(observation)
        if len(observations) == 5:
            _validate_target_observations(observations, target, 5)
        if len(overlay_frames) < 5:
            overlay_frames[index] = frame
    if progress is not None:
        progress.update(
            "样本采集计数已汇总",
            force=True,
            extra=f"decoded={counters['decoded']} | valid={len(samples)}",
        )
        progress.finish_stage(
            f"样本采集完成: decoded={counters['decoded']}, valid={len(samples)}"
        )
    return samples, estimates, tracker_poses, overlay_frames, counters


def _diagnostics(result: Any, samples: Sequence[Any], estimates: Sequence[Any],
                 overlay_frames: Mapping[int, ImageFrame], timeline: Any,
                 camera: Any, max_gap_ms: float) -> tuple:
    """按最终模型生成逐帧 CSV 指标和抽样角点叠加数据。"""
    train = set(result.train_indices)
    validation = set(result.validation_indices)
    rows = []
    overlays = []
    offset_ns = int(round(result.time_offset_ms * 1.0e6))
    for index, (sample, estimate) in enumerate(zip(samples, estimates)):
        target_ns = sample.timestamp_ns + offset_ns
        if timeline.statuses and not tracker_status_valid_at(
            timeline.statuses,
            target_ns,
            max_gap_ms,
            timeline.status_timestamps_ns,
        ):
            continue
        interpolation = interpolate_world_from_tracker(
            timeline.poses,
            target_ns,
            max_gap_ms,
            timeline.pose_timestamps_ns,
        )
        if interpolation is None:
            continue
        predicted_pose = predict_camera_from_board(
            interpolation[0], result.tracker_from_camera,
            result.world_from_board,
        )
        predicted = project_fisheye_points(
            sample.object_points_m, predicted_pose, camera
        )
        errors = np.linalg.norm(predicted - sample.image_points_px, axis=1)
        closure = (interpolation[0] @ result.tracker_from_camera
                   @ sample.camera_from_board)
        closure_error = np.linalg.inv(result.world_from_board) @ closure
        rows.append({
            "timestamp_ns": sample.timestamp_ns,
            "partition": "train" if index in train else (
                "validation" if index in validation else "filtered"),
            "target_type": getattr(sample, "target_type", "aprilgrid"),
            "feature_count": getattr(
                sample, "feature_count", len(sample.object_points_m)
            ),
            "tag_count": getattr(sample, "tag_count", None),
            "pnp_median_px": estimate.median_error_px,
            "pnp_p95_px": estimate.p95_error_px,
            "final_median_px": float(np.median(errors)),
            "final_p95_px": float(np.percentile(errors, 95.0)),
            "closure_translation_mm": float(
                np.linalg.norm(closure_error[:3, 3]) * 1000.0),
            "closure_rotation_deg": float(np.rad2deg(np.linalg.norm(
                Rotation.from_matrix(closure_error[:3, :3]).as_rotvec()))),
            "pose_gap_ms": interpolation[1],
        })
        if index in overlay_frames:
            overlays.append(OverlaySample(
                sample.timestamp_ns, overlay_frames[index].image,
                sample.image_points_px, predicted,
            ))
    return tuple(rows), tuple(overlays)


def run_calibration(arguments: argparse.Namespace) -> PipelineOutcome:
    """执行检测预检或完整 Tracker–鱼眼相机标定流水线。"""
    with CalibrationProgressLogger(
        arguments.output_dir, arguments.progress_interval_seconds
    ) as progress:
        try:
            progress.start_stage("load_inputs", "开始加载配置和标定输入")
            if not matplotlib_available():
                warnings.warn(
                    "缺少 Matplotlib：标定仍会生成 YAML/JSON/CSV，安装后可生成 PNG 诊断图",
                    RuntimeWarning,
                    stacklevel=2,
                )
            target = _load_target_config(
                arguments.target_config, arguments.tag_family
            )
            detector = _make_target_detector(target, arguments.tag_family)
            if arguments.detect_only:
                progress.finish_stage("配置和检测器加载完成")
                return _attach_progress_log(
                    _run_detection_only(arguments, detector, target, progress),
                    progress,
                )
            if not arguments.camera_config:
                raise ValueError("完整标定必须显式指定 --camera-config")
            camera = load_kalibr_camera(arguments.camera_config)
            camera_provenance = {
                "source": "provided",
                "path": str(arguments.camera_config),
                "sha256": _sha256_path(arguments.camera_config),
            }
            timeline = read_tracker_timeline(
                arguments.bag, arguments.tracker_topic,
                arguments.status_topic,
            )
            progress.finish_stage(
                "配置和 Tracker 时间线加载完成: "
                f"poses={len(timeline.poses)}, statuses={len(timeline.statuses)}"
            )
            samples, estimates, tracker_poses, overlay_frames, counters = (
                _collect_samples(
                    arguments, detector, target, camera, timeline, progress
                )
            )
        except DetectionRejected as error:
            return _logged_failure(
                progress,
                arguments.output_dir,
                (
                    "target_validation"
                    if _is_checkerboard(target)
                    else "tag_family_validation"
                ),
                str(error),
            )
        except KeyboardInterrupt:
            progress.fail("用户中断标定")
            raise
        except Exception as error:
            progress.fail(f"未处理异常: {error}")
            raise
        if len(samples) < arguments.minimum_valid_frames:
            return _logged_failure(
                progress,
                arguments.output_dir,
                "sample_collection",
                f"有效标定帧 {len(samples)} 少于 {arguments.minimum_valid_frames}",
                counters,
            )
        progress.start_stage("motion_and_seed", "开始运动去冗余和 Hand-Eye 初值估计")
        diverse = _motion_diverse_indices(tracker_poses)
        if len(diverse) < 8:
            return _logged_failure(
                progress,
                arguments.output_dir,
                "motion_diversity",
                f"运动去冗余后仅 {len(diverse)} 帧，至少需要 8 帧",
                counters,
            )
        try:
            seed = select_handeye_seed(solve_handeye_candidates(
                [tracker_poses[index] for index in diverse],
                [samples[index].camera_from_board for index in diverse],
            ))
        except (HandEyeEstimationError, ValueError) as error:
            return _logged_failure(
                progress, arguments.output_dir, "handeye", str(error), counters
            )
        progress.finish_stage(
            "Hand-Eye 初值估计完成: "
            f"method={seed.method}, diverse_frames={len(diverse)}, "
            f"translation_rmse_mm={seed.translation_rmse_mm:.6f}, "
            f"rotation_rmse_deg={seed.rotation_rmse_deg:.6f}"
        )
        try:
            result = optimize_spatiotemporal(
                samples, timeline, camera, seed.tracker_from_camera,
                seed.world_from_board,
                OptimizationOptions(
                    arguments.time_offset_min_ms,
                    arguments.time_offset_max_ms,
                    arguments.time_offset_step_ms,
                    arguments.max_pose_gap_ms,
                    0.2,
                ),
                progress_callback=_OptimizationProgressAdapter(progress),
            )
        except KeyboardInterrupt:
            progress.fail("用户中断标定")
            raise
        except (HandEyeEstimationError, RuntimeError, ValueError) as error:
            return _logged_failure(
                progress, arguments.output_dir, "optimization", str(error),
                counters,
            )
        progress.start_stage("diagnostics", "开始计算诊断指标和生成标定报告")
        frame_metrics, overlays = _diagnostics(
            result, samples, estimates, overlay_frames, timeline, camera,
            arguments.max_pose_gap_ms,
        )
        thresholds = _thresholds(arguments)
        target_type = _target_type(target)
        result.metrics["target_type"] = target_type
        if samples:
            result.metrics["feature_count"] = int(samples[0].feature_count)
        camera_config_path = Path(arguments.camera_config)
        context = ReportContext(
            Path(arguments.bag), camera_config_path,
            Path(arguments.target_config), arguments.image_topic,
            arguments.tracker_topic, arguments.status_topic,
            arguments.tag_family if target_type == "aprilgrid" else None,
            {
                "frame_stride": arguments.frame_stride,
                "min_tags": arguments.min_tags,
                "max_pose_gap_ms": arguments.max_pose_gap_ms,
                "sample_start_offset_s": arguments.sample_start_offset_s,
                "sample_end_offset_s": arguments.sample_end_offset_s,
                "time_offset_min_ms": arguments.time_offset_min_ms,
                "time_offset_max_ms": arguments.time_offset_max_ms,
                "time_offset_step_ms": arguments.time_offset_step_ms,
                "progress_interval_seconds": (
                    arguments.progress_interval_seconds
                ),
            },
            camera_provenance,
            thresholds, frame_metrics, overlays,
            (f"阶段计数: {json.dumps(counters, ensure_ascii=False)}",),
            target_type=target_type,
            target_spec=_target_spec_snapshot(target),
        )
        paths = write_calibration_report(arguments.output_dir, result, context)
        decision = evaluate_quality(result.metrics, thresholds)
        progress.finish_stage(
            "诊断和报告生成完成，质量门结果="
            f"{'通过' if decision.accepted else '未通过'}"
        )
        paths["calibration_log"] = progress.log_path
        print(json.dumps({
            "accepted": decision.accepted,
            "failures": list(decision.failures),
            "counters": counters,
            "output_paths": {name: str(path) for name, path in paths.items()},
        }, ensure_ascii=False))
        return PipelineOutcome(decision.accepted, paths)


def main(argv: list[str] | None = None) -> None:
    """解析参数、执行标定，并根据质量门设置进程退出码。"""
    parser = build_argument_parser()
    arguments = validate_arguments(
        apply_settings_file(parser.parse_args(argv), parser), parser
    )
    outcome = run_calibration(arguments)
    if outcome.fatal or (
        not outcome.accepted and not arguments.allow_high_residual
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
