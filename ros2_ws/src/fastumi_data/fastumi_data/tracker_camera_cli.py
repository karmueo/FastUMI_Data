"""编排 Tracker–鱼眼相机离线 MCAP 标定流水线和检测预检。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
)
from fastumi_data.tracker_camera_config import (
    CalibrationSettings,
    load_aprilgrid,
    load_kalibr_camera,
)
from fastumi_data.tracker_camera_detection import (
    DetectionRejected,
    OpenCvAprilTagDetector,
    build_aprilgrid_observation,
    validate_tag_family,
)
from fastumi_data.tracker_camera_handeye import (
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
from fastumi_data.tracker_camera_report import (
    OverlaySample,
    QualityThresholds,
    ReportContext,
    evaluate_quality,
    write_calibration_report,
)


@dataclass(frozen=True)
class PipelineOutcome:
    """保存流水线质量判定和已生成输出路径。"""

    accepted: bool
    output_paths: dict[str, Path]


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


def build_argument_parser() -> argparse.ArgumentParser:
    """创建标定命令行解析器并写入已确认的 bag 默认话题。"""
    defaults = CalibrationSettings()
    parser = argparse.ArgumentParser(
        description="从 Vive Tracker 和鱼眼 AprilGrid MCAP 标定刚性外参"
    )
    parser.add_argument("--bag", required=True)
    parser.add_argument("--camera-config", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--settings-config")
    parser.add_argument("--image-topic", default=defaults.image_topic)
    parser.add_argument("--tracker-topic", default=defaults.tracker_topic)
    parser.add_argument("--status-topic", default=defaults.status_topic)
    parser.add_argument("--tag-family", default=defaults.tag_family)
    parser.add_argument(
        "--frame-stride", type=int, default=defaults.frame_stride
    )
    parser.add_argument("--min-tags", type=int, default=defaults.min_tags)
    parser.add_argument(
        "--max-pose-gap-ms", type=float, default=defaults.max_pose_gap_ms
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
                setattr(arguments, attribute, value)
    return arguments


def _thresholds(arguments: argparse.Namespace) -> QualityThresholds:
    """从参数构造报告质量门。"""
    return QualityThresholds(
        int(arguments.minimum_valid_frames),
        float(arguments.validation_median_max_px),
        float(arguments.validation_p95_max_px),
        float(arguments.closure_translation_max_mm),
        float(arguments.closure_rotation_max_deg),
    )


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


def _write_detection_overlay(path: Path, frame: ImageFrame, observation: Any) -> None:
    """把检测角点和 ID 绘制到 detect-only 抽样图像。"""
    canvas = np.asarray(frame.image).copy()
    for index, tag_id in enumerate(observation.tag_ids):
        corners = observation.image_points_px[index * 4:(index + 1) * 4]
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
    detector: OpenCvAprilTagDetector,
    target: Any,
) -> PipelineOutcome:
    """全 bag 检测并用 reservoir 输出 20 帧跨时段预检。"""
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(20260804)
    reservoir = []
    observations = []
    decoded = 0
    rejected = 0
    for frame in iter_image_frames(
        arguments.bag, arguments.image_topic, arguments.frame_stride
    ):
        decoded += 1
        try:
            observation = build_aprilgrid_observation(
                frame.timestamp_ns, detector.detect(frame.image), target,
                arguments.min_tags,
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
        validate_tag_family(observations, target, minimum_probe_frames=5)
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
        _write_detection_overlay(path, frame, observation)
        paths[f"overlay_{index:03d}"] = path
    tag_ids = [tag for item in observations for tag in item.tag_ids]
    summary = {
        "accepted": not failures,
        "failures": failures,
        "decoded_frames": decoded,
        "valid_frames": len(observations),
        "rejected_frames": rejected,
        "probe_frames": len(reservoir),
        "tag_family": target.tag_family,
        "tag_id_min": min(tag_ids) if tag_ids else None,
        "tag_id_max": max(tag_ids) if tag_ids else None,
    }
    summary_path = output_dir / "detection_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["detection_summary"] = summary_path
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


def _collect_samples(arguments: argparse.Namespace, detector: Any, target: Any,
                     camera: Any, timeline: Any) -> tuple:
    """第二遍流式执行状态过滤、AprilGrid 检测、PnP 和零偏移插值。"""
    samples = []
    estimates = []
    tracker_poses = []
    overlay_frames = {}
    counters = {name: 0 for name in (
        "decoded", "status_rejected", "detection_rejected",
        "pnp_rejected", "interpolation_rejected",
    )}
    observations = []
    for frame in iter_image_frames(
        arguments.bag, arguments.image_topic, arguments.frame_stride
    ):
        counters["decoded"] += 1
        if not tracker_status_valid_at(
            timeline.statuses, frame.timestamp_ns, arguments.max_pose_gap_ms
        ):
            counters["status_rejected"] += 1
            continue
        try:
            observation = build_aprilgrid_observation(
                frame.timestamp_ns, detector.detect(frame.image), target,
                arguments.min_tags,
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
            timeline.poses, frame.timestamp_ns, arguments.max_pose_gap_ms
        )
        if interpolation is None:
            counters["interpolation_rejected"] += 1
            continue
        index = len(samples)
        samples.append(CalibrationSample(
            frame.timestamp_ns, observation.object_points_m,
            observation.image_points_px, estimate.camera_from_board,
            observation.tag_count,
        ))
        estimates.append(estimate)
        tracker_poses.append(interpolation[0])
        observations.append(observation)
        if len(overlay_frames) < 5:
            overlay_frames[index] = frame
    validate_tag_family(observations[:5], target, minimum_probe_frames=5)
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
        interpolation = interpolate_world_from_tracker(
            timeline.poses, sample.timestamp_ns + offset_ns, max_gap_ms
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
            "tag_count": sample.tag_count,
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
    camera = load_kalibr_camera(arguments.camera_config)
    target = load_aprilgrid(arguments.target_config, arguments.tag_family)
    detector = OpenCvAprilTagDetector(arguments.tag_family)
    if arguments.detect_only:
        return _run_detection_only(arguments, detector, target)
    timeline = read_tracker_timeline(
        arguments.bag, arguments.tracker_topic, arguments.status_topic
    )
    samples, estimates, tracker_poses, overlay_frames, counters = (
        _collect_samples(arguments, detector, target, camera, timeline)
    )
    if len(samples) < arguments.minimum_valid_frames:
        raise ValueError(
            f"有效标定帧 {len(samples)} 少于 {arguments.minimum_valid_frames}"
        )
    diverse = _motion_diverse_indices(tracker_poses)
    if len(diverse) < 8:
        raise ValueError(f"运动去冗余后仅 {len(diverse)} 帧，至少需要 8 帧")
    seed = select_handeye_seed(solve_handeye_candidates(
        [tracker_poses[index] for index in diverse],
        [samples[index].camera_from_board for index in diverse],
    ))
    result = optimize_spatiotemporal(
        samples, timeline, camera, seed.tracker_from_camera,
        seed.world_from_board,
        OptimizationOptions(
            arguments.time_offset_min_ms, arguments.time_offset_max_ms,
            arguments.time_offset_step_ms, arguments.max_pose_gap_ms, 0.2,
        ),
    )
    frame_metrics, overlays = _diagnostics(
        result, samples, estimates, overlay_frames, timeline, camera,
        arguments.max_pose_gap_ms,
    )
    thresholds = _thresholds(arguments)
    context = ReportContext(
        Path(arguments.bag), Path(arguments.camera_config),
        Path(arguments.target_config), arguments.image_topic,
        arguments.tracker_topic, arguments.status_topic,
        arguments.tag_family,
        {
            "frame_stride": arguments.frame_stride,
            "min_tags": arguments.min_tags,
            "max_pose_gap_ms": arguments.max_pose_gap_ms,
            "time_offset_min_ms": arguments.time_offset_min_ms,
            "time_offset_max_ms": arguments.time_offset_max_ms,
            "time_offset_step_ms": arguments.time_offset_step_ms,
        },
        thresholds, frame_metrics, overlays,
        (f"阶段计数: {json.dumps(counters, ensure_ascii=False)}",),
    )
    paths = write_calibration_report(arguments.output_dir, result, context)
    decision = evaluate_quality(result.metrics, thresholds)
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
    arguments = apply_settings_file(parser.parse_args(argv), parser)
    outcome = run_calibration(arguments)
    if not outcome.accepted and not arguments.allow_high_residual:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
