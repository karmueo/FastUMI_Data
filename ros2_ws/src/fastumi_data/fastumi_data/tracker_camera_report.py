"""评估 Tracker–相机标定质量并生成可复现报告与独立复核工具。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import cv2
try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
except ImportError:
    plt = None
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import yaml

from fastumi_data.pose_math import matrix_to_pose
from fastumi_data.tracker_camera_optimizer import OptimizationResult


@dataclass(frozen=True)
class QualityThresholds:
    """保存默认有效帧、留出集重投影和固定板闭环门限。"""

    minimum_valid_frames: int = 30
    validation_median_max_px: float = 1.0
    validation_p95_max_px: float = 2.0
    closure_translation_max_mm: float = 5.0
    closure_rotation_max_deg: float = 1.0
    reject_time_offset_boundary: bool = True


@dataclass(frozen=True)
class QualityDecision:
    """保存质量门最终判定及全部失败原因。"""

    accepted: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class OverlaySample:
    """保存一帧图像、检测角点和最终模型预测角点。"""

    timestamp_ns: int
    image_bgr: np.ndarray
    observed_points_px: np.ndarray
    predicted_points_px: np.ndarray


@dataclass(frozen=True)
class ReportContext:
    """保存报告所需输入路径、话题、配置快照和逐帧诊断。"""

    bag_path: Path
    camera_config_path: Path
    target_config_path: Path
    image_topic: str
    tracker_topic: str
    status_topic: str
    tag_family: str
    settings_snapshot: Mapping[str, Any]
    camera_provenance: Mapping[str, Any] | None = None
    thresholds: QualityThresholds = QualityThresholds()
    frame_metrics: tuple[Mapping[str, Any], ...] = ()
    overlay_samples: tuple[OverlaySample, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationResult:
    """保存 calibration.yaml 独立复核结果和解析出的质量指标。"""

    valid: bool
    failures: tuple[str, ...]
    metrics: Mapping[str, Any]


def matplotlib_available() -> bool:
    """返回当前环境是否可以生成 Matplotlib 诊断图。"""
    return plt is not None


def evaluate_quality(
    metrics: Mapping[str, Any], thresholds: QualityThresholds
) -> QualityDecision:
    """逐项应用默认质量门并完整返回所有失败原因。"""
    failures = []

    def checked_number(name: str) -> float | None:
        """读取一个有限指标，缺失或非法时登记失败。"""
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError):
            failures.append(f"缺少或无法解析质量指标 {name}")
            return None
        if not np.isfinite(value):
            failures.append(f"质量指标 {name} 不是有限数")
            return None
        return value

    valid_frames = checked_number("valid_frames")
    validation_median = checked_number("validation_median_px")
    validation_p95 = checked_number("validation_p95_px")
    closure_translation = checked_number("closure_translation_rmse_mm")
    closure_rotation = checked_number("closure_rotation_rmse_deg")
    if (
        valid_frames is not None
        and valid_frames < thresholds.minimum_valid_frames
    ):
        failures.append(
            f"有效帧 {int(valid_frames)} 少于 {thresholds.minimum_valid_frames}"
        )
    if (
        validation_median is not None
        and validation_median > thresholds.validation_median_max_px
    ):
        failures.append(
            "验证重投影 median "
            f"{validation_median:.3f} px 超过 "
            f"{thresholds.validation_median_max_px:.3f} px"
        )
    if (
        validation_p95 is not None
        and validation_p95 > thresholds.validation_p95_max_px
    ):
        failures.append(
            f"验证重投影 P95 {validation_p95:.3f} px 超过 "
            f"{thresholds.validation_p95_max_px:.3f} px"
        )
    if (
        closure_translation is not None
        and closure_translation > thresholds.closure_translation_max_mm
    ):
        failures.append(
            f"闭环平移 {closure_translation:.3f} mm 超过 "
            f"{thresholds.closure_translation_max_mm:.3f} mm"
        )
    if (
        closure_rotation is not None
        and closure_rotation > thresholds.closure_rotation_max_deg
    ):
        failures.append(
            f"闭环旋转 {closure_rotation:.3f}° 超过 "
            f"{thresholds.closure_rotation_max_deg:.3f}°"
        )
    if thresholds.reject_time_offset_boundary and bool(
        metrics.get("time_offset_at_boundary", False)
    ):
        failures.append("时间偏移优化结果位于搜索边界")
    return QualityDecision(not failures, tuple(failures))


def transform_document(
    transform: np.ndarray, maps_from: str, maps_to: str
) -> dict[str, Any]:
    """序列化带显式映射方向的刚体变换。"""
    matrix = np.asarray(transform, dtype=np.float64)
    position, quaternion = matrix_to_pose(matrix)
    return {
        "maps_from": maps_from,
        "maps_to": maps_to,
        "matrix": matrix.tolist(),
        "translation_m": position.tolist(),
        "quaternion_xyzw": quaternion.tolist(),
    }


def _sha256_path(path: Path) -> str:
    """流式计算文件或目录内容及相对路径的 SHA-256。"""
    target = Path(path)
    if not target.exists():
        raise ValueError(f"报告输入不存在: {target}")
    digest = hashlib.sha256()
    files = [target] if target.is_file() else sorted(
        item for item in target.rglob("*") if item.is_file()
    )
    for file_path in files:
        relative = file_path.name if target.is_file() else str(
            file_path.relative_to(target)
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    """先写同目录临时文件，再以 replace 原子替换目标文本。"""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_save_figure(path: Path, figure: plt.Figure) -> None:
    """把 Matplotlib 图保存到临时 PNG 后原子替换。"""
    temporary = path.with_name(f".{path.stem}.tmp.png")
    figure.savefig(temporary, dpi=150, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


FRAME_METRIC_FIELDS = (
    "timestamp_ns",
    "partition",
    "tag_count",
    "pnp_median_px",
    "pnp_p95_px",
    "final_median_px",
    "final_p95_px",
    "closure_translation_mm",
    "closure_rotation_deg",
    "pose_gap_ms",
)


def _csv_value(field: str, value: Any) -> str | int:
    """按固定精度把逐帧 CSV 值转换为稳定文本。"""
    if field in {"timestamp_ns", "tag_count"}:
        return int(value)
    if field == "partition":
        return str(value)
    return f"{float(value):.9f}"


def _write_frame_metrics(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    """原子写入固定列名和小数精度的逐帧 CSV。"""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FRAME_METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _csv_value(field, row.get(field, 0))
                    for field in FRAME_METRIC_FIELDS
                }
            )
    temporary.replace(path)


def _write_time_offset_plot(
    path: Path, result: OptimizationResult
) -> None:
    """绘制粗扫描偏移与平移/旋转闭环曲线。"""
    valid_points = [point for point in result.scan_points if point.valid]
    figure, first_axis = plt.subplots(figsize=(7.0, 4.0))
    if valid_points:
        offsets = [point.time_offset_ms for point in valid_points]
        translations = [
            point.translation_rmse_mm for point in valid_points
        ]
        rotations = [point.rotation_rmse_deg for point in valid_points]
        first_axis.plot(offsets, translations, "o-", label="translation RMSE")
        first_axis.set_ylabel("translation RMSE (mm)")
        second_axis = first_axis.twinx()
        second_axis.plot(offsets, rotations, "s-", color="tab:orange")
        second_axis.set_ylabel("rotation RMSE (deg)")
    first_axis.axvline(
        result.time_offset_ms,
        color="tab:red",
        linestyle="--",
        label="optimized offset",
    )
    first_axis.set_xlabel("time offset (ms)")
    first_axis.grid(True, alpha=0.3)
    first_axis.legend(loc="best")
    _atomic_save_figure(path, figure)


def _write_residual_plot(
    path: Path,
    result: OptimizationResult,
    frame_metrics: Sequence[Mapping[str, Any]],
) -> None:
    """绘制逐帧最终残差或训练/验证汇总残差。"""
    figure, axis = plt.subplots(figsize=(7.0, 4.0))
    values = [
        float(row["final_median_px"])
        for row in frame_metrics
        if "final_median_px" in row
    ]
    if values:
        axis.hist(values, bins=min(20, max(5, len(values))))
        axis.set_xlabel("per-frame median reprojection error (px)")
        axis.set_ylabel("frames")
    else:
        labels = ["train median", "validation median"]
        heights = [
            float(result.metrics.get("training_median_px", 0.0)),
            float(result.metrics.get("validation_median_px", 0.0)),
        ]
        axis.bar(labels, heights)
        axis.set_ylabel("reprojection error (px)")
    axis.grid(True, axis="y", alpha=0.3)
    _atomic_save_figure(path, figure)


def _write_overlay(path: Path, sample: OverlaySample) -> None:
    """绘制检测角点与最终模型预测角点叠加图。"""
    image = np.asarray(sample.image_bgr)
    if image.ndim == 3 and image.shape[2] == 3:
        display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display = image
    figure, axis = plt.subplots(figsize=(7.0, 7.0))
    axis.imshow(display, cmap="gray" if display.ndim == 2 else None)
    observed = np.asarray(sample.observed_points_px, dtype=np.float64)
    predicted = np.asarray(sample.predicted_points_px, dtype=np.float64)
    axis.scatter(
        observed[:, 0], observed[:, 1], s=16, c="lime", label="detected"
    )
    axis.scatter(
        predicted[:, 0], predicted[:, 1], s=16, c="red", marker="+",
        label="predicted"
    )
    axis.set_title(f"timestamp_ns={sample.timestamp_ns}")
    axis.legend(loc="best")
    axis.set_axis_off()
    _atomic_save_figure(path, figure)


def write_calibration_report(
    output_dir: Path | str,
    result: OptimizationResult,
    context: ReportContext,
) -> dict[str, Path]:
    """生成版本化 YAML、摘要、逐帧 CSV、诊断曲线和叠加图。"""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    decision = evaluate_quality(result.metrics, context.thresholds)
    warnings = list(context.warnings)
    if plt is None:
        warnings.append("缺少 Matplotlib，已跳过 PNG 诊断图；可安装后重新生成")
    if not context.overlay_samples:
        warnings.append("没有可视化样本，未生成角点叠加图")
    calibration_path = destination / "calibration.yaml"
    summary_path = destination / "summary.json"
    frame_metrics_path = destination / "frame_metrics.csv"
    time_offset_path = destination / "time_offset_scan.png"
    residual_path = destination / "residuals.png"
    document = {
        "schema_version": 1,
        "accepted": decision.accepted,
        "quality_failures": list(decision.failures),
        "coordinate_convention": "^A T_B maps coordinates from B to A",
        "quaternion_order": "xyzw",
        "translation_unit": "m",
        "rotation_unit": "rad",
        "tracker_from_camera": transform_document(
            result.tracker_from_camera, "camera", "tracker"
        ),
        "camera_from_tracker": transform_document(
            result.camera_from_tracker, "tracker", "camera"
        ),
        "world_from_board": transform_document(
            result.world_from_board, "board", "world"
        ),
        "time_offset_ms": float(result.time_offset_ms),
        "train_indices": list(result.train_indices),
        "validation_indices": list(result.validation_indices),
        "metrics": dict(result.metrics),
        "thresholds": asdict(context.thresholds),
        "inputs": {
            "bag": {
                "path": str(context.bag_path),
                "sha256": _sha256_path(context.bag_path),
            },
            "camera_config": {
                "path": str(context.camera_config_path),
                "sha256": _sha256_path(context.camera_config_path),
            },
            "camera_provenance": dict(context.camera_provenance or {}),
            "target_config": {
                "path": str(context.target_config_path),
                "sha256": _sha256_path(context.target_config_path),
            },
        },
        "topics": {
            "image": context.image_topic,
            "tracker_pose": context.tracker_topic,
            "tracker_status": context.status_topic,
        },
        "tag_family": context.tag_family,
        "settings_snapshot": dict(context.settings_snapshot),
        "software_versions": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "warnings": warnings,
    }
    _atomic_write_text(
        calibration_path,
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
    )
    summary = {
        "accepted": decision.accepted,
        "quality_failures": list(decision.failures),
        "metrics": dict(result.metrics),
        "time_offset_ms": float(result.time_offset_ms),
        "warnings": warnings,
    }
    _atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    _write_frame_metrics(frame_metrics_path, context.frame_metrics)
    paths = {
        "calibration": calibration_path,
        "summary": summary_path,
        "frame_metrics": frame_metrics_path,
    }
    if plt is not None:
        _write_time_offset_plot(time_offset_path, result)
        _write_residual_plot(residual_path, result, context.frame_metrics)
        paths["time_offset_plot"] = time_offset_path
        paths["residual_plot"] = residual_path
        for index, sample in enumerate(context.overlay_samples):
            overlay_path = destination / f"overlay_{index:03d}.png"
            _write_overlay(overlay_path, sample)
            paths[f"overlay_{index:03d}"] = overlay_path
    return paths


def _check_transform_document(
    name: str,
    document: Mapping[str, Any],
    failures: list[str],
    expected_maps_from: str,
    expected_maps_to: str,
) -> np.ndarray | None:
    """复核一个序列化变换的矩阵、旋转和四元数。"""
    try:
        matrix = np.asarray(document["matrix"], dtype=np.float64)
        translation = np.asarray(document["translation_m"], dtype=np.float64)
        quaternion = np.asarray(
            document["quaternion_xyzw"], dtype=np.float64
        )
        maps_from = str(document["maps_from"])
        maps_to = str(document["maps_to"])
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"{name} 无法解析: {error}")
        return None
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        failures.append(f"{name} matrix 必须是有限 4x4 数组")
        return None
    if maps_from != expected_maps_from or maps_to != expected_maps_to:
        failures.append(
            f"{name} 映射方向应为 {expected_maps_from} 到 {expected_maps_to}"
        )
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-12):
        failures.append(f"{name} 齐次末行必须为 [0, 0, 0, 1]")
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        failures.append(f"{name} 平移字段必须是三个有限数")
    elif not np.allclose(translation, matrix[:3, 3], atol=1.0e-12):
        failures.append(f"{name} 平移字段与矩阵不一致")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-9):
        failures.append(f"{name} 旋转矩阵不正交")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1.0e-9):
        failures.append(f"{name} 旋转矩阵行列式不为 1")
    if quaternion.shape != (4,) or not np.isclose(
        np.linalg.norm(quaternion), 1.0, atol=1.0e-9
    ):
        failures.append(f"{name} 四元数不是 xyzw 单位四元数")
    else:
        quaternion_rotation = Rotation.from_quat(quaternion).as_matrix()
        if not np.allclose(quaternion_rotation, rotation, atol=1.0e-9):
            failures.append(f"{name} 四元数与矩阵旋转不一致")
    return matrix


def verify_calibration_file(path: str) -> VerificationResult:
    """独立检查结果正逆矩阵、旋转、四元数和质量门字段。"""
    calibration_path = Path(path)
    try:
        document = yaml.safe_load(
            calibration_path.read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as error:
        return VerificationResult(False, (f"无法读取结果文件: {error}",), {})
    if not isinstance(document, Mapping):
        return VerificationResult(False, ("结果文件顶层必须是映射",), {})
    failures = []
    if document.get("schema_version") != 1:
        failures.append("schema_version 必须为 1")
    tracker = _check_transform_document(
        "tracker_from_camera",
        document.get("tracker_from_camera", {}),
        failures,
        "camera",
        "tracker",
    )
    camera = _check_transform_document(
        "camera_from_tracker",
        document.get("camera_from_tracker", {}),
        failures,
        "tracker",
        "camera",
    )
    _check_transform_document(
        "world_from_board",
        document.get("world_from_board", {}),
        failures,
        "board",
        "world",
    )
    if tracker is not None and camera is not None:
        if not (
            np.allclose(tracker @ camera, np.eye(4), atol=1.0e-9)
            and np.allclose(camera @ tracker, np.eye(4), atol=1.0e-9)
        ):
            failures.append("tracker_from_camera 与 camera_from_tracker 不互逆")
    metrics = document.get("metrics", {})
    if not isinstance(metrics, Mapping):
        failures.append("metrics 必须是映射")
        metrics = {}
    try:
        thresholds = QualityThresholds(**document.get("thresholds", {}))
    except (TypeError, ValueError) as error:
        failures.append(f"thresholds 无法解析: {error}")
        thresholds = QualityThresholds()
    decision = evaluate_quality(metrics, thresholds)
    if bool(document.get("accepted")) != decision.accepted:
        failures.append("accepted 字段与按阈值复算的质量判定不一致")
    if not decision.accepted:
        failures.extend(
            f"质量门未通过: {failure}" for failure in decision.failures
        )
    return VerificationResult(not failures, tuple(failures), dict(metrics))


def main(argv: list[str] | None = None) -> None:
    """解析 ``--verify`` 并以退出码 0/2 表示通过或失败。"""
    parser = argparse.ArgumentParser(description="复核 Tracker–相机标定结果")
    parser.add_argument("--verify", required=True, metavar="CALIBRATION_YAML")
    arguments = parser.parse_args(argv)
    result = verify_calibration_file(arguments.verify)
    print(json.dumps(
        {
            "valid": result.valid,
            "failures": list(result.failures),
            "metrics": dict(result.metrics),
        },
        ensure_ascii=False,
        indent=2,
    ))
    if not result.valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
