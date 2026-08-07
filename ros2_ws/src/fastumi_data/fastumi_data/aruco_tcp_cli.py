"""编排双 ArUco Tracker→TCP 离线标定、质量报告和可追溯输出。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from fastumi_data.aruco_tcp_calibration import (
    ArucoTcpCalibrationResult,
    CalibrationThresholds,
    calibrate_frames,
)
from fastumi_data.aruco_tcp_config import (
    ArucoTcpConfig,
    load_aruco_tcp_config,
)
from fastumi_data.aruco_tcp_estimator import (
    DualArucoTcpEstimator,
    FrameEstimationError,
)
from fastumi_data.pose_math import matrix_to_pose
from fastumi_data.tracker_camera_bag import (
    iter_image_frames,
)
from fastumi_data.tracker_camera_config import load_kalibr_camera


DEFAULT_IMAGE_TOPIC = "/xv_sdk/SN250801DR48FB26001253/rgb/image"

# 双 ArUco 标定仅使用图像帧，统一按夹爪完全张开估计 TCP。
FIXED_OPENNESS = 1.0


def _sha256_path(path: Path | str) -> str:
    """流式计算文件或目录内容及相对路径的 SHA-256。"""
    target = Path(path)
    if not target.exists():
        raise ValueError(f"输入路径不存在: {target}")
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


def _transform_document(
    transform: np.ndarray, maps_from: str, maps_to: str
) -> dict[str, Any]:
    """把刚体变换序列化为带方向和单位的 YAML 映射。"""
    matrix = np.asarray(transform, dtype=np.float64)
    position, quaternion = matrix_to_pose(matrix)
    return {
        "maps_from": maps_from,
        "maps_to": maps_to,
        "matrix": matrix.tolist(),
        "translation_m": position.tolist(),
        "quaternion_xyzw": quaternion.tolist(),
    }


def _write_text(path: Path, content: str) -> None:
    """创建父目录并写入 UTF-8 文本。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_tracker_config_serial(path: str | Path) -> str:
    """从 Tracker 参数 YAML 的任意 ROS 参数节点提取 serial。"""
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取 Tracker 配置 {path}: {error}") from error

    def find_serial(value: Any) -> str | None:
        """递归搜索名为 serial 的非空标量。"""
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key) == "serial" and child is not None:
                    serial = str(child).strip()
                    if serial:
                        return serial
                found = find_serial(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = find_serial(child)
                if found:
                    return found
        return None

    serial = find_serial(document)
    if not serial:
        raise ValueError("Tracker 配置必须包含 serial")
    return serial


def _load_tracker_camera_calibration(
    path: str | Path,
) -> tuple[np.ndarray, float, str]:
    """加载已验收的 ``^tracker T_camera`` 和时间偏移。"""
    calibration_path = Path(path)
    try:
        document = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取 Tracker→相机标定 {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise ValueError("Tracker→相机标定必须是 YAML 映射")
    if document.get("accepted") is not True:
        raise ValueError("Tracker→相机源标定必须 accepted=true")
    transform_document = document.get("tracker_from_camera")
    if not isinstance(transform_document, Mapping):
        raise ValueError("Tracker→相机标定缺少 tracker_from_camera")
    matrix = np.asarray(transform_document.get("matrix"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("tracker_from_camera 必须是有限 4x4 矩阵")
    try:
        time_offset_ms = float(document.get("time_offset_ms", 0.0))
    except (TypeError, ValueError) as error:
        raise ValueError("Tracker→相机 time_offset_ms 必须是有限数") from error
    if not np.isfinite(time_offset_ms):
        raise ValueError("Tracker→相机 time_offset_ms 必须是有限数")
    return matrix, time_offset_ms, _sha256_path(calibration_path)


def _overlay_image(image: np.ndarray, frame: Any) -> np.ndarray:
    """在去畸变图像上绘制两个 tag 角点、ID 和 TCP 中心诊断。"""
    canvas = np.asarray(image).copy()
    for tag in (frame.tag0, frame.tag1):
        if tag.corners_px is None:
            continue
        polygon = np.rint(tag.corners_px).astype(np.int32)
        cv2.polylines(canvas, [polygon], True, (0, 255, 0), 2)
        cv2.putText(
            canvas,
            str(tag.tag_id),
            tuple(polygon[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _frame_metric(frame: Any, accepted: bool | None = None) -> dict[str, Any]:
    """把单帧估计转换为 CSV/JSON 可序列化诊断行。"""
    row = {
        "timestamp_ns": int(frame.timestamp_ns),
        "openness": None,
        "tag0_rmse_px": float(frame.tag0_reprojection_rmse_px),
        "tag1_rmse_px": float(frame.tag1_reprojection_rmse_px),
        "measured_distance_m": float(frame.measured_tag_distance_m),
        "expected_distance_m": float(frame.expected_tag_distance_m),
        "distance_error_m": float(
            abs(frame.measured_tag_distance_m - frame.expected_tag_distance_m)
        ),
        "candidate_difference_m": float(
            frame.candidate_translation_difference_m
        ),
    }
    if accepted is not None:
        row["accepted"] = bool(accepted)
    return row


def _write_frame_metrics(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """稳定写入逐帧 CSV，并保留所有额外诊断字段。"""
    fieldnames = [
        "timestamp_ns",
        "openness",
        "tag0_rmse_px",
        "tag1_rmse_px",
        "measured_distance_m",
        "expected_distance_m",
        "distance_error_m",
        "candidate_difference_m",
        "accepted",
    ]
    extras = sorted({key for row in rows for key in row if key not in fieldnames})
    fieldnames.extend(extras)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_calibration_outputs(
    output_dir: Path | str,
    *,
    result: ArucoTcpCalibrationResult,
    aruco_config: ArucoTcpConfig,
    aruco_config_path: Path | str,
    tracker_serial: str,
    time_offset_ms: float,
    bag_uri: Path | str,
    camera_config_path: Path | str,
    tracker_camera_calibration_path: Path | str,
    tracker_config_path: Path | str,
    frame_metrics: Sequence[Mapping[str, Any]],
    overlay_images: Sequence[np.ndarray],
    force: bool = False,
    tracker_from_camera: np.ndarray | None = None,
) -> dict[str, Path]:
    """原子写入标定快照、独立外参、JSON/CSV 报告和叠加图。"""
    destination = Path(output_dir).resolve()
    if destination.exists() and not force:
        raise FileExistsError(f"输出目录已存在；使用 --force 明确覆盖: {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(parent))
    )
    try:
        snapshot_dir = temporary / "calibration_snapshot"
        report_dir = temporary / "calibration_report"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        aruco_snapshot = snapshot_dir / "aruco_to_tcp.yaml"
        shutil.copyfile(aruco_config_path, aruco_snapshot)
        tracker_camera_matrix = (
            np.asarray(tracker_from_camera, dtype=np.float64)
            if tracker_from_camera is not None
            else (
                np.asarray(result.tracker_from_tcp, dtype=np.float64)
                @ np.linalg.inv(np.asarray(result.camera_from_tcp, dtype=np.float64))
                if result.tracker_from_tcp is not None
                and result.camera_from_tcp is not None
                else np.eye(4, dtype=np.float64)
            )
        )
        source_calibration_sha256 = _sha256_path(
            tracker_camera_calibration_path
        )
        source_inputs = {
            "bag": {
                "path": str(Path(bag_uri).resolve()),
                "sha256": _sha256_path(bag_uri),
            },
            "camera_config": {
                "path": str(Path(camera_config_path).resolve()),
                "sha256": _sha256_path(camera_config_path),
            },
            "tracker_camera_calibration": {
                "path": str(
                    Path(tracker_camera_calibration_path).resolve()
                ),
                "sha256": source_calibration_sha256,
            },
            "tracker_config": {
                "path": str(Path(tracker_config_path).resolve()),
                "sha256": _sha256_path(tracker_config_path),
            },
            "aruco_config": {
                "path": str(Path(aruco_config_path).resolve()),
                "sha256": aruco_config.source_sha256,
            },
        }
        report_metrics = dict(result.metrics)
        tracker_document = {
            "schema_version": 2,
            "accepted": True,
            "tracker_serial": tracker_serial,
            "fixture_version": aruco_config.fixture_version,
            "method": "dual_aruco_bootstrap",
            "sample_count": int(result.metrics.get("valid_frames", 0)),
            "translation_rmse_mm": report_metrics.get("translation_rmse_mm"),
            "rotation_rmse_deg": report_metrics.get("rotation_rmse_deg"),
            "time_offset_ms": float(time_offset_ms),
            "source_calibration": {
                "path": str(Path(tracker_camera_calibration_path).resolve()),
                "sha256": source_calibration_sha256,
            },
            "aruco_config_sha256": aruco_config.source_sha256,
            "tracker_from_camera": _transform_document(
                tracker_camera_matrix, "camera", "tracker"
            ),
            "camera_from_tcp": (
                _transform_document(result.camera_from_tcp, "tcp", "camera")
                if result.camera_from_tcp is not None
                else None
            ),
            "tracker_to_tcp": (
                _transform_document(result.tracker_from_tcp, "tcp", "tracker")
                if result.tracker_from_tcp is not None
                else None
            ),
            "quality_metrics": report_metrics,
            "quality_failures": list(result.failures),
        }
        tracker_path = snapshot_dir / "tracker_to_tcp.yaml"
        if result.accepted and result.tracker_from_tcp is not None:
            _write_text(
                tracker_path,
                yaml.safe_dump(
                    tracker_document, allow_unicode=True, sort_keys=False
                ),
            )
        else:
            tracker_path = Path("")
        standalone_transform_path: Path | None = None
        if result.accepted is True and result.tracker_from_tcp is not None:
            standalone_matrix = np.asarray(
                result.tracker_from_tcp, dtype=np.float64
            )
            if standalone_matrix.shape == (4, 4) and np.all(
                np.isfinite(standalone_matrix)
            ):
                standalone_transform_path = (
                    temporary / "tracker_to_tcp_transform.yaml"
                )
                standalone_document = {
                    "tracker_to_tcp": _transform_document(
                        result.tracker_from_tcp, "tcp", "tracker"
                    )
                }
                _write_text(
                    standalone_transform_path,
                    yaml.safe_dump(
                        standalone_document,
                        allow_unicode=True,
                        sort_keys=False,
                    ),
                )
        for index, image in enumerate(overlay_images):
            overlay_path = report_dir / f"overlay_{index:03d}.png"
            if not cv2.imwrite(str(overlay_path), np.asarray(image)):
                raise RuntimeError(f"无法写入标定叠加图: {overlay_path}")
        frame_metrics_path = report_dir / "frame_metrics.csv"
        _write_frame_metrics(frame_metrics_path, frame_metrics)
        summary_outputs = {
            "tracker_to_tcp": (
                "calibration_snapshot/tracker_to_tcp.yaml"
                if result.accepted
                else None
            ),
            "aruco_to_tcp": "calibration_snapshot/aruco_to_tcp.yaml",
            "frame_metrics": "calibration_report/frame_metrics.csv",
        }
        if standalone_transform_path is not None:
            summary_outputs["tracker_to_tcp_transform"] = (
                "tracker_to_tcp_transform.yaml"
            )
        summary = {
            "accepted": bool(result.accepted),
            "calibration_method": "dual_aruco_bootstrap",
            "calibration_assumptions": {
                "fixed_openness": FIXED_OPENNESS,
            },
            "fixture_version": aruco_config.fixture_version,
            "tracker_serial": tracker_serial,
            "time_offset_ms": float(time_offset_ms),
            "metrics": report_metrics,
            "thresholds": asdict(CalibrationThresholds()),
            "rejection_histogram": dict(result.rejection_histogram),
            "quality_failures": list(result.failures),
            "inputs": source_inputs,
            "outputs": summary_outputs,
        }
        summary_path = report_dir / "summary.json"
        _write_text(
            summary_path,
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    output_paths = {
        "summary": destination / "calibration_report" / "summary.json",
        "frame_metrics": destination / "calibration_report" / "frame_metrics.csv",
        "calibration_report": destination / "calibration_report",
        "aruco_snapshot": destination / "calibration_snapshot" / "aruco_to_tcp.yaml",
    }
    if result.accepted:
        output_paths["tracker_to_tcp"] = (
            destination / "calibration_snapshot" / "tracker_to_tcp.yaml"
        )
    if standalone_transform_path is not None:
        output_paths["tracker_to_tcp_transform"] = (
            destination / "tracker_to_tcp_transform.yaml"
        )
    return output_paths


def build_argument_parser() -> argparse.ArgumentParser:
    """创建双 ArUco 标定 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="从去畸变双 ArUco 和 Tracker→Camera 标定生成 Tracker→TCP"
    )
    parser.add_argument("bag_uri", help="唯一 MCAP 输入目录")
    parser.add_argument("--camera-config", required=True)
    parser.add_argument("--aruco-config", required=True)
    parser.add_argument("--tracker-camera-calibration", required=True)
    parser.add_argument("--tracker-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--minimum-frames", type=int, default=30)
    parser.add_argument("--max-reprojection-rmse-px", type=float, default=1.5)
    parser.add_argument("--max-distance-error-mm", type=float, default=5.0)
    parser.add_argument("--max-candidate-difference-mm", type=float, default=5.0)
    parser.add_argument("--max-translation-p95-mm", type=float, default=3.0)
    parser.add_argument("--max-rotation-p95-deg", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    return parser


def _thresholds_from_arguments(arguments: argparse.Namespace) -> CalibrationThresholds:
    """把 CLI 毫米门限转换为内部米制质量门。"""
    return CalibrationThresholds(
        minimum_valid_frames=int(arguments.minimum_frames),
        max_reprojection_rmse_px=float(arguments.max_reprojection_rmse_px),
        max_distance_error_m=float(arguments.max_distance_error_mm) / 1000.0,
        max_candidate_difference_m=(
            float(arguments.max_candidate_difference_mm) / 1000.0
        ),
        max_translation_p95_m=float(arguments.max_translation_p95_mm) / 1000.0,
        max_rotation_p95_deg=float(arguments.max_rotation_p95_deg),
    )


def run_calibration(arguments: argparse.Namespace) -> ArucoTcpCalibrationResult:
    """单遍读取图像，以固定全开开度估计、聚合并写入标定报告。"""
    if arguments.frame_stride <= 0:
        raise ValueError("frame_stride 必须为正整数")
    camera = load_kalibr_camera(arguments.camera_config)
    aruco_config = load_aruco_tcp_config(arguments.aruco_config)
    tracker_from_camera, time_offset_ms, _ = _load_tracker_camera_calibration(
        arguments.tracker_camera_calibration
    )
    tracker_serial = _load_tracker_config_serial(arguments.tracker_config)
    estimator = DualArucoTcpEstimator(camera, aruco_config)
    frames = []
    frame_metrics = []
    overlays: list[np.ndarray] = []
    rejection_histogram: dict[str, int] = {}
    for image_frame in iter_image_frames(
        arguments.bag_uri,
        arguments.image_topic,
        frame_stride=arguments.frame_stride,
    ):
        try:
            frame = estimator.estimate(
                image_frame.image,
                FIXED_OPENNESS,
                image_frame.timestamp_ns,
            )
        except FrameEstimationError as error:
            reason = str(error)
            rejection_histogram[reason] = rejection_histogram.get(reason, 0) + 1
            continue
        frames.append(frame)
        metric = _frame_metric(frame)
        metric["openness"] = FIXED_OPENNESS
        frame_metrics.append(metric)
        if len(overlays) < 5:
            rectified = cv2.remap(
                image_frame.image,
                estimator.rectification_maps[0],
                estimator.rectification_maps[1],
                interpolation=cv2.INTER_LINEAR,
            )
            overlays.append(_overlay_image(rectified, frame))
    result = calibrate_frames(
        frames,
        tracker_from_camera,
        _thresholds_from_arguments(arguments),
    )
    output_dir = Path(arguments.output_dir)
    write_calibration_outputs(
        output_dir,
        result=result,
        aruco_config=aruco_config,
        aruco_config_path=arguments.aruco_config,
        tracker_serial=tracker_serial,
        time_offset_ms=time_offset_ms,
        bag_uri=arguments.bag_uri,
        camera_config_path=arguments.camera_config,
        tracker_camera_calibration_path=arguments.tracker_camera_calibration,
        tracker_config_path=arguments.tracker_config,
        frame_metrics=frame_metrics,
        overlay_images=overlays,
        force=arguments.force,
        tracker_from_camera=tracker_from_camera,
    )
    return result


def main(argv: list[str] | None = None) -> None:
    """解析参数、运行标定并在质量门失败时返回非零退出码。"""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    result = run_calibration(arguments)
    print(
        json.dumps(
            {
                "accepted": result.accepted,
                "metrics": dict(result.metrics),
                "failures": list(result.failures),
            },
            ensure_ascii=False,
        )
    )
    if not result.accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
