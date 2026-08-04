"""验证 Tracker–相机标定质量门、报告文件和独立复核。"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from fastumi_data.pose_math import pose_to_matrix
from fastumi_data.tracker_camera_optimizer import (
    OptimizationResult,
    TimeOffsetScanPoint,
)
from fastumi_data.tracker_camera_report import (
    OverlaySample,
    QualityThresholds,
    ReportContext,
    evaluate_quality,
    verify_calibration_file,
    write_calibration_report,
)


def make_metrics(**overrides: float) -> dict:
    """返回默认满足质量门的优化指标。"""
    metrics = {
        "valid_frames": 48,
        "training_median_px": 0.32,
        "training_p95_px": 0.81,
        "validation_median_px": 0.41,
        "validation_p95_px": 0.92,
        "closure_translation_rmse_mm": 1.2,
        "closure_rotation_rmse_deg": 0.25,
        "time_offset_at_boundary": False,
        "optimizer_cost": 10.0,
        "optimizer_optimality": 1.0e-5,
    }
    metrics.update(overrides)
    return metrics


def make_passing_optimization_result() -> OptimizationResult:
    """构造可序列化的合格优化结果。"""
    tracker_from_camera = pose_to_matrix(
        np.asarray([0.07, -0.025, 0.035]),
        np.asarray([0.0, 0.0, 0.0, 1.0]),
    )
    return OptimizationResult(
        tracker_from_camera=tracker_from_camera,
        camera_from_tracker=np.linalg.inv(tracker_from_camera),
        world_from_board=np.eye(4),
        time_offset_ms=18.2,
        train_indices=(0, 1, 3),
        validation_indices=(2,),
        metrics=make_metrics(),
        scan_points=(
            TimeOffsetScanPoint(16.0, 2.0, 0.4, "PARK", True),
            TimeOffsetScanPoint(18.0, 1.1, 0.2, "PARK", True),
            TimeOffsetScanPoint(20.0, 1.8, 0.35, "PARK", True),
        ),
    )


def make_report_context(tmp_path: Path) -> ReportContext:
    """创建带可哈希输入和一张叠加图的报告上下文。"""
    bag = tmp_path / "input.mcap"
    camera = tmp_path / "camera.yaml"
    target = tmp_path / "target.yaml"
    bag.write_bytes(b"synthetic mcap")
    camera.write_text("camera: synthetic\n", encoding="utf-8")
    target.write_text("target: synthetic\n", encoding="utf-8")
    image = np.full((80, 100, 3), 255, dtype=np.uint8)
    overlay = OverlaySample(
        timestamp_ns=123,
        image_bgr=image,
        observed_points_px=np.asarray([[20.0, 30.0], [40.0, 50.0]]),
        predicted_points_px=np.asarray([[21.0, 30.0], [39.0, 51.0]]),
    )
    return ReportContext(
        bag_path=bag,
        camera_config_path=camera,
        target_config_path=target,
        image_topic="/camera/rgb/image",
        tracker_topic="/vive_tracker/pose",
        status_topic="/vive_tracker/status",
        tag_family="tag36h11",
        settings_snapshot={"frame_stride": 2, "min_tags": 6},
        frame_metrics=(
            {
                "timestamp_ns": 123,
                "partition": "validation",
                "tag_count": 6,
                "pnp_median_px": 0.3,
                "pnp_p95_px": 0.8,
                "final_median_px": 0.4,
                "final_p95_px": 0.9,
                "closure_translation_mm": 1.0,
                "closure_rotation_deg": 0.2,
                "pose_gap_ms": 4.0,
            },
        ),
        overlay_samples=(overlay,),
    )


def test_report_writes_explicit_inverse_transforms(tmp_path: Path) -> None:
    """结果 YAML 中正逆矩阵必须互为逆且标明映射方向。"""
    result = make_passing_optimization_result()
    paths = write_calibration_report(
        tmp_path / "report", result, make_report_context(tmp_path)
    )
    document = yaml.safe_load(
        paths["calibration"].read_text(encoding="utf-8")
    )
    tracker_from_camera = np.asarray(
        document["tracker_from_camera"]["matrix"]
    )
    camera_from_tracker = np.asarray(
        document["camera_from_tracker"]["matrix"]
    )
    np.testing.assert_allclose(
        tracker_from_camera @ camera_from_tracker,
        np.eye(4),
        atol=1.0e-10,
    )
    assert document["quaternion_order"] == "xyzw"
    assert document["accepted"] is True
    assert paths["frame_metrics"].exists()
    assert paths["time_offset_plot"].exists()
    assert paths["residual_plot"].exists()
    assert paths["overlay_000"].exists()
    verification = verify_calibration_file(str(paths["calibration"]))
    assert verification.valid is True
    assert verification.failures == ()


def test_quality_gate_lists_each_failed_metric() -> None:
    """不合格结果应完整列出超限原因。"""
    decision = evaluate_quality(
        make_metrics(
            valid_frames=20,
            validation_median_px=1.4,
            validation_p95_px=2.7,
            closure_translation_rmse_mm=8.0,
            closure_rotation_rmse_deg=1.4,
        ),
        QualityThresholds(),
    )
    assert decision.accepted is False
    assert len(decision.failures) >= 5


def test_verify_rejects_noninverse_transform(tmp_path: Path) -> None:
    """独立复核应拒绝被修改后不再互逆的结果矩阵。"""
    paths = write_calibration_report(
        tmp_path / "report",
        make_passing_optimization_result(),
        make_report_context(tmp_path),
    )
    document = yaml.safe_load(
        paths["calibration"].read_text(encoding="utf-8")
    )
    document["camera_from_tracker"]["matrix"][0][3] += 0.1
    paths["calibration"].write_text(
        yaml.safe_dump(document, allow_unicode=True), encoding="utf-8"
    )
    verification = verify_calibration_file(str(paths["calibration"]))
    assert verification.valid is False
    assert any("互逆" in failure for failure in verification.failures)


def test_report_without_overlays_records_warning(tmp_path: Path) -> None:
    """无可视化样本时仍应生成摘要并明确记录 warning。"""
    context = make_report_context(tmp_path)
    context = ReportContext(
        **{
            **context.__dict__,
            "overlay_samples": (),
        }
    )
    paths = write_calibration_report(
        tmp_path / "report",
        make_passing_optimization_result(),
        context,
    )
    summary = paths["summary"].read_text(encoding="utf-8")
    assert "没有可视化样本" in summary
    assert paths["calibration"].exists()
