"""验证 Tracker–鱼眼标定 CLI 参数、设置覆盖和退出码。"""

import argparse
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fastumi_data.tracker_camera_bag import ImageFrame
from fastumi_data.tracker_camera_config import AprilGridSpec
from fastumi_data.tracker_camera_cli import (
    _OptimizationProgressAdapter,
    PipelineOutcome,
    _collect_samples,
    _run_detection_only,
    apply_settings_file,
    build_argument_parser,
    main,
    run_calibration,
)
from fastumi_data.tracker_camera_detection import RawTagDetection
from fastumi_data.tracker_camera_progress import CalibrationProgressLogger


def minimum_arguments(extra: list[str] | None = None) -> list[str]:
    """返回 CLI 必需参数并附加可选测试参数。"""
    arguments = [
        "--bag",
        "/data/example",
        "--camera-config",
        "camera.yaml",
        "--target-config",
        "target.yaml",
        "--output-dir",
        "/tmp/result",
    ]
    return arguments + (extra or [])


def test_parser_defaults_match_target_bag_topics() -> None:
    """默认话题应匹配已确认的 Tracker–鱼眼 bag。"""
    arguments = build_argument_parser().parse_args(minimum_arguments())
    assert arguments.image_topic == "/tof_stereo_camera/rgb/image_raw"
    assert arguments.tracker_topic == "/vive_tracker/pose"
    assert arguments.status_topic == "/vive_tracker/status"
    assert arguments.camera_config == "camera.yaml"
    assert arguments.tag_family == "tag36h11"
    assert arguments.frame_stride == 2
    assert arguments.min_tags == 6
    assert arguments.progress_interval_seconds == pytest.approx(30.0)


def test_parser_defaults_to_tof_camera_calibration() -> None:
    """省略相机配置时应读取 ToF 包内默认标定。"""
    arguments = build_argument_parser().parse_args(
        [
            "--bag",
            "/data/example",
            "--target-config",
            "target.yaml",
            "--output-dir",
            "/tmp/result",
        ]
    )
    camera_path = Path(arguments.camera_config)
    assert camera_path.name == "calibration.yaml"
    assert camera_path.parent.name == "config"
    assert camera_path.parent.parent.name == "tof_stereo_camera"


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_parser_rejects_invalid_progress_interval(value: str) -> None:
    """进度心跳间隔必须是有限正数。"""
    with pytest.raises(SystemExit) as error:
        build_argument_parser().parse_args(
            minimum_arguments(["--progress-interval-seconds", value])
        )
    assert error.value.code == 2


def test_settings_file_overrides_defaults_but_cli_wins(tmp_path: Path) -> None:
    """设置 YAML 应覆盖默认值，显式命令行参数保持最高优先级。"""
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        """topics:
  image: /configured/rgb/image
filtering:
  frame_stride: 4
  min_tags: 8
optimization:
  time_offset_min_ms: -40.0
""",
        encoding="utf-8",
    )
    parser = build_argument_parser()
    arguments = parser.parse_args(
        minimum_arguments(
            [
                "--settings-config",
                str(settings_path),
                "--min-tags",
                "10",
            ]
        )
    )
    merged = apply_settings_file(arguments, parser)
    assert merged.image_topic == "/configured/rgb/image"
    assert merged.frame_stride == 4
    assert merged.min_tags == 10
    assert merged.time_offset_min_ms == pytest.approx(-40.0)


def test_main_exits_nonzero_when_quality_gate_fails(monkeypatch) -> None:
    """未显式放宽门限时，不合格标定必须返回失败。"""
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.run_calibration",
        lambda arguments: PipelineOutcome(accepted=False, output_paths={}),
    )
    with pytest.raises(SystemExit) as error:
        main(minimum_arguments())
    assert error.value.code == 2


def test_allow_high_residual_preserves_zero_exit(monkeypatch) -> None:
    """显式放宽质量门时可保留报告并以零退出。"""
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.run_calibration",
        lambda arguments: PipelineOutcome(accepted=False, output_paths={}),
    )
    assert main(minimum_arguments(["--allow-high-residual"])) is None


def test_apply_settings_rejects_unknown_keys(tmp_path: Path) -> None:
    """设置文件中的未知键应被拒绝，避免拼写错误静默失效。"""
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "filtering:\n  frame_strde: 3\n", encoding="utf-8"
    )
    parser = build_argument_parser()
    arguments: argparse.Namespace = parser.parse_args(
        minimum_arguments(["--settings-config", str(settings_path)])
    )
    with pytest.raises(ValueError, match="未知设置"):
        apply_settings_file(arguments, parser)


def test_detect_only_failure_still_writes_diagnostics(
    tmp_path: Path, monkeypatch
) -> None:
    """有效帧不足时也应保存统计和已有叠加图供补采判断。"""
    frames = [
        ImageFrame(index, index, np.full((40, 40, 3), 255, np.uint8))
        for index in range(3)
    ]
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.iter_image_frames",
        lambda *args, **kwargs: iter(frames),
    )
    detector = SimpleNamespace(
        settings={
            "max_correction_bits": 3,
            "error_correction_rate": 1.0,
            "perspective_remove_pixel_per_cell": 16,
            "perspective_remove_ignored_margin_per_cell": 0.25,
            "corner_refinement": "contour",
        },
        detect=lambda image: [
            RawTagDetection(
                0,
                np.asarray(
                    [
                        [5.0, 5.0],
                        [20.0, 5.0],
                        [20.0, 20.0],
                        [5.0, 20.0],
                    ]
                ),
                None,
                None,
            )
        ]
    )
    arguments = SimpleNamespace(
        output_dir=str(tmp_path),
        bag="unused",
        image_topic="/camera/image",
        frame_stride=1,
        min_tags=1,
    )
    outcome = _run_detection_only(
        arguments,
        detector,
        AprilGridSpec(6, 6, 0.055, 0.3, "tag36h11"),
    )
    assert outcome.accepted is False
    summary = json.loads(
        outcome.output_paths["detection_summary"].read_text(encoding="utf-8")
    )
    assert summary["valid_frames"] == 3
    assert any("至少 5 帧" in failure for failure in summary["failures"])
    assert summary["tag_count_histogram"] == {"1": 3}
    assert summary["tag_id_frame_counts"] == {"0": 3}
    assert summary["detector_settings"]["max_correction_bits"] == 3
    assert outcome.output_paths["overlay_000"].exists()


def test_full_mode_insufficient_frames_writes_failure_summary(
    tmp_path: Path, monkeypatch
) -> None:
    """完整模式样本不足时应写结构化失败摘要，而不抛出异常。"""
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.load_kalibr_camera",
        lambda path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.load_aprilgrid",
        lambda path, family: AprilGridSpec(6, 6, 0.055, 0.3, family),
    )
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.OpenCvAprilTagDetector",
        lambda family: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.read_tracker_timeline",
        lambda *args: SimpleNamespace(poses=(), statuses=()),
    )
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli._collect_samples",
        lambda *args: ([], [], [], {}, {"decoded": 7}),
    )
    output_dir = tmp_path / "report"
    arguments = build_argument_parser().parse_args(
        minimum_arguments(["--output-dir", str(output_dir)])
    )
    outcome = run_calibration(arguments)
    assert outcome.accepted is False
    summary = json.loads(
        outcome.output_paths["summary"].read_text(encoding="utf-8")
    )
    assert summary["stage"] == "sample_collection"
    assert summary["counters"]["decoded"] == 7
    assert "有效标定帧" in summary["failure"]
    progress_log = outcome.output_paths["calibration_log"]
    assert progress_log.exists()
    log_text = progress_log.read_text(encoding="utf-8")
    assert "stage=load_inputs" in log_text
    assert "stage=sample_collection" in log_text
    assert "status=FAILED" in log_text


def test_collection_checks_configured_offset_status_window(
    monkeypatch,
) -> None:
    """单侧偏移搜索应检查配对窗口，不能按图像原时刻提前拒绝。"""
    frame = ImageFrame(1_000_000_000, 1_000_000_000, np.zeros((4, 4, 3)))
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.iter_image_frames",
        lambda *args: iter([frame]),
    )
    checked_intervals = []

    def valid_interval(samples, start_ns, end_ns, maximum_delta_ms, timestamps):
        """记录采样阶段使用的状态时间窗口。"""
        checked_intervals.append((start_ns, end_ns))
        return True

    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.tracker_status_valid_for_interval",
        valid_interval,
    )
    observation = SimpleNamespace(
        object_points_m=np.zeros((4, 3)),
        image_points_px=np.zeros((4, 2)),
        tag_count=1,
    )
    estimate = SimpleNamespace(camera_from_board=np.eye(4))
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.build_aprilgrid_observation",
        lambda *args: observation,
    )
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.estimate_camera_from_board",
        lambda *args: estimate,
    )
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.interpolate_world_from_tracker",
        lambda *args: (np.eye(4), 1.0),
    )
    arguments = SimpleNamespace(
        bag="unused", image_topic="/camera/image", frame_stride=1,
        max_pose_gap_ms=50.0, min_tags=1,
        time_offset_min_ms=20.0, time_offset_max_ms=60.0,
    )
    timeline = SimpleNamespace(
        poses=(object(), object()), statuses=(object(),),
        pose_timestamps_ns=np.asarray([0, 2_000_000_000]),
        status_timestamps_ns=np.asarray([1_020_000_000]),
    )
    samples, _, _, _, counters = _collect_samples(
        arguments, SimpleNamespace(detect=lambda image: []),
        SimpleNamespace(), SimpleNamespace(), timeline,
    )
    assert len(samples) == 1
    assert counters["status_rejected"] == 0
    assert checked_intervals == [(1_020_000_000, 1_060_000_000)]


def test_collection_writes_final_progress_counts(
    tmp_path: Path, monkeypatch
) -> None:
    """样本采集结束时应强制输出解码帧数和有效样本数。"""
    frames = [
        ImageFrame(index, index, np.zeros((4, 4, 3)))
        for index in range(3)
    ]
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.iter_image_frames",
        lambda *args: iter(frames),
    )
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.tracker_status_valid_for_interval",
        lambda *args: True,
    )
    observation = SimpleNamespace(
        object_points_m=np.zeros((4, 3)),
        image_points_px=np.zeros((4, 2)),
        tag_count=1,
    )
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.build_aprilgrid_observation",
        lambda *args: observation,
    )
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.estimate_camera_from_board",
        lambda *args: SimpleNamespace(camera_from_board=np.eye(4)),
    )
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.interpolate_world_from_tracker",
        lambda *args: (np.eye(4), 1.0),
    )
    arguments = SimpleNamespace(
        bag="unused", image_topic="/camera/image", frame_stride=1,
        max_pose_gap_ms=50.0, min_tags=1,
        time_offset_min_ms=-20.0, time_offset_max_ms=20.0,
    )
    timeline = SimpleNamespace(
        poses=(object(), object()), statuses=(object(),),
        pose_timestamps_ns=np.asarray([0, 10]),
        status_timestamps_ns=np.asarray([0]),
    )
    stream = StringIO()
    with CalibrationProgressLogger(tmp_path, stream=stream) as progress:
        samples, _, _, _, counters = _collect_samples(
            arguments, SimpleNamespace(detect=lambda image: []),
            SimpleNamespace(), SimpleNamespace(), timeline, progress,
        )

    output = stream.getvalue()
    assert len(samples) == 3
    assert counters["decoded"] == 3
    assert "stage=sample_collection" in output
    assert "decoded=3" in output
    assert "valid=3" in output
    assert "status=COMPLETED" in output


def test_optimization_progress_adapter_reports_exact_and_upper_bound_eta(
    tmp_path: Path,
) -> None:
    """粗扫描应给出精确进度，联合优化应标明 ETA 是保守上限。"""
    now = [100.0]
    stream = StringIO()
    with CalibrationProgressLogger(
        tmp_path, clock=lambda: now[0], stream=stream
    ) as progress:
        adapter = _OptimizationProgressAdapter(progress)
        adapter("time_offset_scan", {
            "completed": 1, "total": 5, "offset_ms": -40.0,
            "valid": True, "best_offset_ms": -40.0,
        })
        now[0] += 30.0
        adapter("time_offset_scan", {
            "completed": 2, "total": 5, "offset_ms": -20.0,
            "valid": True, "best_offset_ms": -20.0,
        })
        adapter("joint_optimization_start", {
            "training_samples": 100, "validation_samples": 20,
            "max_nfev": 2000,
        })
        now[0] += 30.0
        adapter("joint_optimization_evaluation", {
            "residual_calls": 140, "approx_nfev": 10,
            "max_nfev": 2000, "residual_rms_px": 1.25,
        })
        adapter("joint_optimization_complete", {
            "residual_calls": 280, "nfev": 20, "status": 2,
            "message": "converged", "cost": 12.0,
        })

    output = stream.getvalue()
    assert "stage=time_offset_scan" in output
    assert "progress=40.0%" in output
    assert "best_offset_ms=-20.000" in output
    assert "stage=joint_optimization" in output
    assert "近似进度" in output
    assert "保守上限 ETA" in output
    assert "nfev=20" in output
    assert "status=COMPLETED" in output
