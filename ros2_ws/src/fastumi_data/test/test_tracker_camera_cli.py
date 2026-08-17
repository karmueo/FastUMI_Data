"""验证 Tracker–鱼眼标定 CLI 参数、设置覆盖和退出码。"""

import argparse
import shutil
import tempfile
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fastumi_data.tracker_camera_cli as cli_module
from fastumi_data.tracker_camera_bag import ImageFrame
from fastumi_data.tracker_camera_config import AprilGridSpec
from fastumi_data.tracker_camera_cli import (
    _OptimizationProgressAdapter,
    PipelineOutcome,
    _collect_samples,
    _iter_sample_frames,
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
    assert arguments.tracker_topic == "/vive_tracker/odom"
    assert arguments.status_topic == "/vive_tracker/status"
    assert arguments.camera_config == "camera.yaml"
    assert arguments.tag_family == "tag36h11"
    assert arguments.frame_stride == 2
    assert arguments.min_tags == 6
    assert arguments.sample_end_offset_s is None
    assert arguments.progress_interval_seconds == pytest.approx(30.0)


def test_parser_omitted_camera_config_enables_automatic_intrinsics() -> None:
    """省略相机配置时应请求自动 Kalibr 内参标定。"""
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
    assert arguments.camera_config is None
    assert arguments.intrinsics_frequency_hz == pytest.approx(4.0)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_parser_rejects_invalid_progress_interval(value: str) -> None:
    """进度心跳间隔必须是有限正数。"""
    with pytest.raises(SystemExit) as error:
        build_argument_parser().parse_args(
            minimum_arguments(["--progress-interval-seconds", value])
        )
    assert error.value.code == 2


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_parser_rejects_invalid_sample_end_offset(value: str) -> None:
    """样本结束偏移必须是有限非负数。"""
    with pytest.raises(SystemExit) as error:
        build_argument_parser().parse_args(
            minimum_arguments(["--sample-end-offset-s", value])
        )
    assert error.value.code == 2


def test_sample_window_stops_after_header_time_limit(monkeypatch) -> None:
    """样本结束偏移应以首个抽帧图像 header 时间为基准并包含边界。"""
    frames = [
        ImageFrame(timestamp, timestamp, np.zeros((2, 2, 3), np.uint8))
        for timestamp in (1_000_000_000, 2_000_000_000, 3_000_000_001)
    ]
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.iter_image_frames",
        lambda *args, **kwargs: iter(frames),
    )
    arguments = SimpleNamespace(
        bag="unused", image_topic="/camera/image", frame_stride=1,
        sample_end_offset_s=1.0,
    )
    selected = list(_iter_sample_frames(arguments))
    assert [frame.timestamp_ns for frame in selected] == [
        1_000_000_000, 2_000_000_000,
    ]


def test_settings_file_overrides_defaults_but_cli_wins(tmp_path: Path) -> None:
    """设置 YAML 应覆盖默认值，显式命令行参数保持最高优先级。"""
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        """topics:
  image: /configured/rgb/image
intrinsics:
  frequency_hz: 3.0
filtering:
  frame_stride: 4
  min_tags: 8
  sample_end_offset_s: 90.0
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
    assert merged.sample_end_offset_s == pytest.approx(90.0)
    assert merged.time_offset_min_ms == pytest.approx(-40.0)
    assert merged.intrinsics_frequency_hz == pytest.approx(3.0)


def test_settings_file_converts_sample_end_offset_string(
    tmp_path: Path,
) -> None:
    """设置文件中的数值字符串应按 CLI 规则转换为浮点数。"""
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        'filtering:\n  sample_end_offset_s: "90.0"\n',
        encoding="utf-8",
    )
    parser = build_argument_parser()
    arguments = parser.parse_args(
        minimum_arguments(["--settings-config", str(settings_path)])
    )

    merged = apply_settings_file(arguments, parser)

    assert merged.sample_end_offset_s == pytest.approx(90.0)
    assert isinstance(merged.sample_end_offset_s, float)


@pytest.mark.parametrize("value", ["-1", ".nan", ".inf", "invalid"])
def test_settings_file_rejects_invalid_sample_end_offset(
    tmp_path: Path, value: str,
) -> None:
    """设置文件中的样本结束偏移也必须是有限非负数。"""
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        f"filtering:\n  sample_end_offset_s: {value}\n",
        encoding="utf-8",
    )
    parser = build_argument_parser()
    arguments = parser.parse_args(
        minimum_arguments(["--settings-config", str(settings_path)])
    )

    with pytest.raises(
        ValueError, match=r"filtering\.sample_end_offset_s"
    ):
        apply_settings_file(arguments, parser)


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
    monkeypatch.setattr("fastumi_data.tracker_camera_cli.sha256_path", lambda path: "test-hash")
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


def test_main_fatal_outcome_ignores_allow_high_residual(monkeypatch) -> None:
    """自动内参致命失败必须始终让主程序返回 2。"""
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.run_calibration",
        lambda arguments: PipelineOutcome(False, {}, fatal=True),
    )
    with pytest.raises(SystemExit) as error:
        main(minimum_arguments(["--allow-high-residual"]))
    assert error.value.code == 2


def test_explicit_camera_config_bypasses_automatic_calibration(
    tmp_path: Path, monkeypatch
) -> None:
    """显式内参配置不能触发自动 Kalibr。"""
    config = tmp_path / "camera.yaml"
    config.write_text("camera: test\n", encoding="utf-8")
    monkeypatch.setattr("fastumi_data.tracker_camera_cli.load_kalibr_camera", lambda path: SimpleNamespace())
    monkeypatch.setattr("fastumi_data.tracker_camera_cli.load_aprilgrid", lambda *args: SimpleNamespace())
    monkeypatch.setattr("fastumi_data.tracker_camera_cli.OpenCvAprilTagDetector", lambda *args: SimpleNamespace())
    monkeypatch.setattr("fastumi_data.tracker_camera_cli._calibrate_automatic_camera", lambda *args: (_ for _ in ()).throw(AssertionError("automatic")))
    monkeypatch.setattr("fastumi_data.tracker_camera_cli.read_tracker_timeline", lambda *args: SimpleNamespace(poses=(), statuses=()))
    monkeypatch.setattr("fastumi_data.tracker_camera_cli._collect_samples", lambda *args: ([], [], [], {}, {}))
    arguments = build_argument_parser().parse_args(
        ["--bag", "bag", "--camera-config", str(config), "--target-config", "target", "--output-dir", str(tmp_path / "out")]
    )
    assert run_calibration(arguments).accepted is False


def test_detect_only_bypasses_camera_loading_and_automatic(
    tmp_path: Path, monkeypatch
) -> None:
    """detect-only 在显式或自动内参分支前返回。"""
    monkeypatch.setattr("fastumi_data.tracker_camera_cli.load_kalibr_camera", lambda *args: (_ for _ in ()).throw(AssertionError("camera")))
    monkeypatch.setattr("fastumi_data.tracker_camera_cli._calibrate_automatic_camera", lambda *args: (_ for _ in ()).throw(AssertionError("automatic")))
    monkeypatch.setattr("fastumi_data.tracker_camera_cli.load_aprilgrid", lambda *args: SimpleNamespace())
    monkeypatch.setattr("fastumi_data.tracker_camera_cli.OpenCvAprilTagDetector", lambda *args: SimpleNamespace())
    monkeypatch.setattr("fastumi_data.tracker_camera_cli._run_detection_only", lambda *args: PipelineOutcome(True, {}))
    arguments = build_argument_parser().parse_args(
        ["--bag", "bag", "--target-config", "target", "--output-dir", str(tmp_path / "out"), "--detect-only"]
    )
    assert run_calibration(arguments).accepted is True


@pytest.mark.parametrize("storage_error", [ValueError, RuntimeError])
def test_automatic_failure_invalidates_marker_and_is_structured(
    tmp_path: Path, monkeypatch, storage_error
) -> None:
    """自动抽取存储失败应删除旧标记并返回结构化致命摘要。"""
    marker = tmp_path / "out" / "camera_intrinsics.yaml"
    marker.parent.mkdir()
    marker.write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr("fastumi_data.tracker_camera_cli.resolve_kalibr_package_prefix", lambda: "/overlay")
    monkeypatch.setattr(
        "fastumi_data.tracker_camera_cli.extract_image_topic_to_sqlite3",
        lambda *args: (_ for _ in ()).throw(storage_error("extract failed")),
    )
    monkeypatch.setattr("fastumi_data.tracker_camera_cli.load_aprilgrid", lambda *args: SimpleNamespace())
    monkeypatch.setattr("fastumi_data.tracker_camera_cli.OpenCvAprilTagDetector", lambda *args: SimpleNamespace())
    arguments = build_argument_parser().parse_args(
        ["--bag", "bag", "--target-config", "target", "--output-dir", str(marker.parent)]
    )
    outcome = run_calibration(arguments)
    assert outcome.fatal is True
    assert not marker.exists()
    assert json.loads((marker.parent / "summary.json").read_text(encoding="utf-8"))["stage"] == "intrinsic_calibration"


def test_automatic_helper_cleans_tempdir_after_extraction_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """自动内参失败后应删除临时 SQLite3/staging 根目录并失效旧 YAML。"""
    target = tmp_path / "target.yaml"
    target.write_text("target_type: aprilgrid\n", encoding="utf-8")
    roots = []
    class TrackingTemporaryDirectory:
        def __init__(self, **kwargs):
            self.path = Path(tempfile.mkdtemp(dir=tmp_path))
            roots.append(self.path)
        def __enter__(self):
            return str(self.path)
        def __exit__(self, *args):
            shutil.rmtree(self.path)
    marker = tmp_path / "out" / "camera_intrinsics.yaml"
    marker.parent.mkdir()
    marker.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(cli_module.tempfile, "TemporaryDirectory", TrackingTemporaryDirectory)
    monkeypatch.setattr(cli_module, "resolve_kalibr_package_prefix", lambda: "/overlay")
    monkeypatch.setattr(cli_module, "extract_image_topic_to_sqlite3", lambda *args: (_ for _ in ()).throw(ValueError("bad bag")))
    arguments = SimpleNamespace(
        output_dir=str(marker.parent), bag="bag", image_topic="/camera/image",
        intrinsics_frequency_hz=4.0, sample_end_offset_s=None, target_config=str(target),
    )
    with pytest.raises(Exception, match="bad bag"):
        cli_module._calibrate_automatic_camera(arguments)
    assert not marker.exists()
    assert roots and not roots[0].exists()


def test_automatic_helper_persists_kalibr_failure_log_after_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    """Kalibr 非零失败日志必须在临时目录销毁后留在输出目录。"""
    output = tmp_path / "out"
    target = tmp_path / "target.yaml"
    target.write_text("target_type: aprilgrid\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "resolve_kalibr_package_prefix", lambda: "/overlay")
    monkeypatch.setattr(
        cli_module, "extract_image_topic_to_sqlite3",
        lambda *args: SimpleNamespace(bag_uri=tmp_path / "images", resolution=(4, 3), frame_count=2),
    )
    def fail_kalibr(command, staging, log_path):
        Path(log_path).write_text("argv: ros2 run kalibr\nexit_status: 9\n", encoding="utf-8")
        raise cli_module.IntrinsicCalibrationError("Kalibr 返回非零退出码 9", Path(log_path))
    monkeypatch.setattr(cli_module, "run_kalibr", fail_kalibr)
    arguments = SimpleNamespace(
        output_dir=str(output), bag="bag", image_topic="/camera/image",
        intrinsics_frequency_hz=4.0, sample_end_offset_s=None, target_config=str(target),
    )
    with pytest.raises(cli_module.IntrinsicCalibrationError) as error:
        cli_module._calibrate_automatic_camera(arguments)
    assert error.value.log_path == output / "camera_intrinsics.log"
    assert error.value.log_path.exists()
    assert "argv: ros2 run kalibr" in error.value.log_path.read_text(encoding="utf-8")
    assert "exit_status: 9" in error.value.log_path.read_text(encoding="utf-8")


def test_automatic_helper_publishes_runner_named_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    """自动流程应从与临时 bag 同目录的 Kalibr 固定命名产物发布最终内参。"""
    output = tmp_path / "out"
    target = tmp_path / "target.yaml"
    target.write_text("target_type: aprilgrid\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_module, "resolve_kalibr_package_prefix", lambda: "/overlay")
    def fake_extract(*args):
        bag_path = Path(args[2])
        bag_path.mkdir()
        return SimpleNamespace(bag_uri=bag_path, resolution=(640, 480), frame_count=3)
    monkeypatch.setattr(cli_module, "extract_image_topic_to_sqlite3", fake_extract)
    def fake_runner(command, staging, log_path):
        staging_path = Path(staging)
        target_argument = Path(command[command.index("--target") + 1])
        assert target_argument == target.resolve()
        assert target_argument.is_absolute()
        assert target_argument.is_file()
        stem = Path(command[command.index("--bag") + 1]).name
        (staging_path / f"camchain-{stem}.yaml").write_text(
            "cam0:\n  camera_model: pinhole\n  distortion_model: equidistant\n"
            "  intrinsics: [400.0, 401.0, 320.0, 240.0]\n"
            "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
            "  resolution: [640, 480]\n  rostopic: /camera/image\n", encoding="utf-8"
        )
        (staging_path / f"results-cam-{stem}.txt").write_text("results\n", encoding="utf-8")
        (staging_path / f"report-cam-{stem}.pdf").write_bytes(b"%PDF")
        Path(log_path).write_text("argv: fake\nexit_status: 0\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "run_kalibr", fake_runner)
    arguments = SimpleNamespace(
        output_dir=str(output), bag="bag", image_topic="/camera/image",
        intrinsics_frequency_hz=4.0, sample_end_offset_s=None, target_config="target.yaml",
    )
    camera, provenance = cli_module._calibrate_automatic_camera(arguments)
    assert camera.resolution == (640, 480)
    assert provenance["source"] == "kalibr_ros2"
    assert provenance["expected_kalibr_commit"]
    assert provenance["expected_compatibility_patch_sha256"]
    assert "kalibr_commit" not in provenance
    assert "compatibility_patch_sha256" not in provenance
    assert (output / "camera_intrinsics.yaml").exists()
    assert (output / "camera_intrinsics_results.txt").exists()
    assert (output / "camera_intrinsics_report.pdf").exists()
    assert (output / "camera_intrinsics.log").exists()


def test_automatic_extraction_failure_does_not_reuse_old_log(
    tmp_path: Path, monkeypatch
) -> None:
    """本次提取失败时不得把输出目录中旧 Kalibr 日志附到新异常。"""
    output = tmp_path / "out"
    output.mkdir()
    target = tmp_path / "target.yaml"
    target.write_text("target_type: aprilgrid\n", encoding="utf-8")
    old_log = output / "camera_intrinsics.log"
    old_log.write_text("old run\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "resolve_kalibr_package_prefix", lambda: "/overlay")
    monkeypatch.setattr(
        cli_module, "extract_image_topic_to_sqlite3",
        lambda *args: (_ for _ in ()).throw(ValueError("new extraction failure")),
    )
    arguments = SimpleNamespace(
        output_dir=str(output), bag="bag", image_topic="/camera/image",
        intrinsics_frequency_hz=4.0, sample_end_offset_s=None, target_config=str(target),
    )
    with pytest.raises(cli_module.IntrinsicCalibrationError) as error:
        cli_module._calibrate_automatic_camera(arguments)
    assert error.value.log_path is None
    assert not old_log.exists()
