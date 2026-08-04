"""验证 Tracker–鱼眼标定 CLI 参数、设置覆盖和退出码。"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fastumi_data.tracker_camera_bag import ImageFrame
from fastumi_data.tracker_camera_config import AprilGridSpec
from fastumi_data.tracker_camera_cli import (
    PipelineOutcome,
    _collect_samples,
    _run_detection_only,
    apply_settings_file,
    build_argument_parser,
    main,
    run_calibration,
)
from fastumi_data.tracker_camera_detection import RawTagDetection


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
    assert arguments.image_topic.endswith("/rgb/image")
    assert arguments.tracker_topic == "/vive_tracker/pose"
    assert arguments.status_topic == "/vive_tracker/status"
    assert arguments.tag_family == "tag36h11"
    assert arguments.frame_stride == 2
    assert arguments.min_tags == 6


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
            "corner_refinement": "subpix",
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
