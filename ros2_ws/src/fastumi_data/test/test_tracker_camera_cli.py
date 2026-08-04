"""验证 Tracker–鱼眼标定 CLI 参数、设置覆盖和退出码。"""

import argparse
from pathlib import Path

import pytest

from fastumi_data.tracker_camera_cli import (
    PipelineOutcome,
    apply_settings_file,
    build_argument_parser,
    main,
)


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
