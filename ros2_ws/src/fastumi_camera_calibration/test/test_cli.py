"""验证独立相机内参标定 CLI 参数、摘要和退出状态。"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import fastumi_camera_calibration.cli as cli
from fastumi_camera_calibration.calibration import IntrinsicCalibrationError


def minimum_arguments() -> list[str]:
    """返回独立内参命令的最小必填参数。"""
    return [
        "--bag", "/data/example",
        "--target-config", "target.yaml",
        "--output-dir", "/tmp/result",
    ]


def test_parser_defaults_and_numeric_validation() -> None:
    """解析器应提供固定话题、4 Hz 默认值和有限数值约束。"""
    arguments = cli.build_argument_parser().parse_args(minimum_arguments())
    assert arguments.image_topic == "/tof_stereo_camera/rgb/image_raw"
    assert arguments.frequency_hz == pytest.approx(4.0)
    assert arguments.sample_end_offset_s is None
    with pytest.raises(SystemExit) as error:
        cli.build_argument_parser().parse_args(
            minimum_arguments() + ["--frequency-hz", "0"]
        )
    assert error.value.code == 2


def test_main_prints_success_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """成功时应输出四类文件路径和 provenance JSON。"""
    # 模拟最终产物集合。
    artifacts = SimpleNamespace(
        yaml_path=tmp_path / "camera_intrinsics.yaml",
        results_path=tmp_path / "camera_intrinsics_results.txt",
        report_path=tmp_path / "camera_intrinsics_report.pdf",
        log_path=tmp_path / "camera_intrinsics.log",
    )
    monkeypatch.setattr(
        cli,
        "calibrate_camera_intrinsics",
        lambda *args: SimpleNamespace(
            artifacts=artifacts,
            provenance={"source": "kalibr_ros2"},
        ),
    )
    assert cli.main(minimum_arguments()) is None
    payload = json.loads(capsys.readouterr().out)
    assert payload["accepted"] is True
    assert payload["provenance"]["source"] == "kalibr_ros2"
    assert payload["output_paths"]["yaml"].endswith(
        "camera_intrinsics.yaml"
    )


def test_main_returns_two_and_reports_retained_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """失败时应返回 2，并且只报告本次保留日志。"""
    # 本次失败日志路径由核心流水线附在异常上。
    log_path = tmp_path / "camera_intrinsics.log"

    def fail(*args: object) -> None:
        """模拟带已保留日志的核心失败。"""
        raise IntrinsicCalibrationError("failed", log_path)

    monkeypatch.setattr(cli, "calibrate_camera_intrinsics", fail)
    with pytest.raises(SystemExit) as error:
        cli.main(minimum_arguments())
    assert error.value.code == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload == {
        "accepted": False,
        "failure": "failed",
        "output_paths": {"log": str(log_path)},
    }
