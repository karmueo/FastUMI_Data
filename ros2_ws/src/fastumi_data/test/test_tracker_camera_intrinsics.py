"""验证自动 Kalibr 内参适配器的命令、包发现、日志和产物校验。"""

from pathlib import Path
import subprocess

import pytest

import fastumi_data.tracker_camera_intrinsics as intrinsics
from fastumi_data.tracker_camera_intrinsics import (
    IntrinsicCalibrationError,
    build_kalibr_command,
    run_kalibr,
    validate_kalibr_artifacts,
)


def test_build_kalibr_command_uses_required_fixed_arguments() -> None:
    """Kalibr 调用应固定为单相机 pinhole-equi、非交互和无 shuffle。"""
    assert build_kalibr_command("/tmp/images", "/camera/image", "target.yaml") == [
        "ros2", "run", "kalibr_imu_camera", "kalibr_calibrate_cameras",
        "--bag", "/tmp/images", "--topics", "/camera/image",
        "--models", "pinhole-equi", "--target", "target.yaml",
        "--verbose", "--no-shuffle", "--dont-show-report",
    ]


def test_resolve_kalibr_package_prefix_success_and_missing(monkeypatch) -> None:
    """包前缀应使用 ament 查询，缺失错误必须给出 overlay 设置说明。"""
    monkeypatch.setattr(intrinsics, "get_package_prefix", lambda name: "/overlay")
    assert intrinsics.resolve_kalibr_package_prefix() == "/overlay"
    def missing(name):
        raise intrinsics.PackageNotFoundError(name)
    monkeypatch.setattr(intrinsics, "get_package_prefix", missing)
    with pytest.raises(IntrinsicCalibrationError, match="kalibr_ros2.repos.*Jazzy"):
        intrinsics.resolve_kalibr_package_prefix()


def test_run_kalibr_logs_argv_status_and_agg(tmp_path: Path, monkeypatch) -> None:
    """即使 Kalibr 无 stdout，日志仍应包含命令和退出状态。"""
    captured = {}
    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, "")
    monkeypatch.setattr(intrinsics.subprocess, "run", fake_run)
    log = tmp_path / "kalibr.log"
    run_kalibr(["kalibr", "--help"], tmp_path, log)
    assert captured["env"]["MPLBACKEND"] == "Agg"
    assert "argv: kalibr --help" in log.read_text(encoding="utf-8")
    assert "exit_status: 0" in log.read_text(encoding="utf-8")


def test_run_kalibr_nonzero_preserves_log(tmp_path: Path, monkeypatch) -> None:
    """非零退出码应成为结构化失败并保留完整日志。"""
    monkeypatch.setattr(
        intrinsics.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 9, "failed"),
    )
    log = tmp_path / "kalibr.log"
    with pytest.raises(IntrinsicCalibrationError, match="非零退出码 9"):
        run_kalibr(["kalibr"], tmp_path, log)
    assert "exit_status: 9" in log.read_text(encoding="utf-8")


def test_validate_artifacts_accepts_loadable_single_camera_yaml(tmp_path: Path) -> None:
    """可由既有加载器读取的单 cam0 输出应通过严格校验。"""
    (tmp_path / "camchain-run.yaml").write_text(
        "cam0:\n  camera_model: pinhole\n  distortion_model: equidistant\n"
        "  intrinsics: [400.0, 401.0, 320.0, 240.0]\n"
        "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        "  resolution: [640, 480]\n  rostopic: /camera/image\n",
        encoding="utf-8",
    )
    (tmp_path / "results-cam-run.txt").write_text("result\n", encoding="utf-8")
    (tmp_path / "report-cam-run.pdf").write_bytes(b"%PDF")
    log = tmp_path / "kalibr.log"
    log.write_text("log\n", encoding="utf-8")
    artifacts = validate_kalibr_artifacts(tmp_path, log, "/camera/image", (640, 480))
    assert artifacts.yaml_path.name == "camchain-run.yaml"


def test_validate_artifacts_rejects_missing_outputs(tmp_path: Path) -> None:
    """成功退出后缺少预期输出仍必须拒绝发布。"""
    log = tmp_path / "kalibr.log"
    log.write_text("log\n", encoding="utf-8")
    with pytest.raises(ValueError, match="camchain"):
        validate_kalibr_artifacts(tmp_path, log, "/camera/image", (640, 480))


def test_intrinsic_error_always_exposes_log_path(tmp_path: Path) -> None:
    """所有结构化自动内参异常都必须暴露可选日志路径。"""
    log = tmp_path / "kalibr.log"
    assert intrinsics.IntrinsicCalibrationError("failed").log_path is None
    assert intrinsics.IntrinsicCalibrationError("failed", log).log_path == log


def test_publish_missing_log_does_not_return_historical_destination(tmp_path: Path) -> None:
    """缺少本次日志时发布函数不能伪造或返回旧日志路径。"""
    destination = tmp_path / "out"
    destination.mkdir()
    old_log = destination / "camera_intrinsics.log"
    old_log.write_text("old\n", encoding="utf-8")
    assert intrinsics.publish_kalibr_log(tmp_path / "missing.log", destination) is None
    assert old_log.read_text(encoding="utf-8") == "old\n"
