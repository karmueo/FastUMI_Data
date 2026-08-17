"""验证独立 Kalibr 内参适配器的命令、日志、产物和流水线。"""

from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import fastumi_camera_calibration.calibration as calibration
from fastumi_camera_calibration.calibration import (
    IntrinsicCalibrationError,
    KalibrArtifacts,
    build_kalibr_command,
    calibrate_camera_intrinsics,
    publish_kalibr_artifacts,
    run_kalibr,
    validate_kalibr_artifacts,
)


def _write_valid_artifacts(path: Path) -> Path:
    """写入符合固定单鱼眼模型的 Kalibr 测试产物。"""
    # YAML 文件模拟 Kalibr camchain 输出。
    yaml_path = path / "camchain-run.yaml"
    yaml_path.write_text(
        "cam0:\n"
        "  camera_model: pinhole\n"
        "  distortion_model: equidistant\n"
        "  intrinsics: [400.0, 401.0, 320.0, 240.0]\n"
        "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        "  resolution: [640, 480]\n"
        "  rostopic: /camera/image\n",
        encoding="utf-8",
    )
    (path / "results-cam-run.txt").write_text(
        "results\n", encoding="utf-8"
    )
    (path / "report-cam-run.pdf").write_bytes(b"%PDF")
    (path / "kalibr.log").write_text("exit_status: 0\n", encoding="utf-8")
    return yaml_path


def test_build_kalibr_command_uses_fixed_single_fisheye_model() -> None:
    """Kalibr 命令应固定单相机 pinhole-equi 和非交互参数。"""
    assert build_kalibr_command(
        "/tmp/images", "/camera/image", "target.yaml"
    ) == [
        "ros2", "run", "kalibr_imu_camera", "kalibr_calibrate_cameras",
        "--bag", "/tmp/images", "--topics", "/camera/image",
        "--models", "pinhole-equi", "--target", "target.yaml",
        "--verbose", "--no-shuffle", "--dont-show-report",
    ]


def test_resolve_kalibr_package_prefix_reports_missing_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺少 Kalibr overlay 时应提供新包资源路径和 source 顺序。"""
    def missing(name: str) -> str:
        """模拟 ament 未找到 Kalibr 包。"""
        raise calibration.PackageNotFoundError(name)

    monkeypatch.setattr(calibration, "get_package_prefix", missing)
    with pytest.raises(
        IntrinsicCalibrationError,
        match="fastumi_camera_calibration/vendor.*Jazzy",
    ):
        calibration.resolve_kalibr_package_prefix()


def test_run_kalibr_records_environment_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """成功和失败运行都应记录参数、Agg 后端与退出状态。"""
    # 捕获 subprocess.run 关键字参数。
    captured = {}

    def fake_run(
        *args: object, **kwargs: object
    ) -> subprocess.CompletedProcess:
        """返回无输出的成功子进程。"""
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, "")

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)
    # 日志路径位于临时工作目录。
    log_path = tmp_path / "kalibr.log"
    run_kalibr(["kalibr", "--help"], tmp_path, log_path)
    assert captured["env"]["MPLBACKEND"] == "Agg"
    assert captured["env"]["QT_QPA_PLATFORM"] == "offscreen"
    assert "argv: kalibr --help" in log_path.read_text(encoding="utf-8")
    assert "exit_status: 0" in log_path.read_text(encoding="utf-8")

    monkeypatch.setattr(
        calibration.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 9, "failed"
        ),
    )
    with pytest.raises(IntrinsicCalibrationError, match="非零退出码 9"):
        run_kalibr(["kalibr"], tmp_path, log_path)
    assert "exit_status: 9" in log_path.read_text(encoding="utf-8")


def test_validate_kalibr_artifacts_enforces_fixed_contract(
    tmp_path: Path,
) -> None:
    """完整且可加载的单 cam0 产物应通过严格校验。"""
    yaml_path = _write_valid_artifacts(tmp_path)
    artifacts = validate_kalibr_artifacts(
        tmp_path, tmp_path / "kalibr.log", "/camera/image", (640, 480)
    )
    assert artifacts.yaml_path == yaml_path

    yaml_path.write_text(
        yaml_path.read_text(encoding="utf-8").replace(
            "distortion_model: equidistant",
            "distortion_model: radtan",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pinhole/equidistant"):
        validate_kalibr_artifacts(
            tmp_path, tmp_path / "kalibr.log", "/camera/image", (640, 480)
        )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("rostopic: /camera/image", "rostopic: /other", "rostopic"),
        ("resolution: [640, 480]", "resolution: [320, 240]", "分辨率"),
        (
            "intrinsics: [400.0, 401.0, 320.0, 240.0]",
            "intrinsics: [.nan, 401.0, 320.0, 240.0]",
            "四个有限数值",
        ),
    ],
)
def test_validate_kalibr_artifacts_rejects_invalid_yaml_values(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    """错误话题、分辨率和非有限内参必须拒绝发布。"""
    yaml_path = _write_valid_artifacts(tmp_path)
    yaml_path.write_text(
        yaml_path.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        validate_kalibr_artifacts(
            tmp_path, tmp_path / "kalibr.log", "/camera/image", (640, 480)
        )


def test_publish_kalibr_artifacts_keeps_compatible_names(
    tmp_path: Path,
) -> None:
    """最终四类产物名称和 YAML 最终路径应保持兼容。"""
    # staging 目录保存模拟 Kalibr 输出。
    staging = tmp_path / "staging"
    staging.mkdir()
    _write_valid_artifacts(staging)
    artifacts = KalibrArtifacts(
        staging / "camchain-run.yaml",
        staging / "results-cam-run.txt",
        staging / "report-cam-run.pdf",
        staging / "kalibr.log",
    )
    # 输出目录模拟用户指定目标。
    output = tmp_path / "output"
    published = publish_kalibr_artifacts(artifacts, output)
    assert published.yaml_path == output / "camera_intrinsics.yaml"
    assert published.results_path.name == "camera_intrinsics_results.txt"
    assert published.report_path.name == "camera_intrinsics_report.pdf"
    assert published.log_path.name == "camera_intrinsics.log"


def test_calibration_pipeline_publishes_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """完整流水线应发布四类文件并记录固定版本和抽帧数量。"""
    # 目标板文件必须能绝对化。
    target = tmp_path / "target.yaml"
    target.write_text("target_type: aprilgrid\n", encoding="utf-8")
    monkeypatch.setattr(
        calibration, "resolve_kalibr_package_prefix", lambda: "/overlay"
    )
    # 补丁资源用临时文件替代，避免测试依赖源码布局。
    patch_path = tmp_path / "compat.patch"
    patch_path.write_text("patch\n", encoding="utf-8")
    monkeypatch.setattr(
        calibration, "kalibr_compatibility_patch_path", lambda: patch_path
    )

    def fake_extract(*args: object, **kwargs: object) -> SimpleNamespace:
        """返回固定临时 bag 元数据。"""
        # bag URI 只用于构造 Kalibr 参数。
        return SimpleNamespace(
            bag_uri=tmp_path / "images",
            resolution=(640, 480),
            frame_count=12,
        )

    monkeypatch.setattr(
        calibration, "extract_image_topic_to_sqlite3", fake_extract
    )

    def fake_run(
        command: list[str], staging_dir: Path, log_path: Path
    ) -> None:
        """在 staging 中写入模拟 Kalibr 产物。"""
        _write_valid_artifacts(Path(staging_dir))

    monkeypatch.setattr(calibration, "run_kalibr", fake_run)
    # 输出目录接收经过校验的最终文件。
    output = tmp_path / "output"
    result = calibrate_camera_intrinsics(
        "source", "/camera/image", target, output
    )
    assert result.provenance["source"] == "kalibr_ros2"
    assert result.provenance["selected_frame_count"] == 12
    assert result.provenance["expected_kalibr_commit"]
    assert result.artifacts.yaml_path.is_file()


def test_calibration_failure_preserves_current_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kalibr 非零失败应使旧 YAML 失效并保留本次日志。"""
    # 创建旧成功标记验证新任务开始时会使其失效。
    output = tmp_path / "output"
    output.mkdir()
    (output / "camera_intrinsics.yaml").write_text(
        "stale\n", encoding="utf-8"
    )
    target = tmp_path / "target.yaml"
    target.write_text("target_type: aprilgrid\n", encoding="utf-8")
    patch_path = tmp_path / "compat.patch"
    patch_path.write_text("patch\n", encoding="utf-8")
    monkeypatch.setattr(
        calibration, "resolve_kalibr_package_prefix", lambda: "/overlay"
    )
    monkeypatch.setattr(
        calibration, "kalibr_compatibility_patch_path", lambda: patch_path
    )
    monkeypatch.setattr(
        calibration,
        "extract_image_topic_to_sqlite3",
        lambda *args, **kwargs: SimpleNamespace(
            bag_uri=tmp_path / "images",
            resolution=(640, 480),
            frame_count=2,
        ),
    )

    def fail_run(
        command: list[str], staging_dir: Path, log_path: Path
    ) -> None:
        """写入本次失败日志并抛出公共异常。"""
        Path(log_path).write_text(
            "exit_status: 9\n", encoding="utf-8"
        )
        raise IntrinsicCalibrationError(
            "Kalibr 返回非零退出码 9", Path(log_path)
        )

    monkeypatch.setattr(calibration, "run_kalibr", fail_run)
    with pytest.raises(IntrinsicCalibrationError) as error:
        calibrate_camera_intrinsics(
            "source", "/camera/image", target, output
        )
    assert error.value.log_path == output / "camera_intrinsics.log"
    assert "exit_status: 9" in error.value.log_path.read_text(
        encoding="utf-8"
    )
    assert not (output / "camera_intrinsics.yaml").exists()
