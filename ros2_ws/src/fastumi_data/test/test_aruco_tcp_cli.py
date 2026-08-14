"""验证双 ArUco 标定 CLI 的仅图像编排、报告溯源和原子输出。"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from fastumi_data.aruco_tcp_cli import (
    DEFAULT_IMAGE_TOPIC,
    build_argument_parser,
    run_calibration,
    write_calibration_outputs,
)
from fastumi_data.aruco_tcp_config import load_aruco_tcp_config
from fastumi_data.tracker_camera_bag import ImageFrame


def _config_document():
    """返回测试使用的 bootstrap ArUco 配置。"""
    return {
        "schema_version": 2,
        "fixture_version": "dual-aruco-bootstrap-v1",
        "aruco": {
            "dictionary_name": "DICT_4X4_50",
            "marker_size_m": 0.016,
            "tag0_id": 0,
            "tag1_id": 1,
        },
        "rectification": {"projection": "reuse_kalibr_intrinsics"},
        "pair_frame": {
            "y_axis_from_tag_id": 0,
            "y_axis_to_tag_id": 1,
            "z_axis_from_marker_normals": True,
            "marker_normal_sign": -1,
        },
        "full_open_geometry": {
            "tag_center_distance_m": 0.126,
            "pair_from_tcp": {
                "translation_m": [0.012, 0.0, 0.018],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        "motion_model": {
            "type": "symmetric_parallel_linear",
            "openness_definition": "0_closed_1_open",
            "closed_tag_center_distance_m": 0.04831,
            "open_tag_center_distance_m": 0.126,
        },
    }


def _make_fixture(tmp_path: Path):
    """创建 CLI 输出测试所需的配置、输入和假结果。"""
    aruco_path = tmp_path / "aruco.yaml"
    aruco_path.write_text(
        yaml.safe_dump(_config_document(), sort_keys=False), encoding="utf-8"
    )
    camera_path = tmp_path / "camera.yaml"
    camera_path.write_text("camera: test\n", encoding="utf-8")
    tracker_camera_path = tmp_path / "tracker_camera.yaml"
    tracker_camera_path.write_text("tracker_camera: test\n", encoding="utf-8")
    tracker_path = tmp_path / "vive.yaml"
    tracker_path.write_text("serial: LHR-TEST\n", encoding="utf-8")
    bag_path = tmp_path / "bag"
    bag_path.mkdir()
    (bag_path / "bag_0.mcap").write_bytes(b"mcap")
    config = load_aruco_tcp_config(str(aruco_path))
    identity = np.eye(4, dtype=np.float64)
    result = SimpleNamespace(
        accepted=True,
        camera_from_tcp=identity,
        tracker_from_tcp=identity,
        valid_frames=tuple(),
        metrics={
            "candidate_frames": 40,
            "single_frame_valid_frames": 40,
            "valid_frames": 40,
            "rejected_frames": 0,
            "translation_p95_mm": 1.2,
            "rotation_p95_deg": 0.8,
        },
        failures=tuple(),
        rejection_histogram={},
    )
    return (
        aruco_path,
        camera_path,
        tracker_camera_path,
        tracker_path,
        bag_path,
        config,
        result,
    )


def test_parser_contains_fixed_calibration_pipeline_arguments():
    """CLI parser 应暴露必需参数并拒绝已移除的绕过参数。"""
    parser = build_argument_parser()
    valid_arguments = [
        "bag",
        "--camera-config",
        "camera.yaml",
        "--aruco-config",
        "aruco.yaml",
        "--tracker-camera-calibration",
        "tracker-camera.yaml",
        "--tracker-config",
        "vive.yaml",
        "--output-dir",
        "derived",
        "--image-topic",
        "/image",
        "--frame-stride",
        "2",
        "--minimum-frames",
        "30",
        "--max-reprojection-rmse-px",
        "1.5",
        "--max-distance-error-mm",
        "5.0",
        "--max-candidate-difference-mm",
        "5.0",
        "--max-translation-p95-mm",
        "3.0",
        "--max-rotation-p95-deg",
        "2.0",
        "--force",
    ]
    arguments = parser.parse_args(valid_arguments)

    assert arguments.bag_uri == "bag"
    assert arguments.image_topic == "/image"
    assert arguments.minimum_frames == 30
    assert arguments.force is True
    for removed_name in (
        "tracker_topic",
        "status_topic",
        "gripper_topic",
        "max_pose_gap_ms",
        "max_gripper_gap_ms",
    ):
        assert not hasattr(arguments, removed_name)


def test_parser_defaults_to_tof_camera_calibration() -> None:
    """省略相机配置时应读取 ToF 包内默认标定。"""
    arguments = build_argument_parser().parse_args(
        [
            "bag",
            "--aruco-config",
            "aruco.yaml",
            "--tracker-camera-calibration",
            "tracker-camera.yaml",
            "--tracker-config",
            "vive.yaml",
            "--output-dir",
            "derived",
        ]
    )
    camera_path = Path(arguments.camera_config)
    assert camera_path.name == "calibration.yaml"
    assert camera_path.parent.name == "config"
    assert camera_path.parent.parent.name == "tof_stereo_camera"


@pytest.mark.parametrize(
    "removed_argument",
    [
        "--tracker-topic",
        "--status-topic",
        "--gripper-topic",
        "--max-pose-gap-ms",
        "--max-gripper-gap-ms",
    ],
)
def test_parser_rejects_removed_timeline_arguments(removed_argument, capsys):
    """仅图像 CLI 必须拒绝旧时间线参数。"""
    parser = build_argument_parser()
    valid_arguments = [
        "bag",
        "--camera-config",
        "camera.yaml",
        "--aruco-config",
        "aruco.yaml",
        "--tracker-camera-calibration",
        "tracker-camera.yaml",
        "--tracker-config",
        "vive.yaml",
        "--output-dir",
        "derived",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args([*valid_arguments, removed_argument, "unused"])
    stderr = capsys.readouterr().err
    assert "unrecognized arguments" in stderr
    assert removed_argument in stderr


def test_output_persists_v2_accepted_extrinsic_and_at_least_five_overlays(
    tmp_path,
):
    """成功输出应写入 v2 已接受外参、报告、CSV 和 bootstrap 溯源。"""
    (
        aruco_path,
        camera_path,
        tracker_camera_path,
        tracker_path,
        bag_path,
        config,
        result,
    ) = _make_fixture(tmp_path)
    output_dir = tmp_path / "derived"

    paths = write_calibration_outputs(
        output_dir,
        result=result,
        aruco_config=config,
        aruco_config_path=aruco_path,
        tracker_serial="LHR-TEST",
        time_offset_ms=2.968,
        bag_uri=bag_path,
        camera_config_path=camera_path,
        tracker_camera_calibration_path=tracker_camera_path,
        tracker_config_path=tracker_path,
        frame_metrics=[
            {
                "timestamp_ns": 1,
                "openness": 1.0,
                "tag0_rmse_px": 0.4,
                "tag1_rmse_px": 0.3,
                "measured_distance_m": 0.126,
                "expected_distance_m": 0.126,
                "candidate_difference_m": 0.001,
                "accepted": True,
            }
        ],
        overlay_images=[np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)],
        force=False,
    )

    tracker_document = yaml.safe_load(
        paths["tracker_to_tcp"].read_text(encoding="utf-8")
    )
    standalone_path = paths["tracker_to_tcp_transform"]
    standalone_document = yaml.safe_load(
        standalone_path.read_text(encoding="utf-8")
    )
    summary = yaml.safe_load(paths["summary"].read_text(encoding="utf-8"))
    assert tracker_document["schema_version"] == 2
    assert tracker_document["accepted"] is True
    assert "verified" not in tracker_document
    assert "calibration_verified" not in tracker_document
    assert tracker_document["method"] == "dual_aruco_bootstrap"
    assert tracker_document["fixture_version"] == "dual-aruco-bootstrap-v1"
    assert tracker_document["time_offset_ms"] == pytest.approx(2.968)
    assert tracker_document["source_calibration"]["sha256"]
    assert tracker_document["aruco_config_sha256"] == config.source_sha256
    assert tracker_document["tracker_from_camera"]["maps_from"] == "camera"
    assert tracker_document["camera_from_tcp"]["maps_to"] == "camera"
    assert tracker_document["tracker_to_tcp"]["maps_to"] == "tracker"
    assert standalone_path == output_dir / "tracker_to_tcp_transform.yaml"
    assert set(standalone_document) == {"tracker_to_tcp"}
    standalone_transform = standalone_document["tracker_to_tcp"]
    assert set(standalone_transform) == {
        "maps_from",
        "maps_to",
        "matrix",
        "translation_m",
        "quaternion_xyzw",
    }
    assert standalone_transform["maps_from"] == "tcp"
    assert standalone_transform["maps_to"] == "tracker"
    np.testing.assert_allclose(
        standalone_transform["matrix"],
        tracker_document["tracker_to_tcp"]["matrix"],
    )
    np.testing.assert_allclose(
        standalone_transform["translation_m"],
        tracker_document["tracker_to_tcp"]["translation_m"],
    )
    np.testing.assert_allclose(
        standalone_transform["quaternion_xyzw"],
        tracker_document["tracker_to_tcp"]["quaternion_xyzw"],
    )
    assert summary["outputs"]["tracker_to_tcp_transform"] == (
        "tracker_to_tcp_transform.yaml"
    )
    assert "calibration_verified" not in summary
    assert summary["calibration_method"] == "dual_aruco_bootstrap"
    assert summary["calibration_assumptions"] == {"fixed_openness": 1.0}
    assert paths["aruco_snapshot"].exists()
    assert paths["frame_metrics"].exists()
    assert len(list(paths["calibration_report"].glob("overlay_*.png"))) == 5


def test_rejected_calibration_omits_standalone_transform(tmp_path):
    """质量门拒绝时不得创建或宣传根目录独立外参。"""
    (
        aruco_path,
        camera_path,
        tracker_camera_path,
        tracker_path,
        bag_path,
        config,
        result,
    ) = _make_fixture(tmp_path)
    result.accepted = False
    output_dir = tmp_path / "rejected"

    paths = write_calibration_outputs(
        output_dir,
        result=result,
        aruco_config=config,
        aruco_config_path=aruco_path,
        tracker_serial="LHR-TEST",
        time_offset_ms=2.968,
        bag_uri=bag_path,
        camera_config_path=camera_path,
        tracker_camera_calibration_path=tracker_camera_path,
        tracker_config_path=tracker_path,
        frame_metrics=[],
        overlay_images=[],
        force=False,
    )

    summary = yaml.safe_load(paths["summary"].read_text(encoding="utf-8"))
    assert "tracker_to_tcp_transform" not in paths
    assert "tracker_to_tcp_transform" not in summary["outputs"]
    assert not (output_dir / "tracker_to_tcp_transform.yaml").exists()


def test_existing_output_requires_force(tmp_path):
    """已有输出目录未显式 force 时必须拒绝覆盖。"""
    (
        aruco_path,
        camera_path,
        tracker_camera_path,
        tracker_path,
        bag_path,
        config,
        result,
    ) = _make_fixture(tmp_path)
    output_dir = tmp_path / "derived"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="force"):
        write_calibration_outputs(
            output_dir,
            result=result,
            aruco_config=config,
            aruco_config_path=aruco_path,
            tracker_serial="LHR-TEST",
            time_offset_ms=0.0,
            bag_uri=bag_path,
            camera_config_path=camera_path,
            tracker_camera_calibration_path=tracker_camera_path,
            tracker_config_path=tracker_path,
            frame_metrics=[],
            overlay_images=[],
            force=False,
        )


def test_run_calibration_injects_fixed_openness_into_each_image_frame(
    monkeypatch,
):
    """图像编排必须给每帧统一注入全开开度并写入同值诊断。"""
    image_frames = [
        ImageFrame(101, 201, np.zeros((4, 4, 3), dtype=np.uint8)),
        ImageFrame(102, 202, np.zeros((4, 4, 3), dtype=np.uint8)),
    ]
    observed_openness = []
    captured = {}

    def estimate(image, openness, timestamp_ns):
        """记录估计输入并返回可供真实指标转换的帧对象。"""
        observed_openness.append(openness)
        tag = SimpleNamespace(corners_px=None, tag_id=0)
        return SimpleNamespace(
            timestamp_ns=timestamp_ns,
            tag0_reprojection_rmse_px=0.4,
            tag1_reprojection_rmse_px=0.3,
            measured_tag_distance_m=0.126,
            expected_tag_distance_m=0.126,
            candidate_translation_difference_m=0.001,
            tag0=tag,
            tag1=tag,
        )

    class FakeEstimator:
        """提供无需 ArUco 检测器的确定性单帧估计替身。"""

        def __init__(self, camera, aruco_config):
            """保存诊断绘制所需的占位去畸变映射。"""
            self.rectification_maps = (None, None)

        def estimate(self, image, openness, timestamp_ns):
            """委托给测试记录器，模拟单帧检测完成。"""
            return estimate(image, openness, timestamp_ns)

    def capture_outputs(output_dir, **kwargs):
        """捕获 CLI 即将持久化的真实逐帧诊断行。"""
        captured["frame_metrics"] = kwargs["frame_metrics"]
        return {}

    result = SimpleNamespace(accepted=True)
    monkeypatch.setattr(
        "fastumi_data.aruco_tcp_cli.load_kalibr_camera", lambda path: object()
    )
    monkeypatch.setattr(
        "fastumi_data.aruco_tcp_cli.load_aruco_tcp_config", lambda path: object()
    )
    monkeypatch.setattr(
        "fastumi_data.aruco_tcp_cli._load_tracker_camera_calibration",
        lambda path: (np.eye(4), 2.968, "source-sha"),
    )
    monkeypatch.setattr(
        "fastumi_data.aruco_tcp_cli._load_tracker_config_serial",
        lambda path: "LHR-TEST",
    )
    monkeypatch.setattr(
        "fastumi_data.aruco_tcp_cli.iter_image_frames",
        lambda *args, **kwargs: iter(image_frames),
    )
    monkeypatch.setattr(
        "fastumi_data.aruco_tcp_cli.DualArucoTcpEstimator", FakeEstimator
    )
    monkeypatch.setattr(
        "fastumi_data.aruco_tcp_cli.calibrate_frames",
        lambda *args: result,
    )
    monkeypatch.setattr(
        "fastumi_data.aruco_tcp_cli.cv2.remap", lambda image, *args, **kwargs: image
    )
    monkeypatch.setattr(
        "fastumi_data.aruco_tcp_cli._overlay_image", lambda image, frame: image
    )
    monkeypatch.setattr(
        "fastumi_data.aruco_tcp_cli.write_calibration_outputs", capture_outputs
    )
    arguments = argparse.Namespace(
        bag_uri="bag",
        camera_config="camera.yaml",
        aruco_config="aruco.yaml",
        tracker_camera_calibration="tracker-camera.yaml",
        tracker_config="vive.yaml",
        output_dir="derived",
        image_topic="/image",
        frame_stride=1,
        minimum_frames=30,
        max_reprojection_rmse_px=1.5,
        max_distance_error_mm=5.0,
        max_candidate_difference_mm=5.0,
        max_translation_p95_mm=3.0,
        max_rotation_p95_deg=2.0,
        force=False,
    )

    assert run_calibration(arguments) is result
    assert observed_openness == [1.0, 1.0]
    assert [row["openness"] for row in captured["frame_metrics"]] == [1.0, 1.0]


def test_default_image_topic_uses_tof_rgb() -> None:
    """双 ArUco 标定默认读取 FastUMI ToF RGB 图像。"""
    assert DEFAULT_IMAGE_TOPIC == "/tof_stereo_camera/rgb/image_raw"
