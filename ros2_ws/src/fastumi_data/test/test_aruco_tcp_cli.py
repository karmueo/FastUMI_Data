"""验证双 ArUco 标定 CLI 参数、报告溯源和原子输出。"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from fastumi_data.aruco_tcp_cli import (
    build_argument_parser,
    write_calibration_outputs,
)
from fastumi_data.aruco_tcp_config import load_aruco_tcp_config


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
    """CLI parser 应同时暴露四类话题、配置、质量和安全参数。"""
    parser = build_argument_parser()
    arguments = parser.parse_args(
        [
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
            "--tracker-topic",
            "/pose",
            "--status-topic",
            "/status",
            "--gripper-topic",
            "/gripper",
            "--frame-stride",
            "2",
            "--max-pose-gap-ms",
            "30",
            "--max-gripper-gap-ms",
            "100",
            "--minimum-frames",
            "30",
            "--force",
        ]
    )

    assert arguments.bag_uri == "bag"
    assert arguments.minimum_frames == 30
    assert arguments.force is True
    assert arguments.gripper_topic == "/gripper"
    with pytest.raises(SystemExit):
        parser.parse_args(["bag", "--allow-unverified"])


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
                "openness": 0.5,
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
    assert "calibration_verified" not in summary
    assert summary["calibration_method"] == "dual_aruco_bootstrap"
    assert paths["aruco_snapshot"].exists()
    assert paths["frame_metrics"].exists()
    assert len(list(paths["calibration_report"].glob("overlay_*.png"))) == 5


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
