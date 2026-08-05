"""测试 FastUMI HDF5 主结构、属性和质量数据。"""

import json

import numpy as np
import pytest

from fastumi_data.models import SynchronizedEpisode


try:
    # 采集机通过 package.xml 安装 h5py；精简开发环境可跳过该单项。
    import h5py
    from fastumi_data.hdf5_writer import (
        build_quality_report,
        write_episode_hdf5,
        write_json_report,
    )
except ModuleNotFoundError:
    h5py = None
    build_quality_report = None
    write_episode_hdf5 = None
    write_json_report = None


@pytest.mark.skipif(h5py is None, reason="当前系统 Python 未安装 h5py")
def test_writes_versioned_fastumi_hdf5(tmp_path) -> None:
    """验证输出包含训练兼容主结构和时间质量字段。"""
    episode = SynchronizedEpisode(
        timestamp_ns=np.arange(10, dtype=np.int64),
        images_rgb=np.zeros((10, 8, 8, 3), dtype=np.uint8),
        qpos=np.tile(
            np.asarray([0, 0, 0, 0, 0, 0, 1, 0.5], dtype=np.float32),
            (10, 1),
        ),
        gripper_observed=np.ones(10, dtype=np.bool_),
        tracker_tracking_ok=np.ones(10, dtype=np.bool_),
        pose_gap_ms=np.zeros(10, dtype=np.float32),
        gripper_gap_ms=np.zeros(10, dtype=np.float32),
    )
    output = tmp_path / "episode_0000.hdf5"

    write_episode_hdf5(
        str(output), episode, "task", "session", 0, 20.0, "hash"
    )

    with h5py.File(output, "r") as root:
        assert root.attrs["schema_version"] == "fastumi_ros2_v1"
        assert root.attrs["pose_frame"] == "episode_start_tcp"
        assert root["observations/qpos"].shape == (10, 8)
        assert root["action"].shape == (10, 8)
        assert root["observations/images/front"].shape == (10, 8, 8, 3)
        assert root["observations/timestamp_ns"].dtype == np.int64
        assert np.all(
            root["observations/quality/tracker_tracking_ok"][:]
        )
        assert root.attrs["tracker_time_offset_ms"] == pytest.approx(0.0)
        assert root.attrs["source_calibration_sha256"] == ""


@pytest.mark.skipif(h5py is None, reason="当前系统 Python 未安装 h5py")
def test_persists_tracker_time_offset_and_source_calibration(tmp_path) -> None:
    """验证 HDF5 与 JSON 报告记录 Tracker 时间偏移和标定来源。"""
    episode = SynchronizedEpisode(
        timestamp_ns=np.arange(10, dtype=np.int64),
        images_rgb=np.zeros((10, 8, 8, 3), dtype=np.uint8),
        qpos=np.tile(
            np.asarray([0, 0, 0, 0, 0, 0, 1, 0.5], dtype=np.float32),
            (10, 1),
        ),
        gripper_observed=np.ones(10, dtype=np.bool_),
        tracker_tracking_ok=np.ones(10, dtype=np.bool_),
        pose_gap_ms=np.zeros(10, dtype=np.float32),
        gripper_gap_ms=np.zeros(10, dtype=np.float32),
    )
    output = tmp_path / "episode_0000.hdf5"
    report_path = tmp_path / "episode_0000.json"
    tracker_time_offset_ms = 2.968089243035214
    source_calibration_sha256 = "source-hash"

    write_episode_hdf5(
        str(output),
        episode,
        "task",
        "session",
        0,
        20.0,
        "runtime-hash",
        tracker_time_offset_ms=tracker_time_offset_ms,
        source_calibration_sha256=source_calibration_sha256,
    )
    report = build_quality_report(
        episode,
        [],
        "runtime-hash",
        tracker_time_offset_ms=tracker_time_offset_ms,
        source_calibration_sha256=source_calibration_sha256,
    )
    write_json_report(str(report_path), report)

    with h5py.File(output, "r") as root:
        assert root.attrs["tracker_time_offset_ms"] == pytest.approx(
            2.968089243035214
        )
        assert root.attrs["source_calibration_sha256"] == "source-hash"
    persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted_report["tracker_time_offset_ms"] == pytest.approx(
        2.968089243035214
    )
    assert persisted_report["source_calibration_sha256"] == "source-hash"
