"""验证 Tracker 到 TCP 外参 YAML 的加载与元数据兼容性。"""

import hashlib

import numpy as np
import pytest
import yaml

from fastumi_data.extrinsic import load_tracker_tcp_extrinsic


def _write_extrinsic(path, **overrides):
    """写入可由外参加载器读取的最小 YAML 文档。"""
    document = {
        "schema_version": 1,
        "tracker_serial": "T265-1234",
        "tracker_to_tcp": {
            "translation_m": [0.1, -0.2, 0.3],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }
    document.update(overrides)
    path.write_text(yaml.safe_dump(document), encoding="utf-8")


def test_load_tracker_tcp_extrinsic_reads_calibration_metadata(tmp_path):
    """加载器应返回时间偏移、来源哈希及既有外参信息。"""
    path = tmp_path / "extrinsic.yaml"
    _write_extrinsic(
        path,
        time_offset_ms=2.968089243035214,
        source_calibration={"sha256": "source-hash"},
    )

    extrinsic = load_tracker_tcp_extrinsic(str(path))

    assert extrinsic.time_offset_ms == pytest.approx(2.968089243035214)
    assert extrinsic.source_calibration_sha256 == "source-hash"
    np.testing.assert_allclose(extrinsic.matrix[:3, 3], [0.1, -0.2, 0.3])
    assert extrinsic.tracker_serial == "T265-1234"
    assert extrinsic.calibration_hash == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    assert extrinsic.metadata["schema_version"] == 1


def test_load_tracker_tcp_extrinsic_defaults_legacy_metadata(tmp_path):
    """未包含新元数据的 schema 1 文件应保持零偏移兼容行为。"""
    legacy_path = tmp_path / "legacy.yaml"
    _write_extrinsic(legacy_path)

    legacy = load_tracker_tcp_extrinsic(str(legacy_path))

    assert legacy.time_offset_ms == 0.0
    assert legacy.source_calibration_sha256 == ""


def test_load_tracker_tcp_extrinsic_rejects_nonfinite_time_offset(tmp_path):
    """加载器应拒绝 NaN 等非有限毫秒时间偏移。"""
    nonfinite_path = tmp_path / "nonfinite.yaml"
    _write_extrinsic(nonfinite_path, time_offset_ms=float("nan"))

    with pytest.raises(ValueError, match="time_offset_ms"):
        load_tracker_tcp_extrinsic(str(nonfinite_path))
