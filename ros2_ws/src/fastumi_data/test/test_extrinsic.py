"""验证 Tracker 到 TCP 外参 YAML 的加载与元数据兼容性。"""

import hashlib

import numpy as np
import pytest
import yaml

from fastumi_data.extrinsic import load_tracker_tcp_extrinsic


def _extrinsic_document(**overrides):
    """构造可由严格 v2 外参加载器读取的最小 YAML 文档。"""
    document = {
        "schema_version": 2,
        "accepted": True,
        "tracker_serial": "T265-1234",
        "tracker_to_tcp": {
            "translation_m": [0.1, -0.2, 0.3],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }
    document.update(overrides)
    return document


def _write_extrinsic(path, **overrides):
    """写入可由严格 v2 外参加载器读取的最小 YAML 文档。"""
    document = _extrinsic_document(**overrides)
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
    assert extrinsic.metadata["schema_version"] == 2
    assert extrinsic.metadata["accepted"] is True
    assert not hasattr(extrinsic, "calibration_verified")


def test_load_tracker_tcp_extrinsic_defaults_optional_metadata(tmp_path):
    """v2 外参缺少可选元数据时应保留既有默认值。"""
    path = tmp_path / "minimal.yaml"
    _write_extrinsic(path)

    extrinsic = load_tracker_tcp_extrinsic(str(path))

    assert extrinsic.time_offset_ms == 0.0
    assert extrinsic.source_calibration_sha256 == ""
    assert extrinsic.calibration_method == ""
    assert extrinsic.aruco_config_sha256 == ""


def test_load_tracker_tcp_extrinsic_preserves_provenance_metadata(tmp_path):
    """严格 v2 外参加载后仍应保留完整 provenance。"""
    path = tmp_path / "extrinsic.yaml"
    _write_extrinsic(
        path,
        method="dual_aruco_bootstrap",
        aruco_config_sha256="aruco-hash",
    )

    extrinsic = load_tracker_tcp_extrinsic(str(path))

    assert extrinsic.calibration_method == "dual_aruco_bootstrap"
    assert extrinsic.aruco_config_sha256 == "aruco-hash"


@pytest.mark.parametrize("accepted", [None, "true", False])
def test_extrinsic_requires_boolean_true_accepted(tmp_path, accepted):
    """外参仅接受 YAML bool true，避免未验收文件被消费。"""
    document = _extrinsic_document()
    if accepted is None:
        document.pop("accepted")
    else:
        document["accepted"] = accepted
    path = tmp_path / "invalid_accepted.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ValueError, match="accepted"):
        load_tracker_tcp_extrinsic(str(path))


def test_load_tracker_tcp_extrinsic_rejects_nonfinite_time_offset(tmp_path):
    """加载器应拒绝 NaN 等非有限毫秒时间偏移。"""
    nonfinite_path = tmp_path / "nonfinite.yaml"
    _write_extrinsic(nonfinite_path, time_offset_ms=float("nan"))

    with pytest.raises(ValueError, match="time_offset_ms"):
        load_tracker_tcp_extrinsic(str(nonfinite_path))


def test_extrinsic_rejects_schema_v1_without_legacy_verification_fields(tmp_path):
    """仅升级版本号前的 v1 外参也必须被拒绝。"""
    path = tmp_path / "schema_v1.yaml"
    _write_extrinsic(path, schema_version=1)

    with pytest.raises(ValueError, match="schema_version"):
        load_tracker_tcp_extrinsic(str(path))


@pytest.mark.parametrize("schema_version", [2.0, True])
def test_extrinsic_requires_integer_schema_version_two(tmp_path, schema_version):
    """外参 schema_version 仅接受 YAML 整数 2。"""
    path = tmp_path / "invalid_schema_version_type.yaml"
    _write_extrinsic(path, schema_version=schema_version)

    with pytest.raises(ValueError, match="schema_version"):
        load_tracker_tcp_extrinsic(str(path))


@pytest.mark.parametrize("legacy_field", ["verified", "calibration_verified"])
def test_extrinsic_rejects_removed_verification_fields(tmp_path, legacy_field):
    """严格 v2 外参不得继续携带已删除的顶层验证字段。"""
    path = tmp_path / f"legacy_{legacy_field}.yaml"
    _write_extrinsic(path, **{legacy_field: True})

    with pytest.raises(ValueError, match=legacy_field):
        load_tracker_tcp_extrinsic(str(path))
