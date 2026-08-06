"""验证双 ArUco 到 TCP 配置的严格校验和确定性运动模型。"""

import hashlib

import numpy as np
import pytest
import yaml

from fastumi_data.aruco_tcp_config import load_aruco_tcp_config


def _document(**overrides):
    """构造包含全部确认几何值的最小配置文档。"""
    document = {
        "schema_version": 2,
        "fixture_version": "dual-aruco-bootstrap-v2",
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
    for key, value in overrides.items():
        document[key] = value
    return document


def _write_config(path, document=None):
    """把配置文档写入临时 YAML 文件。"""
    path.write_text(
        yaml.safe_dump(document or _document(), sort_keys=False),
        encoding="utf-8",
    )


def test_load_v2_config_preserves_sha256_without_verification_field(tmp_path):
    """v2 配置无需验证绕过参数，且应保留源文件哈希。"""
    path = tmp_path / "aruco_to_tcp.yaml"
    _write_config(path)

    config = load_aruco_tcp_config(str(path))

    assert config.schema_version == 2
    assert not hasattr(config, "verified")
    assert config.source_sha256 == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    assert config.aruco.dictionary_name == "DICT_4X4_50"
    assert config.aruco.marker_size_m == pytest.approx(0.016)
    assert config.pair_frame.marker_normal_sign == -1


def test_config_exposes_immutable_pair_from_tcp_transform(tmp_path):
    """pair→TCP 变换应是只读的齐次矩阵并保留米制偏移。"""
    path = tmp_path / "aruco_to_tcp.yaml"
    _write_config(path)

    config = load_aruco_tcp_config(str(path))

    np.testing.assert_allclose(
        config.pair_from_tcp,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.012],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.018],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert config.pair_from_tcp.flags.writeable is False


@pytest.mark.parametrize(
    ("openness", "expected"),
    [
        (0.0, 0.024155),
        (1.0, 0.063),
        (0.5, 0.0435775),
    ],
)
def test_expected_half_distance_follows_linear_motion_model(
    tmp_path, openness, expected
):
    """开度 0、1 及中点应按确认的对称线性模型计算。"""
    path = tmp_path / "aruco_to_tcp.yaml"
    _write_config(path)
    config = load_aruco_tcp_config(str(path))

    assert config.expected_half_distance_m(openness) == pytest.approx(expected)


@pytest.mark.parametrize("openness", [-0.01, 1.01, float("nan")])
def test_expected_half_distance_rejects_invalid_openness(tmp_path, openness):
    """开度必须是 [0, 1] 内的有限数值。"""
    path = tmp_path / "aruco_to_tcp.yaml"
    _write_config(path)
    config = load_aruco_tcp_config(str(path))

    with pytest.raises(ValueError, match="openness"):
        config.expected_half_distance_m(openness)


@pytest.mark.parametrize(
    ("path_key", "bad_value", "message"),
    [
        (("aruco", "tag1_id"), 0, "ID"),
        (("aruco", "dictionary_name"), "UNKNOWN", "字典"),
        (("pair_frame", "marker_normal_sign"), 0, "符号"),
        (("rectification", "projection"), "estimate_new_camera_matrix", "投影"),
        (("motion_model", "closed_tag_center_distance_m"), 0.126, "闭合"),
    ],
)
def test_loader_rejects_unsupported_or_inconsistent_values(
    tmp_path, path_key, bad_value, message
):
    """加载器应拒绝计划列出的字典、ID、投影和几何不一致。"""
    document = _document()
    document[path_key[0]][path_key[1]] = bad_value
    path = tmp_path / "invalid.yaml"
    _write_config(path, document)

    with pytest.raises(ValueError, match=message):
        load_aruco_tcp_config(str(path))


def test_loader_rejects_non_unit_pair_quaternion_and_mismatched_open_distance(
    tmp_path,
):
    """pair 四元数必须单位化，full-open 距离必须与运动模型一致。"""
    document = _document()
    document["full_open_geometry"]["pair_from_tcp"]["quaternion_xyzw"] = [
        0.0,
        0.0,
        0.0,
        2.0,
    ]
    path = tmp_path / "invalid_quaternion.yaml"
    _write_config(path, document)
    with pytest.raises(ValueError, match="单位四元数"):
        load_aruco_tcp_config(str(path))

    document = _document()
    document["motion_model"]["open_tag_center_distance_m"] = 0.125
    path = tmp_path / "invalid_distance.yaml"
    _write_config(path, document)
    with pytest.raises(ValueError, match="全开"):
        load_aruco_tcp_config(str(path))


def test_loader_rejects_schema_v1_without_legacy_verification_fields(tmp_path):
    """仅升级版本号前的 v1 文档也必须被拒绝。"""
    document = _document(schema_version=1)
    path = tmp_path / "schema_v1.yaml"
    _write_config(path, document)

    with pytest.raises(ValueError, match="schema_version"):
        load_aruco_tcp_config(str(path))


@pytest.mark.parametrize("legacy_field", ["verified", "calibration_verified"])
def test_loader_rejects_removed_verification_fields(tmp_path, legacy_field):
    """v2 配置不得继续携带已删除的顶层验证字段。"""
    document = _document(**{legacy_field: True})
    path = tmp_path / f"legacy_{legacy_field}.yaml"
    _write_config(path, document)

    with pytest.raises(ValueError, match=legacy_field):
        load_aruco_tcp_config(str(path))
