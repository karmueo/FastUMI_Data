"""测试 ToF 与 Kalibr 相机标定文件加载和夹爪毫米范围参数校验。"""

from pathlib import Path

from fastumi_gripper_estimator.calibration import (
    load_camera_calibration,
    validate_gripper_distance_range,
)
import numpy as np
import pytest
import yaml


# 仓库内真实 ToF 标定文件，用于验证允许额外顶层字段。
TOF_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "tof_stereo_camera"
    / "config"
    / "calibration.yaml"
)


def write_yaml(path: Path, document) -> None:
    """写入测试 YAML。"""
    path.write_text(
        yaml.safe_dump(document, allow_unicode=True), encoding="utf-8"
    )


def valid_legacy_document():
    """生成有效的 Kalibr cam0 测试数据。"""
    return {
        "cam0": {
            "camera_model": "pinhole",
            "distortion_model": "equidistant",
            "intrinsics": [400.0, 401.0, 320.0, 240.0],
            "distortion_coeffs": [0.08, -0.017, 0.0046, -0.0031],
            "resolution": [640, 480],
        }
    }


def valid_tof_document():
    """生成有效的 ToF rgb 测试数据。"""
    return {
        "rgb": {
            "distortion_model": "fisheye",
            "intrinsics": [405.0, 406.0, 1024.0, 768.0],
            "distortion_coeffs": [0.08, -0.007, -0.006, 0.0004],
            "resolution": [2048, 1536],
        },
        "itof": {"unrelated": True},
        "T_rgb_itof": [[1.0]],
    }


@pytest.mark.parametrize(
    ("document_factory", "expected_resolution"),
    [(valid_legacy_document, (640, 480)), (valid_tof_document, (2048, 1536))],
)
def test_load_supported_camera_calibration(
    tmp_path: Path, document_factory, expected_resolution
) -> None:
    """验证两种支持的标定模式均被转换为矩阵和鱼眼系数。"""
    calibration_path = tmp_path / "camera.yaml"
    write_yaml(calibration_path, document_factory())

    calibration = load_camera_calibration(str(calibration_path))

    assert calibration.camera_matrix.shape == (3, 3)
    assert calibration.camera_matrix[0, 0] > 0.0
    assert calibration.distortion_coefficients.shape == (4, 1)
    assert calibration.resolution == expected_resolution


def test_load_real_tof_calibration() -> None:
    """验证实际 ToF 标定包含的额外顶层字段不会影响 RGB 标定读取。"""
    calibration = load_camera_calibration(str(TOF_CALIBRATION_PATH))

    assert calibration.resolution == (2048, 1536)
    assert calibration.camera_matrix[0, 0] == pytest.approx(405.67036610341245)


@pytest.mark.parametrize(
    "document",
    [
        {
            "rgb": valid_tof_document()["rgb"],
            "cam0": valid_legacy_document()["cam0"],
        },
        {"itof": {}},
    ],
)
def test_ambiguous_or_missing_camera_block_is_rejected(
    tmp_path: Path, document
) -> None:
    """验证同时存在或完全缺失支持相机块时拒绝加载。"""
    calibration_path = tmp_path / "invalid_camera.yaml"
    write_yaml(calibration_path, document)

    with pytest.raises(ValueError):
        load_camera_calibration(str(calibration_path))


@pytest.mark.parametrize(
    ("document_factory", "camera_key", "field"),
    [
        (valid_tof_document, "rgb", "distortion_model"),
        (valid_tof_document, "rgb", "intrinsics"),
        (valid_tof_document, "rgb", "distortion_coeffs"),
        (valid_tof_document, "rgb", "resolution"),
        (valid_legacy_document, "cam0", "camera_model"),
        (valid_legacy_document, "cam0", "distortion_model"),
        (valid_legacy_document, "cam0", "intrinsics"),
        (valid_legacy_document, "cam0", "distortion_coeffs"),
        (valid_legacy_document, "cam0", "resolution"),
    ],
)
def test_missing_required_camera_fields_are_rejected(
    tmp_path: Path, document_factory, camera_key: str, field: str
) -> None:
    """验证 ToF 和 legacy 标定的全部必需字段均不可缺失。"""
    document = document_factory()
    del document[camera_key][field]
    calibration_path = tmp_path / "missing_field.yaml"
    write_yaml(calibration_path, document)

    with pytest.raises(ValueError):
        load_camera_calibration(str(calibration_path))


@pytest.mark.parametrize(
    ("document_factory", "field", "value"),
    [
        (valid_tof_document, "distortion_model", "radtan"),
        (valid_legacy_document, "camera_model", "omni"),
        (valid_legacy_document, "distortion_model", "radtan"),
        (valid_tof_document, "intrinsics", [0.0, 401.0, 320.0, 240.0]),
        (valid_tof_document, "intrinsics", [400.0, -1.0, 320.0, 240.0]),
        (valid_tof_document, "intrinsics", [400.0, 401.0, 320.0]),
        (
            valid_tof_document,
            "intrinsics",
            [400.0, float("nan"), 320.0, 240.0],
        ),
        (valid_tof_document, "distortion_coeffs", [0.1, 0.2]),
        (
            valid_tof_document,
            "distortion_coeffs",
            [0.1, float("inf"), 0.2, 0.3],
        ),
        (valid_tof_document, "resolution", [2048, 0]),
        (valid_tof_document, "resolution", [2048.5, 1536]),
        (valid_tof_document, "resolution", "2048"),
        (valid_tof_document, "resolution", [True, 1536]),
    ],
)
def test_invalid_camera_calibration_is_rejected(
    tmp_path: Path, document_factory, field: str, value
) -> None:
    """验证模型、数值数组和非整数像素分辨率错误均被拒绝。"""
    document = document_factory()
    camera_key = "rgb" if "rgb" in document else "cam0"
    document[camera_key][field] = value
    calibration_path = tmp_path / "invalid_camera.yaml"
    write_yaml(calibration_path, document)

    with pytest.raises(ValueError):
        load_camera_calibration(str(calibration_path))


def test_validate_millimeter_range() -> None:
    """验证有效毫米 ROS 参数校验成功。"""
    distance_range = validate_gripper_distance_range(48.31, 129.0)

    assert distance_range.min_distance_mm == pytest.approx(48.31)
    assert distance_range.max_distance_mm == pytest.approx(129.0)


@pytest.mark.parametrize(
    ("min_distance_mm", "max_distance_mm"),
    [
        (129.0, 48.31),
        (-1.0, 129.0),
        (np.nan, 129.0),
        (48.31, np.inf),
        ("invalid", 129.0),
    ],
)
def test_invalid_range_is_rejected(min_distance_mm, max_distance_mm) -> None:
    """验证非法毫米范围参数会被拒绝。"""
    with pytest.raises(ValueError):
        validate_gripper_distance_range(min_distance_mm, max_distance_mm)


def test_missing_required_files_are_rejected(tmp_path: Path) -> None:
    """验证缺失相机标定文件会被拒绝。"""
    with pytest.raises(ValueError):
        load_camera_calibration(str(tmp_path / "missing.yaml"))
