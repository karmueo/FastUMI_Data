"""测试相机标定文件加载和夹爪毫米范围参数校验。"""

from pathlib import Path

from fastumi_gripper_estimator.calibration import (
    load_camera_calibration,
    validate_gripper_distance_range,
)
import numpy as np
import pytest
import yaml


def write_yaml(path: Path, document) -> None:
    """写入测试 YAML。

    Args:
        path: 临时 YAML 路径。
        document: 待序列化内容。
    """
    path.write_text(
        yaml.safe_dump(document, allow_unicode=True),
        encoding="utf-8",
    )


def valid_camera_document():
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


def test_load_camera_calibration(tmp_path: Path) -> None:
    """验证有效 Kalibr YAML 被转换为矩阵和鱼眼系数。"""
    calibration_path = tmp_path / "camera.yaml"
    write_yaml(calibration_path, valid_camera_document())

    calibration = load_camera_calibration(str(calibration_path))

    assert calibration.camera_matrix.shape == (3, 3)
    assert calibration.camera_matrix[0, 0] == pytest.approx(400.0)
    assert calibration.camera_matrix[1, 1] == pytest.approx(401.0)
    assert calibration.distortion_coefficients.shape == (4, 1)
    assert calibration.resolution == (640, 480)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("camera_model", "omni"),
        ("distortion_model", "radtan"),
        ("intrinsics", [400.0, 401.0, 320.0]),
        ("distortion_coeffs", [0.1, 0.2]),
        ("resolution", [640, 0]),
    ],
)
def test_invalid_camera_calibration_is_rejected(
    tmp_path: Path, field: str, value
) -> None:
    """验证错误模型、字段长度和分辨率会被拒绝。"""
    document = valid_camera_document()
    document["cam0"][field] = value
    calibration_path = tmp_path / "invalid_camera.yaml"
    write_yaml(calibration_path, document)

    with pytest.raises(ValueError):
        load_camera_calibration(str(calibration_path))


def test_validate_millimeter_range() -> None:
    """验证有效毫米 ROS 参数校验成功。"""
    distance_range = validate_gripper_distance_range(
        min_distance_mm=48.31,
        max_distance_mm=129.0,
    )

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
def test_invalid_range_is_rejected(
    min_distance_mm, max_distance_mm
) -> None:
    """验证非法毫米范围参数会被拒绝。"""
    with pytest.raises(ValueError):
        validate_gripper_distance_range(
            min_distance_mm,
            max_distance_mm,
        )


def test_missing_required_files_are_rejected(tmp_path: Path) -> None:
    """验证缺失相机标定文件会被拒绝。"""
    with pytest.raises(ValueError):
        load_camera_calibration(str(tmp_path / "missing.yaml"))
