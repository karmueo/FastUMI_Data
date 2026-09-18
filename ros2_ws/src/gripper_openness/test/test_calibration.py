"""测试相机标定解析、范围统计和夹爪 YAML 往返。"""

from pathlib import Path

from gripper_openness.calibration import (
    load_camera_calibration,
    load_gripper_calibration,
    RangeAccumulator,
    write_gripper_calibration,
)

import numpy as np
import pytest


def _write_camera(path: Path, model: str = "fisheye") -> None:
    """写入测试用相机标定。"""
    path.write_text(
        "cam0:\n"
        "  intrinsics: [400.0, 400.0, 320.0, 240.0]\n"
        f"  distortion_model: {model}\n"
        "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        "  resolution: [640, 480]\n",
        encoding="utf-8",
    )


def _corners(x: float) -> np.ndarray:
    """构造测试用方形标记角点。"""
    return np.asarray([[x, 100], [x + 20, 100], [x + 20, 120], [x, 120]], dtype=float)


def test_camera_models_and_range_round_trip(tmp_path: Path) -> None:
    """验证鱼眼标定读取和范围 YAML 原子往返。"""
    camera_path = tmp_path / "camera.yaml"
    _write_camera(camera_path)
    camera = load_camera_calibration(str(camera_path))
    assert camera.is_fisheye
    assert camera.resolution == (640, 480)

    accumulator = RangeAccumulator()
    accumulator.add(50.0, _corners(100), _corners(300), camera.resolution)
    accumulator.add(100.0, _corners(105), _corners(350), camera.resolution)
    calibration = accumulator.build(16.0, "DICT_4X4_50", 0, 1)
    output = tmp_path / "gripper.yaml"
    write_gripper_calibration(str(output), calibration)
    loaded = load_gripper_calibration(str(output))
    assert loaded.min_marker_dist_mm == 50.0
    assert loaded.max_marker_dist_mm == 100.0
    assert loaded.crop_reference["left_x"] == 100
    with pytest.raises(FileExistsError):
        write_gripper_calibration(str(output), calibration)


def test_range_rejects_degenerate_endpoints() -> None:
    """相同的最小和最大距离不能成为有效标定。"""
    accumulator = RangeAccumulator()
    corners_left = _corners(100)
    corners_right = _corners(300)
    accumulator.add(50.0, corners_left, corners_right, (640, 480))
    with pytest.raises(ValueError, match="两个不同"):
        accumulator.build(16.0, "DICT_4X4_50", 0, 1)
