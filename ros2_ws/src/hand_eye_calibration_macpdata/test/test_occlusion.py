"""夹爪掩膜估计与板视图遮挡判定。"""

from pathlib import Path

import numpy as np
import pytest

from fastumi_bringup.transforms import invert
from hand_eye_calibration_macpdata.board import CheckerboardSpec, load_fisheye_camera
from hand_eye_calibration_macpdata.occlusion import (
    check_board_view, estimate_gripper_mask, load_mask,
)
from hand_eye_calibration_macpdata.synthetic import gripper_occluder, look_at, render_board


CAMERA = Path(__file__).resolve().parents[2] / "fastumi_usb_camera/config/EndEffectorCalib.yaml"
# 与相机同向（相机在板 -z 侧）的 OpenCV 约定板，外框 0.32 x 0.20 m。
SPEC = CheckerboardSpec(7, 4, 0.04, 0.04)


@pytest.fixture(scope="module")
def camera():
    return load_fisheye_camera(CAMERA).scaled(960, 540)


def _views(camera, occluder, count=8):
    center = SPEC.center_point()
    rng = np.random.default_rng(1)
    for _ in range(count):
        offset = np.array([*rng.uniform(-0.15, 0.15, 2), -rng.uniform(0.25, 0.4)])
        camera_from_board = invert(look_at(center + offset, center + np.array(
            [*rng.uniform(-0.05, 0.05, 2), 0.0])))
        yield render_board(SPEC, camera, camera_from_board, occluder=occluder)


def test_mask_covers_fixed_gripper_but_not_scene(camera):
    occluder = gripper_occluder(camera.width, camera.height)
    mask = estimate_gripper_mask(list(_views(camera, occluder)))
    gripper = occluder > 0
    assert np.mean(mask[gripper] > 0) > 0.95
    # 夹爪外（扣除膨胀带与暗角）基本不应被掩膜覆盖。
    center_region = mask[50:300, 200:760]
    assert np.mean(center_region > 0) < 0.02


def test_mask_needs_enough_frames(camera):
    with pytest.raises(ValueError):
        estimate_gripper_mask([np.zeros((540, 960, 3), np.uint8)] * 3)


def test_board_view_checks(camera):
    center = SPEC.center_point()
    mask = np.zeros((camera.height, camera.width), np.uint8)
    centered = invert(look_at(center + np.array([0.0, 0.0, -0.3]), center))
    assert check_board_view(SPEC, camera, centered, mask).ok

    mask[200:340, 400:560] = 255
    result = check_board_view(SPEC, camera, centered, mask)
    assert not result.ok and result.overlap_px > 0

    far = invert(look_at(center + np.array([0.0, 0.0, -3.0]), center))
    assert "过小" in check_board_view(SPEC, camera, far, None).reason
    behind = invert(look_at(center + np.array([0.0, 0.0, -0.3]), center + np.array(
        [0.0, 0.0, -1.0])))
    assert not check_board_view(SPEC, camera, behind, None).ok


def test_load_mask_checks_size(tmp_path):
    import cv2
    path = tmp_path / "mask.png"
    cv2.imwrite(str(path), np.full((10, 20), 7, np.uint8))
    assert load_mask(path, 20, 10).max() == 255
    with pytest.raises(ValueError):
        load_mask(path, 30, 10)
