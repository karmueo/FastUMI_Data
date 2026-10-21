"""Kalibr AprilGrid / 棋盘格配置、鱼眼检测与 PnP、角点顺序约定。"""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

from fastumi_bringup.transforms import invert, rotation_angle
from hand_eye_calibration_macpdata.board import (
    AprilGridSpec, CheckerboardSpec, board_spec_from_dict, load_board_spec,
    load_fisheye_camera, make_detector,
)
from hand_eye_calibration_macpdata.synthetic import gripper_occluder, look_at, render_board


SRC = Path(__file__).resolve().parents[2]
CALIBRATION = SRC.parents[1] / "config/calibration"
CAMERA = SRC / "fastumi_usb_camera/config/EndEffectorCalib.yaml"
APRILGRID = load_board_spec(CALIBRATION / "april_6x6.yaml")
CHECKERBOARD = load_board_spec(CALIBRATION / "checkerboard_11x8.yaml")


@pytest.fixture(scope="module")
def camera():
    return load_fisheye_camera(CAMERA)


def _front_view(spec, offset):
    """相机看向板中心的 ^cam T_board；offset 为板系下相机相对板中心的位置（z 取正面一侧）。"""
    center = spec.center_point()
    offset = np.asarray(offset, dtype=float)
    if isinstance(spec, AprilGridSpec):
        return invert(look_at(center + offset, center, up_hint=(0.0, 1.0, 0.0)))
    offset[2] = -offset[2]
    return invert(look_at(center + offset, center))


def test_loads_repository_board_configs():
    assert APRILGRID == AprilGridSpec(6, 6, 0.055, 0.3)
    assert APRILGRID.corner_count == 144 and APRILGRID.hand_eye_pattern() is None
    assert CHECKERBOARD == CheckerboardSpec(11, 8, 0.03, 0.03)
    assert CHECKERBOARD.corner_count == 88
    assert CHECKERBOARD.hand_eye_pattern() == "2 8 11 0.03"
    wide = load_board_spec(CALIBRATION / "checkerboard_11x8_45mm.yaml")
    assert wide.size_m == pytest.approx((0.54, 0.405))
    for spec in (APRILGRID, CHECKERBOARD, wide):
        assert board_spec_from_dict(spec.to_dict()) == spec


def test_spec_validation(tmp_path):
    with pytest.raises(ValueError, match="一奇一偶"):
        CheckerboardSpec(9, 7, 0.03, 0.03)
    with pytest.raises(ValueError):
        AprilGridSpec(6, 6, 0.055, 0.3, "tag25h9")
    with pytest.raises(ValueError, match="target_type"):
        board_spec_from_dict({"board": {"type": "charuco"}})
    with pytest.raises(ValueError, match="缺少"):
        board_spec_from_dict({"target_type": "aprilgrid", "tagCols": 6})


def test_aprilgrid_object_points_follow_kalibr():
    """Tag0 左下角为原点，x 向右、y 向上，pitch = tagSize·(1+tagSpacing)。"""
    pitch = 0.055 * 1.3
    points = APRILGRID.object_points([0, 1, 2, 3, 4 * 7 + 2])
    assert points[:4, :2] == pytest.approx(np.array(
        [[0, 0], [0.055, 0], [0.055, 0.055], [0, 0.055]]))
    assert points[4, :2] == pytest.approx([pitch + 0.055, pitch + 0.055])
    assert APRILGRID.size_m == pytest.approx((0.055 + 5 * pitch,) * 2)


@pytest.mark.parametrize("spec", [APRILGRID, CHECKERBOARD], ids=["aprilgrid", "checkerboard"])
@pytest.mark.parametrize("offset", [(0.0, 0.0, 0.35), (0.12, -0.06, 0.33), (-0.1, 0.1, 0.4)])
def test_detects_and_recovers_board_pose(camera, spec, offset):
    camera_from_board = _front_view(spec, offset)
    image = render_board(spec, camera, camera_from_board, supersample=1)
    observation = make_detector(spec, camera).detect(image)
    assert observation.corner_count >= 0.85 * spec.corner_count
    assert observation.reprojection_rms_px < 1.0
    error = invert(camera_from_board) @ observation.camera_from_board
    assert np.linalg.norm(error[:3, 3]) < 0.002
    assert rotation_angle(np.eye(3), error[:3, :3]) < math.radians(0.3)


def test_checkerboard_order_is_unique_under_180_degree_roll(camera):
    """相机绕光轴转 180° 时 OpenCV 角点顺序会反转；统一后板系不变。"""
    center = CHECKERBOARD.center_point()
    for up_hint in ((0.0, -1.0, 0.0), (0.0, 1.0, 0.0)):
        camera_from_board = invert(look_at(center + np.array([0.03, 0.02, -0.35]), center,
                                           up_hint=up_hint))
        image = render_board(CHECKERBOARD, camera, camera_from_board, supersample=1)
        observation = make_detector(CHECKERBOARD, camera).detect(image)
        error = invert(camera_from_board) @ observation.camera_from_board
        assert rotation_angle(np.eye(3), error[:3, :3]) < math.radians(0.3)


def test_aprilgrid_back_side_is_not_detected(camera):
    """从背面看（镜像）不应得到位姿，确认渲染与检测的正反面约定一致。"""
    center = APRILGRID.center_point()
    camera_from_board = invert(look_at(center + np.array([0.0, 0.0, -0.35]), center))
    image = render_board(APRILGRID, camera, camera_from_board, supersample=1)
    assert make_detector(APRILGRID, camera).detect(image).camera_from_board is None


def test_aprilgrid_matches_fastumi_data_detector(camera):
    """与 fastumi_data 已在实板验证的 AprilGrid 检测器给出相同的角点–三维点对应。"""
    # fastumi_data 未必已构建；其检测模块是纯 Python，可直接从源码目录导入。
    if str(SRC / "fastumi_data") not in sys.path:
        sys.path.append(str(SRC / "fastumi_data"))
    detection = pytest.importorskip("fastumi_data.tracker_camera_detection")
    config = pytest.importorskip("fastumi_data.tracker_camera_config")
    image = render_board(APRILGRID, camera, _front_view(APRILGRID, (0.05, -0.04, 0.35)),
                         supersample=1)
    reference_spec = config.AprilGridSpec(6, 6, 0.055, 0.3, "tag36h11")
    reference = detection.build_aprilgrid_observation(
        0, detection.OpenCvAprilTagDetector("tag36h11").detect(image), reference_spec, 4)
    observation = make_detector(APRILGRID, camera).detect(image)
    ours = dict(zip(observation.ids.tolist(), observation.pixels))
    reference_ids = [4 * tag + k for tag in reference.tag_ids for k in range(4)]
    assert set(reference_ids) <= set(ours)
    assert APRILGRID.object_points(reference_ids) == pytest.approx(reference.object_points_m)
    pixels = np.array([ours[i] for i in reference_ids])
    # 本模块额外做 X 交点 cornerSubPix，差异应在亚像素内。
    assert np.max(np.linalg.norm(pixels - reference.image_points_px, axis=1)) < 2.0


def test_aprilgrid_partial_occlusion_still_gives_pose(camera):
    offset = (0.0, -0.12, 0.3)
    camera_from_board = _front_view(APRILGRID, offset)
    image = render_board(APRILGRID, camera, camera_from_board,
                         occluder=gripper_occluder(camera.width, camera.height), supersample=1)
    observation = make_detector(APRILGRID, camera).detect(image)
    assert 0 < observation.corner_count < APRILGRID.corner_count
    error = invert(camera_from_board) @ observation.camera_from_board
    assert np.linalg.norm(error[:3, 3]) < 0.003


def test_wrong_image_size_is_rejected(camera):
    detector = make_detector(CHECKERBOARD, camera)
    with pytest.raises(ValueError, match="尺寸"):
        detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
