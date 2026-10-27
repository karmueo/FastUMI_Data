"""双 ArUco d_tag 检测与求解：合成鱼眼图恢复真值、IPPE 歧义消除、鲁棒平均与 YAML 读取。"""

import math
from pathlib import Path

import numpy as np
import pytest

from fastumi_bringup.transforms import axis_angle_matrix, invert, make_transform
from hand_eye_calibration_macpdata.board import load_fisheye_camera
from hand_eye_calibration_macpdata.dual_tag import (
    DualTagError, DualTagEstimator, TagPose, camera_from_dtag, load_matrix_yaml, robust_average,
    select_consistent_pair, transform_difference,
)
from hand_eye_calibration_macpdata.synthetic import render_dual_tags


SRC = Path(__file__).resolve().parents[2]
REPO = SRC.parents[1]
CAMERA = SRC / "fastumi_usb_camera/config/EndEffectorCalib.yaml"


def gripper_dtag_pose(tilt_deg=15.0, origin=(0.005, 0.07, 0.13)):
    """真值 ^cam T_d_tag：夹爪 tag 在相机前下方，正面朝向相机并绕 X 倾斜 tilt_deg。"""
    origin = np.asarray(origin, dtype=float)
    z_axis = -origin / np.linalg.norm(origin)
    x_axis = np.array([1.0, 0.0, 0.0]) - z_axis[0] * z_axis
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])
    return make_transform(rotation @ axis_angle_matrix((1, 0, 0), math.radians(tilt_deg)), origin)


@pytest.fixture(scope="module")
def camera():
    return load_fisheye_camera(CAMERA)


@pytest.mark.parametrize("distance", [0.06, 0.12])
def test_recovers_dtag_pose_from_synthetic_image(camera, distance):
    truth = gripper_dtag_pose()
    image = render_dual_tags(camera, truth, tag_distance_m=distance)
    observation = DualTagEstimator(camera).estimate(image)
    translation_mm, angle_deg = transform_difference(truth, observation.camera_from_dtag)
    assert np.linalg.norm(translation_mm) < 1.0
    assert angle_deg < 2.0
    assert observation.tag_distance_m == pytest.approx(distance, abs=1e-3)
    assert observation.reprojection_rms_px < 1.0
    transform = observation.camera_from_dtag
    # +X 由 ID 0 指向 ID 1；+Z 指向相机（与原点方向相反）。
    assert np.dot(transform[:3, 0], observation.tag1.center - observation.tag0.center) > 0
    assert np.dot(transform[:3, 2], transform[:3, 3]) < 0
    assert np.allclose(np.cross(transform[:3, 2], transform[:3, 0]), transform[:3, 1])


def test_missing_tag_is_rejected(camera):
    blank = np.full((camera.height, camera.width, 3), 100, dtype=np.uint8)
    with pytest.raises(DualTagError, match="缺少 tag"):
        DualTagEstimator(camera).estimate(blank)


def test_wrong_image_size_is_rejected(camera):
    with pytest.raises(DualTagError, match="图像尺寸"):
        DualTagEstimator(camera).estimate(np.zeros((10, 10, 3), dtype=np.uint8))


def _tag(tag_id, center, rotation, rms=0.3):
    return TagPose(tag_id, make_transform(rotation, center), rms, np.zeros((4, 2)))


def test_select_consistent_pair_resolves_ippe_flip():
    truth = gripper_dtag_pose()
    rotation = truth[:3, :3]
    center0 = truth[:3, 3] - 0.03 * rotation[:, 0]
    center1 = truth[:3, 3] + 0.03 * rotation[:, 0]
    # 翻转候选：两枚 tag 的法向同向偏 25°（法向彼此一致，只能靠共面性排除），RMS 反而略小。
    flipped = axis_angle_matrix(rotation[:, 1], math.radians(25.0)) @ rotation
    good0, bad0 = _tag(0, center0, rotation, 0.30), _tag(0, center0, flipped, 0.25)
    good1, bad1 = _tag(1, center1, rotation, 0.30), _tag(1, center1, flipped, 0.25)
    tag0, tag1 = select_consistent_pair([bad0, good0], [bad1, good1])
    assert tag0 is good0 and tag1 is good1
    observation = camera_from_dtag(tag0, tag1)
    assert transform_difference(truth, observation.camera_from_dtag)[1] < 1e-6


def test_camera_from_dtag_rejects_coincident_tags():
    tag = _tag(0, (0.0, 0.05, 0.1), np.eye(3))
    with pytest.raises(DualTagError):
        camera_from_dtag(tag, _tag(1, (0.0, 0.05, 0.1), np.eye(3)))


def test_robust_average_rejects_outlier():
    rng = np.random.default_rng(0)
    truth = gripper_dtag_pose()
    transforms = []
    for _ in range(12):
        noise = make_transform(axis_angle_matrix(rng.normal(size=3), math.radians(0.2)),
                               rng.normal(scale=2e-4, size=3))
        transforms.append(truth @ noise)
    transforms.append(truth @ make_transform(np.eye(3), (0.02, 0.0, 0.0)))
    average = robust_average(transforms)
    assert not average.inliers[-1] and average.inliers[:-1].all()
    translation_mm, angle_deg = transform_difference(truth, average.transform)
    assert np.linalg.norm(translation_mm) < 0.3 and angle_deg < 0.2
    assert average.translation_rms_m < 1e-3 and average.rotation_rms_deg < 0.5


def test_robust_average_keeps_small_sets():
    transforms = [np.eye(4), make_transform(np.eye(3), (0.05, 0.0, 0.0))]
    average = robust_average(transforms)
    assert average.inliers.all()
    assert average.transform[0, 3] == pytest.approx(0.025)


def test_load_matrix_yaml_reads_repo_calibrations():
    dtag_from_tcp = load_matrix_yaml(REPO / "config/calibration/T_d_tag_tcp.yaml", "T_d_tag_tcp")
    camera_from_flange = load_matrix_yaml(REPO / "config/calibration/T_cam_end.yaml",
                                          "cam0.T_cam_hand")
    assert dtag_from_tcp[1, 3] == pytest.approx(0.12)
    assert invert(camera_from_flange)[1, 3] == pytest.approx(0.08812, abs=1e-4)


def test_load_matrix_yaml_rejects_bad_input(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("T: [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]\n")
    with pytest.raises(ValueError, match="刚体"):
        load_matrix_yaml(path, "T")
    with pytest.raises(ValueError, match="缺少键"):
        load_matrix_yaml(path, "cam0.T")
