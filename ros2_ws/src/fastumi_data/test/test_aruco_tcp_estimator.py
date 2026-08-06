"""验证双 ArUco 的 pair 坐标构造、TCP 融合和整幅去畸变检测。"""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from fastumi_data.aruco_tcp_config import load_aruco_tcp_config
from fastumi_data.aruco_tcp_estimator import (
    DualArucoTcpEstimator,
    FrameEstimationError,
    TagPoseEstimate,
    estimate_tcp_from_tag_poses,
)
from fastumi_data.tracker_camera_config import FisheyeCameraModel


def _config_document():
    """返回测试使用的双 ArUco 配置文档。"""
    return {
        "schema_version": 2,
        "fixture_version": "dual-aruco-bootstrap-v1",
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


def _load_config(tmp_path: Path):
    """写入并加载允许使用的测试配置。"""
    path = tmp_path / "aruco_to_tcp.yaml"
    path.write_text(
        yaml.safe_dump(_config_document(), sort_keys=False), encoding="utf-8"
    )
    return load_aruco_tcp_config(str(path))


def _camera() -> FisheyeCameraModel:
    """返回零畸变的 600×600 合成相机。"""
    return FisheyeCameraModel(
        k=np.asarray(
            [[500.0, 0.0, 300.0], [0.0, 500.0, 300.0], [0.0, 0.0, 1.0]]
        ),
        d=np.zeros(4),
        resolution=(600, 600),
    )


def _tag(tag_id: int, center: list[float], normal=(0.0, 0.0, -1.0)):
    """构造带指定中心和法向的人工标签位姿。"""
    rotation = np.eye(4, dtype=np.float64)
    rotation[:3, :3] = np.diag([1.0, 1.0, 1.0])
    rotation[:3, 2] = np.asarray(normal, dtype=np.float64)
    rotation[:3, 0] = [1.0, 0.0, 0.0]
    rotation[:3, 1] = np.cross(rotation[:3, 2], rotation[:3, 0])
    rotation[:3, 1] /= np.linalg.norm(rotation[:3, 1])
    rotation[:3, 2] = np.cross(rotation[:3, 0], rotation[:3, 1])
    rotation[:3, 3] = center
    return TagPoseEstimate(tag_id, rotation, 0.0)


def test_pair_frame_uses_id_order_marker_sign_and_right_handed_axes(tmp_path):
    """ID 0→1 和 -1 法向应构成单位右手相机→pair 方向。"""
    config = _load_config(tmp_path)
    tag0 = _tag(0, [0.0, -0.063, 1.0])
    tag1 = _tag(1, [0.0, 0.063, 1.0])

    estimate = estimate_tcp_from_tag_poses(tag0, tag1, 1.0, config)

    np.testing.assert_allclose(estimate.camera_from_pair[:3, :3], np.eye(3))
    np.testing.assert_allclose(estimate.tcp_from_tag0_m, [0.012, 0.0, 1.018])
    np.testing.assert_allclose(estimate.tcp_from_tag1_m, [0.012, 0.0, 1.018])
    np.testing.assert_allclose(estimate.camera_from_tcp[:3, 3], [0.012, 0.0, 1.018])
    assert estimate.candidate_translation_difference_m == pytest.approx(0.0)
    assert np.linalg.det(estimate.camera_from_pair[:3, :3]) == pytest.approx(1.0)


def test_swapping_centers_reverses_y_and_keeps_rotation_right_handed(tmp_path):
    """交换两个中心后 Y 轴应反向，旋转仍满足右手系。"""
    config = _load_config(tmp_path)
    tag0 = _tag(0, [0.0, 0.063, 1.0])
    tag1 = _tag(1, [0.0, -0.063, 1.0])

    estimate = estimate_tcp_from_tag_poses(tag0, tag1, 1.0, config)

    np.testing.assert_allclose(estimate.camera_from_pair[:3, 1], [0.0, -1.0, 0.0])
    assert np.linalg.det(estimate.camera_from_pair[:3, :3]) == pytest.approx(1.0)


def test_candidate_fusion_weights_lower_reprojection_error(tmp_path):
    """候选融合应按重投影误差平方倒数偏向更可信标签。"""
    config = _load_config(tmp_path)
    tag0 = TagPoseEstimate(
        0,
        _tag(0, [0.0, -0.063, 1.0]).camera_from_tag,
        0.1,
    )
    tag1_transform = _tag(1, [0.002, 0.063, 1.0]).camera_from_tag
    tag1 = TagPoseEstimate(1, tag1_transform, 1.0)

    estimate = estimate_tcp_from_tag_poses(tag0, tag1, 1.0, config)

    distance_to_tag0 = np.linalg.norm(
        estimate.fused_tcp_position_m - estimate.tcp_from_tag0_m
    )
    distance_to_tag1 = np.linalg.norm(
        estimate.fused_tcp_position_m - estimate.tcp_from_tag1_m
    )
    assert distance_to_tag0 < distance_to_tag1


def _synthetic_image():
    """生成含有 ID 0/1 的正视 4×4 ArUco 合成 BGR 图像。"""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image = np.full((600, 600), 255, dtype=np.uint8)
    for marker_id, left in ((0, 140), (1, 420)):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 40)
        image[280:320, left:left + 40] = marker
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_estimator_rectifies_full_image_with_kalibr_k_and_detects_both_tags(
    tmp_path, monkeypatch
):
    """检测必须先用原 K 整幅去畸变，且不得估计新投影内参。"""
    config = _load_config(tmp_path)
    camera = _camera()
    observed = {}
    original = cv2.fisheye.initUndistortRectifyMap

    def record_init(k, d, r, p, size, map_type):
        """记录去畸变初始化参数并调用真实 OpenCV 实现。"""
        observed["k"] = np.array(k, copy=True)
        observed["p"] = np.array(p, copy=True)
        return original(k, d, r, p, size, map_type)

    monkeypatch.setattr(cv2.fisheye, "initUndistortRectifyMap", record_init)
    estimator = DualArucoTcpEstimator(camera, config)
    result = estimator.estimate(_synthetic_image(), openness=1.0, timestamp_ns=123)

    np.testing.assert_allclose(observed["k"], camera.k)
    np.testing.assert_allclose(observed["p"], camera.k)
    assert result.timestamp_ns == 123
    assert {result.tag0.tag_id, result.tag1.tag_id} == {0, 1}
    assert result.tag0.positive_depth is True
    assert result.tag1.positive_depth is True
    assert np.isfinite(result.tag0.reprojection_rmse_px)
    assert np.isfinite(result.tag1.reprojection_rmse_px)
    assert result.tag0.reprojection_rmse_px < 1.0
    assert result.tag1.reprojection_rmse_px < 1.0


def test_estimator_rejects_missing_duplicate_and_wrong_resolution(tmp_path, monkeypatch):
    """缺少 ID、重复 ID 和分辨率错误必须统一报告帧估计失败。"""
    config = _load_config(tmp_path)
    camera = _camera()
    estimator = DualArucoTcpEstimator(camera, config)
    image = _synthetic_image()

    class FakeDetector:
        """返回指定检测结果的最小 ArUco detector。"""

        def __init__(self, identifiers):
            """保存要返回的 ID 列表。"""
            self.identifiers = identifiers

        def detectMarkers(self, _image):
            """返回规则方形角点和配置的 ID。"""
            corners = [
                np.asarray(
                    [[[140.0, 280.0], [180.0, 280.0], [180.0, 320.0], [140.0, 320.0]]]
                )
                for _ in self.identifiers
            ]
            return corners, np.asarray(self.identifiers, dtype=np.int32).reshape(-1, 1), []

    for identifiers, message in [([0], "缺少"), ([0, 0], "重复")]:
        estimator._detector = FakeDetector(identifiers)
        with pytest.raises(FrameEstimationError, match=message):
            estimator.estimate(image, 1.0, 0)

    with pytest.raises(FrameEstimationError, match="分辨率"):
        estimator.estimate(np.zeros((100, 100, 3), dtype=np.uint8), 1.0, 0)


@pytest.mark.parametrize("failure", ["overlap", "normal", "negative", "nonfinite"])
def test_geometry_rejects_degenerate_tag_poses(tmp_path, failure):
    """中心重合、法向退化、负深度和非有限误差都必须拒绝。"""
    config = _load_config(tmp_path)
    if failure == "overlap":
        tag0 = _tag(0, [0.0, 0.0, 1.0])
        tag1 = _tag(1, [0.0, 0.0, 1.0])
    elif failure == "normal":
        tag0 = _tag(0, [0.0, -0.063, 1.0], [0.0, 1.0, 0.0])
        tag1 = _tag(1, [0.0, 0.063, 1.0], [0.0, 1.0, 0.0])
    elif failure == "negative":
        tag0 = _tag(0, [0.0, -0.063, -1.0])
        tag1 = _tag(1, [0.0, 0.063, -1.0])
    else:
        tag0 = TagPoseEstimate(
            0, _tag(0, [0.0, -0.063, 1.0]).camera_from_tag, float("nan")
        )
        tag1 = _tag(1, [0.0, 0.063, 1.0])

    with pytest.raises(FrameEstimationError):
        estimate_tcp_from_tag_poses(tag0, tag1, 1.0, config)
