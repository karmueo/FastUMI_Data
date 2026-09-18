"""测试双 ArUco 检测、三维距离和归一化边界。"""

import cv2
from gripper_openness.calibration import CameraCalibration
from gripper_openness.vision import GripperVision, normalize_distance
import numpy as np


def test_synthetic_markers_are_detected() -> None:
    """合成正视图中的两个标签应得到有限距离。"""
    camera = CameraCalibration(
        camera_matrix=np.asarray(
            [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        distortion_coefficients=np.zeros((4, 1)),
        resolution=(640, 480),
        model="fisheye",
    )
    vision = GripperVision(camera)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    image = np.full((480, 640), 255, dtype=np.uint8)
    for marker_id, center_x in ((0, 150), (1, 450)):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 80)
        image[200:280, center_x - 40:center_x + 40] = marker
    result = vision.detect(image)
    assert result.detected_marker_count == 2
    assert result.valid
    assert result.distance_mm is not None
    assert result.distance_mm > 0.0


def test_normalize_distance_clips_and_checks_range() -> None:
    """归一化在两端裁剪，并拒绝退化范围。"""
    assert normalize_distance(20.0, 50.0, 100.0) == 0.0
    assert normalize_distance(75.0, 50.0, 100.0) == 0.5
    assert normalize_distance(120.0, 50.0, 100.0) == 1.0
