"""验证 AprilGrid 检测后端和二维三维观测组装。"""

import cv2
import numpy as np
import pytest

from fastumi_data.tracker_camera_config import (
    AprilGridSpec,
    tag_object_corners,
)
from fastumi_data.tracker_camera_detection import (
    DetectionRejected,
    OpenCvAprilTagDetector,
    RawTagDetection,
    build_aprilgrid_observation,
    validate_tag_family,
)


def make_spec() -> AprilGridSpec:
    """返回测试使用的 6×6 tag36h11 标定板。"""
    return AprilGridSpec(6, 6, 0.055, 0.3, "tag36h11")


def test_build_observation_sorts_ids_and_matches_object_points() -> None:
    """乱序检测应按 ID 排序并保持二维三维角点对应。"""
    detections = [
        RawTagDetection(7, np.full((4, 2), 70.0), None, None),
        RawTagDetection(0, np.full((4, 2), 10.0), None, None),
        RawTagDetection(35, np.full((4, 2), 350.0), None, None),
    ]
    observation = build_aprilgrid_observation(
        123, detections, make_spec(), min_tags=3
    )
    assert observation.tag_ids == (0, 7, 35)
    assert observation.image_points_px.shape == (12, 2)
    np.testing.assert_allclose(
        observation.object_points_m[:4], tag_object_corners(make_spec(), 0)
    )


def test_duplicate_id_rejects_frame() -> None:
    """同帧重复 ID 会造成不唯一对应，应拒绝整帧。"""
    detection = RawTagDetection(2, np.zeros((4, 2)), None, None)
    with pytest.raises(DetectionRejected, match="重复"):
        build_aprilgrid_observation(
            0, [detection, detection], make_spec(), min_tags=1
        )


def test_out_of_range_ids_are_filtered_and_reported() -> None:
    """目标范围外 ID 应被忽略，同时在有效观测中留下诊断。"""
    detections = [
        RawTagDetection(1, np.zeros((4, 2)), None, None),
        RawTagDetection(80, np.ones((4, 2)), None, None),
    ]
    observation = build_aprilgrid_observation(
        10, detections, make_spec(), min_tags=1
    )
    assert observation.tag_ids == (1,)
    assert observation.ignored_tag_ids == (80,)


def test_invalid_corners_and_too_few_tags_reject_frame() -> None:
    """非有限角点或有效标签不足都不能进入 PnP。"""
    invalid = RawTagDetection(0, np.full((4, 2), np.nan), None, None)
    with pytest.raises(DetectionRejected, match="有限"):
        build_aprilgrid_observation(0, [invalid], make_spec(), min_tags=1)
    valid = RawTagDetection(0, np.zeros((4, 2)), None, None)
    with pytest.raises(DetectionRejected, match="至少 2"):
        build_aprilgrid_observation(0, [valid], make_spec(), min_tags=2)


def test_opencv_detector_decodes_tag36h11() -> None:
    """OpenCV 后端应从合成图像解出正确 ID。"""
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    marker = cv2.aruco.generateImageMarker(dictionary, 5, 240)
    canvas = np.full((320, 320), 255, dtype=np.uint8)
    canvas[40:280, 40:280] = marker
    detections = OpenCvAprilTagDetector("tag36h11").detect(canvas)
    assert [item.tag_id for item in detections] == [5]
    assert detections[0].corners_px.shape == (4, 2)


def test_detector_accepts_bgr_and_rejects_unknown_family() -> None:
    """后端应统一 BGR 输入，并明确拒绝不支持的标签族。"""
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    marker = cv2.aruco.generateImageMarker(dictionary, 3, 120)
    gray = np.pad(marker, 30, constant_values=255)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert OpenCvAprilTagDetector("tag36h11").detect(bgr)[0].tag_id == 3
    with pytest.raises(ValueError, match="不支持"):
        OpenCvAprilTagDetector("tag25h9")


def test_tag_family_probe_requires_enough_valid_frames() -> None:
    """标签族预检必须覆盖指定数量的有效观测。"""
    detection = RawTagDetection(0, np.zeros((4, 2)), None, None)
    observation = build_aprilgrid_observation(
        0, [detection], make_spec(), min_tags=1
    )
    with pytest.raises(DetectionRejected, match="至少 2"):
        validate_tag_family([observation], make_spec(), minimum_probe_frames=2)
    validate_tag_family(
        [observation, observation], make_spec(), minimum_probe_frames=2
    )
