"""验证 AprilGrid 检测后端和二维三维观测组装。"""

import cv2
import numpy as np
import pytest

from fastumi_data.tracker_camera_config import (
    AprilGridSpec,
    CheckerboardSpec,
    checkerboard_object_points,
    tag_object_corners,
)
from fastumi_data.tracker_camera_detection import (
    CheckerboardObservation,
    DetectionRejected,
    OpenCvAprilTagDetector,
    OpenCvCheckerboardDetector,
    RawTagDetection,
    aprilgrid_tag_to_pitch_ratios,
    build_aprilgrid_observation,
    build_checkerboard_observation,
    validate_checkerboard_observations,
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
    assert observation.feature_count == 12
    assert observation.corner_count == 12
    np.testing.assert_allclose(
        observation.object_points_m[:4], tag_object_corners(make_spec(), 0)
    )


def test_aprilgrid_calibration_points_keep_original_corners() -> None:
    """AprilGrid 标定点应保留鱼眼图像中的原始四角对应。"""
    image_corners = np.asarray([
        [10.0, 30.0], [30.0, 30.0],
        [30.0, 10.0], [10.0, 10.0],
    ])
    observation = build_aprilgrid_observation(
        123,
        [RawTagDetection(0, image_corners, None, None)],
        make_spec(),
        min_tags=1,
    )
    np.testing.assert_allclose(observation.image_points_px, image_corners)
    np.testing.assert_allclose(
        observation.object_points_m, tag_object_corners(make_spec(), 0)
    )
    assert observation.feature_count == 4


def test_aprilgrid_tag_to_pitch_ratio_uses_adjacent_tags() -> None:
    """几何诊断应按相邻标签边长和中心距计算无量纲比例。"""
    first = np.asarray([
        [0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 0.0]
    ])
    second = first + np.asarray([13.0, 0.0])
    observation = build_aprilgrid_observation(
        0,
        [
            RawTagDetection(0, first, None, None),
            RawTagDetection(1, second, None, None),
        ],
        make_spec(),
        min_tags=2,
    )
    np.testing.assert_allclose(
        aprilgrid_tag_to_pitch_ratios(observation, make_spec()),
        [10.0 / 13.0],
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
    assert OpenCvAprilTagDetector("tag36h11").settings[
        "corner_refinement"
    ] == "contour"


def test_opencv_detector_corrects_one_bit_damage() -> None:
    """OpenCV 后端应纠正目标板图像中的单个 payload bit 损伤。"""
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    marker = cv2.aruco.generateImageMarker(dictionary, 5, 80)
    marker[20:30, 20:30] = 255 - marker[20:30, 20:30]
    canvas = np.full((160, 160), 255, dtype=np.uint8)
    canvas[40:120, 40:120] = marker

    detections = OpenCvAprilTagDetector("tag36h11").detect(canvas)

    assert [item.tag_id for item in detections] == [5]


def test_opencv_detector_normalizes_kalibr_corner_order() -> None:
    """检测角点应归一化为 Kalibr 的左下、右下、右上、左上顺序。"""
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )
    marker = cv2.aruco.generateImageMarker(dictionary, 5, 240)
    rotated_marker = cv2.rotate(marker, cv2.ROTATE_180)
    canvas = np.full((320, 320), 255, dtype=np.uint8)
    canvas[40:280, 40:280] = rotated_marker

    detections = OpenCvAprilTagDetector("tag36h11").detect(canvas)

    assert [item.tag_id for item in detections] == [5]
    np.testing.assert_allclose(
        detections[0].corners_px,
        np.asarray(
            [
                [40.0, 279.0],
                [279.0, 279.0],
                [279.0, 40.0],
                [40.0, 40.0],
            ]
        ),
        atol=1.1,
    )


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


def make_checkerboard_spec() -> CheckerboardSpec:
    """返回测试使用的 11×8 内部角点棋盘格。"""
    return CheckerboardSpec(11, 8, 0.03, 0.03)


def make_checkerboard_image(spec: CheckerboardSpec) -> np.ndarray:
    """生成带灰色边界的高对比度经典棋盘格图像。"""
    square_size = 40
    board = np.empty(
        (
            (spec.target_rows + 1) * square_size,
            (spec.target_cols + 1) * square_size,
        ),
        dtype=np.uint8,
    )
    for row in range(spec.target_rows + 1):
        for column in range(spec.target_cols + 1):
            value = 255 if (row + column) % 2 == 0 else 0
            board[
                row * square_size:(row + 1) * square_size,
                column * square_size:(column + 1) * square_size,
            ] = value
    image = np.full(
        (board.shape[0] + 80, board.shape[1] + 80), 127, dtype=np.uint8
    )
    image[40:-40, 40:-40] = board
    return image


def test_opencv_checkerboard_detector_accepts_image_formats() -> None:
    """经典检测器应在 mono8、BGR 和 BGRA 输入中返回完整精化网格。"""
    spec = make_checkerboard_spec()
    detector = OpenCvCheckerboardDetector(spec)
    gray = make_checkerboard_image(spec)
    for image in (
        gray,
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA),
    ):
        corners = detector.detect(image)
        assert corners.shape == (spec.corner_count, 2)
        assert np.all(np.isfinite(corners))
    assert detector.settings["pattern_size"] == [11, 8]
    assert detector.settings["corner_subpix_window"] == [11, 11]


@pytest.mark.parametrize(
    "image,message",
    [
        (np.zeros((300, 400, 2), dtype=np.uint8), "mono8"),
        (np.zeros((300, 400), dtype=np.float32), "uint8"),
    ],
)
def test_checkerboard_detector_rejects_invalid_image_formats(
    image: np.ndarray, message: str
) -> None:
    """检测器应拒绝错误通道数和非 uint8 图像。"""
    detector = OpenCvCheckerboardDetector(make_checkerboard_spec())
    with pytest.raises(ValueError, match=message):
        detector.detect(image)


def test_checkerboard_detector_returns_empty_for_incomplete_grid() -> None:
    """没有完整棋盘格时检测器应返回约定的空 Nx2 数组。"""
    detector = OpenCvCheckerboardDetector(make_checkerboard_spec())
    corners = detector.detect(np.full((300, 400), 127, dtype=np.uint8))
    assert corners.shape == (0, 2)


def test_checkerboard_observation_requires_all_configured_corners() -> None:
    """棋盘格观测必须保留完整二维三维对应及固定角点数量。"""
    spec = make_checkerboard_spec()
    image_points = np.arange(spec.corner_count * 2, dtype=np.float64).reshape(
        spec.corner_count, 2
    )
    observation = build_checkerboard_observation(123, image_points, spec)
    assert isinstance(observation, CheckerboardObservation)
    assert observation.feature_count == spec.corner_count
    np.testing.assert_allclose(
        observation.object_points_m, checkerboard_object_points(spec)
    )
    validate_checkerboard_observations([observation] * 5, spec)
    with pytest.raises(DetectionRejected, match="完整 88"):
        build_checkerboard_observation(123, image_points[:-1], spec)
    with pytest.raises(DetectionRejected, match="有限"):
        build_checkerboard_observation(
            123,
            np.full((spec.corner_count, 2), np.nan),
            spec,
        )
