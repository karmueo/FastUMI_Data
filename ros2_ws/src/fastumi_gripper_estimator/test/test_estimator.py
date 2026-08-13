"""测试鱼眼三维毫米估计和无量纲归一化行为。"""

import cv2
from fastumi_gripper_estimator.estimator import GripperOpennessEstimator

import numpy as np
import pytest


# 合成测试使用的 640x480 相机内参。
CAMERA_MATRIX = np.asarray(
    [
        [400.0, 0.0, 320.0],
        [0.0, 400.0, 240.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
# 非零 equidistant 系数用于验证角点校正。
FISHEYE_DISTORTION = np.asarray(
    [0.08, -0.017, 0.0046, -0.0031], dtype=np.float64
)


def make_estimator(
    alpha: float = 1.0,
    closed_distance_mm: float = 40.0,
    open_distance_mm: float = 80.0,
    distortion: np.ndarray = FISHEYE_DISTORTION,
) -> GripperOpennessEstimator:
    """创建使用合成相机参数的测试估计器。

    Args:
        alpha: 指数平滑权重。
        closed_distance_mm: 测试闭合距离，单位为毫米。
        open_distance_mm: 测试张开距离，单位为毫米。
        distortion: 四个 equidistant 鱼眼畸变系数。

    Returns:
        测试用三维估计器。
    """
    return GripperOpennessEstimator(
        camera_matrix=CAMERA_MATRIX,
        distortion_coefficients=distortion,
        closed_distance_mm=closed_distance_mm,
        open_distance_mm=open_distance_mm,
        marker_size_mm=16.0,
        smoothing_alpha=alpha,
    )


def generate_marker(
    dictionary, marker_id: int, marker_size: int
) -> np.ndarray:
    """使用当前 OpenCV 版本支持的 API 生成测试标记。

    Args:
        dictionary: ArUco 预定义字典。
        marker_id: 待生成的标记 ID。
        marker_size: 标记图像的边长，单位为像素。

    Returns:
        单通道 ArUco 标记图像。
    """
    generate_image_marker = getattr(cv2.aruco, "generateImageMarker", None)
    if callable(generate_image_marker):
        return generate_image_marker(dictionary, marker_id, marker_size)
    marker = np.empty((marker_size, marker_size), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, marker_size, marker, 1)
    return marker


def make_marker_image(
    left_center_x: int, right_center_x: int
) -> np.ndarray:
    """生成两枚等大正视 ArUco 标记的灰度图像。

    Args:
        left_center_x: ID 0 标记中心横坐标。
        right_center_x: ID 1 标记中心横坐标。

    Returns:
        包含两个标记的 640x480 灰度图像。
    """
    image = np.full((480, 640), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 70
    marker_y_min = 300
    for marker_id, center_x in (
        (0, left_center_x),
        (1, right_center_x),
    ):
        marker = generate_marker(dictionary, marker_id, marker_size)
        marker_x_min = center_x - marker_size // 2
        image[
            marker_y_min:marker_y_min + marker_size,
            marker_x_min:marker_x_min + marker_size,
        ] = marker
    return image


def test_normalize_distance_is_dimensionless_and_clipped() -> None:
    """验证毫米标定范围映射到无量纲单位区间。"""
    estimator = make_estimator(
        closed_distance_mm=40.0,
        open_distance_mm=80.0,
    )
    assert estimator.normalize_distance(20.0) == pytest.approx(0.0)
    assert estimator.normalize_distance(40.0) == pytest.approx(0.0)
    assert estimator.normalize_distance(60.0) == pytest.approx(0.5)
    assert estimator.normalize_distance(80.0) == pytest.approx(1.0)
    assert estimator.normalize_distance(100.0) == pytest.approx(1.0)


def test_invalid_estimator_parameters_are_rejected() -> None:
    """验证无效相机、距离、标记和平滑参数会被拒绝。"""
    with pytest.raises(ValueError):
        GripperOpennessEstimator(
            np.eye(2),
            FISHEYE_DISTORTION,
            40.0,
            80.0,
        )
    with pytest.raises(ValueError):
        GripperOpennessEstimator(
            CAMERA_MATRIX,
            np.zeros(5),
            40.0,
            80.0,
        )
    with pytest.raises(ValueError):
        make_estimator(closed_distance_mm=80.0, open_distance_mm=40.0)
    with pytest.raises(ValueError):
        GripperOpennessEstimator(
            CAMERA_MATRIX,
            FISHEYE_DISTORTION,
            40.0,
            80.0,
            marker_size_mm=0.0,
        )
    with pytest.raises(ValueError):
        make_estimator(alpha=0.0)
    with pytest.raises(ValueError):
        GripperOpennessEstimator(
            CAMERA_MATRIX,
            FISHEYE_DISTORTION,
            40.0,
            80.0,
            roi_ratios=(0.8, 0.2, 0.1, 0.9),
        )


def test_fisheye_projected_corners_recover_full_pose_in_mm() -> None:
    """验证 equidistant 合成角点可恢复毫米平移和旋转向量。"""
    estimator = make_estimator()
    half_size = 8.0
    object_points = np.asarray(
        [
            [-half_size, half_size, 0.0],
            [half_size, half_size, 0.0],
            [half_size, -half_size, 0.0],
            [-half_size, -half_size, 0.0],
        ],
        dtype=np.float64,
    ).reshape(-1, 1, 3)
    rotation_vector = np.asarray([0.08, -0.04, 0.03], dtype=np.float64)
    expected_position_mm = np.asarray([20.0, -10.0, 180.0], dtype=np.float64)
    distorted_corners, _ = cv2.fisheye.projectPoints(
        object_points,
        rotation_vector,
        expected_position_mm,
        CAMERA_MATRIX,
        FISHEYE_DISTORTION,
    )

    pose = estimator.estimate_marker_pose(distorted_corners)

    assert pose is not None
    assert pose.translation_mm == pytest.approx(expected_position_mm, abs=1e-4)
    assert pose.rotation_vector == pytest.approx(rotation_vector, abs=1e-4)


def test_estimate_marker_position_remains_pose_translation_compatible() -> None:
    """验证旧位置接口继续返回新位姿接口中的毫米平移向量。"""
    estimator = make_estimator()
    distorted_corners, _ = cv2.fisheye.projectPoints(
        estimator._marker_object_points.reshape(-1, 1, 3),
        np.asarray([0.02, -0.03, 0.01], dtype=np.float64),
        np.asarray([15.0, -8.0, 190.0], dtype=np.float64),
        CAMERA_MATRIX,
        FISHEYE_DISTORTION,
    )

    pose = estimator.estimate_marker_pose(distorted_corners)
    position_mm = estimator.estimate_marker_position(distorted_corners)

    assert pose is not None
    assert position_mm is not None
    assert position_mm == pytest.approx(pose.translation_mm, abs=1e-8)


def test_empty_and_missing_marker_images_return_no_estimate() -> None:
    """验证空图像报错且无标记图像不产生输出。"""
    estimator = make_estimator()
    with pytest.raises(ValueError):
        estimator.estimate(np.empty((0, 0), dtype=np.uint8))
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert estimator.estimate(blank_image) is None
    assert estimator.estimate(blank_image) is None


def test_image_resolution_must_match_calibration() -> None:
    """验证与标定宽高不同的输入图像会在三维估计前被拒绝。"""
    estimator = GripperOpennessEstimator(
        camera_matrix=CAMERA_MATRIX,
        distortion_coefficients=FISHEYE_DISTORTION,
        closed_distance_mm=40.0,
        open_distance_mm=80.0,
        image_resolution=(640, 480),
    )

    assert estimator.estimate(
        np.zeros((480, 640, 3), dtype=np.uint8)
    ) is None
    with pytest.raises(ValueError, match="分辨率"):
        estimator.estimate(np.zeros((240, 320, 3), dtype=np.uint8))


def test_falls_back_to_legacy_aruco_detector(monkeypatch) -> None:
    """验证缺少 ArucoDetector 时使用 OpenCV 4.6 的模块级 API。"""
    # 检测调用次数，用于确认估计器确实走过兼容分支。
    detection_calls = []

    def fake_detect_markers(image, dictionary):
        """记录旧版 API 调用并模拟未检测到标记。"""
        detection_calls.append((image, dictionary))
        return (), None, ()

    monkeypatch.setattr(cv2.aruco, "ArucoDetector", None, raising=False)
    monkeypatch.setattr(
        cv2.aruco, "detectMarkers", fake_detect_markers, raising=False
    )
    estimator = make_estimator()

    assert estimator.estimate(np.zeros((480, 640), dtype=np.uint8)) is None
    assert len(detection_calls) == 1


def test_detected_markers_produce_mm_diagnostic_and_unitless_output() -> None:
    """验证双标记生成内部毫米距离和无量纲最终结果。"""
    # 零畸变便于使用未扭曲的合成标记图像。
    estimator = make_estimator(distortion=np.zeros(4))
    result = estimator.estimate(make_marker_image(220, 420))

    assert result is not None
    assert result.distance_mm == pytest.approx(46.4, abs=1.5)
    assert result.raw_openness == pytest.approx(
        estimator.normalize_distance(result.distance_mm)
    )
    assert 0.0 <= result.openness <= 1.0


def test_exponential_smoothing_uses_unitless_values() -> None:
    """验证指数平滑仅作用于无量纲归一化结果。"""
    estimator = make_estimator(
        alpha=0.25,
        closed_distance_mm=10.0,
        open_distance_mm=60.0,
        distortion=np.zeros(4),
    )
    first_result = estimator.estimate(make_marker_image(270, 370))
    second_result = estimator.estimate(make_marker_image(220, 420))

    assert first_result is not None
    assert second_result is not None
    expected_smoothed = (
        0.25 * second_result.raw_openness
        + 0.75 * first_result.raw_openness
    )
    assert second_result.openness == pytest.approx(expected_smoothed)
    assert 0.0 <= second_result.openness <= 1.0
