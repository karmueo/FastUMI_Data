"""测试夹爪像素间距归一化、平滑和输入校验行为。"""

import cv2
from fastumi_gripper_estimator.estimator import GripperOpennessEstimator

import numpy as np
import pytest


def make_estimator(alpha: float = 1.0) -> GripperOpennessEstimator:
    """创建使用简化标定范围的测试估计器。

    Args:
        alpha: 指数平滑权重。

    Returns:
        测试用估计器。
    """
    return GripperOpennessEstimator(
        closed_distance_px=100.0,
        open_distance_px=300.0,
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


def test_normalize_distance_clips_to_unit_interval() -> None:
    """验证标定范围内线性映射且超出范围时裁剪。"""
    estimator = make_estimator()
    assert estimator.normalize_distance(50.0) == pytest.approx(0.0)
    assert estimator.normalize_distance(100.0) == pytest.approx(0.0)
    assert estimator.normalize_distance(200.0) == pytest.approx(0.5)
    assert estimator.normalize_distance(300.0) == pytest.approx(1.0)
    assert estimator.normalize_distance(350.0) == pytest.approx(1.0)


def test_invalid_calibration_is_rejected() -> None:
    """验证无效距离范围、平滑系数和 ROI 会被拒绝。"""
    with pytest.raises(ValueError):
        GripperOpennessEstimator(300.0, 100.0)
    with pytest.raises(ValueError):
        GripperOpennessEstimator(100.0, 300.0, smoothing_alpha=0.0)
    with pytest.raises(ValueError):
        GripperOpennessEstimator(
            100.0,
            300.0,
            roi_ratios=(0.8, 0.2, 0.1, 0.9),
        )


def test_empty_image_is_rejected() -> None:
    """验证空图像不会进入 OpenCV 检测器。"""
    estimator = make_estimator()
    with pytest.raises(ValueError):
        estimator.estimate(np.empty((0, 0), dtype=np.uint8))


def test_missing_markers_returns_none_without_state_update() -> None:
    """验证无标记图像返回空结果且不会产生虚假开合度。"""
    estimator = make_estimator(alpha=0.2)
    blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert estimator.estimate(blank_image) is None
    assert estimator.estimate(blank_image) is None


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


def test_detects_synthetic_marker_distance() -> None:
    """验证两枚合成 ArUco 标记能产生预期的开合度。"""
    estimator = make_estimator()
    # 白色背景为每枚标记保留检测所需的安静边界。
    image = np.full((480, 640), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 70
    marker_y_min = 300
    for marker_id, center_x in ((0, 220), (1, 420)):
        marker = generate_marker(dictionary, marker_id, marker_size)
        marker_x_min = center_x - marker_size // 2
        image[
            marker_y_min:marker_y_min + marker_size,
            marker_x_min:marker_x_min + marker_size,
        ] = marker
    result = estimator.estimate(image)
    assert result is not None
