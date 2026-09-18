"""测试 ROS2 预测节点的消息状态和无效帧处理。"""

from pathlib import Path

import cv2
from gripper_openness.calibration import GripperCalibration, write_gripper_calibration
import numpy as np
import pytest


class _PublisherCapture:
    """保存节点发布的消息，便于无 DDS 订阅器测试。"""

    def __init__(self) -> None:
        """初始化消息列表。"""
        self.messages = []

    def publish(self, message) -> None:
        """保存一条发布消息。"""
        self.messages.append(message)


def _make_frame(gap: int) -> np.ndarray:
    """生成包含两个正视 ArUco 标签的 BGR 图像。"""
    image = np.full((480, 640), 255, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for marker_id, center_x in ((0, 320 - gap // 2), (1, 320 + gap // 2)):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 80)
        image[200:280, center_x - 40:center_x + 40] = marker
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_prediction_node_publishes_state_and_valid_openness(tmp_path: Path) -> None:
    """有效帧发布开度，无效帧只发布 invalid 状态。"""
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("fastumi_interfaces")
    from cv_bridge import CvBridge
    from gripper_openness.gripper_openness_node import GripperOpennessNode

    camera_path = tmp_path / "camera.yaml"
    camera_path.write_text(
        "cam0:\n"
        "  intrinsics: [400.0, 400.0, 320.0, 240.0]\n"
        "  distortion_model: fisheye\n"
        "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        "  resolution: [640, 480]\n",
        encoding="utf-8",
    )
    range_path = tmp_path / "range.yaml"
    write_gripper_calibration(
        str(range_path),
        GripperCalibration(
            16.0,
            "DICT_4X4_50",
            0,
            1,
            39.0,
            59.0,
            (640, 480),
            {"top_y": 200, "bottom_y": 280, "left_x": 100, "right_x": 540},
            2,
        ),
    )
    was_initialized = rclpy.ok()
    if not was_initialized:
        rclpy.init()
    try:
        node = GripperOpennessNode(
            parameter_overrides=[
                ("camera_calibration_path", str(camera_path)),
                ("gripper_calibration_path", str(range_path)),
            ]
        )
        state_capture = _PublisherCapture()
        openness_capture = _PublisherCapture()
        node._state_publisher = state_capture
        node._openness_publisher = openness_capture
        bridge = CvBridge()
        valid_message = bridge.cv2_to_imgmsg(_make_frame(200), encoding="bgr8")
        valid_message.header.stamp.sec = 17
        node._image_callback(valid_message)
        invalid_message = bridge.cv2_to_imgmsg(
            np.zeros((480, 640, 3), dtype=np.uint8), encoding="bgr8"
        )
        invalid_message.header.stamp.sec = 18
        node._image_callback(invalid_message)
        assert len(state_capture.messages) == 2
        assert state_capture.messages[0].valid
        assert state_capture.messages[0].header.stamp.sec == 17
        assert not state_capture.messages[1].valid
        assert len(openness_capture.messages) == 1
        assert 0.0 <= openness_capture.messages[0].data <= 1.0
        node.destroy_node()
    finally:
        if not was_initialized and rclpy.ok():
            rclpy.shutdown()
