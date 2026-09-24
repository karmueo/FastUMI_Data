"""使用模拟相机验证 ROS 图像发布、时间戳和设备故障处理。"""

import time

import cv2
import numpy as np
import pytest
import rclpy
from rclpy.parameter import Parameter
from rclpy.qos import ReliabilityPolicy

from fastumi_usb_camera.usb_camera_node import UsbCameraNode


class FakeCamera:
    """代替真实 USB 设备保存采集回调及关闭状态。"""

    def __init__(self, **config):
        """记录相机配置，不触碰任何真实设备。"""
        self.config = config
        self.callback = None
        self.last_error = None
        self.incomplete_count = 0
        self.closed = False
        self.is_stopped = False

    def start(self, callback):
        """保存图像采集回调。"""
        self.callback = callback

    def stopped(self):
        """报告模拟相机的停止状态。"""
        return self.is_stopped

    def close(self):
        """记录设备关闭。"""
        self.closed = True


class FakePublisher:
    """保存发布消息，以便核对内容和时间戳。"""

    def __init__(self):
        """创建空消息集合。"""
        self.messages = []

    def publish(self, message):
        """记录节点发布的一帧图像。"""
        self.messages.append(message)


@pytest.fixture
def camera_node(request):
    """按用例选择原始或压缩模式，结束后释放 ROS 资源。"""
    compressed = getattr(request, "param", False)
    rclpy.init(args=[
        "--ros-args", "-p", f"publish_compressed:={str(compressed).lower()}"
    ])
    node = UsbCameraNode(camera_factory=FakeCamera)
    try:
        yield node
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


def test_raw_mode_decodes_without_creating_compressed_publisher(camera_node):
    """默认模式仅创建 raw 发布器，图像解码为 bgr8。"""
    node = camera_node
    assert node._image_publisher is not None
    assert node._image_publisher.qos_profile.reliability == ReliabilityPolicy.RELIABLE
    assert node._image_publisher.qos_profile.depth == 20
    raw_publisher = FakePublisher()
    node._image_publisher = raw_publisher
    assert node._compressed_publisher is None
    node.start()
    image = np.zeros((960, 1280, 3), dtype=np.uint8)
    image[:, :, 1] = 87
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    jpeg = encoded.tobytes()
    node._camera.callback(jpeg)
    rclpy.spin_once(node, timeout_sec=1.0)
    node._publish_latest()

    assert len(raw_publisher.messages) == 1
    raw = raw_publisher.messages[0]
    assert (raw.height, raw.width, raw.encoding) == (960, 1280, "bgr8")
    assert raw.header.stamp.sec > 0
    assert raw.header.frame_id == "usb_camera_optical_frame"
    assert node.published_count == 1


@pytest.mark.parametrize("camera_node", [True], indirect=True)
def test_compressed_mode_forwards_jpeg_without_decoding(camera_node, monkeypatch):
    """压缩模式不创建 raw 发布器，保留原始 JPEG 字节且不解码。"""
    node = camera_node
    assert node._image_publisher is None
    assert node._bridge is None
    compressed_publisher = FakePublisher()
    node._compressed_publisher = compressed_publisher

    def fail_if_decoded(*_args, **_kwargs):
        """发现压缩模式触发 JPEG 解码时立即失败。"""
        raise AssertionError("压缩模式不应解码 JPEG")

    monkeypatch.setattr(cv2, "imdecode", fail_if_decoded)
    node.start()
    jpeg = b"\xff\xd8camera-jpeg\xff\xd9"
    node._camera.callback(jpeg)
    rclpy.spin_once(node, timeout_sec=1.0)
    node._publish_latest()

    assert len(compressed_publisher.messages) == 1
    compressed = compressed_publisher.messages[0]
    assert compressed.header.stamp.sec > 0
    assert compressed.header.frame_id == "usb_camera_optical_frame"
    assert compressed.format == "jpeg"
    assert bytes(compressed.data) == jpeg
    assert node.published_count == 1


def test_decode_failure_and_watchdog(camera_node):
    """raw 模式跳过坏 JPEG，设备异常和持续无帧使节点报错。"""
    node = camera_node
    node.start()
    node._camera.callback(b"invalid-jpeg")
    node._publish_latest()
    assert node.decode_failed_count == 1
    assert node.published_count == 0

    node._camera.last_error = OSError("USB disconnected")
    with pytest.raises(RuntimeError, match="采集失败"):
        node._check_camera()
    node._camera.last_error = None
    node._last_complete_at = time.monotonic() - 5.1
    with pytest.raises(RuntimeError, match="连续 5 秒"):
        node._check_camera()


def test_runtime_parameter_change_is_rejected(camera_node):
    """运行时改变相机模式时明确拒绝，避免参数与设备实际状态分离。"""
    results = camera_node.set_parameters([Parameter("width", value=640)])
    assert not results[0].successful
    assert camera_node.get_parameter("width").value == 1280


def test_best_effort_publisher_override():
    """启动参数实际应用到 JPEG/raw 共用的图像发布器。"""
    rclpy.init(args=["--ros-args", "-p", "publish_reliability:=best_effort"])
    node = UsbCameraNode(camera_factory=FakeCamera)
    try:
        assert node._image_publisher.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()
