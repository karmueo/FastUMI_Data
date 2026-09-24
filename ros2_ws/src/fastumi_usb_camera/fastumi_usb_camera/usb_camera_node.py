"""按启动模式将 UVC MJPEG 帧发布为 ROS 2 原始或压缩图像。"""

from __future__ import annotations

import signal
import threading
import time
from typing import Callable

import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from fastumi_usb_camera.capture import UvcCamera
from fastumi_usb_camera.latest_frame import CapturedFrame, LatestFrame


# 连续无完整帧的设备错误门限，单位为秒。
FRAME_TIMEOUT_SECONDS = 5.0
# 解码失败告警的最小间隔，单位为秒。
WARNING_INTERVAL_SECONDS = 5.0


class UsbCameraNode(Node):
    """在 ROS 线程按模式发布图像，在 UVC 线程仅缓存最新帧。"""

    def __init__(self, camera_factory: Callable[..., UvcCamera] = UvcCamera) -> None:
        """声明启动参数、创建发布器并打开指定 USB 相机。"""
        super().__init__("usb_camera_node")
        self.declare_parameter("vendor_id", 0x1BCF)
        self.declare_parameter("product_id", 0x28C4)
        self.declare_parameter("device_uid", "")
        self.declare_parameter("video_device", "")
        self.declare_parameter("width", 1280)
        self.declare_parameter("height", 960)
        self.declare_parameter("fps", 30)
        self.declare_parameter("frame_id", "usb_camera_optical_frame")
        self.declare_parameter("publish_compressed", False)
        self.declare_parameter("publish_reliability", "reliable")
        self.add_on_set_parameters_callback(self._reject_runtime_parameters)

        # 这些参数只在创建节点时读取，不支持运行时改变设备或发布器。
        camera_config = {
            name: self._positive_int(name)
            for name in ("width", "height", "fps")
        }
        device_uid = self.get_parameter("device_uid").value
        if not isinstance(device_uid, str):
            raise ValueError("device_uid 必须是字符串")
        camera_config["device_uid"] = device_uid.strip()
        video_device = self.get_parameter("video_device").value
        if not isinstance(video_device, str):
            raise ValueError("video_device 必须是字符串")
        camera_config["video_device"] = video_device.strip()
        if not camera_config["video_device"]:
            camera_config["vendor_id"] = self._positive_int("vendor_id")
            camera_config["product_id"] = self._positive_int("product_id")
        self._width = camera_config["width"]
        self._height = camera_config["height"]
        self._frame_id = str(self.get_parameter("frame_id").value).strip()
        if not self._frame_id:
            raise ValueError("frame_id 不能为空")
        self._publish_compressed = bool(
            self.get_parameter("publish_compressed").value
        )
        publish_reliability = str(self.get_parameter("publish_reliability").value).strip().lower()
        if publish_reliability not in ("reliable", "best_effort"):
            raise ValueError("publish_reliability 只能是 reliable 或 best_effort")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=(ReliabilityPolicy.RELIABLE if publish_reliability == "reliable"
                         else ReliabilityPolicy.BEST_EFFORT),
            durability=DurabilityPolicy.VOLATILE,
        )
        self._image_publisher = (
            None if self._publish_compressed
            else self.create_publisher(Image, "image_raw", qos)
        )
        self._compressed_publisher = (
            self.create_publisher(CompressedImage, "image_raw/compressed", qos)
            if self._publish_compressed else None
        )
        self._bridge = None if self._publish_compressed else CvBridge()
        self._frames = LatestFrame()
        self.published_count = 0
        self.decode_failed_count = 0
        self._last_warning = 0.0
        self._last_complete_at = time.monotonic()
        self._started = False
        self._closed = False
        self._guard = self.create_guard_condition(self._publish_latest)
        self._watchdog = self.create_timer(0.5, self._check_camera)
        self._diagnostics = self.create_timer(5.0, self._report_counts)
        try:
            self._camera = camera_factory(**camera_config)
        except Exception:
            self.destroy_node()
            raise
        camera_label = camera_config["video_device"] or (
            f"{camera_config['vendor_id']:04x}:{camera_config['product_id']:04x}"
        )
        output_topic = (
            "image_raw/compressed" if self._publish_compressed else "image_raw"
        )
        self.get_logger().info(
            f"USB 相机 {camera_label}，{self._width}x{self._height}@"
            f"{camera_config['fps']} FPS，发布 {output_topic}"
        )

    def _positive_int(self, name: str) -> int:
        """获取并校验正整数相机参数。"""
        value = self.get_parameter(name).value
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} 必须是正整数")
        if name in ("vendor_id", "product_id") and value > 0xFFFF:
            raise ValueError(f"{name} 必须在 USB 16 位标识范围内")
        return value

    def _reject_runtime_parameters(self, _parameters: list) -> SetParametersResult:
        """拒绝运行时修改设备和发布接口参数。"""
        return SetParametersResult(
            successful=False, reason="USB 相机参数仅支持在节点启动时配置"
        )

    def start(self) -> None:
        """在相机线程开始接收 JPEG 帧。"""
        if self._started or self._closed:
            raise RuntimeError("相机节点不能重复启动")
        self._last_complete_at = time.monotonic()
        self._started = True
        self._camera.start(self._receive_frame)

    def _receive_frame(self, jpeg: bytes) -> None:
        """在采集线程记录主机 ROS 时间戳并覆盖旧待处理帧。"""
        stamp = self.get_clock().now().to_msg()
        self._last_complete_at = time.monotonic()
        self._frames.put(CapturedFrame(stamp, jpeg))
        self._guard.trigger()

    def _publish_latest(self) -> None:
        """在 ROS 线程发布最新帧；压缩模式直接转发 JPEG 字节。"""
        frame = self._frames.take()
        if frame is None:
            return
        if self._compressed_publisher is not None:
            compressed = CompressedImage()
            compressed.header.stamp = frame.stamp
            compressed.header.frame_id = self._frame_id
            compressed.format = "jpeg"
            compressed.data = frame.jpeg
            self._compressed_publisher.publish(compressed)
            self.published_count += 1
            return
        image = cv2.imdecode(np.frombuffer(frame.jpeg, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (self._height, self._width):
            self.decode_failed_count += 1
            now = time.monotonic()
            if now - self._last_warning >= WARNING_INTERVAL_SECONDS:
                self.get_logger().warning("JPEG 解码失败或图像尺寸与采集模式不符，已跳过")
                self._last_warning = now
            return
        raw = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
        raw.header.stamp = frame.stamp
        raw.header.frame_id = self._frame_id
        self._image_publisher.publish(raw)
        self.published_count += 1

    def _check_camera(self) -> None:
        """发现采集异常、意外停止或连续无帧时终止节点。"""
        if not self._started or self._closed:
            return
        if self._camera.last_error is not None:
            raise RuntimeError("UVC 相机采集失败") from self._camera.last_error
        if self._camera.stopped():
            raise RuntimeError("UVC 相机采集意外停止")
        if time.monotonic() - self._last_complete_at >= FRAME_TIMEOUT_SECONDS:
            raise RuntimeError("UVC 相机连续 5 秒没有完整图像帧")

    def _report_counts(self) -> None:
        """周期记录采集、发布、积压覆盖和坏帧数量。"""
        self.get_logger().info(
            f"帧统计: 采集={self._frames.captured_count}, "
            f"发布={self.published_count}, 覆盖={self._frames.overwritten_count}, "
            f"不完整={self._camera.incomplete_count}, "
            f"解码失败={self.decode_failed_count}"
        )

    def close(self) -> None:
        """先停止并释放相机，再由调用方销毁 ROS 节点。"""
        if self._closed:
            return
        self._closed = True
        self._camera.close()


def main(args: list[str] | None = None) -> None:
    """启动相机节点，运行 ROS 事件循环并在退出时释放设备。"""
    rclpy.init(args=args)
    node: UsbCameraNode | None = None
    stop_event = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        """将 SIGINT/SIGTERM 转换为事件，避免清理时二次中断。"""
        stop_event.set()

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        node = UsbCameraNode()
        node.start()
        while rclpy.ok() and not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        try:
            if node is not None:
                try:
                    node.close()
                finally:
                    node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
            signal.signal(signal.SIGINT, previous_sigint)
            signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
