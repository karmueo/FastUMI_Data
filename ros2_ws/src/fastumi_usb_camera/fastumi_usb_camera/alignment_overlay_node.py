"""订阅末端相机图像，叠加 UMI 夹爪参考 mask 后发布到独立的 JPEG 话题，供安装对齐查看。

原始相机话题不受影响（录制和推理仍用原图）；输出默认为 ``image_alignment/compressed``，
可用 ``rqt_image_view`` 查看。
"""

from __future__ import annotations

from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

from fastumi_usb_camera.alignment_overlay import TagDetector, load_reference, render

# 码角点误差日志的最小间隔，单位为秒。
LOG_INTERVAL_SECONDS = 2.0


class AlignmentOverlayNode(Node):
    """按限频解码输入图像、叠加参考 mask 并编码为 JPEG 发布。"""

    def __init__(self) -> None:
        super().__init__("alignment_overlay")
        self.declare_parameter("input_topic", "image_raw")
        self.declare_parameter("compressed", False)
        self.declare_parameter("output_topic", "image_alignment/compressed")
        self.declare_parameter("mask_file", "")
        self.declare_parameter("alpha", 0.5)
        self.declare_parameter("max_rate_hz", 10.0)
        self.declare_parameter("detect_tags", True)
        self.declare_parameter("jpeg_quality", 85)

        mask_file = str(self.get_parameter("mask_file").value).strip()
        if not mask_file:
            raise ValueError("mask_file 不能为空")
        self._reference = load_reference(Path(mask_file))
        self._alpha = float(self.get_parameter("alpha").value)
        if not 0.0 <= self._alpha <= 1.0:
            raise ValueError("alpha 必须在 [0, 1] 内")
        rate = float(self.get_parameter("max_rate_hz").value)
        if not rate > 0.0:
            raise ValueError("max_rate_hz 必须为正数")
        self._period = 1.0 / rate
        self._quality = int(self.get_parameter("jpeg_quality").value)
        self._detector = TagDetector() if self.get_parameter("detect_tags").value else None
        self._compressed = bool(self.get_parameter("compressed").value)
        self._bridge = None if self._compressed else CvBridge()
        self._last_processed = 0.0
        self._last_log = 0.0

        input_topic = str(self.get_parameter("input_topic").value)
        # 传感器 QoS 只保留最新帧，兼容 reliable 与 best_effort 发布端。
        self.create_subscription(
            CompressedImage if self._compressed else Image, input_topic,
            self._on_image, qos_profile_sensor_data)
        self._publisher = self.create_publisher(
            CompressedImage, str(self.get_parameter("output_topic").value),
            QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE))
        height, width = self._reference.mask.shape
        self.get_logger().info(
            f"对齐叠加：{mask_file}（{width}x{height}，透明度 {self._alpha:.2f}），"
            f"输入 {self.resolve_topic_name(input_topic)}，"
            f"输出 {self._publisher.topic_name}")

    def _on_image(self, message) -> None:
        """限频处理最新帧；尺寸不符时报错退出，避免显示错误的参考位置。"""
        now = time.monotonic()
        if now - self._last_processed < self._period:
            return
        self._last_processed = now
        if self._compressed:
            frame = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warning("JPEG 解码失败，已跳过")
                return
        else:
            frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        live = self._detector.detect(frame) if self._detector is not None else None
        output, errors = render(frame, self._reference, self._alpha, live)
        ok, jpeg = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            self.get_logger().warning("JPEG 编码失败，已跳过")
            return
        result = CompressedImage()
        result.header = message.header
        result.format = "jpeg"
        result.data = jpeg.tobytes()
        self._publisher.publish(result)
        if errors and now - self._last_log >= LOG_INTERVAL_SECONDS:
            self._last_log = now
            self.get_logger().info("手指码角点误差: " + ", ".join(
                f"id{key}={value:.1f}px" for key, value in errors.items()))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = AlignmentOverlayNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
