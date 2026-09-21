"""ROS 2 夹爪开合度预测节点。"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
from cv_bridge import CvBridge
from fastumi_interfaces.msg import GripperState
from gripper_openness.calibration import (
    load_camera_calibration,
    load_gripper_calibration,
)
from gripper_openness.vision import GripperVision, normalize_distance, VisionResult
from gripper_openness.visualization import draw_vision_result
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class GripperOpennessNode(Node):
    """从双 ArUco 三维距离发布归一化夹爪开合度。"""

    def __init__(self, *, parameter_overrides: Optional[Sequence] = None) -> None:
        """加载相机和夹爪范围标定并创建 ROS 接口。"""
        super().__init__(
            "gripper_openness",
            parameter_overrides=_parameter_overrides(parameter_overrides),
        )
        self._declare_parameters()
        try:
            camera_path = str(self.get_parameter("camera_calibration_path").value)
            range_path = str(self.get_parameter("gripper_calibration_path").value)
            self._camera_calibration = load_camera_calibration(camera_path)
            self._range_calibration = load_gripper_calibration(range_path)
            if self._camera_calibration.resolution != self._range_calibration.resolution:
                raise ValueError(
                    "相机标定分辨率与夹爪范围标定分辨率不一致: "
                    f"{self._camera_calibration.resolution} != "
                    f"{self._range_calibration.resolution}"
                )
            self._vision = GripperVision(
                self._camera_calibration,
                marker_size_mm=self._range_calibration.marker_size_mm,
                dictionary_name=self._range_calibration.dictionary_name,
                left_marker_id=self._range_calibration.left_finger_tag_id,
                right_marker_id=self._range_calibration.right_finger_tag_id,
            )
            self._smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)
            if not (0.0 < self._smoothing_alpha <= 1.0):
                raise ValueError("smoothing_alpha 必须在 (0,1] 范围内")
            self._padding_pixels = int(self.get_parameter("roi_padding_pixels").value)
            if self._padding_pixels < 0:
                raise ValueError("roi_padding_pixels 不能为负数")
        except (ValueError, TypeError, cv2.error) as error:
            self.get_logger().error(f"加载夹爪开合度参数失败: {error}")
            raise RuntimeError("夹爪开合度参数无效") from error
        self._previous_filtered: Optional[float] = None
        self._bridge = CvBridge()
        self._setup_ros_interfaces()
        self.get_logger().info(
            "夹爪开合度节点已启动，订阅 "
            f"{self.get_parameter('image_topic').value}，输出范围 "
            f"{self._range_calibration.min_marker_dist_mm:.3f}.."
            f"{self._range_calibration.max_marker_dist_mm:.3f} mm"
        )

    def _declare_parameters(self) -> None:
        """声明预测节点的 ROS 参数。"""
        self.declare_parameter("image_topic", "/umi_camera/image_raw")
        self.declare_parameter("openness_topic", "/gripper/openness")
        self.declare_parameter("state_topic", "/gripper/state")
        self.declare_parameter("debug_image_topic", "/gripper/openness/debug_image")
        self.declare_parameter("publish_debug_image", False)
        self.declare_parameter("camera_calibration_path", "")
        self.declare_parameter("gripper_calibration_path", "")
        self.declare_parameter("roi_padding_pixels", 50)
        self.declare_parameter("smoothing_alpha", 1.0)

    def _setup_ros_interfaces(self) -> None:
        """创建图像订阅和开度、状态、调试图像发布器。"""
        image_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._openness_publisher = self.create_publisher(
            Float32, str(self.get_parameter("openness_topic").value), 10
        )
        self._state_publisher = self.create_publisher(
            GripperState, str(self.get_parameter("state_topic").value), 10
        )
        self._debug_publisher = None
        if bool(self.get_parameter("publish_debug_image").value):
            self._debug_publisher = self.create_publisher(
                Image, str(self.get_parameter("debug_image_topic").value), image_qos
            )
        self._image_subscription = self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self._image_callback,
            image_qos,
        )

    def _image_callback(self, message: Image) -> None:
        """处理一帧图像并发布带输入时间戳的夹爪状态。"""
        state = GripperState()
        state.header = message.header
        state.raw_openness = math.nan
        state.filtered_openness = math.nan
        state.marker_distance_mm = math.nan
        state.detected_marker_count = 0
        state.valid = False
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            roi = self._expanded_calibration_roi(image.shape[1], image.shape[0])
            result = self._vision.detect(image, roi)
        except (ValueError, cv2.error) as error:
            self._previous_filtered = None
            self._state_publisher.publish(state)
            self.get_logger().error(f"开合度图像处理失败: {error}", throttle_duration_sec=2.0)
            return
        state.detected_marker_count = result.detected_marker_count
        if not result.valid:
            self._previous_filtered = None
            self._state_publisher.publish(state)
            if self._debug_publisher is not None:
                self._publish_debug(image, message, result, "invalid")
            self.get_logger().warning(
                "本帧未获得两枚有效 ArUco 位姿", throttle_duration_sec=2.0
            )
            return

        raw = self._normalize(result.distance_mm)
        filtered = self._filter(raw)
        state.raw_openness = raw
        state.filtered_openness = filtered
        state.marker_distance_mm = result.distance_mm
        state.detected_marker_count = 2
        state.valid = True
        self._state_publisher.publish(state)
        output = Float32()
        output.data = filtered
        self._openness_publisher.publish(output)
        if self._debug_publisher is not None:
            self._publish_debug(image, message, result, f"openness={filtered:.3f}")

    def _expanded_calibration_roi(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """将标定 ROI 向外扩展并裁剪到当前图像边界。"""
        crop = self._range_calibration.crop_reference
        padding = self._padding_pixels
        x_min = max(0, int(crop["left_x"]) - padding)
        y_min = max(0, int(crop["top_y"]) - padding)
        x_max = min(width, int(crop["right_x"]) + padding)
        y_max = min(height, int(crop["bottom_y"]) + padding)
        if not (x_min < x_max and y_min < y_max):
            raise ValueError("扩展后的标定 ROI 无效")
        return x_min, y_min, x_max, y_max

    def _normalize(self, distance_mm: float) -> float:
        """将毫米距离映射并裁剪到 [0,1]。"""
        return normalize_distance(
            distance_mm,
            self._range_calibration.min_marker_dist_mm,
            self._range_calibration.max_marker_dist_mm,
        )

    def _filter(self, raw_value: float) -> float:
        """按指数平滑参数计算当前输出并保存历史值。"""
        if self._previous_filtered is None:
            filtered = raw_value
        else:
            filtered = (
                self._smoothing_alpha * raw_value
                + (1.0 - self._smoothing_alpha) * self._previous_filtered
            )
        self._previous_filtered = float(np.clip(filtered, 0.0, 1.0))
        return self._previous_filtered

    def _publish_debug(
        self, image, message: Image, result: VisionResult, label: str
    ) -> None:
        """绘制并发布带输入时间戳的调试图像。"""
        debug = draw_vision_result(
            image,
            result,
            label,
            camera_calibration=self._camera_calibration,
            marker_size_mm=self._range_calibration.marker_size_mm,
        )
        debug_message = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        debug_message.header = message.header
        self._debug_publisher.publish(debug_message)


def main(args: Optional[List[str]] = None) -> None:
    """启动夹爪开合度预测节点。"""
    rclpy.init(args=args)
    node = None
    try:
        node = GripperOpennessNode()
        rclpy.spin(node)
    except (RuntimeError, ValueError) as error:
        print(f"夹爪开合度节点启动失败: {error}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _parameter_overrides(parameter_overrides: Optional[Sequence]) -> List[Parameter]:
    """兼容 ROS Parameter 对象和测试常用的 ``(name, value)`` 元组。"""
    result = []
    for item in parameter_overrides or []:
        if isinstance(item, Parameter):
            result.append(item)
        else:
            name, value = item
            result.append(Parameter(name=name, value=value))
    return result


if __name__ == "__main__":
    main()
