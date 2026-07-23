"""实现订阅相机图像并发布夹爪归一化开合度的 ROS 2 节点。"""

from typing import List, Optional

import cv2
from cv_bridge import CvBridge
from fastumi_gripper_estimator.estimator import (
    GripperOpennessEstimator,
    OpennessEstimate,
)
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class GripperOpennessNode(Node):
    """从 RGB 图像估计夹爪开合度并发布 ROS 2 话题。"""

    def __init__(self) -> None:
        """读取参数并创建订阅器、开合度发布器和调试发布器。"""
        super().__init__("gripper_openness_estimator")
        self._declare_parameters()

        # 输入、输出和调试话题名称。
        image_topic = str(self.get_parameter("image_topic").value)
        openness_topic = str(self.get_parameter("openness_topic").value)
        debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        # 是否发布带检测标注的调试图像。
        self._publish_debug_image = bool(
            self.get_parameter("publish_debug_image").value
        )
        # ROI 参数按 x_min、y_min、x_max、y_max 顺序传给纯算法模块。
        roi_ratios = tuple(
            float(value) for value in self.get_parameter("roi_ratios").value
        )
        # 视觉估计器保存 ArUco 检测器与平滑状态。
        self._estimator = GripperOpennessEstimator(
            closed_distance_px=float(
                self.get_parameter("closed_distance_px").value
            ),
            open_distance_px=float(self.get_parameter("open_distance_px").value),
            smoothing_alpha=float(self.get_parameter("smoothing_alpha").value),
            dictionary_name=str(self.get_parameter("dictionary_name").value),
            left_marker_id=int(self.get_parameter("left_marker_id").value),
            right_marker_id=int(self.get_parameter("right_marker_id").value),
            roi_ratios=roi_ratios,
        )
        # cv_bridge 负责 ROS Image 与 OpenCV 数组互转。
        self._bridge = CvBridge()
        # 图像流采用小队列和 BEST_EFFORT，避免处理延迟持续累积。
        image_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._openness_publisher = self.create_publisher(
            Float32, openness_topic, 10
        )
        self._debug_publisher = None
        if self._publish_debug_image:
            self._debug_publisher = self.create_publisher(
                Image, debug_image_topic, image_qos
            )
        self._image_subscription = self.create_subscription(
            Image, image_topic, self._image_callback, image_qos
        )
        self.get_logger().info(
            f"订阅 {image_topic}，发布 {openness_topic}；"
            "0 表示闭合，1 表示完全张开"
        )

    def _declare_parameters(self) -> None:
        """声明节点支持的全部 ROS 参数及默认值。"""
        self.declare_parameter(
            "image_topic",
            "/xv_sdk/SN250801DR48FB26001253/"
            "rgb_fisheye_undistorted/image",
        )
        self.declare_parameter("openness_topic", "/gripper/openness")
        self.declare_parameter("debug_image_topic", "/gripper/openness/debug_image")
        self.declare_parameter("publish_debug_image", False)
        self.declare_parameter("dictionary_name", "DICT_4X4_50")
        self.declare_parameter("left_marker_id", 0)
        self.declare_parameter("right_marker_id", 1)
        self.declare_parameter("closed_distance_px", 200.0)
        self.declare_parameter("open_distance_px", 557.0)
        self.declare_parameter("smoothing_alpha", 0.35)
        self.declare_parameter("roi_ratios", [0.15, 0.58, 0.85, 0.82])

    def _image_callback(self, message: Image) -> None:
        """处理一帧图像，在检测有效时发布开合度。

        Args:
            message: 相机发布的 ROS Image 消息。

        Side Effects:
            发布 Float32 开合度；启用调试参数时还会发布标注图像。
        """
        try:
            # BGR 图像同时供估计算法和调试绘图使用。
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            estimate = self._estimator.estimate(image)
        except (ValueError, RuntimeError) as error:
            self.get_logger().error(
                f"图像转换或估计失败: {error}", throttle_duration_sec=2.0
            )
            return
        if estimate is None:
            self.get_logger().warning(
                "夹爪的两枚 ArUco 标记未同时检出，本帧不发布开合度",
                throttle_duration_sec=2.0,
            )
            return

        # 标准 Float32 消息承载 [0, 1] 归一化结果。
        openness_message = Float32()
        openness_message.data = estimate.openness
        self._openness_publisher.publish(openness_message)
        if self._debug_publisher is not None:
            debug_image = self._draw_debug_image(image, estimate)
            debug_message = self._bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
            debug_message.header = message.header
            self._debug_publisher.publish(debug_message)

    @staticmethod
    def _draw_debug_image(image, estimate: OpennessEstimate):
        """在图像副本上绘制 ROI、标记中心、间距和开合度。

        Args:
            image: BGR 输入图像。
            estimate: 当前帧估计结果。

        Returns:
            带可视化标注的 BGR 图像副本。
        """
        # 绘图使用副本，确保不修改 cv_bridge 返回的输入缓存。
        debug_image = image.copy()
        x_min, y_min, x_max, y_max = estimate.roi
        left_point = tuple(int(round(value)) for value in estimate.left_center)
        right_point = tuple(int(round(value)) for value in estimate.right_center)
        cv2.rectangle(debug_image, (x_min, y_min), (x_max, y_max), (255, 180, 0), 2)
        cv2.line(debug_image, left_point, right_point, (0, 255, 0), 3)
        cv2.circle(debug_image, left_point, 8, (0, 0, 255), -1)
        cv2.circle(debug_image, right_point, 8, (255, 0, 0), -1)
        # 调试文字同时展示平滑输出和原始像素距离。
        label = (
            f"openness={estimate.openness:.3f} "
            f"distance={estimate.distance_px:.1f}px"
        )
        cv2.putText(
            debug_image,
            label,
            (x_min, max(30, y_min - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return debug_image


def main(args: Optional[List[str]] = None) -> None:
    """初始化 ROS 2、运行节点并在退出时释放资源。

    Args:
        args: 可选 ROS 2 命令行参数列表。
    """
    rclpy.init(args=args)
    # 节点对象需要在 finally 中显式销毁。
    node = GripperOpennessNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
