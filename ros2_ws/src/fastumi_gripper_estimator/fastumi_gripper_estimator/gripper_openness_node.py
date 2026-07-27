"""订阅原始鱼眼 RGB 图像并发布无量纲夹爪归一化距离。"""

import math
from typing import List, Optional

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from fastumi_interfaces.msg import GripperState
from fastumi_gripper_estimator.calibration import (
    load_camera_calibration,
    validate_gripper_distance_range,
)
from fastumi_gripper_estimator.estimator import (
    GripperOpennessEstimator,
    OpennessEstimate,
)
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


# 默认原始 RGB 图像话题。
DEFAULT_IMAGE_TOPIC = "/xv_sdk/SN250801DR48FB26001253/rgb/image"
# 随包安装的默认 Kalibr equidistant 鱼眼标定文件名。
DEFAULT_CAMERA_CALIBRATION_FILENAME = "camera_calibration.yaml"
# 默认夹爪编号。
DEFAULT_GRIPPER_ID = 0
# 默认左右手指 ArUco 标记编号。
DEFAULT_LEFT_FINGER_TAG_ID = 0
DEFAULT_RIGHT_FINGER_TAG_ID = 1
# 默认闭合和张开标签中心距离，单位为毫米。
DEFAULT_MIN_MARKER_DIST_MM = 48.31
DEFAULT_MAX_MARKER_DIST_MM = 129.0


class GripperOpennessNode(Node):
    """从原始 RGB 图像估计并发布无量纲夹爪归一化距离。"""

    def __init__(self) -> None:
        """加载严格标定参数并创建 ROS 订阅器和发布器。"""
        super().__init__("gripper_openness_estimator")
        self._declare_parameters()

        # 输入、输出和调试话题名称。
        image_topic = str(self.get_parameter("image_topic").value)
        openness_topic = str(self.get_parameter("openness_topic").value)
        state_topic = str(self.get_parameter("state_topic").value)
        debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        # 空参数使用包共享目录中的默认标定，也允许 launch 传入外部标定。
        camera_calibration_path = str(
            self.get_parameter("camera_calibration_path").value
        )
        if not camera_calibration_path.strip():
            camera_calibration_path = self._default_calibration_path()
        # 夹爪编号用于区分配置，毫米距离用于最终归一化。
        gripper_id = int(
            self.get_parameter("gripper_range.gripper_id").value
        )
        min_distance_mm = float(
            self.get_parameter(
                "gripper_range.min_marker_dist_mm"
            ).value
        )
        max_distance_mm = float(
            self.get_parameter(
                "gripper_range.max_marker_dist_mm"
            ).value
        )
        # 是否发布带检测标注的调试图像。
        self._publish_debug_image = bool(
            self.get_parameter("publish_debug_image").value
        )
        # ROI 参数按 x_min、y_min、x_max、y_max 顺序传给纯算法模块。
        roi_ratios = tuple(
            float(value) for value in self.get_parameter("roi_ratios").value
        )

        try:
            # 相机标定和夹爪范围均严格校验，失败时阻止节点运行。
            camera_calibration = load_camera_calibration(
                camera_calibration_path
            )
            # 每帧必须与该宽高一致，才能直接使用标定内参。
            self._calibration_resolution = camera_calibration.resolution
            gripper_range = validate_gripper_distance_range(
                min_distance_mm,
                max_distance_mm,
            )
            self._estimator = GripperOpennessEstimator(
                camera_matrix=camera_calibration.camera_matrix,
                distortion_coefficients=(
                    camera_calibration.distortion_coefficients
                ),
                closed_distance_mm=gripper_range.min_distance_mm,
                open_distance_mm=gripper_range.max_distance_mm,
                marker_size_mm=float(
                    self.get_parameter("marker_size_mm").value
                ),
                smoothing_alpha=float(
                    self.get_parameter("smoothing_alpha").value
                ),
                dictionary_name=str(
                    self.get_parameter("dictionary_name").value
                ),
                left_marker_id=int(
                    self.get_parameter(
                        "gripper_range.left_finger_tag_id"
                    ).value
                ),
                right_marker_id=int(
                    self.get_parameter(
                        "gripper_range.right_finger_tag_id"
                    ).value
                ),
                roi_ratios=roi_ratios,
                image_resolution=camera_calibration.resolution,
            )
        except ValueError as error:
            self.get_logger().error(f"加载三维夹爪估计参数失败: {error}")
            raise RuntimeError("三维夹爪估计参数无效") from error

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
        # 带原图时间戳的状态用于 MCAP 录制、离线同步和质量检查。
        self._state_publisher = self.create_publisher(
            GripperState, state_topic, 10
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
            f"订阅原始 RGB 话题 {image_topic}，发布 {openness_topic} "
            f"和 {state_topic}；"
            "输出为无量纲归一化距离，0 表示闭合，1 表示完全张开"
        )
        self.get_logger().info(
            f"夹爪配置 ID {gripper_id}，内部三维距离范围: "
            f"{gripper_range.min_distance_mm:.3f}~"
            f"{gripper_range.max_distance_mm:.3f} mm"
        )
        self.get_logger().info(
            "相机标定分辨率: "
            f"{self._calibration_resolution[0]}x"
            f"{self._calibration_resolution[1]}"
        )

    @staticmethod
    def _default_calibration_path() -> str:
        """返回随 ROS 包安装的默认相机标定文件路径。

        Returns:
            package share 中默认 Kalibr YAML 的绝对路径。
        """
        package_share = get_package_share_directory(
            "fastumi_gripper_estimator"
        )
        return (
            f"{package_share}/config/"
            f"{DEFAULT_CAMERA_CALIBRATION_FILENAME}"
        )

    def _declare_parameters(self) -> None:
        """声明节点支持的全部 ROS 参数及默认值。"""
        self.declare_parameter("image_topic", DEFAULT_IMAGE_TOPIC)
        self.declare_parameter("openness_topic", "/gripper/openness")
        self.declare_parameter("state_topic", "/gripper/state")
        self.declare_parameter(
            "debug_image_topic", "/gripper/openness/debug_image"
        )
        self.declare_parameter("publish_debug_image", False)
        self.declare_parameter("camera_calibration_path", "")
        self.declare_parameter("marker_size_mm", 16.0)
        self.declare_parameter("dictionary_name", "DICT_4X4_50")
        self.declare_parameter(
            "gripper_range.gripper_id", DEFAULT_GRIPPER_ID
        )
        self.declare_parameter(
            "gripper_range.left_finger_tag_id",
            DEFAULT_LEFT_FINGER_TAG_ID,
        )
        self.declare_parameter(
            "gripper_range.right_finger_tag_id",
            DEFAULT_RIGHT_FINGER_TAG_ID,
        )
        self.declare_parameter(
            "gripper_range.min_marker_dist_mm",
            DEFAULT_MIN_MARKER_DIST_MM,
        )
        self.declare_parameter(
            "gripper_range.max_marker_dist_mm",
            DEFAULT_MAX_MARKER_DIST_MM,
        )
        self.declare_parameter("smoothing_alpha", 0.35)
        self.declare_parameter("roi_ratios", [0.15, 0.58, 0.85, 0.82])

    def _image_callback(self, message: Image) -> None:
        """处理一帧图像，在三维估计有效时发布无量纲结果。

        Args:
            message: 原始鱼眼 RGB 图像消息。

        Side Effects:
            发布 `[0,1]` Float32；启用调试参数时还会发布标注图像。
        """
        # 每帧都会发布状态；无效数值使用 NaN，避免下游误用历史值。
        state_message = GripperState()
        state_message.header = message.header
        state_message.raw_openness = math.nan
        state_message.filtered_openness = math.nan
        state_message.marker_distance_mm = math.nan
        state_message.detected_marker_count = 0
        state_message.valid = False
        try:
            # BGR 图像同时供估计算法和调试绘图使用。
            image = self._bridge.imgmsg_to_cv2(
                message, desired_encoding="bgr8"
            )
            estimate = self._estimator.estimate(image)
        except (ValueError, RuntimeError, cv2.error) as error:
            self._state_publisher.publish(state_message)
            self.get_logger().error(
                f"图像转换或三维估计失败: {error}",
                throttle_duration_sec=2.0,
            )
            return
        if estimate is None:
            state_message.detected_marker_count = (
                self._estimator.last_detected_marker_count
            )
            self._state_publisher.publish(state_message)
            self.get_logger().warning(
                "夹爪标记检测或三维 PnP 无效，本帧状态标记为无效",
                throttle_duration_sec=2.0,
            )
            return

        state_message.raw_openness = estimate.raw_openness
        state_message.filtered_openness = estimate.openness
        state_message.marker_distance_mm = estimate.distance_mm
        state_message.detected_marker_count = 2
        state_message.valid = True
        self._state_publisher.publish(state_message)

        # 兼容话题只承载无量纲 [0,1] 归一化距离。
        openness_message = Float32()
        openness_message.data = estimate.openness
        self._openness_publisher.publish(openness_message)
        if self._debug_publisher is not None:
            debug_image = self._draw_debug_image(image, estimate)
            debug_message = self._bridge.cv2_to_imgmsg(
                debug_image, encoding="bgr8"
            )
            debug_message.header = message.header
            self._debug_publisher.publish(debug_message)

    @staticmethod
    def _draw_debug_image(image, estimate: OpennessEstimate):
        """绘制 ROI、标记中心、内部毫米距离和无量纲输出。

        Args:
            image: BGR 原始鱼眼图像。
            estimate: 当前帧估计结果。

        Returns:
            带可视化标注的 BGR 图像副本。
        """
        # 绘图使用副本，确保不修改 cv_bridge 返回的输入缓存。
        debug_image = image.copy()
        x_min, y_min, x_max, y_max = estimate.roi
        left_point = tuple(int(round(value)) for value in estimate.left_center)
        right_point = tuple(
            int(round(value)) for value in estimate.right_center
        )
        cv2.rectangle(
            debug_image,
            (x_min, y_min),
            (x_max, y_max),
            (255, 180, 0),
            2,
        )
        cv2.line(debug_image, left_point, right_point, (0, 255, 0), 3)
        cv2.circle(debug_image, left_point, 8, (0, 0, 255), -1)
        cv2.circle(debug_image, right_point, 8, (255, 0, 0), -1)
        # 毫米值只显示在调试画面；ROS Float32 输出保持无量纲。
        label = (
            f"openness={estimate.openness:.3f} "
            f"distance={estimate.distance_mm:.1f}mm"
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
    # 初始化失败时仍需在 finally 中关闭 rclpy。
    node: Optional[GripperOpennessNode] = None
    try:
        node = GripperOpennessNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
