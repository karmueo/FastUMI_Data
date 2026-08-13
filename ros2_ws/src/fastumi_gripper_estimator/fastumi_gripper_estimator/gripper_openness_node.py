"""订阅原始鱼眼 RGB 图像并发布无量纲夹爪归一化距离。"""

import math
from typing import List, Optional

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from fastumi_gripper_estimator.calibration import (
    load_camera_calibration,
    validate_gripper_distance_range,
)
from fastumi_gripper_estimator.estimator import (
    GripperOpennessEstimator,
    OpennessEstimate,
)
from fastumi_interfaces.msg import GripperState
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


# 默认 ToF 原始 RGB 图像话题。
DEFAULT_IMAGE_TOPIC = "/tof_stereo_camera/rgb/image_raw"
# 默认夹爪编号。
DEFAULT_GRIPPER_ID = 0
# 默认左右手指 ArUco 标记编号。
DEFAULT_LEFT_FINGER_TAG_ID = 0
DEFAULT_RIGHT_FINGER_TAG_ID = 1
# 默认闭合和张开标签中心距离，单位为毫米。
DEFAULT_MIN_MARKER_DIST_MM = 48.168
DEFAULT_MAX_MARKER_DIST_MM = 126.372


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
        # 空参数使用 ToF 包共享目录中的默认标定，也允许传入外部 Kalibr 标定。
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
            tof_stereo_camera package share 中默认 YAML 的绝对路径。
        """
        package_share = get_package_share_directory("tof_stereo_camera")
        return f"{package_share}/config/calibration.yaml"

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

    def _draw_debug_image(
        self, image: np.ndarray, estimate: OpennessEstimate
    ) -> np.ndarray:
        """在原始鱼眼图像绘制 ROI、距离及两枚标记的相机相对位姿。

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
        self._draw_marker_pose(
            debug_image,
            estimate.left_center,
            self._estimator.left_marker_id,
            estimate.left_pose,
        )
        self._draw_marker_pose(
            debug_image,
            estimate.right_center,
            self._estimator.right_marker_id,
            estimate.right_pose,
        )
        cv2.putText(
            debug_image,
            "pose: tag -> RGB optical camera; X=red Y=green Z=blue",
            (x_min, min(debug_image.shape[0] - 10, y_max + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return debug_image

    def _draw_marker_pose(
        self, debug_image: np.ndarray, center: tuple, marker_id: int, pose
    ) -> None:
        """绘制单枚标记的鱼眼坐标轴和相对相机的位姿文本。

        姿态文本采用 XYZ 轴的 roll-pitch-yaw 约定，满足
        ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``。坐标轴用原始鱼眼内参、
        畸变系数投影，长度为实际标记边长。

        Args:
            debug_image: 待绘制的原始 BGR 鱼眼图像。
            center: 标记在原始图像中的像素中心。
            marker_id: 当前配置中对应的 ArUco 标记 ID。
            pose: 标记坐标系相对于 RGB 光学相机的有限 PnP 位姿。
        """
        projected_axes = self._project_marker_axes(pose)
        if projected_axes is not None:
            origin, axis_x, axis_y, axis_z = (
                tuple(int(round(value)) for value in point)
                for point in projected_axes
            )
            # BGR：X 红、Y 绿、Z 蓝，长度与实际标记边长一致。
            cv2.line(debug_image, origin, axis_x, (0, 0, 255), 2)
            cv2.line(debug_image, origin, axis_y, (0, 255, 0), 2)
            cv2.line(debug_image, origin, axis_z, (255, 0, 0), 2)

        rpy_degrees = self._rotation_vector_to_rpy_degrees(
            pose.rotation_vector
        )
        image_height, image_width = debug_image.shape[:2]
        text_x = min(max(5, int(round(center[0])) + 12), image_width - 255)
        text_y = min(max(18, int(round(center[1])) - 24), image_height - 45)
        translation = pose.translation_mm
        labels = (
            f"tag={marker_id}",
            "t=(%.1f,%.1f,%.1f)mm" % translation,
            "rpy=(%.1f,%.1f,%.1f)deg" % rpy_degrees,
        )
        for index, label in enumerate(labels):
            cv2.putText(
                debug_image,
                label,
                (text_x, text_y + index * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _project_marker_axes(self, pose) -> Optional[np.ndarray]:
        """投影标记原点和三根实际长度坐标轴到原始鱼眼图像。

        Args:
            pose: 标记坐标系相对于 RGB 光学相机的有限 PnP 位姿。

        Returns:
            原点、X、Y、Z 的 ``(4, 2)`` 像素坐标；投影失败或含非有限数值
            时返回 ``None``。
        """
        axis_length_mm = self._estimator.marker_size_mm
        object_points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [axis_length_mm, 0.0, 0.0],
                [0.0, axis_length_mm, 0.0],
                [0.0, 0.0, axis_length_mm],
            ],
            dtype=np.float64,
        ).reshape(-1, 1, 3)
        try:
            image_points, _ = cv2.fisheye.projectPoints(
                object_points,
                np.asarray(pose.rotation_vector, dtype=np.float64),
                np.asarray(pose.translation_mm, dtype=np.float64),
                self._estimator.camera_matrix,
                self._estimator.distortion_coefficients,
            )
        except cv2.error:
            return None
        projected_axes = np.asarray(
            image_points, dtype=np.float64
        ).reshape(4, 2)
        if not np.all(np.isfinite(projected_axes)):
            return None
        return projected_axes

    @staticmethod
    def _rotation_vector_to_rpy_degrees(rotation_vector: tuple) -> tuple:
        """按 XYZ roll-pitch-yaw 约定转换 Rodrigues 旋转向量。

        该约定满足 ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``，返回值单位为度。

        Args:
            rotation_vector: 三维 Rodrigues 旋转向量。

        Returns:
            roll、pitch、yaw 的角度元组。
        """
        rotation_matrix, _ = cv2.Rodrigues(
            np.asarray(rotation_vector, dtype=np.float64)
        )
        cosine_pitch = float(
            np.hypot(rotation_matrix[0, 0], rotation_matrix[1, 0])
        )
        if cosine_pitch > 1e-6:
            roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = math.atan2(-rotation_matrix[2, 0], cosine_pitch)
            yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = math.atan2(-rotation_matrix[2, 0], cosine_pitch)
            yaw = 0.0
        return tuple(math.degrees(value) for value in (roll, pitch, yaw))


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
