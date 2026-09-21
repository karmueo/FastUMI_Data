"""ROS 2 夹爪范围标定节点，支持图像话题和本地视频输入。"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
from cv_bridge import CvBridge
from gripper_openness.calibration import (
    load_camera_calibration,
    RangeAccumulator,
    write_gripper_calibration,
)
from gripper_openness.vision import GripperVision, VisionResult
from gripper_openness.visualization import draw_vision_result
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger


class GripperCalibrationNode(Node):
    """采集夹爪开闭范围并保存为独立 YAML 标定文件。"""

    def __init__(self, *, parameter_overrides: Optional[Sequence] = None) -> None:
        """加载参数、创建输入订阅器和标定控制服务。"""
        super().__init__(
            "gripper_calibration",
            parameter_overrides=_parameter_overrides(parameter_overrides),
        )
        self._declare_parameters()
        self._source_mode = str(self.get_parameter("source_mode").value).lower()
        if self._source_mode not in ("topic", "video"):
            raise RuntimeError("source_mode 必须是 topic 或 video")
        camera_path = str(self.get_parameter("camera_calibration_path").value)
        try:
            self._camera_calibration = load_camera_calibration(camera_path)
            self._marker_size_mm = float(self.get_parameter("marker_size_mm").value)
            self._dictionary_name = str(self.get_parameter("dictionary_name").value)
            self._left_id = int(self.get_parameter("left_finger_tag_id").value)
            self._right_id = int(self.get_parameter("right_finger_tag_id").value)
            self._vision = GripperVision(
                self._camera_calibration,
                marker_size_mm=self._marker_size_mm,
                dictionary_name=self._dictionary_name,
                left_marker_id=self._left_id,
                right_marker_id=self._right_id,
            )
        except (ValueError, TypeError, cv2.error) as error:
            self.get_logger().error(f"加载夹爪标定参数失败: {error}")
            raise RuntimeError("夹爪标定参数无效") from error
        self._bridge = CvBridge()
        self._accumulator = RangeAccumulator()
        self._sampling = self._source_mode == "video"
        self._sampling_frame_count = 0
        self._pair_detection_count = 0
        self._last_result: Optional[VisionResult] = None
        self._video_capture: Optional[cv2.VideoCapture] = None
        self._video_timer = None
        self._debug_publisher = None
        self._setup_ros_interfaces()
        if self._source_mode == "video":
            self._open_video_source()
        self.get_logger().info(
            f"夹爪标定节点已启动，输入模式={self._source_mode}，"
            f"分辨率={self._camera_calibration.resolution[0]}x"
            f"{self._camera_calibration.resolution[1]}，"
            f"图像话题={self.get_parameter('image_topic').value}"
        )

    def _declare_parameters(self) -> None:
        """声明标定节点的 ROS 参数。"""
        self.declare_parameter("image_topic", "/umi_camera/image_raw")
        self.declare_parameter("source_mode", "topic")
        self.declare_parameter("video_path", "")
        self.declare_parameter("video_fps", 30.0)
        self.declare_parameter("camera_calibration_path", "")
        self.declare_parameter("output_path", "")
        self.declare_parameter("overwrite", False)
        self.declare_parameter("publish_debug_image", False)
        self.declare_parameter("debug_image_topic", "/gripper/calibration/debug_image")
        self.declare_parameter("marker_size_mm", 16.0)
        self.declare_parameter("dictionary_name", "DICT_4X4_50")
        self.declare_parameter("left_finger_tag_id", 0)
        self.declare_parameter("right_finger_tag_id", 1)
        self.declare_parameter("roi_ratios", [0.0, 0.0, 1.0, 1.0])

    def _setup_ros_interfaces(self) -> None:
        """创建图像输入、调试图像发布器和控制服务。"""
        image_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._image_subscription = None
        if self._source_mode == "topic":
            self._image_subscription = self.create_subscription(
                Image,
                str(self.get_parameter("image_topic").value),
                self._image_callback,
                image_qos,
            )
        if bool(self.get_parameter("publish_debug_image").value):
            self._debug_publisher = self.create_publisher(
                Image,
                str(self.get_parameter("debug_image_topic").value),
                image_qos,
            )
        self._start_service = self.create_service(Trigger, "~/start", self._start)
        self._save_service = self.create_service(Trigger, "~/save", self._save)
        self._reset_service = self.create_service(Trigger, "~/reset", self._reset)

    def _open_video_source(self) -> None:
        """打开本地视频并创建逐帧定时器。"""
        video_path = str(self.get_parameter("video_path").value)
        if not video_path.strip():
            raise RuntimeError("source_mode=video 时必须设置 video_path")
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        self._video_capture = capture
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        configured_fps = float(self.get_parameter("video_fps").value)
        fps = source_fps if math.isfinite(source_fps) and source_fps > 0.0 else configured_fps
        if not math.isfinite(fps) or fps <= 0.0:
            fps = 30.0
        self._video_timer = self.create_timer(1.0 / fps, self._video_tick)

    def _image_callback(self, message: Image) -> None:
        """处理一帧 ROS 图像；只有 start 后才写入标定样本。"""
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as error:
            self.get_logger().error(f"标定图像转换失败: {error}", throttle_duration_sec=2.0)
            return
        self._process_frame(image, message.header)

    def _video_tick(self) -> None:
        """读取视频下一帧，视频结束时自动保存并停止。"""
        if self._video_capture is None:
            return
        success, image = self._video_capture.read()
        if not success:
            self.get_logger().info("视频读取完成，开始保存夹爪范围标定")
            self._finish_video()
            return
        self._process_frame(image, None)

    def _process_frame(self, image, header) -> None:
        """执行检测、累积有效帧并按需发布调试图像。"""
        try:
            result = self._vision.detect(
                image,
                self._configured_roi(image.shape[1], image.shape[0]),
            )
        except (ValueError, cv2.error) as error:
            self.get_logger().error(f"标定帧处理失败: {error}", throttle_duration_sec=2.0)
            return
        self._last_result = result
        if self._sampling:
            self._sampling_frame_count += 1
            if result.detected_marker_count == 2:
                self._pair_detection_count += 1
            if result.valid:
                self._accumulator.add(
                    result.distance_mm,
                    result.left_corners,
                    result.right_corners,
                    self._camera_calibration.resolution,
                )
        if self._debug_publisher is not None:
            range_values = None
            if self._accumulator.distances_mm:
                range_values = (
                    min(self._accumulator.distances_mm),
                    max(self._accumulator.distances_mm),
                )
            label = "sampling" if self._sampling else "waiting"
            debug = draw_vision_result(
                image,
                result,
                label,
                range_values,
                self._camera_calibration,
                self._marker_size_mm,
            )
            debug_message = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            if header is not None:
                debug_message.header = header
            self._debug_publisher.publish(debug_message)

    def _configured_roi(self, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
        """将可选的比例 ROI 转换为原图像素边界。"""
        values = tuple(float(value) for value in self.get_parameter("roi_ratios").value)
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise ValueError("roi_ratios 必须包含四个有限数")
        x_min, y_min, x_max, y_max = values
        if (x_min, y_min, x_max, y_max) == (0.0, 0.0, 1.0, 1.0):
            return None
        if not (0.0 <= x_min < x_max <= 1.0 and 0.0 <= y_min < y_max <= 1.0):
            raise ValueError("roi_ratios 必须位于 [0,1] 且具有正面积")
        return (
            int(round(x_min * width)),
            int(round(y_min * height)),
            int(round(x_max * width)),
            int(round(y_max * height)),
        )

    def _start(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """清空旧样本并开始实时采样。"""
        if self._sampling:
            response.success = False
            response.message = "标定采样已经在进行"
            return response
        self._accumulator.reset()
        self._sampling_frame_count = 0
        self._pair_detection_count = 0
        self._sampling = True
        response.success = True
        response.message = "已开始夹爪范围采样，请覆盖完整开闭行程"
        self.get_logger().info(response.message)
        return response

    def _save(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """校验有效样本并保存 YAML 标定文件。"""
        try:
            calibration = self._build_calibration()
            output_path = str(self.get_parameter("output_path").value)
            write_gripper_calibration(
                output_path,
                calibration,
                overwrite=bool(self.get_parameter("overwrite").value),
            )
        except (ValueError, FileExistsError, OSError) as error:
            response.success = False
            response.message = f"标定保存失败: {error}"
            self.get_logger().error(response.message)
            return response
        self._sampling = False
        response.success = True
        response.message = (
            f"标定已保存到 {output_path}，有效帧 {calibration.total_valid_frames}，"
            f"范围 {calibration.min_marker_dist_mm:.3f}.."
            f"{calibration.max_marker_dist_mm:.3f} mm"
        )
        self.get_logger().info(response.message)
        return response

    def _reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """清空当前样本并停止实时采样。"""
        self._accumulator.reset()
        self._sampling_frame_count = 0
        self._pair_detection_count = 0
        self._sampling = False
        response.success = True
        response.message = "已清空夹爪范围采样"
        return response

    def _build_calibration(self):
        """从采样器生成待写入的完整标定对象。"""
        if self._accumulator.count == 0:
            if self._sampling_frame_count == 0 and self._source_mode == "topic":
                image_topic = str(self.get_parameter("image_topic").value)
                raise ValueError(f"未收到标定图像，请检查图像话题 {image_topic}")
            if self._pair_detection_count == 0:
                raise ValueError(
                    f"已处理 {self._sampling_frame_count} 帧，但未同时检测到 "
                    f"ArUco ID {self._left_id} 和 {self._right_id}"
                )
            raise ValueError(
                f"已处理 {self._sampling_frame_count} 帧，其中 "
                f"{self._pair_detection_count} 帧检测到双 ArUco，"
                "但双码位姿计算均无效"
            )
        return self._accumulator.build(
            marker_size_mm=self._marker_size_mm,
            dictionary_name=self._dictionary_name,
            left_finger_tag_id=self._left_id,
            right_finger_tag_id=self._right_id,
        )

    def _finish_video(self) -> None:
        """视频结束时保存结果并结束节点定时器。"""
        self._sampling = False
        if self._video_timer is not None:
            self._video_timer.cancel()
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None
        response = self._save(Trigger.Request(), Trigger.Response())
        if not response.success:
            self.get_logger().error(response.message)
        import rclpy

        if rclpy.ok():
            rclpy.shutdown()

    def destroy_node(self):
        """释放视频资源后销毁 ROS 节点。"""
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None
        return super().destroy_node()


def main(args: Optional[List[str]] = None) -> None:
    """启动夹爪范围标定节点。"""
    import rclpy

    rclpy.init(args=args)
    node = None
    try:
        node = GripperCalibrationNode()
        rclpy.spin(node)
    except (RuntimeError, ValueError) as error:
        print(f"夹爪标定节点启动失败: {error}")
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
