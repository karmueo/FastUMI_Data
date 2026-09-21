"""订阅遥操 ROS 话题，并在后台编码图像和保存采集 episode。"""

from __future__ import annotations

from io import BytesIO
import queue
import signal
import threading
import time

from nav_msgs.msg import Odometry
import numpy as np
from PIL import Image as PillowImage
import rclpy
from rclpy.node import Node
from rclpy.validate_full_topic_name import validate_full_topic_name
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.signals import SignalHandlerOptions
from rm_ros_interfaces.msg import Jointpos
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

from tracker_teleoperated.recorder_core import (
    JOINT_NAMES, EpisodeBuffer, RecordingSession,
)
from tracker_teleoperated.recorder_storage import write_episode
from tracker_teleoperated.recording_paths import output_directory


# 记录节点默认只订阅安装在机械臂末端的相机。
DEFAULT_IMAGE_TOPIC = "/wrist_camera/image_raw"
# 后台编码队列的最大待处理图像数，避免写盘阻塞导致内存持续增长。
MAX_QUEUED_IMAGES = 8


def image_to_jpeg(message: Image) -> bytes:
    """按编码和行跨度将 ROS 原始图像转成 JPEG。"""
    encoding = message.encoding.lower()
    formats = {
        "bgr8": ("RGB", "BGR", 3),
        "rgb8": ("RGB", "RGB", 3),
        "mono8": ("L", "L", 1),
    }
    if encoding not in formats:
        raise ValueError(f"不支持的图像编码: {message.encoding}")
    mode, raw_mode, channels = formats[encoding]
    width, height, step = int(message.width), int(message.height), int(message.step)
    if width <= 0 or height <= 0 or step < width * channels:
        raise ValueError("图像尺寸或行跨度无效")
    data = bytes(message.data)
    if len(data) < step * height:
        raise ValueError("图像字节数小于行跨度要求")
    image = PillowImage.frombytes(
        mode, (width, height), data[:step * height], "raw", raw_mode, step
    )
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()


class TrackerTeleopRecorder(Node):
    """独立管理录制状态、话题采样和后台落盘。"""

    def __init__(self, parameter_overrides=None) -> None:
        """读取配置，建立话题订阅并启动图像与保存工作线程。"""
        super().__init__(
            "tracker_teleop_recorder", parameter_overrides=parameter_overrides
        )
        defaults = {
            "joint_state_topic": "/joint_states",
            "joint_action_topic": "/rm_driver/movej_canfd_cmd",
            "gripper_state_topic": "/motion_control/gripper_state",
            "gripper_action_topic": "/motion_control/gripper_command",
            "tracker_odom_topic": "/vive_tracker/odom",
            "image_topic": DEFAULT_IMAGE_TOPIC,
            "dataset_root": "dataset/h5dy_data",
            "dir_name": "test",
            "name": "default_test",
            "record_camera": True,
            "camera_switch_locked": False,
            "camera_fps": 30,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
        self._output_root = output_directory(
            self.get_parameter("dataset_root").value,
            self.get_parameter("dir_name").value,
            self.get_parameter("name").value,
        )
        self._camera_fps = int(self.get_parameter("camera_fps").value)
        if self._camera_fps <= 0:
            raise ValueError("camera_fps 必须大于零")
        self._session = RecordingSession()
        # 最近一次写盘错误供停止服务查询，避免把失败误报为保存成功。
        self._save_error = ""
        # 保证信号与调用者重复关闭时只排入一次退出屏障。
        self._closed = False
        self._queue: queue.Queue = queue.Queue()
        self._queued_images = 0
        self._queue_lock = threading.Lock()
        self._last_drop_warning = 0.0
        self._worker = threading.Thread(
            target=self._work_loop, name="tracker-recorder-writer", daemon=False
        )
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._status = self.create_publisher(
            String, "/tracker_teleoperated/record_status", status_qos
        )
        self._state_publisher = self.create_publisher(
            String, "/tracker_teleoperated/record_state", status_qos
        )
        self.create_service(
            Trigger, "/tracker_teleoperated/stop_recording", self._stop_recording
        )
        self.create_timer(0.5, self._publish_state)
        self.create_subscription(
            String, "/tracker_teleoperated/record_command", self._command, 10
        )
        self.create_subscription(
            JointState, str(self.get_parameter("joint_state_topic").value),
            self._joint_state, 10,
        )
        self.create_subscription(
            Jointpos, str(self.get_parameter("joint_action_topic").value),
            self._joint_action, 10,
        )
        self.create_subscription(
            Float32, str(self.get_parameter("gripper_state_topic").value),
            self._gripper_state, 10,
        )
        self.create_subscription(
            Float32, str(self.get_parameter("gripper_action_topic").value),
            self._gripper_action, 10,
        )
        self.create_subscription(
            Odometry, str(self.get_parameter("tracker_odom_topic").value),
            self._tracker_pose, 10,
        )
        # 订阅代次阻止销毁订阅后仍排队的旧回调进入新录制。
        self._image_generation = 0
        # 设备事务完成前拍摄的排队图像不得进入下一轮录制。
        self._image_min_stamp = 0
        self._image_subscription = None
        if bool(self.get_parameter("record_camera").value):
            self._image_subscription = self.create_subscription(
                Image, str(self.get_parameter("image_topic").value),
                lambda message: self._source_image(message, 0), qos_profile_sensor_data,
            )
        self.add_on_set_parameters_callback(self._change_image_topic)
        self._worker.start()
        self._publish_status(f"记录节点就绪，输出目录: {self._output_root}")
        self._publish_state()

    def _source_image(self, message, generation):
        """仅将当前订阅代次的图像交给编码队列。"""
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if generation == self._image_generation and stamp >= self._image_min_stamp:
            self._image(message)

    def _change_image_topic(self, parameters):
        """空闲时原子替换图像订阅，创建失败保留原订阅与参数。"""
        locks = [p for p in parameters if p.name == 'camera_switch_locked']
        if locks:
            if len(parameters) != 1 or not isinstance(locks[0].value, bool):
                return SetParametersResult(successful=False, reason='录制锁必须单独更新为布尔值')
            if locks[0].value and self._session.snapshot() != 'idle':
                return SetParametersResult(successful=False, reason='仅空闲时允许锁定录制')
            if not locks[0].value:
                self._image_min_stamp = self.get_clock().now().nanoseconds
            return SetParametersResult(successful=True)
        topics = [p for p in parameters if p.name == "image_topic"]
        if not topics:
            return SetParametersResult(successful=True)
        if len(parameters) != 1:
            return SetParametersResult(successful=False, reason="image_topic 必须单独更新")
        topic = topics[0].value
        try:
            if not isinstance(topic, str):
                raise ValueError("图像话题必须为字符串")
            validate_full_topic_name(topic)
            if topic == self.get_parameter("image_topic").value:
                return SetParametersResult(successful=True)
            if self._session.snapshot() != "idle":
                raise ValueError("仅空闲时允许切换末端视频源")
            generation = self._image_generation + 1
            subscription = None
            if bool(self.get_parameter("record_camera").value):
                subscription = self.create_subscription(
                    Image, topic, lambda m: self._source_image(m, generation),
                    qos_profile_sensor_data,
                )
        except Exception as error:
            return SetParametersResult(successful=False, reason=str(error))
        previous = self._image_subscription
        self._image_subscription = subscription
        self._image_generation = generation
        if previous is not None:
            self.destroy_subscription(previous)
        return SetParametersResult(successful=True)

    def _publish_state(self) -> None:
        """周期发布机器可读状态，保存完成后由主线程报告。"""
        self._state_publisher.publish(String(data=self._session.snapshot()))

    def _stop_recording(self, _request, response):
        """幂等停止当前录制；空闲时返回最近保存结果。"""
        self._command(String(data="stop"))
        # 先取状态，保证观察到 idle 时也能看到后台已写入的错误结果。
        state = self._session.snapshot()
        response.success = not bool(self._save_error)
        response.message = self._save_error or state
        return response

    def _now(self) -> float:
        """返回当前 ROS 时间的浮点秒。"""
        return self.get_clock().now().nanoseconds * 1e-9

    def _stamp(self, stamp) -> float:
        """优先使用有效消息头时间戳。"""
        value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return value if value > 0 else self._now()

    def _publish_status(self, text: str) -> None:
        """将录制结果同时发送到日志和 RViz 面板。"""
        self.get_logger().info(text)
        self._status.publish(String(data=text))

    def _command(self, message: String) -> None:
        """响应 a 启停和 b 取消，保存工作交由后台线程完成。"""
        key = message.data.lower()
        if key not in ("a", "b", "stop"):
            return
        if key == 'a' and self._session.snapshot() == 'idle' and self.get_parameter('camera_switch_locked').value:
            self._publish_status('正在切换相机设备，暂时不能开始录制')
            return
        result, episode, generation = self._session.command(key, self._now())
        if result == "started":
            self._save_error = ""
            self._publish_status("录制已开始；按 a 保存，按 b 取消")
        elif result == "discarded":
            self._publish_status("本轮录制已取消")
        elif result == "stopped":
            self._publish_status("录制已停止，正在保存")
            self._queue.put(("save", generation, episode))
        elif result == "busy":
            if key != "stop":
                self._publish_status("正在保存，请等待本轮完成")
        self._publish_state()

    def _joint_state(self, message: JointState) -> None:
        """按固定关节顺序记录反馈位姿。"""
        if len(message.name) != len(message.position):
            return
        positions = dict(zip(message.name, message.position))
        if any(name not in positions for name in JOINT_NAMES):
            return
        values = np.asarray([positions[name] for name in JOINT_NAMES], dtype=np.float32)
        if np.isfinite(values).all():
            self._session.add(
                "joint_state", self._stamp(message.header.stamp), values.tolist()
            )

    def _joint_action(self, message: Jointpos) -> None:
        """记录实际下发的七轴 CANFD 指令。"""
        values = np.asarray(message.joint, dtype=np.float32)
        if message.dof == 7 and values.shape == (7,) and np.isfinite(values).all():
            self._session.add("joint_action", self._now(), values.tolist())

    def _gripper_action(self, message: Float32) -> None:
        """缓存最新归一化夹爪目标。"""
        value = float(message.data)
        if np.isfinite(value):
            self._session.add_gripper_action(float(np.clip(value, 0.0, 1.0)))

    def _gripper_state(self, message: Float32) -> None:
        """按夹爪反馈时间轴记录反馈及保持的最新目标。"""
        value = float(message.data)
        if np.isfinite(value):
            self._session.add_gripper_state(
                self._now(), float(np.clip(value, 0.0, 1.0))
            )

    def _tracker_pose(self, message: Odometry) -> None:
        """保存 Tracker odom 中的位置、xyzw 四元数与 frame_id。"""
        pose = message.pose.pose
        position = [pose.position.x, pose.position.y, pose.position.z]
        orientation = [
            pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w,
        ]
        if (
            not np.isfinite(position + orientation).all()
            or not np.linalg.norm(orientation)
        ):
            return
        self._session.add_tracker(
            self._stamp(message.header.stamp), position,
            orientation, message.header.frame_id,
        )

    def _image(self, message: Image) -> None:
        """快速排队原始图像，由后台线程执行 JPEG 编码。"""
        accepting, generation = self._session.accepts_image()
        if not accepting:
            return
        with self._queue_lock:
            full = self._queued_images >= MAX_QUEUED_IMAGES
            if not full:
                self._queued_images += 1
        if full:
            now = time.monotonic()
            if now - self._last_drop_warning >= 5.0:
                self.get_logger().warning("图像编码积压，已跳过新图像帧")
                self._last_drop_warning = now
            return
        self._queue.put(
            ("image", generation, self._stamp(message.header.stamp), message)
        )

    def _work_loop(self) -> None:
        """按入队顺序处理图像与保存屏障。"""
        while True:
            task = self._queue.get()
            kind = task[0]
            if kind == "shutdown":
                return
            if kind == "image":
                _, generation, timestamp, message = task
                try:
                    jpeg = image_to_jpeg(message)
                    self._session.append_image(generation, timestamp, jpeg)
                except Exception as error:
                    self.get_logger().warning(f"跳过无效图像: {error}")
                finally:
                    with self._queue_lock:
                        self._queued_images -= 1
                continue
            _, _generation, episode = task
            assert isinstance(episode, EpisodeBuffer)
            try:
                path, actions, frames, elapsed = write_episode(
                    self._output_root, episode, self._camera_fps
                )
                self._publish_status(
                    f"已保存 {path}；动作 {actions} 条，视频 {frames} 帧，"
                    f"耗时 {elapsed:.1f} 秒"
                    f"{'；本轮无相机帧' if frames == 0 else ''}"
                )
            except Exception as error:
                self._save_error = str(error)
                self._publish_status(f"保存失败，本轮已丢弃: {error}")
            finally:
                episode.close()
                self._session.finish_save()

    def close(self) -> None:
        """自动保存进行中的录制，并等待图像队列与写盘完成。"""
        if self._closed:
            return
        self._closed = True
        self._command(String(data="stop"))
        self._queue.put(("shutdown",))
        self._worker.join()
        self._session.close()


def main(args=None) -> None:
    """启动记录节点，捕捉退出信号并等待后台写盘。"""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    stop = threading.Event()
    node = TrackerTeleopRecorder()

    def request_stop(_signum, _frame) -> None:
        """通知主循环在下一次 spin 后有序退出。"""
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
