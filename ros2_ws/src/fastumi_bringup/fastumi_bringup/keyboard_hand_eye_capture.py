"""键盘驱动 RM75 带腕部鱼眼相机围绕标定板自动采集眼在手上手眼标定 MCAP。

标定板为 Kalibr target YAML：AprilGrid（tag36h11）或棋盘格，例如
config/calibration/april_6x6.yaml、checkerboard_11x8.yaml。

在 hardware.launch.py 启动后于第二个终端运行：
    ros2 run fastumi_bringup keyboard_hand_eye_capture --ros-args \
        -p board_config:=/abs/FastUMI_Data/config/calibration/april_6x6.yaml

按键：r 设参考位姿 | a 开始/继续自动采集 | 空格 急停并暂停 | c 当前位姿补采 |
      u 撤销上一样本 | m 重新估计夹爪掩膜 | h 回参考位姿 | q 保存并退出

流程：r 记录能看到整块板的参考位姿 → a 先在参考位姿附近做 8 个小幅自举运动（估计夹爪
掩膜和粗外参）→ 按“看向标定板”策略随机规划位姿，静止后采样，直到 num_samples 个。
每个样本写入 MCAP：原始鱼眼 JPEG、^base T_Link7（FK，米/四元数 xyzw）、关节和检测元数据，
四条消息使用相同的 bag 时间戳（图像接收时间）以便离线配对。
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import datetime
import json
import math
import os
from pathlib import Path
import sys
import time

from ament_index_python.packages import get_package_share_directory
import cv2
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import serialize_message
from rm_ros_interfaces.msg import Movej
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Bool, Empty, Float32, String

from fastumi_bringup.hand_eye.board import (
    load_board_spec, load_fisheye_camera, make_detector,
)
from fastumi_bringup.hand_eye.estimate import (
    diversity_stats, estimate_board_in_base, estimate_flange_from_camera, load_prior,
)
from fastumi_bringup.hand_eye.kinematics import JOINT_NAMES, ArmKinematics
from fastumi_bringup.hand_eye.occlusion import (
    check_board_view, estimate_gripper_mask, load_mask,
)
from fastumi_bringup.hand_eye.sampler import (
    PlanContext, SamplerSettings, aim_for_mask, bootstrap_targets, check_reachable, plan_next,
)
from fastumi_bringup.hand_eye.transforms import invert, quaternion_from_matrix
from fastumi_bringup.terminal import TerminalKeyReader


FEEDBACK_FRESHNESS_SECONDS = 0.5
IMAGE_FRESHNESS_SECONDS = 0.5
GRIPPER_TOLERANCE = 0.05
STATIONARY_TOLERANCE_RAD = 1e-4
MASK_FRAME_BUFFER = 12
STOP_RESULT_GRACE_SECONDS = 3.0
# 自举后粗外参下板位姿离散度超过该值时提示外参不可信（毫米）。
BOARD_SPREAD_WARNING_MM = 30.0

SAMPLE_TOPICS = {
    "sample_image": ("/hand_eye/sample_image", "sensor_msgs/msg/CompressedImage"),
    "sample_flange_pose": ("/hand_eye/sample_flange_pose", "geometry_msgs/msg/PoseStamped"),
    "sample_joints": ("/hand_eye/sample_joints", "sensor_msgs/msg/JointState"),
    "sample_meta": ("/hand_eye/sample_meta", "std_msgs/msg/String"),
    "sample_rejected": ("/hand_eye/sample_rejected", "std_msgs/msg/String"),
    "gripper_mask": ("/hand_eye/gripper_mask", "sensor_msgs/msg/Image"),
    "session": ("/hand_eye/session", "std_msgs/msg/String"),
    "joint_states": ("/joint_states", "sensor_msgs/msg/JointState"),
}

HELP_TEXT = ("按键：r 设参考位姿 | a 开始/继续自动采集 | 空格 急停暂停 | c 补采 | "
             "u 撤销 | m 重估掩膜 | h 回参考位姿 | q 保存退出")


@dataclass
class Capture:
    """一次静止位姿下的采样：原始图像消息、解码图、检测、法兰位姿。"""

    image_message: object
    image_bgr: np.ndarray
    receive_ns: int
    joints: np.ndarray
    base_from_flange: np.ndarray
    observation: object


@dataclass
class Motion:
    """进行中的 MoveJ：purpose 为 bootstrap / sample / manual / home。"""

    joints: np.ndarray
    sent_at: float
    purpose: str


def decode_image(message):
    """CompressedImage（JPEG/PNG）或 bgr8/rgb8/mono8 Image 转 BGR uint8。"""
    if isinstance(message, CompressedImage):
        image = cv2.imdecode(np.frombuffer(bytes(message.data), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("压缩图像解码失败")
        return image
    data = np.frombuffer(bytes(message.data), np.uint8)
    if message.encoding in ("bgr8", "rgb8"):
        image = data.reshape(message.height, message.step)[:, :message.width * 3]
        image = image.reshape(message.height, message.width, 3)
        return image[:, :, ::-1].copy() if message.encoding == "rgb8" else image.copy()
    if message.encoding == "mono8":
        image = data.reshape(message.height, message.step)[:, :message.width]
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"不支持的图像编码 {message.encoding}")


def as_compressed(message, image_bgr):
    """样本图像统一存为 CompressedImage；原始 Image 无损编码为 PNG。"""
    if isinstance(message, CompressedImage):
        return message
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("PNG 编码失败")
    return CompressedImage(header=message.header, format="png", data=encoded.tobytes())


def _share_file(package, relative, parameter_name):
    """已安装包 share 目录下的默认文件；包未构建时提示改用显式参数。"""
    try:
        return str(Path(get_package_share_directory(package)) / relative)
    except LookupError as error:
        raise ValueError(f"未找到已安装的 {package}；请构建该包或设置 {parameter_name}") from error


def default_dataset_root():
    """优先 DATASET_ROOT 环境变量，否则使用录制器默认根目录。"""
    if os.environ.get("DATASET_ROOT"):
        return Path(os.environ["DATASET_ROOT"])
    from fastumi_recorder.catalog import default_dataset_root as recorder_root
    return Path(recorder_root())


class HandEyeCapture(Node):
    """手眼标定自动采集节点；非 ROS 逻辑在 fastumi_bringup.hand_eye 中。"""

    def __init__(self, **node_kwargs):
        super().__init__("keyboard_hand_eye_capture", **node_kwargs)
        self._declare_parameters()
        parameter = lambda name: self.get_parameter(name).value  # noqa: E731

        board_config = parameter("board_config")
        if not board_config:
            raise ValueError(
                "必须提供 board_config（Kalibr target YAML，如 config/calibration/april_6x6.yaml）")
        self.spec = load_board_spec(board_config)
        camera_config = parameter("camera_config") or _share_file(
            "fastumi_usb_camera", "config/EndEffectorCalib.yaml", "camera_config")
        self.camera = load_fisheye_camera(camera_config)
        self.camera_config = camera_config
        self.detector = make_detector(self.spec, self.camera)
        urdf_path = parameter("urdf_path") or _share_file(
            "fastumi_bringup", "assets/rm_75_kinematic.urdf", "urdf_path")
        self.urdf_path = urdf_path
        self.kinematics = ArmKinematics.from_urdf(urdf_path)

        self.prior = None
        if parameter("prior_path"):
            self.prior = load_prior(parameter("prior_path"), parameter("prior_convention"))
        self.mask = None
        if parameter("gripper_mask_path"):
            self.mask = load_mask(parameter("gripper_mask_path"),
                                  self.camera.width, self.camera.height)

        self.num_samples = int(parameter("num_samples"))
        self.speed = int(parameter("speed"))
        if not 1 <= self.speed <= 50:
            raise ValueError("speed 需在 1..50（百分比）")
        self.gripper_openness = float(parameter("gripper_openness"))
        self.require_gripper = bool(parameter("require_gripper"))
        self.settle_seconds = float(parameter("settle_seconds"))
        self.motion_timeout = float(parameter("motion_timeout"))
        self.max_reprojection_px = float(parameter("max_reprojection_px"))
        self.min_corner_ratio = float(parameter("min_corner_ratio"))
        self.dry_run = bool(parameter("dry_run"))
        self.record_joint_stream = bool(parameter("record_joint_stream"))
        self.max_attempts = int(parameter("max_attempts_factor")) * self.num_samples
        self.settings = SamplerSettings(
            distance_m=(parameter("distance_min_m"), parameter("distance_max_m")),
            tilt_deg=(parameter("tilt_min_deg"), parameter("tilt_max_deg")),
            roll_deg=parameter("roll_max_deg"),
            workspace_min=tuple(parameter("workspace_min")),
            workspace_max=tuple(parameter("workspace_max")),
            max_joint_step_rad=parameter("max_joint_step_rad"),
        )
        self.rng = np.random.default_rng(
            None if parameter("random_seed") < 0 else parameter("random_seed"))
        self._update_aim()

        root = Path(parameter("dataset_root") or default_dataset_root()).expanduser()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = root / "hand_eye" / f"session_{stamp}"
        self.debug_dir = self.session_dir / "debug"
        self.writer = None

        # 反馈状态（time.monotonic 秒）。
        self.joints = None
        self.joint_history = deque()
        self.last_feedback_at = None
        self.last_valid_feedback_at = None
        self.gripper_state = None
        self.gripper_state_at = None
        self.latest_image = None  # (message, receive_ns, monotonic)

        # 采集状态。
        self.phase = "idle"  # idle / bootstrap / auto / paused / locked / done
        self.paused_phase = None
        self.motion = None
        self.capture_purpose = None
        self.settled_at = None
        self.reference = None
        self.bootstrap_queue = []
        self.bootstrap_captures = []
        self.mask_frames = deque(maxlen=MASK_FRAME_BUFFER)
        self.samples = []  # (index, Capture)
        # 粗外参估计池：所有位姿可靠的帧（含因夹爪重叠未保存的），遮挡只影响最终样本质量。
        self.pose_pool = []  # (^base T_flange, ^cam T_board)
        self.next_index = 0
        self.attempts = 0
        self.flange_from_camera = self.prior
        self.base_from_board = None
        self.planner = ThreadPoolExecutor(max_workers=1)
        self.plan_future = None
        self.finished = False

        self.movej_publisher = self.create_publisher(Movej, "/rm_driver/movej_cmd", 10)
        self.stop_publisher = self.create_publisher(Empty, "/rm_driver/move_stop_cmd", 10)
        self.gripper_publisher = self.create_publisher(
            Float32, "/motion_control/gripper_command", 10)
        self.create_subscription(JointState, "/joint_states", self._joint_state_callback, 50)
        self.create_subscription(
            Bool, "/rm_driver/udp_feedback_valid", self._feedback_valid_callback, 10)
        self.create_subscription(Bool, "/rm_driver/movej_result", self._result_callback, 10)
        self.create_subscription(
            Float32, "/motion_control/gripper_state", self._gripper_callback, 10)
        image_topic = parameter("image_topic")
        image_type = CompressedImage if image_topic.endswith("/compressed") else Image
        self.create_subscription(
            image_type, image_topic, self._image_callback, qos_profile_sensor_data)

        self.get_logger().info(
            f"手眼采集就绪：板 {self.spec.kind}（{self.spec.corner_count} 个特征点，"
            f"{board_config}），图像 {image_topic}，输出 {self.session_dir}"
            + ("（dry_run：只规划不运动）" if self.dry_run else ""))
        self.get_logger().info(HELP_TEXT)

    def _declare_parameters(self):
        """声明所有 ROS 参数及默认值（长度米、角度度/弧度见名称）。"""
        defaults = {
            "board_config": "",
            "camera_config": "",
            "urdf_path": "",
            "prior_path": "",
            "prior_convention": "cam_from_flange",
            "gripper_mask_path": "",
            "image_topic": "/wrist_camera/image_raw/compressed",
            "dataset_root": "",
            "num_samples": 25,
            "speed": 15,
            "gripper_openness": 1.0,
            "require_gripper": True,
            "settle_seconds": 0.8,
            "motion_timeout": 60.0,
            "max_reprojection_px": 1.0,
            "min_corner_ratio": 0.8,
            "dry_run": False,
            "record_joint_stream": True,
            "max_attempts_factor": 3,
            "distance_min_m": 0.25,
            "distance_max_m": 0.45,
            "tilt_min_deg": 10.0,
            "tilt_max_deg": 40.0,
            "roll_max_deg": 45.0,
            "workspace_min": [-0.6, -0.5, 0.16],
            "workspace_max": [0.6, 0.5, 0.60],
            "max_joint_step_rad": 1.2,
            "random_seed": -1,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    # ---------------------------------------------------------------- 回调

    def _joint_state_callback(self, message):
        """按名称读取七关节弧度；同时写入 MCAP 诊断流。"""
        if len(message.name) != len(message.position):
            return
        by_name = dict(zip(message.name, message.position))
        if any(name not in by_name for name in JOINT_NAMES):
            return
        joints = np.array([float(by_name[name]) for name in JOINT_NAMES])
        if not np.all(np.isfinite(joints)):
            return
        now = time.monotonic()
        self.joints = joints
        self.last_feedback_at = now
        self.joint_history.append((now, joints))
        while self.joint_history and now - self.joint_history[0][0] > 3.0:
            self.joint_history.popleft()
        if self.writer is not None and self.record_joint_stream:
            self._write("joint_states", message, self.get_clock().now().nanoseconds)

    def _feedback_valid_callback(self, message):
        """仅接受当前仍有效的驱动 UDP 反馈。"""
        self.last_valid_feedback_at = time.monotonic() if message.data else None

    def _gripper_callback(self, message):
        """夹爪实际开合度 [0, 1]。"""
        if math.isfinite(message.data):
            self.gripper_state = float(message.data)
            self.gripper_state_at = time.monotonic()

    def _image_callback(self, message):
        """只缓存最新一帧及其接收时间，解码推迟到采样时。"""
        self.latest_image = (message, self.get_clock().now().nanoseconds, time.monotonic())

    def _result_callback(self, message):
        """处理 MoveJ 结果；非人工急停导致的失败会锁定节点。"""
        if self.motion is None:
            return
        motion, self.motion = self.motion, None
        if message.data:
            if motion.purpose in ("bootstrap", "sample", "manual"):
                self.capture_purpose = motion.purpose
                self.settled_at = None
            return
        if self.phase == "paused":
            self.get_logger().warning("运动已被急停中断")
            return
        self._lock("MoveJ 执行失败；停止采集，请检查机械臂后重启节点")

    # ---------------------------------------------------------------- 工具

    def _lock(self, reason):
        """进入锁定态，不再下发任何运动。"""
        self.phase = "locked"
        self.capture_purpose = None
        self.bootstrap_queue = []
        self.get_logger().error(reason)

    def _hardware_ready(self):
        """反馈新鲜、驱动有效且有订阅者时才允许运动。"""
        now = time.monotonic()
        return (self.joints is not None and self.last_feedback_at is not None and
                self.last_valid_feedback_at is not None and
                now - self.last_feedback_at <= FEEDBACK_FRESHNESS_SECONDS and
                now - self.last_valid_feedback_at <= FEEDBACK_FRESHNESS_SECONDS and
                (self.dry_run or self.movej_publisher.get_subscription_count() > 0))

    def _gripper_ready(self):
        """夹爪开合度接近目标值（相机视野中的夹爪形状固定）。"""
        if not self.require_gripper:
            return True
        return (self.gripper_state is not None and self.gripper_state_at is not None and
                time.monotonic() - self.gripper_state_at <= 1.0 and
                abs(self.gripper_state - self.gripper_openness) <= GRIPPER_TOLERANCE)

    def _stationary_for(self, seconds):
        """最近 seconds 内关节变化都小于阈值且历史覆盖足够长。"""
        if not self.joint_history:
            return False
        now = time.monotonic()
        if now - self.joint_history[-1][0] > FEEDBACK_FRESHNESS_SECONDS:
            return False
        window = [joints for stamp, joints in self.joint_history if now - stamp <= seconds]
        if len(window) < 2 or now - self.joint_history[0][0] < seconds:
            return False
        stack = np.stack(window)
        return float(np.max(stack.max(axis=0) - stack.min(axis=0))) < STATIONARY_TOLERANCE_RAD

    def _send_move(self, joints, purpose):
        """下发一次阻塞 MoveJ；dry_run 时只打印目标。"""
        if not self._hardware_ready():
            self.get_logger().warning("机械臂反馈未就绪，未下发运动")
            return False
        joints = np.asarray(joints, dtype=float)
        if not self.kinematics.within_limits(joints):
            self._lock("目标关节越过 URDF 限位，已拒绝")
            return False
        degrees = ", ".join(f"{math.degrees(value):.1f}" for value in joints)
        if self.dry_run:
            self.get_logger().info(f"[dry_run] {purpose} 目标关节(°): [{degrees}]")
            return True
        command = Movej()
        command.joint = [float(value) for value in joints]
        command.speed = self.speed
        command.block = True
        command.trajectory_connect = 0
        command.dof = 7
        self.motion = Motion(joints=joints, sent_at=time.monotonic(), purpose=purpose)
        self.movej_publisher.publish(command)
        self.get_logger().info(f"{purpose} MoveJ {self.speed}%: [{degrees}]")
        return True

    def _open_writer(self):
        """首次需要写入时创建会话目录和 MCAP，并写入会话配置。"""
        if self.writer is not None:
            return
        from fastumi_recorder.bag_writer import ProcessBagWriter
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.debug_dir.mkdir()
        self.writer = ProcessBagWriter(self.session_dir / "bag", SAMPLE_TOPICS)
        session = {
            "board": self.spec.to_dict(),
            "hand_eye_pattern": self.spec.hand_eye_pattern(),
            "camera": self.camera.to_dict(),
            "camera_config": str(self.camera_config),
            "urdf_path": str(self.urdf_path),
            "base_frame": "base_link",
            "flange_frame": "Link7",
            "camera_frame": "wrist_camera_optical_frame",
            "pose_convention": "base_from_flange, xyz m, quaternion xyzw",
            "prior_flange_from_camera": None if self.prior is None else self.prior.tolist(),
            "gripper_openness": self.gripper_openness,
            "speed": self.speed,
        }
        self._write("session", String(data=json.dumps(session, ensure_ascii=False)),
                    self.get_clock().now().nanoseconds)

    def _write(self, key, message, timestamp_ns):
        """序列化后交给独立写入进程；写入失败锁定节点。"""
        try:
            self.writer.write(SAMPLE_TOPICS[key][0], serialize_message(message), timestamp_ns)
        except RuntimeError as error:
            self.writer = None
            self._lock(f"MCAP 写入失败：{error}")

    # ---------------------------------------------------------------- 采样

    def _grab(self):
        """取静止后接收的最新一帧，返回 Capture；图像不新或解码失败返回 None。"""
        if self.latest_image is None or self.joints is None:
            return None
        message, receive_ns, received_at = self.latest_image
        if time.monotonic() - received_at > IMAGE_FRESHNESS_SECONDS:
            return None
        try:
            image = decode_image(message)
            observation = self.detector.detect(image)
        except ValueError as error:
            self.get_logger().warning(f"图像处理失败：{error}")
            return None
        joints = self.joints.copy()
        return Capture(message, image, receive_ns, joints,
                       self.kinematics.forward(joints), observation)

    def _quality(self, capture):
        """检测质量与遮挡检查；返回空字符串表示通过。"""
        observation = capture.observation
        if observation.camera_from_board is None:
            return f"未检测到板（角点 {observation.corner_count}）"
        ratio = observation.corner_count / self.spec.corner_count
        if ratio < self.min_corner_ratio:
            return f"角点比例 {ratio:.0%} 不足（可能被夹爪遮挡）"
        if observation.reprojection_rms_px > self.max_reprojection_px:
            return f"重投影 RMS {observation.reprojection_rms_px:.2f}px 过大"
        view = check_board_view(self.spec, self.camera, observation.camera_from_board,
                                self.mask, margin_px=5, min_area_fraction=0.0)
        return "" if view.ok else view.reason

    def _pose_reliable(self, capture):
        """PnP 位姿可用于粗外参估计（不考虑夹爪遮挡）。"""
        observation = capture.observation
        return (observation.camera_from_board is not None and
                observation.corner_count >= 0.5 * self.spec.corner_count and
                observation.reprojection_rms_px <= 2.0 * self.max_reprojection_px)

    def _save_debug(self, capture, tag, reason):
        """保存检测叠加图，便于人工复核遮挡与角点；原因写入文件名旁的 txt。"""
        if self.writer is None:
            return
        image = capture.image_bgr.copy()
        if self.mask is not None:
            image[self.mask > 0] = (image[self.mask > 0] * 0.5 + np.array([0, 0, 127])).astype(
                np.uint8)
        for pixel in capture.observation.pixels:
            cv2.circle(image, tuple(int(round(v)) for v in pixel), 4, (0, 255, 0), -1)
        observation = capture.observation
        if observation.camera_from_board is not None:
            self._draw_axes(image, observation.camera_from_board)
        # OpenCV putText 不支持中文，图上只写 ASCII 摘要。
        cv2.putText(image, f"{tag} corners {observation.corner_count}/{self.spec.corner_count} "
                    f"rms {observation.reprojection_rms_px:.2f}px", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        stem = self.debug_dir / f"{capture.receive_ns}_{tag}"
        cv2.imwrite(str(stem.with_suffix(".jpg")), image)
        if reason != "ok":
            stem.with_suffix(".txt").write_text(reason + "\n", encoding="utf-8")

    def _draw_axes(self, image, camera_from_board, length_m=0.1):
        """画板系原点和 x（红）、y（绿）轴，用于实板上核对原点与方向约定。"""
        axes = np.array([[0, 0, 0], [length_m, 0, 0], [0, length_m, 0]], dtype=np.float64)
        pixels = self.camera.project(axes @ camera_from_board[:3, :3].T + camera_from_board[:3, 3])
        if not np.all(np.isfinite(pixels)):
            return
        origin, x_end, y_end = (tuple(int(round(v)) for v in pixel) for pixel in pixels)
        cv2.arrowedLine(image, origin, x_end, (0, 0, 255), 3)
        cv2.arrowedLine(image, origin, y_end, (0, 255, 0), 3)

    def _accept(self, capture):
        """写入一个样本的四条消息（同一 bag 时间戳）并更新在线估计。"""
        index = self.next_index
        self.next_index += 1
        stamp_ns = capture.receive_ns
        image_message = as_compressed(capture.image_message, capture.image_bgr)
        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.header.stamp.sec, pose.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
        position = capture.base_from_flange[:3, 3]
        quaternion = quaternion_from_matrix(capture.base_from_flange[:3, :3])
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, position)
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = map(float, quaternion)
        joints = JointState(name=list(JOINT_NAMES), position=capture.joints.tolist())
        joints.header = pose.header
        observation = capture.observation
        meta = {
            "index": index,
            "image_receive_ns": stamp_ns,
            "corner_count": observation.corner_count,
            "corner_total": self.spec.corner_count,
            "reprojection_rms_px": observation.reprojection_rms_px,
            "camera_from_board": observation.camera_from_board.tolist(),
            "base_from_flange": capture.base_from_flange.tolist(),
            "gripper_openness": self.gripper_state,
        }
        self._write("sample_image", image_message, stamp_ns)
        self._write("sample_flange_pose", pose, stamp_ns)
        self._write("sample_joints", joints, stamp_ns)
        self._write("sample_meta", String(data=json.dumps(meta)), stamp_ns)
        self.samples.append((index, capture))
        self._save_debug(capture, f"sample{index:03d}", "ok")
        self._update_estimates()
        self.get_logger().info(
            f"样本 {index} 已保存（有效 {len(self.samples)}/{self.num_samples}，角点 "
            f"{observation.corner_count}/{self.spec.corner_count}，RMS "
            f"{observation.reprojection_rms_px:.2f}px）")

    def _update_estimates(self):
        """估计池≥4 时重估粗外参与板位姿，供后续规划使用。"""
        flanges = [flange for flange, _ in self.pose_pool]
        boards = [board for _, board in self.pose_pool]
        if len(flanges) >= 4 and diversity_stats(flanges).rotation_span_deg >= 10.0:
            estimate = estimate_flange_from_camera(flanges, boards)
            if estimate is not None:
                board = estimate_board_in_base(flanges, boards, estimate)
                if board.spread_mm <= BOARD_SPREAD_WARNING_MM or self.flange_from_camera is None:
                    self.flange_from_camera = estimate
                    self.base_from_board = board.base_from_board
                    return
        if self.flange_from_camera is not None and flanges:
            self.base_from_board = estimate_board_in_base(
                flanges, boards, self.flange_from_camera).base_from_board

    def _process_capture(self):
        """静止 settle_seconds 且收到其后的新图像后完成一次采样。"""
        if not self._stationary_for(self.settle_seconds):
            self.settled_at = None
            return
        if self.settled_at is None:
            self.settled_at = time.monotonic()
        if self.latest_image is None or self.latest_image[2] <= self.settled_at:
            return
        if not self._gripper_ready():
            self.get_logger().warning("夹爪开合度未到目标，等待夹爪", throttle_duration_sec=2.0)
            return
        capture = self._grab()
        if capture is None:
            return
        purpose, self.capture_purpose = self.capture_purpose, None
        self.mask_frames.append(capture)
        if self._pose_reliable(capture):
            self.pose_pool.append((capture.base_from_flange,
                                   capture.observation.camera_from_board))
        if purpose == "bootstrap":
            self.bootstrap_captures.append(capture)
            return
        self.attempts += purpose == "sample"
        reason = self._quality(capture)
        if reason:
            self._save_debug(capture, "rejected", reason)
            self.get_logger().warning(f"样本被拒绝：{reason}")
            return
        self._accept(capture)

    # ---------------------------------------------------------------- 阶段推进

    def _finish_bootstrap(self):
        """用自举帧估计夹爪掩膜和粗外参，筛选并保存自举样本，进入自动采样。"""
        captures = self.bootstrap_captures
        self.bootstrap_captures = []
        if self.mask is None:
            self._estimate_mask()
        self._update_estimates()
        for capture in captures:
            reason = self._quality(capture)
            if reason:
                self._save_debug(capture, "rejected", reason)
                self.get_logger().warning(f"自举样本被拒绝：{reason}")
            else:
                self._accept(capture)
        if self.flange_from_camera is None or not self.pose_pool:
            self._lock("自举样本不足以估计粗外参；请检查板是否在视野中央后重启")
            return
        flanges = [flange for flange, _ in self.pose_pool]
        boards = [board for _, board in self.pose_pool]
        board = estimate_board_in_base(flanges, boards, self.flange_from_camera)
        self.base_from_board = board.base_from_board
        if self.prior is not None:
            difference = invert(self.prior) @ self.flange_from_camera
            self.get_logger().info(
                f"粗外参与先验差 {np.linalg.norm(difference[:3, 3]) * 1000:.0f} mm")
        if board.spread_mm > BOARD_SPREAD_WARNING_MM:
            self.get_logger().warning(
                f"粗外参下板位姿离散 {board.spread_mm:.0f} mm / {board.spread_deg:.1f}°，"
                "规划可能偏离；采样会随样本增加自动修正")
        self.phase = "auto"
        self.get_logger().info("自举完成，开始随机位姿采样")

    def _estimate_mask(self):
        """由缓存的多视角帧估计夹爪掩膜并保存到会话目录和 MCAP。"""
        frames = [capture.image_bgr for capture in self.mask_frames]
        hulls = [capture.observation.pixels for capture in self.mask_frames
                 if capture.observation.corner_count >= 3]
        try:
            mask = estimate_gripper_mask(frames, hulls)
        except ValueError as error:
            self.get_logger().warning(f"夹爪掩膜估计失败：{error}")
            return
        coverage = np.count_nonzero(mask) / mask.size
        if coverage > 0.35:
            self.get_logger().warning(
                f"夹爪掩膜覆盖 {coverage:.0%} 画面，疑似背景过于均匀；"
                "可用 gripper_mask_path 提供人工掩膜")
        self.mask = mask
        if self.writer is not None:
            cv2.imwrite(str(self.session_dir / "gripper_mask.png"), mask)
            message = Image(height=mask.shape[0], width=mask.shape[1], encoding="mono8",
                            step=mask.shape[1], data=mask.tobytes())
            message.header.frame_id = "wrist_camera_optical_frame"
            self._write("gripper_mask", message, self.get_clock().now().nanoseconds)
        self.get_logger().info(f"夹爪掩膜已更新，覆盖 {coverage:.1%} 画面")
        self._update_aim()

    def _update_aim(self):
        """按夹爪掩膜更新规划瞄准方向，让板落在未遮挡区域。"""
        pixel, direction = aim_for_mask(self.camera, self.mask)
        self.settings = replace(self.settings, aim_direction=tuple(direction))
        if self.mask is not None:
            self.get_logger().info(f"规划瞄准像素 ({pixel[0]:.0f}, {pixel[1]:.0f})")

    def _plan_context(self):
        """当前规划上下文；粗外参或板位姿未知时返回 None。"""
        if self.flange_from_camera is None or self.base_from_board is None:
            return None
        return PlanContext(
            base_from_board=self.base_from_board,
            flange_from_camera=self.flange_from_camera,
            reference_base_from_camera=self.reference["base_from_flange"] @
            self.flange_from_camera,
            sampled_base_from_flange=[capture.base_from_flange for _, capture in self.samples],
        )

    def tick(self):
        """主循环每次调用：超时检查、采样、推进自举和自动阶段。"""
        if self.motion is not None:
            elapsed = time.monotonic() - self.motion.sent_at
            if self.motion.purpose == "stopped" and elapsed > STOP_RESULT_GRACE_SECONDS:
                # 急停后驱动可能不再回报结果；短暂等待后放弃这次运动。
                self.motion = None
            elif self.motion.purpose != "stopped" and elapsed > self.motion_timeout:
                if not self.dry_run:
                    self.stop_publisher.publish(Empty())
                self.motion = None
                self._lock("等待 MoveJ 结果超时；已请求停止运动并锁定采集，请检查机械臂后重启节点")
        if self.phase in ("locked", "paused", "idle", "done"):
            if self.capture_purpose == "manual" and self.motion is None:
                self._process_capture()
            return
        if self.motion is not None:
            return
        if self.capture_purpose is not None:
            self._process_capture()
            return
        if self.phase == "bootstrap":
            self._step_bootstrap()
        elif self.phase == "auto":
            self._step_auto()

    def _step_bootstrap(self):
        """依次执行自举目标；全部完成后进入自动采样。"""
        if not self.bootstrap_queue:
            self._finish_bootstrap()
            return
        if not self._hardware_ready():
            self.get_logger().warning("等待机械臂反馈", throttle_duration_sec=2.0)
            return
        target = self.bootstrap_queue.pop(0)
        joints, reason = check_reachable(self.kinematics, target, self.joints, self.settings)
        if joints is None:
            self.get_logger().warning(f"跳过自举位姿：{reason}")
            return
        self._send_move(joints, "bootstrap")

    def _step_auto(self):
        """规划在后台线程中进行，保证急停按键始终可响应。"""
        if len(self.samples) >= self.num_samples:
            self._report_done()
            return
        if self.attempts >= self.max_attempts:
            self.phase = "paused"
            self.get_logger().error(f"尝试 {self.attempts} 次仍未采满，已暂停；按 a 继续")
            return
        if self.plan_future is None:
            if not self._hardware_ready():
                self.get_logger().warning("等待机械臂反馈", throttle_duration_sec=2.0)
                return
            context = self._plan_context()
            if context is None:
                self._lock("缺少粗外参或板位姿，无法规划")
                return
            self.plan_future = self.planner.submit(
                plan_next, self.kinematics, self.spec, self.camera, self.mask, context,
                self.joints.copy(), self.rng, self.settings)
            return
        if not self.plan_future.done():
            return
        candidate, reasons = self.plan_future.result()
        self.plan_future = None
        if candidate is None:
            self.phase = "paused"
            self.get_logger().error(f"找不到可行位姿 {reasons}；可调整参考位姿或采样范围后按 a")
            return
        if self.dry_run:
            self.get_logger().info(
                f"[dry_run] 规划第 {len(self.samples) + 1} 个位姿，法兰 "
                f"{np.round(candidate.base_from_flange[:3, 3], 3).tolist()}")
            self.attempts += 1
            self.samples.append((-1, Capture(None, None, 0, candidate.joints,
                                             candidate.base_from_flange, None)))
            return
        self._send_move(candidate.joints, "sample")

    def _report_done(self):
        """采满后打印多样性统计并回到空闲。"""
        self.phase = "done"
        flanges = [capture.base_from_flange for _, capture in self.samples]
        stats = diversity_stats(flanges)
        self.get_logger().info(
            f"采集完成：{stats.sample_count} 个样本，旋转跨度 {stats.rotation_span_deg:.0f}°，"
            f"轴分离 {stats.axis_separation_deg:.0f}°；按 h 回参考位姿，按 q 保存退出")

    # ---------------------------------------------------------------- 按键

    def handle_key(self, key):
        """单键命令分发。"""
        if key is None:
            return
        if key == " ":
            self._emergency_pause()
            return
        if key == "q":
            self._quit()
            return
        if self.phase == "locked":
            self.get_logger().warning("节点已锁定，仅响应 q；请检查机械臂后重启")
            return
        handlers = {"r": self._set_reference, "a": self._start_auto, "c": self._manual_capture,
                    "u": self._undo, "m": self._remask, "h": self._go_home}
        handler = handlers.get(key)
        if handler is not None:
            handler()

    def _busy(self):
        """有运动、待采样或规划中时拒绝新的运动类命令。"""
        if self.motion is not None or self.capture_purpose is not None or (
                self.plan_future is not None):
            self.get_logger().warning("机械臂正在运动或采样中，请先按空格暂停")
            return True
        return False

    def _emergency_pause(self):
        """立即下发 move_stop 并暂停自动流程。"""
        if not self.dry_run:
            self.stop_publisher.publish(Empty())
        if self.motion is not None:
            self.motion = Motion(self.motion.joints, time.monotonic(), "stopped")
        if self.phase in ("bootstrap", "auto"):
            self.paused_phase = self.phase
        if self.phase != "locked":
            self.phase = "paused"
        self.capture_purpose = None
        if self.plan_future is not None:
            self.plan_future.cancel()
            self.plan_future = None
        self.get_logger().warning("已急停并暂停；按 a 继续，h 回参考位姿，q 保存退出")

    def _set_reference(self):
        """当前静止位姿能看到整块板时设为参考位姿。"""
        if self._busy():
            return
        self.gripper_publisher.publish(Float32(data=self.gripper_openness))
        if not self._stationary_for(0.3):
            self.get_logger().warning("机械臂未静止或无关节反馈，无法设定参考位姿")
            return
        capture = self._grab()
        if capture is None:
            self.get_logger().warning("没有新鲜图像，无法设定参考位姿")
            return
        if capture.observation.camera_from_board is None:
            self.get_logger().warning(
                f"当前视野未检测到板（角点 {capture.observation.corner_count}），请调整位姿")
            return
        self._open_writer()
        self.reference = {"joints": capture.joints, "base_from_flange": capture.base_from_flange,
                          "capture": capture}
        self.mask_frames.append(capture)
        if self._pose_reliable(capture):
            self.pose_pool.append((capture.base_from_flange,
                                   capture.observation.camera_from_board))
        distance = capture.observation.camera_from_board[2, 3]
        self.get_logger().info(
            f"参考位姿已设定：角点 {capture.observation.corner_count}/{self.spec.corner_count}，"
            f"板距相机约 {distance:.2f} m；按 a 开始")

    def _start_auto(self):
        """首次进入自举阶段；暂停后继续原阶段。"""
        if self._busy():
            return
        if self.reference is None:
            self.get_logger().warning("请先按 r 设定参考位姿")
            return
        if self.phase == "done":
            self.get_logger().info("已采满；如需更多样本请调大 num_samples 重启")
            return
        self.gripper_publisher.publish(Float32(data=self.gripper_openness))
        self.attempts = 0 if self.phase == "paused" else self.attempts
        if self.phase == "paused" and getattr(self, "paused_phase", None):
            self.phase = self.paused_phase
            self.paused_phase = None
            self.get_logger().info(f"继续 {self.phase} 阶段")
            return
        if self.dry_run:
            if self.flange_from_camera is None:
                self._lock("dry_run 需要 prior_path 提供外参先验")
                return
            observation = self.reference["capture"].observation
            self.base_from_board = (self.reference["base_from_flange"] @
                                    self.flange_from_camera @ observation.camera_from_board)
            self.phase = "auto"
            return
        pivot = np.zeros(3)
        if self.flange_from_camera is not None:
            observation = self.reference["capture"].observation
            board_center = observation.camera_from_board @ np.append(
                self.spec.center_point(), 1.0)
            pivot = (self.flange_from_camera @ board_center)[:3]
        self.bootstrap_queue = bootstrap_targets(self.reference["base_from_flange"], pivot)
        self.bootstrap_captures = []
        self.phase = "bootstrap"
        self.get_logger().info(f"开始自举：{len(self.bootstrap_queue)} 个小幅姿态扰动")

    def _manual_capture(self):
        """在当前静止位姿补采一个样本（例如人工拖动示教后）。"""
        if self._busy():
            return
        if self.writer is None:
            self.get_logger().warning("请先按 r 设定参考位姿")
            return
        self.capture_purpose = "manual"
        self.settled_at = None

    def _undo(self):
        """撤销最近一个样本：MCAP 中追加撤销记录，导出时剔除。"""
        if not self.samples or self.writer is None:
            self.get_logger().warning("没有可撤销的样本")
            return
        index, _ = self.samples.pop()
        self._write("sample_rejected", String(data=json.dumps({"index": index})),
                    self.get_clock().now().nanoseconds)
        if self.phase == "done":
            self.phase = "paused"
        self.get_logger().info(f"已撤销样本 {index}，剩余 {len(self.samples)}")

    def _remask(self):
        """用最近缓存的多视角帧重新估计夹爪掩膜。"""
        self._estimate_mask()

    def _go_home(self):
        """回到参考位姿（不采样）。"""
        if self._busy():
            return
        if self.reference is None:
            self.get_logger().warning("尚未设定参考位姿")
            return
        if self.phase in ("bootstrap", "auto"):
            self.phase = "paused"
        self._send_move(self.reference["joints"], "home")

    def _quit(self):
        """停止运动、关闭 MCAP 并打印导出命令。"""
        if self.motion is not None and not self.dry_run:
            self.stop_publisher.publish(Empty())
        self.finished = True

    def close(self):
        """关闭写入进程并输出摘要；在 destroy_node 前调用。"""
        self.planner.shutdown(wait=False, cancel_futures=True)
        if self.writer is None:
            return
        writer, self.writer = self.writer, None
        try:
            writer.close()
        except RuntimeError as error:
            self.get_logger().error(f"MCAP 关闭失败：{error}")
            return
        summary = {"samples": [index for index, _ in self.samples if index >= 0],
                   "bag": str(self.session_dir / "bag")}
        (self.session_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self.get_logger().info(
            f"已保存 {len(self.samples)} 个样本到 {self.session_dir}\n导出：python "
            f"export_hand_eye_dataset.py --input {self.session_dir} "
            f"--output {self.session_dir / 'hand_eye_result'}")


def main(args=None):
    """运行采集节点及当前终端的逐键读取循环。"""
    if not sys.stdin.isatty():
        print("手眼采集需要交互终端；请在第二个终端运行 ros2 run", file=sys.stderr)
        return 2

    rclpy.init(args=args)
    node = None
    try:
        node = HandEyeCapture()
        with TerminalKeyReader(sys.stdin) as reader:
            while rclpy.ok() and not node.finished:
                rclpy.spin_once(node, timeout_sec=0.02)
                node.handle_key(reader.read_key(0.02))
                node.tick()
    except KeyboardInterrupt:
        if node is not None and node.motion is not None and not node.dry_run:
            node.stop_publisher.publish(Empty())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"手眼采集失败：{error}", file=sys.stderr)
        return 2
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
