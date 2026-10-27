"""键盘驱动 RM75 随机慢动，用腕部鱼眼相机观测夹爪双 ArUco，标定部署端 A_d = ^Cd T_tcp。

相机与双 tag 都固连在 Link7 / 夹爪上，^Cd T_d_tag 理论上不随臂姿变化；随机改变姿态用于
暴露重力方向变化下的夹爪形变、平均检测噪声，并用位姿间离散度评估结果可信度。

在 hardware.launch.py 启动后于第二个终端运行：
    ros2 run hand_eye_calibration_macpdata keyboard_tag_tcp_capture --ros-args \
        -p T_d_tag_tcp_path:=$PWD/../config/calibration/T_d_tag_tcp.yaml \
        -p T_cam_end_path:=$PWD/../config/calibration/T_cam_end.yaml \
        -p T_end_tcp_path:=$PWD/../config/calibration/T_end_tcp.yaml

按键：a 开始/继续 | 空格 急停并暂停 | c 当前位姿补采 | h 回起始位姿 | q 求解并退出

流程：a 以当前静止位姿为起始位姿并先在此采样 → 在起始 Link7 位姿上叠加随机平移/旋转，
IK 与安全过滤后低速 MoveJ → 静止后连续取 frames_per_pose 帧求 ^Cd T_d_tag 并平均为一个
样本 → 采满 num_poses 后回起始位姿，鲁棒平均得到 T_cam_dtag，A_d = T_cam_dtag @ T_d_tag_tcp。
输出 <dataset_root>/tag_tcp/session_<时间>/ 下 result.yaml、samples.csv、images/、debug/。
"""

from dataclasses import dataclass, field
import csv
import datetime
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rm_ros_interfaces.msg import Movej
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Bool, Empty, Float32
import yaml

from fastumi_bringup.kinematics import JOINT_NAMES, ArmKinematics
from fastumi_bringup.terminal import TerminalKeyReader
from fastumi_bringup.transforms import (
    axis_angle_matrix, average_transforms, invert, make_transform, quaternion_from_matrix,
)
from hand_eye_calibration_macpdata.board import load_fisheye_camera
from hand_eye_calibration_macpdata.capture import _share_file, decode_image, default_dataset_root
from hand_eye_calibration_macpdata.dual_tag import (
    DualTagError, DualTagEstimator, draw_observation, load_matrix_yaml, robust_average,
    transform_difference,
)
from hand_eye_calibration_macpdata.sampler import SamplerSettings, check_reachable


FEEDBACK_FRESHNESS_SECONDS = 0.5
IMAGE_FRESHNESS_SECONDS = 0.5
GRIPPER_TOLERANCE = 0.05
STATIONARY_TOLERANCE_RAD = 1e-4
STOP_RESULT_GRACE_SECONDS = 3.0
# 单次随机目标规划的最大尝试数（每次一次 IK）。
MAX_PLAN_TRIES = 60
# 一个位姿采不到足够有效帧时，最多等待的时长（秒），超时记为失败位姿。
POSE_CAPTURE_TIMEOUT_SECONDS = 10.0

HELP_TEXT = "按键：a 开始/继续 | 空格 急停暂停 | c 当前位姿补采 | h 回起始位姿 | q 求解并退出"


@dataclass
class Motion:
    """进行中的 MoveJ：purpose 为 sample / home / return。"""

    joints: np.ndarray
    sent_at: float
    purpose: str


@dataclass
class PoseCapture:
    """一个静止位姿下正在进行的多帧采集。"""

    openness: float
    started_at: float
    settled_at: float | None = None
    last_receive_ns: int = -1
    frames: list = field(default_factory=list)      # (receive_ns, image_bgr, DualTagObservation)
    rejected: dict = field(default_factory=dict)    # 原因 → 次数


@dataclass
class PoseSample:
    """一个位姿的平均结果；变换单位米。"""

    index: int
    receive_ns: int
    joints: np.ndarray
    base_from_flange: np.ndarray
    openness_target: float
    gripper_state: float | None
    camera_from_dtag: np.ndarray
    valid_frames: int
    total_frames: int
    reprojection_rms_px: float
    normal_angle_deg: float
    tag_distance_m: float


def random_flange_target(home_base_from_flange, rng, translation_range_m, rotation_max_deg):
    """在起始 ^base T_Link7 上叠加随机扰动：base 系各轴均匀平移，绕 Link7 原点随机轴旋转。"""
    translation = rng.uniform(-translation_range_m, translation_range_m, 3)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = math.radians(rng.uniform(0.0, rotation_max_deg))
    target = home_base_from_flange @ make_transform(axis_angle_matrix(axis, angle), np.zeros(3))
    target[:3, 3] += translation
    return target


def _pose_text(transform):
    """x y z（米）与四元数 xyzw，用于 CSV。"""
    return [*map(float, transform[:3, 3]), *map(float, quaternion_from_matrix(transform[:3, :3]))]


class TagTcpCapture(Node):
    """双 ArUco 相机→TCP 标定节点；检测算法在 dual_tag.py。"""

    def __init__(self, **node_kwargs):
        super().__init__("keyboard_tag_tcp_capture", **node_kwargs)
        self._declare_parameters()
        parameter = lambda name: self.get_parameter(name).value  # noqa: E731

        dtag_path = parameter("T_d_tag_tcp_path")
        if not dtag_path:
            raise ValueError("必须提供 T_d_tag_tcp_path（如 config/calibration/T_d_tag_tcp.yaml）")
        self.dtag_from_tcp_path = str(Path(dtag_path).expanduser().resolve())
        self.dtag_from_tcp = load_matrix_yaml(self.dtag_from_tcp_path, "T_d_tag_tcp")
        self.cam_end_path = parameter("T_cam_end_path")
        self.camera_from_flange = (load_matrix_yaml(self.cam_end_path, "cam0.T_cam_hand")
                                   if self.cam_end_path else None)
        self.end_tcp_path = parameter("T_end_tcp_path")
        self.flange_from_tcp_nominal = (load_matrix_yaml(self.end_tcp_path, "T_end_tcp")
                                        if self.end_tcp_path else None)

        camera_config = parameter("camera_config") or _share_file(
            "fastumi_usb_camera", "config/EndEffectorCalib.yaml", "camera_config")
        self.camera_config = camera_config
        self.camera = load_fisheye_camera(camera_config)
        self.estimator = DualTagEstimator(
            self.camera, parameter("aruco_dictionary"), float(parameter("marker_size_m")),
            (int(parameter("tag0_id")), int(parameter("tag1_id"))))
        urdf_path = parameter("urdf_path") or _share_file(
            "fastumi_bringup", "assets/rm_75_kinematic.urdf", "urdf_path")
        self.kinematics = ArmKinematics.from_urdf(urdf_path)

        self.num_poses = int(parameter("num_poses"))
        self.frames_per_pose = int(parameter("frames_per_pose"))
        self.min_valid_fraction = float(parameter("min_valid_fraction"))
        self.speed = int(parameter("speed"))
        if not 1 <= self.speed <= 30:
            raise ValueError("speed 需在 1..30（百分比）；本标定只需低速运动")
        self.translation_range_m = float(parameter("translation_range_m"))
        self.rotation_max_deg = float(parameter("rotation_max_deg"))
        if not 0.0 <= self.translation_range_m <= 0.15 or not 0.0 <= self.rotation_max_deg <= 45.0:
            raise ValueError("translation_range_m 需在 0..0.15，rotation_max_deg 需在 0..45")
        self.settle_seconds = float(parameter("settle_seconds"))
        self.motion_timeout = float(parameter("motion_timeout"))
        self.max_reprojection_px = float(parameter("max_reprojection_px"))
        self.max_normal_angle_deg = float(parameter("max_normal_angle_deg"))
        self.openness_values = [float(value) for value in parameter("gripper_openness_values")]
        if not self.openness_values or not all(0.0 <= value <= 1.0
                                               for value in self.openness_values):
            raise ValueError("gripper_openness_values 需为 [0, 1] 内的非空列表")
        self.require_gripper = bool(parameter("require_gripper"))
        self.dry_run = bool(parameter("dry_run"))
        self.max_attempts = int(parameter("max_attempts_factor")) * self.num_poses
        self.settings = SamplerSettings(
            workspace_min=tuple(parameter("workspace_min")),
            workspace_max=tuple(parameter("workspace_max")),
            max_joint_step_rad=float(parameter("max_joint_step_rad")),
        )
        self.rng = np.random.default_rng(
            None if parameter("random_seed") < 0 else parameter("random_seed"))

        root = Path(parameter("dataset_root") or default_dataset_root()).expanduser()
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = root / "tag_tcp" / f"session_{stamp}"

        # 反馈状态（time.monotonic 秒）。
        self.joints = None
        self.joint_history = []
        self.last_feedback_at = None
        self.last_valid_feedback_at = None
        self.gripper_state = None
        self.gripper_state_at = None
        self.latest_image = None  # (message, receive_ns, monotonic)

        # 采集状态。
        self.phase = "idle"  # idle / running / paused / locked / done
        self.motion = None
        self.pose_capture = None
        self.home = None  # {"joints", "base_from_flange"}
        self.samples = []
        self.attempts = 0
        self.solved_count = 0
        self.result = None
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
            f"双 ArUco 相机→TCP 标定就绪：图像 {image_topic}，T_d_tag_tcp {self.dtag_from_tcp_path}，"
            f"输出 {self.session_dir}" + ("（dry_run：只规划不运动）" if self.dry_run else ""))
        self.get_logger().info(HELP_TEXT)

    def _declare_parameters(self):
        defaults = {
            "T_d_tag_tcp_path": "",
            "T_cam_end_path": "",
            "T_end_tcp_path": "",
            "camera_config": "",
            "urdf_path": "",
            "image_topic": "/wrist_camera/image_raw/compressed",
            "aruco_dictionary": "DICT_4X4_50",
            "marker_size_m": 0.016,
            "tag0_id": 0,
            "tag1_id": 1,
            "dataset_root": "",
            "num_poses": 20,
            "frames_per_pose": 15,
            "min_valid_fraction": 0.5,
            "speed": 10,
            "translation_range_m": 0.04,
            "rotation_max_deg": 25.0,
            "workspace_min": [-0.6, -0.5, 0.16],
            "workspace_max": [0.6, 0.5, 0.60],
            "max_joint_step_rad": 0.5,
            "settle_seconds": 1.0,
            "motion_timeout": 60.0,
            "max_reprojection_px": 1.5,
            "max_normal_angle_deg": 10.0,
            "gripper_openness_values": [1.0],
            "require_gripper": True,
            "dry_run": False,
            "max_attempts_factor": 3,
            "random_seed": -1,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    # ---------------------------------------------------------------- 回调

    def _joint_state_callback(self, message):
        """按名称读取七关节弧度，保留最近 3 s 历史用于静止判定。"""
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
            self.joint_history.pop(0)

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
            if motion.purpose == "sample":
                self._begin_pose_capture(self._next_openness())
            elif motion.purpose == "return":
                self.phase = "done"
                self._solve_and_write()
                self.get_logger().info("已回到起始位姿并完成求解；按 q 退出")
            return
        if self.phase == "paused":
            self.get_logger().warning("运动已被急停中断")
            return
        self._lock("MoveJ 执行失败；停止采集，请检查机械臂后重启节点")

    # ---------------------------------------------------------------- 工具

    def _lock(self, reason):
        """进入锁定态，不再下发任何运动。"""
        self.phase = "locked"
        self.pose_capture = None
        self.get_logger().error(reason)

    def _hardware_ready(self):
        """反馈新鲜、驱动有效且有订阅者时才允许运动。"""
        now = time.monotonic()
        return (self.joints is not None and self.last_feedback_at is not None and
                self.last_valid_feedback_at is not None and
                now - self.last_feedback_at <= FEEDBACK_FRESHNESS_SECONDS and
                now - self.last_valid_feedback_at <= FEEDBACK_FRESHNESS_SECONDS and
                (self.dry_run or self.movej_publisher.get_subscription_count() > 0))

    def _gripper_ready(self, openness):
        """夹爪开合度接近目标值。"""
        if not self.require_gripper:
            return True
        return (self.gripper_state is not None and self.gripper_state_at is not None and
                time.monotonic() - self.gripper_state_at <= 1.0 and
                abs(self.gripper_state - openness) <= GRIPPER_TOLERANCE)

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

    def _next_openness(self):
        """各位姿轮流使用 gripper_openness_values 中的开度。"""
        return self.openness_values[len(self.samples) % len(self.openness_values)]

    def _command_gripper(self, openness):
        self.gripper_publisher.publish(Float32(data=float(openness)))

    # ---------------------------------------------------------------- 采样

    def _begin_pose_capture(self, openness):
        """到位后开始多帧采集；先下发夹爪开度。"""
        self._command_gripper(openness)
        self.pose_capture = PoseCapture(openness=openness, started_at=time.monotonic())

    def _process_pose_capture(self):
        """静止且夹爪到位后，逐帧取静止之后接收的新图像，凑满 frames_per_pose 帧后汇总。"""
        capture = self.pose_capture
        now = time.monotonic()
        if now - capture.started_at > POSE_CAPTURE_TIMEOUT_SECONDS + self.settle_seconds:
            self._finish_pose_capture()
            return
        if not (self._stationary_for(self.settle_seconds) and self._gripper_ready(capture.openness)):
            if capture.settled_at is not None:
                # 采集中途又动了：丢弃已采帧重新等待。
                capture.settled_at, capture.frames, capture.rejected = None, [], {}
            return
        if capture.settled_at is None:
            capture.settled_at = now
        if self.latest_image is None:
            return
        message, receive_ns, received_at = self.latest_image
        if (receive_ns == capture.last_receive_ns or received_at < capture.settled_at or
                now - received_at > IMAGE_FRESHNESS_SECONDS):
            return
        capture.last_receive_ns = receive_ns
        try:
            image = decode_image(message)
            observation = self.estimator.estimate(image)
            if observation.reprojection_rms_px > self.max_reprojection_px:
                raise DualTagError("重投影 RMS 过大")
            if observation.normal_angle_deg > self.max_normal_angle_deg:
                raise DualTagError("两 tag 法向夹角过大")
            capture.frames.append((receive_ns, image, observation))
        except (DualTagError, ValueError) as error:
            reason = str(error)
            capture.rejected[reason] = capture.rejected.get(reason, 0) + 1
        if len(capture.frames) + sum(capture.rejected.values()) >= self.frames_per_pose:
            self._finish_pose_capture()

    def _finish_pose_capture(self):
        """把一个位姿的有效帧平均为一个样本；有效帧不足时记失败。"""
        capture, self.pose_capture = self.pose_capture, None
        self.attempts += 1
        total = len(capture.frames) + sum(capture.rejected.values())
        needed = max(1, int(math.ceil(self.min_valid_fraction * self.frames_per_pose)))
        if self.joints is None:
            self.get_logger().warning("没有关节反馈，丢弃该位姿")
            return
        if len(capture.frames) < needed:
            self.get_logger().warning(
                f"位姿有效帧 {len(capture.frames)}/{total} 不足 {needed}，跳过；原因 {capture.rejected}")
            return
        observations = [observation for _, _, observation in capture.frames]
        index = len(self.samples)
        receive_ns, image, first = capture.frames[0]
        joints = self.joints.copy()
        sample = PoseSample(
            index=index, receive_ns=receive_ns, joints=joints,
            base_from_flange=self.kinematics.forward(joints),
            openness_target=capture.openness, gripper_state=self.gripper_state,
            camera_from_dtag=average_transforms(
                [observation.camera_from_dtag for observation in observations]),
            valid_frames=len(observations), total_frames=total,
            reprojection_rms_px=float(np.mean([o.reprojection_rms_px for o in observations])),
            normal_angle_deg=float(np.mean([o.normal_angle_deg for o in observations])),
            tag_distance_m=float(np.mean([o.tag_distance_m for o in observations])))
        self.samples.append(sample)
        self._save_images(index, image, first)
        position = sample.camera_from_dtag[:3, 3] * 1000.0
        self.get_logger().info(
            f"样本 {index}（{len(self.samples)}/{self.num_poses}）：有效帧 {len(observations)}/{total}，"
            f"RMS {sample.reprojection_rms_px:.2f}px，法向夹角 {sample.normal_angle_deg:.1f}°，"
            f"tag 间距 {sample.tag_distance_m * 1000:.1f} mm，d_tag 原点 "
            f"[{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}] mm")

    def _ensure_session_dir(self):
        if not self.session_dir.exists():
            (self.session_dir / "images").mkdir(parents=True)
            (self.session_dir / "debug").mkdir()

    def _save_images(self, index, image, observation):
        """每个位姿保存一张原始图和叠加图（d_tag 轴、TCP 投影点）。"""
        self._ensure_session_dir()
        cv2.imwrite(str(self.session_dir / "images" / f"pose_{index:03d}.jpg"), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(self.session_dir / "debug" / f"pose_{index:03d}.jpg"),
                    draw_observation(image, self.camera, observation, self.dtag_from_tcp))

    # ---------------------------------------------------------------- 求解

    def _solve_and_write(self):
        """鲁棒平均全部样本，写 result.yaml 与 samples.csv，并打印摘要。"""
        if not self.samples or self.solved_count == len(self.samples):
            return self.result
        average = robust_average([sample.camera_from_dtag for sample in self.samples])
        camera_from_dtag = average.transform
        camera_from_tcp = camera_from_dtag @ self.dtag_from_tcp
        inliers = [bool(value) for value in average.inliers]
        result = {
            "description": "部署端双 ArUco 相机→TCP 标定；^a T_b 为 b 系在 a 系中的位姿，平移单位米",
            "camera_config": str(self.camera_config),
            "T_d_tag_tcp_path": self.dtag_from_tcp_path,
            "T_cam_dtag": camera_from_dtag.tolist(),
            "T_d_tag_tcp": self.dtag_from_tcp.tolist(),
            "A_d": camera_from_tcp.tolist(),
            "camera_in_tcp_mm": (invert(camera_from_tcp)[:3, 3] * 1000.0).tolist(),
            "samples": len(self.samples),
            "inliers": int(sum(inliers)),
            "outlier_indices": [s.index for s, keep in zip(self.samples, inliers) if not keep],
            "pose_spread": {"translation_rms_mm": average.translation_rms_m * 1000.0,
                            "rotation_rms_deg": average.rotation_rms_deg},
            "mean_reprojection_rms_px": float(np.mean(
                [s.reprojection_rms_px for s in self.samples])),
            "mean_tag_distance_mm": float(np.mean([s.tag_distance_m for s in self.samples])
                                          * 1000.0),
        }
        if len(self.openness_values) > 1:
            by_openness = {}
            for openness in self.openness_values:
                group = [s.camera_from_dtag for s, keep in zip(self.samples, inliers)
                         if keep and s.openness_target == openness]
                if group:
                    translation, angle = transform_difference(
                        camera_from_dtag, average_transforms(group))
                    by_openness[str(openness)] = {"samples": len(group),
                                                  "delta_translation_mm": translation.tolist(),
                                                  "delta_rotation_deg": angle}
            result["openness_check"] = by_openness
        if self.camera_from_flange is not None:
            flange_from_tcp = invert(self.camera_from_flange) @ camera_from_tcp
            result["T_cam_end_path"] = str(self.cam_end_path)
            result["T_link7_tcp_measured"] = flange_from_tcp.tolist()
            if self.flange_from_tcp_nominal is not None:
                translation, angle = transform_difference(
                    self.flange_from_tcp_nominal, flange_from_tcp)
                result["T_end_tcp_path"] = str(self.end_tcp_path)
                result["vs_nominal_T_end_tcp"] = {"delta_translation_mm": translation.tolist(),
                                                  "delta_rotation_deg": angle}
        self._ensure_session_dir()
        (self.session_dir / "result.yaml").write_text(
            "# A_d = ^Cd T_tcp = T_cam_dtag @ T_d_tag_tcp；D = inv(A_u) @ A_d\n" +
            yaml.safe_dump(result, allow_unicode=True, sort_keys=False), encoding="utf-8")
        self._write_samples_csv(inliers)
        self.solved_count = len(self.samples)
        self.result = result
        spread = result["pose_spread"]
        self.get_logger().info(
            f"求解完成：内点 {result['inliers']}/{result['samples']}，位姿间离散度 "
            f"{spread['translation_rms_mm']:.2f} mm / {spread['rotation_rms_deg']:.2f}°；"
            f"A_d 平移 {np.round(camera_from_tcp[:3, 3] * 1000.0, 1).tolist()} mm，"
            f"相机在 TCP 系 {np.round(result['camera_in_tcp_mm'], 1).tolist()} mm"
            + (f"；与名义 T_end_tcp 差 {np.round(result['vs_nominal_T_end_tcp']['delta_translation_mm'], 1).tolist()} mm / "
               f"{result['vs_nominal_T_end_tcp']['delta_rotation_deg']:.2f}°"
               if "vs_nominal_T_end_tcp" in result else "")
            + f"。结果：{self.session_dir / 'result.yaml'}")
        return result

    def _write_samples_csv(self, inliers):
        header = (["index", "receive_ns", "inlier", "openness_target", "gripper_state",
                   "valid_frames", "total_frames", "reprojection_rms_px", "normal_angle_deg",
                   "tag_distance_mm"] + [f"{name}_rad" for name in JOINT_NAMES] +
                  [f"base_flange_{key}" for key in ("x", "y", "z", "qx", "qy", "qz", "qw")] +
                  [f"cam_dtag_{key}" for key in ("x", "y", "z", "qx", "qy", "qz", "qw")])
        with open(self.session_dir / "samples.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for sample, keep in zip(self.samples, inliers):
                writer.writerow([
                    sample.index, sample.receive_ns, int(keep), sample.openness_target,
                    "" if sample.gripper_state is None else sample.gripper_state,
                    sample.valid_frames, sample.total_frames, sample.reprojection_rms_px,
                    sample.normal_angle_deg, sample.tag_distance_m * 1000.0,
                    *map(float, sample.joints), *_pose_text(sample.base_from_flange),
                    *_pose_text(sample.camera_from_dtag)])

    # ---------------------------------------------------------------- 主循环

    def tick(self):
        """主循环每次调用：超时检查、采样、推进随机位姿。"""
        if self.finished:
            return  # 按 q 后主循环仍会调用一次 tick，退出请求必须阻止后续采样和运动。
        if self.motion is not None:
            elapsed = time.monotonic() - self.motion.sent_at
            if self.motion.purpose == "stopped" and elapsed > STOP_RESULT_GRACE_SECONDS:
                self.motion = None
            elif self.motion.purpose != "stopped" and elapsed > self.motion_timeout:
                if not self.dry_run:
                    self.stop_publisher.publish(Empty())
                self.motion = None
                self._lock("等待 MoveJ 结果超时；已请求停止运动并锁定，请检查机械臂后重启节点")
            return
        if self.pose_capture is not None:
            if self.phase != "locked":
                self._process_pose_capture()
            return
        if self.phase != "running":
            return
        if len(self.samples) >= self.num_poses:
            self._return_home()
            return
        if self.attempts >= self.max_attempts:
            self.phase = "paused"
            self.get_logger().error(f"尝试 {self.attempts} 次仍未采满，已暂停；按 a 继续")
            return
        if not self._hardware_ready():
            self.get_logger().warning("等待机械臂反馈", throttle_duration_sec=2.0)
            return
        self._step_random_pose()

    def _step_random_pose(self):
        """规划一个随机可达目标并下发；找不到时暂停。"""
        reasons = {}
        for _ in range(MAX_PLAN_TRIES):
            target = random_flange_target(self.home["base_from_flange"], self.rng,
                                          self.translation_range_m, self.rotation_max_deg)
            joints, reason = check_reachable(self.kinematics, target, self.joints, self.settings)
            if joints is None:
                key = reason.split(" ")[0]
                reasons[key] = reasons.get(key, 0) + 1
                continue
            self._send_move(joints, "sample")
            if self.dry_run:
                self.attempts += 1
                self.samples.append(None)  # dry_run 只计数，不采样
            return
        self.phase = "paused"
        self.get_logger().error(f"找不到可行随机位姿 {reasons}；可减小扰动范围或调整起始位姿后按 a")

    def _return_home(self):
        """采满后回起始位姿；到位后求解。"""
        if self.dry_run:
            self.samples = [sample for sample in self.samples if sample is not None]
            self.phase = "done"
            self.get_logger().info("[dry_run] 规划完成，未运动、未求解；按 q 退出")
            return
        self.phase = "returning"
        if not self._send_move(self.home["joints"], "return"):
            self.phase = "paused"

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
        handler = {"a": self._start, "c": self._manual_capture, "h": self._go_home}.get(key)
        if handler is not None:
            handler()

    def _busy(self):
        if self.motion is not None or self.pose_capture is not None:
            self.get_logger().warning("机械臂正在运动或采样中，请先按空格暂停")
            return True
        return False

    def _emergency_pause(self):
        """立即下发 move_stop 并暂停自动流程。"""
        if not self.dry_run:
            self.stop_publisher.publish(Empty())
        if self.motion is not None:
            self.motion = Motion(self.motion.joints, time.monotonic(), "stopped")
        if self.phase not in ("locked", "idle", "done"):
            self.phase = "paused"
        self.pose_capture = None
        self.get_logger().warning("已急停并暂停；按 a 继续，h 回起始位姿，q 求解并退出")

    def _start(self):
        """首次以当前静止位姿为起始位姿并先在此采样；暂停后继续。"""
        if self._busy():
            return
        if self.phase == "done":
            self.get_logger().info("已采满；如需更多样本请调大 num_poses 重启")
            return
        if self.home is None:
            if not self._stationary_for(0.3):
                self.get_logger().warning("机械臂未静止或无关节反馈，无法开始")
                return
            self.home = {"joints": self.joints.copy(),
                         "base_from_flange": self.kinematics.forward(self.joints)}
            degrees = ", ".join(f"{math.degrees(value):.1f}" for value in self.joints)
            self.get_logger().info(
                f"起始位姿 [{degrees}]°；随机扰动 ±{self.translation_range_m * 1000:.0f} mm、"
                f"≤{self.rotation_max_deg:.0f}°，速度 {self.speed}%，目标 {self.num_poses} 个位姿")
            self.phase = "running"
            if not self.dry_run:
                self._begin_pose_capture(self._next_openness())
            return
        self.attempts = 0
        self.phase = "running"
        self.get_logger().info("继续采集")

    def _manual_capture(self):
        """在当前静止位姿补采一个样本。"""
        if self._busy() or self.dry_run:
            return
        self._begin_pose_capture(self._next_openness())

    def _go_home(self):
        """回起始位姿（不采样）。"""
        if self._busy():
            return
        if self.home is None:
            self.get_logger().warning("尚未设定起始位姿（按 a）")
            return
        if self.phase == "running":
            self.phase = "paused"
        self._send_move(self.home["joints"], "home")

    def _quit(self):
        """停止运动并结束主循环；close() 中求解。"""
        if self.motion is not None and not self.dry_run:
            self.stop_publisher.publish(Empty())
        self.finished = True

    def close(self):
        """退出前对已有样本求解（若尚未求解）。"""
        self.samples = [sample for sample in self.samples if sample is not None]
        if self.samples:
            self._solve_and_write()
        else:
            self.get_logger().info("没有有效样本，未写出结果")


def main(args=None):
    """运行标定节点及当前终端的逐键读取循环。"""
    if not sys.stdin.isatty():
        print("双 ArUco 标定需要交互终端；请在第二个终端运行 ros2 run", file=sys.stderr)
        return 2

    rclpy.init(args=args)
    node = None
    try:
        node = TagTcpCapture()
        with TerminalKeyReader(sys.stdin) as reader:
            while rclpy.ok() and not node.finished:
                rclpy.spin_once(node, timeout_sec=0.02)
                node.handle_key(reader.read_key(0.02))
                node.tick()
    except KeyboardInterrupt:
        if node is not None and node.motion is not None and not node.dry_run:
            node.stop_publisher.publish(Empty())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"双 ArUco 标定失败：{error}", file=sys.stderr)
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
