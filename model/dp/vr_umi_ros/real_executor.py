"""以严格门控把 Link7 策略序列发送给 RM75 与 Unitree 夹爪。"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import threading
import time

import numpy as np
import rclpy
from fastumi_interfaces.msg import PolicyActionSequence
from geometry_msgs.msg import Pose
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rm_ros_interfaces.msg import Carteposcustom, Rmerr
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float32, String

from diffusion_policy.common.urdf_kinematics import UrdfKinematics
from vr_umi_ros.core import JOINT_NAMES


JOINT_MIN = np.array([-3.1, -2.268, -3.1, -2.355, -3.1, -2.233, -6.28])
JOINT_MAX = np.array([3.1, 2.268, 3.1, 2.355, 3.1, 2.233, 6.28])
WORKSPACE_MIN = np.array([0.05, -0.70, 0.05])
WORKSPACE_MAX = np.array([0.90, 0.70, 1.00])


def pose_arrays(message):
    """把 geometry Pose 转为位置与规范化 xyzw 四元数。"""
    position = np.array(
        [message.position.x, message.position.y, message.position.z], dtype=np.float64)
    quaternion = np.array(
        [message.orientation.x, message.orientation.y,
         message.orientation.z, message.orientation.w], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all() or norm < 1e-8:
        raise ValueError("pose must contain finite position and a valid quaternion")
    return position, quaternion / norm


def pose_error(left_position, left_quaternion, right_position, right_quaternion):
    """返回两姿态的平移米误差和旋转弧度误差。"""
    translation = float(np.linalg.norm(np.asarray(left_position) - np.asarray(right_position)))
    rotation = float(
        (Rotation.from_quat(left_quaternion).inv()
         * Rotation.from_quat(right_quaternion)).magnitude())
    return translation, rotation


def interpolate_pose(start_position, start_quaternion, goal_position, goal_quaternion, ratio):
    """在两个姿态间执行线性位置插值和最短路径旋转插值。"""
    ratio = float(np.clip(ratio, 0.0, 1.0))
    position = np.asarray(start_position) + ratio * (
        np.asarray(goal_position) - np.asarray(start_position))
    start_rotation = Rotation.from_quat(start_quaternion)
    delta = start_rotation.inv() * Rotation.from_quat(goal_quaternion)
    quaternion = (start_rotation * Rotation.from_rotvec(delta.as_rotvec() * ratio)).as_quat()
    return position, quaternion


def rate_limit_pose(position, quaternion, target_position, target_quaternion,
                    max_translation, max_rotation):
    """限制一次控制周期的平移和旋转增量。"""
    delta = np.asarray(target_position) - np.asarray(position)
    distance = float(np.linalg.norm(delta))
    if distance > max_translation:
        delta *= max_translation / distance
    new_position = np.asarray(position) + delta
    current_rotation = Rotation.from_quat(quaternion)
    rotation_delta = current_rotation.inv() * Rotation.from_quat(target_quaternion)
    vector = rotation_delta.as_rotvec()
    angle = float(np.linalg.norm(vector))
    if angle > max_rotation:
        vector *= max_rotation / angle
    new_quaternion = (current_rotation * Rotation.from_rotvec(vector)).as_quat()
    return new_position, new_quaternion


@dataclass
class ExecutionPlan:
    """按 ROS 时钟执行的绝对 Link7 与夹爪目标。"""

    times_ns: np.ndarray
    positions: np.ndarray
    quaternions: np.ndarray
    grippers: np.ndarray
    anchor_time_ns: int
    anchor_position: np.ndarray
    anchor_quaternion: np.ndarray
    anchor_gripper: float
    source: str
    sequence_id: int = 0


def sample_plan(plan, now_ns):
    """跳过过期点，在相邻未来目标之间按绝对时间插值。"""
    if now_ns <= plan.times_ns[0]:
        denominator = max(1, int(plan.times_ns[0] - plan.anchor_time_ns))
        ratio = (now_ns - plan.anchor_time_ns) / denominator
        position, quaternion = interpolate_pose(
            plan.anchor_position, plan.anchor_quaternion,
            plan.positions[0], plan.quaternions[0], ratio)
        gripper = np.interp(
            np.clip(ratio, 0, 1), [0, 1], [plan.anchor_gripper, plan.grippers[0]])
        return position, quaternion, float(gripper)
    index = int(np.searchsorted(plan.times_ns, now_ns, side="right"))
    if index >= len(plan.times_ns):
        return plan.positions[-1], plan.quaternions[-1], float(plan.grippers[-1])
    start = index - 1
    denominator = max(1, int(plan.times_ns[index] - plan.times_ns[start]))
    ratio = (now_ns - plan.times_ns[start]) / denominator
    position, quaternion = interpolate_pose(
        plan.positions[start], plan.quaternions[start],
        plan.positions[index], plan.quaternions[index], ratio)
    gripper = np.interp(ratio, [0, 1], [plan.grippers[start], plan.grippers[index]])
    return position, quaternion, float(gripper)


class RealRobotExecutor(Node):
    """一次性实机 smoke 执行器；默认锁定且不发送硬件命令。"""

    def __init__(self, urdf_path, checkpoint_sha256, enable_motion=False,
                 report_root=None, checkpoint_summary=None):
        super().__init__("fastumi_real_executor")
        self.enable_motion = bool(enable_motion)
        self.fk = UrdfKinematics(Path(urdf_path), JOINT_NAMES)
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_summary = dict(checkpoint_summary or {})
        self.lock = threading.RLock()
        self.phase = "locked"
        self.stop_reason = "not started"
        self.latest_sequence_id = 0
        self.last_observed_sequence_id = 0
        self.arm_after_sequence_id = 0
        self.armed_deadline = 0.0
        self.pending_operator_command = None
        self.pending_operator_deadline = 0.0
        self.self_test_completed = False
        self.inference_metadata = {}
        self.valid_predictions = []
        self.plan = None
        self.joints = None
        self.arm_position = None
        self.arm_quaternion = None
        self.gripper = None
        self.joint_received = self.arm_received = self.gripper_received = 0.0
        self.error_received = 0.0
        self.last_fk_check = 0.0
        self.robot_error_clear = False
        self.fk_match_count = 0
        self.fk_translation_error_max = 0.0
        self.fk_rotation_error_max = 0.0
        self.rm_errors_seen = []
        self.last_command_position = None
        self.last_command_quaternion = None
        self.last_command_gripper = None
        self.last_tick = time.monotonic()
        self.command_count = 0
        self.events = []
        self.inference_latencies = []
        self.command_positions = []
        self.command_grippers = []
        self.feedback_position_min = self.feedback_position_max = None
        self.feedback_gripper_min = self.feedback_gripper_max = None
        self.report_root = Path(report_root) if report_root else None

        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE)
        self.arm_publisher = self.create_publisher(
            Carteposcustom, "/rm_driver/movep_canfd_custom_cmd", command_qos)
        self.gripper_publisher = self.create_publisher(
            Float32, "/motion_control/gripper_command", command_qos)
        self.stop_publisher = self.create_publisher(
            Empty, "/rm_driver/move_stop_cmd", command_qos)
        self.status_publisher = self.create_publisher(
            String, "/fastumi/real/status", 10)
        self.create_subscription(
            PolicyActionSequence, "/fastumi/policy/action_sequence",
            self.on_sequence, command_qos)
        self.create_subscription(JointState, "/joint_states", self.on_joint, 10)
        self.create_subscription(Pose, "/rm_driver/udp_arm_position", self.on_arm, 10)
        self.create_subscription(Float32, "/motion_control/gripper_state", self.on_gripper, 10)
        self.create_subscription(Rmerr, "/rm_driver/udp_rm_err", self.on_error, 10)
        self.create_timer(0.01, self.tick)
        self.create_timer(1.0, self.publish_status)
        mode = "motion-capable but locked" if self.enable_motion else "dry-run"
        self.get_logger().warning(
            f"FastUMI real executor ready in {mode}; t=self-test, a=one policy sequence, "
            "s=stop, q=quit")

    def record_inference(self, message, inference_seconds):
        """接收同进程推理节点提供的准确耗时，用于预热门控。"""
        with self.lock:
            sequence_id = int(message.sequence_id)
            self.inference_metadata[sequence_id] = float(inference_seconds)
            # result_observer 在 DDS 发布前调用；把在途结果纳入“下一条”边界，
            # 确保按键前已开始的预测不会被当成按键后的新序列执行。
            self.latest_sequence_id = max(self.latest_sequence_id, sequence_id)

    def on_joint(self, message):
        """验证并保存七轴反馈。"""
        received = time.monotonic()
        if received - self.joint_received < 0.02:
            return
        try:
            mapping = dict(zip(message.name, message.position))
            values = np.array([mapping[name] for name in JOINT_NAMES], dtype=np.float64)
            if not np.isfinite(values).all() or np.any(values < JOINT_MIN) or np.any(values > JOINT_MAX):
                raise ValueError("joint feedback outside RM75 limits")
        except (KeyError, ValueError, TypeError):
            self._fault("invalid RM75 joint feedback")
            return
        with self.lock:
            self.joints = values
            self.joint_received = received
            self._update_fk_match()

    def on_arm(self, message):
        """保存驱动报告的 Link7/法兰绝对姿态。"""
        received = time.monotonic()
        if received - self.arm_received < 0.02:
            return
        try:
            position, quaternion = pose_arrays(message)
        except ValueError:
            self._fault("invalid RM75 Cartesian feedback")
            return
        with self.lock:
            self.arm_position, self.arm_quaternion = position, quaternion
            self.arm_received = received
            self.feedback_position_min = (position.copy() if self.feedback_position_min is None
                                          else np.minimum(self.feedback_position_min, position))
            self.feedback_position_max = (position.copy() if self.feedback_position_max is None
                                          else np.maximum(self.feedback_position_max, position))
            self._update_fk_match()

    def on_gripper(self, message):
        """保存归一化夹爪实测开度。"""
        value = float(message.data)
        if not np.isfinite(value) or not 0 <= value <= 1:
            self._fault("invalid gripper feedback")
            return
        with self.lock:
            self.gripper = value
            self.gripper_received = time.monotonic()
            self.feedback_gripper_min = (value if self.feedback_gripper_min is None
                                         else min(self.feedback_gripper_min, value))
            self.feedback_gripper_max = (value if self.feedback_gripper_max is None
                                         else max(self.feedback_gripper_max, value))

    def on_error(self, message):
        """RM 驱动以 err_len=1, err=[0] 表示正常；仅非零值算故障。"""
        errors = [int(value) for value in message.err]
        with self.lock:
            self.error_received = time.monotonic()
            self.robot_error_clear = not any(errors)
            if not self.rm_errors_seen or self.rm_errors_seen[-1] != errors:
                self.rm_errors_seen.append(errors)
        if any(errors):
            self._fault(f"RM75 driver error: {errors}")

    def _update_fk_match(self):
        if self.joints is None or self.arm_position is None:
            return
        now = time.monotonic()
        if now - self.last_fk_check < 0.02:
            return
        self.last_fk_check = now
        transform = self.fk.forward(self.joints)
        fk_position = transform[:3, 3]
        fk_quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
        translation, rotation = pose_error(
            fk_position, fk_quaternion, self.arm_position, self.arm_quaternion)
        self.fk_translation_error_max = max(self.fk_translation_error_max, translation)
        self.fk_rotation_error_max = max(self.fk_rotation_error_max, rotation)
        self.fk_match_count = self.fk_match_count + 1 if (
            translation <= 0.002 and rotation <= np.deg2rad(1.0)) else 0

    def _inputs_ready_reason(self, require_policy=True):
        """返回当前拒绝原因；确定性自检执行时不依赖正在暂停的策略推理。"""
        now = time.monotonic()
        ages = {
            "joint feedback": now - self.joint_received,
            "arm feedback": now - self.arm_received,
            "gripper feedback": now - self.gripper_received,
            "RM error feedback": now - self.error_received,
        }
        limits = {"joint feedback": 0.1, "arm feedback": 0.1,
                  "gripper feedback": 0.2, "RM error feedback": 0.1}
        for name, age in ages.items():
            if age > limits[name]:
                return f"{name} missing or stale"
        if not self.robot_error_clear:
            return "RM75 reports a nonzero error"
        if self.fk_match_count < 5:
            return "waiting for five FK/driver agreement samples"
        if require_policy:
            if len(self.valid_predictions) < 2:
                return "waiting for two valid policy predictions"
            if any(latency >= 0.45 for _, latency in self.valid_predictions[-2:]):
                return "recent policy inference latency is not below 0.45 s"
            if now - self.valid_predictions[-1][0] > 1.0:
                return "latest policy prediction is stale"
        if self.arm_publisher.get_subscription_count() != 1:
            return "RM75 Cartesian command subscriber is unavailable or duplicated"
        if self.gripper_publisher.get_subscription_count() != 1:
            return "gripper command subscriber is unavailable or duplicated"
        if self.count_publishers("/rm_driver/movep_canfd_custom_cmd") != 1:
            return "multiple RM75 Cartesian command publishers detected"
        if self.count_publishers("/motion_control/gripper_command") != 1:
            return "multiple gripper command publishers detected"
        if self.count_publishers("/fastumi/policy/action_sequence") != 1:
            return "policy sequence publisher is unavailable or duplicated"
        return None

    def _validate_targets(self, positions, quaternions, start_position, start_quaternion):
        if (not np.isfinite(positions).all() or not np.isfinite(quaternions).all()
                or np.any(positions < WORKSPACE_MIN) or np.any(positions > WORKSPACE_MAX)):
            raise ValueError("policy target is non-finite or outside workspace")
        for position, quaternion in zip(positions, quaternions):
            translation, rotation = pose_error(
                start_position, start_quaternion, position, quaternion)
            if translation > 0.03 or rotation > np.deg2rad(10):
                raise ValueError("policy target exceeds the start-pose smoke envelope")
        previous_position, previous_quaternion = start_position, start_quaternion
        for position, quaternion in zip(positions, quaternions):
            translation, rotation = pose_error(
                previous_position, previous_quaternion, position, quaternion)
            if translation > 0.03 or rotation > np.deg2rad(10):
                raise ValueError("adjacent policy targets exceed the smoke step limit")
            previous_position, previous_quaternion = position, quaternion

    def on_sequence(self, message):
        """校验预测、记录预热，并仅在已武装时接收下一条新序列。"""
        now_ns = self.get_clock().now().nanoseconds
        sequence_id = int(message.sequence_id)
        if sequence_id <= self.last_observed_sequence_id:
            return
        if message.header.frame_id != "base_link" or message.end_frame != "Link7":
            self._fault("policy sequence frame contract mismatch")
            return
        count = len(message.poses)
        if count != 16 or len(message.time_from_start) != count or len(message.gripper_openness) != count:
            self._fault("policy sequence arrays must all contain 16 values")
            return
        base_ns = (int(message.header.stamp.sec) * 1_000_000_000
                   + int(message.header.stamp.nanosec))
        offsets = np.array([
            int(value.sec) * 1_000_000_000 + int(value.nanosec)
            for value in message.time_from_start], dtype=np.int64)
        times = base_ns + offsets
        try:
            decoded = [pose_arrays(value) for value in message.poses]
            positions = np.stack([value[0] for value in decoded])
            quaternions = np.stack([value[1] for value in decoded])
            grippers = np.asarray(message.gripper_openness, dtype=np.float64)
            expected_offsets = np.rint(np.arange(16) / 30.0 * 1e9).astype(np.int64)
            if (not np.array_equal(offsets, expected_offsets)
                    or not np.isfinite(grippers).all()
                    or np.any(grippers < 0) or np.any(grippers > 1)):
                raise ValueError("invalid policy times or gripper targets")
        except ValueError as error:
            self._fault(str(error))
            return
        latency = self.inference_metadata.pop(sequence_id, None)
        if latency is None:
            latency = (now_ns - (
                int(message.header.stamp.sec) * 1_000_000_000
                + int(message.header.stamp.nanosec))) / 1e9
        with self.lock:
            self.last_observed_sequence_id = sequence_id
            self.latest_sequence_id = max(self.latest_sequence_id, sequence_id)
            if np.isfinite(latency) and latency >= 0:
                self.inference_latencies.append(float(latency))
                self.valid_predictions.append((time.monotonic(), float(latency)))
                self.valid_predictions = self.valid_predictions[-10:]
            if self.phase != "armed" or sequence_id <= self.arm_after_sequence_id:
                return
            future = times > now_ns
            if not np.any(future) or now_ns - base_ns > 500_000_000:
                self._fault("policy sequence is fully expired")
                return
            try:
                self._validate_targets(
                    positions, quaternions, self.arm_position, self.arm_quaternion)
            except ValueError as error:
                self._fault(str(error))
                return
            self.plan = ExecutionPlan(
                times[future], positions[future], quaternions[future], grippers[future],
                now_ns, self.arm_position.copy(), self.arm_quaternion.copy(),
                float(self.gripper), "policy", sequence_id)
            self.events.append({"event": "policy_received", "sequence_id": sequence_id})
            self.get_logger().warning(
                f"Policy sequence {sequence_id} staged with {int(np.sum(future))} future steps; "
                "waiting for refreshed hardware feedback")

    def arm_policy(self):
        """锁定到下一条新预测；不会执行按键前已经产生的序列。"""
        with self.lock:
            if not self.enable_motion:
                return False, "start with --enable-motion to permit hardware output"
            if self.phase != "locked":
                return False, f"executor is {self.phase}"
            if not self.self_test_completed:
                return False, "complete the t self-test before arming a policy sequence"
            reason = self._inputs_ready_reason()
            if reason:
                return False, reason
            self.phase = "armed"
            self.arm_after_sequence_id = self.latest_sequence_id
            self.armed_deadline = time.monotonic() + 2.0
            self.stop_reason = "waiting for next policy sequence"
            return True, self.stop_reason

    def should_run_inference(self):
        """执行中或已有待执行策略时暂停推理，为 100 Hz 控制和反馈让出 ROS 线程。"""
        with self.lock:
            return self.phase != "executing" and not (
                self.phase == "armed" and self.plan is not None)

    def queue_operator_command(self, command):
        """排队 t/a 请求，等待 ROS 线程刷新反馈后再执行完整安全门控。"""
        if command not in ("t", "a"):
            raise ValueError("operator command must be 't' or 'a'")
        with self.lock:
            if not self.enable_motion:
                return False, "start with --enable-motion to permit hardware output"
            if self.phase != "locked":
                return False, f"executor is {self.phase}"
            if command == "a" and not self.self_test_completed:
                return False, "complete the t self-test before arming a policy sequence"
            self.pending_operator_command = command
            self.pending_operator_deadline = time.monotonic() + 2.0
            label = "self-test" if command == "t" else "one policy sequence"
            return True, f"{label} queued; waiting for fresh safety inputs"

    def _process_operator_command(self):
        """只在 ROS 执行器线程中兑现排队请求，避免推理期间反馈假过期。"""
        command = self.pending_operator_command
        if command is None:
            return
        reason = self._inputs_ready_reason()
        if time.monotonic() > self.pending_operator_deadline:
            self.pending_operator_command = None
            detail = reason or "request deadline expired before execution"
            self.get_logger().error(f"Operator command rejected: {detail}")
            return
        if reason is not None:
            return
        self.pending_operator_command = None
        if command == "t":
            success, message = self.start_self_test()
        else:
            success, message = self.arm_policy()
        log = self.get_logger().warning if success else self.get_logger().error
        log(("Operator command accepted: " if success else "Operator command rejected: ") + message)

    def start_self_test(self):
        """构造 +Z 1 mm 往返以及安全张开 0.05 的确定性测试计划。"""
        with self.lock:
            if not self.enable_motion:
                return False, "start with --enable-motion to permit hardware output"
            if self.phase != "locked":
                return False, f"executor is {self.phase}"
            reason = self._inputs_ready_reason()
            if reason:
                return False, reason
            now_ns = self.get_clock().now().nanoseconds
            start = self.arm_position.copy()
            raised = start.copy()
            raised[2] += 0.001
            test_gripper = min(1.0, self.gripper + 0.05) if self.gripper <= 0.90 else self.gripper
            self.plan = ExecutionPlan(
                np.array([now_ns + 500_000_000, now_ns + 1_000_000_000]),
                np.stack([raised, start]),
                np.stack([self.arm_quaternion, self.arm_quaternion]),
                np.array([test_gripper, self.gripper]), now_ns, start,
                self.arm_quaternion.copy(), float(self.gripper), "self_test")
            self.phase = "executing"
            self.events.append({"event": "self_test_started"})
            return True, "1 mm +Z / 0.05 gripper self-test started"

    def _publish_target(self, position, quaternion, gripper):
        command = Carteposcustom()
        command.pose.position.x, command.pose.position.y, command.pose.position.z = map(float, position)
        (command.pose.orientation.x, command.pose.orientation.y,
         command.pose.orientation.z, command.pose.orientation.w) = map(float, quaternion)
        command.follow = True
        command.trajectory_mode = 0
        command.radio = 0
        self.arm_publisher.publish(command)
        self.gripper_publisher.publish(Float32(data=float(gripper)))
        self.command_count += 1
        self.command_positions.append(np.asarray(position).tolist())
        self.command_grippers.append(float(gripper))

    def tick(self):
        """100 Hz 执行计划；任何反馈、跟踪或时间异常均 fail closed。"""
        with self.lock:
            now_mono = time.monotonic()
            dt = float(np.clip(now_mono - self.last_tick, 0.001, 0.03))
            self.last_tick = now_mono
            self._process_operator_command()
            if self.phase not in ("armed", "executing"):
                return
            if self.phase == "armed":
                reason = self._inputs_ready_reason()
                if reason:
                    if "missing or stale" in reason and now_mono <= self.armed_deadline:
                        return
                    self._fault(reason)
                    return
                if self.plan is None:
                    if now_mono > self.armed_deadline:
                        self._fault("next policy sequence timed out")
                    return
                now_ns = self.get_clock().now().nanoseconds
                if now_ns > int(self.plan.times_ns[-1]):
                    self._fault("policy sequence is fully expired before execution")
                    return
                self.phase = "executing"
                self.events.append({
                    "event": "policy_started", "sequence_id": self.plan.sequence_id})
                self.get_logger().warning(
                    f"Executing one policy sequence {self.plan.sequence_id}")
                return
            reason = self._inputs_ready_reason(
                require_policy=self.plan is not None and self.plan.source == "policy")
            if reason:
                self._fault(reason)
                return
            if self.phase == "armed":
                return
            if self.plan is None:
                self._fault("execution plan missing")
                return
            now_ns = self.get_clock().now().nanoseconds
            if now_ns > int(self.plan.times_ns[-1]):
                if self.plan.source == "self_test":
                    self.self_test_completed = True
                    self.events.append({"event": "self_test_completed"})
                self._stop("single sequence complete")
                return
            target_position, target_quaternion, target_gripper = sample_plan(self.plan, now_ns)
            position = (self.arm_position.copy() if self.last_command_position is None
                        else self.last_command_position)
            quaternion = (self.arm_quaternion.copy() if self.last_command_quaternion is None
                          else self.last_command_quaternion)
            position, quaternion = rate_limit_pose(
                position, quaternion, target_position, target_quaternion,
                0.020 * dt, 0.2 * dt)
            current_gripper = (self.gripper if self.last_command_gripper is None
                               else self.last_command_gripper)
            gripper_step = 0.25 * dt
            gripper = float(current_gripper + np.clip(
                target_gripper - current_gripper, -gripper_step, gripper_step))
            if self.last_command_position is not None:
                translation, rotation = pose_error(
                    self.arm_position, self.arm_quaternion,
                    self.last_command_position, self.last_command_quaternion)
                if translation > 0.020 or rotation > np.deg2rad(10):
                    self._fault("RM75 command/feedback tracking error exceeded limit")
                    return
            try:
                self._validate_targets(
                    position[None], quaternion[None],
                    self.plan.anchor_position, self.plan.anchor_quaternion)
            except ValueError as error:
                self._fault(str(error))
                return
            self._publish_target(position, quaternion, gripper)
            self.last_command_position = position
            self.last_command_quaternion = quaternion
            self.last_command_gripper = gripper

    def _fault(self, reason):
        with self.lock:
            if self.phase in ("armed", "executing"):
                self._stop(reason, fault=True)

    def _stop(self, reason, fault=False, force=False):
        active = self.phase in ("armed", "executing")
        if self.enable_motion and (active or force):
            self.stop_publisher.publish(Empty())
            if self.gripper is not None:
                self.gripper_publisher.publish(Float32(data=float(self.gripper)))
        self.phase = "locked"
        self.armed_deadline = 0.0
        self.pending_operator_command = None
        self.plan = None
        self.last_command_position = self.last_command_quaternion = None
        self.last_command_gripper = None
        self.stop_reason = reason
        self.events.append({"event": "fault" if fault else "stop", "reason": reason})
        if fault:
            self.get_logger().error(f"Real executor locked: {reason}")
        else:
            self.get_logger().warning(f"Real executor locked: {reason}")

    def emergency_stop(self, reason="operator stop"):
        """人工停止并强制发送 RM 驱动停止消息。"""
        with self.lock:
            self._stop(reason, force=True)

    def publish_status(self):
        """发布便于操作员核对的紧凑 JSON 状态。"""
        with self.lock:
            readiness_reason = self._inputs_ready_reason()
            status = {
                "phase": self.phase, "motion_enabled": self.enable_motion,
                "ready": readiness_reason is None,
                "readiness_reason": readiness_reason,
                "pending_operator_command": self.pending_operator_command,
                "self_test_completed": self.self_test_completed,
                "fk_match_count": self.fk_match_count,
                "valid_predictions": len(self.valid_predictions),
                "last_inference_s": (
                    None if not self.inference_latencies
                    else round(self.inference_latencies[-1], 4)),
                "commands": self.command_count, "stop_reason": self.stop_reason,
            }
        self.status_publisher.publish(String(data=json.dumps(status)))

    def write_report(self):
        """把本次 dry-run/实机 smoke 证据写到 dataset 下。"""
        if self.report_root is None:
            return None
        output = self.report_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output.mkdir(parents=True, exist_ok=False)
        report = {
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint": self.checkpoint_summary,
            "motion_enabled": self.enable_motion,
            "phase": self.phase,
            "stop_reason": self.stop_reason,
            "self_test_completed": self.self_test_completed,
            "inference_latencies_s": self.inference_latencies,
            "valid_prediction_count": len(self.inference_latencies),
            "fk_translation_error_max_m": self.fk_translation_error_max,
            "fk_rotation_error_max_rad": self.fk_rotation_error_max,
            "rm_errors_seen": self.rm_errors_seen,
            "command_count": self.command_count,
            "command_position_min": (
                np.min(self.command_positions, axis=0).tolist() if self.command_positions else None),
            "command_position_max": (
                np.max(self.command_positions, axis=0).tolist() if self.command_positions else None),
            "command_gripper_min": min(self.command_grippers) if self.command_grippers else None,
            "command_gripper_max": max(self.command_grippers) if self.command_grippers else None,
            "feedback_position_min": (
                self.feedback_position_min.tolist()
                if self.feedback_position_min is not None else None),
            "feedback_position_max": (
                self.feedback_position_max.tolist()
                if self.feedback_position_max is not None else None),
            "feedback_gripper_min": self.feedback_gripper_min,
            "feedback_gripper_max": self.feedback_gripper_max,
            "events": self.events,
        }
        path = output / "report.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return path

    def destroy_node(self):
        if hasattr(self, "phase") and self.phase in ("armed", "executing"):
            self.emergency_stop("node shutdown")
        return super().destroy_node()
