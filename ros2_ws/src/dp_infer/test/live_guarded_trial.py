"""在回位状态运行限时 DP 试验，并监控腕部画面和 RM75 反馈。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import cv2
import numpy as np
import rclpy
from fastumi_interfaces.msg import PolicyActionSequence
from rm_ros_interfaces.msg import Jointpos
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Empty, Float32
from std_srvs.srv import Trigger

from diffusion_policy.common.urdf_kinematics import UrdfKinematics
from dp_infer.visual_guard import ball_center_bgr


HOME_JOINTS = np.deg2rad([0, 20, 0, 70, 0, 90, 90])
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))


def outside_trial_region(xyz, home_xyz, max_displacement_m=0.04) -> bool:
    """限制 Link7 相对回位点的米制位移；基座 x 的符号不代表是否靠近球。"""
    displacement = np.asarray(xyz, dtype=np.float64) - np.asarray(home_xyz, dtype=np.float64)
    return bool(
        not np.isfinite(displacement).all()
        or np.linalg.norm(displacement) > max_displacement_m
        or displacement[2] > 0.025
    )


def target_lost_during_control(real, first_command_at, ball, ball_at, now) -> bool:
    """预热阶段不因断帧终止；首条命令后要求最近一次有效球检测不超过 300 ms。"""
    return bool(real and first_command_at is not None
                and (ball is None or now - ball_at > 0.3))


def request_guarded_stop(monitor, task_stop, real):
    """实机先直发驱动停机，再异步请求任务服务确认。"""
    if real:
        monitor.request_stop()
    if task_stop.service_is_ready():
        return task_stop.call_async(Trigger.Request())
    return None


def ball_center(image: Image) -> tuple[float, float] | None:
    """读取 ROS RGB/BGR 帧并复用部署中的球体定位逻辑。"""
    if image.encoding not in ("bgr8", "rgb8") or image.step < image.width * 3:
        return None
    values = np.frombuffer(image.data, dtype=np.uint8)
    if values.size != image.height * image.step:
        return None
    pixels = values.reshape(image.height, image.step)[:, : image.width * 3]
    pixels = pixels.reshape(image.height, image.width, 3)
    if image.encoding == "rgb8":
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    return ball_center_bgr(pixels)


class TrialMonitor(Node):
    """只读追踪实测七轴、夹爪、视频和策略；实机停止命令由调用方触发。"""

    def __init__(self, urdf: Path, real: bool):
        super().__init__("dp_guarded_trial_monitor")
        self.fk = UrdfKinematics(urdf, JOINT_NAMES)
        self.home_xyz = self.fk.forward(HOME_JOINTS)[:3, 3]
        self.real = real
        self.joints = None
        self.xyz = None
        self.joint_at = float("-inf")
        self.gripper = None
        self.min_gripper_feedback = 1.0
        self.gripper_at = float("-inf")
        self.valid = False
        self.valid_at = float("-inf")
        self.ball = None
        self.ball_at = float("-inf")
        self.image_size = None
        self.image_count = 0
        self.last_image_at = float("-inf")
        self.image_gaps = []
        self.active_image_gaps = []
        self.active_ball_misses = 0
        self.policies = []
        self.arm_commands = 0
        self.gripper_commands = 0
        self.debug_commands = 0
        self.stop_commands = 0
        self.stop_requests = 0
        self.stop_results = []
        self.first_command_at = None
        self.control_end_at = None
        self.last_arm_command_at = None
        self.last_debug_command_at = None
        self.close_allowed = False
        self.close_approval_at = float("-inf")
        self.close_approval_true_count = 0
        self.last_gripper_command = None
        self.min_gripper_command = 1.0
        self.unsafe_gripper_command = False
        self.create_subscription(JointState, "/joint_states", self.on_joint, qos_profile_sensor_data)
        self.create_subscription(Float32, "/motion_control/gripper_state", self.on_gripper, 10)
        self.create_subscription(Bool, "/rm_driver/udp_feedback_valid", self.on_valid, 10)
        self.create_subscription(Image, "/wrist_camera/image_decoded", self.on_image, qos_profile_sensor_data)
        self.create_subscription(PolicyActionSequence, "/fastumi/policy/action_sequence", self.on_policy, 10)
        self.create_subscription(Bool, "/fastumi/policy/gripper_close_allowed", self.on_close_approval, 10)
        self.create_subscription(Jointpos, "/rm_driver/movej_canfd_cmd", self.on_arm_command, 10)
        self.create_subscription(Float32, "/motion_control/gripper_command", self.on_gripper_command, 10)
        self.create_subscription(JointState, "/fastumi/rm75/placo/joint_command", self.on_debug, 10)
        self.create_subscription(Empty, "/rm_driver/move_stop_cmd", self.on_stop, 10)
        self.create_subscription(Bool, "/rm_driver/move_stop_result",
                                 lambda message: self.stop_results.append(bool(message.data)), 10)
        self.stop_publisher = self.create_publisher(Empty, "/rm_driver/move_stop_cmd", 10)

    def on_joint(self, message):
        by_name = dict(zip(message.name, message.position))
        if any(name not in by_name for name in JOINT_NAMES):
            return
        positions = np.asarray([by_name[name] for name in JOINT_NAMES], dtype=np.float64)
        if not np.isfinite(positions).all():
            return
        self.joints = positions
        self.joint_at = time.monotonic()
        self.xyz = self.fk.forward(positions)[:3, 3]

    def on_gripper(self, message):
        value = float(message.data)
        self.gripper = value if np.isfinite(value) and 0 <= value <= 1 else None
        if self.gripper is not None:
            self.min_gripper_feedback = min(self.min_gripper_feedback, self.gripper)
        self.gripper_at = time.monotonic()

    def on_valid(self, message):
        self.valid = bool(message.data)
        self.valid_at = time.monotonic()

    def on_image(self, message):
        now = time.monotonic()
        if np.isfinite(self.last_image_at):
            gap = now - self.last_image_at
            self.image_gaps.append(gap)
            if (self.first_command_at is not None and self.control_end_at is None
                    and self.last_image_at >= self.first_command_at):
                self.active_image_gaps.append(gap)
        self.last_image_at = now
        self.image_count += 1
        detected_ball = ball_center(message)
        if detected_ball is not None:
            # 单帧漏检不代表目标消失；从最近一次有效识别起计 300 ms。
            self.ball = detected_ball
            self.ball_at = time.monotonic()
        elif self.first_command_at is not None and self.control_end_at is None:
            self.active_ball_misses += 1
        self.image_size = (message.width, message.height)

    def on_policy(self, message):
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        age_ms = (self.get_clock().now().nanoseconds - stamp) / 1e6
        first, last = message.poses[0], message.poses[-1]
        delta_mm = [
            (getattr(last.position, axis) - getattr(first.position, axis)) * 1000
            for axis in ("x", "y", "z")
        ]
        self.policies.append({
            "id": int(message.sequence_id),
            "age_ms": round(age_ms, 1),
            "delta_mm": np.round(delta_mm, 1).tolist(),
            "min_gripper": round(min(message.gripper_openness), 3),
        })

    def on_arm_command(self, _message):
        self.arm_commands += 1
        self.last_arm_command_at = time.monotonic()
        if self.real and self.first_command_at is None:
            self.first_command_at = self.last_arm_command_at

    def on_close_approval(self, message):
        self.close_allowed = bool(message.data)
        self.close_approval_at = time.monotonic()
        self.close_approval_true_count += int(self.close_allowed)

    def on_gripper_command(self, message):
        self.gripper_commands += 1
        commanded = float(message.data)
        if not np.isfinite(commanded) or not 0 <= commanded <= 1:
            self.unsafe_gripper_command = True
            return
        previous = self.last_gripper_command
        if previous is None:
            previous = self.gripper
        if (previous is not None and commanded < previous - 0.02
                and (not self.close_allowed
                     or time.monotonic() - self.close_approval_at > 0.25
                     or self.first_command_at is None
                     or time.monotonic() - self.first_command_at < 1.5)):
            self.unsafe_gripper_command = True
        self.last_gripper_command = commanded
        self.min_gripper_command = min(self.min_gripper_command, commanded)

    def on_debug(self, _message):
        self.debug_commands += 1
        self.last_debug_command_at = time.monotonic()
        if not self.real and self.first_command_at is None:
            self.first_command_at = self.last_debug_command_at

    def on_stop(self, _message):
        self.stop_commands += 1

    def request_stop(self):
        """发出 RM75 停止命令并记录发出次数，不依赖本节点是否收到回环消息。"""
        self.stop_requests += 1
        self.stop_publisher.publish(Empty())

    def inputs_ready(self, now: float) -> bool:
        """要求机械臂、夹爪及最近一次有效球检测同时新鲜。"""
        return bool(self.joints is not None and now - self.joint_at <= 0.3
                    and self.gripper is not None and now - self.gripper_at <= 0.3
                    and self.valid and now - self.valid_at <= 0.3
                    and self.ball is not None and now - self.ball_at <= 0.3)

    def preflight(self, now: float) -> None:
        """拒绝未回位、缺流或其他命令发布者存在的实机启动。"""
        if not self.inputs_ready(now):
            raise RuntimeError("arm/gripper/video feedback is missing or stale")
        if np.max(np.abs(self.joints - HOME_JOINTS)) > 0.035 or self.gripper < 0.95:
            raise RuntimeError("RM75 and gripper must return to the open start state")
        for topic in ("/rm_driver/movej_cmd", "/rm_driver/movej_canfd_cmd",
                      "/motion_control/gripper_command", "/fastumi/policy/action_sequence"):
            if self.get_publishers_info_by_topic(topic):
                raise RuntimeError(f"another command publisher is active: {topic}")


def terminate(process: subprocess.Popen | None) -> int | None:
    """先让 launch 向子节点发送 SIGINT；超时才逐级升级。"""
    if process is None:
        return None
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real", action="store_true", help="明确启用实机控制；默认 dry-run")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--control-seconds", type=float, default=3.0)
    parser.add_argument("--max-displacement-m", type=float, default=0.04)
    parser.add_argument("--max-command-displacement-m", type=float)
    args = parser.parse_args()
    if (not 0 < args.control_seconds <= 8 or not 1 <= args.steps <= 16
            or not 0 < args.max_displacement_m <= 0.20):
        raise ValueError("trial limits must be 0 < seconds <= 8, 1 <= steps <= 16, "
                         "0 < displacement <= 0.20 m")
    if args.max_command_displacement_m is None:
        args.max_command_displacement_m = max(0.005, args.max_displacement_m - 0.03)
    if not 0 < args.max_command_displacement_m < args.max_displacement_m:
        raise ValueError("command displacement must be below the feedback limit")
    for path in (args.checkpoint, args.urdf):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    root = Path(__file__).resolve().parents[4]
    launch_command = [
        "ros2", "launch", "dp_infer", "dp_infer.launch.py",
        f"checkpoint:={args.checkpoint.resolve()}",
        f"urdf_path:={args.urdf.resolve()}",
        "image_type:=raw", "image_topic:=/wrist_camera/image_decoded",
        f"num_inference_steps:={args.steps}",
        f"dry_run:={'false' if args.real else 'true'}",
        f"max_start_displacement_m:={args.max_command_displacement_m}",
        "max_start_rise_m:=0.015",
    ]
    decoder_command = [
        "ros2", "launch", "fastumi_usb_camera", "receive.launch.py",
        "input_topic:=/wrist_camera/image_raw",
        "output_topic:=/wrist_camera/image_decoded",
    ]
    decoder = controller = None
    # 保持 ROS 上下文有效直至 finally 发出停机命令；默认 rclpy 信号处理会过早关闭它。
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    previous_int_handler = signal.getsignal(signal.SIGINT)
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    shutdown_requested = False

    def request_shutdown(_signum, _frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    monitor = None
    baseline_ball = None
    reason = "not_started"
    start = time.monotonic()
    report = {}
    try:
        monitor = TrialMonitor(args.urdf, args.real)
        with args.output.with_suffix(".decoder.log").open("w") as decoder_log, args.output.with_suffix(".launch.log").open("w") as launch_log:
            decoder = subprocess.Popen(decoder_command, cwd=root, env=environment,
                                       stdout=decoder_log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                rclpy.spin_once(monitor, timeout_sec=0.05)
                if monitor.inputs_ready(time.monotonic()):
                    break
                if decoder.poll() is not None:
                    raise RuntimeError("video decoder exited before a valid frame")
            monitor.preflight(time.monotonic())
            baseline_ball = monitor.ball
            goal_pixel = np.array([monitor.image_size[0] * 0.5, monitor.image_size[1] * 0.67])
            baseline_error = np.linalg.norm(np.asarray(baseline_ball) - goal_pixel)
            controller = subprocess.Popen(launch_command, cwd=root, env=environment,
                                          stdout=launch_log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            task_start = monitor.create_client(Trigger, "/fastumi/policy/start_task")
            task_stop = monitor.create_client(Trigger, "/fastumi/policy/stop_task")
            deadline = time.monotonic() + 60
            while not task_start.service_is_ready() and time.monotonic() < deadline:
                rclpy.spin_once(monitor, timeout_sec=0.05)
                if controller.poll() is not None:
                    raise RuntimeError("policy launch exited before start_task was ready")
            if not task_start.service_is_ready():
                raise TimeoutError("start_task service did not appear")
            start_future = task_start.call_async(Trigger.Request())
            deadline = time.monotonic() + 5
            while not start_future.done() and time.monotonic() < deadline:
                rclpy.spin_once(monitor, timeout_sec=0.05)
            if not start_future.done() or not start_future.result().success:
                raise RuntimeError("start_task failed before guarded trial")
            deadline = time.monotonic() + 60
            while True:
                rclpy.spin_once(monitor, timeout_sec=0.03)
                now = time.monotonic()
                if controller.poll() is not None:
                    reason = "policy_launch_exited"
                    break
                if args.real and monitor.unsafe_gripper_command:
                    reason = "gripper_closed_without_fresh_visual_approval"
                    break
                if monitor.first_command_at is None:
                    if args.real and monitor.stop_commands:
                        reason = "controller_stopped_before_first_arm_command"
                        break
                    if now >= deadline:
                        reason = "no_ik_or_arm_command_before_timeout"
                        break
                    continue
                if target_lost_during_control(
                        args.real, monitor.first_command_at,
                        monitor.ball, monitor.ball_at, now):
                    reason = "ball_lost_from_wrist_image"
                    break
                if now - monitor.first_command_at >= args.control_seconds:
                    reason = "time_limit"
                    break
                if (args.real and monitor.last_arm_command_at is not None
                        and now - monitor.last_arm_command_at > 0.5):
                    reason = "arm_command_stalled"
                    break
                if (not args.real and monitor.last_debug_command_at is not None
                        and now - monitor.last_debug_command_at > 0.5):
                    reason = "debug_command_stalled"
                    break
                if now - monitor.joint_at > 0.3 or not monitor.valid or now - monitor.valid_at > 0.3:
                    reason = "joint_feedback_lost"
                    break
                if monitor.gripper is None or now - monitor.gripper_at > 0.3 or monitor.gripper < 0.30:
                    reason = "gripper_feedback_invalid"
                    break
                if not args.real:
                    continue
                if outside_trial_region(monitor.xyz, monitor.home_xyz,
                                        args.max_displacement_m):
                    reason = "arm_left_small_trial_region"
                    break
            stop_future = request_guarded_stop(monitor, task_stop, args.real)
            control_end_at = time.monotonic()
            monitor.control_end_at = control_end_at
            xyz_at_stop = monitor.xyz.copy() if monitor.xyz is not None else None
            image_age_ms_at_stop = round((control_end_at - monitor.last_image_at) * 1000, 1)
            ball_age_ms_at_stop = round((control_end_at - monitor.ball_at) * 1000, 1)
            if stop_future is not None:
                stop_deadline = time.monotonic() + 4
                while not stop_future.done() and time.monotonic() < stop_deadline:
                    rclpy.spin_once(monitor, timeout_sec=0.05)
            controller_exit = terminate(controller)
            controller = None
            if args.real:
                # 保留解码器约 1 秒以采集停稳后的目标位置，并重复发送停止命令。
                for index in range(20):
                    if index % 5 == 0:
                        monitor.request_stop()
                    rclpy.spin_once(monitor, timeout_sec=0.05)
            decoder_exit = terminate(decoder)
            decoder = None
        report = {
            "mode": "real" if args.real else "dry_run",
            "max_displacement_m": args.max_displacement_m,
            "max_command_displacement_m": args.max_command_displacement_m,
            "reason": reason,
            "elapsed_s": round(time.monotonic() - start, 2),
            "control_s": round(control_end_at - monitor.first_command_at, 2) if monitor.first_command_at else 0,
            "home_xyz": np.round(monitor.home_xyz, 4).tolist(),
            "xyz_at_stop": np.round(xyz_at_stop, 4).tolist() if xyz_at_stop is not None else None,
            "last_xyz": np.round(monitor.xyz, 4).tolist() if monitor.xyz is not None else None,
            "post_stop_drift_m": round(float(np.linalg.norm(monitor.xyz - xyz_at_stop)), 4)
            if xyz_at_stop is not None and monitor.xyz is not None else None,
            "last_gripper": round(monitor.gripper, 4) if monitor.gripper is not None else None,
            "min_gripper_feedback": round(monitor.min_gripper_feedback, 4),
            "last_gripper_command": monitor.last_gripper_command,
            "min_gripper_command": round(monitor.min_gripper_command, 4),
            "close_approval_true_count": monitor.close_approval_true_count,
            "unsafe_gripper_command": monitor.unsafe_gripper_command,
            "baseline_ball_pixel": np.round(baseline_ball, 1).tolist() if baseline_ball else None,
            "last_ball_pixel": np.round(monitor.ball, 1).tolist() if monitor.ball else None,
            "last_ball_error_px": round(float(np.linalg.norm(
                np.asarray(monitor.ball) - goal_pixel)), 1) if monitor.ball else None,
            "baseline_ball_error_px": round(float(baseline_error), 1),
            "image_count": monitor.image_count,
            "image_gap_ms_max": round(max(monitor.image_gaps) * 1000, 1) if monitor.image_gaps else None,
            "image_gap_ms_p95": round(float(np.percentile(monitor.image_gaps, 95)) * 1000, 1) if monitor.image_gaps else None,
            "active_image_gap_ms_max": round(max(monitor.active_image_gaps) * 1000, 1)
            if monitor.active_image_gaps else None,
            "active_ball_misses": monitor.active_ball_misses,
            "image_age_ms_at_stop": image_age_ms_at_stop,
            "ball_age_ms_at_stop": ball_age_ms_at_stop,
            "policies": len(monitor.policies),
            "age_ms_median": round(float(np.median([p["age_ms"] for p in monitor.policies])), 1) if monitor.policies else None,
            "policy_delta_mm_median": np.round(
                np.median([p["delta_mm"] for p in monitor.policies], axis=0), 1
            ).tolist() if monitor.policies else None,
            "policies_predicting_full_close": sum(
                p["min_gripper"] < 0.1 for p in monitor.policies),
            "arm_commands": monitor.arm_commands,
            "gripper_commands": monitor.gripper_commands,
            "debug_commands": monitor.debug_commands,
            "stop_commands": monitor.stop_commands,
            "stop_requests": monitor.stop_requests,
            "stop_results": monitor.stop_results,
            "controller_exit": controller_exit,
            "decoder_exit": decoder_exit,
        }
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if not args.real and (monitor.arm_commands or monitor.gripper_commands or monitor.stop_commands):
            raise RuntimeError("dry-run produced a hardware command")
    finally:
        if args.real and monitor is not None and rclpy.ok():
            try:
                monitor.request_stop()
            except Exception:
                pass
        if controller is not None:
            terminate(controller)
        if decoder is not None:
            terminate(decoder)
        if monitor is not None:
            monitor.destroy_node()
        try:
            rclpy.shutdown()
        finally:
            signal.signal(signal.SIGINT, previous_int_handler)
            signal.signal(signal.SIGTERM, previous_term_handler)


if __name__ == "__main__":
    main()
