"""只读探测当前硬件观测、GPU 推理和视觉闭合许可，不启动运动控制器。"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
from fastumi_interfaces.msg import PolicyActionSequence
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32

from dp_infer.visual_guard import ball_center_bgr
from dp_infer.core import image_to_rgb


class ReadonlyMonitor(Node):
    """仅订阅观测和预测；检查组合 launch 未产生硬件命令发布者。"""

    def __init__(self):
        """订阅腕部图像、七轴反馈、夹爪反馈、视觉许可与策略输出。"""
        super().__init__("dp_live_readonly_probe")
        self.images = 0
        self.packets = 0
        self.image_intervals_ms = []
        self.packet_intervals_ms = []
        self.last_image_at = None
        self.last_packet_at = None
        self.detected = 0
        self.last_ball = None
        self.approvals = []
        self.policies = 0
        self.policy_deltas_mm = []
        self.last_policy_first_xyz = None
        self.last_policy_last_xyz = None
        self.policy_ages_ms = []
        self.coverage_gaps_ms = []
        self.previous_sequence_end_ns = None
        self.joints = 0
        self.grippers = 0
        self.create_subscription(Image, "/wrist_camera/image_decoded", self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(FFMPEGPacket, "/wrist_camera/image_raw/ffmpeg",
                                 self.on_packet, qos_profile_sensor_data)
        self.create_subscription(Bool, "/fastumi/policy/gripper_close_allowed",
                                 lambda message: self.approvals.append(bool(message.data)), 10)
        self.create_subscription(PolicyActionSequence, "/fastumi/policy/action_sequence",
                                 self.on_policy, 10)
        self.create_subscription(JointState, "/joint_states",
                                 lambda _: setattr(self, "joints", self.joints + 1),
                                 qos_profile_sensor_data)
        self.create_subscription(Float32, "/motion_control/gripper_state",
                                 lambda _: setattr(self, "grippers", self.grippers + 1), 10)

    def on_image(self, message):
        """验证当前画面里的目标定位，保留最后一个球心。"""
        import cv2

        now = time.monotonic()
        if self.last_image_at is not None:
            self.image_intervals_ms.append((now - self.last_image_at) * 1000)
        self.last_image_at = now
        self.images += 1
        try:
            rgb = image_to_rgb(message.data, message.height, message.width,
                               message.step, message.encoding)
            ball = ball_center_bgr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except (TypeError, ValueError):
            return
        if ball is not None:
            self.detected += 1
            self.last_ball = [round(value, 1) for value in ball]

    def on_packet(self, _message):
        """记录 H.264 包到达间隔，用于区分采集和解码端断流。"""
        now = time.monotonic()
        if self.last_packet_at is not None:
            self.packet_intervals_ms.append((now - self.last_packet_at) * 1000)
        self.last_packet_at = now
        self.packets += 1

    def on_policy(self, message):
        """记录预测数量及其相对采集时间的年龄。"""
        self.policies += 1
        first, last = message.poses[0], message.poses[-1]
        first_xyz = np.array([first.position.x, first.position.y, first.position.z])
        last_xyz = np.array([last.position.x, last.position.y, last.position.z])
        self.policy_deltas_mm.append(((last_xyz - first_xyz) * 1000).tolist())
        self.last_policy_first_xyz = first_xyz.tolist()
        self.last_policy_last_xyz = last_xyz.tolist()
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        received_ns = self.get_clock().now().nanoseconds
        self.policy_ages_ms.append(round((received_ns - stamp_ns) / 1e6, 1))
        if self.previous_sequence_end_ns is not None and received_ns > self.previous_sequence_end_ns:
            self.coverage_gaps_ms.append(round(
                (received_ns - self.previous_sequence_end_ns) / 1e6, 1))
        last = message.time_from_start[-1]
        self.previous_sequence_end_ns = stamp_ns + last.sec * 1_000_000_000 + last.nanosec


def stop(process):
    """结束本脚本启动的只读子进程。"""
    if process is None:
        return
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    """启动解码和纯推理子进程，采样当前硬件输入但不发布控制指令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if not 3 <= args.seconds <= 30:
        raise ValueError("seconds must be between 3 and 30")
    if not 1 <= args.steps <= 16:
        raise ValueError("steps must be between 1 and 16")
    if not args.checkpoint.is_file() or not args.urdf.is_file():
        raise FileNotFoundError("checkpoint and training URDF must exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[4]
    environment = os.environ.copy()
    decoder_cmd = ["ros2", "launch", "fastumi_usb_camera", "receive.launch.py",
                   "input_topic:=/wrist_camera/image_raw",
                   "output_topic:=/wrist_camera/image_decoded"]
    inference_cmd = [
        "ros2", "launch", "dp_infer", "dp_infer.launch.py",
        f"checkpoint:={args.checkpoint.resolve()}", f"urdf_path:={args.urdf.resolve()}",
        "start_controller:=false", "image_type:=raw",
        "image_topic:=/wrist_camera/image_decoded",
        f"num_inference_steps:={args.steps}",
    ]
    decoder = inference = None
    rclpy.init()
    monitor = ReadonlyMonitor()
    try:
        for topic in ("/rm_driver/movej_canfd_cmd", "/motion_control/gripper_command"):
            if monitor.get_publishers_info_by_topic(topic):
                raise RuntimeError(f"motor command publisher already exists: {topic}")
        with args.output.with_suffix(".decoder.log").open("w") as decoder_log, \
                args.output.with_suffix(".inference.log").open("w") as inference_log:
            decoder = subprocess.Popen(decoder_cmd, cwd=root, env=environment,
                                       stdout=decoder_log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            inference = subprocess.Popen(inference_cmd, cwd=root, env=environment,
                                         stdout=inference_log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            launched_at = time.monotonic()
            ready_deadline = launched_at + 60
            while not monitor.approvals and time.monotonic() < ready_deadline:
                rclpy.spin_once(monitor, timeout_sec=0.05)
                if decoder.poll() is not None or inference.poll() is not None:
                    raise RuntimeError("decoder or inference exited during startup")
            if not monitor.approvals:
                raise TimeoutError("GPU model did not finish startup within 60 seconds")
            startup_s = round(time.monotonic() - launched_at, 2)
            monitor.image_intervals_ms.clear()
            monitor.packet_intervals_ms.clear()
            monitor.last_image_at = None
            monitor.last_packet_at = None
            started = time.monotonic()
            while time.monotonic() - started < args.seconds:
                rclpy.spin_once(monitor, timeout_sec=0.05)
                if decoder.poll() is not None or inference.poll() is not None:
                    raise RuntimeError("decoder or inference exited unexpectedly")
                for topic in ("/rm_driver/movej_canfd_cmd", "/motion_control/gripper_command"):
                    if monitor.get_publishers_info_by_topic(topic):
                        raise RuntimeError(f"unexpected motor command publisher: {topic}")
        report = {
            "mode": "live_readonly", "seconds": args.seconds,
            "num_inference_steps": args.steps,
            "startup_s": startup_s,
            "images": monitor.images, "ball_detected": monitor.detected,
            "h264_packets": monitor.packets,
            "image_gap_ms_max": round(max(monitor.image_intervals_ms), 1)
            if monitor.image_intervals_ms else None,
            "image_gap_ms_p95": round(float(np.percentile(
                monitor.image_intervals_ms, 95)), 1)
            if monitor.image_intervals_ms else None,
            "packet_gap_ms_max": round(max(monitor.packet_intervals_ms), 1)
            if monitor.packet_intervals_ms else None,
            "packet_gap_ms_p95": round(float(np.percentile(
                monitor.packet_intervals_ms, 95)), 1)
            if monitor.packet_intervals_ms else None,
            "last_ball_pixel": monitor.last_ball,
            "close_permission_messages": len(monitor.approvals),
            "close_permission_true": sum(monitor.approvals),
            "policies": monitor.policies,
            "policy_delta_mm_median": np.round(np.median(
                monitor.policy_deltas_mm, axis=0), 1).tolist()
            if monitor.policy_deltas_mm else None,
            "last_policy_first_xyz": np.round(monitor.last_policy_first_xyz, 4).tolist()
            if monitor.last_policy_first_xyz is not None else None,
            "last_policy_last_xyz": np.round(monitor.last_policy_last_xyz, 4).tolist()
            if monitor.last_policy_last_xyz is not None else None,
            "age_ms_min": min(monitor.policy_ages_ms) if monitor.policy_ages_ms else None,
            "age_ms_max": max(monitor.policy_ages_ms) if monitor.policy_ages_ms else None,
            "coverage_gap_count": len(monitor.coverage_gaps_ms),
            "coverage_gap_ms_max": max(monitor.coverage_gaps_ms)
            if monitor.coverage_gaps_ms else 0.0,
            "joints": monitor.joints, "grippers": monitor.grippers,
        }
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if not (monitor.images and monitor.detected and monitor.policies
                and monitor.joints and monitor.grippers and monitor.approvals):
            raise AssertionError("live read-only inference input/output was incomplete")
    finally:
        stop(inference)
        stop(decoder)
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
