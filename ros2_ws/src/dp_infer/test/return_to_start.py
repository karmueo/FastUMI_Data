"""经现场确认后，把 RM75 七轴和夹爪回到本任务起始状态。此脚本会驱动硬件。"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float32


HOME = np.deg2rad([0, 20, 0, 70, 0, 90, 90])
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
MAX_HOME_START_ERROR_DEG = 45.0
MAX_HOME_PATH_DISPLACEMENT_M = 0.20
MAX_HOME_PATH_RISE_M = 0.025
MIN_HOME_PATH_Z_M = 0.20


def validate_homing_start(joints, forward=None):
    """只允许从本任务试验范围内沿安全的关节插值路径回位。"""
    values = np.asarray(joints, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise RuntimeError("current RM75 joints are invalid")
    maximum_error = float(np.rad2deg(np.max(np.abs(values - HOME))))
    if maximum_error > MAX_HOME_START_ERROR_DEG:
        raise RuntimeError(
            f"current RM75 joints are {maximum_error:.2f} degrees from start "
            f"(limit {MAX_HOME_START_ERROR_DEG:.0f})")
    if forward is None:
        return
    home_xyz = np.asarray(forward(HOME)[:3, 3], dtype=np.float64)
    for fraction in np.linspace(0.0, 1.0, 51):
        interpolation = (1.0 - fraction) * values + fraction * HOME
        xyz = np.asarray(forward(interpolation)[:3, 3], dtype=np.float64)
        if (xyz.shape != (3,) or not np.isfinite(xyz).all()
                or np.linalg.norm(xyz - home_xyz) > MAX_HOME_PATH_DISPLACEMENT_M
                or xyz[2] - home_xyz[2] > MAX_HOME_PATH_RISE_M
                or xyz[2] < MIN_HOME_PATH_Z_M):
            raise RuntimeError("RM75 return path leaves the verified trial region")


class StartStateMonitor(Node):
    """只读监控反馈，并在夹爪回位阶段发出全开命令。"""

    def __init__(self):
        """订阅七轴弧度反馈、归一化夹爪反馈和驱动有效标志。"""
        super().__init__("dp_return_to_start")
        self.joints = None
        self.joints_at = float("-inf")
        self.gripper = None
        self.gripper_at = float("-inf")
        self.valid = False
        self.valid_at = float("-inf")
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.create_subscription(Float32, "/motion_control/gripper_state", self.on_gripper, 10)
        self.create_subscription(Bool, "/rm_driver/udp_feedback_valid", self.on_valid, 10)
        self.gripper_publisher = None
        self.stop_publisher = self.create_publisher(Empty, "/rm_driver/move_stop_cmd", 10)

    def on_joints(self, message):
        """按 joint1…joint7 重排实测弧度。"""
        by_name = dict(zip(message.name, message.position))
        if any(name not in by_name for name in JOINT_NAMES):
            return
        values = np.asarray([by_name[name] for name in JOINT_NAMES], dtype=np.float64)
        if np.isfinite(values).all():
            self.joints = values
            self.joints_at = time.monotonic()

    def on_gripper(self, message):
        """缓存 0 闭合、1 张开的实测夹爪开度。"""
        value = float(message.data)
        if np.isfinite(value) and 0 <= value <= 1:
            self.gripper = value
            self.gripper_at = time.monotonic()

    def on_valid(self, message):
        """缓存 RM75 UDP 反馈有效状态。"""
        self.valid = bool(message.data)
        self.valid_at = time.monotonic()

    def feedback_ready(self):
        """只接受最近 300 ms 内的三路有效反馈。"""
        now = time.monotonic()
        return (self.joints is not None and now - self.joints_at < 0.3
                and self.gripper is not None and now - self.gripper_at < 0.3
                and self.valid and now - self.valid_at < 0.3)

    def wait_feedback(self, seconds):
        """等待反馈就绪，超时则禁止后续回位动作。"""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.feedback_ready():
                return
        raise TimeoutError("RM75/gripper feedback did not become fresh and valid")

    def stop_arm(self):
        """在 MoveJ 失败或中断时重复发布驱动停机请求。"""
        for _ in range(5):
            self.stop_publisher.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.05)


def main():
    """张开夹爪，再调用仓库已有的 20% 速度 RM75 MoveJ 回位脚本。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="明确启用机械臂和夹爪回位")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.real:
        raise ValueError("return_to_start requires --real after on-site confirmation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[4]
    home_script = root / "ros2_ws/src/ros2_rm_robot/rm_bringup/scripts/rm_75_initial_pose.py"
    home_python = root / "ros2_ws/.venv-numpy1/bin/python"
    if not home_script.is_file() or not home_python.is_file():
        raise FileNotFoundError("existing RM75 initial-pose script or Python is missing")
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    monitor = StartStateMonitor()
    home_process = None
    try:
        monitor.wait_feedback(5)
        for topic in ("/rm_driver/movej_cmd", "/rm_driver/movej_canfd_cmd",
                      "/motion_control/gripper_command", "/fastumi/policy/action_sequence"):
            if monitor.get_publishers_info_by_topic(topic):
                raise RuntimeError(f"another control publisher is active: {topic}")
        initial_joints_deg = np.rad2deg(monitor.joints).round(2).tolist()
        initial_gripper = monitor.gripper
        from diffusion_policy.common.urdf_kinematics import UrdfKinematics

        training_urdf = root / "dataset/vr_target_umi/rm_75.urdf"
        if not training_urdf.is_file():
            raise FileNotFoundError(f"training URDF is missing: {training_urdf}")
        forward = UrdfKinematics(training_urdf, JOINT_NAMES).forward
        validate_homing_start(monitor.joints, forward)
        monitor.gripper_publisher = monitor.create_publisher(
            Float32, "/motion_control/gripper_command", 10)
        deadline = time.monotonic() + 8
        opened_since = None
        while time.monotonic() < deadline:
            monitor.gripper_publisher.publish(Float32(data=1.0))
            rclpy.spin_once(monitor, timeout_sec=0.1)
            if not monitor.feedback_ready():
                raise RuntimeError("feedback lost during gripper opening")
            if monitor.gripper >= 0.95:
                opened_since = opened_since or time.monotonic()
                if time.monotonic() - opened_since >= 0.5:
                    break
            else:
                opened_since = None
        else:
            raise TimeoutError("gripper did not reach openness 0.95")
        monitor.destroy_publisher(monitor.gripper_publisher)
        monitor.gripper_publisher = None
        command = [
            str(home_python), str(home_script), "--ros-args", "-p",
            "initial_joint_positions:=" + json.dumps(HOME.tolist()),
        ]
        with args.output.with_suffix(".home.log").open("w") as log:
            home_process = subprocess.Popen(command, cwd=root, env=os.environ.copy(),
                                            stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            try:
                home_result = home_process.wait(timeout=125)
            except subprocess.TimeoutExpired:
                os.killpg(home_process.pid, signal.SIGINT)
                home_process.wait(timeout=5)
                raise TimeoutError("RM75 initial-pose mover timed out")
        if home_result != 0:
            raise RuntimeError(f"RM75 initial-pose mover failed with exit {home_result}")
        monitor.wait_feedback(3)
        settled_since = None
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            rclpy.spin_once(monitor, timeout_sec=0.05)
            if not monitor.feedback_ready():
                raise RuntimeError("feedback lost after RM75 homing")
            at_home = (np.max(np.abs(monitor.joints - HOME)) <= 0.035
                       and monitor.gripper >= 0.95)
            if at_home:
                settled_since = settled_since or time.monotonic()
                if time.monotonic() - settled_since >= 0.5:
                    break
            else:
                settled_since = None
        else:
            raise RuntimeError("RM75/gripper did not settle at start state")
        report = {
            "initial_joint_deg": initial_joints_deg,
            "initial_gripper": initial_gripper,
            "final_joint_deg": np.rad2deg(monitor.joints).round(2).tolist(),
            "final_gripper": round(monitor.gripper, 4),
            "max_home_error_deg": round(float(np.rad2deg(
                np.max(np.abs(monitor.joints - HOME)))), 3),
            "feedback_valid": monitor.valid,
            "home_result": home_result,
        }
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    except BaseException:
        if home_process is not None:
            monitor.stop_arm()
        raise
    finally:
        if home_process is not None and home_process.poll() is None:
            os.killpg(home_process.pid, signal.SIGINT)
            try:
                home_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.stop_arm()
                os.killpg(home_process.pid, signal.SIGKILL)
                home_process.wait(timeout=5)
        if monitor.gripper_publisher is not None:
            monitor.destroy_publisher(monitor.gripper_publisher)
        monitor.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
