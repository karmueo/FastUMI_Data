"""在隔离 ROS domain 中回放录制观测，验证真实 DP 输出和 Placo dry-run。"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import cv2
from fastumi_interfaces.msg import PolicyActionSequence
import h5py
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rm_ros_interfaces.msg import Jointpos
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Float32


JOINT_NAMES = [f"joint{index}" for index in range(1, 8)]


def load_frames(episode, count):
    """读取实际录制图像和各图像时刻最近的七轴/夹爪反馈。"""
    with h5py.File(episode / "proprio.hdf5", "r") as source:
        times = source["observations/images/cam_gripper_timestamp"][:count]
        joint_times = source["observations/joint_state/timestamp"][:]
        joints = source["observations/joint_state/qpos"][:]
        gripper_times = source["observations/gripper_state/timestamp"][:]
        grippers = source["observations/gripper_state/position"][:, 0]
    camera = cv2.VideoCapture(str(episode / "gripper.mp4"))
    frames = []
    try:
        for stamp in times:
            ok, bgr = camera.read()
            if not ok:
                raise ValueError("video ended before HDF5 timestamps")
            encoded, jpeg = cv2.imencode(".jpg", bgr)
            if not encoded:
                raise ValueError("cannot encode replay JPEG")
            joint_index = int(np.argmin(np.abs(joint_times - stamp)))
            gripper_index = int(np.argmin(np.abs(gripper_times - stamp)))
            frames.append((jpeg.tobytes(), joints[joint_index].tolist(), float(grippers[gripper_index])))
    finally:
        camera.release()
    return frames


def run(args):
    """只发布观测；组合 launch 固定 dry_run，不触发硬件命令。"""
    domain = os.environ.get("ROS_DOMAIN_ID", "")
    if not domain.isdigit() or int(domain) == 0:
        raise ValueError("set a nonzero isolated ROS_DOMAIN_ID before replay")
    if not args.episode.is_dir() or not args.checkpoint.is_file() or not args.urdf.is_file():
        raise FileNotFoundError("episode, checkpoint and URDF must exist")
    frames = load_frames(args.episode, args.frames)
    if len(frames) < 2:
        raise ValueError("at least two recorded frames are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = rclpy.create_node("dp_infer_smoke_replay")
    # 不向已运行硬件的 DDS domain 发送伪造的录制反馈。
    time.sleep(0.5)
    if any(node.count_publishers(topic) for topic in (
            "/joint_states", "/wrist_camera/image_raw/compressed",
            "/motion_control/gripper_state")):
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError("ROS domain already has observation publishers")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    images = node.create_publisher(CompressedImage, "/wrist_camera/image_raw/compressed", 10)
    joints = node.create_publisher(JointState, "/joint_states", 10)
    grippers = node.create_publisher(Float32, "/motion_control/gripper_state", 10)
    predictions, debug_joints = [], []
    arm_commands, gripper_commands, close_permissions = [], [], []
    node.create_subscription(
        PolicyActionSequence, "/fastumi/policy/action_sequence",
        lambda message: predictions.append((message, node.get_clock().now().nanoseconds)), 10)
    node.create_subscription(
        JointState, "/fastumi/rm75/placo/joint_command",
        lambda message: debug_joints.append(message), 10)
    node.create_subscription(
        Jointpos, "/rm_driver/movej_canfd_cmd",
        lambda message: arm_commands.append(message), 10)
    node.create_subscription(
        Float32, "/motion_control/gripper_command",
        lambda message: gripper_commands.append(message), 10)
    node.create_subscription(
        Bool, "/fastumi/policy/gripper_close_allowed",
        lambda message: close_permissions.append(bool(message.data)), 10)

    launch_log = args.output.with_suffix(".launch.log")
    command = [
        "ros2", "launch", "dp_infer", "dp_infer.launch.py",
        f"checkpoint:={args.checkpoint}", f"urdf_path:={args.urdf}",
        "dry_run:=true", "num_inference_steps:=8",
    ]
    process = None
    try:
        with launch_log.open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=os.environ.copy())
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("launch exited before subscriptions were ready")
                if all(publisher.get_subscription_count() >= 1
                       for publisher in (images, joints, grippers)):
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError("inference subscriptions did not appear")
            # 为 DDS 发现和控制器的关节订阅再留少量时间。
            time.sleep(0.5)
            started = time.monotonic()
            for index, (jpeg, angles, openness) in enumerate(frames):
                stamp = node.get_clock().now().to_msg()
                joint = JointState()
                joint.header.stamp = stamp
                joint.name = JOINT_NAMES
                joint.position = angles
                joints.publish(joint)
                grippers.publish(Float32(data=openness))
                image = CompressedImage()
                image.header.stamp = stamp
                image.format = "jpeg"
                image.data = jpeg
                images.publish(image)
                until = started + (index + 1) / 30.0
                time.sleep(max(0, until - time.monotonic()))
            time.sleep(0.8)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        executor.shutdown()
        spinner.join(timeout=5)
        node.destroy_node()
        rclpy.shutdown()

    ages_ms = []
    coverage_gaps_ms = []
    previous_end_ns = None
    for message, received_ns in predictions:
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        age_ms = (received_ns - stamp_ns) / 1e6
        ages_ms.append(age_ms)
        offsets = np.asarray([
            item.sec * 1_000_000_000 + item.nanosec
            for item in message.time_from_start
        ], dtype=np.int64)
        positions = np.asarray([
            [pose.position.x, pose.position.y, pose.position.z]
            for pose in message.poses
        ])
        quaternions = np.asarray([
            [pose.orientation.x, pose.orientation.y,
             pose.orientation.z, pose.orientation.w]
            for pose in message.poses
        ])
        gripper_values = np.asarray(message.gripper_openness)
        if (message.header.frame_id != "base_link" or message.end_frame != "Link7"
                or len(message.poses) != 16 or len(message.time_from_start) != 16
                or len(message.gripper_openness) != 16 or not 0 <= age_ms <= 500
                or not np.array_equal(offsets, np.rint(np.arange(16) / 30 * 1e9).astype(np.int64))
                or not np.isfinite(positions).all() or not np.isfinite(quaternions).all()
                or not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-5)
                or not np.isfinite(gripper_values).all()
                or np.any((gripper_values < 0) | (gripper_values > 1))):
            raise AssertionError("invalid or expired policy output")
        if previous_end_ns is not None and received_ns > previous_end_ns:
            coverage_gaps_ms.append((received_ns - previous_end_ns) / 1e6)
        previous_end_ns = stamp_ns + int(offsets[-1])
    for message in debug_joints:
        if (message.name != JOINT_NAMES or len(message.position) != 7
                or not np.isfinite(message.position).all()):
            raise AssertionError("invalid Placo dry-run joint target")
    result = {
        "recorded_frames": len(frames), "policy_sequences": len(predictions),
        "debug_joint_targets": len(debug_joints),
        "dry_run_arm_commands": len(arm_commands),
        "dry_run_gripper_commands": len(gripper_commands),
        "close_permission_messages": len(close_permissions),
        "close_permission_true": sum(close_permissions),
        "prediction_age_ms_min": min(ages_ms) if ages_ms else None,
        "prediction_age_ms_max": max(ages_ms) if ages_ms else None,
        "coverage_gap_count": len(coverage_gaps_ms),
        "coverage_gap_ms_max": max(coverage_gaps_ms) if coverage_gaps_ms else 0.0,
        "launch_log": str(launch_log),
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if (not predictions or not debug_joints or not close_permissions
            or arm_commands or gripper_commands):
        raise AssertionError("dry-run inference/IK/output contract failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=180)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
