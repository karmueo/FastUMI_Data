"""实际 ROS 2 launch 启动 DP 节点，并通过 DDS 回放验证发布、缺流、过期和重置。"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
import zarr

from fastumi_interfaces.msg import PolicyActionSequence
from diffusion_policy.common.urdf_kinematics import UrdfKinematics
from dp_infer.core import JOINT_NAMES
from dp_infer.node import stamp_ns


class ReplayProbe(Node):
    """回放一段匹配的真实 RGB/关节/夹爪，并检查订阅到的预测消息。"""

    def __init__(self, dataset, joint_dataset, urdf_path, topics):
        """预加载一段真实 episode，并校验两套数据的时间轴和训练 URDF。"""
        super().__init__("dp_infer_replay_probe")
        poses = zarr.open_group(str(dataset), mode="r")
        joints = zarr.open_group(str(joint_dataset), mode="r")
        first_end = int(poses["meta/episode_ends"][0])
        count = min(first_end, 180)  # 有界回放缓存，最长六秒原始数据。
        if count < 2 or first_end != int(joints["meta/episode_ends"][0]):
            raise ValueError("Pose and joint datasets must contain the same episode")
        np.testing.assert_array_equal(poses["data/timestamp"][:count], joints["data/timestamp"][:count])
        np.testing.assert_array_equal(poses["data/source_image_index"][:count],
                                      joints["data/source_image_index"][:count])
        urdf = Path(urdf_path)
        if hashlib.sha256(urdf.read_bytes()).hexdigest() != poses.attrs["urdf_sha256"]:
            raise ValueError("Replay URDF differs from the training dataset")
        self.images = poses["data/camera0_rgb"][:count]  # 已由训练转换器 letterbox 的 RGB。
        self.joints = joints["data/robot0_joint_pos"][:count]
        self.grippers = joints["data/robot0_gripper_position"][:count, 0]
        np.testing.assert_allclose(UrdfKinematics(urdf, JOINT_NAMES).forward(self.joints)[:, :3, 3],
                                   poses["data/robot0_eef_pos"][:count], atol=1e-6)
        self.index = 0  # 到段尾后固定最后一帧，避免跨 episode 拼接。
        self.mode = "all"  # all、missing_gripper 或 stale，供验收阶段切换。
        self.sent_stamps = set()  # 用于验证输出 header 对应真实发布过的图像。
        self.messages = []  # 已校验的输出消息。
        self.image_pub = self.create_publisher(Image, topics["image_topic"], qos_profile_sensor_data)
        self.joint_pub = self.create_publisher(JointState, topics["joint_topic"], qos_profile_sensor_data)
        self.gripper_pub = self.create_publisher(Float32, topics["gripper_topic"], qos_profile_sensor_data)
        self.result_sub = self.create_subscription(PolicyActionSequence, topics["output_topic"],
                                                   self.on_result, 10)
        self.reset_client = self.create_client(Trigger, topics["reset_service"])
        self.timer = self.create_timer(1 / 30, self.publish_frame)

    def publish_frame(self):
        """按实时 30 Hz 发布匹配帧；过期模式把带 header 的消息时间回拨一秒。"""
        current = self.get_clock().now().nanoseconds
        stamp = current - 1_000_000_000 if self.mode == "stale" else current
        joint = JointState()
        joint.header.stamp.sec, joint.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        joint.name = list(JOINT_NAMES)
        joint.position = self.joints[self.index].astype(float).tolist()
        rgb = self.images[self.index]
        image = Image()
        image.header.stamp = joint.header.stamp
        image.header.frame_id = "camera0"
        image.height, image.width = rgb.shape[:2]
        image.encoding = "rgb8"
        image.step = image.width * 3
        image.data = rgb.tobytes()
        self.joint_pub.publish(joint)
        if self.mode != "missing_gripper":
            self.gripper_pub.publish(Float32(data=float(self.grippers[self.index])))
        self.image_pub.publish(image)
        self.sent_stamps.add(stamp)
        self.index = min(self.index + 1, len(self.images) - 1)

    def on_result(self, message):
        """验证坐标系、数组配对、时间、四元数、夹爪范围及结果序号。"""
        assert message.header.frame_id == "base_link" and message.end_frame == "Link7"
        assert stamp_ns(message.header.stamp) in self.sent_stamps
        assert len(message.poses) == len(message.gripper_openness) == len(message.time_from_start) == 16
        quaternions = np.array([[pose.orientation.x, pose.orientation.y, pose.orientation.z,
                                 pose.orientation.w] for pose in message.poses])
        positions = np.array([[pose.position.x, pose.position.y, pose.position.z] for pose in message.poses])
        assert np.isfinite(positions).all()
        np.testing.assert_allclose(np.linalg.norm(quaternions, axis=-1), 1, atol=1e-6)
        assert np.all(np.sum(quaternions[1:] * quaternions[:-1], axis=-1) >= 0)
        assert np.isfinite(message.gripper_openness).all()
        assert all(0 <= value <= 1 for value in message.gripper_openness)
        np.testing.assert_allclose([stamp_ns(value) / 1e9 for value in message.time_from_start],
                                   np.arange(16) / 30, atol=1e-9)
        if self.messages:
            assert message.sequence_id > self.messages[-1].sequence_id
            assert message.episode_id >= self.messages[-1].episode_id
        self.messages.append(message)


def spin_until(executor, predicate, timeout_s, child=None):
    """处理 ROS 回调直到条件满足，同时检查外部 launch 是否意外退出。"""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("ROS smoke test timed out waiting for valid predictions/service")
        if child is not None and child.poll() is not None:
            raise RuntimeError(f"Inference launch exited early with status {child.returncode}")
        executor.spin_once(timeout_sec=0.01)


def spin_for(executor, duration_s):
    """在短时验收窗口内持续处理订阅和工作线程结果。"""
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.01)


def main():
    """通过 ROS 2 launch 启动独立节点并回放真实数据，保存验收报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--joint-dataset", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    topics = {  # 私有话题名称避免真实机器人订阅本次回放。
        "image_topic": "/dp_infer_smoke/image", "joint_topic": "/dp_infer_smoke/joints",
        "gripper_topic": "/dp_infer_smoke/gripper", "output_topic": "/dp_infer_smoke/actions",
        "reset_service": "/dp_infer_smoke/reset",
    }
    command = [  # 节点进程由实际 ros2 launch 管理。
        "ros2", "launch", "dp_infer", "dp_infer.launch.py",
        f"checkpoint:={args.checkpoint.resolve()}", f"urdf_path:={args.urdf.resolve()}",
        f"device:={args.device}",
        *[f"{name}:={value}" for name, value in topics.items()],
    ]
    rclpy.init()
    probe = None
    child = None
    executor = SingleThreadedExecutor()
    try:
        probe = ReplayProbe(args.dataset, args.joint_dataset, args.urdf, topics)
        executor.add_node(probe)
        with (args.output.parent / "launch.log").open("w") as launch_log:
            started = time.monotonic()
            child = subprocess.Popen(command, stdout=launch_log, stderr=subprocess.STDOUT,
                                     cwd="/tmp", start_new_session=True)
            spin_until(executor, lambda: len(probe.messages) >= 3, args.timeout, child)
            startup_s = time.monotonic() - started
            probe.mode = "missing_gripper"
            spin_for(executor, 0.8)
            settled_count = len(probe.messages)
            spin_for(executor, 0.6)
            assert len(probe.messages) == settled_count, "Predicted from missing/stale gripper feedback"
            probe.mode = "stale"
            spin_for(executor, 0.8)
            stale_count = len(probe.messages)
            spin_for(executor, 0.6)
            assert len(probe.messages) == stale_count, "Predicted from expired image/joint state"
            spin_until(executor, probe.reset_client.service_is_ready, 5, child)
            reset = probe.reset_client.call_async(Trigger.Request())
            spin_until(executor, reset.done, 5, child)
            assert reset.result().success
            probe.mode = "all"
            spin_until(executor, lambda: sum(message.episode_id == 1 for message in probe.messages) >= 3,
                       args.timeout, child)
            report = {
                "checkpoint": str(args.checkpoint.resolve()), "prediction_shape": [1, 16, 10],
                "startup_seconds": startup_s, "published_messages": len(probe.messages),
                "messages_by_episode": dict(Counter(message.episode_id for message in probe.messages)),
                "missing_gripper_rejected": True, "expired_inputs_rejected": True,
                "episode_reset_verified": True,
                "sequence_ids": [message.sequence_id for message in probe.messages],
                "observation_stamps_ns": [stamp_ns(message.header.stamp) for message in probe.messages],
                "passed": True,
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
    finally:
        executor.shutdown()
        if probe is not None:
            probe.destroy_node()
        if child is not None and child.poll() is None:
            # 由 launch 向节点转发一次 SIGINT，避免进程组重复中断 ROS 资源清理。
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        rclpy.shutdown()


if __name__ == "__main__":
    main()
