"""实际 ROS 2 launch 启动 LeRobot 推理节点，用数据集图像回放验证发布、时效与重置。

不接硬件：图像取自 ``compare_policy.py export-reference`` 导出的留出 episode 样本（已 letterbox），
关节为固定姿态，夹爪为固定开度，按实时 30 Hz 发布。用法（仓库根目录，已加载 ROS 与工作区）::

    ROS_DOMAIN_ID=181 ros2_ws/.venv-dp-lerobot/bin/python \\
        ros2_ws/src/dp_infer_lerobot/test/smoke_dp_infer_lerobot.py \\
        --checkpoint <pretrained_model> --sample dataset/validation/dp_lerobot_v4/reference_ddim16.npz \\
        [--backend tensorrt_fp16 --export-dir <lerobot-export 模型包>] \\
        --report dataset/validation/dp_lerobot_v4/smoke_ddim16.json
"""

import argparse
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

from fastumi_interfaces.msg import PolicyActionSequence
from dp_infer.core import JOINT_NAMES
from dp_infer.node import stamp_ns

# 回放使用的固定 RM75 关节角（弧度），只需让 FK 有效。
REPLAY_JOINTS = [0.0, 0.4, 0.0, 1.4, 0.0, 1.2, 0.0]
TOPICS = {"image_topic": "/usb_camera/image_raw", "joint_topic": "/joint_states",
          "gripper_topic": "/gripper/openness", "output_topic": "/fastumi/policy/action_sequence",
          "reset_service": "/fastumi/policy/reset_episode"}  # 与节点默认话题一致。


class ReplayProbe(Node):
    """按 30 Hz 发布图像/关节/夹爪，并记录与校验收到的策略序列。"""

    def __init__(self, sample_path, gripper):
        """加载样本中的当前帧图像序列，创建发布器、订阅与重置客户端。"""
        super().__init__("dp_infer_lerobot_replay_probe")
        sample = np.load(sample_path)
        # (N,224,224,3) uint8 RGB：每个样本取当前帧（第 2 帧）。
        self.images = np.round(sample["image"][:, -1].transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        self.gripper = float(gripper)  # 回放夹爪开度。
        self.index = 0  # 当前回放图像序号，循环播放。
        self.sent_stamps = set()  # 已发布图像的采集时间，用于核对输出 header。
        self.messages = []  # (接收时间纳秒, 消息)。
        self.image_pub = self.create_publisher(Image, TOPICS["image_topic"], qos_profile_sensor_data)
        self.joint_pub = self.create_publisher(JointState, TOPICS["joint_topic"], qos_profile_sensor_data)
        self.gripper_pub = self.create_publisher(Float32, TOPICS["gripper_topic"], qos_profile_sensor_data)
        self.result_sub = self.create_subscription(PolicyActionSequence, TOPICS["output_topic"], self.on_result, 10)
        self.reset_client = self.create_client(Trigger, TOPICS["reset_service"])
        self.timer = self.create_timer(1 / 30, self.publish_frame)

    def publish_frame(self):
        """同一时间戳发布一帧图像与关节，并发布夹爪。"""
        now = self.get_clock().now().nanoseconds
        joint = JointState()
        joint.header.stamp.sec, joint.header.stamp.nanosec = divmod(now, 1_000_000_000)
        joint.name = list(JOINT_NAMES)
        joint.position = list(REPLAY_JOINTS)
        rgb = self.images[(self.index // 3) % len(self.images)]  # 每张图保持 0.1 s，模拟连续画面。
        image = Image()
        image.header.stamp = joint.header.stamp
        image.header.frame_id = "camera0"
        image.height, image.width = rgb.shape[:2]
        image.encoding = "rgb8"
        image.step = image.width * 3
        image.data = rgb.tobytes()
        self.joint_pub.publish(joint)
        self.gripper_pub.publish(Float32(data=self.gripper))
        self.image_pub.publish(image)
        self.sent_stamps.add(now)
        self.index += 1

    def on_result(self, message):
        """记录接收时间与消息。"""
        self.messages.append((self.get_clock().now().nanoseconds, message))


def check_message(message, sent_stamps, received_ns, max_result_age_s):
    """校验消息结构与接收时的结果年龄（秒，包含传输延迟）；不合格抛出 AssertionError。"""
    age_ns = received_ns - stamp_ns(message.header.stamp)
    assert 0 <= age_ns <= round(max_result_age_s * 1e9), (
        f"Result age {age_ns / 1e9:.6f} s outside [0, {max_result_age_s}] s")
    assert message.header.frame_id == "base_link" and message.end_frame == "Link7"
    assert len(message.poses) == len(message.gripper_openness) == len(message.time_from_start) == 8
    assert stamp_ns(message.header.stamp) in sent_stamps, "header must be an observed image stamp"
    offsets = [stamp_ns(item) for item in message.time_from_start]
    assert offsets == [round(k * 1e9 / 30) for k in range(8)]
    for pose in message.poses:
        q = pose.orientation
        assert abs(np.linalg.norm([q.x, q.y, q.z, q.w]) - 1) < 1e-6
        assert np.isfinite([pose.position.x, pose.position.y, pose.position.z]).all()
    assert all(0 <= value <= 1 for value in message.gripper_openness)


def spin_for(executor, seconds):
    """运行执行器指定秒数。"""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.01)


def main():
    """启动 launch、回放、重置并输出报告；失败时返回非零。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="", help="PyTorch 后端的 pretrained_model 目录")
    parser.add_argument("--backend", default="", help="pytorch / tensorrt_fp32 / tensorrt_fp16，空则沿用参数文件")
    parser.add_argument("--export-dir", default="", help="TensorRT 后端的 lerobot-export 模型包")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--urdf", default="assets/rm_75.urdf")
    parser.add_argument("--end-camera", default="config/calibration/T_cam_end.yaml")
    parser.add_argument("--deploy-tcp", default="config/calibration/T_d_cam_tcp.yaml")
    parser.add_argument("--noise-scheduler", default="")
    parser.add_argument("--num-inference-steps", default="")
    parser.add_argument("--gripper", type=float, default=0.8)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--max-result-age-s", type=float, default=0.5,
                        help="接收端允许的最大结果年龄，包含传输延迟，默认 0.5 秒")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    if not np.isfinite(args.max_result_age_s) or args.max_result_age_s <= 0:
        parser.error("--max-result-age-s must be positive and finite")
    if not args.checkpoint and not args.export_dir:
        parser.error("--checkpoint or --export-dir is required")
    command = ["ros2", "launch", "dp_infer_lerobot", "dp_infer_lerobot.launch.py",
               f"urdf_path:={Path(args.urdf).resolve()}",
               f"end_camera_extrinsic:={Path(args.end_camera).resolve()}",
               f"deploy_tcp_extrinsic:={Path(args.deploy_tcp).resolve()}"]
    for name, value in (("checkpoint", args.checkpoint), ("export_dir", args.export_dir)):
        if value:
            command.append(f"{name}:={Path(value).resolve()}")
    if args.backend:
        command.append(f"backend:={args.backend}")
    if args.noise_scheduler:
        command.append(f"noise_scheduler:={args.noise_scheduler}")
    if args.num_inference_steps:
        command.append(f"num_inference_steps:={args.num_inference_steps}")
    log_path = Path(args.report or "smoke").with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rclpy.init()
    probe = ReplayProbe(args.sample, args.gripper)
    executor = SingleThreadedExecutor()
    executor.add_node(probe)
    report = {"command": command, "max_result_age_s": args.max_result_age_s, "passed": False}
    try:
        started = time.monotonic()
        while not probe.messages and time.monotonic() - started < args.startup_timeout:
            if process.poll() is not None:
                raise RuntimeError(f"launch exited with {process.returncode}; see {log_path}")
            spin_for(executor, 0.5)
        if not probe.messages:
            raise RuntimeError("No policy message before startup timeout")
        report["first_message_after_s"] = time.monotonic() - started
        for received, message in probe.messages:
            check_message(message, probe.sent_stamps, received, args.max_result_age_s)
        probe.messages.clear()
        spin_for(executor, args.duration)
        steady = list(probe.messages)
        for received, message in steady:
            check_message(message, probe.sent_stamps, received, args.max_result_age_s)
        ages = [(received - stamp_ns(message.header.stamp)) / 1e6 for received, message in steady]
        report.update(messages=len(steady), rate_hz=len(steady) / args.duration,
                      age_ms={"median": float(np.median(ages)), "max": float(np.max(ages))},
                      episode_ids=sorted({int(message.episode_id) for _, message in steady}))
        old_episode = max(report["episode_ids"])
        if not probe.reset_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("reset service unavailable")
        future = probe.reset_client.call_async(Trigger.Request())
        reset_called = probe.get_clock().now().nanoseconds
        while not future.done():
            executor.spin_once(timeout_sec=0.05)
        probe.messages.clear()
        spin_for(executor, 2.0)
        after = list(probe.messages)
        for received, message in after:
            check_message(message, probe.sent_stamps, received, args.max_result_age_s)
        stale = [message for _, message in after
                 if message.episode_id <= old_episode or stamp_ns(message.header.stamp) < reset_called]
        report.update(reset_response=future.result().message, messages_after_reset=len(after),
                      stale_after_reset=len(stale))
        first, last = steady[0][1], steady[-1][1]
        report["example"] = {"sequence_id": int(last.sequence_id),
                             "first_position": [first.poses[0].position.x, first.poses[0].position.y,
                                                first.poses[0].position.z],
                             "gripper": list(map(float, last.gripper_openness))}
        report["passed"] = bool(steady) and bool(after) and not stale
    except Exception as error:  # 报告失败原因后继续清理。
        report["error"] = repr(error)
    finally:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        executor.remove_node(probe)
        probe.destroy_node()
        rclpy.shutdown()
    logs = log_path.read_text()
    report["inference_ms"] = [float(part.split("=")[1]) for line in logs.splitlines()
                              for part in line.split() if part.startswith("inference_ms=")]
    if report["inference_ms"]:
        report["inference_ms"] = {"median": float(np.median(report["inference_ms"])),
                                  "max": float(np.max(report["inference_ms"]))}
    report["log"] = str(log_path)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
