"""实际 ROS 2 launch 启动 LeRobot 推理节点（不启动控制器），回放图像验证发布、时效与停止/重启。

不接硬件：图像取自 ``compare_policy.py export-reference`` 导出的留出 episode 样本（已 letterbox），
未给样本时使用合成图像；关节为固定姿态，夹爪为固定开度，按实时 30 Hz 发布到隔离话题。
节点默认待命，脚本调用 ``start_task`` 开始推理；“重置”为 ``stop_task`` 后再 ``start_task``。
用法（仓库根目录，已加载 ROS 与工作区，使用隔离 ROS_DOMAIN_ID）::

    ROS_DOMAIN_ID=181 ros2_ws/.venv-dp-lerobot/bin/python \\
        ros2_ws/src/dp_infer_lerobot/test/smoke_dp_infer_lerobot.py \\
        --checkpoint <pretrained_model> --sequential-execution false --num-inference-steps 2 \\
        [--backend tensorrt_fp16 --export-dir <本机 TensorRT 模型包>，此时不传 --checkpoint] \\
        --report dataset/validation/dp_lerobot_v4/smoke_ddim2.json
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
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
from dp_infer.common import JOINT_NAMES
from dp_infer.node import stamp_ns

# 回放使用的固定 RM75 关节角（弧度），只需让 FK 有效。
REPLAY_JOINTS = [0.0, 0.4, 0.0, 1.4, 0.0, 1.2, 0.0]
# 回放图像使用独立 raw 话题，避免与真实相机混淆；其余与 launch 默认值一致（不启动控制器）。
TOPICS = {"image_topic": "/dp_infer_lerobot_smoke/image_raw", "joint_topic": "/joint_states",
          "gripper_topic": "/motion_control/gripper_state", "output_topic": "/fastumi/policy/action_sequence",
          "start_service": "/fastumi/policy/start_task", "stop_service": "/fastumi/policy/stop_task"}


class ReplayProbe(Node):
    """按 30 Hz 发布图像/关节/夹爪，并记录与校验收到的策略序列。"""

    def __init__(self, sample_path, gripper):
        """加载样本中的当前帧图像序列（或合成图像），创建发布器、订阅与任务服务客户端。"""
        super().__init__("dp_infer_lerobot_replay_probe")
        if sample_path:
            sample = np.load(sample_path)
            # (N,224,224,3) uint8 RGB：每个样本取当前帧（第 2 帧）。
            self.images = np.round(sample["image"][:, -1].transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        else:
            # 合成 240×320 图像，只验证链路与时效（含 letterbox），不代表策略质量。
            # 尺寸不宜过大：rclpy 填充大块 Image.data 较慢，会使回放低于 30 Hz、两帧历史无法匹配。
            rng = np.random.default_rng(0)
            self.images = rng.integers(0, 256, size=(4, 240, 320, 3), dtype=np.uint8)
        self.gripper = float(gripper)  # 回放夹爪开度。
        self.index = 0  # 当前回放图像序号，循环播放。
        self.sent_stamps = set()  # 已发布图像的采集时间，用于核对输出 header。
        self.messages = []  # (接收时间纳秒, 消息)。
        self.image_pub = self.create_publisher(Image, TOPICS["image_topic"], qos_profile_sensor_data)
        self.joint_pub = self.create_publisher(JointState, TOPICS["joint_topic"], qos_profile_sensor_data)
        self.gripper_pub = self.create_publisher(Float32, TOPICS["gripper_topic"], qos_profile_sensor_data)
        self.result_sub = self.create_subscription(PolicyActionSequence, TOPICS["output_topic"], self.on_result, 10)
        self.start_client = self.create_client(Trigger, TOPICS["start_service"])
        self.stop_client = self.create_client(Trigger, TOPICS["stop_service"])
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


def check_message(message, sent_stamps, received_ns, max_result_age_s, sequential):
    """校验消息结构与接收时的结果年龄（秒，包含传输延迟）；不合格抛出 AssertionError。

    流水线模式下 header 为观测图像采集时间、点时间为 k/30；串行模式下 header 为发布时刻，
    运行层把全部点推迟一帧（(k+1)/30）。
    """
    age_ns = received_ns - stamp_ns(message.header.stamp)
    assert 0 <= age_ns <= round(max_result_age_s * 1e9), (
        f"Result age {age_ns / 1e9:.6f} s outside [0, {max_result_age_s}] s")
    assert message.header.frame_id == "base_link" and message.end_frame == "Link7"
    assert len(message.poses) == len(message.gripper_openness) == len(message.time_from_start) == 8
    if not sequential:
        assert stamp_ns(message.header.stamp) in sent_stamps, "header must be an observed image stamp"
    offsets = [stamp_ns(item) for item in message.time_from_start]
    assert offsets == [round((k + sequential) * 1e9 / 30) for k in range(8)]
    for pose in message.poses:
        q = pose.orientation
        assert abs(np.linalg.norm([q.x, q.y, q.z, q.w]) - 1) < 1e-6
        assert np.isfinite([pose.position.x, pose.position.y, pose.position.z]).all()
    assert all(0 <= value <= 1 for value in message.gripper_openness)


def call(executor, client, timeout_s=10.0):
    """同步调用 Trigger 服务并返回响应；超时抛出 RuntimeError。"""
    if not client.wait_for_service(timeout_sec=timeout_s):
        raise RuntimeError(f"{client.srv_name} unavailable")
    future = client.call_async(Trigger.Request())
    deadline = time.monotonic() + timeout_s
    while not future.done():
        if time.monotonic() > deadline:
            raise RuntimeError(f"{client.srv_name} timed out")
        executor.spin_once(timeout_sec=0.05)
    return future.result()


def start_task(executor, probe, timeout_s):
    """反复请求开始任务，直到三路输入新鲜、节点接受。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        response = call(executor, probe.start_client)
        if response.success:
            return response.message
        spin_for(executor, 0.5)
    raise RuntimeError(f"start_task rejected: {response.message}")


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
    parser.add_argument("--export-dir", default="", help="TensorRT 后端的本机模型包（build_tensorrt_engines.py 生成）")
    parser.add_argument("--deployment-config", default="", help="部署档，空则用随包 dp_umi_v4；v5 用 dp_umi_v5_deployment.yaml")
    parser.add_argument("--sample", default="", help="export-reference 导出的 NPZ；空则使用合成图像")
    parser.add_argument("--urdf", default="assets/rm_75.urdf")
    parser.add_argument("--end-camera", default="config/calibration/T_cam_end.yaml")
    parser.add_argument("--deploy-tcp", default="config/calibration/T_d_cam_tcp.yaml")
    parser.add_argument("--noise-scheduler", default="")
    parser.add_argument("--num-inference-steps", default="")
    parser.add_argument("--sequential-execution", choices=("true", "false"), default="false")
    parser.add_argument("--gripper", type=float, default=0.8)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--max-result-age-s", type=float, default=0.3,
                        help="接收端允许的最大结果年龄，包含传输延迟，默认 0.3 秒（8 个动作约 0.233 秒）")
    parser.add_argument("--report", default="")
    args = parser.parse_args()
    if not np.isfinite(args.max_result_age_s) or args.max_result_age_s <= 0:
        parser.error("--max-result-age-s must be positive and finite")
    if not args.checkpoint and not args.export_dir:
        parser.error("--checkpoint or --export-dir is required")
    # 用当前（推理环境）解释器运行 ros2：系统 Python 可能加载用户目录的 NumPy 2，导致 SciPy 导入失败。
    ros2 = shutil.which("ros2") or "/opt/ros/humble/bin/ros2"
    command = [sys.executable, ros2, "launch", "dp_infer_lerobot", "dp_infer_lerobot.launch.py",
               f"urdf_path:={Path(args.urdf).resolve()}",
               f"end_camera_extrinsic:={Path(args.end_camera).resolve()}",
               f"deploy_tcp_extrinsic:={Path(args.deploy_tcp).resolve()}",
               "start_controller:=false", "image_type:=raw", f"image_topic:={TOPICS['image_topic']}",
               f"sequential_execution:={args.sequential_execution}"]
    sequential = args.sequential_execution == "true"
    for name, value in (("checkpoint", args.checkpoint), ("export_dir", args.export_dir),
                        ("deployment_config", args.deployment_config)):
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
        while not probe.start_client.service_is_ready() and time.monotonic() - started < args.startup_timeout:
            if process.poll() is not None:
                raise RuntimeError(f"launch exited with {process.returncode}; see {log_path}")
            spin_for(executor, 0.5)
        report["start"] = start_task(executor, probe, 10.0)
        while not probe.messages and time.monotonic() - started < args.startup_timeout:
            if process.poll() is not None:
                raise RuntimeError(f"launch exited with {process.returncode}; see {log_path}")
            spin_for(executor, 0.5)
        if not probe.messages:
            raise RuntimeError("No policy message before startup timeout")
        report["first_message_after_s"] = time.monotonic() - started
        for received, message in probe.messages:
            check_message(message, probe.sent_stamps, received, args.max_result_age_s, sequential)
        probe.messages.clear()
        spin_for(executor, args.duration)
        steady = list(probe.messages)
        for received, message in steady:
            check_message(message, probe.sent_stamps, received, args.max_result_age_s, sequential)
        ages = [(received - stamp_ns(message.header.stamp)) / 1e6 for received, message in steady]
        report.update(messages=len(steady), rate_hz=len(steady) / args.duration,
                      age_ms={"median": float(np.median(ages)), "max": float(np.max(ages))},
                      episode_ids=sorted({int(message.episode_id) for _, message in steady}))
        old_episode = max(report["episode_ids"])
        stop = call(executor, probe.stop_client)
        reset_called = probe.get_clock().now().nanoseconds
        probe.messages.clear()
        spin_for(executor, 1.0)
        report["messages_while_stopped"] = len(probe.messages)
        probe.messages.clear()
        restart = start_task(executor, probe, 10.0)
        spin_for(executor, 3.0)
        after = list(probe.messages)
        for received, message in after:
            check_message(message, probe.sent_stamps, received, args.max_result_age_s, sequential)
        stale = [message for _, message in after
                 if message.episode_id <= old_episode or stamp_ns(message.header.stamp) < reset_called]
        report.update(reset_response=f"{stop.message}; {restart}", messages_after_reset=len(after),
                      stale_after_reset=len(stale))
        first, last = steady[0][1], steady[-1][1]
        report["example"] = {"sequence_id": int(last.sequence_id),
                             "first_position": [first.poses[0].position.x, first.poses[0].position.y,
                                                first.poses[0].position.z],
                             "gripper": list(map(float, last.gripper_openness))}
        report["passed"] = (bool(steady) and bool(after) and not stale
                            and report["messages_while_stopped"] == 0)
    except Exception as error:  # 报告失败原因后继续清理。
        report["error"] = repr(error)
    finally:
        if process.poll() is None:
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
