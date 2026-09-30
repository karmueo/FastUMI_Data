"""重放录制 MCAP 的三路观测，验证真实 C++ 推理和 Placo dry-run。"""

import argparse
from array import array
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from fastumi_interfaces.msg import PolicyActionSequence
from fastumi_interfaces.srv import ResetPolicyController, SetNumInferenceSteps
from rm_ros_interfaces.msg import Jointpos
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Bool, Float32
from std_srvs.srv import Trigger


JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


def load_frames(episode, count, work):
    """复用 MCAP 解码器；录制命令只读取，不进入回放发布链路。"""
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root))
    from convert_hardware_mcap import _decode_frame, _fps, _read_episode, _write_video

    frames = []
    with tempfile.TemporaryFile(mode="w+b") as spool:
        data = _read_episode(episode / "bag", spool, require_tracker=False)
        camera = data["camera"]
        if data["camera_mode"] == "h264":
            first = next(i for i, item in enumerate(camera) if item.keyframe)
            camera = camera[first:]
            video = work / "replay.mp4"
            _write_video(video, camera, "h264", spool, _fps(camera), threads=2)
            capture = cv2.VideoCapture(str(video))
        else:
            capture = None
        joints = data["series"]["joint_state"]
        grippers = data["series"]["gripper_state"]
        joint_times = np.array([item[0] for item in joints], dtype=np.int64)
        gripper_times = np.array([item[0] for item in grippers], dtype=np.int64)
        try:
            for record in camera[:count]:
                if capture:
                    ok, bgr = capture.read()
                    if not ok:
                        raise ValueError("H.264 replay ended early")
                else:
                    bgr = _decode_frame(spool, record, data["camera_mode"])
                q = joints[int(np.argmin(np.abs(joint_times - record.bag_ns)))][1]
                width = grippers[int(np.argmin(np.abs(gripper_times - record.bag_ns)))][1]
                frames.append((bgr, list(map(float, q)), float(width)))
        finally:
            if capture:
                capture.release()
    if len(frames) < 30:
        raise ValueError("Replay requires at least 30 recorded camera frames")
    return frames


def run(args):
    """在独立 domain 中固定 dry-run，校验完整输出和服务状态切换。"""
    domain = os.environ.get("ROS_DOMAIN_ID", "")
    if not domain.isdigit() or int(domain) == 0:
        raise ValueError("Set a nonzero isolated ROS_DOMAIN_ID")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="trt-replay-", dir=args.output.parent) as temporary:
        frames = load_frames(args.episode.resolve(), args.frames, Path(temporary))
    rclpy.init()
    node = rclpy.create_node("dp_trt_smoke_replay")
    image_topic = "/wrist_camera/replay_raw" if args.image_type == "raw" else "/wrist_camera/replay_compressed"
    image_class = Image if args.image_type == "raw" else CompressedImage
    images = node.create_publisher(image_class, image_topic, 10)
    joints = node.create_publisher(JointState, "/joint_states", 10)
    gripper = node.create_publisher(Float32, "/motion_control/gripper_state", 10)
    predictions, debug_joints, arm_commands, gripper_commands, approvals = [], [], [], [], []
    subscriptions = [
        node.create_subscription(PolicyActionSequence, "/fastumi/policy/action_sequence",
                                 lambda m: predictions.append((m, node.get_clock().now().nanoseconds)), 10),
        node.create_subscription(JointState, "/fastumi/rm75/placo/joint_command", debug_joints.append, 10),
        node.create_subscription(Jointpos, "/rm_driver/movej_canfd_cmd", arm_commands.append, 10),
        node.create_subscription(Float32, "/motion_control/gripper_command", gripper_commands.append, 10),
        node.create_subscription(Bool, "/fastumi/policy/gripper_close_allowed",
                                 lambda m: approvals.append(m.data), 10),
    ]
    services = {name: node.create_client(Trigger, f"/fastumi/policy/{name}")
                for name in ("start_task", "stop_task", "reset_episode", "return_to_start")}
    steps_client = node.create_client(SetNumInferenceSteps, "/fastumi/policy/set_inference_steps")
    controller_probes = [node.create_client(ResetPolicyController, f"/fastumi/rm75/placo/{name}")
                         for name in ("start_task", "reset_episode", "return_to_start")]
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    process = None
    enabled = threading.Event()
    feeding = threading.Event()
    feeding.set()
    feeder_errors = []
    frame_index = [0]
    # 开始任务必须满足原控制器的静止条件；准备阶段重发同一真实观测，
    # 确认开始后才推进录制轨迹。重启时也先固定一帧等待反馈稳定。
    settling_frame = [frames[0]]
    replay_thread = None
    launch_log = args.output.with_suffix(".launch.log")

    def publish_frame(frame):
        bgr, angles, width = frame
        stamp = node.get_clock().now().to_msg()
        joint = JointState()
        joint.header.stamp = stamp
        joint.name = JOINT_NAMES
        joint.position = angles
        joints.publish(joint)
        gripper.publish(Float32(data=width))
        image = image_class()
        image.header.stamp = stamp
        if args.image_type == "raw":
            image.height, image.width = bgr.shape[:2]
            image.encoding = "bgr8"
            image.step = image.width * 3
            # array('B') 走 ROS 消息的原生字节路径，避免 Python 逐像素类型检查
            # 使大尺寸 raw 图像在生成时间戳后耗尽观测的新鲜度窗口。
            image.data = array("B", bgr.tobytes())
        else:
            success, encoded = cv2.imencode(".jpg", bgr)
            if not success:
                raise ValueError("JPEG replay encoding failed")
            image.format = "jpeg"
            image.data = array("B", encoded.tobytes())
        images.publish(image)

    def replay():
        try:
            deadline = time.monotonic()
            while enabled.is_set():
                if feeding.is_set():
                    publish_frame(settling_frame[0] if settling_frame[0] is not None
                                  else frames[frame_index[0] % len(frames)])
                    frame_index[0] += 1
                deadline += 1 / 30
                time.sleep(max(0, deadline - time.monotonic()))
        except Exception as error:
            feeder_errors.append(str(error))

    def trigger(name, expected=True, startup=False):
        if not services[name].wait_for_service(timeout_sec=3):
            raise TimeoutError(f"Missing {name}")
        ready_deadline = time.monotonic() + 5
        while True:
            future = services[name].call_async(Trigger.Request())
            deadline = time.monotonic() + 5
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not future.done():
                raise AssertionError(f"{name}: timeout")
            result = future.result()
            if bool(result.success) == expected:
                return result.message
            transient = result.message in (
                "Controller service is unavailable", "RM75 or gripper is not stable at start",
                "Image, joints or gripper feedback is stale")
            if not startup or not transient or time.monotonic() >= ready_deadline:
                raise AssertionError(f"{name}: {result}")
            # 仅首次开始允许等待 DDS 发现及新鲜反馈；不绕过节点/控制器门控。
            time.sleep(0.15)

    service_results = {}
    try:
        time.sleep(0.5)
        if any(node.count_publishers(topic) for topic in (
                "/joint_states", "/motion_control/gripper_state",
                "/rm_driver/movej_canfd_cmd", "/motion_control/gripper_command")):
            # 自己是观测发布者：关节和夹爪仅允许各一个，命令必须为零。
            if (node.count_publishers("/joint_states") != 1
                    or node.count_publishers("/motion_control/gripper_state") != 1
                    or node.count_publishers("/rm_driver/movej_canfd_cmd")
                    or node.count_publishers("/motion_control/gripper_command")):
                raise RuntimeError("Replay domain already contains hardware/command publishers")
        command = ["ros2", "launch", "dp_infer_tensorrt", "dp_infer_tensorrt.launch.py",
                   f"engine_dir:={args.engine_dir.resolve()}", f"urdf_path:={args.urdf.resolve()}",
                   "dry_run:=true", "start_controller:=true", f"image_type:={args.image_type}",
                   f"image_topic:={image_topic}", "num_inference_steps:=8", "precision:=fp16"]
        with launch_log.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=os.environ.copy())
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Launch exited; see {launch_log}")
                # 同时等待推理端和控制器；只看到推理订阅时控制器可能尚未发现反馈。
                if (images.get_subscription_count() >= 1
                        and joints.get_subscription_count() >= 2
                        and gripper.get_subscription_count() >= 2):
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError("Inference subscriptions missing")
            enabled.set()
            replay_thread = threading.Thread(target=replay, daemon=True)
            replay_thread.start()
            for client in controller_probes:
                if not client.wait_for_service(timeout_sec=5):
                    raise TimeoutError("Placo task services were not discovered")
            time.sleep(1.0)
            if predictions:
                raise AssertionError("Managed node published before start_task")
            service_results["start"] = trigger("start_task", startup=True)
            settling_frame[0] = None
            time.sleep(args.frames / 30)
            if not predictions or not debug_joints:
                raise AssertionError("Missing TensorRT prediction or Placo dry-run target")
            first_episode = predictions[-1][0].episode_id
            service_results["stop"] = trigger("stop_task")
            stopped_count = len(predictions)
            time.sleep(0.3)
            if len(predictions) != stopped_count:
                raise AssertionError("Prediction published after stop acknowledgement")
            feeding.clear()
            settling_frame[0] = frames[frame_index[0] % len(frames)]
            time.sleep(0.3)
            service_results["stale_start_rejected"] = trigger("start_task", expected=False)
            feeding.set()
            time.sleep(1.0)
            service_results["restart"] = trigger("start_task", startup=True)
            settling_frame[0] = None
            time.sleep(0.8)
            if len(predictions) <= stopped_count or predictions[-1][0].episode_id <= first_episode:
                raise AssertionError("Restart did not use a new episode")
            request = SetNumInferenceSteps.Request(num_inference_steps=16)
            future = steps_client.call_async(request)
            deadline = time.monotonic() + 3
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            if not future.done() or not future.result().success:
                raise AssertionError("Dynamic denoising service failed")
            time.sleep(0.8)
            service_results["reset"] = trigger("reset_episode")
            service_results["dry_run_return_rejected"] = trigger("return_to_start", expected=False)
            if feeder_errors:
                raise RuntimeError(feeder_errors)
            if arm_commands or gripper_commands:
                raise AssertionError("dry-run emitted real hardware commands")
    finally:
        enabled.clear()
        if replay_thread:
            replay_thread.join(timeout=3)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        executor.shutdown()
        spinner.join(timeout=3)
        node.destroy_node()
        rclpy.shutdown()

    ages, sequence_ids = [], []
    for message, received in predictions:
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        age = (received - stamp) / 1e6
        offsets = np.array([t.sec * 1_000_000_000 + t.nanosec for t in message.time_from_start])
        positions = np.array([[p.position.x, p.position.y, p.position.z] for p in message.poses])
        rotations = np.array([[p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
                              for p in message.poses])
        widths = np.array(message.gripper_openness)
        if (message.header.frame_id != "base_link" or message.end_frame != "Link7"
                or positions.shape != (16, 3) or rotations.shape != (16, 4) or widths.shape != (16,)
                or not np.array_equal(offsets, np.rint(np.arange(16) / 30 * 1e9).astype(np.int64))
                or not 0 <= age <= 510 or not np.isfinite(positions).all()
                or not np.isfinite(rotations).all() or not np.allclose(np.linalg.norm(rotations, axis=1), 1, atol=1e-5)
                or not np.isfinite(widths).all() or np.any((widths < 0) | (widths > 1))):
            raise AssertionError("Invalid policy message")
        ages.append(age)
        sequence_ids.append(message.sequence_id)
    if any(b <= a for a, b in zip(sequence_ids, sequence_ids[1:])):
        raise AssertionError("Sequence IDs did not increase")
    for message in debug_joints:
        if message.name != JOINT_NAMES or len(message.position) != 7 or not np.isfinite(message.position).all():
            raise AssertionError("Invalid Placo dry-run joints")
    if arm_commands or gripper_commands or not approvals:
        raise AssertionError("dry-run command or guard contract failed")
    report = {"status": "passed", "image_type": args.image_type, "source_frames": len(frames),
              "start_preparation": "Hold a recorded observation at 30 Hz until stable; advance recording after controller acknowledgement",
              "replayed_frames": frame_index[0], "policy_sequences": len(predictions),
              "debug_joint_targets": len(debug_joints), "arm_commands": len(arm_commands),
              "gripper_commands": len(gripper_commands), "guard_messages": len(approvals),
              "guard_true": sum(approvals), "age_p50_ms": float(np.percentile(ages, 50)),
              "age_p95_ms": float(np.percentile(ages, 95)), "age_max_ms": max(ages),
              "services": service_results, "launch_log": str(launch_log)}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    """指定录制 episode、模型目录、训练 URDF 和报告位置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-type", choices=("raw", "compressed"), default="compressed")
    parser.add_argument("--frames", type=int, default=180)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
