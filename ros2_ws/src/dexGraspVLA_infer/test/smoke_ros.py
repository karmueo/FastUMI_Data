"""隔离 ROS 域中的真实 GPU + dry-run 集成测试，使用录制画面和模拟反馈。

只启动 dry-run 关节执行器，检查域内无驱动/夹爪命令发布者。
不会启动 hardware bringup、机械臂驱动或任何真实移动节点。
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from fastumi_interfaces.msg import PolicyJointActionSequence
from fastumi_interfaces.srv import SetNumInferenceSteps
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger

from dexgraspvla_infer.offline import load_samples


def main():
    """验证真实推理、50Hz debug、键盘接口、丢失后停车和回位拒绝。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--predictions", type=int, default=10)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if os.environ.get("ROS_DOMAIN_ID") != "153" or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise RuntimeError("Run with ROS_DOMAIN_ID=153 ROS_LOCALHOST_ONLY=1 to isolate this smoke test")
    log_root = root / "dataset/dexgraspvla_validation"
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "dataset/h5dy_data/rm75_DexGraspVLA/2026.10.01/15.52_train_dexgraspvla_controller_grasp_rm75/checkpoints/latest.ckpt"
    assets = root / ".local/dexgraspvla"
    gpu_cmd = [sys.executable, "-m", "dexgraspvla_infer.cli", "--ros-args",
               "-p", f"checkpoint:={checkpoint}", "-p", f"model_root:={root / 'model/DexGraspVLA'}",
               "-p", f"asset_root:={assets}", "-p", f"urdf_path:={root / 'model/DexGraspVLA/assets/rm75/rm_75.urdf'}",
               "-p", f"num_inference_steps:={args.steps}"]
    controller_cmd = [str(root / "ros2_ws/.venv-numpy2/bin/python"), "-m", "fastumi_rm75.rm75_joint_controller",
                      "--ros-args", "-p", "dry_run:=true"]
    bgr, _, _ = next(load_samples(root / "dataset/h5dy_data/rm75_vla/jingbao_merge/episode_78/proprio.hdf5", 1))
    native_source = Path(__file__).parent / "native_camera"
    native_build = log_root / "native_camera_build"
    subprocess.run(["cmake", "-S", str(native_source), "-B", str(native_build)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["cmake", "--build", str(native_build), "-j2"], check=True, stdout=subprocess.DEVNULL)
    frame_file = log_root / "fixed_frame.bgr"
    frame_file.write_bytes(bgr.tobytes())
    files = [open(log_root / "smoke_gpu.log", "w"), open(log_root / "smoke_controller.log", "w")]
    processes = [subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                 for command, stream in zip((gpu_cmd, controller_cmd), files)]
    rclpy.init()
    node = Node("dexgraspvla_smoke_source")
    joints = node.create_publisher(JointState, "/joint_states", 1)
    gripper = node.create_publisher(Float32, "/motion_control/gripper_state", 1)
    valid = node.create_publisher(Bool, "/rm_driver/udp_feedback_valid", 1)
    camera_process = subprocess.Popen([str(native_build / "mock_camera"), str(frame_file), str(bgr.shape[0]), str(bgr.shape[1])],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    processes.append(camera_process)
    home = np.deg2rad([0, 20, 0, 70, 0, 90, 90]).tolist()
    state, control, debug_times, sequences, metrics = {}, {}, [], [], []
    watch_running = [False]

    def feedback():
        message = JointState()
        message.header.stamp = node.get_clock().now().to_msg()
        message.name = [f"joint{i}" for i in range(1, 8)]
        message.position = home
        joints.publish(message)

    def gripper_feedback():
        gripper.publish(Float32(data=1.0))

    def udp_feedback():
        valid.publish(Bool(data=True))

    def policy_status(message):
        state.clear()
        state.update(json.loads(message.data))

    def controller_status(message):
        control.clear()
        control.update(json.loads(message.data))

    node.create_timer(0.005, feedback)
    node.create_timer(0.02, gripper_feedback)
    node.create_timer(0.05, udp_feedback)
    node.create_subscription(String, "/fastumi/policy/status", policy_status, 10)
    node.create_subscription(String, "/fastumi/rm75/joint/status", controller_status, 10)
    node.create_subscription(String, "/fastumi/policy/metrics", lambda m: metrics.append(json.loads(m.data)), 100)
    node.create_subscription(JointState, "/fastumi/rm75/joint/joint_command",
                             lambda m: debug_times.append(m.header.stamp.sec + m.header.stamp.nanosec / 1e9), 100)
    node.create_subscription(PolicyJointActionSequence, "/fastumi/policy/joint_action_sequence", sequences.append, 10)
    clients = {name: node.create_client(Trigger, "/fastumi/policy/" + name)
               for name in ("start_task", "stop_task", "return_to_start")}
    steps_client = node.create_client(SetNumInferenceSteps, "/fastumi/policy/set_inference_steps")

    def wait(predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("A smoke-test child exited; inspect smoke logs")
            rclpy.spin_once(node, timeout_sec=0.005)
            if watch_running[0] and state.get("state") not in ("running", "starting"):
                raise RuntimeError(f"Task stopped before completing predictions: {state}")
            if predicate():
                return
        raise TimeoutError(f"Smoke test timed out; policy={state}, controller={control}")

    def call(name):
        future = clients[name].call_async(Trigger.Request())
        wait(future.done, 5)
        return future.result()

    report = {}
    try:
        wait(lambda: state.get("ready") and control.get("dry_run"), 90)
        for topic in ("/rm_driver/movej_canfd_cmd", "/motion_control/gripper_command", "/rm_driver/movej_cmd", "/rm_driver/move_stop_cmd"):
            assert node.count_publishers(topic) == 0, f"Unexpected hardware publisher: {topic}"
        request = SetNumInferenceSteps.Request()
        request.num_inference_steps = args.steps
        future = steps_client.call_async(request)
        wait(future.done, 5)
        assert future.result().success
        started = call("start_task")
        assert started.success, started.message
        wait(lambda: state.get("state") == "running")
        watch_running[0] = True
        wait(lambda: len(metrics) >= args.predictions, max(30, args.predictions * 3))
        watch_running[0] = False
        assert len(sequences) and len(debug_times) >= 10
        assert call("stop_task").success
        wait(lambda: state.get("state") == "idle" and not control.get("enabled"))
        stopped_count = len(debug_times)
        until = time.monotonic() + 0.5
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=0.01)
        assert len(debug_times) <= stopped_count + 1, "commands continued after stop"
        assert not call("return_to_start").success
        wait(lambda: state.get("state") == "fault")
        assert call("stop_task").success
        wait(lambda: state.get("state") == "idle")
        settle_until = time.monotonic() + 0.6
        while time.monotonic() < settle_until:
            rclpy.spin_once(node, timeout_sec=0.005)
        assert call("start_task").success
        wait(lambda: control.get("enabled"))
        camera_process.send_signal(signal.SIGINT)
        camera_process.wait(timeout=5)
        processes.remove(camera_process)
        wait(lambda: state.get("state") == "idle" and not control.get("enabled"))
        from dexgraspvla_infer.benchmark import evaluate
        report = {"passed": True, "predictions": len(metrics), "debug_frames": len(debug_times),
                  "debug_hz": 1 / float(np.median(np.diff(debug_times))),
                  "no_hardware_publishers": True, "steps": args.steps,
                  "precision": "float32",
                  "start_stop_home_and_camera_loss": True, "performance": evaluate(metrics, args.predictions)}
    except Exception as error:
        report = {"passed": False, "error": str(error), "policy": state, "controller": control,
                  "predictions": len(metrics), "debug_frames": len(debug_times), "metrics": metrics}
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        node.destroy_node()
        rclpy.shutdown()
        for stream in files:
            stream.close()
    print(json.dumps(report))


if __name__ == "__main__":
    main()
