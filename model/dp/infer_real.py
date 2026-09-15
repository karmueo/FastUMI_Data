"""读取 RM75 实机观测并持续发布 FastUMI Link7 策略序列。"""

import argparse
from pathlib import Path

DEFAULT_CHECKPOINT = Path.home() / "data/model/DP/checkpoints/best.ckpt"
DEFAULT_URDF = Path(__file__).resolve().parent / "assets/rm_75_kinematic.urdf"


def parse_args(argv=None):
    """解析实机专用参数；ROS 参数仍可放在 ``--ros-args`` 后。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", "--ckpt_path", dest="checkpoint", type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Link7 checkpoint (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--camera-topic", default="/camera/image_raw/compressed")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--gripper-state-topic", default="/motion_control/gripper_state")
    parser.add_argument("--max-inference-hz", type=float, default=10.0)
    arguments, unknown = parser.parse_known_args(argv)
    if unknown and "--ros-args" not in unknown:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    return arguments


def validate_paths(arguments):
    """在模型分配 GPU 内存前检查部署输入。"""
    if not arguments.checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {arguments.checkpoint}")
    if not arguments.urdf_path.is_file():
        raise ValueError(f"URDF does not exist: {arguments.urdf_path}")


def main(argv=None):
    """加载可信模型并运行纯推理 ROS 节点。"""
    arguments = parse_args(argv)
    validate_paths(arguments)

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.signals import SignalHandlerOptions

    from vr_umi_ros.core import (PolicyEngine, TRUSTED_LEGACY_CHECKPOINT_SHA256,
                                 TRUSTED_LEGACY_URDF_SHA256, file_sha256)
    from vr_umi_ros.node import VrUmiInferenceNode

    checkpoint_sha256 = file_sha256(arguments.checkpoint)
    import torch
    import cv2

    # 单 CPU worker 避免与 ROS 回调争用 AGX 核心；策略前向主要由 CUDA 完成。
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)

    rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
    inference_node = None
    ros_executor = MultiThreadedExecutor(num_threads=4)
    try:
        engine = PolicyEngine(
            arguments.checkpoint, arguments.device,
            trusted_legacy_sha256=TRUSTED_LEGACY_CHECKPOINT_SHA256,
            checkpoint_sha256=checkpoint_sha256)
        if (engine.legacy_missing_urdf_hash
                and file_sha256(arguments.urdf_path) != TRUSTED_LEGACY_URDF_SHA256):
            raise ValueError("trusted legacy checkpoint requires the bundled verified RM75 URDF")
        inference_node = VrUmiInferenceNode(
            engine=engine,
            parameter_defaults={
                "urdf_path": str(arguments.urdf_path),
                "image_topic": arguments.camera_topic,
                "image_type": "compressed",
                "joint_topic": arguments.joint_topic,
                "gripper_topic": arguments.gripper_state_topic,
                "synchronous_inference": False,
                "max_inference_hz": arguments.max_inference_hz,
            })
        ros_executor.add_node(inference_node)
        print(
            f"checkpoint={arguments.checkpoint}\n"
            f"checkpoint_sha256={checkpoint_sha256}\n"
            "mode=inference-only\n"
            "output=/fastumi/policy/action_sequence",
            flush=True)
        ros_executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ros_executor.shutdown()
        if inference_node is not None:
            ros_executor.remove_node(inference_node)
            inference_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
