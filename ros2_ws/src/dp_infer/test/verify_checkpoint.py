"""从一轮实际录制取两帧，按固定种子生成可比较的模型预测。"""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import h5py
import numpy as np
import torch


def observations_from_episode(episode, urdf, core):
    """取视频前两帧及其最近关节/夹爪状态，构造训练格式观测。"""
    from diffusion_policy.common.urdf_kinematics import UrdfKinematics

    with h5py.File(episode / "proprio.hdf5", "r") as source:
        camera_times = source["observations/images/cam_gripper_timestamp"][:2]
        joint_times = source["observations/joint_state/timestamp"][:]
        joint_values = source["observations/joint_state/qpos"][:]
        gripper_times = source["observations/gripper_state/timestamp"][:]
        gripper_values = source["observations/gripper_state/position"][:, 0]
    camera = cv2.VideoCapture(str(episode / "gripper.mp4"))
    kinematics = UrdfKinematics(urdf, core.JOINT_NAMES)
    history = []
    try:
        for index, camera_time in enumerate(camera_times):
            valid, image = camera.read()
            if not valid:
                raise ValueError("recorded video has fewer than two frames")
            joint_index = int(np.argmin(np.abs(joint_times - camera_time)))
            gripper_index = int(np.argmin(np.abs(gripper_times - camera_time)))
            history.append(core.Observation(
                index * 33_333_333,
                core.letterbox_rgb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                kinematics.forward(joint_values[joint_index]),
                float(gripper_values[gripper_index]),
            ))
    finally:
        camera.release()
    return core.build_observations(history, history[0].pose), history[-1].pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--episode", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--num-inference-steps", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    if args.reference:
        # 原入口由模型项目提供；单独进程运行，避免同名 diffusion_policy 混用。
        sys.path[:] = [
            entry for entry in sys.path
            if "/ros2_ws/install/dp_infer/" not in entry
            and not entry.endswith("/ros2_ws/src/dp_infer")
        ]
        sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "model/dp"))
        from vr_umi_ros import core
        engine = core.PolicyEngine(args.checkpoint, "cuda:0")
        if args.num_inference_steps != 16:
            raise ValueError("reference checkpoint uses 16 DDIM steps")
    else:
        from dp_infer import core
        engine = core.PolicyEngine(
            args.checkpoint, "cuda:0", num_inference_steps=args.num_inference_steps)
    observations, reference_pose = observations_from_episode(args.episode, args.urdf, core)
    context = core.InferenceContext(reference_pose, 33_333_333, 0, 1)
    torch.manual_seed(42)
    torch.cuda.synchronize()
    started = time.perf_counter()
    prediction = engine.predict(observations, context)
    torch.cuda.synchronize()
    duration = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        positions=prediction.positions,
        quaternions=prediction.quaternions,
        gripper_openness=prediction.gripper_openness,
        time_from_start=prediction.time_from_start,
    )
    print(json.dumps({
        "reference": args.reference, "steps": args.num_inference_steps,
        "prediction_seconds": duration, "output": str(args.output),
        "finite": bool(np.isfinite(prediction.positions).all()
                       and np.isfinite(prediction.quaternions).all()
                       and np.isfinite(prediction.gripper_openness).all()),
    }))


if __name__ == "__main__":
    main()
