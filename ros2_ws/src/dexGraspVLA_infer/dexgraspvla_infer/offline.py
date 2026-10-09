"""离线 HDF5 相机/状态 -> YOLO/SAM/Cutie -> 真实 controller；无 ROS 发布。"""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from dexgraspvla_infer.core import Observation
from dexgraspvla_infer.model_runtime import ModelRuntime


def perception_runtime(assets, device):
    """创建使用本地来源和权重的感知组件。"""
    from dexgraspvla_infer.detector import JingbaoDetector
    from dexgraspvla_infer.perception import MaskValidator, SamCutieTracker, TargetPerception
    sys.path.insert(0, str(assets / "third_party/Cutie"))
    tracker = SamCutieTracker(sam_checkpoint=assets / "weights/segmentation/sam_vit_h_4b8939.pth",
                             cutie_checkpoint=assets / "weights/tracking/cutie-base-mega.pth",
                             torch_hub_dir=assets / "torch_hub", device=device,
                             validator=MaskValidator(min_area_fraction=0.0005, max_area_fraction=0.1, max_area_ratio=4))
    return TargetPerception(JingbaoDetector(assets / "weights/detection/yolo26s_jingbao.pt", device=device), tracker)


def load_samples(path, count=4):
    """读取本仓库硬件 HDF5 结构，按相机时间选最近反馈。"""
    import h5py
    with h5py.File(path, "r") as file:
        if "observations/images/cam_gripper_timestamp" in file:
            times = np.asarray(file["observations/images/cam_gripper_timestamp"])
            joints = file["observations/joint_state/qpos"]
            joint_times = np.asarray(file["observations/joint_state/timestamp"])
            grippers = file["observations/gripper_state/position"]
            gripper_times = np.asarray(file["observations/gripper_state/timestamp"])
            capture = cv2.VideoCapture(str(Path(path).parent / "gripper.mp4"))
            try:
                for index in range(min(count, len(times))):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                    ok, bgr = capture.read()
                    if not ok:
                        raise ValueError(f"Cannot decode video frame {index}")
                    stamp = float(times[index])
                    qi, gi = int(np.argmin(abs(joint_times - stamp))), int(np.argmin(abs(gripper_times - stamp)))
                    if max(abs(joint_times[qi] - stamp), abs(gripper_times[gi] - stamp)) > 0.05:
                        raise ValueError("offline feedback exceeds 50ms sync tolerance")
                    yield bgr, np.r_[np.asarray(joints[qi]), float(grippers[gi, 0])], int(stamp * 1e9)
            finally:
                capture.release()
            return
        cameras = file["observation/camera"]
        # Repository recordings use wrist_camera; accept a sole named camera.
        if "timestamp" not in cameras:
            cameras = cameras["wrist_camera"] if "wrist_camera" in cameras else cameras[next(iter(cameras))]
        times = np.asarray(cameras["timestamp"]).reshape(-1)
        images = cameras["image"] if "image" in cameras else cameras["color"]
        joints = file["observation/robot_state/joint_position"]
        joint_times = np.asarray(file["observation/robot_state/timestamp"]).reshape(-1)
        grippers = file["observation/gripper_state"]
        gripper_times = np.asarray(grippers["timestamp"]).reshape(-1)
        gripper_values = grippers["position"] if "position" in grippers else grippers["gripper_position"]
        for index in range(min(count, len(times))):
            value = np.asarray(images[index])
            bgr = cv2.imdecode(value.astype(np.uint8), cv2.IMREAD_COLOR) if value.ndim == 1 else value
            if bgr is None:
                raise ValueError("failed to decode HDF5 camera frame")
            stamp = float(times[index])
            q = np.asarray(joints[int(np.argmin(abs(joint_times - stamp)))]).reshape(7)
            g = float(np.asarray(gripper_values[int(np.argmin(abs(gripper_times - stamp)))]).reshape(-1)[0])
            yield bgr, np.r_[q, g], int(stamp * 1e9)


def main():
    """记录有限值、范围、动作形状和耗时；结果只写指定输出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "model-root", "asset-root", "urdf", "hdf5", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    runtime = ModelRuntime(args.checkpoint, args.model_root, args.asset_root, args.urdf, args.device)
    perception = perception_runtime(args.asset_root, args.device)
    report = []
    for index, (bgr, state, stamp) in enumerate(load_samples(args.hdf5, args.samples)):
        started = time.monotonic()
        if perception.tracker.initialized:
            result = perception.update(bgr)
        else:
            result, _ = perception.initialize(bgr)
        mask_seconds = time.monotonic() - started
        obs = Observation(stamp, 1, bgr, result.mask, state)
        predicted = runtime.predict(obs, args.steps)
        report.append({"sample": index, "shape": list(predicted.shape), "mask_s": mask_seconds,
                       "total_s": time.monotonic() - started, "finite": bool(np.isfinite(predicted).all()),
                       "gripper_range": [float(predicted[:, 7].min()), float(predicted[:, 7].max())]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
