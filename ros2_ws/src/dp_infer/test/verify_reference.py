"""独立重放参考推理输入，并比较安装后 DP 包与训练侧的固定随机种子输出。"""

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from diffusion_policy.common.urdf_kinematics import UrdfKinematics
from dp_infer.core import (
    InferenceContext, JOINT_NAMES, Observation, PolicyEngine,
    build_observations, decode_actions, letterbox_rgb, validate_urdf,
)


def _sha256(path):
    """流式计算文件摘要，供参考输入和模型完整性检查。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare(actual, expected, atol, rtol):
    """逐字段比较形状和数值，返回最大绝对误差及通过状态。"""
    if actual.shape != expected.shape:
        return {"passed": False, "shape": list(actual.shape), "expected_shape": list(expected.shape)}
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        "passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "shape": list(actual.shape), "max_abs_error": float(delta.max(initial=0)),
    }


def main():
    """加载参考 NPZ、运行新引擎、写出完整比较报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads((args.reference_dir / "manifest.json").read_text())  # 基准完整性清单。
    input_path = args.reference_dir / "reference_input.npz"
    output_path = args.reference_dir / "reference_output.npz"
    integrity = {  # 输入、预期输出、模型和 URDF 必须与基准完全一致。
        "input": _sha256(input_path) == manifest["input_sha256"],
        "output": _sha256(output_path) == manifest["output_sha256"],
        "checkpoint": _sha256(args.checkpoint) == manifest["checkpoint_sha256"],
        "urdf": _sha256(args.urdf) == manifest["urdf_sha256"],
    }
    if not all(integrity.values()):
        raise ValueError(f"Reference integrity check failed: {integrity}")

    with np.load(input_path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}  # 不保留已关闭归档视图。
    engine = PolicyEngine(args.checkpoint, args.device,
                          expected_urdf_sha256=manifest["urdf_sha256"])
    validate_urdf(args.urdf, engine.cfg, manifest["urdf_sha256"])
    kinematics = UrdfKinematics(args.urdf, JOINT_NAMES)  # 与在线节点相同的 FK。
    poses = kinematics.forward(values["joint_positions"])
    start_pose = kinematics.forward(values["start_joint_positions"])
    history = [  # 两帧训练一致的观测。
        Observation(int(values["stamps_ns"][index]),
                    letterbox_rgb(values["images_rgb"][index]),
                    poses[index], float(values["gripper_openness"][index]))
        for index in range(2)
    ]
    observations = build_observations(history, start_pose)
    seed = int(manifest["seed"])  # 对扩散噪声应用基准随机种子。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    raw_action = engine.predict_raw(observations)
    sequence = decode_actions(raw_action[0], InferenceContext(poses[-1], 0, 0, 0))
    actual = {
        **{f"observation__{name}": value for name, value in observations.items()},
        "raw_action": raw_action, "positions": sequence.positions,
        "quaternions": sequence.quaternions,
        "gripper_openness": sequence.gripper_openness,
        "time_from_start": sequence.time_from_start,
    }
    with np.load(output_path, allow_pickle=False) as archive:
        expected = {name: archive[name] for name in archive.files}  # 基准输出数组。
    if set(actual) != set(expected):
        raise ValueError("Reference output fields differ")
    comparisons = {  # 记录每个输出的误差和验收状态。
        name: _compare(actual[name], expected[name], args.atol, args.rtol)
        for name in sorted(actual)
    }
    observation_fields = [name for name in comparisons if name.startswith("observation__")]
    observation_exact = all(np.array_equal(actual[name], expected[name])
                            for name in observation_fields)
    report = {
        "passed": observation_exact and all(value["passed"] for value in comparisons.values()),
        "observation_inputs_exact": observation_exact, "integrity": integrity,
        "comparisons": comparisons, "atol": args.atol, "rtol": args.rtol,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
