"""生成并校验可跨机器复制的 VR UMI 确定性参考推理样本。"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random

import numpy as np
import torch
import zarr

from diffusion_policy.common.urdf_kinematics import UrdfKinematics
from vr_umi_ros.core import (InferenceContext, JOINT_NAMES, Observation, PolicyEngine,
                             build_observations, letterbox_rgb, validate_urdf)


SCHEMA_VERSION = 1  # 参考文件格式版本，变更字段语义时递增。
INPUT_FILE = "reference_input.npz"  # 可复制到其他机器的固定原始输入。
OUTPUT_FILE = "reference_output.npz"  # 基准机器生成的模型与解码输出。
MANIFEST_FILE = "manifest.json"  # 记录摘要、运行环境和比较约定。


def sha256_file(path):
    """流式计算文件 SHA-256，避免将大型 checkpoint 整体读入内存。"""
    digest = hashlib.sha256()  # 当前文件的增量摘要。
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def episode_bounds(episode_ends, index):
    """返回给定样本所在 episode 的左闭右开边界。"""
    ends = np.asarray(episode_ends, dtype=np.int64)  # 所有 episode 的累计结束下标。
    episode = int(np.searchsorted(ends, index, side="right"))  # 样本所属 episode 编号。
    if index < 0 or episode >= len(ends):
        raise ValueError(f"Sample index is outside the dataset: {index}")
    start = 0 if episode == 0 else int(ends[episode - 1])  # 当前 episode 起始下标。
    return start, int(ends[episode])


def save_reference_input(dataset_path, joint_dataset_path, sample_index, output_path):
    """从匹配数据集中提取两帧原始输入，并写入不含 pickle 对象的 NPZ。"""
    poses = zarr.open_group(str(dataset_path), mode="r")
    joints = zarr.open_group(str(joint_dataset_path), mode="r")
    pose_ends = poses["meta/episode_ends"][:]
    joint_ends = joints["meta/episode_ends"][:]
    np.testing.assert_array_equal(pose_ends, joint_ends)
    episode_start, episode_end = episode_bounds(pose_ends, sample_index)
    if sample_index <= episode_start:
        raise ValueError("sample-index must have a preceding frame in the same episode")
    indices = np.array([sample_index - 1, sample_index], dtype=np.int64)  # 两帧历史下标。
    pose_data = poses["data"]
    joint_data = joints["data"]
    np.testing.assert_array_equal(pose_data["source_image_index"][indices],
                                  joint_data["source_image_index"][indices])
    np.testing.assert_array_equal(pose_data["timestamp"][indices], joint_data["timestamp"][indices])
    timestamps = np.asarray(pose_data["timestamp"][indices], dtype=np.float64)  # 数据集秒时间戳。
    stamps_ns = np.rint((timestamps - timestamps[0]) * 1e9).astype(np.int64)  # 可移植相对纳秒时间。
    np.savez_compressed(
        output_path,
        schema_version=np.array(SCHEMA_VERSION, dtype=np.int64),
        sample_index=np.array(sample_index, dtype=np.int64),
        episode_start_index=np.array(episode_start, dtype=np.int64),
        episode_end_index=np.array(episode_end, dtype=np.int64),
        stamps_ns=stamps_ns,
        images_rgb=np.asarray(pose_data["camera0_rgb"][indices], dtype=np.uint8),
        joint_positions=np.asarray(joint_data["robot0_joint_pos"][indices], dtype=np.float64),
        gripper_openness=np.asarray(joint_data["robot0_gripper_position"][indices, 0],
                                    dtype=np.float64),
        start_joint_positions=np.asarray(joint_data["robot0_joint_pos"][episode_start],
                                         dtype=np.float64),
    )


def load_input(path):
    """安全读取并严格校验参考输入的字段、形状、范围和有限性。"""
    with np.load(path, allow_pickle=False) as archive:
        required = {"schema_version", "sample_index", "episode_start_index", "episode_end_index",
                    "stamps_ns", "images_rgb", "joint_positions", "gripper_openness",
                    "start_joint_positions"}
        if set(archive.files) != required or int(archive["schema_version"]) != SCHEMA_VERSION:
            raise ValueError("Unsupported or incomplete reference input")
        values = {name: archive[name].copy() for name in required}  # 关闭归档后仍可使用的数组。
    expected_shapes = {
        "stamps_ns": (2,), "images_rgb": (2, 224, 224, 3), "joint_positions": (2, 7),
        "gripper_openness": (2,), "start_joint_positions": (7,),
    }
    for name, shape in expected_shapes.items():
        if values[name].shape != shape:
            raise ValueError(f"Invalid reference input shape for {name}: {values[name].shape}")
    for name in ("joint_positions", "gripper_openness", "start_joint_positions"):
        if not np.isfinite(values[name]).all():
            raise ValueError(f"Reference input contains non-finite {name}")
    if np.any((values["gripper_openness"] < 0) | (values["gripper_openness"] > 1)):
        raise ValueError("Reference gripper values must be normalized to [0,1]")
    return values


def build_reference_observations(values, urdf_path):
    """对原始输入执行在线节点等价的图像预处理、FK 和五键观测构造。"""
    kinematics = UrdfKinematics(urdf_path, JOINT_NAMES)  # 部署 URDF 对应的正运动学。
    poses = kinematics.forward(values["joint_positions"])  # 两帧 base→Link7 位姿。
    start_pose = kinematics.forward(values["start_joint_positions"])  # episode 起始位姿。
    history = [
        Observation(int(values["stamps_ns"][index]), letterbox_rgb(values["images_rgb"][index]),
                    poses[index], float(values["gripper_openness"][index]))
        for index in range(2)
    ]
    return build_observations(history, start_pose), poses[-1]


def seed_inference(seed):
    """在每次参考推理前重置 CPU/CUDA 随机状态，使扩散采样输入固定。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_reference(engine, observations, reference_pose, seed):
    """运行固定随机种子的原始预测与绝对动作解码，返回可保存数组。"""
    from vr_umi_ros.core import decode_actions

    seed_inference(seed)
    raw_action = engine.predict_raw(observations)  # 模型原始相对动作。
    context = InferenceContext(reference_pose, 0, 0, 0)  # 解码所需的冻结参考位姿。
    sequence = decode_actions(raw_action[0], context)
    return {
        **{f"observation__{name}": value for name, value in observations.items()},
        "raw_action": raw_action,
        "positions": sequence.positions,
        "quaternions": sequence.quaternions,
        "gripper_openness": sequence.gripper_openness,
        "time_from_start": sequence.time_from_start,
    }


def environment_info(device):
    """返回便于定位跨机器差异的核心软件与设备版本。"""
    info = {
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "device": device, "cuda": torch.version.cuda,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(torch.device(device))
    return info


def compare_arrays(actual, expected, atol, rtol):
    """比较两个数组并返回形状、最大绝对/相对误差和通过状态。"""
    if actual.shape != expected.shape:
        return {"passed": False, "actual_shape": list(actual.shape),
                "expected_shape": list(expected.shape), "reason": "shape_mismatch"}
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    denominator = np.maximum(np.abs(expected.astype(np.float64)), np.finfo(np.float64).tiny)
    return {
        "passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "shape": list(actual.shape), "max_abs_error": float(difference.max(initial=0)),
        "max_rel_error": float((difference / denominator).max(initial=0)),
    }


def create_reference(args):
    """创建固定输入、期望输出与完整性清单。"""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / INPUT_FILE  # 原始参考输入路径。
    output_path = args.output_dir / OUTPUT_FILE  # 基准输出路径。
    save_reference_input(args.dataset, args.joint_dataset, args.sample_index, input_path)
    values = load_input(input_path)
    dataset_root = zarr.open_group(str(args.dataset), mode="r")  # 训练数据记录的 URDF 契约。
    expected_urdf_sha256 = dataset_root.attrs["urdf_sha256"]  # 兼容未内嵌摘要的旧 checkpoint。
    engine = PolicyEngine(args.checkpoint, args.device,
                          expected_urdf_sha256=expected_urdf_sha256)
    validate_urdf(args.urdf, engine.cfg, expected_urdf_sha256)
    observations, reference_pose = build_reference_observations(values, args.urdf)
    outputs = run_reference(engine, observations, reference_pose, args.seed)
    np.savez_compressed(output_path, **outputs)
    manifest = {
        "schema_version": SCHEMA_VERSION, "seed": args.seed,
        "input_sha256": sha256_file(input_path), "output_sha256": sha256_file(output_path),
        "checkpoint_sha256": sha256_file(args.checkpoint), "urdf_sha256": sha256_file(args.urdf),
        "sample_index": args.sample_index, "environment": environment_info(args.device),
        "comparison_defaults": {"atol": args.atol, "rtol": args.rtol},
    }
    (args.output_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def verify_reference(args):
    """在当前机器重跑参考输入，输出逐字段误差报告，并以退出码表示结果。"""
    input_path = args.reference_dir / INPUT_FILE  # 待验证的固定输入。
    output_path = args.reference_dir / OUTPUT_FILE  # 基准机器期望输出。
    manifest_path = args.reference_dir / MANIFEST_FILE  # 完整性与运行参数清单。
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    integrity = {
        "input": sha256_file(input_path) == manifest["input_sha256"],
        "output": sha256_file(output_path) == manifest["output_sha256"],
        "checkpoint": sha256_file(args.checkpoint) == manifest["checkpoint_sha256"],
        "urdf": sha256_file(args.urdf) == manifest["urdf_sha256"],
    }
    if not all(integrity.values()):
        raise ValueError(f"Reference integrity check failed: {integrity}")
    values = load_input(input_path)
    engine = PolicyEngine(args.checkpoint, args.device,
                          expected_urdf_sha256=manifest["urdf_sha256"])
    validate_urdf(args.urdf, engine.cfg, manifest["urdf_sha256"])
    observations, reference_pose = build_reference_observations(values, args.urdf)
    actual = run_reference(engine, observations, reference_pose, int(manifest["seed"]))
    with np.load(output_path, allow_pickle=False) as archive:
        expected = {name: archive[name] for name in archive.files}  # 基准机器保存的全部输出。
        if set(actual) != set(expected):
            raise ValueError("Reference output fields do not match the current schema")
        comparisons = {
            name: compare_arrays(actual[name], expected[name], args.atol, args.rtol)
            for name in sorted(actual)
        }
    observation_fields = [name for name in comparisons if name.startswith("observation__")]
    observation_exact = all(np.array_equal(actual[name], expected[name]) for name in observation_fields)
    report = {
        "passed": observation_exact and all(item["passed"] for item in comparisons.values()),
        "observation_inputs_exact": observation_exact, "integrity": integrity,
        "atol": args.atol, "rtol": args.rtol, "environment": environment_info(args.device),
        "comparisons": comparisons,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


def parser():
    """构建 create/verify 命令行参数解析器。"""
    root = argparse.ArgumentParser(description=__doc__)  # 顶层命令解析器。
    commands = root.add_subparsers(dest="command", required=True)  # 两种互斥工作模式。
    create = commands.add_parser("create", help="生成固定输入和基准输出")
    create.add_argument("--checkpoint", required=True, type=Path)
    create.add_argument("--dataset", required=True, type=Path)
    create.add_argument("--joint-dataset", required=True, type=Path)
    create.add_argument("--urdf", required=True, type=Path)
    create.add_argument("--output-dir", required=True, type=Path)
    create.add_argument("--sample-index", type=int, default=30)
    create.add_argument("--seed", type=int, default=20260904)
    create.add_argument("--device", default="cuda:0")
    create.add_argument("--atol", type=float, default=1e-4)
    create.add_argument("--rtol", type=float, default=1e-4)
    create.set_defaults(handler=create_reference)
    verify = commands.add_parser("verify", help="重跑固定输入并与基准输出比较")
    verify.add_argument("--checkpoint", required=True, type=Path)
    verify.add_argument("--urdf", required=True, type=Path)
    verify.add_argument("--reference-dir", required=True, type=Path)
    verify.add_argument("--report", required=True, type=Path)
    verify.add_argument("--device", default="cuda:0")
    verify.add_argument("--atol", type=float, default=1e-4)
    verify.add_argument("--rtol", type=float, default=1e-4)
    verify.set_defaults(handler=verify_reference)
    return root


def main():
    """解析参数并执行参考样本创建或验证。"""
    args = parser().parse_args()  # 已选择子命令的参数集合。
    torch.set_num_threads(4)
    args.handler(args)


if __name__ == "__main__":
    main()
