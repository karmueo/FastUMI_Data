"""对照 LeRobot 原版推理与本包移植实现，并在留出 episode 上评估开环误差。

三个子命令分别在不同环境运行，均从仓库根目录执行：

- ``export-reference``：LeRobot fork 训练环境（如 ``/home/scl/work/lerobot/.venv``）。从 LeRobotDataset
  取留出 episode 的真实样本，用原版 ``DiffusionPolicy`` + 处理链推理，固定初始噪声与随机种子。
- ``export-candidate``：本包推理环境（``ros2_ws/.venv-dp-lerobot``）。读取参考 NPZ 的同一输入，
  用 :class:`dp_infer_lerobot.engine.PolicyEngine` 推理；``--backend tensorrt_fp32/fp16 --export-dir``
  改用 LeRobot 导出的 TensorRT 模型包。
- ``compare``：任意含 NumPy/SciPy 的环境。默认逐项比较归一化状态、模型动作与最终动作（atol/rtol）；
  ``--physical`` 改为按最终绝对动作的物理门槛验收（位置 mm、旋转角 °、夹爪绝对差，用于 TensorRT），
  失败返回非零；报告还给出两者相对示教动作的开环误差。

动作 k 对应观测时刻 + k/30 秒，示教真值取同一 episode 的第 t..t+7 帧 action。
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

IMAGE_KEY = "observation.images.front"  # 相机特征键。
STATE_KEY = "observation.state"  # 原始 7D 状态键。
ACTION_STEPS = 8  # 每次推理的动作数。
HORIZON = 16  # 扩散去噪序列长度。
MODEL_DIM = 10  # 模型布局动作维度。
COMPARED = ("normalized_state", "model_actions", "actions")  # compare 子命令逐项比较的数组。
# --physical 默认门槛，与 LeRobot 导出验证一致：位置 1 mm、旋转 0.5°、夹爪 0.01。
PHYSICAL_LIMITS = {"position_mm": 1.0, "rotation_deg": 0.5, "gripper": 0.01}


def held_out_episodes(dataset, eval_split):
    """复现训练时的留出规则：每个任务的最后 ceil(n × eval_split) 个 episode。"""
    episode_tasks = dataset.meta.episodes["tasks"]
    by_task = {}
    for index in range(dataset.num_episodes):
        by_task.setdefault(episode_tasks[index][0] if episode_tasks[index] else "", []).append(index)
    held = []
    for episodes in by_task.values():
        held.extend(episodes[len(episodes) - math.ceil(len(episodes) * eval_split):])
    return sorted(held)


def sample_indices(dataset, episodes, per_episode, rng):
    """在每个 episode 内随机取 ``per_episode`` 个满足前 1 帧、后 7 帧均存在的全局帧号。"""
    meta = dataset.meta.episodes
    indices = []
    for episode in episodes:
        start, stop = int(meta["dataset_from_index"][episode]), int(meta["dataset_to_index"][episode])
        if stop - start < ACTION_STEPS + 1:
            continue
        candidates = np.arange(start + 1, stop - ACTION_STEPS + 1)
        indices.extend(int(i) for i in rng.choice(candidates, min(per_episode, len(candidates)), replace=False))
    return sorted(indices)


def export_reference(args):
    """在 LeRobot fork 环境中导出真实样本、固定噪声及原版推理结果。"""
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    checkpoint = Path(args.checkpoint)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.noise_scheduler_type = args.scheduler
    config.num_inference_steps = args.steps
    config.device = args.device
    policy = DiffusionPolicy.from_pretrained(checkpoint, config=config).to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=str(checkpoint))
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root, delta_timestamps={
        IMAGE_KEY: [-1 / 30, 0.0], STATE_KEY: [-1 / 30, 0.0],
        "action": [k / 30 for k in range(ACTION_STEPS)]})
    episodes = held_out_episodes(dataset, args.eval_split)
    if args.max_episodes:
        episodes = episodes[:: max(1, len(episodes) // args.max_episodes)][: args.max_episodes]
    rng = np.random.default_rng(args.seed)
    indices = sample_indices(dataset, episodes, args.per_episode, rng)
    records = {key: [] for key in ("index", "episode", "image", "state", "gt_actions", "noise", *COMPARED)}
    for number, index in enumerate(indices):
        item = dataset[index]
        image = item[IMAGE_KEY].numpy().astype(np.float32)  # (2,3,224,224)，[0,1]
        state = item[STATE_KEY].numpy().astype(np.float32)  # (2,7)
        noise = rng.standard_normal((1, HORIZON, MODEL_DIM)).astype(np.float32)
        batch = preprocessor({IMAGE_KEY: torch.from_numpy(image[None]), STATE_KEY: torch.from_numpy(state[None])})
        torch.manual_seed(args.seed + number)  # DDPM 每步方差噪声来自全局 RNG。
        with torch.inference_mode():
            model_actions = policy.predict_action_chunk(
                batch, noise=torch.from_numpy(noise).to(args.device))
            actions = postprocessor(model_actions)
        records["index"].append(index)
        records["episode"].append(int(item["episode_index"]))
        records["image"].append(image)
        records["state"].append(state)
        records["gt_actions"].append(item["action"].numpy().astype(np.float32))
        records["noise"].append(noise[0])
        records["normalized_state"].append(batch[STATE_KEY].cpu().numpy()[0])
        records["model_actions"].append(model_actions.cpu().numpy()[0])
        records["actions"].append(actions.cpu().numpy()[0])
    meta = {"checkpoint": str(checkpoint), "scheduler": args.scheduler, "steps": args.steps,
            "seed": args.seed, "episodes": episodes, "torch": torch.__version__,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
    save(args.output, records, meta)


def export_candidate(args):
    """在本包推理环境中用相同输入、噪声与种子执行移植实现。"""
    import torch

    from dp_infer_lerobot.deployment import load_profile
    from dp_infer_lerobot.engine import PolicyEngine

    sample = np.load(args.sample)
    meta = json.loads(str(sample["meta"]))
    if args.backend == "pytorch" and not args.checkpoint:
        raise SystemExit("--checkpoint is required for the pytorch backend")
    engine = PolicyEngine(args.checkpoint, load_profile(args.deployment_config), args.device,
                          meta["scheduler"], meta["steps"], backend=args.backend, export_dir=args.export_dir)
    records = {key: [] for key in (*COMPARED, "latency_ms")}
    for number in range(len(sample["index"])):
        observations = {IMAGE_KEY: sample["image"][number][None], STATE_KEY: sample["state"][number][None]}
        torch.manual_seed(meta["seed"] + number)
        if engine.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        actions, debug = engine.predict_raw(observations, noise=sample["noise"][number][None], return_debug=True)
        records["latency_ms"].append((time.perf_counter() - started) * 1000)
        records["normalized_state"].append(debug[f"normalized.{STATE_KEY}"][0])
        records["model_actions"].append(debug["model_actions"][0])
        records["actions"].append(actions[0])
    meta.update(candidate_torch=torch.__version__, candidate=engine.description)
    save(args.output, records, meta)


def save(path, records, meta):
    """把逐样本列表堆叠后连同 JSON 元信息写入压缩 NPZ。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in records.items()}
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)
    print(f"Wrote {path}: {len(next(iter(arrays.values())))} samples")


def pose_errors(actions, truth):
    """返回 ``(位置误差 mm, 旋转误差 度, 夹爪绝对误差)``，形状均为 (N, 8)。"""
    from scipy.spatial.transform import Rotation

    position = np.linalg.norm(actions[..., :3] - truth[..., :3], axis=-1) * 1000
    predicted = Rotation.from_rotvec(actions[..., 3:6].reshape(-1, 3))
    expected = Rotation.from_rotvec(truth[..., 3:6].reshape(-1, 3))
    rotation = np.degrees((predicted.inv() * expected).magnitude()).reshape(actions.shape[:-1])
    return position, rotation, np.abs(actions[..., 6] - truth[..., 6])


def summarize(actions, truth):
    """汇总开环误差：整体均值/P95 及末步（k=7）均值。"""
    result = {}
    for name, values in zip(("position_mm", "rotation_deg", "gripper"), pose_errors(actions, truth)):
        result[name] = {"mean": float(values.mean()), "p95": float(np.percentile(values, 95)),
                        "last_step_mean": float(values[:, -1].mean())}
    return result


def compare(args):
    """逐项比较参考与候选数组，输出报告；任一项超出容差返回 1。"""
    reference, candidate = np.load(args.reference), np.load(args.candidate)
    truth = reference["gt_actions"]
    # 基线：保持当前位姿与夹爪不动，衡量模型是否提供了有效的运动预测。
    hold = np.repeat(reference["state"][:, -1:, :], ACTION_STEPS, axis=1)
    report = {"meta": json.loads(str(candidate["meta"])), "samples": int(len(truth)),
              "atol": args.atol, "rtol": args.rtol, "checks": {},
              "physical_limits": PHYSICAL_LIMITS if args.physical else None,
              "latency_ms": {"median": float(np.median(candidate["latency_ms"])),
                             "p95": float(np.percentile(candidate["latency_ms"], 95))},
              "openloop": {"reference": summarize(reference["actions"], truth),
                           "candidate": summarize(candidate["actions"], truth),
                           "hold_current_pose": summarize(hold, truth)}}
    passed = True
    for key in COMPARED:
        difference = np.abs(reference[key] - candidate[key])
        ok = bool(np.allclose(candidate[key], reference[key], atol=args.atol, rtol=args.rtol))
        report["checks"][key] = {"passed": ok, "max_abs": float(difference.max())}
        if not args.physical:
            passed &= ok
    if args.physical:
        # TensorRT 与 PyTorch 内核不同，逐元素容差不适用；按最终绝对动作的物理偏差验收。
        errors = pose_errors(candidate["actions"], reference["actions"])
        for (name, limit), values in zip(PHYSICAL_LIMITS.items(), errors):
            ok = bool(np.isfinite(candidate["actions"]).all() and values.max() <= limit)
            report["checks"][f"actions.{name}"] = {"passed": ok, "max": float(values.max()),
                                                    "p95": float(np.percentile(values, 95))}
            passed &= ok
    report["passed"] = passed
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if passed else 1


def main(argv=None):
    """解析子命令并执行。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("export-reference")
    ref.add_argument("--checkpoint", required=True)
    ref.add_argument("--repo-id", required=True)
    ref.add_argument("--dataset-root", required=True)
    ref.add_argument("--eval-split", type=float, default=0.1)
    ref.add_argument("--max-episodes", type=int, default=8)
    ref.add_argument("--per-episode", type=int, default=4)
    ref.add_argument("--scheduler", choices=("DDPM", "DDIM"), default="DDPM")
    ref.add_argument("--steps", type=int, default=100)
    ref.add_argument("--seed", type=int, default=0)
    ref.add_argument("--device", default="cuda:0")
    ref.add_argument("--output", required=True)
    cand = sub.add_parser("export-candidate")
    cand.add_argument("--checkpoint", default="", help="PyTorch 后端必填的 pretrained_model 目录")
    cand.add_argument("--backend", choices=("pytorch", "tensorrt_fp32", "tensorrt_fp16"), default="pytorch")
    cand.add_argument("--export-dir", default="", help="TensorRT 后端使用的 lerobot-export 模型包")
    cand.add_argument("--deployment-config", required=True)
    cand.add_argument("--sample", required=True)
    cand.add_argument("--device", default="cuda:0")
    cand.add_argument("--output", required=True)
    cmp = sub.add_parser("compare")
    cmp.add_argument("--reference", required=True)
    cmp.add_argument("--candidate", required=True)
    cmp.add_argument("--report", default="")
    cmp.add_argument("--atol", type=float, default=1e-5)
    cmp.add_argument("--rtol", type=float, default=1e-4)
    cmp.add_argument("--physical", action="store_true", help="按物理门槛验收最终动作（TensorRT 对照）")
    args = parser.parse_args(argv)
    return {"export-reference": export_reference, "export-candidate": export_candidate,
            "compare": compare}[args.command](args) or 0


if __name__ == "__main__":
    sys.exit(main())
