#!/usr/bin/env python3
"""在分别具备模型环境的主机导出参考/候选结果，再离线比较；不涉及 ROS 或硬件。"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

# 支持从源码运行，也支持已安装的 ROS 包。
package_root = Path(__file__).resolve().parents[1]
if (package_root / "dp_infer_lerobot").is_dir():
    sys.path.insert(0, str(package_root))
    sys.path.insert(0, str(package_root.parent / "dp_infer"))


def export(args):
    """显式执行真实模型，仅由用户在训练/Orin 环境调用。"""
    import torch

    from dp_infer_lerobot.deployment import sha256_file

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.sample:
        with np.load(args.sample, allow_pickle=False) as sample:
            image, state, noise = (sample[key].copy() for key in ("image", "state", "noise"))
    else:
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (1, 2, 3, 224, 224), dtype=np.uint8).astype(np.float32) / 255
        state = np.array([[[0.75], [0.8]]], dtype=np.float32)
        noise = rng.standard_normal((1, 16, 10)).astype(np.float32)
    if image.shape != (1, 2, 3, 224, 224) or state.shape != (1, 2, 1) or noise.shape != (1, 16, 10):
        raise ValueError("Invalid sample shapes")
    if any(not np.isfinite(value).all() for value in (image, state, noise)):
        raise ValueError("Non-finite sample")
    batch = {"observation.images.cam0": torch.as_tensor(image, device=args.device),
             "observation.state": torch.as_tensor(state, device=args.device)}
    if args.command == "export-reference":
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors

        if importlib.metadata.version("lerobot") != "0.6.0":
            raise ValueError("Reference environment must use LeRobot 0.6.0")
        cfg = DiffusionConfig.from_pretrained(args.checkpoint)
        cfg.device = args.device
        cfg.use_amp = False
        cfg.compile_model = False
        policy = DiffusionPolicy.from_pretrained(args.checkpoint, config=cfg, local_files_only=True)
        policy.to(args.device).eval()
        pre, post = make_pre_post_processors(
            cfg, pretrained_path=args.checkpoint,
            preprocessor_overrides={"device_processor": {"device": args.device}})
    else:
        from dp_infer_lerobot.deployment import load_profile
        from dp_infer_lerobot.engine import PolicyEngine

        engine = PolicyEngine(args.checkpoint, load_profile(args.deployment_config), args.device, args.steps)
        policy, pre, post = engine.policy, engine.processors.normalize, engine.processors.unnormalize
    policy.diffusion.num_inference_steps = args.steps
    noise_tensor = torch.as_tensor(noise, device=args.device)
    with torch.inference_mode():
        normalized = pre(batch)
        images = normalized["observation.images.cam0"]
        features = policy.diffusion.rgb_encoder(images.flatten(0, 1))
        grouped = dict(normalized)
        grouped["observation.images"] = images.unsqueeze(2)
        conditioning = policy.diffusion._prepare_global_conditioning(grouped)
        policy.diffusion.noise_scheduler.set_timesteps(args.steps)
        first_t = policy.diffusion.noise_scheduler.timesteps[0]
        denoised = policy.diffusion.unet(
            noise_tensor, torch.full((1,), int(first_t), device=args.device, dtype=torch.long),
            global_cond=conditioning)
        actions_norm = policy.predict_action_chunk(normalized, noise=noise_tensor.clone())
        actions = post(actions_norm)
    tensors = {"normalized_image": normalized["observation.images.cam0"],
               "normalized_state": normalized["observation.state"], "features": features,
               "conditioning": conditioning, "denoiser_first": denoised,
               "actions_normalized": actions_norm, "actions": actions}
    metadata = {"steps": args.steps, "device": args.device, "mode": args.command,
                "versions": {name: importlib.metadata.version(name)
                             for name in ("torch", "torchvision", "diffusers", "numpy")},
                "checkpoint_files": {p.name: sha256_file(p) for p in Path(args.checkpoint).iterdir()
                                     if p.name in ("config.json", "model.safetensors", "policy_preprocessor.json",
                                                   "policy_postprocessor.json") or p.name.endswith("_processor.safetensors")}}
    arrays = {key: value.detach().float().cpu().numpy() for key, value in tensors.items()}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Model returned non-finite trace")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(stream, image=image, state=state, noise=noise,
                            metadata=np.array(json.dumps(metadata)), **arrays)
    print(f"Exported {args.command} trace to {output}")


def compare(args):
    """固定输入/噪声/模型指纹后逐阶段比较；不一致返回非零，不自动放宽容差。"""
    with np.load(args.reference, allow_pickle=False) as reference, np.load(args.candidate, allow_pickle=False) as candidate:
        ref_meta, got_meta = (json.loads(data["metadata"].item()) for data in (reference, candidate))
        if ref_meta["mode"] != "export-reference" or got_meta["mode"] != "export-candidate":
            raise ValueError("Expected an original LeRobot reference and an adapted candidate")
        if any(ref_meta[key] != got_meta[key] for key in ("steps", "checkpoint_files")):
            raise ValueError("Denoising steps or checkpoint fingerprints differ")
        for key in ("image", "state", "noise"):
            if not np.array_equal(reference[key], candidate[key]):
                raise ValueError(f"Inputs differ: {key}")
        report = {"reference": ref_meta, "candidate": got_meta, "atol": args.atol,
                  "rtol": args.rtol, "stages": {}}
        for key in ("normalized_image", "normalized_state", "features", "conditioning",
                    "denoiser_first", "actions_normalized", "actions"):
            expected, actual = reference[key], candidate[key]
            same_shape = expected.shape == actual.shape
            finite = np.isfinite(expected).all() and np.isfinite(actual).all()
            passed = bool(same_shape and finite and np.allclose(expected, actual, atol=args.atol, rtol=args.rtol))
            report["stages"][key] = {"passed": passed,
                "max_abs_error": float(np.max(np.abs(expected - actual))) if same_shape and finite else None}
    report["passed"] = all(stage["passed"] for stage in report["stages"].values())
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "report": str(path)}))
    return 0 if report["passed"] else 1


def main():
    """仅解析命令；--help/compare 不导入 PyTorch。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("export-reference", "export-candidate"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--sample", help="NPZ containing image/state/noise; candidate should reuse reference NPZ")
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--steps", type=int, choices=range(1, 51), default=16)
        if name == "export-candidate":
            command.add_argument("--deployment-config", required=True)
    command = commands.add_parser("compare")
    command.add_argument("--reference", required=True)
    command.add_argument("--candidate", required=True)
    command.add_argument("--report", required=True)
    command.add_argument("--atol", type=float, default=1e-5)
    command.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.command == "compare":
        if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
            parser.error("tolerances must be finite and nonnegative")
        return compare(args)
    export(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
