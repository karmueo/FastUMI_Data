"""离线加载关节 DP checkpoint，评估验证集并保存首批原始动作预测。"""

import argparse
import json
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import evaluate_policy


def main():
    """加载可信本地 checkpoint，输出验证指标与 NPZ 动作；不连接机器人。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    payload = torch.load(args.checkpoint, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    if cfg.training.get("tf32", False):
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    if cfg.task.get("action_layout") != "joint8":
        raise ValueError("Checkpoint must explicitly declare joint8 actions")
    cfg.task.dataset_path = str(args.dataset.resolve())
    # 加载 checkpoint 自带权重，避免再次访问预训练权重源。
    cfg.policy.obs_encoder.pretrained = False
    policy = hydra.utils.instantiate(cfg.policy)
    key = "ema_model" if cfg.training.use_ema else "model"
    policy.load_state_dict(payload["state_dicts"][key])
    policy.to(args.device).eval()
    dataset = hydra.utils.instantiate(cfg.task.dataset).get_validation_dataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    metrics = evaluate_policy(policy, loader, args.device, "joint8", sample=True, max_steps=args.max_steps)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(args.checkpoint.resolve()), "dataset": str(args.dataset.resolve()),
              "validation_windows": len(dataset), "max_steps": args.max_steps, "metrics": metrics}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    batch = next(iter(loader))
    obs = dict_apply(batch["obs"], lambda value: value.to(args.device))
    with torch.no_grad():
        torch.manual_seed(42)
        pred = policy.predict_action(obs)["action_pred"].cpu().numpy()
    np.savez_compressed(args.output / "predictions.npz", predicted_action=pred, target_action=batch["action"].numpy())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
