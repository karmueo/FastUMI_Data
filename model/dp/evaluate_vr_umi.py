"""离线评估 Link7 末端策略 checkpoint，并保存可解码的 10D 预测和误差。"""

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
from umi.common.pose_util import pose10d_to_mat


def load_policy(checkpoint, device):
    """加载本地可信 checkpoint 的 EMA 权重与契约，禁止关节 checkpoint 混用。"""
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    if list(cfg.shape_meta.action.shape) != [10] or cfg.task.get("contract", {}).get("end_frame") != "Link7":
        raise ValueError("Expected a Link7 pose10 checkpoint")
    if cfg.training.get("tf32", False):
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    cfg.policy.obs_encoder.pretrained = False
    policy = hydra.utils.instantiate(cfg.policy)
    key = "ema_model" if cfg.training.use_ema else "model"
    policy.load_state_dict(payload["state_dicts"][key])
    policy.to(device).eval()
    return policy, cfg


def decode_pose_actions(actions):
    """将任意批次维度的 [...,10] 动作解码为 [...,4,4]；适配原工具的二维输入。"""
    values = np.asarray(actions)
    if values.shape[-1] != 10 or not np.isfinite(values).all():
        raise ValueError("Expected finite 10D pose actions")
    matrices = pose10d_to_mat(values[..., :9].reshape(-1, 9))
    return matrices.reshape(values.shape[:-1] + (4, 4))


def main():
    """评估 checkpoint 自带验证划分，输出完整指标和首批动作/旋转矩阵。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    policy, cfg = load_policy(args.checkpoint, args.device)
    cfg.task.dataset_path = str(args.dataset.resolve())
    dataset = hydra.utils.instantiate(cfg.task.dataset).get_validation_dataset()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    metrics = evaluate_policy(policy, loader, args.device, "pose10", sample=True, max_steps=args.max_steps)
    batch = next(iter(loader))
    with torch.no_grad():
        torch.manual_seed(42)
        pred = policy.predict_action(dict_apply(batch["obs"], lambda value: value.to(args.device)))["action_pred"].cpu().numpy()
    matrices = decode_pose_actions(pred)
    rotations = matrices[..., :3, :3]
    if pred.shape[1:] != (16, 10) or not np.isfinite(pred).all():
        raise ValueError("Invalid predicted action shape or values")
    if not np.allclose(rotations @ rotations.swapaxes(-1, -2), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
        raise ValueError("Invalid decoded rotations")
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "predictions.npz", predicted_action=pred,
                        target_action=batch["action"].numpy(), relative_transforms=matrices)
    report = {"checkpoint": str(args.checkpoint.resolve()), "validation_windows": len(dataset),
              "max_steps": args.max_steps, "prediction_shape": list(pred.shape),
              "rotation_decode_verified": True, "metrics": metrics}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
