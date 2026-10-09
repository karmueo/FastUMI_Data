"""严格恢复 RM75 controller 并复现训练预处理；无需训练数据。"""

from pathlib import Path
import sys
import time

import numpy as np

from dexgraspvla_infer.core import denormalize_actions, normalization_bounds, normalize_state


def preprocess_rgbm(bgr, mask):
    """BGR [0,255] 与二值 mask -> float32 [1,1,4,392,518]。"""
    import torch
    import torch.nn.functional as functional

    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected uint8 BGR image")
    if mask.dtype != np.uint8 or mask.shape != bgr.shape[:2] or not np.isin(mask, [0, 255]).all():
        raise ValueError("expected same-frame uint8 0/255 mask")
    height = max(14, round(bgr.shape[0] / bgr.shape[1] * 518 / 14) * 14)
    rgb = torch.from_numpy(bgr.copy()).permute(2, 0, 1).float()[None] / 255
    target = torch.from_numpy(mask.copy()).float()[None, None]
    rgb = functional.interpolate(rgb, (height, 518), mode="bilinear", align_corners=False)
    target = (functional.interpolate(target, (height, 518), mode="nearest") > 200).float()
    combined = torch.cat((rgb, target), dim=1)
    if combined.shape[-2:] != (392, 518):
        combined = functional.interpolate(combined, (392, 518), mode="bilinear", align_corners=False)
    return combined[:, None]


class ModelRuntime:
    """模型只在 GPU worker 中调用；去噪参数在每次推理开始时冻结。"""

    def __init__(self, checkpoint, model_root, asset_root, urdf, device="cuda:0"):
        import torch
        import hydra
        from omegaconf import OmegaConf, open_dict

        self.torch = torch
        # Leave CPU capacity for ROS image/feedback callbacks on the 12-core Orin.
        torch.set_num_threads(2)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run on the GPU-accessible host")
        model_root, asset_root = Path(model_root), Path(asset_root)
        if not (model_root / "controller").is_dir():
            raise FileNotFoundError(f"missing DexGraspVLA source: {model_root}")
        sys.path.insert(0, str(model_root))
        # User-selected local workspace checkpoint contains OmegaConf metadata.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = payload["cfg"]
        if not OmegaConf.is_config(cfg):
            cfg = OmegaConf.create(cfg)
        shape = OmegaConf.to_container(cfg.shape_meta, resolve=True)
        if (shape["obs"]["rgbm"]["shape"] != [4, 392, 518]
                or shape["obs"]["right_state"]["shape"] != [8]
                or shape["action"] != {"shape": [8], "horizon": 64}
                or any(v["horizon"] != 1 for v in shape["obs"].values())
                or cfg.training.use_ema):
            raise ValueError("checkpoint is not the expected single-frame RM75 controller")
        policy_cfg = OmegaConf.create(OmegaConf.to_container(cfg.policy, resolve=True))
        with open_dict(policy_cfg):
            policy_cfg.obs_encoder.model_config.head.source_dir = str(asset_root / "third_party/dinov2")
            policy_cfg.obs_encoder.model_config.head.local_weights_path = str(asset_root / "weights/dinov2/dinov2_vitb14_pretrain.pth")
            policy_cfg.start_ckpt_path = None
        self.policy = hydra.utils.instantiate(policy_cfg)
        self.policy.load_state_dict(payload["state_dicts"]["model"], strict=True)
        self.policy.to(self.device).eval()
        self.lower, self.upper = normalization_bounds(urdf)
        self.max_steps = min(50, int(self.policy.noise_scheduler.config.num_train_timesteps))
        self.last_diagnostic = {}
        self.loaded_epoch = payload.get("pickles", {}).get("epoch")

    def predict(self, observation, steps=16):
        """返回物理单位的 [64,8]；不改变任何时间戳。"""
        self.torch.set_num_threads(2)
        if not 1 <= int(steps) <= self.max_steps:
            raise ValueError(f"inference steps must be in [1,{self.max_steps}]")
        self.policy.num_inference_steps = int(steps)
        started = time.monotonic()
        state = normalize_state(observation.state, self.lower, self.upper)
        inputs = {
            "rgbm": preprocess_rgbm(observation.bgr, observation.mask).to(self.device),
            "right_state": self.torch.from_numpy(state)[None, None].to(self.device),
        }
        prepared = time.monotonic()
        with self.torch.inference_mode():
            actions = self.policy.predict_action(inputs).cpu().numpy()[0]
        self.last_diagnostic = {"steps": int(steps), "torch_threads": self.torch.get_num_threads(),
                                "prepare_s": prepared - started, "network_s": time.monotonic() - prepared}
        return denormalize_actions(actions, self.lower, self.upper)

    def warmup(self, observation, steps=16):
        """用真实观测预热网络，不将预热动作发布给执行器。"""
        self.torch.set_num_threads(2)
        self.policy.num_inference_steps = int(steps)
        started = time.monotonic()
        inputs = {
            "rgbm": preprocess_rgbm(observation.bgr, observation.mask).to(self.device),
            "right_state": self.torch.from_numpy(normalize_state(
                observation.state, self.lower, self.upper))[None, None].to(self.device),
        }
        prepared = time.monotonic()
        with self.torch.inference_mode():
            result = self.policy.predict_action(inputs).cpu().numpy()
        self.last_diagnostic = {"steps": int(steps), "torch_threads": self.torch.get_num_threads(),
                                "prepare_s": prepared - started, "network_s": time.monotonic() - prepared}
        if result.shape != (1, 64, 8) or not np.isfinite(result).all():
            raise ValueError("invalid warmup output")
