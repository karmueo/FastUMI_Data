"""读取 checkpoint 的受限处理器协议，精确复现 LeRobot MEAN_STD/MIN_MAX。"""

import json
from pathlib import Path

import numpy as np

IMAGE = "observation.images.cam0"
STATE = "observation.state"
NORM_MAP = {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}
FEATURES = {IMAGE: {"type": "VISUAL", "shape": [3, 224, 224]},
            STATE: {"type": "STATE", "shape": [1]},
            "action": {"type": "ACTION", "shape": [10]}}


def read_processor_spec(checkpoint, inverse=False):
    """只接受此次模型保存的确定处理链，禁止任意类导入或外部统计路径。"""
    root = Path(checkpoint)
    name = "policy_postprocessor" if inverse else "policy_preprocessor"
    raw = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
    steps = raw.get("steps", [])
    expected = (["unnormalizer_processor", "device_processor"] if inverse else
                ["rename_observations_processor", "to_batch_processor", "device_processor", "normalizer_processor"])
    if raw.get("name") != name or [step.get("registry_name") for step in steps] != expected:
        raise ValueError(f"Unsupported {name} pipeline")
    if not inverse and (steps[0].get("config") != {"rename_map": {}} or steps[1].get("config") != {}):
        raise ValueError("Unsupported observation renaming/batching")
    device = steps[1 if inverse else 2].get("config", {})
    if set(device) != {"device", "float_dtype"} or device["float_dtype"] is not None:
        raise ValueError("Unsupported processor dtype conversion")
    entry = steps[0 if inverse else 3]
    cfg = entry.get("config", {})
    expected_features = {"action": FEATURES["action"]} if inverse else FEATURES
    if (cfg.get("norm_map") != NORM_MAP or cfg.get("features") != expected_features
            or set(cfg) != {"eps", "features", "norm_map"}
            or cfg["eps"] != 1e-8):
        raise ValueError("Unsupported normalization contract")
    filename = entry.get("state_file", "")
    if not filename or Path(filename).name != filename or not filename.endswith(".safetensors"):
        raise ValueError("Processor state_file must be a local safetensors filename")
    return root / filename, float(cfg["eps"])


def validate_stats(stats, inverse=False):
    """确认广播形状与数值，防止错误键悄悄广播为另一种特征。"""
    required = {"action.min": (10,), "action.max": (10,)} if inverse else {
        f"{IMAGE}.mean": (3, 1, 1), f"{IMAGE}.std": (3, 1, 1),
        f"{STATE}.min": (1,), f"{STATE}.max": (1,)}
    for key, shape in required.items():
        value = np.asarray(stats[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid normalization statistic {key}")
    prefix = "action" if inverse else STATE
    if np.any(stats[f"{prefix}.max"] < stats[f"{prefix}.min"]):
        raise ValueError("Normalization max is smaller than min")
    if not inverse and np.any(stats[f"{IMAGE}.std"] < 0):
        raise ValueError("Image standard deviation is negative")


def normalize_numpy(observations, stats, eps=1e-8):
    """无 GPU 的参考公式；不截断超出训练范围的有效物理状态。"""
    validate_stats(stats)
    denominator = stats[f"{STATE}.max"] - stats[f"{STATE}.min"]
    denominator = np.where(denominator == 0, eps, denominator)
    return {IMAGE: (observations[IMAGE] - stats[f"{IMAGE}.mean"]) / (stats[f"{IMAGE}.std"] + eps),
            STATE: 2 * (observations[STATE] - stats[f"{STATE}.min"]) / denominator - 1}


def unnormalize_numpy(actions, stats, eps=1e-8):
    """动作反归一化；仅零跨度替换 eps，与上游公式一致。"""
    validate_stats(stats, inverse=True)
    denominator = stats["action.max"] - stats["action.min"]
    denominator = np.where(denominator == 0, eps, denominator)
    return (actions + 1) / 2 * denominator + stats["action.min"]


class TorchProcessors:
    """在目标设备执行处理器数值运算，不安装通用 LeRobot pipeline。"""

    def __init__(self, checkpoint, device):
        import torch
        from safetensors.numpy import load_file

        self.torch = torch
        pre, self.pre_eps = read_processor_spec(checkpoint)
        post, self.post_eps = read_processor_spec(checkpoint, inverse=True)
        pre_stats, post_stats = load_file(str(pre)), load_file(str(post))
        validate_stats(pre_stats)
        validate_stats(post_stats, inverse=True)
        self.pre = {key: torch.as_tensor(value.copy(), device=device, dtype=torch.float32)
                    for key, value in pre_stats.items()}
        self.post = {key: torch.as_tensor(value.copy(), device=device, dtype=torch.float32)
                     for key, value in post_stats.items()}

    def normalize(self, batch):
        """保持与上游相同的运算顺序和 FP32 类型。"""
        denominator = self.pre[f"{STATE}.max"] - self.pre[f"{STATE}.min"]
        denominator = self.torch.where(denominator == 0, self.pre_eps, denominator)
        return {IMAGE: (batch[IMAGE] - self.pre[f"{IMAGE}.mean"]) / (self.pre[f"{IMAGE}.std"] + self.pre_eps),
                STATE: 2 * (batch[STATE] - self.pre[f"{STATE}.min"]) / denominator - 1}

    def unnormalize(self, actions):
        """还原八个10维动作，包括连续物理夹爪开度。"""
        denominator = self.post["action.max"] - self.post["action.min"]
        denominator = self.torch.where(denominator == 0, self.post_eps, denominator)
        return (actions + 1) / 2 * denominator + self.post["action.min"]
