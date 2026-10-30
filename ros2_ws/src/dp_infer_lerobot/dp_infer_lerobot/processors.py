"""读取 checkpoint 的受限处理器协议，复现 LeRobot 归一化与 SE(3) 相对位姿转换。

旧契约预处理：原始 state 历史 (B, 2, 7) → 相对当前帧 ``inv(T_cur) @ T``（xyz + rot6d，当前帧写成精确单位位姿）
→ MIN_MAX 归一化；图像 MEAN_STD 归一化。后处理：动作 MIN_MAX 反归一化 → ``T_cur @ T_rel``
还原为原始布局 ``[xyz, 轴角, 夹爪]``。位姿函数逐字摘自 LeRobot fork 6168a52
``lerobot/processor/relative_pose_processor.py``（Apache-2.0），保证与上游 FP32 结果一致。
新契约输入和输出保持回合起点坐标，仅执行归一化和反归一化。
"""

import json
from pathlib import Path

import numpy as np

from .config import IMAGE_KEY, MODEL_DIM, STATE_KEY

NORM_MAP = {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}  # 训练归一化方式。
RAW_FEATURES = {STATE_KEY: {"type": "STATE", "shape": [7]},
                IMAGE_KEY: {"type": "VISUAL", "shape": [3, 224, 224]},
                "action": {"type": "ACTION", "shape": [7]}}  # 处理链 JSON 中记录的原始特征。
POSE_STARTS = [0]  # state/action 中唯一位姿组的起始下标。
PRE_STEPS = ["rename_observations_processor", "to_batch_processor", "device_processor",
             "relative_pose_processor", "normalizer_processor"]  # 预处理链必须严格按此顺序。
POST_STEPS = ["unnormalizer_processor", "absolute_pose_processor", "device_processor"]  # 后处理链顺序。


def read_processor_spec(checkpoint, inverse=False, pose_frame=None):
    """校验处理链 JSON，返回 ``(统计量 safetensors 路径, eps)``。

    只接受本模型保存的确定处理链，禁止任意类导入或外部统计路径。

    Raises:
        ValueError: 步骤、配置或统计文件名与契约不符。
    """
    root = Path(checkpoint)
    name = "policy_postprocessor" if inverse else "policy_preprocessor"
    raw = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
    steps = raw.get("steps", [])
    names = [step.get("registry_name") for step in steps]
    legacy = POST_STEPS if inverse else PRE_STEPS
    fixed = ["unnormalizer_processor", "device_processor"] if inverse else [
        "rename_observations_processor", "to_batch_processor", "device_processor", "normalizer_processor"]
    if pose_frame is None:
        pose_frame = "current_tcp" if names == legacy else "episode_start_tcp"
    if pose_frame not in ("current_tcp", "episode_start_tcp") or raw.get("name") != name or names != (
            legacy if pose_frame == "current_tcp" else fixed):
        raise ValueError(f"Unsupported {name} pipeline")
    configs = [step.get("config", {}) for step in steps]
    if inverse:
        pose_cfg, device_cfg, entry = (configs[1] if pose_frame == "current_tcp" else None), configs[-1], steps[0]
        expected_pose = {"enabled": True, "state_pose_starts": POSE_STARTS, "action_pose_starts": None}
        expected_features = {"action": RAW_FEATURES["action"]}
    else:
        if configs[0] != {"rename_map": {}} or configs[1] != {}:
            raise ValueError("Unsupported observation renaming/batching")
        pose_cfg, device_cfg, entry = (configs[3] if pose_frame == "current_tcp" else None), configs[2], steps[-1]
        expected_pose = {"enabled": True, "state_pose_starts": POSE_STARTS,
                         "action_pose_starts": None, "n_obs_steps": 2}
        expected_features = RAW_FEATURES
    if pose_frame == "current_tcp" and pose_cfg != expected_pose:
        raise ValueError("Unsupported relative pose configuration")
    if set(device_cfg) != {"device", "float_dtype"} or device_cfg["float_dtype"] is not None:
        raise ValueError("Unsupported processor dtype conversion")
    cfg = entry.get("config", {})
    if (set(cfg) != {"eps", "features", "norm_map"} or cfg["norm_map"] != NORM_MAP
            or cfg["features"] != expected_features or cfg["eps"] != 1e-8):
        raise ValueError("Unsupported normalization contract")
    filename = entry.get("state_file", "")
    if not filename or Path(filename).name != filename or not filename.endswith(".safetensors"):
        raise ValueError("Processor state_file must be a local safetensors filename")
    return root / filename, float(cfg["eps"])


def validate_stats(stats, inverse=False, model_dim=MODEL_DIM):
    """确认所需统计量的形状与数值，防止错误键被悄悄广播。

    旧相对位姿空间为 10 维，新回合起点空间为 7 维；部分维度 min==max 属正常。
    """
    required = {"action.min": (model_dim,), "action.max": (model_dim,)} if inverse else {
        f"{IMAGE_KEY}.mean": (3, 1, 1), f"{IMAGE_KEY}.std": (3, 1, 1),
        f"{STATE_KEY}.min": (model_dim,), f"{STATE_KEY}.max": (model_dim,)}
    for key, shape in required.items():
        value = np.asarray(stats.get(key))
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid normalization statistic {key}")
    prefix = "action" if inverse else STATE_KEY
    if np.any(stats[f"{prefix}.max"] < stats[f"{prefix}.min"]):
        raise ValueError("Normalization max is smaller than min")
    if not inverse and np.any(stats[f"{IMAGE_KEY}.std"] < 0):
        raise ValueError("Image standard deviation is negative")
    return {key: np.asarray(stats[key], dtype=np.float32) for key in required}


class TorchProcessors:
    """在目标设备上执行处理器数值运算，不安装通用 LeRobot pipeline。"""

    def __init__(self, checkpoint, device, pose_frame="current_tcp"):
        """读取并校验两份统计量，复制到 ``device`` 上的 FP32 张量。"""
        import torch
        from safetensors.numpy import load_file

        from . import relative_pose

        self.pose_frame = pose_frame
        model_dim = MODEL_DIM if pose_frame == "current_tcp" else 7
        self.torch = torch
        self.pose = relative_pose  # 惰性导入的 torch 位姿函数模块。
        pre, self.pre_eps = read_processor_spec(checkpoint, pose_frame=pose_frame)
        post, self.post_eps = read_processor_spec(checkpoint, inverse=True, pose_frame=pose_frame)
        self.pre = {key: torch.as_tensor(value.copy(), device=device)
                    for key, value in validate_stats(load_file(str(pre)), model_dim=model_dim).items()}
        self.post = {key: torch.as_tensor(value.copy(), device=device)
                     for key, value in validate_stats(load_file(str(post)), inverse=True, model_dim=model_dim).items()}

    def _min_max(self, stats, prefix, eps):
        """返回 ``(min, 分母)``；零跨度分母替换为 eps，与上游一致。"""
        minimum = stats[f"{prefix}.min"]
        denominator = stats[f"{prefix}.max"] - minimum
        eps = self.torch.tensor(eps, device=denominator.device, dtype=denominator.dtype)
        return minimum, self.torch.where(denominator == 0, eps, denominator)

    def preprocess(self, image, state):
        """原始观测 → 归一化模型输入。

        Args:
            image: (B, 2, 3, 224, 224) RGB [0,1]。
            state: (B, 2, 7) 原始布局历史，最后一帧为当前帧。

        Returns:
            ``(batch, reference)``：归一化输入字典，以及当前帧原始 state (B, 7) 供后处理还原。
        """
        reference = state[:, -1]
        relative = state
        if self.pose_frame == "current_tcp":
            relative = self.pose.to_relative_pose(state, reference, POSE_STARTS, POSE_STARTS)
            relative = self.pose.set_frame_to_identity(relative, -1, POSE_STARTS)
        minimum, denominator = self._min_max(self.pre, STATE_KEY, self.pre_eps)
        normalized_state = 2 * (relative - minimum) / denominator - 1
        normalized_image = ((image - self.pre[f"{IMAGE_KEY}.mean"])
                            / (self.pre[f"{IMAGE_KEY}.std"] + self.pre_eps))
        return {IMAGE_KEY: normalized_image, STATE_KEY: normalized_state}, reference

    def postprocess(self, actions, reference):
        """归一化模型动作反归一化；旧契约再还原当前帧坐标，新契约保留回合起点坐标。"""
        minimum, denominator = self._min_max(self.post, "action", self.post_eps)
        actions = (actions + 1) / 2 * denominator + minimum
        if self.pose_frame == "current_tcp":
            return self.pose.to_absolute_pose(actions, reference, POSE_STARTS, POSE_STARTS)
        return actions
