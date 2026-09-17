"""加载 RM75 Link7 扩散策略 checkpoint 并恢复训练时使用的模型权重。"""

from pathlib import Path

import dill
import hydra
import torch


def load_policy(checkpoint: Path, device: str):
    """从可信 checkpoint 加载 EMA/model 权重，并返回策略与训练配置。

    Args:
        checkpoint: 本地训练生成的 Link7 checkpoint 文件。
        device: PyTorch 设备名，如 ``cuda:0`` 或 ``cpu``。

    Returns:
        已置为 eval 模式的策略及其训练配置。

    Raises:
        ValueError: checkpoint 类型或目标设备与当前机器不匹配。
        RuntimeError: 权重、模型结构或依赖版本不兼容。
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint must name an existing file: {checkpoint}")
    try:
        target_device = torch.device(device)
    except RuntimeError as error:
        raise ValueError(f"Invalid device: {device}") from error
    if target_device.type == "cuda":
        index = target_device.index or 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device is unavailable: {device}")
    if target_device.type not in ("cuda", "cpu"):
        raise ValueError(f"Unsupported inference device: {device}")

    # 训练产物使用 dill 序列化；只接受部署人员提供的可信 checkpoint。
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False)
    cfg = payload["cfg"]
    if list(cfg.shape_meta.action.shape) != [10] or cfg.task.get("contract", {}).get("end_frame") != "Link7":
        raise ValueError("Expected a Link7 pose10 checkpoint")
    if cfg.training.get("tf32", False):
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    # checkpoint 含完整视觉权重，实例化时不访问外部预训练模型服务。
    cfg.policy.obs_encoder.pretrained = False
    policy = hydra.utils.instantiate(cfg.policy)
    weight_key = "ema_model" if cfg.training.use_ema else "model"
    policy.load_state_dict(payload["state_dicts"][weight_key])
    policy.to(target_device).eval()
    return policy, cfg
