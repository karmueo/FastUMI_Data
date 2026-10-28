"""目标机上的 FP32 PyTorch 推理；模型依赖仅在引擎创建时加载。"""

from pathlib import Path

import numpy as np

from .config import IMAGE_KEY, MODEL_DIM, RAW_DIM, STATE_KEY, DiffusionConfig
from .core import ACTION_STEPS, decode_actions, validate_sequence
from .deployment import validate_checkpoint


class PolicyEngine:
    """单线程使用的无队列模型；ROS 层负责提供完整两帧观测窗口。"""

    def __init__(self, checkpoint, profile, device="cuda:0", noise_scheduler="", num_inference_steps=0,
                 processors=()):
        """校验指纹后构建网络并严格加载权重。

        Args:
            checkpoint: LeRobot ``pretrained_model`` 目录。
            profile: :func:`deployment.load_profile` 返回的部署档。
            device: 推理设备，如 ``cuda:0``。
            noise_scheduler: 空字符串沿用训练 DDPM，或 ``DDPM``/``DDIM``。
            num_inference_steps: 0 表示使用 checkpoint 默认（DDPM 100 步）。
            processors: 可选后处理扩展，``process(sequence, context)``。

        Raises:
            ValueError/RuntimeError: 指纹、配置、权重或 CUDA 不可用。
        """
        import torch
        from safetensors.torch import load_file

        from .network import InferencePolicy
        from .processors import TorchProcessors

        root = Path(checkpoint)
        validate_checkpoint(root, profile)
        self.cfg = DiffusionConfig.from_file(root / "config.json", noise_scheduler or None,
                                             num_inference_steps or None)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in the selected inference environment")
        # 保持 PyTorch 默认精度设置（卷积允许 TF32、matmul 不允许），与 LeRobot 训练/推理一致；
        # 全 FP32 卷积在 RTX 5090 上单步 U-Net 约 11 ms，TF32 约 3.3 ms。
        self.policy = InferencePolicy(self.cfg)
        weights = load_file(str(root / "model.safetensors"), device="cpu")
        # BatchNorm 的 num_batches_tracked 计数器为 int64，其余必须是 FP32。
        if any(value.dtype != (torch.int64 if name.endswith(".num_batches_tracked") else torch.float32)
               for name, value in weights.items()):
            raise ValueError("Expected FP32 checkpoint tensors")
        self.policy.load_state_dict(weights, strict=True)
        self.policy.to(device=self.device, dtype=torch.float32).eval()
        self.processors = TorchProcessors(root, self.device)
        self.postprocessors = tuple(processors)

    @property
    def description(self):
        """日志用的调度器与去噪步数描述。"""
        return f"{self.cfg.noise_scheduler_type}x{self.policy.diffusion.num_inference_steps}"

    def predict_raw(self, observations, noise=None, return_debug=False):
        """执行一次推理，返回 CPU ``[1,8,7]`` 原始布局绝对动作。

        Args:
            observations: :func:`core.build_observations` 的输出。
            noise: 可选初始噪声 ``[1,16,10]``，用于离线数值对照。
            return_debug: 为真时额外返回归一化输入和模型输出，供对照工具使用。
        """
        import torch

        expected = {IMAGE_KEY: (1, 2, 3, 224, 224), STATE_KEY: (1, 2, RAW_DIM)}
        if set(observations) != set(expected):
            raise ValueError("Unexpected observation keys")
        inputs = {}
        for key, shape in expected.items():
            value = np.asarray(observations[key], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid model input {key}")
            inputs[key] = torch.as_tensor(value, device=self.device)
        if noise is not None:
            noise = torch.as_tensor(np.asarray(noise, dtype=np.float32), device=self.device)
            if noise.shape != (1, self.cfg.horizon, MODEL_DIM) or not torch.isfinite(noise).all():
                raise ValueError("Expected finite [1,16,10] initial noise")
        with torch.inference_mode():
            batch, reference = self.processors.preprocess(inputs[IMAGE_KEY], inputs[STATE_KEY])
            model_actions = self.policy.predict_action_chunk(batch, noise=noise)
            prediction = self.processors.postprocess(model_actions, reference).cpu().numpy()
        if prediction.shape != (1, ACTION_STEPS, RAW_DIM) or not np.isfinite(prediction).all():
            raise ValueError("Expected finite [1,8,7] prediction")
        if return_debug:
            debug = {f"normalized.{key}": value.cpu().numpy() for key, value in batch.items()}
            debug["model_actions"] = model_actions.cpu().numpy()
            return prediction, debug
        return prediction

    def predict(self, observations, context):
        """推理并换算为 Link7 目标序列，再依次执行后处理扩展。"""
        sequence = decode_actions(self.predict_raw(observations)[0], context)
        for processor in self.postprocessors:
            sequence = validate_sequence(processor.process(sequence, context))
        return sequence
