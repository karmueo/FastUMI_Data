"""目标机上的 FP32 PyTorch 推理；模型依赖仅在引擎创建时加载。"""

from pathlib import Path

import numpy as np

from .config import DiffusionConfig, validate_steps
from .core import decode_actions, validate_sequence
from .deployment import validate_checkpoint


class PolicyEngine:
    """单线程使用的无队列模型；共享 ROS 层负责提供完整观测历史。"""

    def __init__(self, checkpoint, profile, device="cuda:0", num_inference_steps=16, processors=()):
        import torch
        from safetensors.torch import load_file

        from .network import InferencePolicy
        from .processors import TorchProcessors

        validate_checkpoint(checkpoint, profile)
        self.cfg = DiffusionConfig.from_file(Path(checkpoint) / "config.json")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in the selected inference environment")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.policy = InferencePolicy(self.cfg)
        weights = load_file(str(Path(checkpoint) / "model.safetensors"), device="cpu")
        if any(value.dtype != torch.float32 for value in weights.values()):
            raise ValueError("Expected FP32 checkpoint tensors")
        self.policy.load_state_dict(weights, strict=True)
        self.policy.to(device=self.device, dtype=torch.float32).eval()
        self.policy.diffusion.num_inference_steps = validate_steps(num_inference_steps)
        self.processors = TorchProcessors(checkpoint, self.device)
        self.postprocessors = tuple(processors)

    def predict_raw(self, observations, noise=None, num_inference_steps=None):
        """返回 CPU [1,8,10] 物理动作；可注入固定噪声进行离线数值对照。"""
        import torch

        if num_inference_steps is not None:
            self.policy.diffusion.num_inference_steps = validate_steps(num_inference_steps)
        expected = {"observation.images.cam0": (1, 2, 3, 224, 224), "observation.state": (1, 2, 1)}
        if set(observations) != set(expected):
            raise ValueError("Unexpected observation keys")
        batch = {}
        for key, shape in expected.items():
            value = np.asarray(observations[key], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid model input {key}")
            batch[key] = torch.as_tensor(value, device=self.device)
        if noise is not None:
            noise = torch.as_tensor(noise, device=self.device, dtype=torch.float32)
            if noise.shape != (1, 16, 10) or not torch.isfinite(noise).all():
                raise ValueError("Expected finite [1,16,10] initial noise")
        with torch.inference_mode():
            prediction = self.policy.predict_action_chunk(self.processors.normalize(batch), noise=noise)
            prediction = self.processors.unnormalize(prediction).cpu().numpy()
        if prediction.shape != (1, 8, 10) or not np.isfinite(prediction).all():
            raise ValueError("Expected finite [1,8,10] prediction")
        return prediction

    def predict(self, observations, context, num_inference_steps=None):
        """每次预测独立产生八个局部增量，转换为九点绝对目标。"""
        sequence = decode_actions(self.predict_raw(observations, num_inference_steps=num_inference_steps)[0], context)
        for processor in self.postprocessors:
            sequence = validate_sequence(processor.process(sequence, context))
        return sequence
