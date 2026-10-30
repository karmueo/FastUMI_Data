"""目标机上的 FP32 PyTorch 或 TensorRT 推理；模型依赖仅在引擎创建时加载。"""

from pathlib import Path

import numpy as np

from .config import IMAGE_KEY, RAW_DIM, STATE_KEY, DiffusionConfig, validate_backend
from .core import ACTION_STEPS, decode_actions, validate_sequence
from .deployment import validate_checkpoint, validate_export


class PolicyEngine:
    """单线程使用的无队列模型；ROS 层负责提供完整两帧观测窗口。"""

    def __init__(self, checkpoint, profile, device="cuda:0", noise_scheduler="", num_inference_steps=0,
                 processors=(), backend="pytorch", export_dir=""):
        """校验指纹后构建网络：PyTorch 严格加载 checkpoint 权重，TensorRT 加载模型包 engine。

        Args:
            checkpoint: LeRobot ``pretrained_model`` 目录；TensorRT 后端不使用。
            profile: :func:`deployment.load_profile` 返回的部署档。
            device: 推理设备，如 ``cuda:0``；TensorRT 须为构建 engine 的同型号 GPU。
            noise_scheduler: 空字符串沿用训练 DDPM，或 ``DDPM``/``DDIM``。
            num_inference_steps: 0 表示使用 checkpoint 默认（DDPM 100 步）。
            processors: 可选后处理扩展，``process(sequence, context)``。
            backend: ``pytorch``、``tensorrt_fp32`` 或 ``tensorrt_fp16``。
            export_dir: TensorRT 后端使用的 ``lerobot-export`` 模型包目录。

        Raises:
            ValueError/RuntimeError: 指纹、配置、权重、engine 或 CUDA 不可用。
        """
        import torch

        from .processors import TorchProcessors

        precision = validate_backend(backend)
        self.backend = backend
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in the selected inference environment")
        if precision is None:
            root = Path(checkpoint)
            validate_checkpoint(root, profile)
        else:
            if not export_dir:
                raise ValueError("TensorRT backends require export_dir")
            manifest = validate_export(export_dir, profile, precision)
            root = Path(export_dir) / "assets"  # 与 checkpoint 指纹一致的配置及处理器统计量。
        self.cfg = DiffusionConfig.from_file(root / "config.json", noise_scheduler or None,
                                             num_inference_steps or None,
                                             profile["contract"].get("pose_frame", "current_tcp"))
        if precision is None:
            self.policy = self._load_torch_policy(root)
        else:
            from .tensorrt_policy import TensorRTPolicy, TensorRTSession

            session = TensorRTSession(export_dir, manifest, self.device, precision)
            self.policy = TensorRTPolicy(self.cfg, manifest, session, self.device)
        self.processors = TorchProcessors(root, self.device, self.cfg.pose_frame)
        self.postprocessors = tuple(processors)

    def _load_torch_policy(self, root):
        """构建 PyTorch 网络并严格加载 FP32 权重。"""
        import torch
        from safetensors.torch import load_file

        from .network import InferencePolicy

        # 保持 PyTorch 默认精度设置（卷积允许 TF32、matmul 不允许），与 LeRobot 训练/推理一致；
        # 全 FP32 卷积在 RTX 5090 上单步 U-Net 约 11 ms，TF32 约 3.3 ms。
        policy = InferencePolicy(self.cfg)
        weights = load_file(str(root / "model.safetensors"), device="cpu")
        # BatchNorm 的 num_batches_tracked 计数器为 int64，其余必须是 FP32。
        if any(value.dtype != (torch.int64 if name.endswith(".num_batches_tracked") else torch.float32)
               for name, value in weights.items()):
            raise ValueError("Expected FP32 checkpoint tensors")
        policy.load_state_dict(weights, strict=True)
        return policy.to(device=self.device, dtype=torch.float32).eval()

    @property
    def description(self):
        """日志用的后端、调度器与去噪步数描述。"""
        return f"{self.backend}:{self.cfg.noise_scheduler_type}x{self.cfg.num_inference_steps}"

    def predict_raw(self, observations, noise=None, return_debug=False):
        """执行一次推理，返回 CPU ``[1,8,7]`` 原始布局动作；新模型相对于回合起点。

        Args:
            observations: :func:`core.build_observations` 的输出。
            noise: 可选初始噪声 ``[1,16,D]``，D 按部署契约为 10 或 7。
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
            if noise.shape != (1, self.cfg.horizon, self.cfg.model_action_dim) or not torch.isfinite(noise).all():
                raise ValueError("Invalid initial noise shape or values")
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
        sequence = decode_actions(self.predict_raw(observations)[0], context, self.cfg.pose_frame)
        for processor in self.postprocessors:
            sequence = validate_sequence(processor.process(sequence, context))
        return sequence
