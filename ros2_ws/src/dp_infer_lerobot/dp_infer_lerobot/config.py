"""不依赖训练框架的严格配置解析，只接受已验证的 dp_umi_v4 / dp_umi_v5 网络契约。

LeRobot fork 的 ``DiffusionConfig`` 在加载时会派生 ``crop_shape``（``resize_shape × crop_ratio``）
以及相对位姿下的模型维度。本模块逐项核对 ``config.json``，复现这些派生量，
拒绝任何未知架构，避免仅凭同为 7 维状态就误加载其他模型。
"""

import json
from pathlib import Path
from types import SimpleNamespace

IMAGE_KEY = "observation.images.front"  # 唯一相机输入键。
STATE_KEY = "observation.state"  # 原始布局 [x,y,z,rx,ry,rz,gripper]。
RAW_DIM = 7  # 原始 state/action 维度。
MODEL_DIM = 10  # 相对位姿后 [xyz, rot6d, gripper] 维度。
MAX_INFERENCE_STEPS = 100  # 各变体训练时间步数的上限；实际上限为所加载模型的 num_train_timesteps。
NOISE_SCHEDULERS = ("DDPM", "DDIM")  # 推理可选调度器；DDIM 复用 DDPM 训练的噪声表。
# 推理后端：PyTorch 读取 checkpoint；TensorRT 读取本机构建的模型包中对应精度的 engine。
BACKENDS = ("pytorch", "tensorrt_fp32", "tensorrt_fp16")


def validate_backend(value):
    """校验后端名并返回 TensorRT 精度（``fp32``/``fp16``），PyTorch 返回 ``None``。"""
    if value not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}")
    return value.removeprefix("tensorrt_") if value != "pytorch" else None

# config.json 中所有变体都必须逐字相等的字段。
COMMON = {
    "type": "diffusion",
    "n_obs_steps": 2,
    "input_features": {
        STATE_KEY: {"type": "STATE", "shape": [RAW_DIM]},
        IMAGE_KEY: {"type": "VISUAL", "shape": [3, 224, 224]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [RAW_DIM]}},
    "horizon": 16,
    "n_action_steps": 8,
    "use_relative_pose": True,
    "relative_pose_state_starts": [0],
    "relative_pose_action_starts": None,
    "normalization_mapping": {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"},
    "vision_backbone": "resnet18",
    "resize_shape": [224, 224],
    "crop_ratio": 0.95,
    "crop_is_random": True,
    "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
    "use_group_norm": False,
    "spatial_softmax_num_keypoints": 32,
    "use_separate_rgb_encoder_per_camera": True,
    "kernel_size": 5,
    "n_groups": 8,
    "diffusion_step_embed_dim": 128,
    "use_film_scale_modulation": True,
    "beta_schedule": "squaredcos_cap_v2",
    "beta_start": 0.0001,
    "beta_end": 0.02,
    "prediction_type": "epsilon",
    "clip_sample": True,
    "clip_sample_range": 1.0,
    "compile_model": False,
}
# 已验证的网络变体：U-Net 通道宽度与训练噪声表不同，编码器、输入输出与前后处理一致。
# 同一变体内三个字段必须同时匹配，避免拼出未经验证的组合。
VARIANTS = {
    "dp_umi_v4": {"down_dims": [512, 1024, 2048], "noise_scheduler_type": "DDPM", "num_train_timesteps": 100},
    "dp_umi_v5": {"down_dims": [256, 512, 1024], "noise_scheduler_type": "DDIM", "num_train_timesteps": 50},
}
SUPPORTED = dict(COMMON, **VARIANTS["dp_umi_v4"])  # dp_umi_v4 的完整受支持配置。


class DiffusionConfig(SimpleNamespace):
    """为移植网络提供与上游一致的只读约定属性。"""

    @classmethod
    def from_file(cls, path, noise_scheduler_type=None, num_inference_steps=None):
        """读取并校验 ``config.json``，返回可直接构建网络的配置。

        Args:
            path: checkpoint 目录下的 ``config.json``。
            noise_scheduler_type: ``None`` 沿用训练的 DDPM；``"DDIM"`` 用同一噪声表做确定性少步去噪。
            num_inference_steps: ``None`` 时使用 checkpoint 值，未设置则等于训练步数 100。

        Raises:
            ValueError: 架构、步数或调度器不受支持。
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, expected in COMMON.items():
            if raw.get(key) != expected:
                raise ValueError(f"Unsupported model configuration: {key}")
        variant = next((name for name, fields in VARIANTS.items()
                        if all(raw.get(key) == value for key, value in fields.items())), None)
        if variant is None:
            raise ValueError("Unsupported model configuration: down_dims/noise_scheduler_type/num_train_timesteps")
        if raw.get("use_amp", False) or raw.get("use_peft", False):
            raise ValueError("AMP/PEFT checkpoint configuration is not supported")
        scheduler = noise_scheduler_type or raw["noise_scheduler_type"]
        if scheduler not in NOISE_SCHEDULERS:
            raise ValueError(f"noise_scheduler_type must be one of {NOISE_SCHEDULERS}")
        steps = num_inference_steps if num_inference_steps is not None else raw.get("num_inference_steps")
        maximum = raw["num_train_timesteps"]
        steps = maximum if steps is None else validate_steps(steps, maximum)
        result = cls(**{key: raw[key] for key in (*COMMON, *VARIANTS[variant])})
        result.variant = variant
        result.noise_scheduler_type = scheduler
        result.num_inference_steps = steps
        # 上游 __post_init__：resize 后按比例中心裁剪（评估时），224×0.95 → 212。
        result.resize_shape = tuple(raw["resize_shape"])
        result.crop_shape = tuple(int(size * raw["crop_ratio"]) for size in raw["resize_shape"])
        # 权重全部来自 model.safetensors；构建时不下载 torchvision 预训练权重。
        result.pretrained_backbone_weights = None
        result.compile_mode = "reduce-overhead"
        result.gradient_checkpointing = False
        result.image_features = {IMAGE_KEY: SimpleNamespace(shape=(3, 224, 224))}
        result.env_state_feature = None
        result.model_state_dim = MODEL_DIM
        result.model_action_dim = MODEL_DIM
        return result


def validate_steps(value, maximum=MAX_INFERENCE_STEPS):
    """去噪步数只接受 1～``maximum`` 的整数（默认 100，加载模型后为其训练时间步数），排除布尔值。"""
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"num_inference_steps must be an integer in [1, {maximum}]")
    return value
