"""不依赖训练框架的严格配置解析，只接受已验证的 jingbao 网络契约。"""

import json
from pathlib import Path
from types import SimpleNamespace

SUPPORTED = {
    'type': 'diffusion',
    'n_obs_steps': 2,
    'input_features': {'observation.images.cam0': {'type': 'VISUAL', 'shape': [3, 224, 224]}, 'observation.state': {'type': 'STATE', 'shape': [1]}},
    'output_features': {'action': {'type': 'ACTION', 'shape': [10]}},
    'horizon': 16,
    'n_action_steps': 8,
    'normalization_mapping': {'VISUAL': 'MEAN_STD', 'STATE': 'MIN_MAX', 'ACTION': 'MIN_MAX'},
    'vision_backbone': 'resnet18',
    'resize_shape': [236, 236],
    'crop_shape': [224, 224],
    'crop_is_random': True,
    'pretrained_backbone_weights': None,
    'use_group_norm': True,
    'spatial_softmax_num_keypoints': 32,
    'use_separate_rgb_encoder_per_camera': False,
    'down_dims': [256, 512, 1024],
    'kernel_size': 5,
    'n_groups': 8,
    'diffusion_step_embed_dim': 128,
    'use_film_scale_modulation': True,
    'noise_scheduler_type': 'DDIM',
    'num_train_timesteps': 50,
    'beta_schedule': 'squaredcos_cap_v2',
    'beta_start': 0.0001,
    'beta_end': 0.02,
    'prediction_type': 'epsilon',
    'clip_sample': True,
    'clip_sample_range': 1.0,
    'compile_model': False,
}


class DiffusionConfig(SimpleNamespace):
    """为移植网络提供与上游一致的只读约定属性。"""

    @classmethod
    def from_file(cls, path):
        """拒绝未知架构，避免仅凭同为10维动作误加载其他模型。"""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, expected in SUPPORTED.items():
            if raw.get(key) != expected:
                raise ValueError(f"Unsupported model configuration: {key}")
        steps = raw.get("num_inference_steps")
        validate_steps(steps)
        if raw.get("use_amp", False) or raw.get("use_peft", False):
            raise ValueError("AMP/PEFT checkpoint configuration is not supported")
        result = cls(**{key: raw[key] for key in SUPPORTED}, num_inference_steps=steps)
        result.image_features = {
            "observation.images.cam0": SimpleNamespace(shape=(3, 224, 224))}
        result.robot_state_feature = SimpleNamespace(shape=(1,))
        result.action_feature = SimpleNamespace(shape=(10,))
        result.env_state_feature = None
        return result


def validate_steps(value):
    """去噪步数只接受1～50的整数，排除布尔值。"""
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 50:
        raise ValueError("num_inference_steps must be an integer in [1, 50]")
    return value
