"""验证图像训练 workspace 的离线 runner 和固定 batch 动作指标逻辑。"""

from unittest.mock import Mock

import pytest
import torch

from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    _compute_action_mse_metrics,
    _instantiate_env_runner,
    _sample_training_action_metrics,
)


class _IdentityActionNormalizer:
    """为采样 helper 测试提供恒等动作反归一化参数。"""

    def __init__(self):
        # 动作缩放参数，覆盖单臂 10D 动作。
        scale = torch.ones(10)
        # 动作偏移参数，覆盖单臂 10D 动作。
        offset = torch.zeros(10)
        self.params_dict = {"scale": scale, "offset": offset}


def test_null_env_runner_is_supported(tmp_path):
    """env_runner 为 null 时不应触发 Hydra 实例化。"""
    assert _instantiate_env_runner(None, tmp_path) is None


def test_action_mse_uses_position_rotation_and_width_slices():
    """动作 MSE 应分别覆盖位置、rotation-6D 和夹爪分量。"""
    pred = torch.ones((2, 16, 10))
    target = torch.zeros_like(pred)

    metrics = _compute_action_mse_metrics("train", pred, target)

    assert set(metrics) == {
        "train_action_mse_error",
        "train_action_mse_error_pos",
        "train_action_mse_error_rot",
        "train_action_mse_error_width",
    }
    assert all(torch.equal(value, torch.tensor(1.0)) for value in metrics.values())


def test_action_mse_rejects_non_10d_per_robot_action():
    """每个机器人动作维度不是 10 时应报告实际维度。"""
    pred = torch.ones((2, 16, 9))

    with pytest.raises(ValueError, match="9"):
        _compute_action_mse_metrics("train", pred, torch.zeros_like(pred))


def test_fixed_batch_sampling_predicts_once():
    """一次固定 batch 采样事件只调用一次策略预测。"""
    batch = {
        "obs": {"state": torch.zeros((2, 2, 1))},
        "action": torch.ones((2, 16, 10)),
    }
    policy = Mock()
    policy.normalizer = {"action": _IdentityActionNormalizer()}
    policy.predict_action.return_value = {"action_pred": torch.zeros_like(batch["action"])}

    metrics = _sample_training_action_metrics(policy, batch)

    assert policy.predict_action.call_count == 1
    assert set(metrics) == {
        "train_action_mse_error",
        "train_action_mse_error_pos",
        "train_action_mse_error_rot",
        "train_action_mse_error_width",
    }
    assert all(value.ndim == 0 and torch.isfinite(value) for value in metrics.values())
