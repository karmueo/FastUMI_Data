"""Check deterministic inputs and numeric failure rules for the ONNX bridge."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from export_onnx import OBS_KEYS, select_weight_key
from verify_onnx import (compare_arrays, initial_noise_for_reference,
                         unnormalize_action, validate_observations)


def test_ema_choice_follows_training_config_and_requires_weights():
    """A checkpoint marked for EMA must not silently use raw model weights."""
    cfg = SimpleNamespace(training=SimpleNamespace(use_ema=True))
    payload = {"state_dicts": {"model": {}}}
    with pytest.raises(ValueError, match="ema_model"):
        select_weight_key(cfg, payload)
    payload["state_dicts"]["ema_model"] = {"weight": torch.ones(1)}
    assert select_weight_key(cfg, payload) == "ema_model"
    cfg.training.use_ema = False
    assert select_weight_key(cfg, payload) == "model"


def test_observation_contract_rejects_wrong_shape_and_rgb_range():
    """The deployed encoder accepts only ordered, finite FP32 training inputs."""
    shapes = {name: [1, 2, 1] for name in OBS_KEYS}
    shapes["camera0_rgb"] = [1, 2, 3, 4, 4]
    obs = {name: torch.zeros(shapes[name]) for name in OBS_KEYS}
    validate_observations(obs, shapes)
    obs["robot0_eef_pos"] = torch.zeros(1, 3, 1)
    with pytest.raises(ValueError, match="robot0_eef_pos"):
        validate_observations(obs, shapes)
    obs["robot0_eef_pos"] = torch.zeros(shapes["robot0_eef_pos"])
    obs["camera0_rgb"][0, 0, 0, 0, 0] = 1.01
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        validate_observations(obs, shapes)


def test_captured_noise_is_the_next_policy_draw():
    """Reference inference consumes precisely the noise later reused by ONNX."""
    policy = SimpleNamespace(action_horizon=16, action_dim=10,
                             dtype=torch.float32, device=torch.device("cpu"))
    obs = {"camera0_rgb": torch.zeros(1, 2, 3, 4, 4)}
    noise = initial_noise_for_reference(policy, obs, seed=42)
    assert torch.equal(noise, torch.randn(1, 16, 10))
    assert torch.equal(noise, initial_noise_for_reference(policy, obs, seed=42))


def test_action_coefficients_match_normalizer_inverse():
    """Manifest coefficients reconstruct physical action values exactly."""
    normalized = np.array([[[0.5, -0.25]]], dtype=np.float32)
    actual = unnormalize_action(normalized, {"scale": [2.0, 4.0],
                                                  "offset": [0.1, -0.5]})
    np.testing.assert_allclose(actual, [[[0.2, 0.0625]]], rtol=0, atol=1e-7)
    with pytest.raises(ValueError, match="coefficients"):
        unnormalize_action(normalized, {"scale": [0.0, 4.0], "offset": [0, 0]})


def test_comparison_rejects_nonfinite_shape_and_above_threshold():
    """A bad conversion is reported as failure rather than averaged away."""
    expected = np.array([0.0, 1.0], dtype=np.float32)
    assert compare_arrays(expected.copy(), expected, 1e-4, 1e-3)["passed"]
    assert not compare_arrays(np.array([0.01, 1.0]), expected, 1e-4, 1e-3)["passed"]
    assert compare_arrays(np.array([np.nan, 1.0]), expected, 1e-4, 1e-3)["reason"] == "non_finite"
    assert compare_arrays(np.zeros(3), expected, 1e-4, 1e-3)["reason"] == "shape_mismatch"
