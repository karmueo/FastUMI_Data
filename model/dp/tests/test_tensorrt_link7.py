"""Check Link7 TensorRT provenance, DDIM replay, and physical error semantics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import DDIMScheduler

from tensorrt_link7 import (OBS_KEYS, latency_stats, load_engine_bundle,
                           load_onnx_bundle, metrics, physical_errors, sample_trajectory,
                           sha256_file, unnormalize_action)


def example_manifest():
    """Construct the fixed batch Link7 interface without large graph files."""
    shapes = {
        "camera0_rgb": [1, 2, 3, 224, 224],
        "robot0_eef_pos": [1, 2, 3],
        "robot0_eef_rot_axis_angle": [1, 2, 6],
        "robot0_gripper_width": [1, 2, 1],
        "robot0_eef_rot_axis_angle_wrt_start": [1, 2, 6],
    }
    return {
        "schema_version": 1, "batch_size": 1, "dtype": "float32",
        "observation_order": list(OBS_KEYS), "observation_shapes": shapes,
        "sample_shape": [1, 16, 10], "timestep_shape": [1],
        "global_cond_shape": [1, 1568], "num_inference_steps": 16,
        "scheduler": {"prediction_type": "epsilon"},
        "contract": {"end_frame": "Link7"}, "action_layout": "pose10",
        "weights": "ema_model", "checkpoint_sha256": "abc",
        "graphs": {},
    }


def test_onnx_bundle_rejects_changed_graph_and_checkpoint(tmp_path):
    """Source changes must stop conversion before any TensorRT build."""
    manifest = example_manifest()
    for graph in ("obs_encoder", "denoiser"):
        filename = graph + ".onnx"
        path = tmp_path / filename
        path.write_bytes(graph.encode())
        manifest["graphs"][graph] = {"file": filename, "sha256": sha256_file(path)}
    checkpoint = tmp_path / "latest.ckpt"
    checkpoint.write_bytes(b"weights")
    manifest["checkpoint_sha256"] = sha256_file(checkpoint)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert load_onnx_bundle(tmp_path, checkpoint)["weights"] == "ema_model"
    checkpoint.write_bytes(b"different")
    with pytest.raises(ValueError, match="Checkpoint"):
        load_onnx_bundle(tmp_path, checkpoint)
    (tmp_path / "denoiser.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="ONNX hash"):
        load_onnx_bundle(tmp_path)


def test_engine_bundle_rejects_wrong_source_and_engine_hash(tmp_path, monkeypatch):
    """Serialized plans cannot be used after their source or own bytes change."""
    import tensorrt as trt

    manifest = example_manifest()
    for graph in ("obs_encoder", "denoiser"):
        manifest["graphs"][graph] = {"sha256": graph}
        filename = ("denoiser.fp16_fallback_fp32.plan" if graph == "denoiser"
                    else "obs_encoder.fp16.plan")
        (tmp_path / filename).write_bytes(graph.encode())
    built = {
        "schema_version": 1, "source_checkpoint_sha256": "abc",
        "source_graphs": {graph: graph for graph in ("obs_encoder", "denoiser")},
        "tensorrt_version": trt.__version__, "gpu_name": "test-gpu",
        "compute_capability": [8, 7],
        "engines": {"fp16": {graph: {
            "file": ("denoiser.fp16_fallback_fp32.plan" if graph == "denoiser"
                     else "obs_encoder.fp16.plan"),
            "effective_precision": "fp32" if graph == "denoiser" else "fp16",
            "sha256": sha256_file(tmp_path / (
                "denoiser.fp16_fallback_fp32.plan" if graph == "denoiser"
                else "obs_encoder.fp16.plan"))}
            for graph in ("obs_encoder", "denoiser")}},
    }
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: SimpleNamespace(name="test-gpu", major=8, minor=7))
    (tmp_path / "manifest.json").write_text(json.dumps(built))
    assert len(load_engine_bundle(tmp_path, manifest, "fp16")) == 2
    (tmp_path / "denoiser.fp16_fallback_fp32.plan").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Engine hash"):
        load_engine_bundle(tmp_path, manifest, "fp16")
    manifest["graphs"]["obs_encoder"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="source ONNX"):
        load_engine_bundle(tmp_path, manifest, "fp16")


def test_ddim_replay_uses_explicit_noise_and_inverse_normalizer():
    """Given a fixed noise tensor, replay matches independent DDIM updates."""
    config = {"num_train_timesteps": 4, "beta_start": 0.0001,
              "beta_end": 0.02, "beta_schedule": "linear",
              "prediction_type": "epsilon", "clip_sample": True}
    manifest = {"scheduler": config, "num_inference_steps": 2,
                "action_normalizer": {"scale": [2] * 10, "offset": [0.25] * 10}}
    initial = torch.linspace(-0.5, 0.5, 20).reshape(1, 2, 10)
    observations = {"example": torch.zeros(1)}
    encoder = lambda obs: torch.zeros(1, 4)
    denoiser = lambda sample, step, cond: torch.full_like(sample, 0.1)
    condition, action, predictions = sample_trajectory(
        encoder, denoiser, observations, initial, manifest, capture_steps=True)
    scheduler = DDIMScheduler.from_config(config)
    scheduler.set_timesteps(2)
    expected = initial.clone()
    for step in scheduler.timesteps:
        expected = scheduler.step(torch.full_like(expected, 0.1), int(step), expected).prev_sample
    expected = (expected - 0.25) / 2
    torch.testing.assert_close(action, expected)
    assert condition.shape == (1, 4)
    assert predictions.shape == (2, 1, 2, 10)
    torch.testing.assert_close(unnormalize_action(initial, manifest), (initial - 0.25) / 2)


def test_action_error_reports_physical_units_and_rejects_invalid_rotations():
    """A 1 mm translation and quarter turn have readable physical errors."""
    reference = np.zeros((1, 1, 10), dtype=np.float32)
    reference[..., 3:9] = [1, 0, 0, 0, 1, 0]
    actual = reference.copy()
    actual[..., 0] = 0.001
    actual[..., 3:9] = [0, -1, 0, 1, 0, 0]
    actual[..., 9] = 0.02
    error = physical_errors(actual, reference)
    assert error["max_position_mm"] == pytest.approx(1.0)
    assert error["max_rotation_deg"] == pytest.approx(90.0)
    assert error["max_gripper"] == pytest.approx(0.02)
    invalid = actual.copy()
    invalid[..., 3:9] = 0
    with pytest.raises(ValueError, match="Degenerate rotation"):
        physical_errors(invalid, reference)
    with pytest.raises(ValueError, match="Nonfinite"):
        metrics(np.array([np.nan]), np.array([1.0]))


def test_latency_summary_uses_measured_call_rate():
    """Reported throughput and percentiles come from the same timings."""
    result = latency_stats([1.0, 2.0, 3.0, 4.0])
    assert result["mean_ms"] == 2.5
    assert result["median_ms"] == 2.5
    assert result["p90_ms"] == pytest.approx(3.7)
    assert result["calls_per_second"] == 400
    with pytest.raises(ValueError, match="positive"):
        latency_stats([1.0, 0.0])
