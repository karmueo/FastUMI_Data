"""Compare two exported ONNX graphs against a trusted PyTorch Link7 checkpoint."""

import argparse
import json
from pathlib import Path

import hydra
import numpy as np
import onnxruntime as ort
import torch
from diffusers import DDIMScheduler

from export_onnx import (Denoiser, OBS_KEYS, ObservationEncoder, input_shapes,
                         load_policy, sha256_file)


def compare_arrays(actual, expected, atol, rtol):
    """Compute error metrics and a strict finite-aware allclose decision."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        return {"passed": False, "actual_shape": list(actual.shape),
                "expected_shape": list(expected.shape), "reason": "shape_mismatch"}
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return {"passed": False, "shape": list(actual.shape), "reason": "non_finite"}
    error = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(error)
    return {
        "passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
        "shape": list(actual.shape),
        "max_abs_error": float(absolute.max(initial=0)),
        "mean_abs_error": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
    }


def unnormalize_action(sample, normalizer):
    """Invert the checkpoint's action scale and offset in FP32."""
    scale = np.asarray(normalizer["scale"], dtype=np.float32)
    offset = np.asarray(normalizer["offset"], dtype=np.float32)
    if np.any(scale == 0) or not np.isfinite(scale).all() or not np.isfinite(offset).all():
        raise ValueError("Invalid action normalization coefficients")
    return (np.asarray(sample, dtype=np.float32) - offset) / scale


def initial_noise_for_reference(policy, observations, seed):
    """Capture the exact initial noise consumed by the next predict_action call."""
    torch.manual_seed(seed)
    state = torch.random.get_rng_state()
    batch = next(iter(observations.values())).shape[0]
    noise = torch.randn((batch, policy.action_horizon, policy.action_dim),
                        dtype=policy.dtype, device=policy.device)
    torch.random.set_rng_state(state)
    return noise


def make_session(path):
    """Create a reproducible CPU ONNX Runtime session with bounded threads."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=["CPUExecutionProvider"])


def validate_observations(observations, shapes):
    """Check names, shapes, float dtypes and the RGB input range."""
    if set(observations) != set(OBS_KEYS):
        raise ValueError("Expected the five Link7 observation keys")
    for name in OBS_KEYS:
        value = observations[name]
        if list(value.shape) != shapes[name] or value.dtype != torch.float32:
            raise ValueError(f"Invalid observation shape or dtype for {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite observation: {name}")
    image = observations["camera0_rgb"]
    if torch.any(image < 0) or torch.any(image > 1):
        raise ValueError("camera0_rgb must be in [0,1]")


def select_indices(dataset_length, count):
    """Spread deterministic validation windows across the full dataset."""
    if count < 1 or count > dataset_length:
        raise ValueError(f"num-samples must be between 1 and {dataset_length}")
    return np.linspace(0, dataset_length - 1, count, dtype=np.int64).tolist()


def check_bundle(onnx_dir, manifest, checkpoint, cfg, weight_key):
    """Reject stale, incomplete, or contract-incompatible exported graphs."""
    if manifest["schema_version"] != 1 or manifest["batch_size"] != 1:
        raise ValueError("Unsupported ONNX manifest")
    if manifest["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("ONNX bundle was exported from a different checkpoint")
    if manifest["weights"] != weight_key or manifest["observation_shapes"] != input_shapes(cfg):
        raise ValueError("ONNX bundle does not match the checkpoint contract")
    if manifest["num_inference_steps"] != int(cfg.policy.num_inference_steps):
        raise ValueError("ONNX bundle has different diffusion step count")
    paths = {}
    for name in ("obs_encoder", "denoiser"):
        entry = manifest["graphs"][name]
        path = onnx_dir / entry["file"]
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Missing or changed ONNX graph: {name}")
        paths[name] = path
    return paths


def run_sample(policy, encoder, denoiser, sessions, observations, manifest, seed, atol, rtol):
    """Run the original policy and matched PyTorch/ONNX diffusion trajectories."""
    validate_observations(observations, manifest["observation_shapes"])
    initial_noise = initial_noise_for_reference(policy, observations, seed)
    with torch.inference_mode():
        reference = policy.predict_action(observations)["action_pred"].cpu().numpy()
        pt_condition = encoder(*(observations[name] for name in OBS_KEYS))
    onnx_condition = sessions["obs_encoder"].run(None, {
        name: observations[name].numpy() for name in OBS_KEYS
    })[0]
    condition_comparison = compare_arrays(onnx_condition, pt_condition.numpy(), atol, rtol)

    # Recreate each deterministic DDIM update using the exported scheduler config.
    pt_scheduler = DDIMScheduler.from_config(manifest["scheduler"])
    onnx_scheduler = DDIMScheduler.from_config(manifest["scheduler"])
    pt_scheduler.set_timesteps(manifest["num_inference_steps"])
    onnx_scheduler.set_timesteps(manifest["num_inference_steps"])
    pt_sample = initial_noise.clone()
    onnx_sample = initial_noise.clone()
    step_actual = []
    step_expected = []
    with torch.inference_mode():
        for step in pt_scheduler.timesteps:
            timestep = np.array([float(step)], dtype=np.float32)
            pt_prediction = denoiser(pt_sample, torch.from_numpy(timestep), pt_condition)
            # Isolated graph comparison: both denoisers receive the same sample and condition.
            checked_prediction = sessions["denoiser"].run(None, {
                "sample": pt_sample.numpy(), "timestep": timestep,
                "global_cond": pt_condition.numpy(),
            })[0]
            step_actual.append(checked_prediction)
            step_expected.append(pt_prediction.numpy())
            onnx_prediction = sessions["denoiser"].run(None, {
                "sample": onnx_sample.numpy(), "timestep": timestep,
                "global_cond": onnx_condition,
            })[0]
            pt_sample = pt_scheduler.step(pt_prediction, int(step), pt_sample).prev_sample
            onnx_sample = onnx_scheduler.step(torch.from_numpy(onnx_prediction),
                                              int(step), onnx_sample).prev_sample
        pt_action = policy.normalizer["action"].unnormalize(pt_sample).numpy()

    onnx_action = unnormalize_action(onnx_sample.numpy(), manifest["action_normalizer"])
    step_comparison = compare_arrays(np.stack(step_actual), np.stack(step_expected), atol, rtol)
    reconstruction = compare_arrays(pt_action, reference, atol, rtol)
    action_comparison = compare_arrays(onnx_action, reference, atol, rtol)
    groups = {
        "position": compare_arrays(onnx_action[..., :3], reference[..., :3], atol, rtol),
        "rotation_6d": compare_arrays(onnx_action[..., 3:9], reference[..., 3:9], atol, rtol),
        "gripper": compare_arrays(onnx_action[..., 9:], reference[..., 9:], atol, rtol),
    }
    passed = all(item["passed"] for item in (
        condition_comparison, step_comparison, reconstruction, action_comparison, *groups.values()
    ))
    result = {
        "passed": passed, "encoder": condition_comparison, "denoiser": step_comparison,
        "reference_reconstruction": reconstruction, "action": action_comparison,
        "action_groups": groups,
    }
    arrays = {
        "initial_noise": initial_noise.numpy(), "pytorch_condition": pt_condition.numpy(),
        "onnx_condition": onnx_condition, "pytorch_action": reference,
        "onnx_action": onnx_action, "pytorch_denoiser": np.stack(step_expected),
        "onnx_denoiser": np.stack(step_actual),
        **{f"obs__{name}": observations[name].numpy() for name in OBS_KEYS},
    }
    return result, arrays


def main():
    """Verify fixed validation windows and save numeric evidence even on mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--onnx-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()
    if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        parser.error("atol and rtol must be finite and nonnegative")
    torch.set_num_threads(4)
    policy, cfg, weight_key = load_policy(args.checkpoint)
    onnx_dir = args.onnx_dir.expanduser().resolve()
    manifest = json.loads((onnx_dir / "manifest.json").read_text(encoding="utf-8"))
    paths = check_bundle(onnx_dir, manifest, args.checkpoint, cfg, weight_key)
    sessions = {name: make_session(path) for name, path in paths.items()}
    cfg.task.dataset.dataset_path = str(args.dataset.expanduser().resolve())
    split_path = args.checkpoint.parent.parent / "dataset_split.json"
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        cfg.task.dataset.train_episode_indices = split["train_episode_indices"]
        cfg.task.dataset.val_episode_indices = split["val_episode_indices"]
    # Zero these values explicitly, including for checkpoints trained with augmentation.
    cfg.task.dataset.repeat_frame_prob = 0.0
    cfg.task.dataset.start_pose_noise_std = 0.0
    dataset = hydra.utils.instantiate(cfg.task.dataset).get_validation_dataset()
    indices = select_indices(len(dataset), args.num_samples)
    encoder = ObservationEncoder(policy).eval()
    denoiser = Denoiser(policy).eval()
    results = []
    records = []
    for index in indices:
        batch = dataset[index]
        observations = {name: batch["obs"][name].unsqueeze(0).float() for name in OBS_KEYS}
        result, arrays = run_sample(policy, encoder, denoiser, sessions, observations,
                                    manifest, args.seed + index, args.atol, args.rtol)
        result["validation_index"] = index
        result["seed"] = args.seed + index
        results.append(result)
        records.append(arrays)
        print(f"validation window {index}: {'pass' if result['passed'] else 'FAIL'}", flush=True)

    report = {
        "passed": all(item["passed"] for item in results),
        "checkpoint": str(args.checkpoint.resolve()), "dataset": str(args.dataset.resolve()),
        "weights": weight_key, "runtime": "onnxruntime-cpu", "dtype": "float32",
        "atol": args.atol, "rtol": args.rtol,
        "current_dataset_episodes": dataset.replay_buffer.n_episodes,
        "training_split_episodes": (
            len(split["train_episode_indices"]) + len(split["val_episode_indices"])
            if split_path.is_file() else None
        ),
        "validation_windows": len(dataset),
        "split_manifest": str(split_path.resolve()) if split_path.is_file() else None,
        "sample_indices": indices, "samples": results,
    }
    for stage in ("encoder", "denoiser", "reference_reconstruction", "action"):
        values = [item[stage]["max_abs_error"] for item in results
                  if "max_abs_error" in item[stage]]
        report[f"max_{stage}_abs_error"] = max(values) if values else None
    onnx_dir.joinpath("validation_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(onnx_dir / "validation_samples.npz", **{
        name: np.stack([record[name] for record in records]) for name in records[0]
    })
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
