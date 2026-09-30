"""Compare TensorRT FP32/FP16 Link7 engines with checkpoint and saved ONNX references."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tensorrt_link7 import (ENGINE_PRECISIONS, OBS_KEYS, EngineRunner,
                           load_engine_bundle, load_onnx_bundle, load_policy,
                           load_samples, make_scheduler, metrics,
                           physical_errors, require_trt10, sample_trajectory,
                           save_json, torch_observation)


def compare_action(actual, expected):
    """Measure normalized components and physical pose10 action error."""
    return {"all": metrics(actual, expected),
            "position": metrics(actual[..., :3], expected[..., :3]),
            "rotation_6d": metrics(actual[..., 3:9], expected[..., 3:9]),
            "gripper": metrics(actual[..., 9:], expected[..., 9:]),
            "physical": physical_errors(actual, expected)}


def main():
    """Replay all saved validation windows with identical initial noise."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--onnx-dir", required=True, type=Path)
    parser.add_argument("--engine-dir", type=Path)
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()
    require_trt10()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    onnx_dir = args.onnx_dir.expanduser().resolve()
    engine_dir = (args.engine_dir or onnx_dir.parents[1] / "tensorrt" / onnx_dir.name).expanduser().resolve()
    manifest = load_onnx_bundle(onnx_dir, args.checkpoint)
    arrays = load_samples(onnx_dir, manifest)
    if args.num_samples < 1 or args.num_samples > len(arrays["initial_noise"]):
        parser.error(f"num-samples must be in [1,{len(arrays['initial_noise'])}]")
    paths = {precision: load_engine_bundle(engine_dir, manifest, precision)
             for precision in ENGINE_PRECISIONS}
    policy = load_policy(args.checkpoint, manifest)
    runners = {precision: {graph: EngineRunner(path, manifest, graph)
                           for graph, path in paths[precision].items()}
               for precision in ENGINE_PRECISIONS}

    def pt_encoder(obs):
        return policy.obs_encoder.inference(policy.normalizer.normalize(obs))

    def pt_denoiser(sample, step, cond):
        return policy.model(sample, step, global_cond=cond)

    def trt_encoder(precision, obs):
        return runners[precision]["obs_encoder"](obs)

    def trt_denoiser(precision, sample, step, cond):
        return runners[precision]["denoiser"](
            {"sample": sample, "timestep": step, "global_cond": cond})

    per_sample = []
    saved = {name: [] for name in (
        "pytorch_cuda_condition", "pytorch_cuda_action",
        "tensorrt_fp32_condition", "tensorrt_fp32_action",
        "tensorrt_fp16_condition", "tensorrt_fp16_action",
        "pytorch_cuda_denoiser", "tensorrt_fp32_isolated_denoiser",
        "tensorrt_fp16_isolated_denoiser")}
    with torch.inference_mode():
        for index in range(args.num_samples):
            observations = torch_observation(arrays, index)
            initial_noise = torch.from_numpy(arrays["initial_noise"][index]).to("cuda:0")
            pt_cond, pt_action, _ = sample_trajectory(
                pt_encoder, pt_denoiser, observations, initial_noise, manifest)
            pt_cond_cpu = pt_cond.detach().cpu().numpy()
            pt_action_cpu = pt_action.detach().cpu().numpy()

            # Each denoiser sees the same PyTorch sample, step, and condition.
            scheduler = make_scheduler(manifest)
            step_sample = initial_noise.clone()
            isolated = {"pytorch_cuda": [], "tensorrt_fp32": [], "tensorrt_fp16": []}
            for timestep in scheduler.timesteps:
                step = scheduler.step_inputs[int(timestep)]
                pt_prediction = pt_denoiser(step_sample, step, pt_cond)
                isolated["pytorch_cuda"].append(pt_prediction.detach().cpu().numpy())
                for precision in ENGINE_PRECISIONS:
                    prediction = trt_denoiser(precision, step_sample, step, pt_cond)
                    isolated[f"tensorrt_{precision}"].append(prediction.detach().cpu().numpy())
                step_sample = scheduler.step(pt_prediction, int(timestep), step_sample).prev_sample
            isolated = {key: np.stack(value) for key, value in isolated.items()}

            results = {}
            for precision in ENGINE_PRECISIONS:
                condition, action, _ = sample_trajectory(
                    lambda obs, p=precision: trt_encoder(p, obs),
                    lambda sample, step, cond, p=precision: trt_denoiser(p, sample, step, cond),
                    observations, initial_noise, manifest)
                results[precision] = (condition.detach().cpu().numpy(),
                                      action.detach().cpu().numpy())

            one = {
                "index": index,
                "saved_reference": {
                    "pytorch_cuda_vs_saved_pytorch_condition": metrics(
                        pt_cond_cpu, arrays["pytorch_condition"][index]),
                    "pytorch_cuda_vs_saved_pytorch_action": compare_action(
                        pt_action_cpu, arrays["pytorch_action"][index]),
                    "saved_onnx_vs_saved_pytorch_action": compare_action(
                        arrays["onnx_action"][index], arrays["pytorch_action"][index]),
                    "pytorch_cuda_vs_saved_pytorch_denoiser": metrics(
                        isolated["pytorch_cuda"], arrays["pytorch_denoiser"][index]),
                },
                "precision": {},
            }
            for precision in ENGINE_PRECISIONS:
                condition, action = results[precision]
                one["precision"][precision] = {
                    "encoder_vs_pytorch_cuda": metrics(condition, pt_cond_cpu),
                    "isolated_denoiser_vs_pytorch_cuda": metrics(
                        isolated[f"tensorrt_{precision}"], isolated["pytorch_cuda"]),
                    "action_vs_pytorch_cuda": compare_action(action, pt_action_cpu),
                    "action_vs_saved_onnx": compare_action(action, arrays["onnx_action"][index]),
                }
                saved[f"tensorrt_{precision}_condition"].append(condition)
                saved[f"tensorrt_{precision}_action"].append(action)
                saved[f"tensorrt_{precision}_isolated_denoiser"].append(
                    isolated[f"tensorrt_{precision}"])
            one["fp16_vs_fp32"] = {
                "encoder": metrics(results["fp16"][0], results["fp32"][0]),
                "action": compare_action(results["fp16"][1], results["fp32"][1]),
            }
            per_sample.append(one)
            saved["pytorch_cuda_condition"].append(pt_cond_cpu)
            saved["pytorch_cuda_action"].append(pt_action_cpu)
            saved["pytorch_cuda_denoiser"].append(isolated["pytorch_cuda"])
            print(f"Validated sample {index + 1}/{args.num_samples}", flush=True)

    report = {
        "status": "finite",
        "acceptance_threshold": None,
        "acceptance_note": "No task accuracy threshold is set; finite numerical loss is reported only.",
        "backend_effective_precisions": {
            "pytorch_cuda": {"obs_encoder": "fp32", "denoiser": "fp32"},
            "tensorrt_fp32": {"obs_encoder": "fp32", "denoiser": "fp32"},
            "tensorrt_fp16": {"obs_encoder": "fp16", "denoiser": "fp32_fallback"},
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "onnx_dir": str(onnx_dir), "engine_dir": str(engine_dir),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "num_samples": args.num_samples,
        "num_inference_steps": manifest["num_inference_steps"],
        "sample_results": per_sample,
    }
    report["maxima_across_samples"] = {
        precision: {
            "encoder_max_abs": max(item["precision"][precision]["encoder_vs_pytorch_cuda"]["max_abs"]
                                   for item in per_sample),
            "isolated_denoiser_max_abs": max(
                item["precision"][precision]["isolated_denoiser_vs_pytorch_cuda"]["max_abs"]
                for item in per_sample),
            "action_max_abs": max(item["precision"][precision]["action_vs_pytorch_cuda"]["all"]["max_abs"]
                                  for item in per_sample),
            **{name: max(item["precision"][precision]["action_vs_pytorch_cuda"]["physical"][name]
                         for item in per_sample)
               for name in ("max_position_mm", "max_rotation_deg", "max_gripper")},
        } for precision in ENGINE_PRECISIONS
    }
    report_dir = engine_dir / "reports"
    save_json(report_dir / "precision_report.json", report)
    np.savez_compressed(report_dir / "precision_arrays.npz",
                        **{name: np.stack(value) for name, value in saved.items()})
    print(json.dumps({"status": report["status"], "samples": args.num_samples,
                      "report": str(report_dir / "precision_report.json")}, indent=2))


if __name__ == "__main__":
    main()
