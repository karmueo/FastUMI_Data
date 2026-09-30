"""用现有固定噪声样本验证 C++ TensorRT，并生成独立数值与性能报告。"""

import argparse
import copy
import gc
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def prepare_samples(arrays, directory):
    """将 NPZ 转成 C++ 可读取的连续小端 FP32，保留相同初始噪声。"""
    directory.mkdir(parents=True, exist_ok=True)
    count = len(arrays["initial_noise"])
    for index in range(count):
        for key, values in arrays.items():
            if key.startswith("obs__") or key == "initial_noise":
                name = key.removeprefix("obs__")
                np.asarray(values[index], dtype="<f4").tofile(directory / f"{index}_{name}.bin")
    (directory / "samples.json").write_text(json.dumps({"count": count}), encoding="utf-8")


def read_predictions(directory, count):
    """读取 C++ 核心的物理 pose10 动作与编码器 condition。"""
    action = np.stack([np.fromfile(directory / f"{i}_action.bin", dtype="<f4").reshape(1, 16, 10)
                       for i in range(count)])
    condition = np.stack([np.fromfile(directory / f"{i}_condition.bin", dtype="<f4").reshape(1, 1568)
                          for i in range(count)])
    if not np.isfinite(action).all() or not np.isfinite(condition).all():
        raise ValueError("C++ returned nonfinite action or condition")
    return action, condition


def main():
    """16 步对照已保存 TensorRT 结果，8 步对照现有 Python TensorRT 实现。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / "model/dp"))
    import torch
    from tensorrt_link7 import (EngineRunner, load_engine_bundle, load_onnx_bundle,
                               load_samples, metrics, physical_errors, require_trt10,
                               sample_trajectory, torch_observation)

    require_trt10()
    engine_dir = args.engine_dir.resolve()
    onnx_dir = engine_dir.parents[1] / "onnx" / engine_dir.name
    output = (args.output or engine_dir / "reports/cpp_ros2").resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_onnx_bundle(onnx_dir)
    arrays = load_samples(onnx_dir, manifest)
    with np.load(engine_dir / "reports/precision_arrays.npz", allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    prepare_samples(arrays, output / "inputs")
    report = {"status": "passed", "atol": 1e-5, "rtol": 1e-4,
              "samples": len(arrays["initial_noise"]), "results": {},
              "source_checkpoint_sha256": manifest["checkpoint_sha256"],
              "note": "Numerical/runtime acceptance only; task success accuracy is not measured."}
    all_passed = True
    for precision in ("fp32", "fp16"):
        for steps in (16, 8):
            key = f"{precision}_{steps}"
            destination = output / key
            destination.mkdir(exist_ok=True)
            command = [str(args.runner.resolve()), "--engine-dir", str(engine_dir),
                       "--input-dir", str(output / "inputs"), "--output-dir", str(destination),
                       "--precision", precision, "--steps", str(steps),
                       "--warmup", str(args.warmup), "--iterations", str(args.iterations)]
            with (destination / "runner.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            action, condition = read_predictions(destination, report["samples"])
            if steps == 16:
                expected_action = saved[f"tensorrt_{precision}_action"]
                expected_condition = saved[f"tensorrt_{precision}_condition"]
                if expected_action.shape != action.shape or expected_condition.shape != condition.shape:
                    raise ValueError("Saved TensorRT reference shape does not match sample count")
            else:
                paths = load_engine_bundle(engine_dir, manifest, precision)
                encoder = EngineRunner(paths["obs_encoder"], manifest, "obs_encoder")
                denoiser = EngineRunner(paths["denoiser"], manifest, "denoiser")
                schedule_manifest = copy.deepcopy(manifest)
                schedule_manifest["num_inference_steps"] = steps
                actions, conditions = [], []
                with torch.inference_mode():
                    for index in range(report["samples"]):
                        observations = torch_observation(arrays, index)
                        noise = torch.from_numpy(arrays["initial_noise"][index]).to("cuda:0")
                        cond, predicted, _ = sample_trajectory(
                            lambda obs: encoder(obs),
                            lambda sample, timestep, global_cond: denoiser(
                                {"sample": sample, "timestep": timestep, "global_cond": global_cond}),
                            observations, noise, schedule_manifest)
                        conditions.append(cond.cpu().numpy().copy())
                        actions.append(predicted.cpu().numpy().copy())
                expected_action = np.stack(actions)
                expected_condition = np.stack(conditions)
                del encoder, denoiser, cond, predicted, noise, observations
                gc.collect()
                torch.cuda.empty_cache()
            passed = bool(np.allclose(action, expected_action, atol=1e-5, rtol=1e-4)
                          and np.allclose(condition, expected_condition, atol=1e-5, rtol=1e-4))
            result = {"passed": passed, "action": metrics(action, expected_action),
                      "condition": metrics(condition, expected_condition),
                      "physical": physical_errors(action, expected_action),
                      "benchmark": json.loads((destination / "benchmark.json").read_text())}
            if steps == 16:
                result["vs_pytorch_fp32"] = physical_errors(action, saved["pytorch_cuda_action"])
            report["results"][key] = result
            all_passed &= passed
            np.savez_compressed(destination / "comparison.npz", action=action, reference=expected_action,
                                condition=condition, reference_condition=expected_condition)
            print(f"{key}: passed={passed}, action_max_abs={result['action']['max_abs']:.3g}, "
                  f"mean_ms={result['benchmark']['full_end_to_end']['mean_ms']:.2f}", flush=True)
    report["status"] = "passed" if all_passed else "failed"
    (output / "runtime_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all_passed:
        raise SystemExit("C++ numerical parity failed; see runtime_report.json")


if __name__ == "__main__":
    main()
