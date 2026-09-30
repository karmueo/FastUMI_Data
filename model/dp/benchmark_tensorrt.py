"""Benchmark Link7 PyTorch CUDA FP32 and TensorRT FP32/FP16 inference."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import torch

from tensorrt_link7 import (ENGINE_PRECISIONS, EngineRunner,
                           latency_stats, load_engine_bundle, load_onnx_bundle,
                           load_policy, load_samples, make_scheduler, require_trt10,
                           sample_trajectory, save_json, torch_observation)


def optional_device_status():
    """Capture available read-only Jetson power and clock telemetry."""
    readings = {}
    try:
        result = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True,
                                timeout=3, check=False)
        if result.returncode == 0 and result.stdout.strip():
            readings["nvpmodel"] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        # tegrastats on JetPack 6 streams until stopped; timeout captures one sample.
        subprocess.run(["tegrastats", "--interval", "100"],
                       capture_output=True, timeout=0.4, check=False)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or b""
        sample = output.decode("utf-8", errors="replace").splitlines()
        if sample:
            readings["tegrastats"] = sample[0]
    except FileNotFoundError:
        pass
    return readings


def measure_events(call, warmup, iterations, count):
    """Measure a GPU-resident component with synchronized CUDA events."""
    with torch.inference_mode():
        for index in range(warmup):
            call(index % count)
        torch.cuda.synchronize()
        values = []
        for index in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call(index % count)
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
    return latency_stats(values)


def measure_wall(call, warmup, iterations, count):
    """Measure full inference using synchronized wall-clock latency."""
    with torch.inference_mode():
        for index in range(warmup):
            call(index % count)
        torch.cuda.synchronize()
        values = []
        for index in range(iterations):
            torch.cuda.synchronize()
            started = time.perf_counter()
            call(index % count)
            torch.cuda.synchronize()
            values.append(1000 * (time.perf_counter() - started))
    result = latency_stats(values)
    result["predictions_per_second"] = result["calls_per_second"]
    return result


def main():
    """Benchmark equal shapes, samples, DDIM steps, and streams by backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--onnx-dir", required=True, type=Path)
    parser.add_argument("--engine-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup must be nonnegative and iterations must be positive")
    require_trt10()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    onnx_dir = args.onnx_dir.expanduser().resolve()
    engine_dir = (args.engine_dir or onnx_dir.parents[1] / "tensorrt" / onnx_dir.name).expanduser().resolve()
    manifest = load_onnx_bundle(onnx_dir, args.checkpoint)
    arrays = load_samples(onnx_dir, manifest)
    paths = {precision: load_engine_bundle(engine_dir, manifest, precision)
             for precision in ENGINE_PRECISIONS}
    policy = load_policy(args.checkpoint, manifest)
    runners = {precision: {graph: EngineRunner(path, manifest, graph)
                           for graph, path in paths[precision].items()}
               for precision in ENGINE_PRECISIONS}
    count = len(arrays["initial_noise"])
    resident_obs = [torch_observation(arrays, index) for index in range(count)]
    resident_noise = [torch.from_numpy(arrays["initial_noise"][index]).to("cuda:0")
                      for index in range(count)]
    status = optional_device_status()
    reports = {}

    for backend in ("pytorch_cuda_fp32", "tensorrt_fp32", "tensorrt_fp16"):
        if backend == "pytorch_cuda_fp32":
            def encoder(obs):
                return policy.obs_encoder.inference(policy.normalizer.normalize(obs))

            def denoiser(sample, step, cond):
                return policy.model(sample, step, global_cond=cond)
        else:
            precision = backend.removeprefix("tensorrt_")

            def encoder(obs, p=precision):
                return runners[p]["obs_encoder"](obs)

            def denoiser(sample, step, cond, p=precision):
                return runners[p]["denoiser"](
                    {"sample": sample, "timestep": step, "global_cond": cond})

        with torch.inference_mode():
            conditions = [encoder(obs).detach().clone() for obs in resident_obs]
        scheduler = make_scheduler(manifest)
        step = scheduler.step_inputs[int(scheduler.timesteps[0])]
        print(f"Benchmarking {backend}", flush=True)
        encoder_stats = measure_events(lambda i: encoder(resident_obs[i]),
                                       args.warmup, args.iterations, count)
        denoiser_stats = measure_events(
            lambda i: denoiser(resident_noise[i], step, conditions[i]),
            args.warmup, args.iterations, count)

        def full_resident(index):
            return sample_trajectory(encoder, denoiser, resident_obs[index],
                                     resident_noise[index], manifest, scheduler=scheduler)[1]

        resident_stats = measure_wall(full_resident, args.warmup, args.iterations, count)

        def full_end_to_end(index):
            obs = {name: torch.from_numpy(arrays[f"obs__{name}"][index]).to("cuda:0")
                   for name in manifest["observation_order"]}
            noise = torch.from_numpy(arrays["initial_noise"][index]).to("cuda:0")
            action = sample_trajectory(encoder, denoiser, obs, noise, manifest,
                                       scheduler=scheduler)[1]
            return action.cpu()

        end_to_end_stats = measure_wall(full_end_to_end, args.warmup, args.iterations, count)
        reports[backend] = {
            "encoder_gpu": encoder_stats, "denoiser_gpu": denoiser_stats,
            "full_gpu_resident": resident_stats, "full_end_to_end": end_to_end_stats,
        }
        print(f"{backend}: full end-to-end mean {end_to_end_stats['mean_ms']:.2f} ms", flush=True)

    speedups = {}
    for backend in ("tensorrt_fp32", "tensorrt_fp16"):
        speedups[backend] = {}
        for component in reports[backend]:
            target = reports[backend][component]["mean_ms"]
            speedups[backend][component] = {
                "vs_pytorch_cuda_fp32": reports["pytorch_cuda_fp32"][component]["mean_ms"] / target,
                "vs_tensorrt_fp32": reports["tensorrt_fp32"][component]["mean_ms"] / target,
            }
    report = {
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "tensorrt": __import__("tensorrt").__version__,
        "device_status": status,
        "device_status_capture": "before timed sections; tegrastats is a single idle snapshot",
        "warmup": args.warmup, "iterations": args.iterations,
        "validation_samples_rotated": count,
        "num_inference_steps": manifest["num_inference_steps"],
        "tf32_enabled": False,
        "backend_effective_precisions": {
            "pytorch_cuda_fp32": {"obs_encoder": "fp32", "denoiser": "fp32"},
            "tensorrt_fp32": {"obs_encoder": "fp32", "denoiser": "fp32"},
            "tensorrt_fp16": {"obs_encoder": "fp16", "denoiser": "fp32_fallback"},
        },
        "measurement": {
            "component": "CUDA events, GPU-resident inputs and outputs",
            "full_gpu_resident": "Synchronized wall-clock, GPU-resident inputs and outputs",
            "full_end_to_end": "Synchronized wall-clock, CPU inputs to GPU and action back to CPU",
            "excluded": ["model loading", "engine loading", "file I/O", "warmup"],
        },
        "latencies": reports, "speedups": speedups,
    }
    output = engine_dir / "reports" / "benchmark_report.json"
    save_json(output, report)
    print(json.dumps({"report": str(output), "end_to_end_speedups": {
        key: value["full_end_to_end"] for key, value in speedups.items()}}, indent=2))


if __name__ == "__main__":
    main()
