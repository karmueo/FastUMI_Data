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


BENCHMARK_BACKENDS = (
    ("pytorch_cuda_fp32", "PyTorch CUDA FP32", "#333333"),
    ("tensorrt_fp32", "TensorRT FP32", "#0072B2"),
    ("tensorrt_fp16", "TensorRT FP16*", "#D55E00"),
)
BENCHMARK_COMPONENTS = (
    ("encoder_gpu", "观测编码器"),
    ("denoiser_gpu", "单步去噪器"),
    ("full_gpu_resident", "完整策略（GPU 驻留）"),
    ("full_end_to_end", "完整策略（端到端）"),
)


def save_benchmark_plots(report, report_dir):
    """Render cached latency (ms), full predictions/s, and mean-latency speedups.

    Save two 200 DPI PNGs without loading models or running GPU inference.
    Statistics and speedups are taken directly from the existing JSON schema.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "benchmark_summary.png"
    latency_path = report_dir / "benchmark_latency_stats.png"
    latencies = report["latencies"]
    footer = (
        f"设备：{report['gpu']}  |  PyTorch {report['torch']}  |  "
        f"CUDA {report['cuda']}  |  TensorRT {report['tensorrt']}\n"
        f"每项预热：{report['warmup']} 次  |  每项测速：{report['iterations']} 次  |  "
        f"轮换样本：{report['validation_samples_rotated']} 个  |  "
        f"DDIM：{report['num_inference_steps']} 步  |  "
        f"TF32：{'开启' if report['tf32_enabled'] else '关闭'}\n"
        "* FP16 编码器 + FP32 回退去噪器\n"
        "计时：组件使用 CUDA events；完整策略使用同步墙钟；端到端包含输入上传和动作下载。\n"
        "不计入模型及引擎加载、文件读取和预热；吞吐量表示每秒完整预测次数。"
    )

    def label_bars(axis, bars, suffix=""):
        """Annotate measured values and leave room above zero-based bars."""
        axis.bar_label(bars, labels=[f"{bar.get_height():.2f}{suffix}" for bar in bars],
                       padding=4, fontsize=9)
        axis.autoscale(enable=True, axis="y")
        axis.margins(y=0.25)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)

    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP",
                                             "WenQuanYi Micro Hei", "Microsoft YaHei",
                                             "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False,
                         "font.size": 11, "axes.titlesize": 13,
                         "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 3, figsize=(16, 10))
        try:
            figure.suptitle("TensorRT 推理性能总览", fontsize=18, y=0.97)
            figure.text(0.5, 0.925, "延迟越低越好；吞吐量和加速比越高越好",
                        ha="center", fontsize=11)
            figure.subplots_adjust(top=0.86, bottom=0.24, hspace=0.6, wspace=0.32)
            for axis, (component, title) in zip(axes.flat, BENCHMARK_COMPONENTS):
                values = [latencies[name][component]["mean_ms"]
                          for name, _, _ in BENCHMARK_BACKENDS]
                bars = axis.bar(range(3), values, width=0.55,
                                color=[color for _, _, color in BENCHMARK_BACKENDS])
                label_bars(axis, bars)
                axis.set(title=f"{title}平均延迟", ylabel="延迟 (ms)")
            values = [latencies[name]["full_end_to_end"]["predictions_per_second"]
                      for name, _, _ in BENCHMARK_BACKENDS]
            bars = axes[1, 1].bar(range(3), values, width=0.55,
                                   color=[color for _, _, color in BENCHMARK_BACKENDS])
            label_bars(axes[1, 1], bars)
            axes[1, 1].set(title="端到端完整预测吞吐量", ylabel="完整预测次数（次/s）")
            for axis in list(axes.flat)[:5]:
                axis.set_xticks(range(3), ["PyTorch\nCUDA FP32", "TensorRT\nFP32",
                                           "TensorRT\nFP16*"])
            speedup_axis = axes[1, 2]
            for offset, (backend, label, color) in zip((-0.18, 0.18), BENCHMARK_BACKENDS[1:]):
                values = [report["speedups"][backend]["full_end_to_end"][baseline]
                          for baseline in ("vs_pytorch_cuda_fp32", "vs_tensorrt_fp32")]
                bars = speedup_axis.bar([index + offset for index in range(2)], values,
                                        width=0.32, color=color, label=label)
                label_bars(speedup_axis, bars, "×")
            speedup_axis.axhline(1, color="#777777", linestyle="--", linewidth=1)
            speedup_axis.set_xticks(range(2), ["相对 PyTorch\nCUDA FP32", "相对 TensorRT\nFP32"])
            speedup_axis.set(title="端到端加速比", ylabel="加速倍数（1× 为基准）")
            speedup_axis.legend(fontsize=9, loc="upper right")
            figure.text(0.5, 0.035, footer, ha="center", va="bottom", fontsize=10,
                        linespacing=1.6)
            figure.savefig(summary_path, dpi=200)
        finally:
            plt.close(figure)

        figure, axes = plt.subplots(2, 2, figsize=(16, 11))
        try:
            figure.suptitle("TensorRT 延迟统计对比", fontsize=18, y=0.97)
            figure.subplots_adjust(top=0.87, bottom=0.23, hspace=0.4, wspace=0.25)
            statistic_keys = ("mean_ms", "median_ms", "p90_ms", "p95_ms")
            for axis, (component, title) in zip(axes.flat, BENCHMARK_COMPONENTS):
                for offset, (backend, label, color) in zip((-0.24, 0, 0.24), BENCHMARK_BACKENDS):
                    values = [latencies[backend][component][key] for key in statistic_keys]
                    bars = axis.bar([index + offset for index in range(4)], values,
                                    width=0.21, color=color, label=label)
                    label_bars(axis, bars)
                axis.set_xticks(range(4), ["平均值", "中位数", "P90", "P95"])
                axis.set(title=title, ylabel="延迟 (ms)")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                          ncol=3, frameon=False)
            figure.text(0.5, 0.035, footer, ha="center", va="bottom", fontsize=10,
                        linespacing=1.6)
            figure.savefig(latency_path, dpi=200)
        finally:
            plt.close(figure)
    return summary_path, latency_path


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
    summary_path, latency_path = save_benchmark_plots(report, output.parent)
    print(json.dumps({"report": str(output),
                     "summary_plot": str(summary_path), "latency_plot": str(latency_path),
                     "end_to_end_speedups": {
        key: value["full_end_to_end"] for key, value in speedups.items()}}, indent=2))


if __name__ == "__main__":
    main()
