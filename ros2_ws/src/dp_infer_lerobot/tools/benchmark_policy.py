#!/usr/bin/env python3
"""在目标机上对比 PyTorch 与 TensorRT 后端的推理延迟、分阶段耗时、内存与动作误差。

每个后端都经 :class:`dp_infer_lerobot.engine.PolicyEngine` 正式加载（TensorRT 包须已通过
``build_tensorrt_engines.py`` 验证），依次测量各去噪步数：

- 端到端 ``predict_raw``（numpy 输入 → CPU 绝对动作）的中位 / P95 / 最大延迟；
- 分阶段耗时：预处理、编码器、单步去噪（U-Net / engine）、调度器与其余开销、后处理
  （每阶段前后同步 CUDA，单独一轮测量，不计入端到端统计）；
- 相对 PyTorch 同步数、同初始噪声的最终动作物理误差。

在推理环境运行（仓库根目录）::

    PYTHONPATH=ros2_ws/src/dp_infer:ros2_ws/src/dp_infer_lerobot \\
    ros2_ws/.venv-dp-lerobot/bin/python ros2_ws/src/dp_infer_lerobot/tools/benchmark_policy.py \\
        --checkpoint dataset/lerobot/jingbao_models/012000/pretrained_model \\
        --export-dir dataset/lerobot/jingbao_models/lerobot_onnx/dp_umi_v4/lerobot_trt_orin \\
        --video dataset/lerobot/jingbao/videos/observation.images.cam0/chunk-000/file-000.mp4 \\
        --output dataset/validation/dp_lerobot_v4/benchmark_orin.json
"""

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))  # 复用同目录构建脚本的输入构造与误差统计。

from build_tensorrt_engines import DEFAULT_PROFILE, error_summary, make_inputs, read_json, tegra_release  # noqa: E402

BACKENDS = ("pytorch", "tensorrt_fp32", "tensorrt_fp16")  # 默认测量顺序；PyTorch 须在前作为误差基准。


class StageTimer:
    """包装可调用对象，前后同步 CUDA 后记录墙钟耗时（ms）。"""

    def __init__(self, device):
        """``device`` 为 CUDA 设备；CPU 时不同步。"""
        import torch

        self.sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
        self.times = defaultdict(list)  # 阶段名 → 每次调用耗时 ms。
        self.enabled = False  # 只在分阶段测量轮记录。

    def wrap(self, name, function):
        """返回记录 ``name`` 耗时的包装函数。"""
        def timed(*args, **kwargs):
            if not self.enabled:
                return function(*args, **kwargs)
            self.sync()
            started = time.perf_counter()
            result = function(*args, **kwargs)
            self.sync()
            self.times[name].append((time.perf_counter() - started) * 1000)
            return result
        return timed


def instrument(engine, timer):
    """在引擎实例上挂载分阶段计时（只替换实例属性，不改类）。"""
    engine.processors.preprocess = timer.wrap("preprocess", engine.processors.preprocess)
    engine.processors.postprocess = timer.wrap("postprocess", engine.processors.postprocess)
    engine.policy.predict_action_chunk = timer.wrap("sampling", engine.policy.predict_action_chunk)
    if hasattr(engine.policy, "diffusion"):
        diffusion = engine.policy.diffusion
        diffusion._prepare_global_conditioning = timer.wrap("encoder", diffusion._prepare_global_conditioning)
        diffusion.unet.forward = timer.wrap("denoiser", diffusion.unet.forward)
    else:
        session = engine.policy.session
        run = session.run
        session.run = lambda stage, inputs: timer.wrap(stage, run)(stage, inputs)


def stage_summary(timer, steps, iterations):
    """把分阶段记录汇总为每次推理的平均耗时；去噪按步数给出单步与合计。"""
    mean = {name: float(np.mean(values)) for name, values in timer.times.items()}
    denoiser_total = float(np.sum(timer.times["denoiser"]) / iterations)
    return {
        "preprocess_ms": mean["preprocess"], "encoder_ms": mean["encoder"],
        "denoiser_step_ms": mean["denoiser"], "denoiser_total_ms": denoiser_total,
        "scheduler_and_other_ms": mean["sampling"] - mean["encoder"] - denoiser_total,
        "postprocess_ms": mean["postprocess"], "steps": steps,
    }


def measure(engine, images, states, noises, steps, iterations, warmup):
    """测量一个步数设置：端到端延迟、分阶段耗时与各输入的动作，返回 ``(结果, actions)``。"""
    import torch

    from dp_infer_lerobot.config import IMAGE_KEY, STATE_KEY

    observations = [{IMAGE_KEY: image[None], STATE_KEY: state[None]} for image, state in zip(images, states)]
    engine.set_inference_steps(steps)
    for index in range(warmup):
        engine.predict_raw(observations[index % len(observations)])
    latency = []
    for index in range(iterations):
        torch.cuda.synchronize(engine.device)
        started = time.perf_counter()
        engine.predict_raw(observations[index % len(observations)])
        latency.append((time.perf_counter() - started) * 1000)
    timer = engine.timer
    timer.times.clear()
    timer.enabled = True
    stage_iterations = max(3, iterations // 3)
    for index in range(stage_iterations):
        engine.predict_raw(observations[index % len(observations)])
    timer.enabled = False
    actions = []
    for observation, noise in zip(observations, noises):
        torch.manual_seed(0)
        actions.append(engine.predict_raw(observation, noise=noise)[0])
    latency = np.asarray(latency)
    result = {"median_ms": float(np.median(latency)), "p95_ms": float(np.percentile(latency, 95)),
              "max_ms": float(latency.max()), "mean_ms": float(latency.mean()), "iterations": iterations,
              "stages": stage_summary(timer, steps, stage_iterations)}
    return result, np.asarray(actions)


def memory_used_mib(torch, device):
    """设备已用内存（MiB）；Orin 为 CPU/GPU 共享内存，包含其他进程，只作量级参考。"""
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 2**20


def environment():
    """记录功耗模式、GPU 频率与软件版本，便于复现。"""
    import tensorrt as trt
    import torch

    info = {"l4t": tegra_release(), "torch": torch.__version__, "tensorrt": trt.__version__,
            "gpu": torch.cuda.get_device_name(0)}
    try:
        info["nvpmodel"] = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True,
                                          timeout=10).stdout.strip().splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        info["nvpmodel"] = ""
    for devfreq in Path("/sys/class/devfreq").glob("*.gpu"):
        info["gpu_freq_hz"] = {name: int((devfreq / name).read_text()) for name in ("min_freq", "max_freq", "cur_freq")
                               if (devfreq / name).is_file()}
    return info


def markdown(results, steps_list):
    """生成延迟与加速比 Markdown 表（相对 PyTorch 同步数的中位延迟）。"""
    lines = ["| 后端 | " + " | ".join(f"DDIM{steps} 中位 / P95 (ms)" for steps in steps_list) + " |",
             "| --- |" + " --- |" * len(steps_list)]
    base = results.get("pytorch", {}).get("steps", {})
    for backend, result in results.items():
        cells = []
        for steps in steps_list:
            value = result["steps"][str(steps)]
            speedup = (f" ({base[str(steps)]['median_ms'] / value['median_ms']:.1f}×)"
                       if backend != "pytorch" and str(steps) in base else "")
            cells.append(f"{value['median_ms']:.1f} / {value['p95_ms']:.1f}{speedup}")
        lines.append(f"| {backend} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    """逐后端加载并测量，写出 JSON 报告。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--export-dir", default="")
    parser.add_argument("--deployment-config", default=str(DEFAULT_PROFILE))
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--noise-scheduler", choices=("DDPM", "DDIM"), default="DDIM")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=4, help="误差对比与轮换输入的样本数")
    parser.add_argument("--sample", default="")
    parser.add_argument("--video", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    import torch

    from dp_infer_lerobot.config import MODEL_DIM
    from dp_infer_lerobot.deployment import load_profile
    from dp_infer_lerobot.engine import PolicyEngine

    profile = load_profile(args.deployment_config)
    images, states, source = make_inputs(args.sample, args.video, args.samples)
    noises = [np.random.default_rng(index).standard_normal((1, 16, MODEL_DIM)).astype(np.float32)
              for index in range(len(images))]
    report = {"environment": environment(), "inputs": source, "scheduler": args.noise_scheduler,
              "warmup": args.warmup, "backends": {}}
    if args.export_dir:
        manifest = read_json(Path(args.export_dir) / "manifest.json")
        report["engines"] = {stage: {precision: {key: record.get(key) for key in ("build_seconds", "size_bytes")}
                                     for precision, record in spec["engines"].items()}
                             for stage, spec in manifest["stages"].items()}
    reference = {}  # PyTorch 各步数的动作，作为误差基准。
    for backend in args.backends:
        print(f"[benchmark] loading {backend}", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.device)
        before = memory_used_mib(torch, args.device)
        started = time.perf_counter()
        engine = PolicyEngine(args.checkpoint, profile, args.device, args.noise_scheduler, args.steps[0],
                              backend=backend, export_dir=args.export_dir)
        load_s = time.perf_counter() - started
        started = time.perf_counter()
        engine.warmup()  # 与节点启动一致；首个调用含 cuDNN 搜索 / TensorRT 初始化。
        first_ms = (time.perf_counter() - started) * 1000
        engine.timer = StageTimer(engine.device)
        instrument(engine, engine.timer)
        result = {"load_s": load_s, "first_call_ms": first_ms, "steps": {}}
        for steps in args.steps:
            measured, actions = measure(engine, images, states, noises, steps, args.iterations, args.warmup)
            if backend == "pytorch":
                reference[steps] = actions
            elif steps in reference:
                measured["actions_vs_pytorch"] = error_summary(actions, reference[steps])
            result["steps"][str(steps)] = measured
            print(f"[benchmark] {backend} DDIM{steps}: {measured['median_ms']:.1f} ms "
                  f"(p95 {measured['p95_ms']:.1f})", flush=True)
        result["memory"] = {"device_used_delta_mib": memory_used_mib(torch, args.device) - before,
                            "torch_max_allocated_mib": torch.cuda.max_memory_allocated(args.device) / 2**20}
        report["backends"][backend] = result
        del engine
        gc.collect()  # 释放上一后端的权重 / engine，避免叠加到下一后端的内存统计。
    report["markdown"] = markdown(report["backends"], args.steps)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report["markdown"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
