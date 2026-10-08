"""Compare TensorRT FP32/FP16 Link7 engines with checkpoint and saved ONNX references."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from tensorrt_link7 import (ENGINE_PRECISIONS, EngineRunner,
                           load_engine_bundle, load_onnx_bundle, load_policy,
                           load_samples, make_scheduler, metrics,
                           physical_errors, require_trt10, rotation_matrices, sample_trajectory,
                           save_json, torch_observation)


PLOT_STYLES = {
    "pytorch_cuda": {"label": "PyTorch CUDA FP32", "color": "#333333", "linestyle": "-"},
    "tensorrt_fp32": {"label": "TensorRT FP32", "color": "#0072B2", "linestyle": "--"},
    "tensorrt_fp16": {"label": "TensorRT FP16*", "color": "#D55E00", "linestyle": ":"},
}
PRECISION_NOTE = "* FP16 编码器 + FP32 回退去噪器"


def action_error_curves(actual, reference):
    """Return per-step pose10 errors in mm, degrees, and gripper encoding units.

    Inputs are [prediction_steps, 10], relative to the current observed end
    frame. Rotation distances use the same row-major 6D convention as the report.
    """
    metrics(actual, reference)
    if actual.ndim != 2 or actual.shape[-1] != 10:
        raise ValueError("Expected [prediction_steps, 10] pose10 action")
    actual_rot = Rotation.from_matrix(rotation_matrices(actual[:, 3:9]))
    reference_rot = Rotation.from_matrix(rotation_matrices(reference[:, 3:9]))
    return {
        "position_mm": 1000 * np.linalg.norm(actual[:, :3] - reference[:, :3], axis=-1),
        "rotation_deg": np.rad2deg((actual_rot * reference_rot.inv()).magnitude()),
        "gripper": np.abs(actual[:, 9] - reference[:, 9]),
    }


def save_precision_plots(report, arrays, report_dir):
    """Save a summary and one action PNG per [sample, 1, horizon, 10] array.

    Only cached CPU arrays are used; no model loading or GPU inference runs.
    Existing sample_NNN.png files outside this report's indices are removed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report_dir = Path(report_dir)
    sample_dir = report_dir / "action_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    reference = arrays["pytorch_cuda_action"]
    if (reference.ndim != 4 or reference.shape[0] != report["num_samples"]
            or reference.shape[1] != 1 or reference.shape[-1] != 10):
        raise ValueError("Expected action arrays with shape [num_samples, 1, horizon, 10]")
    for precision in ENGINE_PRECISIONS:
        metrics(arrays[f"tensorrt_{precision}_action"], reference)

    summary_path = report_dir / "precision_summary.png"
    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Noto Sans CJK SC", "Noto Sans CJK JP",
                                             "WenQuanYi Micro Hei", "Microsoft YaHei",
                                             "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False,
                         "font.size": 11, "axes.titlesize": 13,
                         "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        try:
            figure.suptitle("TensorRT 精度验证（基准：PyTorch CUDA FP32）", fontsize=18)
            panels = (
                ("max_position_mm", "位置误差", "最大距离差 (mm)"),
                ("max_rotation_deg", "旋转误差", "最大角度差 (deg)"),
                ("max_gripper", "夹爪误差", "最大归一化编码差"),
                ("encoder_max_abs", "观测编码器误差", "最大绝对误差"),
                ("isolated_denoiser_max_abs", "单步去噪器误差", "最大绝对误差"),
            )
            for axis, (key, title, unit) in zip(axes.flat, panels):
                values = np.array([report["maxima_across_samples"][p][key]
                                   for p in ENGINE_PRECISIONS])
                positive = values[values > 0]
                if len(positive):
                    axis.set_yscale("symlog", linthresh=float(positive.min()) / 10)
                axis.bar(range(2), values, width=0.5,
                         color=[PLOT_STYLES[f"tensorrt_{p}"]["color"]
                                for p in ENGINE_PRECISIONS])
                for index, value in enumerate(values):
                    axis.annotate(f"{value:.3e}", (index, value),
                                  xytext=(0, 7), textcoords="offset points", ha="center")
                axis.set_xticks(range(2), ["FP32", "FP16*"])
                axis.set(title=title, ylabel=unit)
                axis.set_ylim(bottom=0)
                axis.margins(y=0.25)
                axis.grid(axis="y", alpha=0.2)
                axis.set_axisbelow(True)
            axes[1, 2].axis("off")
            axes[1, 2].text(
                0, 0.95,
                "验证说明\n\n"
                f"样本数：{report['num_samples']}\n"
                f"DDIM 推理步数：{report['num_inference_steps']}\n"
                f"预测长度：{reference.shape[2]} 步\n"
                f"数值状态：{'全部有限' if report['status'] == 'finite' else report['status']}\n\n"
                "未设置验收阈值。\n"
                "数值有限不代表达到任务精度。\n\n"
                "柱状图展示所有样本中的最大误差。\n"
                "非零面板使用对称对数轴（symlog）。\n\n"
                f"{PRECISION_NOTE}\n\n"
                f"模型检查点 SHA256：\n{report['checkpoint_sha256'][:16]}...",
                transform=axes[1, 2].transAxes, va="top", linespacing=1.6,
            )
            figure.savefig(summary_path, dpi=200)
        finally:
            plt.close(figure)

        sample_paths = []
        steps = np.arange(1, reference.shape[2] + 1)
        for row, sample in enumerate(report["sample_results"]):
            index = sample["index"]
            figure, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
            try:
                figure.suptitle(
                    f"样本 {index}：相对当前观测 Link7 末端的动作\n"
                    f"{PRECISION_NOTE}", fontsize=16,
                )
                actions = {name: arrays[f"{name}_action"][row, 0]
                           for name in PLOT_STYLES}
                for axis, component, label in zip(axes.flat, (0, 1, 2, 9),
                                                 ("X 位置", "Y 位置", "Z 位置", "夹爪")):
                    for name, action in actions.items():
                        values = action[:, component] * (1000 if component < 3 else 1)
                        axis.plot(steps, values, **PLOT_STYLES[name], linewidth=1.8)
                    axis.set(title=f"{label}预测", ylabel=(
                        "相对位置 (mm)" if component < 3 else "归一化编码"))
                for precision in ENGINE_PRECISIONS:
                    name = f"tensorrt_{precision}"
                    errors = action_error_curves(actions[name], actions["pytorch_cuda"])
                    for axis, key, title, unit in (
                        (axes[2, 0], "position_mm", "相对 PyTorch 的位置误差", "距离差 (mm)"),
                        (axes[2, 1], "rotation_deg", "相对 PyTorch 的旋转误差", "角度差 (deg)"),
                    ):
                        values = errors[key]
                        style = dict(PLOT_STYLES[name])
                        style["label"] += f"（最大值 {values.max():.3e}）"
                        axis.plot(steps, values, **style, linewidth=1.8)
                        axis.set(title=title, ylabel=unit)
                for axis in axes.flat:
                    axis.set_xlabel("预测步")
                    axis.set_xticks(steps[::max(1, len(steps) // 8)])
                    axis.grid(alpha=0.2)
                    axis.legend(fontsize=9)
                for axis in axes[2]:
                    axis.set_ylim(bottom=0)
                    axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
                path = sample_dir / f"sample_{index:03d}.png"
                figure.savefig(path, dpi=200)
                sample_paths.append(path)
            finally:
                plt.close(figure)

    expected_names = {path.name for path in sample_paths}
    for path in sample_dir.iterdir():
        if (path.is_file() and re.fullmatch(r"sample_\d{3,}\.png", path.name)
                and path.name not in expected_names):
            path.unlink()
    return summary_path, sample_paths


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
    saved_arrays = {name: np.stack(value) for name, value in saved.items()}
    np.savez_compressed(report_dir / "precision_arrays.npz", **saved_arrays)
    summary_path, sample_paths = save_precision_plots(report, saved_arrays, report_dir)
    print(json.dumps({"status": report["status"], "samples": args.num_samples,
                      "report": str(report_dir / "precision_report.json"),
                      "summary_plot": str(summary_path),
                      "sample_plots_dir": str(sample_paths[0].parent)}, indent=2))


if __name__ == "__main__":
    main()
