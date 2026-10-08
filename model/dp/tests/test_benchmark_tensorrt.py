"""Check cached benchmark plotting without model loading or GPU timing."""

from copy import deepcopy

import numpy as np
import pytest

from benchmark_tensorrt import save_benchmark_plots


def example_report(single_iteration=False):
    """Construct distinct component timings, including a backend slower than baseline."""
    backends = ("pytorch_cuda_fp32", "tensorrt_fp32", "tensorrt_fp16")
    components = ("encoder_gpu", "denoiser_gpu", "full_gpu_resident", "full_end_to_end")
    latencies = {}
    for backend, factor in zip(backends, (1, 0.5, 2)):
        latencies[backend] = {}
        for component, mean in zip(components, (12, 8, 40, 50)):
            mean *= factor
            latencies[backend][component] = {
                "mean_ms": mean,
                "median_ms": mean if single_iteration else mean * 0.9,
                "p90_ms": mean if single_iteration else mean * 1.1,
                "p95_ms": mean if single_iteration else mean * 1.2,
                "calls_per_second": 1000 / mean,
            }
            if component.startswith("full_"):
                latencies[backend][component]["predictions_per_second"] = 1000 / mean
    speedups = {
        backend: {component: {
            "vs_pytorch_cuda_fp32": latencies[backends[0]][component]["mean_ms"]
            / latencies[backend][component]["mean_ms"],
            "vs_tensorrt_fp32": latencies[backends[1]][component]["mean_ms"]
            / latencies[backend][component]["mean_ms"],
        } for component in components} for backend in backends[1:]
    }
    return {
        "gpu": "test GPU", "torch": "2.8.0", "cuda": "12.6", "tensorrt": "10.3",
        "warmup": 0 if single_iteration else 20,
        "iterations": 1 if single_iteration else 100,
        "validation_samples_rotated": 8, "num_inference_steps": 16,
        "tf32_enabled": False, "latencies": latencies, "speedups": speedups,
    }


@pytest.mark.parametrize("single_iteration", [False, True])
def test_cached_plots_preserve_statistics_and_baselines(tmp_path, monkeypatch, single_iteration):
    """Rendered bars use the correct report fields and preserve slowdown values below 1x."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from PIL import Image

    report = example_report(single_iteration)
    original = deepcopy(report)
    raw = tmp_path / "benchmark_report.json"
    raw.write_bytes(b"existing report")
    captured = []
    savefig = Figure.savefig

    def inspect_and_save(figure, path, **kwargs):
        captured.append(figure)
        savefig(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_and_save)
    summary, statistics = save_benchmark_plots(report, tmp_path)
    assert summary.name == "benchmark_summary.png"
    assert statistics.name == "benchmark_latency_stats.png"
    for path in (summary, statistics):
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert min(image.size) >= 2000
    assert report == original
    assert raw.read_bytes() == b"existing report"
    assert plt.get_fignums() == []

    backends = ("pytorch_cuda_fp32", "tensorrt_fp32", "tensorrt_fp16")
    components = ("encoder_gpu", "denoiser_gpu", "full_gpu_resident", "full_end_to_end")
    summary_axes = captured[0].axes
    assert len(summary_axes) == 6
    for axis, component in zip(summary_axes, components):
        expected = [report["latencies"][backend][component]["mean_ms"] for backend in backends]
        assert [bar.get_height() for bar in axis.patches] == pytest.approx(expected)
        assert [label.get_text() for label in axis.texts] == [f"{value:.2f}" for value in expected]
        assert axis.get_ylabel() == "延迟 (ms)"
    throughput = summary_axes[4]
    assert [bar.get_height() for bar in throughput.patches] == pytest.approx([
        report["latencies"][backend]["full_end_to_end"]["predictions_per_second"]
        for backend in backends])
    assert throughput.get_ylabel() == "完整预测次数（次/s）"
    speedups = summary_axes[5]
    expected = [report["speedups"][backend]["full_end_to_end"][baseline]
                for backend in backends[1:]
                for baseline in ("vs_pytorch_cuda_fp32", "vs_tensorrt_fp32")]
    assert [bar.get_height() for bar in speedups.patches] == pytest.approx(expected)
    assert any(value < 1 for value in expected)
    np.testing.assert_array_equal(speedups.lines[0].get_ydata(), [1, 1])
    for axis in summary_axes:
        assert axis.get_ylim()[0] == 0
        assert axis.get_ylim()[1] > max(bar.get_height() for bar in axis.patches)
        assert axis.get_yscale() == "linear"

    statistics_axes = captured[1].axes
    assert len(statistics_axes) == 4
    for axis, component in zip(statistics_axes, components):
        expected = [report["latencies"][backend][component][key]
                    for backend in backends
                    for key in ("mean_ms", "median_ms", "p90_ms", "p95_ms")]
        assert [bar.get_height() for bar in axis.patches] == pytest.approx(expected)
        assert [label.get_text() for label in axis.get_xticklabels()] == [
            "平均值", "中位数", "P90", "P95"]
        assert axis.get_ylabel() == "延迟 (ms)"
        assert axis.get_ylim()[0] == 0
        assert axis.get_ylim()[1] > max(bar.get_height() for bar in axis.patches)
        assert axis.get_yscale() == "linear"
    for figure in captured:
        footer = figure.texts[-1].get_text()
        assert "FP32 回退去噪器" in footer
        assert "CUDA events" in footer
        assert "输入上传和动作下载" in footer
        assert "每秒完整预测次数" in footer
        assert f"每项测速：{report['iterations']} 次" in footer


def test_plot_closes_figure_on_save_failure(tmp_path, monkeypatch):
    """A filesystem error must not leave a Matplotlib figure open."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    def fail_save(*args, **kwargs):
        raise OSError("cannot save PNG")

    monkeypatch.setattr(Figure, "savefig", fail_save)
    with pytest.raises(OSError, match="cannot save PNG"):
        save_benchmark_plots(example_report(), tmp_path)
    assert plt.get_fignums() == []
