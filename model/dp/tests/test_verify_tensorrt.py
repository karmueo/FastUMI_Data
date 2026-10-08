"""Check physical error curves and headless plots without running GPU inference."""

import numpy as np
import pytest

from tensorrt_link7 import physical_errors
from verify_tensorrt import action_error_curves, save_precision_plots


def example_outputs(num_samples=1, fp32_offset=0.0, fp16_offset=0.0):
    """Build valid row-major pose10 actions with a three-step prediction horizon."""
    reference = np.zeros((num_samples, 1, 3, 10), dtype=np.float32)
    reference[..., 3:9] = [1, 0, 0, 0, 1, 0]
    reference[..., 0] = np.array([0.001, 0.002, 0.003])
    reference[..., 9] = np.array([0.1, 0.2, 0.3])
    arrays = {"pytorch_cuda_action": reference}
    maxima = {}
    for precision, offset in (("fp32", fp32_offset), ("fp16", fp16_offset)):
        action = reference.copy()
        action[..., 1] += offset
        arrays[f"tensorrt_{precision}_action"] = action
        maxima[precision] = {
            **physical_errors(action, reference),
            "encoder_max_abs": offset,
            "isolated_denoiser_max_abs": offset,
        }
    report = {
        "num_samples": num_samples,
        "num_inference_steps": 16,
        "status": "finite",
        "checkpoint_sha256": "a" * 64,
        "sample_results": [{"index": index} for index in range(num_samples)],
        "maxima_across_samples": maxima,
    }
    return report, arrays


def test_action_error_curves_preserve_units_and_step_order():
    """Millimetre translations and a quarter turn match report statistics."""
    _, arrays = example_outputs()
    reference = arrays["pytorch_cuda_action"][0, 0]
    actual = reference.copy()
    actual[:, 1] = [0.001, 0.003, 0.002]
    actual[1, 3:9] = [0, -1, 0, 1, 0, 0]
    actual[:, 9] += [0.01, 0.03, 0.02]
    curves = action_error_curves(actual, reference)
    np.testing.assert_allclose(curves["position_mm"], [1, 3, 2])
    np.testing.assert_allclose(curves["rotation_deg"], [0, 90, 0], atol=1e-10)
    np.testing.assert_allclose(curves["gripper"], [0.01, 0.03, 0.02], rtol=1e-6)
    statistics = physical_errors(actual, reference)
    for key, suffix in (("position_mm", "position_mm"),
                        ("rotation_deg", "rotation_deg"), ("gripper", "gripper")):
        assert curves[key].max() == pytest.approx(statistics[f"max_{suffix}"])
        assert curves[key].mean() == pytest.approx(statistics[f"mean_{suffix}"])
    for curve in action_error_curves(reference, reference).values():
        np.testing.assert_array_equal(curve, np.zeros(3))
    with pytest.raises(ValueError, match="pose10"):
        action_error_curves(reference[None], reference[None])
    invalid = actual.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        action_error_curves(invalid, reference)


@pytest.mark.parametrize("fp32_offset,fp16_offset", [(0, 0), (0, 0.03), (1e-12, 0.03)])
def test_headless_single_sample_plots(tmp_path, monkeypatch, fp32_offset, fp16_offset):
    """Plots retain zero errors, precise summary values, and the original horizon."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from PIL import Image

    report, arrays = example_outputs(fp32_offset=fp32_offset, fp16_offset=fp16_offset)
    savefig = Figure.savefig
    captured = []

    def inspect_and_save(figure, path, **kwargs):
        captured.append(figure)
        savefig(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_and_save)
    summary, samples = save_precision_plots(report, arrays, tmp_path)
    assert summary.name == "precision_summary.png"
    assert [path.name for path in samples] == ["sample_000.png"]
    for path in [summary, *samples]:
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert min(image.size) >= 1800
    assert plt.get_fignums() == []
    summary_axes = captured[0].axes
    for axis, key in zip(summary_axes, ("max_position_mm", "max_rotation_deg", "max_gripper",
                                       "encoder_max_abs", "isolated_denoiser_max_abs")):
        values = [report["maxima_across_samples"][p][key] for p in ("fp32", "fp16")]
        assert [bar.get_height() for bar in axis.patches] == pytest.approx(values)
        assert [label.get_text() for label in axis.texts] == [f"{v:.3e}" for v in values]
        assert axis.get_yscale() == ("symlog" if any(values) else "linear")
    assert "未设置验收阈值" in summary_axes[5].texts[0].get_text()
    assert "FP32 回退去噪器" in summary_axes[5].texts[0].get_text()
    sample_axes = captured[1].axes
    assert "相对当前观测 Link7 末端" in captured[1]._suptitle.get_text()
    for axis in sample_axes:
        assert axis.get_xlabel() == "预测步"
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), [1, 2, 3])
    for axis, component in zip(sample_axes, (0, 1, 2, 9)):
        expected = arrays["pytorch_cuda_action"][0, 0, :, component]
        np.testing.assert_allclose(
            axis.lines[0].get_ydata(), expected * (1000 if component < 3 else 1))
        assert len(axis.lines) == 3
    for axis, key in zip(sample_axes[4:], ("position_mm", "rotation_deg")):
        for line, precision in zip(axis.lines, ("fp32", "fp16")):
            errors = action_error_curves(arrays[f"tensorrt_{precision}_action"][0, 0],
                                         arrays["pytorch_cuda_action"][0, 0])
            np.testing.assert_allclose(line.get_ydata(), errors[key])


def test_rerun_cleans_only_stale_generated_sample_images(tmp_path):
    """Reducing the sample count preserves raw outputs and unrelated files."""
    report, arrays = example_outputs(num_samples=2)
    _, samples = save_precision_plots(report, arrays, tmp_path)
    assert len(samples) == 2
    preserved = [tmp_path / "precision_report.json", tmp_path / "precision_arrays.npz",
                 tmp_path / "action_samples" / "notes.png",
                 tmp_path / "action_samples" / "sample_custom.png"]
    for path in preserved:
        path.write_bytes(b"keep")
    report, arrays = example_outputs()
    _, samples = save_precision_plots(report, arrays, tmp_path)
    assert len(samples) == 1
    assert not (tmp_path / "action_samples" / "sample_001.png").exists()
    assert all(path.read_bytes() == b"keep" for path in preserved)
