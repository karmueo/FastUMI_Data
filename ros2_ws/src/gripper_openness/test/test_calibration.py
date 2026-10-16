"""测试相机标定解析、范围统计和夹爪 YAML 往返。"""

from pathlib import Path

from gripper_openness.calibration import (
    CalibrationQualityConfig,
    load_camera_calibration,
    load_gripper_calibration,
    RangeAccumulator,
    write_gripper_calibration,
)

import numpy as np
import pytest


def _write_camera(path: Path, model: str = "fisheye") -> None:
    """写入测试用相机标定。"""
    path.write_text(
        "cam0:\n"
        "  intrinsics: [400.0, 400.0, 320.0, 240.0]\n"
        f"  distortion_model: {model}\n"
        "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        "  resolution: [640, 480]\n",
        encoding="utf-8",
    )


def _corners(x: float) -> np.ndarray:
    """构造测试用方形标记角点。"""
    return np.asarray([[x, 100], [x + 20, 100], [x + 20, 120], [x, 120]], dtype=float)


def _add(accumulator: RangeAccumulator, distance: float, **metrics) -> None:
    """按距离生成左右码角点并加入一帧。"""
    accumulator.add(
        distance, _corners(100), _corners(100 + 2 * distance), (640, 480), **metrics
    )


def _stroke(accumulator: RangeAccumulator, low: float = 50.0, high: float = 100.0) -> None:
    """模拟闭合停留、往返过渡和全开停留的完整行程，带 ±0.3 mm 噪声。"""
    noise = (0.0, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.0)
    for value in noise:
        _add(accumulator, low + value, reprojection_error_px=0.3, relative_rotation_deg=1.0)
    for value in np.linspace(low + 5.0, high - 5.0, 6):
        _add(accumulator, float(value), reprojection_error_px=0.3, relative_rotation_deg=1.0)
    for value in noise:
        _add(accumulator, high + value, reprojection_error_px=0.3, relative_rotation_deg=1.0)


def test_camera_models_and_range_round_trip(tmp_path: Path) -> None:
    """验证鱼眼标定读取和范围、统计 YAML 原子往返。"""
    camera_path = tmp_path / "camera.yaml"
    _write_camera(camera_path)
    camera = load_camera_calibration(str(camera_path))
    assert camera.is_fisheye
    assert camera.resolution == (640, 480)

    accumulator = RangeAccumulator()
    _stroke(accumulator)
    calibration = accumulator.build(16.0, "DICT_4X4_50", 0, 1)
    output = tmp_path / "gripper.yaml"
    write_gripper_calibration(str(output), calibration)
    loaded = load_gripper_calibration(str(output))
    assert loaded.min_marker_dist_mm == pytest.approx(50.0)
    assert loaded.max_marker_dist_mm == pytest.approx(100.0)
    assert loaded.crop_reference["left_x"] == 100
    assert loaded.total_valid_frames == 22
    assert loaded.statistics["min_endpoint"]["frames"] == 8
    assert loaded.statistics["warnings"] == []
    with pytest.raises(FileExistsError):
        write_gripper_calibration(str(output), calibration)


def test_legacy_yaml_without_statistics_still_loads(tmp_path: Path) -> None:
    """旧版不含 calibration_statistics 的标定文件仍可读取。"""
    output = tmp_path / "legacy.yaml"
    output.write_text(
        "gripper_calibration:\n"
        "  marker_size_mm: 16.0\n"
        "  min_marker_dist_mm: 40.0\n"
        "  max_marker_dist_mm: 120.0\n"
        "  resolution: [640, 480]\n"
        "  crop_reference: {top_y: 10, bottom_y: 100, left_x: 10, right_x: 600}\n",
        encoding="utf-8",
    )
    assert load_gripper_calibration(str(output)).statistics is None


def test_isolated_extremes_do_not_move_endpoints() -> None:
    """单帧向外的离群距离不能拉偏端点，只体现在原始极值统计中。"""
    accumulator = RangeAccumulator()
    _add(accumulator, 44.0)
    _stroke(accumulator)
    _add(accumulator, 107.0)
    calibration = accumulator.build(16.0, "DICT_4X4_50", 0, 1)
    assert calibration.min_marker_dist_mm == pytest.approx(50.0)
    assert calibration.max_marker_dist_mm == pytest.approx(100.0)
    assert calibration.statistics["raw_min_mm"] == 44.0
    assert calibration.statistics["raw_max_mm"] == 107.0


def test_quality_gates_reject_bad_frames() -> None:
    """重投影误差、像素边长和两码相对姿态门限各自剔除坏帧并分别计数。"""
    accumulator = RangeAccumulator()
    _stroke(accumulator)
    # 闭合端外侧的连续坏帧本身足以形成假平台，必须被门限挡住。
    for _ in range(6):
        _add(accumulator, 40.0, reprojection_error_px=3.5, relative_rotation_deg=1.0)
    tiny_left = np.asarray([[100, 100], [105, 100], [105, 105], [100, 105]], dtype=float)
    for _ in range(6):
        accumulator.add(
            110.0, tiny_left, _corners(300), (640, 480),
            reprojection_error_px=0.3, relative_rotation_deg=1.0,
        )
    for _ in range(6):
        _add(accumulator, 115.0, reprojection_error_px=0.3, relative_rotation_deg=38.0)
    calibration = accumulator.build(16.0, "DICT_4X4_50", 0, 1)
    assert calibration.min_marker_dist_mm == pytest.approx(50.0)
    assert calibration.max_marker_dist_mm == pytest.approx(100.0)
    assert calibration.total_valid_frames == 22
    assert calibration.statistics["rejected_frames"] == {
        "reprojection_error": 6,
        "marker_side": 6,
        "relative_rotation": 6,
    }
    assert calibration.statistics["relative_rotation_median_deg"] == 1.0


def test_all_frames_rejected_reports_counts() -> None:
    """全部样本被门限剔除时报告各门限计数。"""
    accumulator = RangeAccumulator()
    for distance in (50.0, 100.0):
        _add(accumulator, distance, reprojection_error_px=9.0)
    with pytest.raises(ValueError, match="重投影误差 2"):
        accumulator.build(16.0, "DICT_4X4_50", 0, 1)


def test_range_rejects_missing_or_single_dwell() -> None:
    """没有停留段或只停留在一端时拒绝保存。"""
    accumulator = RangeAccumulator()
    for distance in np.linspace(50.0, 100.0, 20):
        _add(accumulator, float(distance))
    with pytest.raises(ValueError, match="停留段"):
        accumulator.build(16.0, "DICT_4X4_50", 0, 1)

    accumulator.reset()
    for _ in range(8):
        _add(accumulator, 50.0)
    with pytest.raises(ValueError, match="两个不同"):
        accumulator.build(16.0, "DICT_4X4_50", 0, 1)


def test_short_endpoint_dwell_is_rejected() -> None:
    """端点平台帧数少于 min_endpoint_frames 时给出具体端点。"""
    accumulator = RangeAccumulator(CalibrationQualityConfig(min_endpoint_frames=10))
    _stroke(accumulator)
    with pytest.raises(ValueError, match="闭合端停留帧 8 少于 10"):
        accumulator.build(16.0, "DICT_4X4_50", 0, 1)


def test_frame_gaps_break_dwell_windows() -> None:
    """源帧序号间隔过大的样本不能拼成停留窗口。"""
    accumulator = RangeAccumulator()
    for index in range(10):
        _add(accumulator, 50.0, frame_index=index * 10)
    for index in range(10):
        _add(accumulator, 100.0, frame_index=1000 + index * 10)
    with pytest.raises(ValueError, match="停留段"):
        accumulator.build(16.0, "DICT_4X4_50", 0, 1)


def test_pass_through_beyond_plateau_warns() -> None:
    """大量越过平台的过渡帧会在统计中给出未停留警告。"""
    accumulator = RangeAccumulator()
    _stroke(accumulator)
    for value in np.linspace(101.0, 118.0, 6):
        _add(accumulator, float(value), frame_index=200 + int(value) * 3)
    calibration = accumulator.build(16.0, "DICT_4X4_50", 0, 1)
    assert calibration.max_marker_dist_mm == pytest.approx(100.0)
    assert any("全开端可能未停留" in item for item in calibration.statistics["warnings"])


def test_static_noise_is_not_a_range() -> None:
    """夹爪静止时 ±1 mm 噪声不能被当作开闭两端。"""
    accumulator = RangeAccumulator()
    pattern = (125.0, 125.6, 126.2, 125.4, 124.6, 125.2, 126.0, 125.8)
    for repeat in range(10):
        for value in pattern:
            _add(accumulator, value + 0.02 * repeat)
    with pytest.raises(ValueError, match="两个不同"):
        accumulator.build(16.0, "DICT_4X4_50", 0, 1)


def test_short_stroke_is_rejected() -> None:
    """端点平台分离但行程小于 min_range_mm 时拒绝保存。"""
    accumulator = RangeAccumulator()
    _stroke(accumulator, low=50.0, high=54.0)
    with pytest.raises(ValueError, match="min_range_mm"):
        accumulator.build(16.0, "DICT_4X4_50", 0, 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_reprojection_error_px": 0.0},
        {"plateau_band_mm": float("nan")},
        {"plateau_window_frames": 1},
        {"min_endpoint_frames": 0},
        {"min_range_mm": -1.0},
    ],
)
def test_quality_config_rejects_invalid_values(overrides) -> None:
    """非正、非有限或过小的门限参数在构造时报错。"""
    with pytest.raises(ValueError):
        CalibrationQualityConfig(**overrides)
