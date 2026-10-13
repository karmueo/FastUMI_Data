"""验证 30 Hz Link7 选帧：同频逐帧、丢帧提示、间隔异常与高帧率抽帧路径。"""

from pathlib import Path

import numpy as np
import pytest

from fastumi_data.frame_selection import (
    FRAME_DROP,
    FRAME_INTERVAL_ANOMALY,
    FRAME_RATE_DEVIATION,
    FRAME_REUSE,
    HIGH_RATE,
    LOW_EFFECTIVE_RATE,
    MATCH_ERROR_EXCEEDED,
    SAME_RATE,
    TIMESTAMP_NOT_INCREASING,
    FrameSelectionConfig,
    fit_frame_rate,
    nominal_offset_ns,
    select_frames,
    select_high_rate,
    select_same_rate,
    selection_path_for,
    windows_crossing_drops,
)


FIXTURES = Path(__file__).parent / "fixtures"
START_NS = 1_791_787_535_000_000_000


def regular(rate_hz: float, seconds: float, start_ns: int = START_NS) -> np.ndarray:
    """生成指定速率的理想图像时间戳。"""
    count = int(round(rate_hz * seconds)) + 1
    return start_ns + np.round(np.arange(count) * 1.0e9 / rate_hz).astype(np.int64)


def codes(items) -> set:
    """返回问题代码集合。"""
    return {item["code"] for item in items}


@pytest.mark.parametrize("fps", [30, 30.0, 29.97, 30.3, 29.7])
def test_same_rate_path_covers_thirty_hz_and_declared_external_rates(fps) -> None:
    """标称 30、29.97 及 ±1% 边界都进入同频逐帧路径。"""
    assert selection_path_for(fps) == SAME_RATE


@pytest.mark.parametrize("fps", [60, 90, 120.0])
def test_high_rate_path_requires_at_least_sixty_hz(fps) -> None:
    """标称 60、90 Hz 进入高帧率抽帧路径。"""
    assert selection_path_for(fps) == HIGH_RATE


@pytest.mark.parametrize("fps", [25, 29.6, 30.31, 45, 59.9, 0, -30, float("nan"), float("inf")])
def test_unsupported_nominal_rates_are_configuration_errors(fps) -> None:
    """不属于两条路径或非有限正数的标称值在转换前报错。"""
    with pytest.raises(ValueError):
        selection_path_for(fps)


def test_ideal_thirty_hz_outputs_every_frame_with_nominal_times() -> None:
    """理想 30 Hz：源索引连续、名义时间为 n/30 且按整数纳秒舍入。"""
    times = regular(30, 5)
    selection = select_same_rate(times, 30)
    assert selection.accepted
    np.testing.assert_array_equal(selection.positions, np.arange(len(times)))
    np.testing.assert_array_equal(selection.frame_numbers, np.arange(len(times)))
    nominal = selection.nominal_timestamp_ns
    assert nominal[:4].tolist() == [0, 33_333_333, 66_666_667, 100_000_000]
    assert np.all(np.abs(np.diff(nominal) - 1e9 / 30) <= 1)
    assert not selection.dropped_frames_before.any()


@pytest.mark.parametrize("actual_rate", [30.0616, 29.97])
def test_rate_offset_sequences_longer_than_sixty_seconds_are_never_skipped(actual_rate) -> None:
    """30.0616 与 29.97 Hz 实际速率持续 70 s：不跳帧、不复用、不拒绝，并报告名义累积差。"""
    times = regular(actual_rate, 70)
    selection = select_same_rate(times, 30)
    assert selection.accepted, selection.rejections
    assert len(selection) == len(times)
    np.testing.assert_array_equal(np.diff(selection.positions), 1)
    assert not selection.dropped_frames_before.any()
    assert selection.stats["fitted_frame_rate_hz"] == pytest.approx(actual_rate, rel=1e-6)
    accumulated_ms = (selection.nominal_timestamp_ns[-1] - (times[-1] - times[0])) / 1e6
    expected_ms = (len(times) - 1) * (1000 / 30 - 1000 / actual_rate)
    assert accumulated_ms == pytest.approx(expected_ms, abs=1e-3)
    assert abs(accumulated_ms) > 10


def test_jitter_within_tolerance_is_normal() -> None:
    """±8 ms 以内的相邻间隔抖动全部判定为正常帧。"""
    rng = np.random.default_rng(3)
    times = regular(30, 20) + rng.integers(-3_900_000, 3_900_000, size=601)
    selection = select_same_rate(times, 30)
    assert selection.accepted
    assert not selection.warnings
    assert selection.stats["interval_deviation_ms"]["max"] < 8


@pytest.mark.parametrize("missing", [[100], [100, 101, 102], [50, 300, 301]])
def test_single_and_consecutive_drops_warn_and_keep_episode(missing) -> None:
    """单次与连续丢帧产生 FRAME_DROP 提示、正确的丢帧计数与名义时间跳变。"""
    full = regular(30, 20)
    times = np.delete(full, missing)
    selection = select_same_rate(times, 30)
    assert selection.accepted
    assert FRAME_DROP in codes(selection.warnings)
    dropped = selection.dropped_frames_before
    assert int(dropped.sum()) == len(missing)
    np.testing.assert_array_equal(selection.frame_numbers, np.delete(np.arange(len(full)), missing))
    jumps = np.diff(selection.nominal_timestamp_ns)
    np.testing.assert_allclose(jumps, (1 + dropped[1:]) * 1e9 / 30, atol=1)
    assert selection.stats["fitted_frame_rate_hz"] == pytest.approx(30, rel=1e-9)


def test_interval_outside_integer_multiple_tolerance_rejects() -> None:
    """间隔偏离整数倍周期超过容差（约 1.5 个周期）时拒绝整条 episode。"""
    times = regular(30, 5)
    times[40:] += 17_000_000
    selection = select_same_rate(times, 30)
    assert not selection.accepted
    assert FRAME_INTERVAL_ANOMALY in codes(selection.rejections)


@pytest.mark.parametrize("change", ["duplicate", "backwards"])
def test_nonmonotonic_or_duplicate_timestamps_reject(change) -> None:
    """重复或倒退的时间戳无法归类，拒绝整条 episode。"""
    times = regular(30, 5)
    times[20] = times[19] if change == "duplicate" else times[19] - 1_000_000
    selection = select_same_rate(times, 30)
    assert FRAME_INTERVAL_ANOMALY in codes(selection.rejections)


def test_rate_deviation_beyond_half_percent_rejects() -> None:
    """实际速率比标称快 1%（声明错误）时拟合偏差超限被拒绝。"""
    selection = select_same_rate(regular(30.3, 20), 30)
    assert FRAME_RATE_DEVIATION in codes(selection.rejections)
    assert select_same_rate(regular(30.12, 20), 30).accepted


def test_many_drops_below_ninety_five_percent_effective_rate_reject() -> None:
    """丢帧使有效帧率低于标称值 95% 时拒绝。"""
    full = regular(30, 20)
    times = np.delete(full, np.arange(5, len(full) - 5, 10))
    selection = select_same_rate(times, 30)
    assert LOW_EFFECTIVE_RATE in codes(selection.rejections)
    assert selection.stats["effective_rate_threshold_hz"] == pytest.approx(28.5)


def test_subset_rebases_nominal_time_and_first_drop_count() -> None:
    """裁剪后以新首帧为零点，首帧丢帧计数为 0。"""
    times = np.delete(regular(30, 5), [10])
    subset = select_same_rate(times, 30).subset(10, 30)
    assert subset.nominal_timestamp_ns[0] == 0
    assert subset.dropped_frames_before[0] == 0


def test_windows_crossing_drops_counts_affected_training_windows() -> None:
    """统计跨越丢帧点的训练窗口数。"""
    dropped = np.zeros(40, dtype=np.int32)
    dropped[20] = 1
    assert windows_crossing_drops(dropped, 17) == 16
    assert windows_crossing_drops(np.zeros(40, dtype=np.int32), 17) == 0


def test_fit_frame_rate_uses_estimated_frame_numbers() -> None:
    """拟合帧率以估计帧号为横轴，丢帧不引入偏差。"""
    times = regular(30.06, 10)
    numbers = np.arange(len(times))
    assert fit_frame_rate(np.delete(times, [5, 6]), np.delete(numbers, [5, 6])) == pytest.approx(
        30.06, rel=1e-6
    )


@pytest.mark.parametrize("fps", [90, 60])
def test_high_rate_grid_selects_strictly_increasing_frames(fps) -> None:
    """高帧率路径：精确 30 Hz 网格，零复用，匹配误差在半源周期内。"""
    times = regular(fps * 1.002, 10)
    selection = select_high_rate(times, fps)
    assert selection.accepted, selection.rejections
    assert np.all(np.diff(selection.positions) > 0)
    assert np.all(np.abs(selection.match_error_ns) <= 0.5e9 / fps + 2e6)
    assert not selection.dropped_frames_before.any()
    nominal = selection.nominal_timestamp_ns
    assert set(np.diff(nominal).tolist()) <= {33_333_333, 33_333_334}


@pytest.mark.parametrize("fps", [60, 90, 120])
@pytest.mark.parametrize("last_grid", [15, 16, 17])
@pytest.mark.parametrize("end_delta_ns", [-1, 0, 1])
def test_high_rate_grid_keeps_all_points_within_rounded_endpoint(
    fps, last_grid, end_delta_ns,
) -> None:
    """整秒、向下/向上舍入的端点及其 ±1 ns 边界均准确保留范围内网格点。"""
    count = last_grid * (fps // 30) + 1
    times = START_NS + np.round(np.arange(count) * 1e9 / fps).astype(np.int64)
    times[-1] += end_delta_ns
    selection = select_high_rate(times, fps)
    assert selection.accepted, selection.rejections
    expected_count = last_grid + (end_delta_ns >= 0)
    np.testing.assert_array_equal(selection.frame_numbers, np.arange(expected_count))
    np.testing.assert_array_equal(
        selection.positions, np.arange(expected_count) * (fps // 30),
    )
    assert START_NS + selection.nominal_timestamp_ns[-1] <= times[-1]


def test_high_rate_jitter_and_isolated_drops_pass() -> None:
    """90 Hz 带抖动与单个丢帧时仍能在门限内选帧。"""
    rng = np.random.default_rng(7)
    times = regular(90, 10) + rng.integers(-1_500_000, 1_500_000, size=901)
    times = np.delete(times, [100, 400])
    assert select_high_rate(times, 90).accepted


def test_high_rate_local_consecutive_drops_exceed_match_error() -> None:
    """局部连续丢帧导致匹配误差超过半源周期加裕量时拒绝。"""
    times = np.delete(regular(90, 10), np.arange(300, 310))
    selection = select_high_rate(times, 90)
    assert MATCH_ERROR_EXCEEDED in codes(selection.rejections)


def test_high_rate_reuse_is_rejected() -> None:
    """实际源速率接近 30 Hz 却声明 60 Hz 时出现复用或超限，均拒绝。"""
    selection = select_high_rate(regular(31, 10), 60)
    assert codes(selection.rejections) & {FRAME_REUSE, MATCH_ERROR_EXCEEDED, LOW_EFFECTIVE_RATE}


def test_high_rate_match_error_boundary() -> None:
    """匹配误差门限为 0.5/f_source + 2 ms：边界内通过，超出拒绝。"""
    base = regular(90, 3)
    # 90 Hz 门限约 7.556 ms；网格对齐的源帧序号为 3k，相邻帧距网格 ±11.1 ms。
    aligned = np.arange(3, len(base) - 1, 3)
    inside = base.copy()
    inside[aligned] += 7_000_000
    selection = select_high_rate(inside, 90)
    assert selection.accepted, selection.rejections
    assert np.max(np.abs(selection.match_error_ns)) == 7_000_000
    outside = base.copy()
    outside[aligned] += 8_000_000
    assert MATCH_ERROR_EXCEEDED in codes(select_high_rate(outside, 90).rejections)
    # 网格对齐帧单独丢失同样超出门限，而不是改用较远的相邻帧。
    dropped = np.delete(base, [30])
    assert MATCH_ERROR_EXCEEDED in codes(select_high_rate(dropped, 90).rejections)


def test_high_rate_low_effective_rate_rejects() -> None:
    """高帧率源有效帧率低于标称 95% 时拒绝。"""
    times = regular(80, 10)
    assert LOW_EFFECTIVE_RATE in codes(select_high_rate(times, 90).rejections)


def test_high_rate_requires_strictly_increasing_timestamps() -> None:
    """高帧率路径遇到倒序时间直接拒绝。"""
    times = regular(90, 2)
    times[5] = times[4]
    assert TIMESTAMP_NOT_INCREASING in codes(select_high_rate(times, 90).rejections)


def test_select_frames_dispatches_by_nominal_rate() -> None:
    """按标称值而非实测频率分派路径。"""
    assert select_frames(regular(30, 2), 30).path == SAME_RATE
    assert select_frames(regular(90, 2), 90).path == HIGH_RATE
    with pytest.raises(ValueError):
        select_frames(regular(30, 2), 25)


def test_nominal_offsets_use_exact_integer_rounding() -> None:
    """名义纳秒偏移用整数有理运算，不累积舍入误差。"""
    offsets = nominal_offset_ns(np.arange(30 * 3600 + 1))
    assert offsets[-1] == 3600 * 1_000_000_000
    assert set(np.diff(offsets).tolist()) == {33_333_333, 33_333_334}


def test_configuration_thresholds_must_be_positive() -> None:
    """门限必须为有限正数。"""
    with pytest.raises(ValueError):
        FrameSelectionConfig(interval_tolerance_s=0).validate()
    with pytest.raises(ValueError):
        FrameSelectionConfig(min_effective_rate_ratio=1.5).validate()


@pytest.mark.parametrize("name", ["tag_jpeg_1", "tag_jpeg_2"])
def test_recorded_jpeg_timestamps_pass_same_rate_regression(name) -> None:
    """两条实际 JPEG 记录的图像时间戳在 30 Hz 同频路径全程通过且无丢帧。"""
    path = FIXTURES / f"{name}_image_stamps.txt"
    times = np.loadtxt(path, dtype=np.int64)
    selection = select_same_rate(times, 30)
    assert selection.accepted, selection.rejections
    assert len(selection) == len(times)
    assert not selection.warnings
    assert selection.stats["fitted_rate_deviation_ratio"] == pytest.approx(0.00205, abs=2e-4)
    interval = selection.stats["interval_ms"]
    assert 31.0 < interval["min"] and interval["max"] < 37.0
