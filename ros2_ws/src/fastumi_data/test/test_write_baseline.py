"""验证写入基准的纯逻辑限额、分位数与重叠汇总。"""

import pytest

from fastumi_data.write_baseline import (
    _overlap_report,
    percentile_summary,
    validate_settings,
)


def test_percentile_summary_interpolates_and_handles_empty_input() -> None:
    """统计结果按线性插值计算分位数，空序列返回空统计。"""
    result = percentile_summary([10, 20, 30, 40])

    assert result == {
        "count": 4,
        "mean_ns": 25.0,
        "p50_ns": 25.0,
        "p95_ns": 38.5,
        "p99_ns": 39.7,
        "max_ns": 40,
    }
    assert percentile_summary([])["max_ns"] is None


def test_limits_bound_duration_disk_and_frame_memory() -> None:
    """限额校验返回预算并拒绝不适用模式和超出上限的配置。"""
    settings = validate_settings(
        "queued", "raw", 10.0, 30.0, 640, 480, 512.0, 1024.0
    )

    assert settings["frame_limit"] == 300
    assert settings["message_limit"] == 600
    assert settings["max_input_bytes"] == 512 * 1024 * 1024
    disk_limited = validate_settings(
        "queued", "raw", 1.0, 1.0, 640, 480, 16.0, 8.0
    )
    assert disk_limited["max_input_bytes"] < 16 * 1024 * 1024
    with pytest.raises(ValueError, match="raw 不适用"):
        validate_settings(
            "continuous", "raw", 1.0, 30.0, 640, 480, 16.0, 16.0
        )
    with pytest.raises(ValueError, match="不超过 600"):
        validate_settings(
            "queued", "raw", 601.0, 30.0, 640, 480, 16.0, 1024.0
        )
    with pytest.raises(ValueError, match="磁盘上限不足"):
        validate_settings(
            "queued", "raw", 10.0, 30.0, 640, 480, 1.0, 1.0
        )


def test_long_write_overlap_counts_are_correlational_only() -> None:
    """重叠汇总报告时间相交数量，不推断磁盘故障原因。"""
    write = {"start_ns": 10, "end_ns": 40, "duration_ns": 30}
    callback = {"start_ns": 20, "end_ns": 25, "duration_ns": 5}
    report = _overlap_report(
        [write], [callback], [0, 10, 40, 70], 20, 4, 20
    )

    assert report["long_write_count"] == 1
    assert report["long_callback_count"] == 1
    assert report["long_heartbeat_gap_count"] == 2
    assert report["long_writes"] == [{
        "write_index": 0,
        "overlapping_long_callbacks": 1,
        "overlapping_long_image_callback_gaps": 0,
        "overlapping_long_heartbeats": 1,
    }]


def test_touching_write_and_heartbeat_windows_do_not_overlap() -> None:
    """仅在两个时间窗口有正长度交集时才统计重叠。"""
    report = _overlap_report(
        [{"start_ns": 10, "end_ns": 40, "duration_ns": 30}],
        [], [40, 70], 20, 4, 20,
    )

    assert report["long_writes"][0]["overlapping_long_heartbeats"] == 0
