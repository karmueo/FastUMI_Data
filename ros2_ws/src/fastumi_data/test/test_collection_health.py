"""验证采集健康监控：新鲜度、有效性、匹配窗口、乱序和夹爪滞后。"""

import math
from typing import Dict

from fastumi_data.collection_health import (
    GRIPPER,
    IMAGE,
    TRACKER,
    CollectionHealthMonitor,
    HealthConfig,
    preflight_issues,
)


# 毫秒到纳秒的换算。
MS = 1_000_000
# 基准源时间戳。
T0 = 1_700_000_000_000_000_000
ALL_FOUND = {IMAGE: True, TRACKER: True, GRIPPER: True}


class _Clock:
    """同时驱动单调时钟和墙钟的可控时钟。"""

    def __init__(self) -> None:
        """单调时钟从 0 开始，墙钟与源时间戳同域。"""
        self.mono = 0
        self.wall = T0

    def advance(self, milliseconds: float) -> None:
        """两个时钟同步前进。"""
        self.mono += int(milliseconds * MS)
        self.wall += int(milliseconds * MS)


def _monitor(clock: _Clock, **overrides: float) -> CollectionHealthMonitor:
    """创建使用可控时钟的监控器。"""
    return CollectionHealthMonitor(
        HealthConfig(**overrides), lambda: clock.mono, lambda: clock.wall
    )


def _feed(
    monitor: CollectionHealthMonitor,
    clock: _Clock,
    frames: int,
    tracker_offset_ms: float = 5.0,
    gripper: bool = True,
    tracker: bool = True,
    gripper_valid: bool = True,
    openness: float = 0.5,
) -> None:
    """以 30 Hz 图像、100 Hz Tracker 的节奏喂入数据。"""
    monitor.observe_tracker_status(True, True, 3)
    for _ in range(frames):
        source = clock.wall
        monitor.observe_image(source, True)
        if tracker:
            monitor.observe_tracker_pose(
                source + int(tracker_offset_ms * MS), True
            )
        clock.advance(2)
        if gripper:
            monitor.observe_gripper(source, gripper_valid, openness)
        clock.advance(31)
        monitor.observe_tracker_status(True, True, 3)


def _issue_codes(monitor: CollectionHealthMonitor, config: HealthConfig) -> Dict:
    """返回预检阻塞码集合。"""
    snapshot = monitor.snapshot(ALL_FOUND)
    return {issue.code for issue in preflight_issues(snapshot, config)}


def test_healthy_streams_report_rates_and_pair_metrics() -> None:
    """验证健康输入的新鲜度、频率、配对源时间差和到达时间差。"""
    clock = _Clock()
    monitor = _monitor(clock)
    _feed(monitor, clock, 90)

    snapshot = monitor.snapshot(ALL_FOUND)

    for stream in (snapshot.image, snapshot.tracker, snapshot.gripper):
        assert stream.discovered and stream.fresh and stream.valid
    assert snapshot.image.matched and snapshot.tracker.matched
    assert snapshot.gripper.matched
    assert 25 < snapshot.image.rate_hz < 35
    assert snapshot.gripper.source_delta_ms == 0.0
    assert snapshot.gripper.pair_arrival_delta_ms == 2.0
    assert math.isclose(snapshot.tracker.source_delta_ms, 5.0)
    assert snapshot.tracker.pair_arrival_delta_ms == 0.0
    assert math.isnan(snapshot.image.source_delta_ms)
    assert snapshot.alarms == ()
    assert snapshot.gripper_valid and snapshot.gripper_percent == 50.0
    assert snapshot.tracker_tracking_ok


def test_stationary_but_publishing_tracker_stays_healthy() -> None:
    """验证只要持续发布，位姿不变的 Tracker 不被判为失效。"""
    clock = _Clock()
    monitor = _monitor(clock)
    for _ in range(60):
        source = clock.wall
        monitor.observe_image(source, True)
        monitor.observe_tracker_pose(source, True)  # 位姿内容恒定不影响健康。
        monitor.observe_tracker_status(True, True, 3)
        monitor.observe_gripper(source, True, 0.4)
        clock.advance(33)

    snapshot = monitor.snapshot(ALL_FOUND)

    assert snapshot.tracker.fresh and snapshot.tracker.valid
    assert "TRACKER_STALE" not in snapshot.alarms


def test_missing_publishers_block_start_per_stream() -> None:
    """验证缺失节点（无发布者）在预检中按路给出原因。"""
    clock = _Clock()
    monitor = _monitor(clock)
    _feed(monitor, clock, 30)

    snapshot = monitor.snapshot({IMAGE: True, TRACKER: False, GRIPPER: False})
    codes = {i.code for i in preflight_issues(snapshot, monitor.config)}

    assert codes == {"TRACKER_NOT_DISCOVERED", "GRIPPER_NOT_DISCOVERED"}
    assert "TRACKER_NOT_DISCOVERED" in snapshot.alarms


def test_stream_that_stops_becomes_stale_after_its_own_timeout() -> None:
    """验证断流后各路按各自超时判定过期：Tracker 0.25 s，图像 1 s。"""
    clock = _Clock()
    monitor = _monitor(clock)
    _feed(monitor, clock, 30)

    clock.advance(300)
    mid = monitor.snapshot(ALL_FOUND)
    clock.advance(800)
    late = monitor.snapshot(ALL_FOUND)

    assert not mid.tracker.fresh and mid.image.fresh
    assert "TRACKER_STALE" in mid.alarms and "IMAGE_STALE" not in mid.alarms
    assert not late.image.fresh and "IMAGE_STALE" in late.alarms
    assert late.image.age_s > 1.0


def test_never_received_stream_is_stale_with_unknown_age() -> None:
    """验证从未收到数据的输入不新鲜，年龄为 -1，时间指标不可用。"""
    clock = _Clock()
    monitor = _monitor(clock)

    snapshot = monitor.snapshot(ALL_FOUND)

    assert not snapshot.image.fresh and snapshot.image.age_s == -1.0
    assert math.isnan(snapshot.image.message_age_ms)
    assert math.isnan(snapshot.image.rate_hz)
    assert {"IMAGE_STALE", "TRACKER_STALE", "GRIPPER_STALE"} <= set(snapshot.alarms)


def test_invalid_tracking_state_makes_tracker_invalid() -> None:
    """验证 Tracker 位姿无效或跟踪状态非 RUNNING_OK 时不可采集。"""
    clock = _Clock()
    monitor = _monitor(clock)
    _feed(monitor, clock, 30)
    monitor.observe_tracker_status(True, True, 4)  # RUNNING_OUT_OF_RANGE

    snapshot = monitor.snapshot(ALL_FOUND)
    codes = _issue_codes(monitor, monitor.config)

    assert snapshot.tracker.fresh and not snapshot.tracker.valid
    assert not snapshot.tracker_tracking_ok
    assert snapshot.tracker_tracking_state == 4
    assert "TRACKER_INVALID" in snapshot.alarms and "TRACKER_INVALID" in codes
    assert "RUNNING_OK" in snapshot.tracker.detail

    monitor.observe_tracker_status(False, True, 3)
    assert "未连接" in monitor.snapshot(ALL_FOUND).tracker.detail
    monitor.observe_tracker_status(True, False, 3)
    assert "位姿无效" in monitor.snapshot(ALL_FOUND).tracker.detail


def test_tracker_status_that_stops_makes_tracker_invalid() -> None:
    """验证状态消息过期后不能再把 Tracker 视为有效。"""
    clock = _Clock()
    monitor = _monitor(clock)
    _feed(monitor, clock, 10)
    monitor.observe_tracker_status(True, True, 3)
    for _ in range(20):  # 位姿继续发布，但状态停止。
        monitor.observe_tracker_pose(clock.wall, True)
        clock.advance(33)

    snapshot = monitor.snapshot(ALL_FOUND)

    assert snapshot.tracker.fresh and not snapshot.tracker.valid
    assert "状态" in snapshot.tracker.detail


def test_nan_gripper_is_invalid_and_percent_unavailable() -> None:
    """验证夹爪 NaN 或 valid=false 时无效，百分比为 NaN。"""
    clock = _Clock()
    monitor = _monitor(clock)
    _feed(monitor, clock, 30, openness=math.nan)

    snapshot = monitor.snapshot(ALL_FOUND)

    assert snapshot.gripper.fresh and not snapshot.gripper.valid
    assert not snapshot.gripper_valid and math.isnan(snapshot.gripper_percent)
    assert "GRIPPER_INVALID" in snapshot.alarms

    _feed(monitor, clock, 30, gripper_valid=False, openness=0.5)
    assert not monitor.snapshot(ALL_FOUND).gripper.valid


def test_out_of_order_stamps_alarm_then_clear() -> None:
    """验证源时间戳回退触发乱序报警并在保持期后清除。"""
    clock = _Clock()
    monitor = _monitor(clock)
    _feed(monitor, clock, 30)
    monitor.observe_image(clock.wall - 500 * MS, True)

    assert "IMAGE_OUT_OF_ORDER" in monitor.snapshot(ALL_FOUND).alarms
    assert monitor.snapshot(ALL_FOUND).image.out_of_order == 1

    _feed(monitor, clock, 100)
    assert "IMAGE_OUT_OF_ORDER" not in monitor.snapshot(ALL_FOUND).alarms


def test_gripper_without_matching_image_stamp_is_unmatched() -> None:
    """验证夹爪源时间戳与任何图像都不相同时标记匹配缺失且指标不可用。"""
    clock = _Clock()
    monitor = _monitor(clock)
    monitor.observe_tracker_status(True, True, 3)
    for _ in range(30):
        monitor.observe_image(clock.wall, True)
        monitor.observe_tracker_pose(clock.wall, True)
        monitor.observe_gripper(clock.wall + 1, True, 0.5)  # 差 1 ns，不相同。
        clock.advance(33)

    snapshot = monitor.snapshot(ALL_FOUND)

    assert not snapshot.gripper.matched
    assert math.isnan(snapshot.gripper.source_delta_ms)
    assert math.isnan(snapshot.gripper.pair_arrival_delta_ms)
    assert "GRIPPER_UNMATCHED" in snapshot.alarms
    assert not snapshot.image.matched


def test_tracker_outside_match_window_is_unmatched_inside_is_matched() -> None:
    """验证 Tracker 匹配窗口为 30 ms：±29 ms 匹配，40 ms 不匹配。"""
    for offset_ms, expected in ((29.0, True), (-29.0, True), (40.0, False)):
        clock = _Clock()
        monitor = _monitor(clock)
        monitor.observe_tracker_status(True, True, 3)
        for _ in range(10):  # 帧间隔 200 ms，每帧附近只有一个 Tracker 样本。
            monitor.observe_image(clock.wall, True)
            monitor.observe_tracker_pose(clock.wall + int(offset_ms * MS), True)
            monitor.observe_gripper(clock.wall, True, 0.5)
            monitor.observe_tracker_status(True, True, 3)
            clock.advance(200)
        clock.advance(-150)  # 保持各路在 0.25 s 超时内。

        snapshot = monitor.snapshot(ALL_FOUND)

        assert snapshot.tracker.matched is expected
        if expected:
            assert math.isclose(snapshot.tracker.source_delta_ms, abs(offset_ms))
        else:
            assert "TRACKER_UNMATCHED" in snapshot.alarms


def test_gripper_processing_lag_raises_alarm() -> None:
    """验证夹爪源时间戳落后最新图像超过门限时报警并给出滞后量。"""
    clock = _Clock()
    monitor = _monitor(clock)
    monitor.observe_tracker_status(True, True, 3)
    stamps = []
    for _ in range(40):
        stamps.append(clock.wall)
        monitor.observe_image(clock.wall, True)
        monitor.observe_tracker_pose(clock.wall, True)
        if len(stamps) > 12:  # 夹爪始终落后 12 帧（约 400 ms）。
            monitor.observe_gripper(stamps[-13], True, 0.5)
        monitor.observe_tracker_status(True, True, 3)
        clock.advance(33)

    snapshot = monitor.snapshot(ALL_FOUND)

    assert snapshot.gripper_lag_ms > 250.0
    assert "GRIPPER_LAG" in snapshot.alarms
    assert snapshot.gripper.fresh and snapshot.gripper.matched


def test_message_age_uses_wall_clock_against_source_stamp() -> None:
    """验证消息年龄为当前墙钟减源时间戳，包含传输和排队。"""
    clock = _Clock()
    monitor = _monitor(clock)
    monitor.observe_image(clock.wall - 40 * MS, True)

    snapshot = monitor.snapshot(ALL_FOUND)

    assert math.isclose(snapshot.image.message_age_ms, 40.0)


def test_empty_image_marks_stream_invalid() -> None:
    """验证尺寸或长度异常的图像使图像输入无效。"""
    clock = _Clock()
    monitor = _monitor(clock)
    monitor.observe_image(clock.wall, False, "数据为空")

    snapshot = monitor.snapshot(ALL_FOUND)

    assert snapshot.image.fresh and not snapshot.image.valid
    assert snapshot.image.detail == "数据为空"
    assert "IMAGE_INVALID" in snapshot.alarms


def test_missing_frames_raise_gap_alarm_then_clear() -> None:
    """验证相邻源时间戳间隔突增（丢帧）触发 GAP 报警，保持期后清除。

    实机默认 DDS 配置曾在 30 fps 1080p 下静默丢约 7% 的图像。
    """
    clock = _Clock()
    monitor = _monitor(clock)
    _feed(monitor, clock, 40)
    assert "IMAGE_GAP" not in monitor.snapshot(ALL_FOUND).alarms

    clock.advance(33 * 4)  # 连续丢 4 帧后图像恢复。
    _feed(monitor, clock, 3)

    assert "IMAGE_GAP" in monitor.snapshot(ALL_FOUND).alarms
    _feed(monitor, clock, 100)
    assert "IMAGE_GAP" not in monitor.snapshot(ALL_FOUND).alarms


def test_ordinary_jitter_does_not_raise_gap_alarm() -> None:
    """验证 ±20% 的正常抖动不会被误判为丢帧。"""
    clock = _Clock()
    monitor = _monitor(clock)
    monitor.observe_tracker_status(True, True, 3)
    for index in range(80):
        monitor.observe_image(clock.wall, True)
        monitor.observe_tracker_pose(clock.wall, True)
        monitor.observe_gripper(clock.wall, True, 0.5)
        monitor.observe_tracker_status(True, True, 3)
        clock.advance(33 * (1.2 if index % 2 else 0.8))

    assert not any(alarm.endswith("_GAP") for alarm in monitor.snapshot(ALL_FOUND).alarms)


def test_single_dropped_frame_raises_gap_alarm() -> None:
    """验证只丢一帧（间隔约为 2 倍）也会报警；实机默认 DDS 配置下丢帧多为单帧。"""
    clock = _Clock()
    monitor = _monitor(clock)
    _feed(monitor, clock, 40)

    clock.advance(33)  # 少一帧：下一帧间隔约 66 ms。
    _feed(monitor, clock, 2)

    assert "IMAGE_GAP" in monitor.snapshot(ALL_FOUND).alarms
