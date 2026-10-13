"""验证可配置频率窗口在高帧率下的预检、缓存保留与录制中的低频报警。"""

import pytest

from fastumi_data.collection_health import (
    GRIPPER,
    IMAGE,
    TRACKER,
    CollectionHealthMonitor,
    HealthConfig,
    preflight_issues,
)
from fastumi_data.collection_state import QualityAccumulator


MS = 1_000_000
T0 = 1_700_000_000_000_000_000
ALL_FOUND = {IMAGE: True, TRACKER: True, GRIPPER: True}
CONFIG = HealthConfig(expected_image_rate_hz=30.0, expected_gripper_rate_hz=30.0)


class Rig:
    """可控时钟驱动的监控器。"""

    def __init__(self, config: HealthConfig = CONFIG) -> None:
        """单调时钟从 0 开始，墙钟与源时间同域。"""
        self.mono = 0
        self.source = T0
        self.config = config
        self.monitor = CollectionHealthMonitor(config, lambda: self.mono, lambda: self.source)

    def run(self, seconds: float, rate_hz: float = 30.0, receive_every: int = 1,
            gripper_valid: bool = True, gripper_every: int = 1) -> None:
        """按 ``rate_hz`` 推进源时间；``receive_every`` > 1 模拟接收积压（源间隔正常）。"""
        period_ns = int(1e9 / rate_hz)
        steps = int(round(seconds * rate_hz))
        pending = []
        for step in range(steps):
            pending.append(self.source)
            if step % receive_every == 0:
                # 积压时每 receive_every 步才送出最早的一帧，其余留在队列中（接收速率下降）。
                stamp = pending.pop(0)
                self.monitor.observe_image(stamp, True)
                self.monitor.observe_tracker_pose(stamp, True)
                self.monitor.observe_tracker_status(True, True, 3)
                if step % gripper_every == 0:
                    self.monitor.observe_gripper(stamp, gripper_valid, 0.5)
            self.mono += period_ns
            self.source += period_ns

    def codes(self) -> set:
        """当前预检阻塞码。"""
        snapshot = self.monitor.snapshot(ALL_FOUND)
        return {issue.code for issue in preflight_issues(snapshot, self.config)}

    def alarms(self) -> tuple:
        """当前报警码。"""
        return self.monitor.snapshot(ALL_FOUND).alarms


def test_start_blocked_until_full_two_second_window() -> None:
    """窗口不足 2 s 时阻止开始，积满且达标后放行。"""
    rig = Rig()
    rig.run(1.0)
    assert {"IMAGE_RATE_WINDOW_INCOMPLETE", "GRIPPER_RATE_WINDOW_INCOMPLETE"} <= rig.codes()
    rig.run(1.5)
    assert rig.codes() == set()
    snapshot = rig.monitor.snapshot(ALL_FOUND)
    assert snapshot.image.rate_window_complete
    assert 29.0 < snapshot.image.source_rate_hz < 31.0
    assert 29.0 < snapshot.image.receive_rate_hz < 31.0


@pytest.mark.parametrize("rate_hz, window_s", [(240.0, 2.0), (30.0, 10.0)])
def test_rate_window_exceeding_256_samples_becomes_ready(rate_hz, window_s) -> None:
    """高帧率或长窗口均积满后放行，持续运行时仍保留完整窗口并限制缓存增长。"""
    rig = Rig(HealthConfig(
        rate_window_s=window_s,
        expected_image_rate_hz=rate_hz,
        expected_gripper_rate_hz=rate_hz,
    ))
    rig.run(window_s / 2, rate_hz=rate_hz)
    assert {"IMAGE_RATE_WINDOW_INCOMPLETE", "GRIPPER_RATE_WINDOW_INCOMPLETE"} <= rig.codes()
    rig.run(window_s, rate_hz=rate_hz)
    assert rig.codes() == set()
    rig.run(20.0, rate_hz=rate_hz)
    assert rig.codes() == set()
    snapshot = rig.monitor.snapshot(ALL_FOUND)
    for stream in (snapshot.image, snapshot.gripper):
        assert stream.rate_window_complete
        assert stream.source_rate_hz == pytest.approx(rate_hz, rel=0.01)
        assert stream.receive_rate_hz == pytest.approx(rate_hz, rel=0.01)
    # 缓存量由时间窗口限制，不能随总运行时间增长。
    for state in rig.monitor._streams.values():
        assert len(state.samples) <= rate_hz * window_s + 2


def test_stable_low_rate_blocks_start_even_with_regular_intervals() -> None:
    """稳定 20 Hz 间隔规则，相邻间隔检查不报丢帧，但绝对速率低于 28.5 Hz 阻止开始。"""
    rig = Rig()
    rig.run(3.0, rate_hz=20.0)
    codes = rig.codes()
    assert {"IMAGE_LOW_RATE", "GRIPPER_LOW_RATE"} <= codes
    assert "IMAGE_GAP" not in rig.alarms()


def test_high_rate_window_detects_rate_drop_after_warmup() -> None:
    """高帧率窗口成熟后降频仍触发低频预检，不能退回窗口未满而绕过检查。"""
    rig = Rig(HealthConfig(expected_image_rate_hz=240.0, expected_gripper_rate_hz=240.0))
    rig.run(3.0, rate_hz=240.0)
    assert rig.codes() == set()
    rig.run(3.0, rate_hz=120.0)
    assert rig.codes() == {"IMAGE_LOW_RATE", "GRIPPER_LOW_RATE"}
    snapshot = rig.monitor.snapshot(ALL_FOUND)
    for stream in (snapshot.image, snapshot.gripper):
        assert stream.rate_window_complete
        assert stream.receive_rate_hz == pytest.approx(120.0, rel=0.01)


def test_rate_at_threshold_passes() -> None:
    """28.6 Hz 高于 30 × 0.95 门限，不阻止。"""
    rig = Rig()
    rig.run(3.0, rate_hz=28.6)
    assert not {"IMAGE_LOW_RATE", "GRIPPER_LOW_RATE"} & rig.codes()


def test_invalid_gripper_states_do_not_count_toward_rate() -> None:
    """只统计有效夹爪状态：一半无效时夹爪速率不足。"""
    rig = Rig()
    rig.run(3.0, gripper_every=2)
    snapshot = rig.monitor.snapshot(ALL_FOUND)
    assert snapshot.gripper.below_rate
    assert "GRIPPER_LOW_RATE" in rig.codes()
    assert not snapshot.image.below_rate


def test_receive_backlog_is_distinguished_from_source_rate() -> None:
    """源时间戳 30 Hz 正常但接收积压：源速率达标、接收速率不足，仍阻止开始。"""
    rig = Rig()
    rig.run(1.0)
    rig.run(3.0, receive_every=2)
    snapshot = rig.monitor.snapshot(ALL_FOUND)
    assert snapshot.image.source_rate_hz > 28.5
    assert snapshot.image.receive_rate_hz < 28.5
    assert "IMAGE_LOW_RATE" in rig.codes()


def test_recording_alarm_requires_two_seconds_and_clears_after_two_seconds() -> None:
    """录制中连续低于门限 2 s 才报警，恢复后连续达标 2 s 再清除，历史保留在质量统计。"""
    rig = Rig()
    quality = QualityAccumulator()
    rig.run(3.0)
    elapsed = 0.0

    def tick() -> tuple:
        nonlocal elapsed
        snapshot = rig.monitor.snapshot(ALL_FOUND)
        quality.update(elapsed, snapshot, ())
        elapsed += 0.5
        return snapshot.alarms

    assert "IMAGE_LOW_RATE" not in tick()
    seen_low = []
    for _ in range(12):
        rig.run(0.5, rate_hz=15.0)
        seen_low.append("IMAGE_LOW_RATE" in tick())
    # 窗口需先降到门限下，再持续 2 s 才报警。
    assert not seen_low[0] and seen_low[-1]
    first_alarm = seen_low.index(True)
    assert first_alarm >= 4
    recovered = []
    for _ in range(12):
        rig.run(0.5)
        recovered.append("IMAGE_LOW_RATE" in tick())
    assert recovered[0] and not recovered[-1]
    assert recovered.index(False) >= 4
    history = quality.to_dict()
    assert history["has_alarms"]
    assert history["alarms"]["IMAGE_LOW_RATE"]["count"] == 1
    assert "image.source_rate_hz" in history["timing"]
    assert "image.receive_rate_hz" in history["timing"]


def test_expected_rate_zero_disables_absolute_check() -> None:
    """期望频率为 0（外部相机未声明）时不检查绝对速率。"""
    rig = Rig(HealthConfig())
    rig.run(3.0, rate_hz=20.0)
    assert not {"IMAGE_LOW_RATE", "IMAGE_RATE_WINDOW_INCOMPLETE"} & rig.codes()
