"""测试归一化平行夹爪端点映射和模拟适配器。"""

import pytest

from fastumi_rm75.gripper import (
    MockParallelGripperAdapter,
    openness_to_position,
    position_to_openness,
)


def test_maps_and_clips_normalized_openness() -> None:
    """验证 0、1 和越界开度映射到标定端点。"""
    assert openness_to_position(0.0, 0.01, 0.07) == pytest.approx(0.01)
    assert openness_to_position(1.0, 0.01, 0.07) == pytest.approx(0.07)
    assert openness_to_position(2.0, 0.01, 0.07) == pytest.approx(0.07)
    assert position_to_openness(0.04, 0.01, 0.07) == pytest.approx(0.5)
    assert position_to_openness(0.10, 0.01, 0.07) == pytest.approx(1.0)


def test_mock_adapter_records_command_and_stop() -> None:
    """验证模拟适配器支持命令、反馈、端点标定和停止。"""
    adapter = MockParallelGripperAdapter()

    adapter.calibrate_endpoints(0.01, 0.07)
    adapter.command(-0.5)
    assert adapter.last_openness == pytest.approx(0.0)
    assert adapter.state_feedback() == pytest.approx(0.0)
    assert adapter.closed_position_m == pytest.approx(0.01)
    assert adapter.open_position_m == pytest.approx(0.07)
    assert not adapter.stopped

    adapter.stop()
    assert adapter.stopped
