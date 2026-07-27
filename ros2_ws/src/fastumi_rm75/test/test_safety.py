"""测试 RM75 工作空间、目标步长和 watchdog 安全门限。"""

import numpy as np

from fastumi_rm75.safety import SafetyGate, SafetyLimits


def _gate() -> SafetyGate:
    """创建确定性测试安全门。"""
    return SafetyGate(
        SafetyLimits(
            workspace_min_m=np.asarray([0.0, -1.0, 0.0]),
            workspace_max_m=np.asarray([1.0, 1.0, 1.0]),
            joint_min_rad=np.deg2rad(
                np.asarray([-178, -130, -178, -135, -178, -128, -360])
            ),
            joint_max_rad=np.deg2rad(
                np.asarray([178, 130, 178, 135, 178, 128, 360])
            ),
            maximum_step_translation_m=0.02,
            maximum_step_rotation_rad=0.2,
            state_timeout_s=0.1,
            target_timeout_s=0.2,
        )
    )


def test_rejects_workspace_and_large_step() -> None:
    """验证越界目标和过大相邻位移均被拒绝。"""
    gate = _gate()
    previous = np.eye(4)
    previous[:3, 3] = [0.5, 0.0, 0.5]
    outside = previous.copy()
    outside[0, 3] = 1.2
    large_step = previous.copy()
    large_step[0, 3] += 0.03

    assert "工作空间" in gate.validate_target(outside, previous)
    assert "平移步长" in gate.validate_target(large_step, previous)


def test_watchdog_reports_stale_target() -> None:
    """验证新鲜状态配合过期策略目标会触发目标超时。"""
    gate = _gate()
    now_ns = 1_000_000_000

    reason = gate.stale_reason(
        now_ns,
        state_timestamp_ns=950_000_000,
        target_timestamp_ns=700_000_000,
    )

    assert "策略目标" in reason


def test_watchdog_rejects_future_timestamp() -> None:
    """验证明显超前的消息时间戳会被视为时钟域异常。"""
    gate = _gate()
    now_ns = 1_000_000_000

    reason = gate.stale_reason(
        now_ns,
        state_timestamp_ns=1_500_000_000,
        target_timestamp_ns=now_ns,
    )

    assert "时间戳超前" in reason


def test_rejects_joint_limit_violation() -> None:
    """验证任一关节超过 RM75 配置角度范围都会被拒绝。"""
    positions = np.zeros(7, dtype=np.float64)
    positions[1] = np.deg2rad(131.0)

    reason = _gate().validate_joint_positions(positions)

    assert "关节 2" in reason
