"""绝对关节序列校验、延迟过滤、插值及跨序列限速。"""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from fastumi_rm75.joint_control import decode_joint_trajectory, limit_joint_step, sample_joint_trajectory


def stamp(ns):
    sec, nano = divmod(ns, 10**9)
    return NS(sec=sec, nanosec=nano)


def message():
    return NS(header=NS(stamp=stamp(10**9), frame_id="base_link"), joint_names=[f"joint{i}" for i in range(1, 8)],
              points=[NS(positions=[i * 0.01] * 7, time_from_start=stamp(i * 20_000_000),
                         velocities=[], accelerations=[], effort=[]) for i in range(64)],
              gripper_openness=np.linspace(1, 0, 64).tolist())


def decode(msg=None, now=1_500_000_000):
    return decode_joint_trajectory(msg or message(), now, np.zeros(7), 0.8, np.full(7, -1), np.ones(7))


def test_expired_points_are_removed_without_shifting_time():
    trajectory = decode()
    assert trajectory.times_ns[0] == 1_520_000_000
    assert trajectory.times_ns[-1] == 2_260_000_000
    q, g = sample_joint_trajectory(trajectory, 1_510_000_000)
    np.testing.assert_allclose(q, 0.13)
    assert 0 <= g <= 1
    with pytest.raises(ValueError, match="exhausted"):
        sample_joint_trajectory(trajectory, 2_260_000_000)
    with pytest.raises(ValueError, match="regressed"):
        sample_joint_trajectory(trajectory, 1_499_999_999)
    with pytest.raises(ValueError, match="expired"):
        decode(now=2_300_000_000)


@pytest.mark.parametrize("case", ["names", "length", "nan", "limit", "gripper", "offset", "velocity", "future"])
def test_rejects_invalid_sequences(case):
    msg = message()
    if case == "names":
        msg.joint_names.reverse()
    elif case == "length":
        msg.gripper_openness.pop()
    elif case == "nan":
        msg.points[0].positions[0] = np.nan
    elif case == "limit":
        msg.points[-1].positions[0] = 1.1
    elif case == "gripper":
        msg.gripper_openness[0] = -0.1
    elif case == "offset":
        msg.points[1].time_from_start = stamp(0)
    elif case == "velocity":
        msg.points[0].velocities = [0] * 7
    else:
        msg.header.stamp = stamp(3_000_000_000)
    with pytest.raises(ValueError):
        decode(msg)


def test_delayed_tick_and_new_sequence_cannot_expand_velocity_step():
    previous = np.zeros(7)
    velocity = np.ones(7) * 0.25
    limited = limit_joint_step(np.ones(7), previous, velocity, 0.5, 0.02)
    np.testing.assert_allclose(limited, 0.005)
    switched = limit_joint_step(-np.ones(7), limited, velocity, 0.02, 0.02)
    np.testing.assert_allclose(switched, 0)
