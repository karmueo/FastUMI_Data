"""验证夹爪遥操的跟随、保持、输入校验和超时状态转换。"""

import math

import pytest

from tracker_teleoperated.gripper_follow import GripperFollower


def test_follow_requires_real_feedback_and_valid_estimate():
    """确认未收到真实反馈时绝不转发预测，输入齐备后直接跟随。"""
    follower = GripperFollower(0.25, 0.25)
    follower.update_estimate(True, 0.8, 10.0)

    missing = follower.step(True, 10.0)
    assert missing.mode == "feedback_missing"
    assert missing.command is None

    follower.update_feedback(0.3, 10.0)
    following = follower.step(True, 10.0)
    assert following.mode == "following"
    assert following.command == pytest.approx(0.8)


def test_pause_holds_recent_measured_openness_once():
    """确认暂停时保持真实位置，不继续发送上一预测目标。"""
    follower = GripperFollower(0.25, 0.25)
    follower.update_feedback(0.35, 10.0)
    follower.update_estimate(True, 0.9, 10.0)
    assert follower.step(True, 10.0).command == pytest.approx(0.9)

    follower.update_feedback(0.48, 10.02)
    paused = follower.step(False, 10.02)
    assert paused.mode == "paused"
    assert paused.command == pytest.approx(0.48)
    assert follower.step(False, 10.03).command is None


def test_paused_startup_holds_when_first_feedback_arrives():
    """确认默认暂停且反馈稍后到达时，也会建立一次保持目标。"""
    follower = GripperFollower(0.25, 0.25)
    assert follower.step(False, 10.0).command is None
    follower.update_feedback(0.42, 10.01)
    assert follower.step(False, 10.01).command == pytest.approx(0.42)
    assert follower.step(False, 10.02).command is None


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -0.01, 1.01])
def test_invalid_prediction_holds_measured_openness_and_recovers(bad_value):
    """确认无效预测不会透传，下一帧有效预测可自动恢复。"""
    follower = GripperFollower(0.25, 0.25)
    follower.update_feedback(0.2, 10.0)
    follower.update_estimate(True, 0.8, 10.0)
    assert follower.step(True, 10.0).command == pytest.approx(0.8)

    assert not follower.update_estimate(True, bad_value, 10.01)
    held = follower.step(True, 10.01)
    assert held.mode == "estimate_invalid"
    assert held.command == pytest.approx(0.2)
    assert follower.step(True, 10.02).command is None

    follower.update_estimate(True, 0.4, 10.03)
    resumed = follower.step(True, 10.03)
    assert resumed.mode == "following"
    assert resumed.command == pytest.approx(0.4)


def test_explicit_invalid_frame_holds_immediately():
    """确认 valid=false 的一帧立即撤销此前预测。"""
    follower = GripperFollower(0.25, 0.25)
    follower.update_feedback(0.3, 10.0)
    follower.update_estimate(True, 0.7, 10.0)
    follower.step(True, 10.0)

    assert not follower.update_estimate(False, math.nan, 10.01)
    assert follower.step(True, 10.01).command == pytest.approx(0.3)


def test_prediction_timeout_holds_then_resumes_on_new_frame():
    """确认预测断流超过阈值后保持实测开度，新帧到达后恢复。"""
    follower = GripperFollower(0.25, 0.25)
    follower.update_feedback(0.4, 10.0)
    follower.update_estimate(True, 0.8, 10.0)
    follower.step(True, 10.0)
    follower.update_feedback(0.5, 10.2)

    timed_out = follower.step(True, 10.26)
    assert timed_out.mode == "estimate_timeout"
    assert timed_out.command == pytest.approx(0.5)
    assert follower.step(True, 10.27).command is None

    follower.update_estimate(True, 0.6, 10.28)
    assert follower.step(True, 10.28).command == pytest.approx(0.6)


def test_feedback_timeout_holds_last_measured_value():
    """确认真实反馈断流时只发送一次最后的实测开度。"""
    follower = GripperFollower(0.25, 0.25)
    follower.update_feedback(0.3, 10.0)
    follower.update_estimate(True, 0.8, 10.0)
    follower.step(True, 10.0)
    follower.update_estimate(True, 0.9, 10.2)

    timed_out = follower.step(True, 10.26)
    assert timed_out.mode == "feedback_timeout"
    assert timed_out.command == pytest.approx(0.3)
    assert follower.step(True, 10.27).command is None

    follower.update_feedback(0.45, 10.28)
    assert follower.step(True, 10.28).command == pytest.approx(0.9)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -0.01, 1.01])
def test_invalid_feedback_stops_following(bad_value):
    """确认非法真实反馈导致夹爪保持最近一次合法实测值。"""
    follower = GripperFollower(0.25, 0.25)
    follower.update_feedback(0.3, 10.0)
    follower.update_estimate(True, 0.8, 10.0)
    follower.step(True, 10.0)

    assert not follower.update_feedback(bad_value, 10.01)
    held = follower.step(True, 10.01)
    assert held.mode == "feedback_invalid"
    assert held.command == pytest.approx(0.3)


@pytest.mark.parametrize("timeout", [0.0, -1.0, math.nan, math.inf])
def test_invalid_timeout_is_rejected(timeout):
    """确认节点启动时拒绝无效夹爪超时参数。"""
    with pytest.raises(ValueError, match="正有限数"):
        GripperFollower(timeout, 0.25)
    with pytest.raises(ValueError, match="正有限数"):
        GripperFollower(0.25, timeout)
