"""测试实际反馈、限速、失联及恢复时的夹爪运动约束。"""

import math

import pytest

from unitree_gripper.control import GripperControl, rad_to_ratio, ratio_to_rad


def test_endpoint_mapping_and_invalid_command():
    assert ratio_to_rad(0.0, 0.02, 5.0) == pytest.approx(0.02)
    assert ratio_to_rad(1.0, 0.02, 5.0) == pytest.approx(5.0)
    assert rad_to_ratio(0.02, 0.02, 5.0) == pytest.approx(0.0)
    assert rad_to_ratio(5.0, 0.02, 5.0) == pytest.approx(1.0)
    assert rad_to_ratio(10.0, 0.02, 5.0) == pytest.approx(1.0)
    controller = GripperControl()
    assert controller.target_ratio == 1.0
    for value in (-0.1, 1.1, math.nan, math.inf, -math.inf):
        assert not controller.set_target(value)
        assert controller.target_ratio == 1.0
    assert controller.set_target(0.5)


def test_first_feedback_starts_from_actual_and_is_speed_limited():
    controller = GripperControl()
    assert controller.next_commands(10.0, 0.02) == {}
    assert controller.current_ratio(10.0) is None
    assert controller.update_feedback("left", 0.02, 10.0)
    commands = controller.next_commands(10.0, 0.02)
    assert commands == {"left": pytest.approx(0.14)}
    assert controller.current_ratio(10.0) == pytest.approx(0.0)


def test_hold_error_limits_stalled_motor():
    controller = GripperControl()
    controller.update_feedback("left", 0.02, 10.0)
    for tick in range(1, 8):
        commands = controller.next_commands(10.0 + tick * 0.02, 0.02)
        assert commands["left"] <= 0.02 + 0.30 + 1e-9
    assert commands["left"] == pytest.approx(0.32)


def test_only_fresh_sides_contribute_to_command_and_state():
    controller = GripperControl()
    controller.update_feedback("left", 0.02, 10.0)
    controller.update_feedback("right", 5.0, 10.0)
    assert controller.current_ratio(10.0) == pytest.approx(0.5)
    assert set(controller.next_commands(10.0, 0.02)) == {"left", "right"}
    controller.update_feedback("right", 5.0, 10.6)
    assert set(controller.next_commands(10.6, 0.02)) == {"right"}
    assert controller.current_ratio(10.6) == pytest.approx(1.0)
    assert controller.next_commands(11.2, 0.02) == {}
    assert controller.current_ratio(11.2) is None
    controller.update_feedback("left", 1.0, 11.3)
    assert controller.next_commands(11.3, 0.02)["left"] == pytest.approx(1.12)


def test_invalid_feedback_is_ignored():
    controller = GripperControl()
    assert not controller.update_feedback("left", math.nan, 1.0)
    assert not controller.update_feedback("unknown", 2.0, 1.0)
    assert controller.next_commands(1.0, 0.02) == {}
