"""无 ROS 网络验证键盘按键到急停/回位/关节运动服务调用的映射。"""

import math
import time
from unittest.mock import Mock

import pytest
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from fastumi_interfaces.srv import MoveJoints

from fastumi_arm_keyboard.keyboard_control import (
    ArmKeyboardControl, main, parse_joint_input, parse_speed_input,
)


class SimpleResponse:
    def __init__(self, success, message):
        self.success = success
        self.message = message


class FakeFuture:
    def __init__(self):
        self.result_value = None
        self.finished = False
        self.cancelled = False

    def done(self):
        return self.finished

    def result(self):
        return self.result_value

    def cancel(self):
        self.cancelled = True

    def finish(self, success, message=""):
        self.result_value = SimpleResponse(success=success, message=message)
        self.finished = True


class FakeClient:
    def __init__(self, ready=True):
        self.ready = ready
        self.futures = []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        future = FakeFuture()
        future.request = request
        self.futures.append(future)
        return future


@pytest.fixture
def control(monkeypatch):
    """仅构造按键逻辑状态，用假客户端替代 ROS 服务客户端。"""
    node = ArmKeyboardControl.__new__(ArmKeyboardControl)
    node.service_names = {"stop": "/fastumi/arm/emergency_stop",
                          "home": "/fastumi/arm/return_home",
                          "move": "/fastumi/arm/move_joints"}
    node.service_clients = {"stop": FakeClient(), "home": FakeClient(), "move": FakeClient()}
    node.pending = {}
    node.move_speed = 20
    node.current_degrees = None
    node.joint_state_at = None
    node._reset_input()
    node.messages = []
    monkeypatch.setattr(node, "_emit", node.messages.append)
    monkeypatch.setattr(node, "_echo", node.messages.append)
    return node


CURRENT_RADIANS = [0.0, 0.5, 0.0, 1.0, 0.0, 1.5, 1.2]


def feed_joint_state(node, positions=CURRENT_RADIANS):
    node._joint_state_callback(JointState(
        name=[f"joint{index}" for index in range(1, 8)], position=positions))


def type_text(node, text):
    for key in text:
        node.handle_key(key)


def test_space_stops_and_enter_homes(control):
    control.handle_key("x")
    control.handle_key(None)
    assert control.pending == {}
    control.handle_key(" ")
    control.handle_key("\n")
    assert isinstance(control.service_clients["stop"].futures[0].request, Trigger.Request)
    assert isinstance(control.service_clients["home"].futures[0].request, Trigger.Request)
    control.handle_key("\r")
    assert len(control.service_clients["home"].futures) == 1
    assert "已有运动请求" in control.messages[-1]


def test_stop_supersedes_pending_home(control):
    control.handle_key("\n")
    control.handle_key(" ")
    home = control.service_clients["home"].futures[0]
    stop = control.service_clients["stop"].futures[0]
    home.finish(False, "回位被急停中断")
    stop.finish(True, "RM75 轨迹已停止")
    control.check_responses()
    assert control.pending == {}
    assert any("紧急停止：成功" in message for message in control.messages)
    assert any("已被紧急停止取代" in message for message in control.messages)
    # 急停结果确认后可以再次回位。
    control.handle_key("\n")
    assert len(control.service_clients["home"].futures) == 2


def test_unavailable_service_not_called(control):
    control.service_clients["stop"].ready = False
    control.handle_key(" ")
    assert control.service_clients["stop"].futures == []
    assert "不可用" in control.messages[-1]


def test_timeout_cancels_and_reports(control, monkeypatch):
    control.handle_key("\n")
    control.pending["home"].deadline = 0.0
    control.check_responses()
    assert control.service_clients["home"].futures[0].cancelled
    assert "超时" in control.messages[-1]
    assert control.pending == {}


def test_main_requires_tty(monkeypatch, capsys):
    stdin = Mock()
    stdin.isatty.return_value = False
    monkeypatch.setattr("sys.stdin", stdin)
    assert main([]) == 2
    assert "交互终端" in capsys.readouterr().err


CURRENT_DEGREES = [math.degrees(value) for value in CURRENT_RADIANS]


def test_parse_joint_input():
    assert parse_joint_input("", 12.5) == 12.5
    assert parse_joint_input("  ", 12.5) == 12.5
    assert parse_joint_input(" -30.5 ", 12.5) == -30.5
    for bad in ("x", "1 2", "nan", "inf"):
        with pytest.raises(ValueError):
            parse_joint_input(bad, 0.0)


def test_joint_input_requires_fresh_joint_state(control):
    control.handle_key("j")
    assert control.input_stage is None
    assert "joint_states" in control.messages[-1]
    feed_joint_state(control)
    control.joint_state_at = time.monotonic() - 5.0
    control.handle_key("j")
    assert control.input_stage is None


def test_joint_by_joint_input_keeps_blank_and_calls_move_joints(control):
    feed_joint_state(control)
    control.handle_key("J")
    assert control.input_stage == "joints"
    assert any(f"joint4={CURRENT_DEGREES[3]:.2f}" in m for m in control.messages)
    assert control.messages[-1].startswith("joint1（当前 0.00）")

    type_text(control, "10\n")          # joint1
    type_text(control, "\n")            # joint2 保持
    type_text(control, "-5x")
    control.handle_key("\x7f")
    type_text(control, "\n")            # joint3 = -5
    assert control.messages[-1].startswith("joint4")
    type_text(control, "\n\n\n")        # joint4..6 保持
    assert control.input_stage == "joints"
    type_text(control, "45\n")          # joint7
    assert control.input_stage == "speed"
    assert "当前 20%" in control.messages[-1]
    type_text(control, "\n")            # 速度保持
    assert control.input_stage == "confirm"
    assert "y/n" in control.messages[-1]
    assert any("速度：20%" in m for m in control.messages)

    # 确认阶段只认 y/n，其他键（包括空格、回车）被忽略。
    type_text(control, " \nq")
    assert control.input_stage == "confirm"
    assert control.service_clients["move"].futures == []
    assert control.service_clients["stop"].futures == []

    control.handle_key("y")
    assert control.input_stage is None
    request = control.service_clients["move"].futures[0].request
    assert isinstance(request, MoveJoints.Request)
    expected = [10.0, CURRENT_DEGREES[1], -5.0, *CURRENT_DEGREES[3:6], 45.0]
    assert list(request.joint_degrees) == pytest.approx(expected)
    assert request.speed == 20
    assert "move" in control.pending


def test_blank_keeps_value_from_input_start(control):
    feed_joint_state(control)
    control.handle_key("j")
    # 输入期间反馈变化不影响“保持不变”的值。
    feed_joint_state(control, [0.3] * 7)
    type_text(control, "\n" * 8 + "y")
    request = control.service_clients["move"].futures[0].request
    assert list(request.joint_degrees) == pytest.approx(CURRENT_DEGREES)


def test_invalid_joint_value_reprompts_same_joint(control):
    feed_joint_state(control)
    control.handle_key("j")
    type_text(control, "abc\n")
    assert any("输入无效" in m for m in control.messages)
    assert control.messages[-1].startswith("joint1")
    assert control.input_degrees == []
    type_text(control, "1\n")
    assert control.input_degrees == [1.0]


@pytest.mark.parametrize("keys", ["\x1b", "1\n2\x1b", "\n" * 7 + "\x1b", "\n" * 8 + "n",
                                  "\n" * 8 + "\x1b"])
def test_joint_input_cancel_paths(control, keys):
    feed_joint_state(control)
    control.handle_key("j")
    type_text(control, keys)
    assert control.input_stage is None
    assert control.service_clients["move"].futures == []
    # 取消后恢复普通按键：空格又是急停。
    control.handle_key(" ")
    assert len(control.service_clients["stop"].futures) == 1


def test_space_is_not_stop_while_typing(control):
    feed_joint_state(control)
    control.handle_key("j")
    control.handle_key(" ")
    assert control.service_clients["stop"].futures == []


def test_motion_keys_blocked_while_motion_pending(control):
    feed_joint_state(control)
    control.handle_key("\n")
    control.handle_key("j")
    assert control.input_stage is None
    assert "已有运动请求" in control.messages[-1]


def test_stop_supersedes_pending_move(control):
    feed_joint_state(control)
    control.handle_key("j")
    type_text(control, "\n" * 6 + "1\n\ny")
    control.handle_key(" ")
    control.service_clients["move"].futures[0].finish(False, "运动被急停中断")
    control.service_clients["stop"].futures[0].finish(True, "RM75 轨迹已停止")
    control.check_responses()
    assert control.pending == {}
    assert any("关节运动：已被紧急停止取代" in m for m in control.messages)


def test_parse_speed_input():
    assert parse_speed_input("", 25) == 25
    assert parse_speed_input(" 40 ", 25) == 40
    for bad in ("0", "101", "fast", "2.5"):
        with pytest.raises(ValueError):
            parse_speed_input(bad, 25)


def test_speed_step_sets_and_keeps_new_speed(control):
    feed_joint_state(control)
    control.handle_key("j")
    type_text(control, "\n" * 6 + "1\n")
    type_text(control, "150\n")
    assert control.input_stage == "speed"
    assert any("输入无效" in m for m in control.messages)
    type_text(control, "35\n")
    assert control.input_stage == "confirm"
    assert any("速度：35%" in m for m in control.messages)
    control.handle_key("y")
    assert control.service_clients["move"].futures[0].request.speed == 35

    # 下一次 J 显示并默认沿用新速度。
    control.check_responses()
    control.pending.clear()
    control.handle_key("j")
    type_text(control, "\n" * 6 + "2\n")
    assert "当前 35%" in control.messages[-1]
    type_text(control, "\ny")
    assert control.service_clients["move"].futures[1].request.speed == 35
