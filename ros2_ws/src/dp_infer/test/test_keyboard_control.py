"""验证终端按键到任务服务的映射及异步结果处理。"""

from collections import deque
from concurrent.futures import Future
import os
import termios
import time
from types import SimpleNamespace

import pytest
import rclpy
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger
from fastumi_interfaces.srv import SetNumInferenceSteps

from dp_infer import keyboard_control


class FakeClient:
    """记录 Trigger 请求，并由测试控制异步答复。"""

    def __init__(self, ready=True):
        self.ready = ready
        self.requests = []
        self.futures = []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        future = Future()
        self.requests.append(request)
        self.futures.append(future)
        return future

    def respond(self, success, message):
        self.futures[-1].set_result(Trigger.Response(success=success, message=message))


class FakeParameterClient(FakeClient):
    """模拟推理节点的 GetParameters 服务。"""

    def respond(self, value):
        self.futures[-1].set_result(GetParameters.Response(values=[
            ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=value)
        ]))


class FakeStepsClient(FakeClient):
    """模拟去噪步数设置服务。"""

    def respond(self, success, message):
        self.futures[-1].set_result(
            SetNumInferenceSteps.Response(success=success, message=message))


@pytest.fixture
def control():
    """绕过 DDS 创建，使用可控的 Trigger 客户端测试按键逻辑。"""
    node = object.__new__(keyboard_control.KeyboardControl)
    clients = {operation: FakeClient() for operation in keyboard_control.SERVICE_NAMES}
    node.task_clients = clients
    node.steps_parameter_client = FakeParameterClient()
    node.steps_service_client = FakeStepsClient()
    node.step_directions = deque()
    node.steps_call = None
    node.startup_steps_pending = False
    node.pending = {}
    node.motion_locked = False
    yield node, clients


def test_keys_call_correct_trigger_services_and_report_results(control, capsys):
    """回车、空格和两种退格编码均调用正确服务并显示实际答复。"""
    node, clients = control
    capsys.readouterr()
    for key, operation in (("\r", "start"), ("\n", "start"),
                           (" ", "home"), ("\x7f", "stop"), ("\b", "stop")):
        node.handle_key(key)
        assert len(clients[operation].requests) >= 1
        assert isinstance(clients[operation].requests[-1], Trigger.Request)
        assert "已发送请求" in capsys.readouterr().out
        clients[operation].respond(operation != "home", "模拟服务结果")
        node.check_responses()
        output = capsys.readouterr().out
        assert ("success=False" if operation == "home" else "success=True") in output
        assert "模拟服务结果" in output


def test_other_keys_and_duplicate_requests_are_ignored(control, capsys):
    """无关按键不调用服务，未完成的同一操作不重复发送。"""
    node, clients = control
    for key in ("a", "\x1b", None, ""):
        node.handle_key(key)
    assert all(not client.requests for client in clients.values())
    node.handle_key(" ")
    node.handle_key(" ")
    node.handle_key("\n")
    assert len(clients["home"].requests) == 1
    assert not clients["start"].requests
    assert "忽略重复按键" in capsys.readouterr().out


def test_service_unavailable_and_exception_reported(control, capsys):
    """服务不可用或未来结果抛异常时，仍可继续监听。"""
    node, clients = control
    clients["start"].ready = False
    node.handle_key("\n")
    assert not clients["start"].requests
    assert "不可用" in capsys.readouterr().out
    clients["start"].ready = True
    node.handle_key("\n")
    clients["start"].futures[-1].set_exception(RuntimeError("测试异常"))
    node.check_responses()
    assert "测试异常" in capsys.readouterr().out


def test_stop_preempts_home_and_late_success_is_ignored(control, capsys):
    """回位等待期间停止可立刻发送，旧回位成功不能当作当前成功。"""
    node, clients = control
    node.handle_key(" ")
    node.handle_key("\x7f")
    node.handle_key("\b")
    assert len(clients["stop"].requests) == 1
    clients["home"].respond(True, "旧回位答复")
    node.check_responses()
    output = capsys.readouterr().out
    assert "忽略迟到结果" in output
    assert "回到初始状态：成功" not in output
    clients["stop"].respond(True, "已停止")
    node.check_responses()
    assert "停止任务：成功" in capsys.readouterr().out
    node.handle_key("\n")
    assert len(clients["start"].requests) == 1


def test_home_timeout_locks_motion_until_stop_succeeds(control, capsys):
    """回位未知结果下仅允许请求停止，停机成功后再允许开始。"""
    node, clients = control
    node.handle_key(" ")
    node.pending["home"].deadline = 0.0
    node.check_responses()
    assert node.motion_locked
    node.handle_key("\n")
    node.handle_key(" ")
    assert not clients["start"].requests
    assert len(clients["home"].requests) == 1
    node.handle_key("\x7f")
    clients["stop"].respond(False, "停机失败")
    node.check_responses()
    assert node.motion_locked
    node.handle_key("\b")
    clients["stop"].respond(True, "已停止")
    node.check_responses()
    assert not node.motion_locked
    node.handle_key("\n")
    assert len(clients["start"].requests) == 1
    assert "等待服务响应超时" in capsys.readouterr().out


def test_noninteractive_terminal_fails_before_ros_init(monkeypatch, capsys):
    """缺少交互终端时明确报错，不尝试创建 ROS 节点。"""
    monkeypatch.setattr(keyboard_control.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    assert keyboard_control.main() == 2
    assert "需要交互终端" in capsys.readouterr().err


def test_terminal_reader_restores_settings_after_exception():
    """逐键读取后，即使发生异常也恢复终端属性。"""
    master_fd, slave_fd = os.openpty()
    stream = os.fdopen(slave_fd, "r")
    try:
        original = termios.tcgetattr(slave_fd)
        with pytest.raises(RuntimeError, match="测试退出"):
            with keyboard_control.TerminalKeyReader(stream) as reader:
                os.write(master_fd, b"\x7f")
                assert reader.read_key(0.5) == "\x7f"
                raise RuntimeError("测试退出")
        assert termios.tcgetattr(slave_fd) == original
    finally:
        stream.close()
        os.close(master_fd)


def test_terminal_reader_recognizes_left_and_right_sequences():
    """cbreak 终端将 CSI 与应用光标模式的方向键读成完整事件。"""
    master_fd, slave_fd = os.openpty()
    stream = os.fdopen(slave_fd, "r")
    try:
        with keyboard_control.TerminalKeyReader(stream) as reader:
            for sequence in (b"\x1b[C", b"\x1b[D", b"\x1bOC", b"\x1bOD"):
                os.write(master_fd, sequence)
                assert reader.read_key(0.5) == sequence.decode("ascii")
    finally:
        stream.close()
        os.close(master_fd)


@pytest.mark.parametrize("key,current,target", [
    ("\x1b[C", 1, 2), ("\x1b[C", 8, 16), ("\x1b[C", 32, 32),
    ("\x1b[C", 50, 32), ("\x1b[D", 2, 2), ("\x1b[D", 5, 2),
    ("\x1b[D", 32, 16),
])
def test_direction_keys_use_real_parameter_and_clamp(control, capsys, key, current, target):
    """每次先查询权威值，边界也调用设置服务，然后读回并显示。"""
    node, _ = control
    node.handle_key(key)
    assert node.steps_parameter_client.requests[-1].names == ["num_inference_steps"]
    node.steps_parameter_client.respond(current)
    node.check_responses()
    request = node.steps_service_client.requests[-1]
    assert isinstance(request, SetNumInferenceSteps.Request)
    assert request.num_inference_steps == target
    node.steps_service_client.respond(True, "已设置")
    node.check_responses()
    assert len(node.steps_parameter_client.requests) == 2
    node.steps_parameter_client.respond(target)
    node.check_responses()
    output = capsys.readouterr().out
    assert f"目标={target}" in output
    assert f"当前 num_inference_steps={target}" in output
    assert "success=True" in output


def test_rapid_direction_keys_are_processed_in_order(control):
    """连按不跳过任何一次，后续按键均基于再次查询的值。"""
    node, _ = control
    for key in ("\x1b[C", "\x1b[C", "\x1b[D"):
        node.handle_key(key)
    assert len(node.steps_parameter_client.requests) == 1
    for before, target in ((8, 16), (16, 32), (32, 16)):
        node.steps_parameter_client.respond(before)
        node.check_responses()
        assert node.steps_service_client.requests[-1].num_inference_steps == target
        node.steps_service_client.respond(True, "已设置")
        node.check_responses()
        node.steps_parameter_client.respond(target)
        node.check_responses()
    assert len(node.steps_service_client.requests) == 3
    assert not node.step_directions and node.steps_call is None


def test_stop_clears_queued_keys_without_waiting_for_inflight_adjustment(control, capsys):
    """停止请求立即发送，尚未开始的方向键请求被清除。"""
    node, clients = control
    node.handle_key("\x1b[C")
    node.handle_key("\x1b[C")
    node.handle_key("\x7f")
    assert len(clients["stop"].requests) == 1
    assert not node.step_directions
    node.steps_parameter_client.respond(8)
    node.check_responses()
    node.steps_service_client.respond(True, "已设置")
    node.check_responses()
    node.steps_parameter_client.respond(16)
    node.check_responses()
    assert len(node.steps_service_client.requests) == 1
    assert "已清除 1 个" in capsys.readouterr().out


def test_failed_call_timeout_and_mismatched_readback(control, capsys):
    """失败与超时后读回真实值；不把目标值当作当前值。"""
    node, _ = control
    node.handle_key("\x1b[C")
    node.steps_parameter_client.respond(8)
    node.check_responses()
    node.steps_service_client.respond(False, "策略拒绝")
    node.check_responses()
    node.steps_parameter_client.respond(8)
    node.check_responses()
    assert "success=False，目标=16，当前 num_inference_steps=8" in capsys.readouterr().out

    node.handle_key("\x1b[C")
    node.steps_parameter_client.respond(8)
    node.check_responses()
    node.steps_call.deadline = 0.0
    node.check_responses()
    node.steps_parameter_client.respond(16)
    node.check_responses()
    output = capsys.readouterr().out
    assert "等待服务响应超时" in output
    assert "success=False，目标=16，当前 num_inference_steps=16" in output

    node.handle_key("\x1b[C")
    node.steps_parameter_client.respond(8)
    node.check_responses()
    node.steps_service_client.respond(True, "已设置")
    node.check_responses()
    node.steps_parameter_client.respond(4)
    node.check_responses()
    output = capsys.readouterr().out
    assert "success=True，目标=16，当前 num_inference_steps=4" in output
    assert "读回值与目标不同" in output


def test_missing_parameter_readback_never_guesses_current_value(control, capsys):
    """读取失败时显示未知值，下一次按键仍重新查询。"""
    node, _ = control
    node.handle_key("\x1b[C")
    node.steps_parameter_client.futures[-1].set_result(GetParameters.Response(values=[]))
    node.check_responses()
    assert not node.steps_service_client.requests
    assert "当前值未确认" in capsys.readouterr().out
    node.handle_key("\x1b[C")
    node.steps_parameter_client.respond(8)
    node.check_responses()
    node.steps_service_client.respond(True, "已设置")
    node.check_responses()
    node.steps_parameter_client.futures[-1].set_exception(RuntimeError("读回故障"))
    node.check_responses()
    assert "当前值未确认" in capsys.readouterr().out


def test_startup_reports_parameter_value(control, capsys):
    """启动阶段异步读取一次节点参数。"""
    node, _ = control
    node.startup_steps_pending = True
    node.startup_steps_deadline = time.monotonic() + 5
    node.check_responses()
    node.steps_parameter_client.respond(8)
    node.check_responses()
    assert "当前 num_inference_steps=8" in capsys.readouterr().out


def test_keyboard_queries_real_ros_parameter_service(capsys):
    """ROS 往返测试确认键盘显示来自目标节点参数。"""
    rclpy.init()
    server = Node("dp_infer")
    server.declare_parameter("num_inference_steps", 8)

    def set_steps(request, response):
        result = server.set_parameters_atomically([
            Parameter("num_inference_steps", value=request.num_inference_steps)])
        response.success = result.successful
        response.message = "已设置" if result.successful else result.reason
        return response

    server.create_service(
        SetNumInferenceSteps, "/fastumi/policy/set_inference_steps", set_steps)
    keyboard = keyboard_control.KeyboardControl()
    executor = SingleThreadedExecutor()
    executor.add_node(server)
    executor.add_node(keyboard)
    try:
        deadline = time.monotonic() + 5
        while (keyboard.startup_steps_pending or keyboard.steps_call is not None):
            executor.spin_once(timeout_sec=0.01)
            keyboard.check_responses()
            assert time.monotonic() < deadline
        keyboard.handle_key("\x1b[C")
        while keyboard.steps_call is not None or keyboard.step_directions:
            executor.spin_once(timeout_sec=0.01)
            keyboard.check_responses()
            assert time.monotonic() < deadline
        assert server.get_parameter("num_inference_steps").value == 16
        output = capsys.readouterr().out
        assert "当前 num_inference_steps=8" in output
        assert "success=True，目标=16，当前 num_inference_steps=16" in output
    finally:
        executor.remove_node(keyboard)
        executor.remove_node(server)
        executor.shutdown()
        keyboard.destroy_node()
        server.destroy_node()
        rclpy.shutdown()
