"""验证控制器退出时先发送停止命令，再关闭 ROS 上下文。"""

import signal

import pytest
from rclpy.signals import SignalHandlerOptions

from fastumi_rm75 import rm75_placo_controller as controller


@pytest.mark.parametrize("termination", ["keyboard", "sigint", "sigterm", "interrupted_conversion", "nested_sigint"])
def test_signal_keeps_context_alive_until_node_destroyed(monkeypatch, termination):
    """SIGINT/SIGTERM 应由 Python 处理，以便节点销毁时仍能发布停止指令。"""
    events = []

    class FakeNode:
        def destroy_node(self):
            if termination == "nested_sigint":
                signal.raise_signal(signal.SIGINT)
            events.append("destroy")

    def fake_init(*, args, signal_handler_options):
        assert args is None
        assert signal_handler_options == SignalHandlerOptions.NO
        events.append("init")

    def fake_spin(node):
        assert isinstance(node, FakeNode)
        if termination == "sigterm":
            signal.raise_signal(signal.SIGTERM)
        if termination == "sigint":
            signal.raise_signal(signal.SIGINT)
        if termination == "interrupted_conversion":
            try:
                signal.raise_signal(signal.SIGINT)
            except KeyboardInterrupt:
                raise RuntimeError("Unable to convert call argument to Python object")
        if termination == "nested_sigint":
            signal.raise_signal(signal.SIGINT)
        raise KeyboardInterrupt

    monkeypatch.setattr(controller, "Rm75PlacoController", FakeNode)
    monkeypatch.setattr(controller.rclpy, "init", fake_init)
    monkeypatch.setattr(controller.rclpy, "spin", fake_spin)
    monkeypatch.setattr(controller.rclpy, "ok", lambda: True)
    monkeypatch.setattr(controller.rclpy, "shutdown", lambda: events.append("shutdown"))

    previous_int_handler = signal.getsignal(signal.SIGINT)
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    controller.main()

    assert events == ["init", "destroy", "shutdown"]
    assert signal.getsignal(signal.SIGINT) is previous_int_handler
    assert signal.getsignal(signal.SIGTERM) is previous_term_handler
