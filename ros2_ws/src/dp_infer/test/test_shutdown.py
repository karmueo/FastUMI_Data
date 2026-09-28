"""验证组合 launch 多次发送中断信号时推理节点能完成清理。"""

import signal

import cv2
import torch
from rclpy.signals import SignalHandlerOptions

from dp_infer import node as inference_node


def test_duplicate_sigint_keeps_cleanup_running(monkeypatch):
    """首个信号退出 spin，清理期间的第二个信号不能打断工作线程等待。"""
    events = []

    class FakeNode:
        def destroy_node(self):
            signal.raise_signal(signal.SIGINT)
            events.append("destroy")

    def fake_init(*, args, signal_handler_options):
        assert args is None
        assert signal_handler_options == SignalHandlerOptions.NO
        events.append("init")

    def fake_spin(_node):
        signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(cv2, "setNumThreads", lambda _: None)
    monkeypatch.setattr(inference_node, "DpInferenceNode", FakeNode)
    monkeypatch.setattr(inference_node.rclpy, "init", fake_init)
    monkeypatch.setattr(inference_node.rclpy, "spin", fake_spin)
    monkeypatch.setattr(inference_node.rclpy, "ok", lambda: True)
    monkeypatch.setattr(inference_node.rclpy, "shutdown", lambda: events.append("shutdown"))

    previous_int = signal.getsignal(signal.SIGINT)
    previous_term = signal.getsignal(signal.SIGTERM)
    inference_node.main()

    assert events == ["init", "destroy", "shutdown"]
    assert signal.getsignal(signal.SIGINT) is previous_int
    assert signal.getsignal(signal.SIGTERM) is previous_term
