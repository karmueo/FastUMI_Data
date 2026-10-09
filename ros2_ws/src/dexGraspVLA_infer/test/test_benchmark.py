"""验证性能评估会排除实际轨迹耗尽的档位，无需 ROS 图或硬件。"""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from dexgraspvla_infer import benchmark


@pytest.mark.parametrize(
    "reason,enabled,selected_steps",
    [
        ("trajectory horizon exhausted", False, 12),
        ("episode reset", False, 16),
        ("trajectory horizon exhausted", True, 16),
    ],
)
def test_controller_exhaustion_excludes_otherwise_passing_profile(
    monkeypatch, tmp_path, reason, enabled, selected_steps
):
    """耗尽后无后续预测间隔时仍排除该档位，并去重重复状态消息。"""
    metric_topic = "/fastumi/policy/metrics"
    status_topic = "/fastumi/rm75/joint/status"
    callbacks, events = {}, []
    for steps in (16, 12, 8, 4):
        for index in range(2):
            events.append((metric_topic, {
                "steps": steps, "age_s": 0.45,
                "interval_s": None if index == 0 else 0.5,
            }))
        if steps == 16:
            status = {"episode_id": 7, "enabled": enabled, "reason": reason}
            events.extend([(status_topic, status), (status_topic, status)])

    class FakeNode:
        """只记录订阅回调；测试消息由模拟 executor 顺序交付。"""

        def __init__(self, name):
            self.name = name

        def create_subscription(self, message_type, topic, callback, qos):
            callbacks[topic] = callback

        def destroy_node(self):
            pass

    def spin_once(node, timeout_sec):
        topic, value = events.pop(0)
        callbacks[topic](SimpleNamespace(data=json.dumps(value)))

    ros = ModuleType("rclpy")
    ros.init = lambda: None
    ros.ok = lambda: bool(events)
    ros.spin_once = spin_once
    ros.shutdown = lambda: None
    ros_node = ModuleType("rclpy.node")
    ros_node.Node = FakeNode
    messages = ModuleType("std_msgs.msg")
    messages.String = SimpleNamespace
    for name, module in (
        ("rclpy", ros), ("rclpy.node", ros_node),
        ("std_msgs", ModuleType("std_msgs")), ("std_msgs.msg", messages),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    output = tmp_path / "benchmark.json"
    monkeypatch.setattr(sys, "argv", [
        "benchmark", "--output", str(output), "--minimum", "2", "--warmup", "0",
    ])
    benchmark.main()

    report = json.loads(output.read_text())
    assert report["profiles"]["16"]["passed"]
    assert report["selected_steps"] == selected_steps
    if selected_steps == 12:
        assert report["controller_exhaustion_events"] == [dict(status, steps=16)]
    else:
        assert report["controller_exhaustion_events"] == []
