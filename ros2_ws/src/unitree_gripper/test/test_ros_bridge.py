"""隔离 ROS 域并模拟 DDS 反馈，验证话题闭环不触碰实机。"""

from types import SimpleNamespace

import pytest
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32

import unitree_gripper.gripper_node as gripper_node


def test_command_and_real_feedback_topics(monkeypatch):
    writes = []

    class FakePublisher:
        def __init__(self, topic, message_type):
            self.topic = topic

        def Init(self):
            pass

        def Write(self, message):
            writes.append((self.topic, float(message.cmds[0].q)))
            return True

        def Close(self):
            pass

    class FakeSubscriber:
        def __init__(self, topic, message_type):
            self.topic = topic

        def Init(self, handler, queue_length):
            pass

        def Read(self, timeout):
            if "left" in self.topic:
                return SimpleNamespace(states=[SimpleNamespace(q=0.02)])
            return None

        def Close(self):
            pass

    monkeypatch.setattr(gripper_node, "ChannelFactoryInitialize", lambda *_: None)
    monkeypatch.setattr(gripper_node, "ChannelPublisher", FakePublisher)
    monkeypatch.setattr(gripper_node, "ChannelSubscriber", FakeSubscriber)

    rclpy.init(domain_id=214)
    bridge = None
    probe = None
    try:
        bridge = gripper_node.Dex1GripperNode()
        probe = Node("gripper_test_probe")
        states = []
        probe.create_subscription(
            Float32, "/motion_control/gripper_state",
            lambda message: states.append(message.data), 10,
        )
        command_pub = probe.create_publisher(
            Float32, "/motion_control/gripper_command", 10
        )
        for _ in range(10):
            rclpy.spin_once(bridge, timeout_sec=0.03)
            rclpy.spin_once(probe, timeout_sec=0.03)
            if writes and states:
                break
        assert writes
        assert writes[0][0] == "rt/dex1/left/cmd"
        assert writes[0][1] > 0.02
        assert states[-1] == pytest.approx(0.0)

        command_pub.publish(Float32(data=0.0))
        for _ in range(10):
            rclpy.spin_once(bridge, timeout_sec=0.03)
            if bridge._controller.target_ratio == 0.0:
                break
        assert bridge._controller.target_ratio == 0.0
        command_pub.publish(Float32(data=float("nan")))
        for _ in range(10):
            rclpy.spin_once(bridge, timeout_sec=0.03)
        assert bridge._controller.target_ratio == 0.0
    finally:
        if probe is not None:
            probe.destroy_node()
        if bridge is not None:
            bridge.destroy_node()
        rclpy.shutdown()


def test_main_cleans_up_after_external_ros_shutdown(monkeypatch):
    """launch 终止节点时，ROS 已关闭也能无异常清理资源。"""
    destroyed = []

    class FakeNode:
        def destroy_node(self):
            destroyed.append(True)

    def interrupt_spin(_node):
        rclpy.shutdown()
        raise ExternalShutdownException()

    monkeypatch.setattr(gripper_node, "Dex1GripperNode", FakeNode)
    monkeypatch.setattr(gripper_node.rclpy, "spin", interrupt_spin)
    gripper_node.main()
    assert destroyed == [True]
