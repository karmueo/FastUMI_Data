"""使用确定性假引擎验证 DP ROS 节点的发布、重置与时间门控。"""

from concurrent.futures import Future
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from std_srvs.srv import Trigger

from dp_infer.core import InferenceContext, decode_actions
from dp_infer.node import DpInferenceNode, sequence_message, stamp_ns
from test_core import _add_frame


@pytest.fixture
def node(tmp_path):
    """构造与 ROS 接口相连、但不加载真实 checkpoint 的推理节点。"""
    path = tmp_path / "arm.urdf"
    pieces = ['<robot name="test"><link name="base_link"/>']
    for index in range(1, 8):
        parent = "base_link" if index == 1 else f"Link{index - 1}"
        pieces.append(
            f'<link name="Link{index}"/><joint name="joint{index}" type="revolute">'
            f'<parent link="{parent}"/><child link="Link{index}"/>'
            '<origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>'
        )
    pieces.append("</robot>")
    path.write_text("".join(pieces))
    rclpy.init(args=["--ros-args", "-p", f"urdf_path:={path}"])
    contract = SimpleNamespace(urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    engine = SimpleNamespace(cfg=SimpleNamespace(task=SimpleNamespace(contract=contract)))
    instance = DpInferenceNode(engine=engine)
    messages = []  # 模拟 ROS 发布器以检查输出消息。
    instance.publisher = SimpleNamespace(publish=messages.append)
    yield instance, messages
    instance.destroy_node()
    rclpy.shutdown()


def _prediction(context):
    """生成 16 步单位旋转的有效动作序列。"""
    actions = np.tile(np.r_[np.zeros(3), [1, 0, 0, 0, 1, 0], 0.4], (16, 1))
    return decode_actions(actions, context)


def test_default_interfaces_and_message_stamp(node):
    """默认输入接入当前采集话题，输出时间保持为观测采集时间。"""
    instance, _ = node
    assert instance.parameter("image_topic") == "/usb_camera/image_raw"
    assert instance.parameter("joint_topic") == "/joint_states"
    assert instance.parameter("gripper_topic") == "/gripper/openness"
    assert instance.parameter("output_topic") == "/fastumi/policy/action_sequence"
    context = InferenceContext(np.eye(4), 1_783_456_789_123_456_789, 5, 17)
    message = sequence_message(_prediction(context), context)
    assert stamp_ns(message.header.stamp) == context.stamp_ns
    assert message.header.frame_id == "base_link" and message.end_frame == "Link7"
    assert message.episode_id == 5 and message.sequence_id == 17
    assert len(message.poses) == len(message.gripper_openness) == len(message.time_from_start) == 16
    assert stamp_ns(message.time_from_start[-1]) == 500_000_000


def test_reset_expired_result_and_fresh_publish(node):
    """重置或结果超时丢弃旧任务；新鲜结果完整发布。"""
    instance, messages = node
    now = instance.get_clock().now().nanoseconds
    old = InferenceContext(np.eye(4), now, 0, 1)
    instance.future = Future()
    instance.active_context = old
    instance.pending = ({}, old)
    response = instance.on_reset(Trigger.Request(), Trigger.Response())
    assert response.success and instance.buffer.episode_id == 1 and instance.pending is None
    instance.future.set_result(_prediction(old))
    instance.tick()
    assert messages == []
    expired = InferenceContext(np.eye(4), now - 600_000_000, 1, 2)
    instance.active_context = expired
    instance.future = Future()
    instance.future.set_result(_prediction(expired))
    instance.tick()
    assert messages == []
    fresh = InferenceContext(np.eye(4), instance.get_clock().now().nanoseconds, 1, 3)
    instance.active_context = fresh
    instance.future = Future()
    instance.future.set_result(_prediction(fresh))
    instance.tick()
    assert len(messages) == 1 and messages[0].sequence_id == 3


def test_pending_latest_and_clock_rollback(node):
    """在途任务保留最新输入；ROS 时钟回退清空历史并丢弃旧结果。"""
    instance, messages = node
    now = instance.get_clock().now().nanoseconds
    instance.future = Future()
    _add_frame(instance.buffer, now - 90_000_000)
    _add_frame(instance.buffer, now - 56_666_667)
    instance.tick()
    first_stamp = instance.pending[1].stamp_ns
    _add_frame(instance.buffer, now - 23_333_334)
    instance.tick()
    assert instance.pending[1].stamp_ns > first_stamp
    instance.active_context = InferenceContext(np.eye(4), now, 0, 1)
    instance.future.set_result(_prediction(instance.active_context))
    instance.last_clock_ns = now + 1_000_000_000
    instance.tick()
    assert instance.buffer.episode_id == 1
    assert instance.pending is None and messages == []
