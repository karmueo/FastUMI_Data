"""使用确定性假引擎验证 DP ROS 节点的发布、重置与时间门控。"""

import asyncio
from concurrent.futures import Future
import hashlib
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from fastumi_interfaces.srv import ResetPolicyController
from std_srvs.srv import Trigger

from dp_infer.core import InferenceContext, decode_actions
from dp_infer.node import DpInferenceNode, sequence_message, stamp_ns
from test_core import _add_frame
from test_visual_guard import _scene


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
    rclpy.init(args=["--ros-args", "-p", f"urdf_path:={path}",
                     "-p", "require_controller_reset:=false"])
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
    assert instance.parameter("image_topic") == "/wrist_camera/image_raw/compressed"
    assert instance.parameter("image_type") == "compressed"
    assert instance.parameter("num_inference_steps") == 8
    assert instance.parameter("joint_topic") == "/joint_states"
    assert instance.parameter("gripper_topic") == "/motion_control/gripper_state"
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
    response = asyncio.run(instance.on_reset(Trigger.Request(), Trigger.Response()))
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


def test_close_guard_publishes_vision_permission_and_clears_on_reset(node):
    instance, _ = node
    decisions = []
    instance.close_guard_publisher = SimpleNamespace(publish=decisions.append)
    for location, expected in (((900, 350), False), ((650, 650), True)):
        rgb = cv2.cvtColor(_scene(location), cv2.COLOR_BGR2RGB)
        instance.publish_close_guard(rgb, instance.get_clock().now().nanoseconds)
        assert decisions[-1].data is expected
    rgb = cv2.cvtColor(_scene((650, 650)), cv2.COLOR_BGR2RGB)
    now_ns = instance.get_clock().now().nanoseconds
    instance.publish_close_guard(rgb, now_ns - instance.buffer.timeout_ns - 1)
    assert decisions[-1].data is False
    instance.publish_close_guard(rgb, now_ns + 1_000_000_000)
    assert decisions[-1].data is False
    instance.reset_episode()
    assert decisions[-1].data is False


def test_reset_response_waits_for_controller_acknowledgement():
    """外部重置服务在控制器确认停止前不得返回成功。"""
    async def exercise():
        future = asyncio.Future()
        calls = []
        fake = SimpleNamespace(
            buffer=SimpleNamespace(episode_id=1),
            _controller_reset_future=None,
            _reset_acknowledged=False,
        )

        def reset_episode():
            calls.append("reset")
            fake._controller_reset_future = future

        def finish_reset():
            calls.append("ack")
            fake._reset_acknowledged = bool(future.result().success)

        fake.reset_episode = reset_episode
        fake._finish_controller_reset = finish_reset
        task = asyncio.create_task(DpInferenceNode.on_reset(
            fake, Trigger.Request(), Trigger.Response()))
        await asyncio.sleep(0)
        assert calls == ["reset"] and not task.done()
        future.set_result(SimpleNamespace(success=True))
        result = await task
        assert result.success and calls == ["reset", "ack"]

    asyncio.run(exercise())


def test_reset_service_waits_for_controller_service(node):
    """单线程执行器也能处理控制器确认，外部服务随后才返回成功。"""
    instance, _ = node
    instance._require_controller_reset = True
    controller = Node("reset_controller_stub")
    caller = Node("reset_caller")
    events = []

    def stop_controller(request, response):
        events.append(request.episode_id)
        response.success = True
        response.message = "stopped"
        return response

    controller.create_service(
        ResetPolicyController, "/fastumi/rm75/placo/reset_episode", stop_controller)
    reset_client = caller.create_client(Trigger, "/fastumi/policy/reset_episode")
    executor = SingleThreadedExecutor()
    for participant in (instance, controller, caller):
        executor.add_node(participant)
    try:
        deadline = time.monotonic() + 2
        while not instance.controller_reset_client.service_is_ready() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert instance.controller_reset_client.service_is_ready()
        future = reset_client.call_async(Trigger.Request())
        while not future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert future.done() and future.result().success
        assert events == [1] and instance.buffer.episode_id == 1
    finally:
        executor.remove_node(instance)
        executor.remove_node(controller)
        executor.remove_node(caller)
        executor.shutdown()
        caller.destroy_node()
        controller.destroy_node()
