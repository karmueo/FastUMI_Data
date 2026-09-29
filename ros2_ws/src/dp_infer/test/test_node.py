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
from rclpy.parameter import Parameter
from fastumi_interfaces.srv import ResetPolicyController, SetNumInferenceSteps
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


def test_set_inference_steps_service_validates_and_syncs_parameter(node):
    """服务只接受 1～50，成功后 ROS 参数与下一次推理配置一致。"""
    instance, _ = node
    assert instance.parameter("set_inference_steps_service") == "/fastumi/policy/set_inference_steps"
    for steps in (1, 50, 16):
        response = instance.on_set_inference_steps(
            SetNumInferenceSteps.Request(num_inference_steps=steps),
            SetNumInferenceSteps.Response())
        assert response.success and "next inference" in response.message
        assert instance.parameter("num_inference_steps") == steps
        assert instance._requested_steps == steps
    for steps in (0, -1, 51):
        response = instance.on_set_inference_steps(
            SetNumInferenceSteps.Request(num_inference_steps=steps),
            SetNumInferenceSteps.Response())
        assert not response.success and "[1, 50]" in response.message
        assert instance.parameter("num_inference_steps") == 16
        assert instance._requested_steps == 16
    rejected = instance.set_parameters_atomically([
        Parameter("num_inference_steps", value=True)])
    assert not rejected.successful and instance._requested_steps == 16


def test_set_inference_steps_ros_service_round_trip(node):
    """真实 ROS 服务请求可更新配置，并向客户端返回结果。"""
    instance, _ = node
    caller = Node("set_inference_steps_caller")
    client = caller.create_client(
        SetNumInferenceSteps, "/fastumi/policy/set_inference_steps")
    executor = SingleThreadedExecutor()
    executor.add_node(instance)
    executor.add_node(caller)
    try:
        deadline = time.monotonic() + 2
        while not client.service_is_ready() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert client.service_is_ready()
        future = client.call_async(SetNumInferenceSteps.Request(num_inference_steps=12))
        while not future.done() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert future.done() and future.result().success
        assert instance.parameter("num_inference_steps") == 12
    finally:
        executor.remove_node(instance)
        executor.remove_node(caller)
        executor.shutdown()
        caller.destroy_node()


def test_set_inference_steps_keeps_inflight_and_uses_latest_on_next_submit(node, monkeypatch):
    """在途结果照常发布；连续修改只影响下一次提交的任务。"""
    instance, messages = node
    now = instance.get_clock().now().nanoseconds
    context = InferenceContext(np.eye(4), now, instance.buffer.episode_id, 1)
    instance.future = Future()
    instance.active_context = context
    instance.active_num_inference_steps = 8
    instance.pending = ({}, context)
    instance.engine.predict = lambda *_: None
    submissions = []

    def submit(*args):
        submissions.append(args)
        return Future()

    monkeypatch.setattr(instance.worker, "submit", submit)
    for steps in (16, 4):
        response = instance.on_set_inference_steps(
            SetNumInferenceSteps.Request(num_inference_steps=steps),
            SetNumInferenceSteps.Response())
        assert response.success
    instance.tick()
    assert not submissions and instance.active_num_inference_steps == 8
    instance.future.set_result(_prediction(context))
    instance.tick()
    assert len(messages) == 1 and messages[0].sequence_id == 1
    assert len(submissions) == 1 and submissions[0][-1] == 4
    assert instance.active_num_inference_steps == 4


def test_reset_expired_result_and_fresh_publish(node):
    """重置或结果超时丢弃旧任务；新鲜结果完整发布。"""
    instance, messages = node
    warnings = []
    instance.warn = warnings.append
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
    assert "episode changed" in warnings[-1]
    assert "predicted_episode=0 current_episode=1" in warnings[-1]
    expired = InferenceContext(np.eye(4), now - 600_000_000, 1, 2)
    instance.active_context = expired
    instance.future = Future()
    instance.future.set_result(_prediction(expired))
    instance.tick()
    assert messages == []
    assert "result expired" in warnings[-1]
    assert "limit_ms=500.0" in warnings[-1]
    assert "denoise_steps=8" in warnings[-1]
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


def test_managed_task_starts_stops_and_drops_inflight_result(node):
    """组合入口待命，停止后旧模型结果不得重新发布。"""
    instance, messages = node
    instance._task_control_enabled = True
    instance.task_state = "idle"
    now = instance.get_clock().now().nanoseconds
    _add_frame(instance.buffer, now - 20_000_000)
    start = asyncio.run(instance.on_start_task(Trigger.Request(), Trigger.Response()))
    assert start.success and instance.task_state == "running"
    episode = instance.buffer.episode_id
    old = InferenceContext(np.eye(4), now, episode, 1)
    instance.future = Future()
    instance.active_context = old
    stop = asyncio.run(instance.on_stop_task(Trigger.Request(), Trigger.Response()))
    assert stop.success and instance.task_state == "idle"
    assert instance.buffer.episode_id > episode
    instance.future.set_result(_prediction(old))
    instance.tick()
    assert messages == []
    repeated = asyncio.run(instance.on_stop_task(Trigger.Request(), Trigger.Response()))
    assert repeated.success
    assert instance.buffer.episode_id == episode + 1
    reset = asyncio.run(instance.on_reset(Trigger.Request(), Trigger.Response()))
    assert reset.success and instance.task_state == "idle"
    home = asyncio.run(instance.on_return_to_start(Trigger.Request(), Trigger.Response()))
    assert not home.success and "unavailable" in home.message


def test_stop_interrupts_pending_start_and_return(node):
    """旧服务确认不得覆盖后续停止建立的待命状态。"""
    async def exercise():
        instance, _ = node
        instance._task_control_enabled = True
        instance._require_controller_reset = True
        instance.task_state = "idle"
        _add_frame(instance.buffer, instance.get_clock().now().nanoseconds - 20_000_000)
        start_future = asyncio.Future()
        calls = []

        async def controller_call(client, timeout):
            calls.append(client)
            if client is instance.controller_start_client:
                return await start_future
            return True, "stopped"

        instance._controller_call = controller_call
        start_task = asyncio.create_task(instance.on_start_task(
            Trigger.Request(), Trigger.Response()))
        await asyncio.sleep(0)
        assert instance.task_state == "starting"
        stop = await instance.on_stop_task(Trigger.Request(), Trigger.Response())
        assert stop.success and instance.task_state == "idle"
        start_future.set_result((True, "late start"))
        assert not (await start_task).success
        assert instance.task_state == "idle"

        home_future = asyncio.Future()

        async def home_call(client, timeout):
            if client is instance.controller_return_client:
                return await home_future
            return True, "stopped"

        instance._controller_call = home_call
        home_task = asyncio.create_task(instance.on_return_to_start(
            Trigger.Request(), Trigger.Response()))
        await asyncio.sleep(0)
        assert instance.task_state == "homing"
        stopped = await instance.on_stop_task(Trigger.Request(), Trigger.Response())
        assert stopped.success and instance.task_state == "idle"
        home_future.set_result((True, "late home"))
        assert not (await home_task).success
        assert instance.task_state == "idle"

    asyncio.run(exercise())


def test_idle_stop_invalidates_a_start_that_timed_out(node):
    """开始响应超时后，停止仍须让控制器拒绝迟到的旧 episode。"""
    instance, _ = node
    instance._task_control_enabled = True
    instance._require_controller_reset = True
    instance.task_state = "idle"
    _add_frame(instance.buffer, instance.get_clock().now().nanoseconds - 20_000_000)
    calls = []

    async def controller_call(client, _timeout):
        calls.append((client, instance.buffer.episode_id))
        if client is instance.controller_start_client:
            return False, "Controller service timed out"
        return True, "Driver stop acknowledged"

    instance._controller_call = controller_call
    start = asyncio.run(instance.on_start_task(Trigger.Request(), Trigger.Response()))
    assert not start.success and instance.task_state == "idle"

    stop = asyncio.run(instance.on_stop_task(Trigger.Request(), Trigger.Response()))
    assert stop.success and instance.task_state == "idle"
    assert calls == [
        (instance.controller_start_client, 1),
        (instance.controller_reset_client, 2),
    ]


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


def test_managed_public_services_coordinate_with_controller(node):
    """单线程执行器可完成开始、停机、回位的内部服务往返。"""
    instance, _ = node
    instance._task_control_enabled = True
    instance._require_controller_reset = True
    instance.task_state = "idle"
    controller = Node("task_controller_stub")
    caller = Node("task_caller")
    events = []

    def handler(name):
        def handle(request, response):
            events.append((name, request.episode_id))
            response.success = True
            response.message = name
            return response
        return handle

    for name, topic in (
            ("start", "/fastumi/rm75/placo/start_task"),
            ("stop", "/fastumi/rm75/placo/reset_episode"),
            ("home", "/fastumi/rm75/placo/return_to_start")):
        controller.create_service(ResetPolicyController, topic, handler(name))
    clients = {
        name: caller.create_client(Trigger, f"/fastumi/policy/{name}")
        for name in ("start_task", "stop_task", "return_to_start")
    }
    executor = SingleThreadedExecutor()
    for participant in (instance, controller, caller):
        executor.add_node(participant)
    try:
        deadline = time.monotonic() + 3
        while not all(client.service_is_ready() for client in clients.values()):
            executor.spin_once(timeout_sec=0.01)
            assert time.monotonic() < deadline
        _add_frame(instance.buffer, instance.get_clock().now().nanoseconds - 20_000_000)
        for name in ("start_task", "stop_task", "return_to_start"):
            future = clients[name].call_async(Trigger.Request())
            while not future.done() and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.01)
            assert future.done() and future.result().success
        assert [name for name, _ in events] == ["start", "stop", "stop", "home"]
        assert [episode for _, episode in events] == [1, 2, 3, 4]
        assert instance.task_state == "idle"
    finally:
        for participant in (instance, controller, caller):
            executor.remove_node(participant)
        executor.shutdown()
        caller.destroy_node()
        controller.destroy_node()
