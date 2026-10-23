"""使用确定性假引擎验证 DP ROS 节点的发布、重置与时间门控。"""

import asyncio
from concurrent.futures import Future
import hashlib
from threading import Event
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CompressedImage
from fastumi_interfaces.srv import (
    CancelRecordingRequest, ResetPolicyController, SetNumInferenceSteps, StartRecording,
)
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


def test_link7_correction_applies_to_joint_observation_and_published_targets(node):
    """FK 位姿右乘 C⁻¹ 进入观测缓存，发布目标右乘 C；默认无修正文件时为单位阵。"""
    from scipy.spatial.transform import Rotation
    from sensor_msgs.msg import JointState

    instance, messages = node
    np.testing.assert_array_equal(instance.link7_correction, np.eye(4))
    correction = np.eye(4)
    correction[:3, :3] = Rotation.from_euler("x", 15, degrees=True).as_matrix()
    correction[:3, 3] = [0.0, 0.005, -0.01]
    instance.link7_correction = correction
    instance.link7_correction_inverse = np.linalg.inv(correction)
    angles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    joints = JointState(name=[f"joint{i}" for i in range(1, 8)], position=angles)
    joints.header.stamp.sec = 1
    instance.on_joint(joints)
    np.testing.assert_allclose(
        instance.buffer.joints[-1][1], instance.fk.forward(angles) @ np.linalg.inv(correction))
    context = InferenceContext(np.eye(4), instance.get_clock().now().nanoseconds, 0, 4)
    instance.active_context = context
    instance.future = Future()
    instance.future.set_result(_prediction(context))
    instance.tick()
    pose = messages[-1].poses[0]
    np.testing.assert_allclose(
        [pose.position.x, pose.position.y, pose.position.z], correction[:3, 3], atol=1e-9)
    published = Rotation.from_quat(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
    assert (published * Rotation.from_matrix(correction[:3, :3]).inv()).magnitude() < 1e-9


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
        instance._guard_future.result(timeout=5)
        instance._finish_close_guard()
        assert decisions[-1].data is expected
    rgb = cv2.cvtColor(_scene((650, 650)), cv2.COLOR_BGR2RGB)
    now_ns = instance.get_clock().now().nanoseconds
    instance.publish_close_guard(rgb, now_ns - instance.buffer.timeout_ns - 1)
    assert decisions[-1].data is False
    instance.publish_close_guard(rgb, now_ns + 1_000_000_000)
    assert decisions[-1].data is False
    instance.reset_episode()
    assert decisions[-1].data is False


def test_slow_close_guard_does_not_block_observation_windows(node, monkeypatch):
    """检测阻塞时仍接收连续图像及状态，且只保留一个检测任务。"""
    instance, _ = node
    entered, release = Event(), Event()

    def slow_detection(_rgb):
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release detection")
        return True

    monkeypatch.setattr("dp_infer.node.close_guard_allowed", slow_detection)
    now = [1_000_000_000]
    monkeypatch.setattr(instance, "get_clock", lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=now[0])))
    _, jpeg = cv2.imencode(".jpg", np.zeros((24, 32, 3), dtype=np.uint8))
    windows = []
    try:
        for index in range(10):
            now[0] = 1_000_000_000 + round(index * 1e9 / 30)
            message = CompressedImage(data=jpeg.tobytes(), format="jpeg")
            message.header.stamp.sec, message.header.stamp.nanosec = divmod(now[0], 10**9)
            instance.buffer.add_pose(now[0], np.eye(4))
            instance.buffer.add_gripper(now[0], 1.0)
            instance.on_compressed_image(message)
            if index == 0:
                assert entered.wait(timeout=2)
                first_future = instance._guard_future
            assert instance._guard_future is first_future
            window = instance.buffer.poll(now[0])
            if window is not None:
                windows.append(window[-1].stamp_ns)
        assert len(windows) == 9
        assert not first_future.done()
    finally:
        release.set()


@pytest.mark.parametrize("invalidation", [
    "reset", "clear_task", "invalid_image", "stale_image", "idle", "expired", "future", "error",
])
def test_inflight_close_guard_cannot_restore_invalid_permission(node, monkeypatch, invalidation):
    """重置、坏图像、暂停、过期及检测失败均不能被在途的允许结果覆盖。"""
    instance, _ = node
    decisions = []
    instance.close_guard_publisher = SimpleNamespace(publish=decisions.append)
    now = [1_000_000_000]
    monkeypatch.setattr(instance, "get_clock", lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(nanoseconds=now[0])))
    future = Future()
    future.set_running_or_notify_cancel()
    instance._guard_future = future
    instance._guard_context = (now[0], instance.buffer.episode_id, instance._guard_generation)
    if invalidation == "reset":
        instance.reset_episode()
    elif invalidation == "clear_task":
        instance._clear_task()
    elif invalidation == "invalid_image":
        instance.on_compressed_image(CompressedImage())
    elif invalidation == "stale_image":
        instance.publish_close_guard(None, now[0] - instance.buffer.timeout_ns - 1)
    elif invalidation == "idle":
        instance._task_control_enabled = True
        instance.task_state = "idle"
    elif invalidation == "expired":
        now[0] += instance.buffer.timeout_ns + 1
    elif invalidation == "future":
        now[0] -= 1
    if invalidation == "error":
        future.set_exception(ValueError("invalid detection"))
    else:
        future.set_result(True)
    instance._finish_close_guard()
    assert decisions and all(not decision.data for decision in decisions)


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


def _recording_node(node, start_result=(True, "ok", "rec-1"), controller_ok=True):
    """开启任务录制，并用假服务记录录制器与控制器的调用顺序。"""
    instance, _ = node
    instance._task_control_enabled = True
    instance._require_controller_reset = True
    instance._record_task = True
    instance.task_state = "idle"
    _add_frame(instance.buffer, instance.get_clock().now().nanoseconds - 20_000_000)
    calls = []

    async def recording_call(operation, **_identities):
        calls.append(("recording", operation, instance._recording_id))
        return start_result if operation == "start" else (True, "STOP_ACCEPTED", "rec-1")

    async def controller_call(client, _timeout):
        calls.append(("controller", client))
        if client is instance.controller_start_client:
            return controller_ok, "started" if controller_ok else "rejected"
        return True, "stopped"

    instance._recording_call = recording_call
    instance._controller_call = controller_call
    return instance, calls


def test_task_recording_starts_before_controller_and_stops_after(node):
    instance, calls = _recording_node(node)
    start = asyncio.run(instance.on_start_task(Trigger.Request(), Trigger.Response()))
    assert start.success and "recording rec-1 started" in start.message
    assert calls == [("recording", "start", ""),
                     ("controller", instance.controller_start_client)]
    calls.clear()
    stop = asyncio.run(instance.on_stop_task(Trigger.Request(), Trigger.Response()))
    assert stop.success and "recording rec-1 saving" in stop.message
    assert calls == [("controller", instance.controller_reset_client),
                     ("recording", "stop", "rec-1")]
    assert instance._recording_id == ""


def test_task_recording_failure_only_warns(node):
    instance, calls = _recording_node(node, start_result=(False, "Recording service is unavailable", ""))
    start = asyncio.run(instance.on_start_task(Trigger.Request(), Trigger.Response()))
    assert start.success and instance.task_state == "running"
    assert "recording not started: Recording service is unavailable" in start.message
    stop = asyncio.run(instance.on_stop_task(Trigger.Request(), Trigger.Response()))
    assert stop.success and "recording" not in stop.message
    assert [call for call in calls if call[0] == "recording"] == [("recording", "start", "")]


def test_task_recording_cancelled_when_controller_rejects_start(node):
    instance, calls = _recording_node(node, controller_ok=False)
    start = asyncio.run(instance.on_start_task(Trigger.Request(), Trigger.Response()))
    assert not start.success and instance.task_state == "idle"
    assert "recording rec-1 cancelled" in start.message
    assert calls[-1] == ("recording", "cancel", "rec-1")
    assert instance._recording_id == ""


def test_task_recording_disabled_by_default(node):
    instance, calls = _recording_node(node)
    instance._record_task = False
    assert asyncio.run(instance.on_start_task(Trigger.Request(), Trigger.Response())).success
    assert asyncio.run(instance.on_stop_task(Trigger.Request(), Trigger.Response())).success
    assert all(call[0] == "controller" for call in calls)


def test_recording_start_timeout_cancels_the_durable_request(node):
    """开始应答丢失时按同一 UUID 撤销，防止留下没有 recording_id 的后台录制。"""
    instance, _ = _recording_node(node)
    instance._recording_call = DpInferenceNode._recording_call.__get__(instance)
    requests = []

    async def service_call(client, request, _timeout, _label):
        requests.append(request)
        if client is instance.recording_clients["start"]:
            assert instance._recording_request_id == request.request_id
            return None, "Recording service timed out"
        assert client is instance.recording_clients["cancel_request"]
        return CancelRecordingRequest.Response(
            success=True, state="cancelled", recording_id="late-rec", message="CANCELLED"), ""

    instance._service_call = service_call
    start = asyncio.run(instance.on_start_task(Trigger.Request(), Trigger.Response()))
    assert start.success and "recording not started" in start.message
    assert requests[0].request_id == requests[1].request_id
    assert instance._recording_id == instance._recording_request_id == ""
    assert asyncio.run(instance.on_stop_task(Trigger.Request(), Trigger.Response())).success
    assert len(requests) == 2


def test_unconfirmed_recording_start_can_be_cancelled_on_later_stop(node):
    """开始与撤销都暂时不可用时保留 UUID，下一次停止重试同一个请求。"""
    instance, _ = _recording_node(node)
    instance._recording_call = DpInferenceNode._recording_call.__get__(instance)
    available = False
    request_ids = []

    async def service_call(client, request, _timeout, _label):
        request_ids.append(request.request_id)
        if not available:
            return None, "Recording service is unavailable"
        assert client is instance.recording_clients["cancel_request"]
        return CancelRecordingRequest.Response(success=True, state="cancelled"), ""

    instance._service_call = service_call
    assert asyncio.run(instance.on_start_task(Trigger.Request(), Trigger.Response())).success
    pending_id = instance._recording_request_id
    assert pending_id and not instance._recording_id
    available = True
    assert asyncio.run(instance.on_stop_task(Trigger.Request(), Trigger.Response())).success
    assert request_ids == [pending_id, pending_id, pending_id]
    assert instance._recording_request_id == instance._recording_id == ""


@pytest.mark.parametrize("terminal_state", ["failed", "completed"])
def test_terminal_recording_request_needs_no_further_cleanup(node, terminal_state):
    """录制器已持久化失败或完成状态时，即使 cancel 返回 false 也无活动录制。"""
    instance, _ = node
    instance._recording_request_id = "old-request"

    async def service_call(*_args):
        return CancelRecordingRequest.Response(success=False, state=terminal_state), ""

    instance._service_call = service_call
    assert "cancelled" in asyncio.run(instance._finish_recording("stop"))
    assert not instance._recording_request_id


def test_recording_stop_failure_retains_identity_and_idle_stop_retries(node):
    """录制停止失败不影响停机结果，保留会话；仅推理模式 idle 下也可重试。"""
    instance, _ = _recording_node(node)
    instance.task_state = "running"
    instance._recording_id = "rec-to-retry"
    instance._recording_request_id = "request-to-retry"
    attempts = []

    async def recording_call(operation, **identities):
        attempts.append((operation, identities))
        return (len(attempts) > 1, "unavailable" if len(attempts) == 1 else "STOP_ACCEPTED", "")

    instance._recording_call = recording_call
    stop = asyncio.run(instance.on_stop_task(Trigger.Request(), Trigger.Response()))
    assert stop.success and "stop failed" in stop.message
    assert instance._recording_id == "rec-to-retry"
    assert instance._recording_request_id == "request-to-retry"
    instance._require_controller_reset = False
    retry = asyncio.run(instance.on_stop_task(Trigger.Request(), Trigger.Response()))
    assert retry.success and "saving" in retry.message
    assert attempts[0] == attempts[1]
    assert instance._recording_id == instance._recording_request_id == ""


def test_new_task_retries_old_recording_cleanup_before_opening_new_session(node):
    """收尾失败后开始下一轮时，先重试停止旧录制，再创建新录制。"""
    instance, calls = _recording_node(node)
    instance._recording_id = "old-rec"
    instance._recording_request_id = "old-request"
    result = asyncio.run(instance.on_start_task(Trigger.Request(), Trigger.Response()))
    assert result.success
    assert calls[:2] == [("recording", "stop", "old-rec"), ("recording", "start", "")]
    assert instance._recording_id == "rec-1"


def test_unconfirmed_cleanup_does_not_reuse_old_recording_for_new_task(node):
    """旧录制无法停止时保持可重试身份，并明确报告新任务未开新录制。"""
    instance, _ = _recording_node(node)
    instance._recording_id = "old-rec"
    attempts = []

    async def recording_call(operation, **_identities):
        attempts.append(operation)
        return False, "unavailable", ""

    instance._recording_call = recording_call
    start = asyncio.run(instance.on_start_task(Trigger.Request(), Trigger.Response()))
    assert start.success and "recording not started" in start.message
    assert instance._recording_id == "old-rec"
    assert attempts == ["stop"]


def test_task_stays_stopping_until_recording_response_arrives(node):
    """录制停止在途期间不允许新任务；应答后下一轮必须开启新录制。"""
    instance, _ = _recording_node(node)

    async def exercise():
        instance.task_state = "running"
        instance._recording_id = "old-rec"
        finish = asyncio.Future()
        calls = []

        async def recording_call(operation, **_identities):
            calls.append(operation)
            if operation == "stop":
                return await finish
            return True, "STARTED", "new-rec"

        instance._recording_call = recording_call
        stopping = asyncio.create_task(instance.on_stop_task(Trigger.Request(), Trigger.Response()))
        await asyncio.sleep(0)
        assert instance.task_state == "stopping"
        start = await instance.on_start_task(Trigger.Request(), Trigger.Response())
        assert not start.success and "stopping" in start.message
        assert calls == ["stop"]
        finish.set_result((True, "STOP_ACCEPTED", "old-rec"))
        assert (await stopping).success
        assert instance.task_state == "idle" and not instance._recording_id
        _add_frame(instance.buffer, instance.get_clock().now().nanoseconds - 20_000_000)
        assert (await instance.on_start_task(Trigger.Request(), Trigger.Response())).success
        assert calls == ["stop", "start"] and instance._recording_id == "new-rec"

    asyncio.run(exercise())


def test_superseded_stop_cannot_clear_newer_fault_state(node):
    """录制收尾期间时钟回退等操作使旧停止失效，迟到应答不能覆盖 fault。"""
    instance, _ = _recording_node(node)

    async def exercise():
        instance.task_state, instance._recording_id = "running", "rec-1"
        finish = asyncio.Future()

        async def recording_call(*_args, **_kwargs):
            return await finish

        instance._recording_call = recording_call
        stopping = asyncio.create_task(instance.on_stop_task(Trigger.Request(), Trigger.Response()))
        await asyncio.sleep(0)
        instance._clear_task()
        instance.task_state = "fault"
        finish.set_result((True, "STOP_ACCEPTED", "rec-1"))
        result = await stopping
        assert not result.success and "superseded" in result.message
        assert instance.task_state == "fault"

    asyncio.run(exercise())


def test_late_recording_start_cannot_overwrite_restarted_task(node):
    """开始在途时停止按 UUID 撤销；其迟到应答不能覆盖下一轮录制身份。"""
    instance, _ = _recording_node(node)
    instance._recording_call = DpInferenceNode._recording_call.__get__(instance)

    async def exercise():
        old_start = asyncio.Future()
        starts, cancellations = [], []

        async def service_call(client, request, _timeout, _label):
            if client is instance.recording_clients["start"]:
                starts.append(request.request_id)
                if len(starts) == 1:
                    return await old_start
                return StartRecording.Response(success=True, recording_id="new-rec"), ""
            cancellations.append(request.request_id)
            return CancelRecordingRequest.Response(success=True, state="cancelled"), ""

        instance._service_call = service_call
        starting = asyncio.create_task(instance.on_start_task(Trigger.Request(), Trigger.Response()))
        await asyncio.sleep(0)
        assert instance.task_state == "starting" and starts
        assert (await instance.on_stop_task(Trigger.Request(), Trigger.Response())).success
        assert cancellations == starts
        assert not instance._recording_id and not instance._recording_request_id
        _add_frame(instance.buffer, instance.get_clock().now().nanoseconds - 20_000_000)
        assert (await instance.on_start_task(Trigger.Request(), Trigger.Response())).success
        old_start.set_result((StartRecording.Response(success=True, recording_id="old-rec"), ""))
        assert not (await starting).success
        assert instance.task_state == "running" and instance._recording_id == "new-rec"
        assert instance._recording_request_id == starts[1]

    asyncio.run(exercise())


@pytest.mark.parametrize("lose_start_response", [False, True])
def test_task_recording_with_real_ros_recorder(node, tmp_path, lose_start_response):
    """隔离 ROS 服务验证 MCAP 话题、时间戳、帧数，以及开始应答丢失后的实际撤销。"""
    rosbag2_py = pytest.importorskip("rosbag2_py")
    recorder_module = pytest.importorskip("fastumi_recorder.node")
    if "mcap" not in rosbag2_py.get_registered_writers():
        pytest.skip("rosbag2_storage_mcap is not installed")
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float32
    from rclpy.serialization import deserialize_message
    from fastumi_interfaces.msg import PolicyActionSequence
    from fastumi_bringup.initial_pose import load_initial_joint_positions

    instance, _ = _recording_node(node)
    instance._recording_call = DpInferenceNode._recording_call.__get__(instance)
    instance.timer.cancel()  # 此测试仅验证服务和数据录制，不调度假引擎推理。
    if lose_start_response:
        async def service_call(client, request, timeout, label):
            result, error = await DpInferenceNode._service_call(
                instance, client, request, timeout, label)
            if client is instance.recording_clients["start"] and result is not None:
                assert result.success
                return None, "Simulated lost start response"
            return result, error

        instance._service_call = service_call
    prefix = "/test_task_recording"
    recorder = recorder_module.RecorderNode(parameter_overrides=[
        Parameter("dataset_root", value=str(tmp_path)),
        Parameter("joint_state_topic", value=prefix + "/joint"),
        Parameter("gripper_state_topic", value=prefix + "/gripper"),
        Parameter("image_transport", value="raw"),
        Parameter("image_topic", value=prefix + "/image"),
        Parameter("extra_topics", value=prefix + "/action:fastumi_interfaces/msg/PolicyActionSequence"),
    ])
    caller = Node("task_recording_smoke_caller")
    clients = {name: caller.create_client(Trigger, "/fastumi/policy/" + name)
               for name in ("start_task", "stop_task")}
    publishers = {name: caller.create_publisher(kind, prefix + "/" + name, 10)
                  for name, kind in (("joint", JointState), ("gripper", Float32),
                                     ("image", Image), ("action", PolicyActionSequence))}
    executor = SingleThreadedExecutor()
    participants = (instance, recorder, caller)
    for participant in participants:
        executor.add_node(participant)

    def publish_inputs():
        stamp = caller.get_clock().now().nanoseconds
        joints = JointState(name=[f"joint{i}" for i in range(1, 8)],
                            position=list(load_initial_joint_positions()))
        joints.header.stamp.sec, joints.header.stamp.nanosec = divmod(stamp, 10**9)
        frame = Image(width=16, height=16, step=48, encoding="rgb8", data=bytes(768))
        frame.header.stamp = joints.header.stamp
        publishers["joint"].publish(joints)
        publishers["gripper"].publish(Float32(data=1.0))
        publishers["image"].publish(frame)
        context = InferenceContext(np.eye(4), stamp, instance.buffer.episode_id, 1)
        publishers["action"].publish(sequence_message(_prediction(context), context))

    def pump_until(condition):
        deadline = time.monotonic() + 5
        while not condition():
            assert time.monotonic() < deadline
            publish_inputs()
            executor.spin_once(timeout_sec=0.01)

    def call(name):
        future = clients[name].call_async(Trigger.Request())
        pump_until(future.done)
        assert future.result().success, future.result().message
        return future.result()

    try:
        pump_until(lambda: all(client.service_is_ready() for client in
                              (*clients.values(), *instance.recording_clients.values()))
                   and all(pub.get_subscription_count() for pub in publishers.values())
                   and all(recorder.engine.status()[field] >= 0
                           for field in ("joint_age", "gripper_age", "image_age")))
        _add_frame(instance.buffer, instance.get_clock().now().nanoseconds - 20_000_000)
        start = call("start_task")
        if lose_start_response:
            assert "recording not started" in start.message
            assert recorder.engine.state == "idle"
            assert not instance._recording_request_id and not instance._recording_id
            assert not list(tmp_path.glob("*/*/.recording-*"))
            assert not recorder.engine.catalog.entries
            assert call("stop_task").success
        else:
            recording_id = instance._recording_id
            assert recording_id and recorder.engine.state == "recording"
            pump_until(lambda: recorder.engine._counts["received_frames"] >= 2)
            assert "saving" in call("stop_task").message
            pump_until(lambda: recorder.engine.state != "saving")
            assert recorder.engine.state == "idle", recorder.engine.last_error
            info = recorder.engine.catalog.entries[recording_id]
            bag = tmp_path / info["relative_path"] / "bag"
            assert info["image_frames"] >= 2 and (bag / "metadata.yaml").is_file()
            assert list(bag.glob("*.mcap"))
            reader = rosbag2_py.SequentialReader()
            reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                        rosbag2_py.ConverterOptions("", ""))
            seen = set()
            image_count = 0
            while reader.has_next():
                topic, serialized, timestamp_ns = reader.read_next()
                assert timestamp_ns > 0
                seen.add(topic)
                if topic == prefix + "/image":
                    frame = deserialize_message(serialized, Image)
                    assert 0 < stamp_ns(frame.header.stamp) <= timestamp_ns
                    assert frame.encoding == "rgb8" and len(frame.data) == 768
                    image_count += 1
            assert seen == {prefix + "/" + name for name in publishers}
            assert image_count == info["image_frames"]
            del reader
    finally:
        recorder.close()
        for participant in participants:
            executor.remove_node(participant)
        executor.shutdown()
        recorder.destroy_node()
        caller.destroy_node()
