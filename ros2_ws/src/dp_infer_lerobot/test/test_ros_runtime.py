"""仅在隔离 ROS 环境运行；两种后端使用假引擎，不加载模型或连接硬件。"""

import asyncio
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from std_srvs.srv import Trigger
from dp_infer.common import InferenceContext
from dp_infer.core import decode_actions as legacy_decode
from dp_infer.node import DpInferenceNode, stamp_ns
from dp_infer_lerobot.core import decode_actions
from dp_infer_lerobot.deployment import sha256_file
from dp_infer_lerobot.node import LerobotInferenceNode
from fastumi_rm75.placo_control import decode_trajectory


@pytest.fixture(params=['legacy', 'lerobot'])
def runtime(request):
    src = Path(__file__).resolve().parents[2]
    urdf = src / 'fastumi_rm75/assets/rm_75_kinematic.urdf'
    profile = src / 'dp_infer_lerobot/config/jingbao_deployment.yaml'
    args = ['--ros-args', '-p', f'urdf_path:={urdf}', '-p', 'task_control_enabled:=true',
            '-p', 'require_controller_reset:=false']
    if request.param == 'lerobot':
        args += ['-p', f'deployment_config:={profile}']
    rclpy.init(args=args)
    cfg = SimpleNamespace(task=SimpleNamespace(contract=SimpleNamespace(urdf_sha256=sha256_file(urdf))))
    node_type = DpInferenceNode if request.param == 'legacy' else LerobotInferenceNode
    node = node_type(engine=SimpleNamespace(cfg=cfg))
    published = []
    node.publisher = SimpleNamespace(publish=published.append)
    try:
        yield node, published, request.param
    finally:
        node.destroy_node()
        rclpy.shutdown()


def prediction(node, kind, age=0):
    stamp = node.get_clock().now().nanoseconds - age
    context = InferenceContext(np.eye(4), stamp, node.buffer.episode_id, 1)
    count = 16 if kind == 'legacy' else 8
    values = np.tile(np.r_[np.zeros(3), [1, 0, 0, 0, 1, 0], .7], (count, 1))
    return context, (legacy_decode if kind == 'legacy' else decode_actions)(values, context)


def test_backend_defaults_and_old_episode_result_is_discarded(runtime):
    node, published, kind = runtime
    assert node.parameter('num_inference_steps') == (8 if kind == 'legacy' else 16)
    node.task_state = 'running'
    ctx, sequence = prediction(node, kind)
    node.future, node.active_context = Future(), ctx
    node.future.set_result(sequence)
    node.tick()
    assert len(published[-1].poses) == (16 if kind == 'legacy' else 9)
    node.future, node.active_context = Future(), ctx
    node.reset_episode()
    node.future.set_result(sequence)
    node.tick()
    assert len(published) == 1


def test_short_horizon_expires_before_generic_timeout(runtime):
    node, published, kind = runtime
    node.task_state = 'running'
    ctx, sequence = prediction(node, kind, age=300_000_000)
    node.future, node.active_context = Future(), ctx
    node.future.set_result(sequence)
    node.tick()
    assert bool(published) == (kind == 'legacy')


@pytest.mark.parametrize('steps', [1, 2, 8, 32])
def test_sequential_execution_preserves_requested_actions(runtime, steps):
    """两种后端的动作数、执行时间和控制器接收结果一致，包含单步与超长请求。"""
    node, published, kind = runtime
    node.task_state = 'running'
    node._sequential = True
    node._execution_steps = steps
    node.link7_correction = np.eye(4)
    context, _ = prediction(node, kind)
    count = 16 if kind == 'legacy' else 8
    actions = np.tile(np.r_[0.01, 0, 0, [1, 0, 0, 0, 1, 0], .7], (count, 1))
    if kind == 'legacy':
        # 旧后端的每个预测点均相对于观测位姿，而非逐步累积。
        actions[:, 0] = np.arange(1, count + 1) * 0.01
    sequence = (legacy_decode if kind == 'legacy' else decode_actions)(actions, context)
    node.future, node.active_context = Future(), context
    node.future.set_result(sequence)
    node.tick()

    assert len(published) == 1
    message = published[0]
    executed = min(steps, count)
    assert len(message.poses) == executed + (kind == 'lerobot')
    assert message.poses[-1].position.x == pytest.approx(executed * 0.01)
    assert stamp_ns(message.time_from_start[-1]) == round(executed * 1e9 / 30)
    published_ns = stamp_ns(message.header.stamp)
    assert node.execution_end_ns == published_ns + stamp_ns(message.time_from_start[-1]) + node._settle_ns

    # 模拟发布后收到消息：旧后端的每个动作都应有未来时刻，LeRobot 的 T0 是起始锚点。
    trajectory = decode_trajectory(message, published_ns + 1_000_000, np.eye(4), .7)
    assert len(trajectory.positions) == executed
    assert trajectory.positions[-1, 0] == pytest.approx(executed * 0.01)


def test_start_stop_recording_and_return_use_shared_lifecycle(runtime):
    node, _, _ = runtime
    calls = []
    node._fresh_inputs = lambda: True
    node._record_task = True

    async def record(operation, **kwargs):
        calls.append(operation)
        return True, 'ok', 'recording-1'

    async def control(client, timeout):
        calls.append('control')
        return True, 'ok'

    node._recording_call = record
    node._controller_call = control
    started = asyncio.run(node.on_start_task(Trigger.Request(), Trigger.Response()))
    assert started.success and node.task_state == 'running' and calls == ['start']
    old_episode = node.buffer.episode_id
    stopped = asyncio.run(node.on_stop_task(Trigger.Request(), Trigger.Response()))
    assert stopped.success and node.task_state == 'idle' and calls[-1] == 'stop'
    assert node.buffer.episode_id > old_episode and node.pending is None
    node._require_controller_reset = True
    returned = asyncio.run(node.on_return_to_start(Trigger.Request(), Trigger.Response()))
    assert returned.success and node.task_state == 'idle'
    assert calls[-2:] == ['control', 'control']
