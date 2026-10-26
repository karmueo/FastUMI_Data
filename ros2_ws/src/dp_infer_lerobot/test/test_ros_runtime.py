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
from dp_infer.node import DpInferenceNode
from dp_infer_lerobot.core import decode_actions
from dp_infer_lerobot.deployment import sha256_file
from dp_infer_lerobot.node import LerobotInferenceNode


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
