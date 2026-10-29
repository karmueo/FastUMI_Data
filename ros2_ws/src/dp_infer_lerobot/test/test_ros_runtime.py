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


SRC = Path(__file__).resolve().parents[2]
REPO = SRC.parents[1]


@pytest.fixture(params=['legacy', 'lerobot'])
def runtime(request):
    urdf = SRC / 'fastumi_rm75/assets/rm_75_kinematic.urdf'
    args = ['--ros-args', '-p', 'task_control_enabled:=true', '-p', 'require_controller_reset:=false']
    if request.param == 'lerobot':
        # 推理 FK 使用训练 URDF，TCP 偏移来自仓库部署外参。
        urdf = REPO / 'assets/rm_75.urdf'
        args += ['-p', f'deployment_config:={SRC / "dp_infer_lerobot/config/dp_umi_v4_deployment.yaml"}',
                 '-p', f'end_camera_extrinsic:={REPO / "config/calibration/T_cam_end.yaml"}',
                 '-p', f'deploy_tcp_extrinsic:={REPO / "config/calibration/T_d_cam_tcp.yaml"}']
    args += ['-p', f'urdf_path:={urdf}']
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


def actions(kind, x=None):
    """构造两种后端的动作：旧后端 [16,10] 相对 rot6d，新后端 [8,7] 绝对 TCP 轴角。"""
    count = 16 if kind == 'legacy' else 8
    values = (np.tile(np.r_[np.zeros(3), [1, 0, 0, 0, 1, 0], .7], (count, 1)) if kind == 'legacy'
              else np.tile(np.r_[np.zeros(6), .7], (count, 1)))
    if x is not None:
        values[:, 0] = x
    return values


def prediction(node, kind, age=0):
    stamp = node.get_clock().now().nanoseconds - age
    context = InferenceContext(np.eye(4), stamp, node.buffer.episode_id, 1)
    return context, (legacy_decode if kind == 'legacy' else decode_actions)(actions(kind), context)


def test_backend_defaults_and_old_episode_result_is_discarded(runtime):
    node, published, kind = runtime
    assert node.parameter('num_inference_steps') == (8 if kind == 'legacy' else 16)
    node.task_state = 'running'
    ctx, sequence = prediction(node, kind)
    node.future, node.active_context = Future(), ctx
    node.future.set_result(sequence)
    node.tick()
    assert len(published[-1].poses) == (16 if kind == 'legacy' else 8)
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


def test_lerobot_observes_tcp_and_publishes_link7():
    """观测位姿为 FK @ T_Link7_TCP；TCP 目标发布前换算回 Link7。"""
    from dp_infer_lerobot import deployment
    urdf = REPO / 'assets/rm_75.urdf'
    rclpy.init(args=['--ros-args', '-p', f'urdf_path:={urdf}', '-p', 'require_controller_reset:=false',
                     '-p', f'deployment_config:={SRC / "dp_infer_lerobot/config/dp_umi_v4_deployment.yaml"}',
                     '-p', f'end_camera_extrinsic:={REPO / "config/calibration/T_cam_end.yaml"}',
                     '-p', f'deploy_tcp_extrinsic:={REPO / "config/calibration/T_d_cam_tcp.yaml"}'])
    node = LerobotInferenceNode(engine=SimpleNamespace())
    try:
        link7_from_tcp, _ = deployment.load_deploy_tcp(REPO / 'config/calibration/T_cam_end.yaml',
                                                       REPO / 'config/calibration/T_d_cam_tcp.yaml')
        assert 0.2 < np.linalg.norm(link7_from_tcp[:3, 3]) < 0.25
        np.testing.assert_allclose(node.link7_correction_inverse, link7_from_tcp, atol=1e-9)
        q = np.array([0.1, -0.3, 0.2, 1.2, -0.1, 0.5, 0.3])
        tcp = node.fk.forward(q) @ node.link7_correction_inverse
        published = []
        node.publisher = SimpleNamespace(publish=published.append)
        node.task_state = 'running'
        from dp_infer_lerobot.core import pose_to_raw
        values = np.tile(np.r_[pose_to_raw(tcp), .5], (8, 1))
        context = InferenceContext(tcp, node.get_clock().now().nanoseconds, node.buffer.episode_id, 1)
        node.future, node.active_context = Future(), context
        node.future.set_result(decode_actions(values, context))
        node.tick()
        pose = published[-1].poses[0]
        np.testing.assert_allclose([pose.position.x, pose.position.y, pose.position.z],
                                   node.fk.forward(q)[:3, 3], atol=1e-9)
    finally:
        node.destroy_node()
        rclpy.shutdown()


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
    # 两种后端的每个预测点都是相对观测位姿/绝对目标，而非逐步累积。
    values = actions(kind, np.arange(1, count + 1) * 0.01)
    sequence = (legacy_decode if kind == 'legacy' else decode_actions)(values, context)
    node.future, node.active_context = Future(), context
    node.future.set_result(sequence)
    node.tick()

    assert len(published) == 1
    message = published[0]
    executed = min(steps, count)
    assert len(message.poses) == executed
    assert message.poses[-1].position.x == pytest.approx(executed * 0.01)
    assert stamp_ns(message.time_from_start[-1]) == round(executed * 1e9 / 30)
    published_ns = stamp_ns(message.header.stamp)
    assert node.execution_end_ns == published_ns + stamp_ns(message.time_from_start[-1]) + node._settle_ns

    # 模拟发布后收到消息：两种后端首点都是动作而非锚点，运行层整体推迟一帧。
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


@pytest.mark.parametrize('backend, values, message', [
    ('tensorrt_fp16', {'export_dir': ''}, 'export_dir'),
    ('pytorch', {'checkpoint': '/nonexistent'}, 'checkpoint'),
    ('onnx', {}, 'backend'),
])
def test_lerobot_backend_requires_model_directory_before_loading(backend, values, message):
    """后端名非法或缺少对应模型目录时，在加载模型前拒绝。"""
    from dp_infer_lerobot.node import LerobotBackend

    parameters = {'backend': backend, 'checkpoint': '', 'export_dir': '', **values}
    with pytest.raises(ValueError, match=message):
        LerobotBackend().create_engine(parameters.__getitem__)
