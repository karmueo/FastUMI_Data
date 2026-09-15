"""在 ROS2 环境中验证在途结果的 episode/年龄门控、最新观测覆盖和消息序列化。"""

from concurrent.futures import Future
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

# 无 ROS 安装时只跳过适配层测试，核心测试仍可独立执行。
pytest.importorskip("rclpy")
pytest.importorskip("fastumi_interfaces.msg")
import rclpy
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger

from vr_umi_ros.core import InferenceContext, decode_actions
from vr_umi_ros import node as node_module
from vr_umi_ros.node import (VrUmiInferenceNode, format_prediction,
                             predict_with_timing, sequence_message, stamp_ns)
from test_vr_umi import minimal_urdf
from test_vr_umi_ros_core import add_frame


@pytest.fixture
def node(tmp_path):
    """构造无需模型和真实 URDF 的节点，并记录发布而不连接控制话题。"""
    urdf = minimal_urdf(tmp_path / "arm.urdf", 7)
    rclpy.init(args=["--ros-args", "-p", f"urdf_path:={urdf}"])
    contract = SimpleNamespace(urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest())
    engine = SimpleNamespace(cfg=SimpleNamespace(task=SimpleNamespace(contract=contract)))
    instance = VrUmiInferenceNode(engine=engine)
    messages = []
    instance.publisher = SimpleNamespace(publish=messages.append)
    yield instance, messages
    instance.destroy_node()
    rclpy.shutdown()


def prediction(context):
    """构造有效 16 步序列，模拟已经完成的模型结果。"""
    actions = np.tile(np.r_[np.zeros(3), [1, 0, 0, 0, 1, 0], 0.4], (16, 1))
    return decode_actions(actions, context)


def test_inflight_reset_rejects_old_episode(node):
    """重置后，即使旧模型任务成功完成也不可发布，历史和待处理窗口全部清空。"""
    instance, messages = node
    now = instance.get_clock().now().nanoseconds
    context = InferenceContext(np.eye(4), now, 0, 1)
    instance.last_joint_stamp_ns = now
    instance.future = Future()
    instance.active_context = context
    instance.pending = ({}, context)
    response = instance.on_reset(Trigger.Request(), Trigger.Response())
    assert response.success and instance.buffer.episode_id == 1 and instance.pending is None
    assert instance.last_joint_stamp_ns == -1
    instance.future.set_result((prediction(context), 0.0123))
    instance.tick()
    assert messages == [] and instance.future is None


def test_expired_result_and_valid_message(node, monkeypatch):
    """500ms 外的结果被丢弃，新鲜结果保留原始时间和 16 个配对动作。"""
    instance, messages = node
    formatted = []
    monkeypatch.setattr(
        node_module, "format_prediction",
        lambda message, seconds, age_ns: formatted.append(message.sequence_id) or "prediction")
    observed_seconds = []
    instance.result_observer = lambda message, seconds: observed_seconds.append(seconds)
    now = instance.get_clock().now().nanoseconds
    old = InferenceContext(np.eye(4), now - 600_000_000, 0, 1)
    instance.active_context = old
    instance.future = Future()
    instance.future.set_result((prediction(old), 0.0456))
    instance.tick()
    assert messages == [] and formatted == [] and observed_seconds == []
    current = InferenceContext(np.eye(4), instance.get_clock().now().nanoseconds, 0, 2)
    instance.active_context = current
    instance.future = Future()
    instance.future.set_result((prediction(current), 0.0123))
    instance.tick()
    assert len(messages) == 1
    message = messages[0]
    assert stamp_ns(message.header.stamp) == current.stamp_ns
    assert message.sequence_id == 2 and message.header.frame_id == "base_link"
    assert len(message.poses) == len(message.gripper_openness) == len(message.time_from_start) == 16
    assert stamp_ns(message.time_from_start[-1]) == 500_000_000
    assert formatted == [2] and observed_seconds == [0.0123]


def test_inflight_keeps_latest_pending_window(node):
    """推理未完成时回调继续更新缓存，待处理窗口始终覆盖为最新帧。"""
    instance, messages = node
    now = instance.get_clock().now().nanoseconds
    instance.future = Future()
    add_frame(instance.buffer, now - 90_000_000)
    add_frame(instance.buffer, now - 56_666_667)
    instance.tick()
    first_stamp = instance.pending[1].stamp_ns
    add_frame(instance.buffer, now - 23_333_334)
    instance.tick()
    assert instance.pending[1].stamp_ns > first_stamp
    assert not instance.pending[1].reference_pose.flags.writeable
    assert not instance.future.done() and messages == []
    instance.future.cancel()


def test_message_nanoseconds_are_preserved():
    """大纪元纳秒戳不经过浮点转换，时间和位姿一一对应。"""
    context = InferenceContext(np.eye(4), 1_783_456_789_123_456_789, 5, 17)
    message = sequence_message(prediction(context), context)
    assert stamp_ns(message.header.stamp) == context.stamp_ns
    assert message.episode_id == 5 and message.sequence_id == 17


def test_prediction_log_formats_complete_fixed_width_table():
    """发布日志包含耗时、元数据和完整的 16 步绝对目标。"""
    context = InferenceContext(np.eye(4), 1_783_456_789_123_456_789, 5, 17)
    message = sequence_message(prediction(context), context)
    output = format_prediction(message, 0.012345, 45_670_000)
    lines = output.splitlines()
    assert lines[0] == (
        "Prediction published | sequence=17 | episode=5 | "
        "inference=12.35 ms | age=45.67 ms | steps=16")
    assert lines[1].split() == [
        "step", "time_ms", "|", "x_m", "y_m", "z_m", "|",
        "qx", "qy", "qz", "qw", "|", "gripper"]
    assert len(lines) == 19
    assert lines[3].split() == [
        "0", "0.0", "|", "0.00000", "0.00000", "0.00000", "|",
        "0.00000", "0.00000", "0.00000", "1.00000", "|", "0.4000"]
    assert lines[-1].split()[0:2] == ["15", "500.0"]


def test_predict_with_timing_excludes_caller_polling(monkeypatch):
    """计时只包围策略调用，并原样返回动作序列。"""
    context = InferenceContext(np.eye(4), 123, 0, 1)
    expected = prediction(context)
    engine = SimpleNamespace(predict=lambda observations, value: expected)
    clock = iter((10.0, 10.025))
    monkeypatch.setattr(node_module.time, "perf_counter", lambda: next(clock))
    actual, seconds = predict_with_timing(engine, {"input": "value"}, context)
    assert actual is expected
    assert seconds == pytest.approx(0.025)


def test_synchronous_inference_uses_measured_policy_time(node, monkeypatch):
    """同步入口与异步入口使用相同计时包装器并传递耗时。"""
    instance, messages = node
    instance.set_parameters([Parameter("synchronous_inference", value=True)])
    context = InferenceContext(
        np.eye(4), instance.get_clock().now().nanoseconds, 0, 0)
    instance.pending = ({"input": "value"}, context)
    instance.engine = SimpleNamespace(
        predict=lambda observations, value: prediction(value))
    observed_seconds = []
    instance.result_observer = lambda message, seconds: observed_seconds.append(seconds)
    monkeypatch.setattr(node_module, "format_prediction", lambda *args: "prediction")
    clock = iter((20.0, 20.018))
    monkeypatch.setattr(node_module.time, "perf_counter", lambda: next(clock))

    instance.tick()

    assert len(messages) == 1
    assert instance.last_inference_seconds == pytest.approx(0.018)
    assert observed_seconds == pytest.approx([0.018])
