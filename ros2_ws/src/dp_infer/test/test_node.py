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


def _urdf(tmp_path):
    """写一个七个 z 轴转动关节、零偏置的测试 URDF。"""
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
    return path


def _engine(path, **contract):
    """构造只携带契约的假引擎。"""
    contract.setdefault("end_frame", "Link7")
    contract = SimpleNamespace(urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **contract)
    return SimpleNamespace(cfg=SimpleNamespace(task=SimpleNamespace(contract=contract)))


@pytest.fixture
def node(tmp_path):
    """构造与 ROS 接口相连、但不加载真实 checkpoint 的推理节点。"""
    path = _urdf(tmp_path)
    rclpy.init(args=["--ros-args", "-p", f"urdf_path:={path}"])
    instance = DpInferenceNode(engine=_engine(path))
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


def test_tcp_checkpoint_converts_fk_to_tcp_and_checks_axes(tmp_path):
    """TCP checkpoint 观测为 FK·T_Link7_TCP；缺外参或两端轴向不一致时拒绝启动。"""
    import yaml
    from scipy.spatial.transform import Rotation

    path = _urdf(tmp_path)
    camera_from_link7 = np.eye(4)
    camera_from_link7[:3, 3] = [0.0, 0.09, -0.016]
    camera_from_tcp = np.eye(4)
    camera_from_tcp[:3, :3] = Rotation.from_rotvec([1.6, 0, 0]).as_matrix()
    camera_from_tcp[:3, 3] = [0.0, 0.074, 0.21]
    end = tmp_path / "T_cam_end.yaml"
    end.write_text(yaml.safe_dump({"cam0": {"T_cam_hand": camera_from_link7.tolist()}}))
    tcp = tmp_path / "T_d_cam_tcp.yaml"
    tcp.write_text(yaml.safe_dump({"T_d_cam_tcp": camera_from_tcp.tolist()}))
    engine = _engine(path, end_frame="tcp", umi_camera_from_tcp=camera_from_tcp.tolist())
    rclpy.init(args=["--ros-args", "-p", f"urdf_path:={path}"])
    try:
        with pytest.raises(ValueError, match="require end_camera_extrinsic"):
            DpInferenceNode(engine=engine)
    finally:
        rclpy.shutdown()
    rclpy.init(args=["--ros-args", "-p", f"urdf_path:={path}",
                     "-p", f"end_camera_extrinsic:={end}", "-p", f"deploy_tcp_extrinsic:={tcp}"])
    try:
        instance = DpInferenceNode(engine=engine)
        expected = np.linalg.inv(camera_from_link7) @ camera_from_tcp
        np.testing.assert_allclose(instance.link7_from_tool, expected, atol=1e-12)
        instance.on_joint(SimpleNamespace(
            name=[f"joint{index}" for index in range(1, 8)], position=[0.0] * 7,
            header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0))))
        np.testing.assert_allclose(instance.buffer.joints[-1][1], expected, atol=1e-12)
        instance.destroy_node()
        flipped = np.array(camera_from_tcp)
        flipped[:3, :3] = flipped[:3, :3] @ Rotation.from_rotvec([0, 0, np.pi]).as_matrix()
        with pytest.raises(ValueError, match="axis convention"):
            DpInferenceNode(engine=_engine(path, end_frame="tcp", umi_camera_from_tcp=flipped.tolist()))
    finally:
        rclpy.shutdown()
