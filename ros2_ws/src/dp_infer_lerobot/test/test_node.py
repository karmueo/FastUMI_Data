"""使用确定性假引擎验证 LeRobot ROS 节点的启动校验、TCP 观测、发布与重置。"""

from concurrent.futures import Future
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

rclpy = pytest.importorskip("rclpy")
from std_srvs.srv import Trigger  # noqa: E402

from dp_infer.core import InferenceContext  # noqa: E402
from dp_infer.node import stamp_ns  # noqa: E402
from dp_infer_lerobot import core, deployment  # noqa: E402
from dp_infer_lerobot.node import LerobotInferenceNode  # noqa: E402

PROFILE = Path(__file__).resolve().parents[1] / "config/dp_umi_v4_deployment.yaml"  # 随包部署档。


def _files(tmp_path, umi_rotation=(0.0, 0.0, 0.0)):
    """写测试 URDF、外参和绑定它们的部署档，返回 ROS 参数列表与 T_Link7_TCP。"""
    urdf = tmp_path / "arm.urdf"
    pieces = ['<robot name="test"><link name="base_link"/>']
    for index in range(1, 8):
        parent = "base_link" if index == 1 else f"Link{index - 1}"
        pieces.append(
            f'<link name="Link{index}"/><joint name="joint{index}" type="revolute">'
            f'<parent link="{parent}"/><child link="Link{index}"/>'
            '<origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>')
    pieces.append("</robot>")
    urdf.write_text("".join(pieces))
    camera_from_link7 = np.eye(4)
    camera_from_link7[:3, 3] = [0.0, 0.09, -0.016]
    camera_from_tcp = np.eye(4)
    camera_from_tcp[:3, :3] = Rotation.from_rotvec([1.6, 0, 0]).as_matrix()
    camera_from_tcp[:3, 3] = [0.0, 0.074, 0.21]
    end = tmp_path / "T_cam_end.yaml"
    end.write_text(yaml.safe_dump({"cam0": {"T_cam_hand": camera_from_link7.tolist()}}))
    tcp = tmp_path / "T_d_cam_tcp.yaml"
    tcp.write_text(yaml.safe_dump({"T_d_cam_tcp": camera_from_tcp.tolist()}))
    umi = camera_from_tcp.copy()
    umi[:3, :3] = umi[:3, :3] @ Rotation.from_rotvec(umi_rotation).as_matrix()
    profile = copy.deepcopy(deployment.load_profile(PROFILE))
    profile.update(urdf_sha256=deployment.sha256_file(urdf), umi_camera_from_tcp=umi.tolist())
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(profile))
    args = ["--ros-args", "-p", f"urdf_path:={urdf}", "-p", f"deployment_config:={profile_path}",
            "-p", f"end_camera_extrinsic:={end}", "-p", f"deploy_tcp_extrinsic:={tcp}"]
    return args, np.linalg.inv(camera_from_link7) @ camera_from_tcp


@pytest.fixture
def node(tmp_path):
    """构造与 ROS 接口相连、但不加载真实 checkpoint 的推理节点。"""
    args, link7_from_tcp = _files(tmp_path)
    rclpy.init(args=args)
    instance = LerobotInferenceNode(engine=SimpleNamespace(
        description="fake", predict=lambda observations, context: _prediction(context)))
    messages = []  # 模拟 ROS 发布器以检查输出消息。
    instance.publisher = SimpleNamespace(publish=messages.append)
    yield instance, messages, link7_from_tcp
    instance.destroy_node()
    rclpy.shutdown()


def _prediction(context):
    """生成 8 个静止 TCP 动作并换算为 Link7 序列。"""
    actions = np.tile(np.r_[0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0.6], (core.ACTION_STEPS, 1))
    return core.decode_actions(actions, context)


def _add_frame(instance, stamp):
    """向同步缓存加入一帧匹配时间戳的图像、关节与夹爪。"""
    instance.buffer.add_image(stamp, np.zeros((3, 224, 224), dtype=np.float32))
    instance.on_joint(SimpleNamespace(
        name=[f"joint{index}" for index in range(1, 8)], position=[0.1] * 7,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000))))
    instance.buffer.add_gripper(stamp, 0.4)


def test_tcp_observation_and_eight_point_message(node):
    """关节 FK 右乘 T_Link7_TCP 作为观测；两帧窗口进入待推理；发布 8 点 Link7 消息。"""
    instance, messages, link7_from_tcp = node
    np.testing.assert_allclose(instance.link7_from_tool, link7_from_tcp, atol=1e-12)
    now = instance.get_clock().now().nanoseconds
    instance.future = Future()  # 占住工作线程，只检查待推理窗口。
    _add_frame(instance, now - 40_000_000)
    _add_frame(instance, now - 6_666_667)
    instance.tick()
    observations, context = instance.pending
    expected = instance.fk.forward(np.full(7, 0.1)) @ link7_from_tcp
    np.testing.assert_allclose(context.reference_pose, expected, atol=1e-12)
    np.testing.assert_allclose(observations["observation.state"][0, 1, :6], core.pose_to_raw(expected), atol=1e-6)
    instance.active_context = InferenceContext(np.eye(4), now, 0, 4, link7_from_tcp)
    instance.future.set_result(_prediction(instance.active_context))
    instance.tick()
    assert len(messages) == 1
    message = messages[0]
    assert message.header.frame_id == "base_link" and message.end_frame == "Link7"
    assert len(message.poses) == len(message.gripper_openness) == len(message.time_from_start) == 8
    assert stamp_ns(message.time_from_start[-1]) == round(7e9 / 30)
    target = np.eye(4)
    target[:3, 3] = [0.4, 0.0, 0.3]
    link7 = target @ np.linalg.inv(link7_from_tcp)
    np.testing.assert_allclose([message.poses[0].position.x, message.poses[0].position.y,
                                message.poses[0].position.z], link7[:3, 3], atol=1e-9)


def test_reset_and_expired_results_are_dropped(node):
    """重置后旧 episode 结果丢弃；过期结果丢弃；新鲜结果发布。"""
    instance, messages, link7_from_tcp = node
    now = instance.get_clock().now().nanoseconds
    old = InferenceContext(np.eye(4), now, 0, 1, link7_from_tcp)
    instance.future, instance.active_context = Future(), old
    response = instance.on_reset(Trigger.Request(), Trigger.Response())
    assert response.success and instance.buffer.episode_id == 1
    instance.future.set_result(_prediction(old))
    instance.tick()
    expired = InferenceContext(np.eye(4), now - 600_000_000, 1, 2, link7_from_tcp)
    instance.future, instance.active_context = Future(), expired
    instance.future.set_result(_prediction(expired))
    instance.tick()
    assert messages == []
    fresh = InferenceContext(np.eye(4), instance.get_clock().now().nanoseconds, 1, 3, link7_from_tcp)
    instance.future, instance.active_context = Future(), fresh
    instance.future.set_result(_prediction(fresh))
    instance.tick()
    assert [message.sequence_id for message in messages] == [3]


def test_startup_rejects_mismatched_tcp_axes(tmp_path):
    """部署 TCP 与示教 TCP 轴向相差 180° 时拒绝启动。"""
    args, _ = _files(tmp_path, umi_rotation=(0.0, 0.0, np.pi))
    rclpy.init(args=args)
    try:
        with pytest.raises(ValueError, match="axis convention"):
            LerobotInferenceNode(engine=SimpleNamespace())
    finally:
        rclpy.shutdown()


@pytest.mark.parametrize("max_axis_deg, accepted", [(15.0, True), (10.0, False)])
def test_tcp_axis_threshold_from_parameter_file(tmp_path, max_axis_deg, accepted):
    """12° 外参差按参数文件阈值验收：15° 允许启动，10° 拒绝。"""
    args, _ = _files(tmp_path, umi_rotation=(0.0, 0.0, np.radians(12)))
    config = tmp_path / "node.yaml"
    config.write_text(yaml.safe_dump({"dp_infer_lerobot": {
        "ros__parameters": {"max_tcp_axis_deg": max_axis_deg}}}))
    rclpy.init(args=[*args, "--params-file", str(config)])
    instance = None
    try:
        if accepted:
            instance = LerobotInferenceNode(engine=SimpleNamespace())
            assert instance.parameter("max_tcp_axis_deg") == max_axis_deg
        else:
            with pytest.raises(ValueError, match="axis convention"):
                LerobotInferenceNode(engine=SimpleNamespace())
    finally:
        if instance is not None:
            instance.destroy_node()
        rclpy.shutdown()


def test_tensorrt_backend_requires_export_dir(tmp_path):
    """TensorRT 后端缺少模型包目录时在加载模型前拒绝启动。"""
    args, _ = _files(tmp_path)
    rclpy.init(args=[*args, "-p", "backend:=tensorrt_fp16"])
    try:
        with pytest.raises(ValueError, match="export_dir"):
            LerobotInferenceNode()
    finally:
        rclpy.shutdown()
