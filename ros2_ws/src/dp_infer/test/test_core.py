"""验证独立 DP 核心的输入契约、同步、动作解码和 URDF 检查。"""

import hashlib
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
from scipy.spatial.transform import Rotation
import yaml

from dp_infer.core import (
    InferenceContext, OBS_SHAPES, Observation, build_observations,
    compare_tcp_frames, contract_end_frame, decode_actions, image_to_rgb, letterbox_rgb,
    load_deploy_tcp, ordered_joints, validate_contract, validate_urdf,
)
from dp_infer.synchronizer import ObservationBuffer
from umi.common.pose_util import mat_to_pose10d


def _contract_config(digest):
    """构造与 Link7 模型相同的最小推理契约。"""
    shape_meta = {  # 五键两帧观测以及 16 步、10D 动作。
        "obs": {
            name: {"shape": list(shape), "horizon": 2, "down_sample_steps": 1,
                   "latency_steps": 0}
            for name, shape in OBS_SHAPES.items()
        },
        "action": {"shape": [10], "horizon": 16, "down_sample_steps": 1,
                   "latency_steps": 0},
    }
    contract = {  # RM75 Link7 坐标、频率和夹爪约定。
        "base_frame": "base_link", "end_frame": "Link7", "tool_offset": "identity",
        "gripper_representation": "normalized_0_1",
        "action_reference": "current_observed_end_frame", "frequency_hz": 30,
        "urdf_sha256": digest,
    }
    return OmegaConf.create({
        "task": {"contract": contract, "pose_repr": {
            "obs_pose_repr": "relative", "action_pose_repr": "relative"}},
        "shape_meta": shape_meta,
    })


def _add_frame(buffer, stamp_ns, pose=None, with_gripper=True):
    """向同步缓存加入一帧具有匹配时间戳的观测。"""
    pose = np.eye(4) if pose is None else pose
    buffer.add_image(stamp_ns, np.zeros((3, 224, 224), dtype=np.float32))
    buffer.add_pose(stamp_ns, pose)
    if with_gripper:
        buffer.add_gripper(stamp_ns, 0.4)


def test_image_stride_color_and_joint_order():
    """带行填充 BGR 图像正确转 RGB；关节按名称提取并拒绝坏值。"""
    bgr = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    padded = np.zeros((2, 12), dtype=np.uint8)
    padded[:, :9] = bgr.reshape(2, 9)
    rgb = image_to_rgb(padded.tobytes(), 2, 3, 12, "bgr8")
    np.testing.assert_array_equal(rgb, bgr[..., ::-1])
    image = letterbox_rgb(rgb)
    assert image.shape == (3, 224, 224) and image.dtype == np.float32
    np.testing.assert_array_equal(
        ordered_joints([f"joint{i}" for i in range(7, 0, -1)], list(range(7, 0, -1))),
        np.arange(1, 8),
    )
    with pytest.raises(ValueError):
        ordered_joints(["joint1"] * 7, [0.0] * 7)
    with pytest.raises(ValueError):
        ordered_joints([f"joint{i}" for i in range(1, 8)], [np.nan] * 7)


def test_observation_and_absolute_action_contract():
    """五键两帧形状、最新相对位姿和非平凡绝对位姿解码保持训练约定。"""
    reference = np.eye(4)
    reference[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_matrix()
    reference[:3, 3] = [0.3, -0.2, 0.1]
    start = np.eye(4)
    history = [
        Observation(0, np.zeros((3, 224, 224), dtype=np.float32), start, 0.2),
        Observation(33_333_333, np.ones((3, 224, 224), dtype=np.float32), reference, 0.7),
    ]
    observations = build_observations(history, start)
    assert set(observations) == set(OBS_SHAPES)
    for key, shape in OBS_SHAPES.items():
        assert observations[key].shape == (1, 2, *shape)
    np.testing.assert_allclose(observations["robot0_eef_pos"][0, -1], 0, atol=1e-12)
    relative = np.eye(4)
    relative[:3, :3] = Rotation.from_euler("xyz", [-0.4, 0.1, 0.3]).as_matrix()
    relative[:3, 3] = [0.1, 0.2, -0.05]
    actions = np.tile(np.r_[mat_to_pose10d(relative), 1.4], (16, 1))
    sequence = decode_actions(actions, InferenceContext(reference, 33_333_333, 0, 1))
    expected = reference @ relative
    np.testing.assert_allclose(sequence.positions, np.tile(expected[:3, 3], (16, 1)))
    np.testing.assert_allclose(
        Rotation.from_quat(sequence.quaternions).as_matrix(),
        np.tile(expected[:3, :3], (16, 1, 1)), atol=1e-12,
    )
    np.testing.assert_array_equal(sequence.gripper_openness, np.ones(16))
    np.testing.assert_allclose(sequence.time_from_start, np.arange(16) / 30)
    with pytest.raises(ValueError):
        decode_actions(np.full((16, 10), np.nan), InferenceContext(reference, 0, 0, 1))


def test_sync_missing_stale_gap_and_reset():
    """缺流、断续历史和过期窗口不进入推理；重置后只使用新 episode。"""
    buffer = ObservationBuffer()
    now = 1_000_000_000
    _add_frame(buffer, now - 70_000_000, with_gripper=False)
    assert buffer.poll(now) is None
    buffer.add_gripper(now - 70_000_000, 0.3)
    _add_frame(buffer, now - 36_666_667)
    result = buffer.poll(now)
    assert result is not None and len(result) == 2
    assert buffer.poll(now) is None
    _add_frame(buffer, now - 2_000_000)
    assert buffer.poll(now) is not None
    _add_frame(buffer, now + 100_000_000)
    assert buffer.poll(now + 100_000_000) is None
    assert buffer.poll(now + 400_000_000) is None
    buffer.reset()
    assert buffer.episode_id == 1 and buffer.start_pose is None
    _add_frame(buffer, now + 500_000_000)
    assert buffer.poll(now + 500_000_000) is None
    with pytest.raises(ValueError):
        buffer.add_gripper(now, 1.5)


def test_checkpoint_and_urdf_contract(tmp_path):
    """合法摘要通过，错误末端、历史长度或 URDF 内容必须被拒绝。"""
    path = tmp_path / "arm.urdf"
    path.write_text("<robot name='test'/>")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    config = _contract_config(digest)
    assert validate_contract(config) == digest
    validate_urdf(path, config)
    config.task.contract.end_frame = "tool"
    with pytest.raises(ValueError):
        validate_contract(config)
    config.task.contract.end_frame = "Link7"
    config.shape_meta.action.horizon = 8
    with pytest.raises(ValueError):
        validate_contract(config)
    path.write_text("<robot name='changed'/>")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_urdf(path, config)


def _transform(rotvec, translation):
    """构造齐次变换。"""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def test_tcp_contract_is_accepted_only_with_matching_tool_offset(tmp_path):
    """TCP 契约须声明 tool_offset=deployment_calibration；与 Link7 的单位阵声明不可混用。"""
    config = _contract_config("a" * 64)
    assert contract_end_frame(config) == "Link7"
    config.task.contract.end_frame = "tcp"
    with pytest.raises(ValueError, match="Link7/TCP"):
        validate_contract(config)
    config.task.contract.tool_offset = "deployment_calibration"
    assert validate_contract(config) == "a" * 64
    assert contract_end_frame(config) == "tcp"


def test_tcp_round_trip_reproduces_demonstrated_tcp_motion():
    """TCP 观测 = FK(Link7)·T_Link7_TCP；解码后的 Link7 目标再乘 T_Link7_TCP 等于参考 TCP·相对动作。"""
    link7_from_tcp = _transform([0.02, -0.03, 3.1], [-0.002, -0.002, 0.226])
    base_from_link7 = _transform([0.3, -0.5, 0.2], [0.4, 0.1, 0.5])
    reference_tcp = base_from_link7 @ link7_from_tcp
    relative = _transform([0.1, -0.2, 0.05], [0.03, -0.01, 0.02])
    actions = np.tile(np.r_[mat_to_pose10d(relative), 0.5], (16, 1))
    context = InferenceContext(reference_tcp, 0, 0, 1, link7_from_tcp)
    sequence = decode_actions(actions, context)
    target_link7 = np.eye(4)
    target_link7[:3, :3] = Rotation.from_quat(sequence.quaternions[0]).as_matrix()
    target_link7[:3, 3] = sequence.positions[0]
    np.testing.assert_allclose(target_link7 @ link7_from_tcp, reference_tcp @ relative, atol=1e-9)
    # 纯绕 TCP x 轴旋转时 TCP 原点不动，而约 226 mm 外的 Link7 目标随之平移。
    spin = np.tile(np.r_[mat_to_pose10d(_transform([0.3, 0, 0], [0, 0, 0])), 0.5], (16, 1))
    spun = decode_actions(spin, context)
    spun_link7 = np.eye(4)
    spun_link7[:3, :3] = Rotation.from_quat(spun.quaternions[0]).as_matrix()
    spun_link7[:3, 3] = spun.positions[0]
    np.testing.assert_allclose((spun_link7 @ link7_from_tcp)[:3, 3], reference_tcp[:3, 3], atol=1e-9)
    assert np.linalg.norm(spun.positions[0] - base_from_link7[:3, 3]) > 0.05


def test_load_deploy_tcp_and_axis_check(tmp_path):
    """部署外参组合 inverse(T_cam_end)·T_d_cam_tcp；两端 TCP 轴向相差过大时拒绝。"""
    camera_from_link7 = _transform([3.1, 0.05, 0.0], [0.0, 0.09, -0.016])
    camera_from_tcp = _transform([1.62, 0.02, 0.01], [-0.002, 0.074, 0.21])
    end = tmp_path / "T_cam_end.yaml"
    end.write_text(yaml.safe_dump({"cam0": {"T_cam_hand": camera_from_link7.tolist()}}))
    tcp = tmp_path / "T_d_cam_tcp.yaml"
    tcp.write_text(yaml.safe_dump({"T_d_cam_tcp": camera_from_tcp.tolist()}))
    link7_from_tcp, loaded = load_deploy_tcp(end, tcp)
    np.testing.assert_allclose(link7_from_tcp, np.linalg.inv(camera_from_link7) @ camera_from_tcp,
                               atol=1e-12)
    np.testing.assert_allclose(loaded, camera_from_tcp, atol=1e-12)
    umi = camera_from_tcp @ _transform([0.05, 0, 0], [0.0, 0.0, 0.0085])
    offset_mm, angle = compare_tcp_frames(umi, camera_from_tcp)
    assert offset_mm == pytest.approx(8.5, abs=0.5) and angle == pytest.approx(np.degrees(0.05))
    flipped = camera_from_tcp @ _transform([0, 0, np.pi / 2], [0, 0, 0])
    with pytest.raises(ValueError, match="axis convention"):
        compare_tcp_frames(flipped, camera_from_tcp)
    tcp.write_text(yaml.safe_dump({"T_u_cam_tcp": camera_from_tcp.tolist()}))
    with pytest.raises(ValueError, match="T_d_cam_tcp"):
        load_deploy_tcp(end, tcp)
