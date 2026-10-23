"""验证独立 DP 核心的输入契约、同步、动作解码和 URDF 检查。"""

import hashlib
from types import SimpleNamespace

import cv2
import numpy as np
from omegaconf import OmegaConf
import pytest
from scipy.spatial.transform import Rotation

from dp_infer.core import (
    InferenceContext, OBS_SHAPES, Observation, build_observations,
    PolicyEngine,
    compressed_image_to_rgb, correct_sequence, decode_actions, image_to_rgb, letterbox_rgb,
    load_link7_correction, ordered_joints,
    validate_contract, validate_urdf,
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


def test_policy_engine_applies_steps_before_prediction():
    """推理入口在同一工作线程中更新策略步数并生成动作。"""
    engine = PolicyEngine.__new__(PolicyEngine)
    engine.policy = SimpleNamespace(num_inference_steps=8)
    engine.processors = []
    seen = []

    def predict_raw(_observations):
        seen.append(engine.policy.num_inference_steps)
        actions = np.tile(np.r_[np.zeros(3), [1, 0, 0, 0, 1, 0], 0.4], (1, 16, 1))
        return actions

    engine.predict_raw = predict_raw
    context = InferenceContext(np.eye(4), 0, 0, 0)
    sequence = engine.predict({}, context, 16)
    assert seen == [16] and sequence.positions.shape == (16, 3)


def test_image_stride_color_and_joint_order():
    """带行填充 BGR 图像正确转 RGB；关节按名称提取并拒绝坏值。"""
    bgr = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    padded = np.zeros((2, 12), dtype=np.uint8)
    padded[:, :9] = bgr.reshape(2, 9)
    rgb = image_to_rgb(padded.tobytes(), 2, 3, 12, "bgr8")
    np.testing.assert_array_equal(rgb, bgr[..., ::-1])
    ok, jpeg = cv2.imencode(".jpg", bgr)
    assert ok
    np.testing.assert_array_equal(
        compressed_image_to_rgb(jpeg.tobytes()),
        cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
    )
    with pytest.raises(ValueError):
        compressed_image_to_rgb(b"")
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


def _correction():
    """约 15° 绕 x、厘米级平移的刚体修正，与实测量级一致。"""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("x", 14.9, degrees=True).as_matrix()
    matrix[:3, 3] = [-0.002, 0.0055, -0.0092]
    return matrix


def test_link7_correction_file_round_trip_and_rejects_non_rigid(tmp_path):
    """空路径为单位阵；文件按 9 位小数保存后重新正交化；非刚体被拒绝。"""
    np.testing.assert_array_equal(load_link7_correction(""), np.eye(4))
    path = tmp_path / "link7_correction.yaml"
    matrix = _correction()
    path.write_text(f"link7_correction:\n  T_virtual_link7: {np.round(matrix, 9).tolist()}\n")
    loaded = load_link7_correction(path)
    np.testing.assert_allclose(loaded, matrix, atol=1e-8)
    np.testing.assert_allclose(loaded[:3, :3] @ loaded[:3, :3].T, np.eye(3), atol=1e-12)
    scaled = matrix.copy()
    scaled[:3, :3] *= 1.01
    path.write_text(f"link7_correction:\n  T_virtual_link7: {scaled.tolist()}\n")
    with pytest.raises(ValueError, match="rigid"):
        load_link7_correction(path)


def test_corrected_hold_action_targets_the_measured_link7_pose():
    """观测右乘 C⁻¹ 后，原地保持的策略动作经右乘 C 发布，目标恰为实测 Link7，不会跳变。"""
    correction = _correction()
    measured = np.eye(4)
    measured[:3, :3] = Rotation.from_euler("xyz", [170, 10, -30], degrees=True).as_matrix()
    measured[:3, 3] = [0.3, 0.2, 0.15]
    virtual = measured @ np.linalg.inv(correction)
    hold = np.tile(np.r_[np.zeros(3), [1, 0, 0, 0, 1, 0], 0.9], (16, 1))
    sequence = correct_sequence(
        decode_actions(hold, InferenceContext(virtual, 1, 0, 1)), correction)
    np.testing.assert_allclose(sequence.positions, np.tile(measured[:3, 3], (16, 1)), atol=1e-9)
    expected = Rotation.from_matrix(measured[:3, :3])
    for quaternion in sequence.quaternions:
        assert (Rotation.from_quat(quaternion) * expected.inv()).magnitude() < 1e-9
    np.testing.assert_allclose(sequence.gripper_openness, 0.9)
    identity = decode_actions(hold, InferenceContext(measured, 1, 0, 1))
    assert correct_sequence(identity, np.eye(4)) is identity
