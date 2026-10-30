"""验证 TCP 观测构造、动作到 Link7 的换算、序列校验与配置解析。"""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dp_infer.core import InferenceContext, Observation
from dp_infer_lerobot import core
from dp_infer_lerobot.config import IMAGE_KEY, STATE_KEY, SUPPORTED, DiffusionConfig


def _pose(rotvec, translation):
    """由轴角和平移构造 4×4 位姿。"""
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    pose[:3, 3] = translation
    return pose


def test_pose_raw_round_trip():
    """位姿与 [xyz, 轴角] 互转无损。"""
    pose = _pose([0.2, -0.4, 1.1], [0.3, -0.1, 0.5])
    np.testing.assert_allclose(core.raw_to_pose(core.pose_to_raw(pose)), pose, atol=1e-12)


def test_build_observations_layout_and_validation():
    """两帧观测转为 [1,2,3,224,224] 图像与 [1,2,7] 原始状态；越界夹爪被拒绝。"""
    image = np.full((3, 224, 224), 0.5, dtype=np.float32)
    first = Observation(1, image, _pose([0, 0, 0.1], [0.1, 0, 0.2]), 0.8)
    second = Observation(2, image, _pose([0, 0, 0.2], [0.1, 0.01, 0.2]), 0.7)
    observations = core.build_observations([first, second])
    assert observations[IMAGE_KEY].shape == (1, 2, 3, 224, 224)
    np.testing.assert_allclose(observations[STATE_KEY][0, 1], [0.1, 0.01, 0.2, 0, 0, 0.2, 0.7], atol=1e-7)
    with pytest.raises(ValueError, match="state"):
        core.build_observations([first, Observation(2, image, second.pose, 1.5)])
    with pytest.raises(ValueError, match="Exactly two"):
        core.build_observations([first])


def test_decode_actions_converts_tcp_to_link7():
    """TCP 目标右乘 inverse(T_Link7_TCP) 得到 Link7；时间为 k/30，夹爪截断。"""
    link7_from_tcp = _pose([0.0, 0.0, np.pi / 2], [0.0, 0.0, 0.2])
    tcp = [_pose([0, 0.1 * k, 0], [0.4 + 0.01 * k, 0, 0.3]) for k in range(core.ACTION_STEPS)]
    actions = np.stack([np.r_[core.pose_to_raw(pose), 0.5] for pose in tcp])
    actions[-1, 6] = 1.2
    context = InferenceContext(np.eye(4), 0, 0, 1, link7_from_tcp)
    sequence = core.decode_actions(actions, context)
    for k, pose in enumerate(tcp):
        expected = pose @ np.linalg.inv(link7_from_tcp)
        np.testing.assert_allclose(sequence.positions[k], expected[:3, 3], atol=1e-12)
        rotation = Rotation.from_quat(sequence.quaternions[k]).as_matrix()
        np.testing.assert_allclose(rotation, expected[:3, :3], atol=1e-9)
    np.testing.assert_allclose(sequence.time_from_start, np.arange(8) / 30)
    assert sequence.gripper_openness[-1] == 1.0
    with pytest.raises(ValueError, match="actions"):
        core.decode_actions(actions[:7], context)


def test_config_parses_contract_and_rejects_changes(tmp_path):
    """接受 dp_umi_v4 配置并派生 crop212/10 维；架构字段变化或步数越界被拒绝。"""
    raw = dict(SUPPORTED, num_inference_steps=None, use_amp=False, use_peft=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    config = DiffusionConfig.from_file(path)
    assert config.crop_shape == (212, 212) and config.model_action_dim == 10
    assert config.noise_scheduler_type == "DDPM" and config.num_inference_steps == 100
    assert config.pretrained_backbone_weights is None
    ddim = DiffusionConfig.from_file(path, "DDIM", 16)
    assert ddim.noise_scheduler_type == "DDIM" and ddim.num_inference_steps == 16
    with pytest.raises(ValueError, match="num_inference_steps"):
        DiffusionConfig.from_file(path, "DDIM", 101)
    path.write_text(json.dumps(dict(raw, crop_ratio=1.0)))
    with pytest.raises(ValueError, match="crop_ratio"):
        DiffusionConfig.from_file(path)


def test_episode_start_coordinates_and_gripper():
    """非平凡起点和工具偏移下，观测与动作按固定起点往返。"""
    start = _pose([0.3, -0.2, 0.7], [0.4, -0.3, 0.2])
    relative = _pose([-0.1, 0.4, 0.2], [0.1, -0.05, 0.08])
    tool = _pose([0.1, 0.2, -0.1], [0.02, 0.03, 0.15])
    image = np.zeros((3, 224, 224), dtype=np.float32)
    history = [Observation(1, image, start, 0.7499), Observation(2, image, start @ relative, 0.75)]
    state = core.build_observations(history, start, "episode_start_tcp")[STATE_KEY][0]
    np.testing.assert_allclose(core.raw_to_pose(state[0]), np.eye(4), atol=1e-7)
    np.testing.assert_allclose(core.raw_to_pose(state[1]), relative, atol=1e-7)
    np.testing.assert_array_equal(state[:, 6], [0, 1])
    actions = np.tile(np.r_[core.pose_to_raw(relative), 0.42], (8, 1))
    sequence = core.decode_actions(actions, InferenceContext(start, 2, 0, 1, tool), "episode_start_tcp")
    expected = start @ relative @ np.linalg.inv(tool)
    np.testing.assert_allclose(sequence.positions, np.tile(expected[:3, 3], (8, 1)), atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(sequence.quaternions[0]).as_matrix(), expected[:3, :3])
    np.testing.assert_allclose(sequence.gripper_openness, 0.42)
    for invalid in (float("nan"), -0.01, 1.01):
        with pytest.raises(ValueError, match="gripper"):
            core.build_observations([history[0], Observation(2, image, start, invalid)], start, "episode_start_tcp")


def test_episode_start_config_must_match_explicit_contract(tmp_path):
    """7 维检查点必须由显式起点契约加载，不能由配置字段单独推断。"""
    raw = dict(SUPPORTED, use_relative_pose=False, num_inference_steps=None)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    cfg = DiffusionConfig.from_file(path, pose_frame="episode_start_tcp")
    assert cfg.model_state_dim == cfg.model_action_dim == 7
    with pytest.raises(ValueError, match="use_relative_pose"):
        DiffusionConfig.from_file(path)
    raw["use_relative_pose"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="use_relative_pose"):
        DiffusionConfig.from_file(path, pose_frame="episode_start_tcp")


def test_first_observation_padding_and_reset():
    """首帧只补齐一次，后续真实历史、缺流恢复与重置使用各自起点。"""
    from dp_infer.synchronizer import ObservationBuffer

    buffer = ObservationBuffer(pad_first_observation=True)
    image = np.zeros((3, 224, 224), np.float32)
    start = _pose([0, 0, 0.4], [0.2, 0.1, 0.3])
    buffer.add_image(100_000_000, image)
    assert buffer.poll(100_000_000) is None
    buffer.add_pose(100_000_000, start)
    buffer.add_gripper(100_000_000, 0.75)
    first = buffer.poll(100_000_000)
    assert first[0] is first[1] and len(buffer.history) == 1
    assert buffer.poll(100_000_000) is None
    later = 133_333_333
    buffer.add_image(later, image)
    buffer.add_pose(later, start)
    buffer.add_gripper(later, 0.5)
    assert buffer.poll(later)[0] is first[0]
    np.testing.assert_array_equal(buffer.start_pose, start)
    buffer.reset()
    assert buffer.start_pose is None and buffer.episode_id == 1
    assert buffer.poll(later) is None
