"""验证 TCP 观测构造、TCP 解码与 Link7 换算、结果时效、序列校验与配置解析。"""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dp_infer.common import InferenceContext, Observation
from dp_infer_lerobot import core
from dp_infer_lerobot.config import COMMON, IMAGE_KEY, STATE_KEY, SUPPORTED, VARIANTS, DiffusionConfig


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


def test_decode_then_correct_converts_tcp_to_link7():
    """解码得到 base→TCP；右乘 C=inverse(T_Link7_TCP) 得到 Link7；时间为 k/30，夹爪截断。"""
    link7_from_tcp = _pose([0.0, 0.0, np.pi / 2], [0.0, 0.0, 0.2])
    tcp = [_pose([0, 0.1 * k, 0], [0.4 + 0.01 * k, 0, 0.3]) for k in range(core.ACTION_STEPS)]
    actions = np.stack([np.r_[core.pose_to_raw(pose), 0.5] for pose in tcp])
    actions[-1, 6] = 1.2
    context = InferenceContext(np.eye(4), 0, 0, 1)
    decoded = core.decode_actions(actions, context)
    np.testing.assert_allclose(decoded.positions, [pose[:3, 3] for pose in tcp], atol=1e-12)
    sequence = core.correct_sequence(decoded, np.linalg.inv(link7_from_tcp))
    for k, pose in enumerate(tcp):
        expected = pose @ np.linalg.inv(link7_from_tcp)
        np.testing.assert_allclose(sequence.positions[k], expected[:3, 3], atol=1e-12)
        rotation = Rotation.from_quat(sequence.quaternions[k]).as_matrix()
        np.testing.assert_allclose(rotation, expected[:3, :3], atol=1e-9)
    np.testing.assert_allclose(sequence.time_from_start, np.arange(8) / 30)
    assert sequence.gripper_openness[-1] == 1.0
    with pytest.raises(ValueError, match="actions"):
        core.decode_actions(actions[:7], context)
    with pytest.raises(ValueError, match="rigid"):
        core.correct_sequence(decoded, np.diag([2.0, 1, 1, 1]))


def test_result_is_usable_until_last_action():
    """采集时间之前或末点（7/30 秒）到期后的结果都不可用。"""
    actions = np.tile(np.r_[0.4, 0, 0.3, 0, 0, 0, 0.5], (core.ACTION_STEPS, 1))
    sequence = core.decode_actions(actions)
    context = InferenceContext(np.eye(4), 1_000, 0, 1)
    assert not core.result_is_usable(sequence, context, 999)
    assert core.result_is_usable(sequence, context, 1_000)
    assert core.result_is_usable(sequence, context, 1_000 + core.LAST_POINT_NS - 1)
    assert not core.result_is_usable(sequence, context, 1_000 + core.LAST_POINT_NS)


def test_longer_sequences_keep_eighth_point_expiry():
    """多发布 horizon 后续点（最多 15）时时间仍为 k/30；过期判定仍以第 8 点为准，16 点被拒绝。"""
    actions = np.tile(np.r_[0.4, 0, 0.3, 0, 0, 0, 0.5], (core.MAX_ACTION_STEPS, 1))
    actions[:, 0] += np.arange(core.MAX_ACTION_STEPS) * 0.001
    sequence = core.decode_actions(actions)
    assert len(sequence.positions) == core.MAX_ACTION_STEPS
    np.testing.assert_allclose(sequence.time_from_start, np.arange(core.MAX_ACTION_STEPS) / 30)
    corrected = core.correct_sequence(sequence, np.eye(4))
    np.testing.assert_allclose(corrected.positions[:, 0], actions[:, 0])
    context = InferenceContext(np.eye(4), 1_000, 0, 1)
    assert not core.result_is_usable(sequence, context, 1_000 + core.LAST_POINT_NS)
    with pytest.raises(ValueError, match="actions"):
        core.decode_actions(np.vstack([actions, actions[:1]]))


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



def test_config_accepts_v5_variant_and_rejects_mixed_fields(tmp_path):
    """dp_umi_v5（窄 U-Net、DDIM 50 步）可解析且步数上限为 50；两个变体字段混搭被拒绝。"""
    raw = dict(COMMON, **VARIANTS["dp_umi_v5"], num_inference_steps=16, use_amp=False, use_peft=False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    config = DiffusionConfig.from_file(path)
    assert config.variant == "dp_umi_v5" and config.down_dims == [256, 512, 1024]
    assert config.noise_scheduler_type == "DDIM" and config.num_inference_steps == 16
    assert DiffusionConfig.from_file(path, "DDIM", 50).num_inference_steps == 50
    with pytest.raises(ValueError, match=r"\[1, 50\]"):
        DiffusionConfig.from_file(path, "DDIM", 51)
    path.write_text(json.dumps(dict(raw, down_dims=VARIANTS["dp_umi_v4"]["down_dims"])))
    with pytest.raises(ValueError, match="down_dims"):
        DiffusionConfig.from_file(path)


def test_episode_start_observation_and_action_roundtrip():
    """非单位起点下输入与动作互逆，TCP 偏移仅在发布前应用一次。"""
    start = _pose([0.2, -0.3, 0.8], [0.4, -0.1, 0.3])
    relative = _pose([0.1, 0.2, -0.1], [0.02, 0.03, -0.01])
    image = np.zeros(core.IMAGE_SHAPE, dtype=np.float32)
    history = [Observation(1, image, start, 0.749), Observation(2, image, start @ relative, 0.75)]
    state = core.build_observations(history, start, 'episode_start_tcp')[STATE_KEY][0]
    np.testing.assert_allclose(state[0], np.zeros(7), atol=1e-7)
    np.testing.assert_allclose(core.raw_to_pose(state[1]), relative, atol=1e-7)
    assert state[1, 6] == 1
    values = np.tile(state[1], (8, 1))
    context = InferenceContext(start, 2, 0, 1)
    sequence = core.decode_actions(values, context, 'episode_start_tcp')
    offset = _pose([0, 0, 0.2], [0, 0, 0.195])
    result = core.correct_sequence(sequence, np.linalg.inv(offset))
    expected = start @ relative @ np.linalg.inv(offset)
    np.testing.assert_allclose(result.positions[0], expected[:3, 3], atol=1e-7)
    np.testing.assert_allclose(Rotation.from_quat(result.quaternions[0]).as_matrix(), expected[:3, :3], atol=1e-7)


def test_episode_first_frame_padding_and_reset():
    """首帧仅提供一次，回合内起点不随窗口变化，重置后重新捕获。"""
    from dp_infer.synchronizer import ObservationBuffer
    buffer = ObservationBuffer(pad_first_observation=True)
    image = np.zeros(core.IMAGE_SHAPE, dtype=np.float32)
    start = _pose([0, 0, 0.2], [0.1, 0.2, 0.3])
    for stamp, pose in [(1_000_000_000, start), (1_033_333_333, np.eye(4))]:
        buffer.add_image(stamp, image)
        buffer.add_pose(stamp, pose)
        buffer.add_gripper(stamp, 1.0)
        window = buffer.poll(stamp)
        assert window is not None
        np.testing.assert_allclose(buffer.start_pose, start)
        assert buffer.poll(stamp) is None
    buffer.reset()
    assert buffer.start_pose is None
    stamp = 2_000_000_000
    buffer.add_image(stamp, image)
    buffer.add_pose(stamp, np.eye(4))
    buffer.add_gripper(stamp, 0.0)
    first, second = buffer.poll(stamp)
    assert first is second
    np.testing.assert_allclose(buffer.start_pose, np.eye(4))


def test_episode_first_poll_with_backlog():
    """首次轮询积压两个同步时刻时先补回合首帧，再提供真实历史窗口。"""
    from dp_infer.synchronizer import ObservationBuffer

    buffer = ObservationBuffer(pad_first_observation=True)
    for base in (1_000_000_000, 2_000_000_000):
        for stamp in (base, base + 33_333_333):
            buffer.add_image(stamp, np.zeros(core.IMAGE_SHAPE, dtype=np.float32))
            buffer.add_pose(stamp, np.eye(4))
            buffer.add_gripper(stamp, 1.0)
        first, duplicate = buffer.poll(stamp)
        assert first is duplicate and first.stamp_ns == base
        previous, latest = buffer.poll(stamp)
        assert previous is first and latest.stamp_ns == stamp
        assert buffer.poll(stamp) is None
        buffer.reset()
