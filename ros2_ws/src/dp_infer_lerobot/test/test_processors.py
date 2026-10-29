"""验证处理链契约解析、归一化公式与 SE(3) 相对/绝对位姿转换（不加载真实模型）。"""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

torch = pytest.importorskip("torch")

from dp_infer_lerobot import relative_pose  # noqa: E402
from dp_infer_lerobot.config import IMAGE_KEY, STATE_KEY  # noqa: E402
from dp_infer_lerobot.processors import (POST_STEPS, PRE_STEPS, RAW_FEATURES,  # noqa: E402
                                         TorchProcessors, read_processor_spec, validate_stats)

NORM = {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}


def _write_checkpoint(root, stats_pre, stats_post):
    """写一份与 dp_umi_v4 相同结构的处理链 JSON 及统计量。"""
    from safetensors.numpy import save_file

    pre_configs = [{"rename_map": {}}, {}, {"device": "cpu", "float_dtype": None},
                   {"enabled": True, "state_pose_starts": [0], "action_pose_starts": None, "n_obs_steps": 2},
                   {"eps": 1e-8, "features": RAW_FEATURES, "norm_map": NORM}]
    pre = {"name": "policy_preprocessor",
           "steps": [{"registry_name": name, "config": config} for name, config in zip(PRE_STEPS, pre_configs)]}
    pre["steps"][4]["state_file"] = "pre.safetensors"
    post_configs = [{"eps": 1e-8, "features": {"action": RAW_FEATURES["action"]}, "norm_map": NORM},
                    {"enabled": True, "state_pose_starts": [0], "action_pose_starts": None},
                    {"device": "cpu", "float_dtype": None}]
    post = {"name": "policy_postprocessor",
            "steps": [{"registry_name": name, "config": config} for name, config in zip(POST_STEPS, post_configs)]}
    post["steps"][0]["state_file"] = "post.safetensors"
    (root / "policy_preprocessor.json").write_text(json.dumps(pre))
    (root / "policy_postprocessor.json").write_text(json.dumps(post))
    save_file(stats_pre, str(root / "pre.safetensors"))
    save_file(stats_post, str(root / "post.safetensors"))
    return pre, post


def _stats():
    """构造含一维零跨度（当前帧单位位姿）的统计量。"""
    minimum = np.linspace(-1, 0, 10).astype(np.float32)
    maximum = (minimum + np.linspace(0.5, 2, 10)).astype(np.float32)
    maximum[0] = minimum[0]  # 零跨度维度，分母应替换为 eps。
    pre = {f"{IMAGE_KEY}.mean": np.full((3, 1, 1), 0.4, np.float32),
           f"{IMAGE_KEY}.std": np.full((3, 1, 1), 0.2, np.float32),
           f"{STATE_KEY}.min": minimum, f"{STATE_KEY}.max": maximum}
    post = {"action.min": minimum, "action.max": maximum + 1}
    return pre, post


def test_spec_accepts_contract_and_rejects_changes(tmp_path):
    """接受训练保存的处理链；步骤顺序或相对位姿配置改变即拒绝。"""
    pre_stats, post_stats = _stats()
    pre, post = _write_checkpoint(tmp_path, pre_stats, post_stats)
    assert read_processor_spec(tmp_path)[0].name == "pre.safetensors"
    assert read_processor_spec(tmp_path, inverse=True)[0].name == "post.safetensors"
    pre["steps"][3]["config"]["n_obs_steps"] = 1
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps(pre))
    with pytest.raises(ValueError, match="relative pose"):
        read_processor_spec(tmp_path)
    post["steps"] = post["steps"][::-1]
    (tmp_path / "policy_postprocessor.json").write_text(json.dumps(post))
    with pytest.raises(ValueError, match="pipeline"):
        read_processor_spec(tmp_path, inverse=True)


def test_validate_stats_rejects_wrong_shape():
    """统计量必须是 10 维相对空间；7 维原始空间统计被拒绝。"""
    pre, _ = _stats()
    pre[f"{STATE_KEY}.min"] = np.zeros(7, np.float32)
    with pytest.raises(ValueError, match="statistic"):
        validate_stats(pre)


def test_relative_absolute_round_trip_and_identity_frame():
    """相对化后当前帧为精确单位位姿，相对动作经绝对还原回到原值。"""
    rng = np.random.default_rng(0)
    rotvec = Rotation.random(3, random_state=1).as_rotvec()
    raw = np.concatenate([rng.normal(size=(3, 3)) * 0.2, rotvec, rng.uniform(size=(3, 1))], axis=1)
    raw = torch.tensor(raw[None], dtype=torch.float64)  # (1,3,7)
    reference = raw[:, -1]
    relative = relative_pose.set_frame_to_identity(
        relative_pose.to_relative_pose(raw, reference, [0], [0]), -1, [0])
    torch.testing.assert_close(relative[0, -1, :9], torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0.], dtype=torch.float64))
    restored = relative_pose.to_absolute_pose(relative, reference, [0], [0])
    torch.testing.assert_close(restored[..., :3], raw[..., :3])
    torch.testing.assert_close(restored[..., 6], raw[..., 6])
    expected = Rotation.from_rotvec(raw[0, :, 3:6].numpy())
    actual = Rotation.from_rotvec(restored[0, :, 3:6].numpy())
    assert np.allclose((expected.inv() * actual).magnitude(), 0, atol=1e-6)


def test_relative_pose_matches_matrix_definition():
    """相对位姿等于 inv(T_ref) @ T，rot6d 为旋转矩阵前两行。"""
    reference = np.r_[0.1, -0.2, 0.3, Rotation.from_euler("xyz", [0.3, -0.2, 0.5]).as_rotvec(), 0.5]
    value = np.r_[0.15, -0.1, 0.25, Rotation.from_euler("xyz", [0.1, 0.2, 0.9]).as_rotvec(), 0.7]
    result = relative_pose.to_relative_pose(torch.tensor(value[None]), torch.tensor(reference[None]), [0], [0])[0]

    def matrix(raw):
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = Rotation.from_rotvec(raw[3:6]).as_matrix(), raw[:3]
        return pose

    expected = np.linalg.inv(matrix(reference)) @ matrix(value)
    np.testing.assert_allclose(result[:3].numpy(), expected[:3, 3], atol=1e-9)
    np.testing.assert_allclose(result[3:9].numpy(), expected[:2, :3].reshape(-1), atol=1e-9)
    assert result[9].item() == pytest.approx(0.7)


def test_torch_processors_formulas(tmp_path):
    """MIN_MAX/MEAN_STD 与上游一致：零跨度用 eps；反归一化后还原绝对位姿。"""
    pre_stats, post_stats = _stats()
    _write_checkpoint(tmp_path, pre_stats, post_stats)
    processors = TorchProcessors(tmp_path, "cpu")
    image = torch.full((1, 2, 3, 224, 224), 0.6)
    state = torch.tensor([[[0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.8], [0.1, 0.01, 0.2, 0.0, 0.0, 0.1, 0.8]]])
    batch, reference = processors.preprocess(image, state)
    torch.testing.assert_close(batch[IMAGE_KEY], torch.full_like(image, (0.6 - 0.4) / (0.2 + 1e-8)))
    minimum = torch.from_numpy(pre_stats[f"{STATE_KEY}.min"])
    denominator = torch.from_numpy(pre_stats[f"{STATE_KEY}.max"]) - minimum
    denominator[0] = 1e-8
    relative = relative_pose.set_frame_to_identity(relative_pose.to_relative_pose(state, state[:, -1], [0], [0]), -1, [0])
    torch.testing.assert_close(batch[STATE_KEY], 2 * (relative - minimum) / denominator - 1)
    torch.testing.assert_close(reference, state[:, -1])
    actions = torch.zeros(1, 8, 10)
    restored = processors.postprocess(actions, reference)
    assert restored.shape == (1, 8, 7)
