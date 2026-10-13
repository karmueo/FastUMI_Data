"""验证示教 Link7 格式的固定夹爪归一化契约和数据/训练频率契约检查。"""

import numpy as np
import pytest
import torch
import zarr

from diffusion_policy.dataset.umi_dataset import UmiDataset, uses_fixed_gripper_range
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import check_dataset_frequency
from test_vr_umi import umi_shape


LINK7_ATTRS = {
    "format": "fastumi-rm75-link7-v1", "complete": True, "frequency": 30,
    "normalization_contract": "fixed-gripper-0-1-v1", "gripper_lower": 0.0, "gripper_upper": 1.0,
    "urdf_sha256": "a" * 64,
}


def link7_zarr(path, gripper, attrs=None):
    """构造三段 Link7 Zarr；``gripper`` 为每帧开度。"""
    count = len(gripper)
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(LINK7_ATTRS if attrs is None else attrs)
    data = root.create_group("data")
    t = np.arange(count, dtype=np.float64)
    position = np.c_[t / 100, np.sin(t / 10) / 50, np.zeros(count)].astype("f4")
    rotvec = np.c_[np.zeros(count), np.zeros(count), t / 200].astype("f4")
    data.create_dataset("camera0_rgb", data=np.zeros((count, 8, 8, 3), dtype="u1"))
    data.create_dataset("robot0_eef_pos", data=position)
    data.create_dataset("robot0_eef_rot_axis_angle", data=rotvec)
    data.create_dataset("robot0_gripper_width", data=np.asarray(gripper, dtype="f4")[:, None])
    starts = np.repeat([0, 20, 40], 20)
    data.create_dataset("robot0_demo_start_pose", data=np.c_[position[starts], rotvec[starts]])
    data.create_dataset("action", data=np.c_[position, rotvec, gripper].astype("f4"))
    data.create_dataset("timestamp", data=np.tile(np.arange(20) / 30, 3))
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=np.array([20, 40, 60]))
    meta.create_dataset("episode_names", data=np.array(["r/episode_0000", "r/episode_0001", "r/episode_0002"]))
    return path


def dataset(path):
    """以相对位姿和固定划分加载数据集。"""
    return UmiDataset(umi_shape(), str(path),
                      pose_repr={"obs_pose_repr": "relative", "action_pose_repr": "relative"},
                      train_episode_indices=[0, 1], val_episode_indices=[2],
                      start_pose_noise_std=0, normalizer_num_workers=0)


def gripper_params(normalizer):
    """返回观测与动作夹爪维的 (scale, offset)。"""
    obs = normalizer["robot0_gripper_width"].params_dict
    action = normalizer["action"].params_dict
    return (obs["scale"].tolist(), obs["offset"].tolist(),
            action["scale"][-1].item(), action["offset"][-1].item())


@pytest.mark.parametrize("gripper", [np.linspace(0.3, 0.5, 60), np.full(60, 0.4)])
def test_link7_gripper_uses_fixed_zero_one_bounds(tmp_path, gripper) -> None:
    """部分开度范围或恒定开度都使用固定 [0,1] 输入边界，映射到 [-1,1]。"""
    normalizer = dataset(link7_zarr(tmp_path / "link7.zarr", gripper)).get_normalizer()
    assert gripper_params(normalizer) == ([2.0], [-1.0], 2.0, -1.0)
    values = torch.tensor([[0.0], [0.25], [1.0]])
    normalized = normalizer["robot0_gripper_width"].normalize(values)
    np.testing.assert_allclose(normalized.numpy()[:, 0], [-1.0, -0.5, 1.0], atol=1e-6)
    np.testing.assert_allclose(normalizer["robot0_gripper_width"].unnormalize(normalized), values, atol=1e-6)


def test_link7_gripper_matches_v2_checkpoint_scaling(tmp_path) -> None:
    """新格式夹爪 normalizer 与 rm75-umi-pose-v2 的固定缩放一致，位置仍按数据统计。"""
    gripper = np.linspace(0.3, 0.5, 60)
    link7 = dataset(link7_zarr(tmp_path / "link7.zarr", gripper)).get_normalizer()
    v2_attrs = dict(LINK7_ATTRS, format="rm75-umi-pose-v2", normalization_contract="urdf-joint-limits-v1")
    v2 = dataset(link7_zarr(tmp_path / "v2.zarr", gripper, v2_attrs)).get_normalizer()
    assert gripper_params(link7) == gripper_params(v2)
    legacy_attrs = dict(LINK7_ATTRS, format="rm75-umi-pose-v1")
    legacy = dataset(link7_zarr(tmp_path / "v1.zarr", gripper, legacy_attrs)).get_normalizer()
    assert gripper_params(legacy) != gripper_params(link7)


@pytest.mark.parametrize("change", [
    {"normalization_contract": None},
    {"normalization_contract": "urdf-joint-limits-v1"},
    {"gripper_upper": 2.0},
    {"gripper_lower": None},
    {"gripper_lower": True},
])
def test_link7_missing_or_conflicting_contract_is_rejected(change) -> None:
    """契约缺失或冲突时直接失败。"""
    attrs = dict(LINK7_ATTRS)
    for key, value in change.items():
        if value is None:
            attrs.pop(key)
        else:
            attrs[key] = value
    with pytest.raises(ValueError):
        uses_fixed_gripper_range(attrs)


def test_other_formats_keep_their_normalization() -> None:
    """v2 仍要求 URDF 契约，其余格式使用数据统计边界。"""
    assert uses_fixed_gripper_range({"format": "rm75-umi-pose-v2",
                                     "normalization_contract": "urdf-joint-limits-v1"})
    with pytest.raises(ValueError):
        uses_fixed_gripper_range({"format": "rm75-umi-pose-v2"})
    assert not uses_fixed_gripper_range({"format": "rm75-umi-pose-v1"})
    assert not uses_fixed_gripper_range({})


def test_dataset_frequency_must_match_training_contract() -> None:
    """新格式必须声明 30 Hz 且与 task.contract.frequency_hz 一致。"""
    contract = {"frequency_hz": 30}
    check_dataset_frequency(LINK7_ATTRS, contract)
    with pytest.raises(ValueError, match="does not match"):
        check_dataset_frequency(dict(LINK7_ATTRS, frequency=20), contract)
    with pytest.raises(ValueError, match="requires"):
        check_dataset_frequency({k: v for k, v in LINK7_ATTRS.items() if k != "frequency"}, contract)
    with pytest.raises(ValueError, match="requires"):
        check_dataset_frequency(LINK7_ATTRS, {})
    # 旧格式缺少频率属性时保持兼容，存在时同样核对。
    check_dataset_frequency({"format": "rm75-umi-pose-v2"}, contract)
    with pytest.raises(ValueError):
        check_dataset_frequency({"format": "rm75-umi-pose-v2", "frequency": 20}, contract)
