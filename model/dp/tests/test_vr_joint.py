"""验证遥操转换、因果对齐、关节采样、归一化隔离和评估指标。"""

import json
from pathlib import Path
from unittest.mock import Mock

import cv2
import h5py
import numpy as np
import pytest
import torch
import zarr
from hydra import compose, initialize_config_dir

from convert_vr_target import STREAMS, causal_indices, convert, letterbox_rgb, read_episode
from diffusion_policy.dataset.vr_joint_image_dataset import VrJointImageDataset
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    _compute_action_mse_metrics, evaluate_policy,
)


def make_episode(path, offset=0):
    """生成低帧率测试视频与不同频率状态流，包含故意损坏的无用 JSON。"""
    path.mkdir()
    times = np.arange(25) / 10 + 100
    with h5py.File(path / "proprio.hdf5", "w") as file:
        file.create_dataset("observations/images/cam_gripper_timestamp", data=times)
        for key, (group, field, dimension) in STREAMS.items():
            ts = np.arange(49) / 20 + 100
            values = np.tile(np.arange(49)[:, None] / 100, (1, dimension)).astype(np.float32)
            if key == "joint_action":
                values += 1 + offset
            elif key == "robot0_joint_pos":
                values += offset
            file.create_dataset(f"{group}/timestamp", data=ts)
            file.create_dataset(f"{group}/{field}", data=values)
    writer = cv2.VideoWriter(str(path / "gripper.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    for _ in times:
        writer.write(np.full((24, 32, 3), (10, 20, 240), dtype=np.uint8))
    writer.release()
    (path / "gripper.json").write_text("broken JSON must never be read", encoding="utf-8")


def shape_meta():
    """返回测试用 8D 动作与三路历史观测契约。"""
    return {"obs": {"camera0_rgb": {"shape": [3, 32, 32], "horizon": 2},
                    "robot0_joint_pos": {"shape": [7], "horizon": 2},
                    "robot0_gripper_position": {"shape": [1], "horizon": 2}},
            "action": {"shape": [8], "horizon": 16}}


def test_causal_alignment_and_letterbox():
    """前值保持不能选择未来样本，缩放保持 RGB 顺序与居中黑边。"""
    assert causal_indices(np.array([0., 1., 2.]), np.array([0., .9, 1., 1.9])).tolist() == [0, 0, 1, 1]
    with pytest.raises(ValueError):
        causal_indices(np.array([0., 1.]), np.array([-1.]))
    image = letterbox_rgb(np.full((12, 16, 3), (10, 20, 240), dtype=np.uint8), 32)
    assert image.shape == (32, 32, 3)
    assert image[16, 16].tolist() == [240, 20, 10]
    assert np.all(image[:4] == 0)


def test_conversion_sampling_and_train_only_normalization(tmp_path):
    """忽略 JSON、使用控制目标并验证 episode 顺序、窗口边界和训练统计隔离。"""
    source = tmp_path / "source"
    source.mkdir()
    for number in [10, 2, 0]:
        make_episode(source / f"episode_{number}", offset=number)
    output = tmp_path / "target.zarr"
    report = convert(source, output, frequency=10, image_size=32)
    assert [item["episode"] for item in report["episodes"]] == ["episode_0", "episode_2", "episode_10"]
    dataset = VrJointImageDataset(output, shape_meta(), val_ratio=1 / 3)
    assert len(dataset) == 20
    assert len(dataset.get_validation_dataset()) == 10
    split = dataset.get_split_manifest()
    assert set(split["train"]).isdisjoint(split["validation"])
    batch = dataset[0]
    assert batch["obs"]["camera0_rgb"].shape == (2, 3, 32, 32)
    assert torch.equal(batch["obs"]["camera0_rgb"][0], batch["obs"]["camera0_rgb"][1])
    assert batch["action"].shape == (16, 8)
    assert batch["action"][0, 0] - batch["obs"]["robot0_joint_pos"][0, 0] == 1
    for current, start in dataset.indices:
        end = dataset.ends[np.searchsorted(dataset.starts, start)]
        assert current + 16 <= end
    # 修改验证段为极值，训练 normalizer 不应受影响。
    normalizer_before = dataset.get_normalizer()["action"].params_dict["scale"].clone()
    root = zarr.open_group(str(output), mode="r+")
    for start, end, selected in zip(dataset.starts, dataset.ends, dataset.val_mask):
        if selected:
            root["data/action"][start:end] = 10000
    normalizer_after = dataset.get_normalizer()["action"].params_dict["scale"]
    assert torch.equal(normalizer_before, normalizer_after)
    with pytest.raises(Exception):
        convert(source, output, image_size=32)


@pytest.mark.parametrize("failure", ["nan", "timestamps", "missing", "video"])
def test_invalid_episode_fails(tmp_path, failure):
    """损坏数值、时间戳、字段或视频不能静默进入训练集。"""
    episode = tmp_path / "episode_0"
    make_episode(episode)
    with h5py.File(episode / "proprio.hdf5", "r+") as file:
        if failure == "nan":
            file["action/joint_action/position"][0, 0] = np.nan
        elif failure == "timestamps":
            file["action/joint_action/timestamp"][1] = 0
        elif failure == "missing":
            del file["action/gripper_action/position"]
        else:
            del file["observations/images/cam_gripper_timestamp"]
            file.create_dataset("observations/images/cam_gripper_timestamp", data=np.arange(26) / 10 + 100)
    with pytest.raises((ValueError, KeyError)):
        read_episode(episode)


def test_joint_metrics_and_weighted_validation():
    """8D 指标正确拆分关节与夹爪，验证按 batch 样本数加权。"""
    predicted = torch.ones(2, 16, 8)
    predicted[..., 7] = 2
    metrics = _compute_action_mse_metrics("val", predicted, torch.zeros_like(predicted), "joint8")
    assert metrics["val_action_mse_error_joint"] == 1
    assert metrics["val_action_mse_error_gripper"] == 4
    policy = Mock(side_effect=[torch.tensor(1.), torch.tensor(4.)])
    batches = [{"obs": {}, "action": torch.zeros(2, 16, 8)}, {"obs": {}, "action": torch.zeros(1, 16, 8)}]
    assert evaluate_policy(policy, batches, "cpu")["val_loss"] == 2


def test_joint_training_config():
    """Hydra 必须选中关节任务、验证 checkpoint 和离线训练配置。"""
    config_dir = Path(__file__).resolve().parents[1] / "diffusion_policy/config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="train_diffusion_unet_timm_vr_joint_workspace")
    assert cfg.task.action_layout == "joint8"
    assert cfg.task.shape_meta.action.shape == [8]
    assert cfg.training.num_epochs == 120
    assert cfg.training.enable_validation
    assert cfg.checkpoint.topk.monitor_key == "val_loss"
    assert cfg.logging.mode == "offline"


def test_encoder_eval_preprocessing_matches_inference():
    """验证和推理使用相同归一化，训练增强不能泄漏到验证。"""
    from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder

    # 最小编码器替身避免测试下载权重，并直接覆盖共享图像处理逻辑。
    encoder = TimmObsEncoder.__new__(TimmObsEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.key_transform_map = torch.nn.ModuleDict({"camera": torch.nn.Identity()})
    encoder.apply_image_normalization = True
    encoder.image_mean = (0.5, 0.5, 0.5)
    encoder.image_std = (0.25, 0.25, 0.25)
    image = torch.ones(2, 3, 4, 4)
    assert torch.equal(encoder._prepare_image("camera", image, False), torch.full_like(image, 2))
