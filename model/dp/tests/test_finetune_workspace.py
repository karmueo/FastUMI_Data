"""验证独立微调只迁移策略权重，并能完成一次小数据训练。"""

import copy
import os
import shutil
import subprocess
from pathlib import Path

import dill
import hydra
import pytest
import timm
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace,
    resumed_training_position,
)


SHAPE_META = {
    "obs": {"state": {"shape": [1]}},
    "action": {"shape": [10]},
}


class TinyPolicy(nn.Module):
    """使用一维观测和 10D 动作的轻量训练策略。"""

    def __init__(self):
        super().__init__()
        self.obs_encoder = nn.Linear(1, 1)
        self.model = nn.Linear(1, 10)
        self.normalizer = LinearNormalizer()

    @property
    def device(self):
        return self.model.weight.device

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch):
        observation = batch["obs"]["state"].mean(dim=1)
        prediction = self.model(self.obs_encoder(observation)).unsqueeze(1)
        target = self.normalizer["action"].normalize(batch["action"])
        return (prediction - target).square().mean()


class TinyDataset(BaseImageDataset):
    """两个训练窗口和两个验证窗口，动作尺度与源 checkpoint 不同。"""

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "obs": {"state": torch.tensor([[1.0], [2.0]])},
            "action": torch.full((16, 10), float(index + 2)),
        }

    def get_validation_dataset(self):
        return self

    def get_normalizer(self):
        normalizer = LinearNormalizer()
        normalizer.fit({
            "state": torch.tensor([[[1.0], [2.0]], [[1.0], [2.0]]]),
            "action": torch.stack([self[index]["action"] for index in range(len(self))]),
        })
        return normalizer


def tiny_config():
    """提供微调训练循环需要的最小 Hydra 配置。"""
    return OmegaConf.create({
        "shape_meta": SHAPE_META,
        "policy": {"_target_": "tiny.policy", "obs_encoder": {"pretrained": False}},
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1e-3},
        "training": {
            "seed": 42, "resume": False, "ckpt_path": None,
            "finetune_ckpt_path": None, "save_optimizer": False,
            "use_ema": True, "tf32": False, "lr_scheduler": "cosine",
            "lr_warmup_steps": 0, "num_epochs": 1,
            "gradient_accumulate_every": 1, "enable_validation": True,
            "best_monitor": "val_loss", "freeze_encoder": False,
            "checkpoint_every": 1, "val_every": 1, "sample_every": 0,
            "max_train_steps": 1, "max_val_steps": 1,
            "tqdm_interval_sec": 0.01, "debug": False,
            "rollout_every": 999999,
        },
        "task": {
            "dataset": {"_target_": "tiny.dataset"},
            "env_runner": None, "action_layout": "pose10",
        },
        "dataloader": {"batch_size": 1, "num_workers": 0, "shuffle": False},
        "val_dataloader": {"batch_size": 1, "num_workers": 0, "shuffle": False},
        "ema": {"_target_": "diffusion_policy.model.diffusion.ema_model.EMAModel"},
        "logging": {"project": "finetune-test", "mode": "offline"},
        "checkpoint": {
            "save_last_ckpt": True, "save_last_snapshot": False,
            "topk": {"monitor_key": "val_loss", "mode": "min", "k": 1,
                     "format_str": "epoch={epoch:04d}-val_loss={val_loss:.6f}.ckpt"},
        },
    })


@pytest.fixture
def tiny_instances(monkeypatch):
    """仅替换耗时的策略和数据集，其余训练组件使用真实实现。"""
    original_instantiate = hydra.utils.instantiate

    def instantiate(config, *args, **kwargs):
        target = config.get("_target_") if config is not None else None
        if target == "tiny.policy":
            return TinyPolicy()
        if target == "tiny.dataset":
            return TinyDataset()
        return original_instantiate(config, *args, **kwargs)

    monkeypatch.setattr(hydra.utils, "instantiate", instantiate)


def source_checkpoint(tmp_path, cfg):
    """保存具有不同 model/EMA 权重、旧优化器动量和旧进度的 checkpoint。"""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    workspace = TrainDiffusionUnetImageWorkspace(cfg, output_dir=str(source_dir))
    old_normalizer = LinearNormalizer()
    old_normalizer.fit({
        "state": torch.tensor([[[10.0]], [[20.0]]]),
        "action": torch.stack([torch.full((16, 10), value) for value in (10.0, 20.0)]),
    })
    workspace.model.set_normalizer(old_normalizer)
    workspace.ema_model.set_normalizer(old_normalizer)
    workspace.model(torch.utils.data.default_collate([TinyDataset()[0]])).backward()
    workspace.optimizer.step()
    workspace.optimizer.zero_grad()
    with torch.no_grad():
        workspace.model.model.weight.fill_(1.0)
        workspace.ema_model.model.weight.fill_(2.0)
    workspace.epoch = 8
    workspace.global_step = 30
    workspace.best_loss = 0.01
    workspace.checkpoint_next_epoch = 9
    workspace.checkpoint_next_global_step = 31
    workspace.exclude_keys = ()
    return Path(workspace.save_checkpoint(use_thread=False))


def test_finetune_prefers_ema_and_keeps_new_training_state(tmp_path, tiny_instances):
    source_cfg = tiny_config()
    source = source_checkpoint(tmp_path, source_cfg)
    cfg = tiny_config()
    cfg.optimizer.lr = 3e-5
    cfg.training.finetune_ckpt_path = str(source)
    workspace = TrainDiffusionUnetImageWorkspace(cfg, output_dir=str(tmp_path / "new"))

    assert workspace.load_finetune_checkpoint(source) == "ema_model"
    assert torch.all(workspace.model.model.weight == 2)
    assert torch.equal(workspace.model.model.weight, workspace.ema_model.model.weight)
    assert workspace.epoch == workspace.global_step == 0
    assert workspace.best_loss == float("inf")
    assert workspace.checkpoint_next_epoch is None
    assert workspace.checkpoint_next_global_step is None
    assert workspace.optimizer.state == {}
    assert workspace.optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    assert workspace.output_dir == str(tmp_path / "new")
    assert workspace.exclude_keys == ()


@pytest.mark.parametrize("mode", ["finetune", "resume"])
def test_checkpoint_initialization_skips_hub_without_changing_encoder_lr(
        tmp_path, tiny_instances, monkeypatch, mode):
    """加载完整 checkpoint 时不请求 Hub，仍按预训练编码器设置学习率。"""
    source = source_checkpoint(tmp_path, tiny_config())
    cfg = tiny_config()
    cfg.policy.obs_encoder._target_ = (
        "diffusion_policy.model.vision.timm_obs_encoder.TimmObsEncoder"
    )
    cfg.policy.obs_encoder.pretrained = True
    cfg.optimizer.lr = 3e-5
    if mode == "finetune":
        cfg.training.finetune_ckpt_path = str(source)
    else:
        cfg.training.resume = True
        cfg.training.ckpt_path = str(source)

    original_instantiate = hydra.utils.instantiate

    def assert_policy_config(config, *args, **kwargs):
        if config.get("_target_") == "tiny.policy":
            assert config.obs_encoder.pretrained is True
            assert config.obs_encoder.load_pretrained_weights is False
        return original_instantiate(config, *args, **kwargs)

    monkeypatch.setattr(hydra.utils, "instantiate", assert_policy_config)
    workspace = TrainDiffusionUnetImageWorkspace(cfg, output_dir=str(tmp_path / "new"))
    assert workspace.optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    assert workspace.optimizer.param_groups[1]["lr"] == pytest.approx(3e-6)
    assert "load_pretrained_weights" not in cfg.policy.obs_encoder


def test_timm_encoder_can_skip_hub_without_changing_frozen_architecture(monkeypatch):
    """跳过下载不改变预训练模型的冻结与 BatchNorm 处理逻辑。"""
    create_model = timm.create_model
    requested = []

    def record_model(*args, **kwargs):
        requested.append(kwargs["pretrained"])
        return create_model(*args, **kwargs)

    monkeypatch.setattr(timm, "create_model", record_model)
    encoder = TimmObsEncoder(
        shape_meta={"obs": {"camera0_rgb": {"shape": [3, 32, 32], "type": "rgb"}}},
        model_name="resnet18", pretrained=True, frozen=True,
        global_pool="", transforms=None, use_group_norm=True,
        feature_aggregation="avg", load_pretrained_weights=False,
    )
    assert requested == [False]
    assert any(isinstance(module, nn.BatchNorm2d) for module in encoder.modules())
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_finetune_falls_back_to_model_weights(tmp_path, tiny_instances):
    cfg = tiny_config()
    source = source_checkpoint(tmp_path, cfg)
    payload = torch.load(source, pickle_module=dill)
    del payload["state_dicts"]["ema_model"]
    torch.save(payload, source, pickle_module=dill)
    workspace = TrainDiffusionUnetImageWorkspace(cfg, output_dir=str(tmp_path / "new"))

    assert workspace.load_finetune_checkpoint(source) == "model"
    assert torch.all(workspace.model.model.weight == 1)
    assert torch.equal(workspace.model.model.weight, workspace.ema_model.model.weight)


def test_finetune_rejects_missing_weights_and_incompatible_shapes(tmp_path, tiny_instances):
    cfg = tiny_config()
    source = source_checkpoint(tmp_path, cfg)
    workspace = TrainDiffusionUnetImageWorkspace(cfg, output_dir=str(tmp_path / "new"))

    with pytest.raises(FileNotFoundError, match="does not exist"):
        workspace.load_finetune_checkpoint(tmp_path / "absent.ckpt")
    missing_cfg = tiny_config()
    missing_cfg.training.finetune_ckpt_path = str(tmp_path / "absent.ckpt")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        TrainDiffusionUnetImageWorkspace(missing_cfg, output_dir=str(tmp_path / "new"))

    payload = torch.load(source, pickle_module=dill)
    saved_states = payload["state_dicts"]
    payload["state_dicts"] = {}
    torch.save(payload, source, pickle_module=dill)
    with pytest.raises(ValueError, match="no model weights"):
        workspace.load_finetune_checkpoint(source)

    payload["state_dicts"] = saved_states
    changed = copy.deepcopy(cfg)
    changed.shape_meta.obs.state.shape = [2]
    workspace.cfg = changed
    torch.save(payload, source, pickle_module=dill)
    with pytest.raises(ValueError, match="observation shape"):
        workspace.load_finetune_checkpoint(source)
    changed.shape_meta.obs = {"other": {"shape": [1]}}
    with pytest.raises(ValueError, match="observation keys"):
        workspace.load_finetune_checkpoint(source)
    changed.shape_meta.obs = {"state": {"shape": [1]}}
    changed.shape_meta.action.shape = [8]
    with pytest.raises(ValueError, match="action shape"):
        workspace.load_finetune_checkpoint(source)

    workspace.cfg = cfg
    payload["state_dicts"]["ema_model"]["model.weight"] = torch.ones(9, 1)
    torch.save(payload, source, pickle_module=dill)
    with pytest.raises(ValueError, match="incompatible"):
        workspace.load_finetune_checkpoint(source)


def test_small_data_finetune_saves_fresh_optimizer_and_normalizer(tmp_path, tiny_instances,
                                                                 monkeypatch):
    cfg = tiny_config()
    source = source_checkpoint(tmp_path, cfg)
    cfg.training.finetune_ckpt_path = str(source)
    cfg.optimizer.lr = 3e-5
    output = tmp_path / "finetune"
    output.mkdir()
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(output))
    workspace = TrainDiffusionUnetImageWorkspace(cfg, output_dir=str(output))
    workspace.run()

    best = output / "checkpoints" / "best.ckpt"
    latest = output / "checkpoints" / "latest.ckpt"
    assert best.is_file() and latest.is_file()
    payload = torch.load(latest, pickle_module=dill)
    saved = {key: dill.loads(value) for key, value in payload["pickles"].items()}
    assert saved["checkpoint_next_epoch"] == 1
    assert saved["best_loss"] < float("inf")
    assert payload["state_dicts"]["optimizer"]["state"]
    assert payload["state_dicts"]["optimizer"]["param_groups"][0]["initial_lr"] == pytest.approx(3e-5)
    assert workspace.model.normalizer["action"].params_dict["input_stats"]["min"][0] == 2
    assert workspace.ema_model.normalizer["action"].params_dict["input_stats"]["min"][0] == 2
    assert resumed_training_position(
        saved["epoch"], saved["global_step"], saved["checkpoint_next_epoch"],
        saved["checkpoint_next_global_step"]
    )[0] == 1

    resume_cfg = tiny_config()
    resume_cfg.training.resume = True
    resume_cfg.training.ckpt_path = str(latest)
    resumed = TrainDiffusionUnetImageWorkspace(resume_cfg, output_dir=str(tmp_path / "resume"))
    resumed.load_checkpoint(path=latest)
    assert resumed.optimizer.state
    assert resumed.epoch == 1
    assert resumed.global_step == saved["global_step"] < 30


def test_finetune_and_resume_are_mutually_exclusive(tmp_path, tiny_instances):
    cfg = tiny_config()
    cfg.training.resume = True
    cfg.training.finetune_ckpt_path = "source.ckpt"
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainDiffusionUnetImageWorkspace(cfg, output_dir=str(tmp_path))


@pytest.mark.parametrize("name", [
    "train_diffusion_unet_timm_umi_workspace",
    "train_diffusion_unet_timm_vr_joint_workspace",
    "train_diffusion_unet_timm_vr_umi_workspace",
])
def test_training_configs_inherit_finetune_path(name):
    config_dir = Path(__file__).parents[1] / "diffusion_policy" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=name)
        overridden = compose(config_name=name,
                             overrides=["training.finetune_ckpt_path=/tmp/source.ckpt"])
    assert cfg.training.finetune_ckpt_path is None
    assert overridden.training.finetune_ckpt_path == "/tmp/source.ckpt"


def test_link7_script_passes_finetune_overrides(tmp_path):
    """使用临时 Accelerate 记录 argv，避免启动 GPU 训练。"""
    project = tmp_path / "project"
    script_dir = project / "model" / "dp"
    script_dir.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parents[1] / "train_vr_umi.sh", script_dir / "train_vr_umi.sh")
    executable = script_dir / ".venv" / "bin" / "accelerate"
    executable.parent.mkdir(parents=True)
    executable.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
    executable.chmod(0o755)
    dataset = tmp_path / "dataset.zarr"
    dataset.mkdir()
    checkpoint = tmp_path / "source.ckpt"
    checkpoint.touch()
    run_root = project / "dataset" / "h5dy_data" / "rm75_umi"
    run_root.mkdir(parents=True)
    capture = tmp_path / "args.txt"
    environment = os.environ.copy()
    environment.update({
        "DATASET_PATH": str(dataset), "FINETUNE_CKPT": str(checkpoint),
        "LEARNING_RATE": "3e-5", "NUM_EPOCHS": "20", "NUM_PROCESSES": "1",
        "CAPTURE_ARGS": str(capture),
    })
    subprocess.run(["bash", str(script_dir / "train_vr_umi.sh"), str(run_root / "new_run")],
                   env=environment, check=True, capture_output=True, text=True)
    arguments = capture.read_text().splitlines()
    assert f"training.finetune_ckpt_path={checkpoint}" in arguments
    assert "optimizer.lr=3e-5" in arguments
    assert "training.num_epochs=20" in arguments
    assert f"task.dataset_path={dataset}" in arguments
