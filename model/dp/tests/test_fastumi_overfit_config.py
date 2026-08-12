"""验证 FastUMI 固定 batch 过拟合 Hydra 配置的关键契约。"""

import pathlib

from hydra import compose, initialize_config_dir

from diffusion_policy.workspace import train_diffusion_unet_image_workspace  # noqa: F401


CONFIG_NAME = "train_diffusion_unet_timm_fastumi_overfit_workspace"


def _compose_config(overrides=()):
    """加载 FastUMI 专用 Hydra 配置。"""
    config_dir = str(pathlib.Path(__file__).parents[1].joinpath("diffusion_policy", "config"))
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=CONFIG_NAME, overrides=list(overrides))


def test_fastumi_overfit_config_has_fixed_batch_contract():
    """专用配置应继承 UMI Timm 策略并启用固定 batch 过拟合参数。"""
    cfg = _compose_config()

    assert cfg.policy._target_ == "diffusion_policy.policy.diffusion_unet_timm_policy.DiffusionUnetTimmPolicy"
    assert cfg.policy.obs_encoder.model_name == "vit_base_patch16_clip_224.openai"
    assert cfg.policy.obs_encoder.pretrained is True
    assert cfg.policy.noise_scheduler._target_ == "diffusers.DDIMScheduler"
    assert cfg.task.action_horizon == 16
    assert cfg.task.img_obs_horizon == 2
    assert cfg.dataloader.batch_size == 4
    assert cfg.dataloader.shuffle is False
    assert cfg.training.max_train_steps == 1
    assert cfg.training.freeze_encoder is True
    assert cfg.task.env_runner is None
    assert cfg.logging.project == "fastumi-pick-place"
    assert cfg.logging.mode == "online"
    assert cfg.checkpoint.topk.monitor_key == "train_action_mse_error"

    dataset_path = pathlib.Path(str(cfg.task.dataset_path))
    assert not dataset_path.is_absolute()
    assert "pick_place_dp.zarr" not in str(dataset_path)


def test_fastumi_overfit_config_accepts_dataset_path_override():
    """数据路径应通过 task.dataset_path 命令行覆盖。"""
    dataset_path = "/tmp/example-fastumi.zarr"
    cfg = _compose_config([f"task.dataset_path={dataset_path}"])

    assert cfg.task.dataset_path == dataset_path
