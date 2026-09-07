"""验证本地 FastUMI DP 入口、checkpoint 契约与真实 Zarr 只读采样。"""

import hashlib
import os
import pathlib
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import hydra
import numpy as np
import pytest
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from diffusion_policy.dataset.umi_dataset import UmiDataset, _open_replay_store
from infer_fastumi_sim import (
    DEFAULT_DIFFUSION_POLICY_ROOT,
    FASTUMI_RGB_KEY,
    validate_fastumi_shape_meta,
)
from infer_real import load_controller_factory


PROJECT_ROOT = pathlib.Path(__file__).parents[1]
"""目标 DP 项目根目录。"""

CANONICAL_OBS_KEYS = {
    "camera0_rgb",
    "robot0_eef_pos",
    "robot0_eef_rot_axis_angle",
    "robot0_gripper_width",
    "robot0_eef_rot_axis_angle_wrt_start",
}
"""canonical FastUMI checkpoint 的五个观测键。"""


def _compose_fastumi_config(overrides=()):
    """组合 FastUMI canonical Hydra 配置。"""
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = PROJECT_ROOT / "diffusion_policy" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(
            config_name="train_diffusion_unet_timm_fastumi_overfit_workspace",
            overrides=list(overrides),
        )


def _legacy_cfg(obs_keys, action_shape):
    """构造仅供 legacy shape 校验使用的最小配置对象。"""
    obs = {key: SimpleNamespace() for key in obs_keys}
    return SimpleNamespace(
        shape_meta=SimpleNamespace(
            obs=obs,
            action=SimpleNamespace(shape=action_shape),
        )
    )


def test_supported_entrypoints_are_target_local():
    """支持的入口不得包含原仓库导入或机器特定绝对路径。"""
    entrypoints = ["infer_sim.py", "infer_fastumi_sim.py", "infer_real.py"]
    for entrypoint in entrypoints:
        content = (PROJECT_ROOT / entrypoint).read_text(encoding="utf-8")
        assert "from src." not in content
        assert "/home/" not in content

    portable_files = [
        "README.md",
        "diffusion_policy/config/train_diffusion_unet_timm_umi_workspace.yaml",
    ]
    for relative_path in portable_files:
        content = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert "/home/" not in content


def test_real_controller_factory_is_loaded_from_explicit_module(monkeypatch):
    """真实机器人控制器工厂应从调用方指定的模块加载。"""

    class TestControllerFactory:
        """提供测试使用的最小控制器工厂接口。"""

        @staticmethod
        def create_controller(name):
            """返回控制器名称，证明工厂接口可调用。"""
            return name

    module = ModuleType("fastumi_test_controller")
    module.ControllerFactory = TestControllerFactory
    monkeypatch.setitem(sys.modules, module.__name__, module)

    factory = load_controller_factory(
        "fastumi_test_controller:ControllerFactory"
    )

    assert factory is TestControllerFactory
    assert factory.create_controller("realman_gen72") == "realman_gen72"


@pytest.mark.parametrize(
    "factory_spec",
    ["missing_separator", ":ControllerFactory", "module:"],
)
def test_real_controller_factory_rejects_invalid_spec(factory_spec):
    """控制器工厂路径格式错误时应在硬件启动前给出明确错误。"""
    with pytest.raises(ValueError, match="模块:属性"):
        load_controller_factory(factory_spec)


def test_canonical_hydra_contract_is_five_key_ten_dimensional():
    """canonical 训练配置必须保持五键观测和每臂 10 维动作。"""
    cfg = _compose_fastumi_config()

    assert cfg._target_ == (
        "diffusion_policy.workspace.train_diffusion_unet_image_workspace."
        "TrainDiffusionUnetImageWorkspace"
    )
    assert set(cfg.shape_meta.obs.keys()) == CANONICAL_OBS_KEYS
    assert list(cfg.shape_meta.action.shape) == [10]
    assert not pathlib.Path(str(cfg.task.dataset_path)).is_absolute()


def test_base_umi_resume_checkpoint_requires_explicit_override():
    """基础 UMI 配置不得携带开发者机器上的 checkpoint 路径。"""
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_dir = PROJECT_ROOT / "diffusion_policy" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="train_diffusion_unet_timm_umi_workspace")

    assert cfg.training.ckpt_path is None


def test_legacy_adapter_rejects_canonical_contract():
    """legacy 图像适配器只能接受一键七维 checkpoint。"""
    assert pathlib.Path(DEFAULT_DIFFUSION_POLICY_ROOT).resolve() == PROJECT_ROOT.resolve()
    validate_fastumi_shape_meta(_legacy_cfg({FASTUMI_RGB_KEY}, [7]))

    with pytest.raises(ValueError, match="obs keys"):
        validate_fastumi_shape_meta(_legacy_cfg(CANONICAL_OBS_KEYS, [10]))
    with pytest.raises(ValueError, match="action shape"):
        validate_fastumi_shape_meta(_legacy_cfg({FASTUMI_RGB_KEY}, [10]))


@pytest.mark.parametrize(
    "entrypoint",
    ["train.py", "infer_sim.py", "infer_fastumi_sim.py", "infer_real.py"],
)
def test_entrypoint_help_uses_local_imports(entrypoint):
    """各入口的帮助文本应在目标环境中退出成功，无需上游路径。"""
    result = subprocess.run(
        [sys.executable, entrypoint, "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _dataset_digest(path):
    """流式计算 ZIP 或目录 Zarr 的内容摘要，验证采样未写入源数据。"""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
    for item in files:
        digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
        with item.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def test_real_fastumi_dataset_is_read_only_when_requested():
    """环境变量指定真实数据集时读取其关键字段并采样 canonical 数据集。"""
    dataset_value = os.environ.get("FASTUMI_DATASET")
    if not dataset_value:
        pytest.skip("FASTUMI_DATASET 未设置")

    dataset_path = pathlib.Path(dataset_value)
    before_digest = _dataset_digest(dataset_path)
    required_data_keys = {
        "camera0_rgb",
        "robot0_eef_pos",
        "robot0_eef_rot_axis_angle",
        "robot0_gripper_width",
        "robot0_demo_start_pose",
    }
    with _open_replay_store(dataset_path) as store:
        root = zarr.open_group(store=store, mode="r")
        assert required_data_keys <= set(root["data"].keys())
        assert "episode_ends" in root["meta"]

    cfg = _compose_fastumi_config([f"task.dataset_path={dataset_path}"])
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    sample = dataset[0]
    assert set(sample["obs"].keys()) == CANONICAL_OBS_KEYS
    assert tuple(sample["action"].shape) == (16, 10)
    assert np.isfinite(sample["action"].numpy()).all()
    assert _dataset_digest(dataset_path) == before_digest
