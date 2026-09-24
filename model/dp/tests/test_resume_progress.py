"""Regression checks for epoch boundaries when resuming training."""

import dill
import pytest
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace,
    resumed_training_position,
)


def test_legacy_checkpoint_advances_past_completed_epoch():
    epoch, global_step = resumed_training_position(43, 30876)

    assert (epoch, global_step) == (44, 30877)
    assert len(range(epoch, 120)) == 76


def test_current_checkpoint_uses_explicit_next_position():
    assert resumed_training_position(43, 30876, 44, 30877) == (44, 30877)
    assert resumed_training_position(120, 100000, 120, 100000) == (120, 100000)


def test_partial_checkpoint_progress_is_rejected():
    with pytest.raises(ValueError, match="incomplete training progress"):
        resumed_training_position(43, 30876, next_epoch=44)


def test_checkpoint_persists_next_training_position(tmp_path):
    workspace = TrainDiffusionUnetImageWorkspace.__new__(TrainDiffusionUnetImageWorkspace)
    BaseWorkspace.__init__(workspace, OmegaConf.create({}), output_dir=str(tmp_path))
    workspace.epoch = 43
    workspace.global_step = 30876
    workspace.best_loss = 0.1
    workspace.checkpoint_next_epoch = 44
    workspace.checkpoint_next_global_step = 30877

    path = workspace.save_checkpoint(use_thread=False)
    payload = torch.load(path, pickle_module=dill)
    saved = {key: dill.loads(value) for key, value in payload["pickles"].items()}

    assert resumed_training_position(
        saved["epoch"], saved["global_step"], saved["checkpoint_next_epoch"],
        saved["checkpoint_next_global_step"]
    ) == (44, 30877)
