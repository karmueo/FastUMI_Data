"""Exercise Target category selection, fixed URDF bounds and shared split."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from convert_vr_target import convert, select_episodes
from convert_vr_target_to_umi import convert as convert_pose
from diffusion_policy.common.fixed_normalization import fixed_range_normalizer
from diffusion_policy.dataset.vr_joint_image_dataset import VrJointImageDataset
from diffusion_policy.dataset.umi_dataset import UmiDataset
from run_target_training import assess_run, stratified_split, training_overrides
from test_vr_joint import make_episode, shape_meta
from test_vr_umi import umi_shape


def urdf_with_limits(path):
    """Build a seven-joint chain with finite limits for conversion tests."""
    parts = ['<robot name="test"><link name="base_link"/>']
    for index in range(1, 8):
        parent = "base_link" if index == 1 else f"Link{index-1}"
        parts.append(f'<link name="Link{index}"/><joint name="joint{index}" type="revolute">'
                     f'<parent link="{parent}"/><child link="Link{index}"/>'
                     '<origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>'
                     '<limit lower="-10" upper="10"/></joint>')
    parts.append("</robot>")
    path.write_text("".join(parts), encoding="utf-8")
    return path


def test_target_selection_split_and_fixed_normalization(tmp_path):
    """Same episode number in separate Target categories stays unique."""
    source = tmp_path / "source"
    for category in ("Target", "Target2"):
        (source / category).mkdir(parents=True)
        make_episode(source / category / "episode_0")
        make_episode(source / category / "episode_1")
    (source / "Apple").mkdir()
    make_episode(source / "Apple" / "episode_0")
    assert len(select_episodes(source)) == 4
    with pytest.raises(ValueError, match="Target"):
        select_episodes(source, ["Apple"])
    urdf = urdf_with_limits(tmp_path / "arm.urdf")
    joint_path = tmp_path / "joint.zarr"
    report = convert(source, joint_path, frequency=10, image_size=32,
                     categories=["Target", "Target2"], urdf_path=urdf)
    assert report["episode_count"] == 4
    joint = zarr.open_group(str(joint_path), mode="r")
    names = joint["meta/episode_names"][:].tolist()
    assert names == ["Target/episode_0", "Target/episode_1",
                     "Target2/episode_0", "Target2/episode_1"]
    split = stratified_split(names)
    assert len(split["train_episode_indices"]) == 2
    assert len(split["val_episode_indices"]) == 2
    dataset = VrJointImageDataset(joint_path, shape_meta(),
                                  train_episode_indices=split["train_episode_indices"],
                                  val_episode_indices=split["val_episode_indices"])
    assert dataset[0]["action"].shape == (16, 8)
    normalizer = dataset.get_normalizer()
    physical = np.array([[0., 10., -10., 0., 0., 0., 0., .5]], dtype=np.float32)
    normalized = normalizer["action"].normalize(physical)
    np.testing.assert_allclose(normalized.detach(), [[0, 1, -1, 0, 0, 0, 0, 0]])
    np.testing.assert_allclose(normalizer["action"].unnormalize(normalized).detach(), physical)
    np.testing.assert_allclose(normalizer["robot0_joint_pos"].normalize(np.zeros((1, 7))).detach(), 0)
    pose_path = tmp_path / "pose.zarr"
    convert_pose(joint_path, urdf, pose_path)
    pose = zarr.open_group(str(pose_path), mode="r")
    assert pose.attrs["format"] == "rm75-umi-pose-v2"
    assert pose["meta/episode_names"][:].tolist() == names
    pose_shape = umi_shape()
    pose_shape["obs"]["camera0_rgb"]["shape"] = [3, 32, 32]
    umi = UmiDataset(pose_shape, str(pose_path),
                     pose_repr={"obs_pose_repr": "relative", "action_pose_repr": "relative"},
                     train_episode_indices=split["train_episode_indices"],
                     val_episode_indices=split["val_episode_indices"],
                     start_pose_noise_std=0, normalizer_num_workers=0)
    assert umi[0]["action"].shape == (16, 10)
    assert sum(value.shape[-1] for value in umi[0]["obs"].values() if value.ndim == 2) == 16
    pose_normalizer = umi.get_normalizer()
    np.testing.assert_allclose(
        pose_normalizer["robot0_gripper_width"].normalize(np.array([[0], [1]], dtype=np.float32)).detach(),
        [[-1], [1]])


def test_parallel_conversion_matches_serial(tmp_path):
    """Concurrent decoding preserves arrays, episode order, report and v2 metadata."""
    source = tmp_path / "source"
    for category in ("Target", "Target2"):
        (source / category).mkdir(parents=True)
        for index in range(2):
            make_episode(source / category / f"episode_{index}", offset=index)
    urdf = urdf_with_limits(tmp_path / "arm.urdf")
    outputs = [tmp_path / "serial.zarr", tmp_path / "parallel.zarr"]
    reports = [convert(source, output, frequency=10, image_size=32,
                       urdf_path=urdf, workers=workers)
               for output, workers in zip(outputs, (1, 4))]
    assert reports[0] == reports[1]
    serial, parallel = [zarr.open_group(str(path), mode="r") for path in outputs]
    assert dict(serial.attrs) == dict(parallel.attrs)
    assert serial.attrs["format"] == "rm75-joint-image-v2"
    for group in ("data", "meta"):
        assert set(serial[group].array_keys()) == set(parallel[group].array_keys())
        for key in serial[group].array_keys():
            np.testing.assert_array_equal(serial[f"{group}/{key}"][:],
                                          parallel[f"{group}/{key}"][:])


def test_parallel_conversion_reports_bad_episode(tmp_path):
    """A worker failure identifies its episode and never marks output complete."""
    source = tmp_path / "Target"
    source.mkdir()
    make_episode(source / "episode_0")
    make_episode(source / "episode_1")
    with h5py.File(source / "episode_1/proprio.hdf5", "r+") as file:
        file["action/joint_action/position"][0, 0] = np.nan
    output = tmp_path / "bad.zarr"
    with pytest.raises(ValueError, match="episode_1"):
        convert(source, output, frequency=10, image_size=32, workers=2)
    assert not zarr.open_group(str(output), mode="r").attrs["complete"]
    with pytest.raises(ValueError, match="workers"):
        convert(source, tmp_path / "invalid.zarr", workers=0)


def test_fixed_bounds_reject_invalid_ranges():
    """The physical normalization contract cannot silently use broken limits."""
    with pytest.raises(ValueError):
        fixed_range_normalizer([1], [1])
    with pytest.raises(ValueError):
        fixed_range_normalizer([np.nan], [1])


def test_five_epoch_overrides_use_full_data(tmp_path):
    """The production launch keeps full epochs and passes identical indices."""
    split = {"train_episode_indices": [0, 2], "val_episode_indices": [1, 3]}
    for representation in ("joint", "pose"):
        overrides = training_overrides(representation, Path("/tmp/example.zarr"), split,
                                       5, 32, 2, 16, tmp_path)
        assert "training.num_epochs=5" in overrides
        assert "training.max_train_steps=null" in overrides
        assert "training.max_val_steps=null" in overrides
        assert "training.sample_every=0" in overrides
        assert "task.dataset.train_episode_indices=[0, 2]" in overrides
        assert "task.dataset.val_episode_indices=[1, 3]" in overrides


def test_five_epoch_acceptance_and_curve(tmp_path):
    """Record an auditable pass and reject a validation loss that rises."""
    (tmp_path / "checkpoints").mkdir()
    for name in ("best.ckpt", "latest.ckpt"):
        (tmp_path / "checkpoints" / name).touch()
    rows = [{"epoch": index, "train_loss": 1 - index * .1,
             "val_loss": 1 - index * .05} for index in range(5)]
    log = tmp_path / "logs.json.txt"
    log.write_text("\n".join(json.dumps(row) for row in rows))
    assert assess_run(tmp_path, 5)["passed"]
    assert (tmp_path / "loss_curve.png").is_file()
    for row in rows:
        row["val_loss"] = 1 + row["epoch"] * .1
    log.write_text("\n".join(json.dumps(row) for row in rows))
    assert not assess_run(tmp_path, 5)["passed"]
