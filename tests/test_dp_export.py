"""测试 FastUMI HDF5 到 Diffusion Policy Zarr 的分批导出结构。"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import zarr

from data_processing_tcp_to_dp import discover_hdf5_files, export_zarr
from replay_buffer import ReplayBuffer


def _write_episode(path: Path, episode_length: int) -> None:
    """写入一条确定性的最小 FastUMI HDF5。"""
    action = np.tile(
        np.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5],
            dtype=np.float32,
        ),
        (episode_length, 1),
    )
    with h5py.File(path, "w") as root:
        root.attrs["image_encoding"] = "rgb8"
        root.attrs["sample_rate_hz"] = 20.0
        images = root.create_group("observations").create_group("images")
        images.create_dataset(
            "front",
            data=np.zeros(
                (episode_length, 16, 16, 3), dtype=np.uint8
            ),
        )
        root.create_dataset("action", data=action)


def test_exports_batched_zarr_and_replay_buffer(tmp_path: Path) -> None:
    """验证键名、episode 结束索引和 ReplayBuffer 磁盘读取。"""
    _write_episode(tmp_path / "episode_0000.hdf5", 3)
    _write_episode(tmp_path / "episode_0001.hdf5", 4)
    output = tmp_path / "task_dp.zarr"

    export_zarr(
        discover_hdf5_files(str(tmp_path)),
        str(output),
        (8, 8),
        compression_level=99,
        force=False,
        legacy_bgr=False,
        batch_size=2,
    )

    root = zarr.open(str(output), mode="r")
    replay_buffer = ReplayBuffer.create_from_group(root)
    assert replay_buffer.n_steps == 7
    assert replay_buffer.n_episodes == 2
    assert root["data/camera0_rgb"].shape == (7, 8, 8, 3)
    assert root["meta/episode_ends"][:].tolist() == [3, 7]
    assert root.attrs["schema_version"] == "fastumi_dp_v1"


def test_force_preserves_existing_output_when_export_fails(
    tmp_path: Path,
) -> None:
    """验证强制重建失败时仍保留已有 Zarr 数据。"""
    output = tmp_path / "task_dp.zarr"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("keep", encoding="utf-8")
    invalid_episode = tmp_path / "episode_0000.hdf5"
    with h5py.File(invalid_episode, "w"):
        pass

    with pytest.raises(ValueError):
        export_zarr(
            [invalid_episode],
            str(output),
            (8, 8),
            compression_level=99,
            force=True,
            legacy_bgr=False,
            batch_size=2,
        )

    assert marker.read_text(encoding="utf-8") == "keep"


def test_force_replaces_existing_output_after_success(
    tmp_path: Path,
) -> None:
    """验证强制重建成功后才移除已有 Zarr 数据。"""
    episode = tmp_path / "episode_0000.hdf5"
    _write_episode(episode, 3)
    output = tmp_path / "task_dp.zarr"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_text("replace", encoding="utf-8")

    export_zarr(
        [episode],
        str(output),
        (8, 8),
        compression_level=99,
        force=True,
        legacy_bgr=False,
        batch_size=2,
    )

    assert not marker.exists()
    assert zarr.open(str(output), mode="r")["meta/episode_ends"][:].tolist() == [
        3
    ]
