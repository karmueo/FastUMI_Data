"""验证 UMI 数据集对目录和 ZIP Zarr 容器的只读读取能力。"""

import re

import numpy as np
import pytest
import zarr

from diffusion_policy.dataset.umi_dataset import _open_replay_store


def _write_minimal_store(tmp_path, container):
    """创建包含一个数组的最小 Zarr 容器。"""
    if container == "directory":
        dataset_path = tmp_path / "minimal.zarr"
        store = zarr.DirectoryStore(str(dataset_path))
    else:
        dataset_path = tmp_path / "minimal.zarr.zip"
        store = zarr.ZipStore(str(dataset_path), mode="w")

    try:
        root = zarr.group(store=store)
        root.create_dataset("data/value", data=np.array([1, 2, 3]))
    finally:
        store.close()
    return dataset_path


@pytest.mark.parametrize("container", ["directory", "zip"])
def test_open_replay_store_reads_directory_and_zip(tmp_path, container):
    """目录和 ZIP Zarr 均可被只读 Store 接口打开。"""
    dataset_path = _write_minimal_store(tmp_path, container)

    with _open_replay_store(dataset_path) as store:
        root = zarr.open_group(store=store, mode="r")
        np.testing.assert_array_equal(root["data/value"][:], np.array([1, 2, 3]))


def test_open_replay_store_reports_missing_absolute_path(tmp_path):
    """不存在的路径应在异常中明确指出绝对路径。"""
    missing_path = tmp_path / "missing.zarr"

    with pytest.raises(FileNotFoundError, match=re.escape(str(missing_path.resolve()))):
        with _open_replay_store(missing_path):
            pass


def test_open_replay_store_reports_invalid_file_path(tmp_path):
    """普通文件不应被误认为有效的 Zarr 容器。"""
    dataset_path = tmp_path / "invalid.zarr"
    dataset_path.write_text("not a zarr container", encoding="utf-8")

    with pytest.raises(ValueError, match=re.escape(str(dataset_path.resolve()))):
        with _open_replay_store(dataset_path):
            pass


def test_open_replay_store_preserves_consumer_value_error(tmp_path):
    """消费者在 with 体内抛出的 ValueError 应保持原异常和消息。"""
    dataset_path = _write_minimal_store(tmp_path, "directory")

    with pytest.raises(ValueError, match="consumer sentinel"):
        with _open_replay_store(dataset_path):
            raise ValueError("consumer sentinel")


def test_open_replay_store_preserves_consumer_key_error(tmp_path):
    """消费者在 with 体内抛出的 KeyError 应保持原异常和消息。"""
    dataset_path = _write_minimal_store(tmp_path, "directory")

    with pytest.raises(KeyError, match="consumer sentinel"):
        with _open_replay_store(dataset_path):
            raise KeyError("consumer sentinel")
