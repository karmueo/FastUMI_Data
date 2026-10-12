"""验证网页预览的抽样、文件格式和已有远端介绍的保留行为。"""

import csv
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
import pytest
import yaml
import zarr

from datatool.modelscope_preview import (
    CONFIG_NAME, CSV_PATH, END_MARKER, START_MARKER, build_preview, merge_card,
    open_source, publish_preview, sample_indices,
)
import upload_dataset_to_modelscope as upload


@pytest.fixture
def source(tmp_path):
    """生成两段轨迹和纯红 RGB 图像，便于验证抽样值与颜色。"""
    path = tmp_path / "target.zarr"
    root = zarr.open_group(str(path), mode="w")
    root.attrs.update(format="rm75-umi-pose-v2", position_unit="m", rotation_unit="rad",
                      source_dataset="/private/machine/recordings", complete=True)
    root.create_dataset("meta/episode_ends", data=np.array([3, 7], dtype="i8"))
    root.create_dataset("meta/episode_names", data=np.array(["episode_0", "episode_1"]))
    root.create_dataset("data/action", data=np.arange(21, dtype="f4").reshape(7, 3),
                        chunks=(2, 3))
    root.create_dataset("data/robot0_eef_pos", data=np.ones((7, 3), dtype="f4"))
    root.create_dataset("data/timestamp", data=np.arange(7, dtype="f8") / 30)
    images = np.zeros((7, 8, 8, 3), dtype="u1")
    images[..., 0] = 128
    root.create_dataset("data/camera0_rgb", data=images, chunks=(1, 8, 8, 3))
    return path


class FakeApi:
    """支持元数据合并和单文件上传的离线仓库。"""

    def __init__(self, remote):
        self.remote = dict(remote)
        self.uploads = []

    def get_repo(self, **kwargs):
        return {}

    def list_repo_files(self, **kwargs):
        return [SimpleNamespace(path=path) for path in self.remote]

    def download_file(self, *, file_path, cache_dir, **kwargs):
        path = Path(cache_dir) / file_path
        path.write_bytes(self.remote[file_path])
        return path

    def upload_file(self, *, path_in_repo, path_or_fileobj, **kwargs):
        self.uploads.append(path_in_repo)
        self.remote[path_in_repo] = Path(path_or_fileobj).read_bytes()


def read_rows(path):
    """读取 UTF-8 CSV 的字段和行。"""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def test_preview_values_metadata_images_and_card(source, tmp_path):
    """生成可解析的标量 CSV、对应字段信息及正确 RGB 截图。"""
    bundle = build_preview(source, tmp_path / "preview", "train/target.zarr.zip", "data", 4)
    headers, rows = read_rows(bundle.directory / CSV_PATH)
    assert [int(row["frame_index"]) for row in rows] == [0, 2, 4, 6]
    assert [int(row["episode_index"]) for row in rows] == [0, 0, 1, 1]
    assert [int(row["episode_frame_index"]) for row in rows] == [0, 2, 1, 3]
    assert rows[-1]["episode_name"] == "episode_1"
    assert float(rows[-1]["action_0"]) == 18
    assert float(rows[-1]["timestamp"]) == pytest.approx(6 / 30)
    info = json.loads((bundle.directory / "dataset_infos.json").read_text())[CONFIG_NAME]
    assert set(info["features"]) == set(headers)
    assert info["features"]["action_0"]["dtype"] == "float64"
    assert info["splits"]["train"]["num_examples"] == 4
    card = (bundle.directory / "README.md").read_text()
    config = yaml.safe_load(card.split("---", 2)[1])["configs"][0]
    assert config["config_name"] == CONFIG_NAME
    assert config["data_files"] == [{"split": "train", "path": CSV_PATH}]
    assert "train/target.zarr.zip" in card
    assert "/private/machine/recordings" not in card
    assert bundle.images == 4
    for relative in bundle.files:
        if relative.endswith(".jpg"):
            image = cv2.imread(str(bundle.directory / relative))
            assert image.shape == (8, 8, 3)
            assert image[0, 0, 2] > 100 and image[0, 0, 0] < 5
            assert f"]({relative})" in card


def test_full_dataset_is_not_loaded_or_rewritten(source, tmp_path):
    """抽样行数和图片数量有界，源数据不变。"""
    before = upload.snapshot_directory(source)
    bundle = build_preview(source, tmp_path / "preview", "target.zip", "data", 1000)
    assert bundle.rows == 7
    assert bundle.images == 6
    assert upload.snapshot_directory(source) == before
    assert not any(item.relative_path.endswith(".csv") for item in before)


def test_zip_input_can_be_previewed_without_extracting(source, tmp_path):
    """已打包的 ZIP 可直接读取，不需要复制完整数据。"""
    archive = tmp_path / "target.zarr.zip"
    upload.pack_directory(source, archive, upload.snapshot_directory(source))
    bundle = build_preview(archive, tmp_path / "preview", archive.name, "data", 3)
    _, rows = read_rows(bundle.directory / CSV_PATH)
    assert [int(row["frame_index"]) for row in rows] == [0, 3, 6]
    assert bundle.images == 3


def test_flat_zip_root_is_supported(source, tmp_path):
    """兼容根目录直接存放 .zgroup 的标准 Zarr ZIP。"""
    archive = tmp_path / "flat.zip"
    with zipfile.ZipFile(archive, "w") as packed:
        for item in upload.snapshot_directory(source):
            packed.write(source / item.relative_path, item.relative_path)
    with open_source(archive) as root:
        assert root["data/action"].shape == (7, 3)


def test_merge_preserves_remote_body_license_and_other_configs():
    """更新生成段落时保留用户正文、许可证及其他预览配置。"""
    existing = ("---\nlicense: mit\nconfigs:\n- config_name: previous\n  default: true\n"
                "  data_files: old.csv\n---\n\n# 原有介绍\n\n用户编写的内容。\n")
    card = merge_card(existing, "## 抽样预览\n第一次")
    card = merge_card(card, "## 抽样预览\n第二次")
    metadata = yaml.safe_load(card.split("---", 2)[1])
    assert metadata["license"] == "mit"
    assert metadata["configs"][1]["data_files"] == "old.csv"
    assert metadata["configs"][0]["default"] is True
    assert "用户编写的内容。" in card
    assert "第一次" not in card
    assert "第二次" in card
    assert card.count(START_MARKER) == card.count(END_MARKER) == 1


@pytest.mark.parametrize("card", ["---\nlicense: mit", "---\n- bad\n---\n",
                                  "---\nconfigs: wrong\n---\n", START_MARKER])
def test_invalid_remote_card_is_not_overwritten(card):
    """损坏的现有卡片必须报错，不静默丢弃其内容。"""
    with pytest.raises(ValueError):
        merge_card(card, "preview")


def test_publish_only_preview_preserves_archive_and_remote_metadata(source):
    """补传轨迹、截图和配置，完整 ZIP 不会重新上传。"""
    previous_info = {"old": {"features": {"text": {"_type": "Value", "dtype": "string"}}}}
    api = FakeApi({"target.zarr.zip": b"original large archive",
                   "README.md": b"# Existing user card\n",
                   "dataset_infos.json": json.dumps(previous_info).encode()})
    bundle = publish_preview(api, source, "target.zarr.zip", "owner/data", "master",
                             4, require_archive=True)
    assert "target.zarr.zip" not in api.uploads
    assert api.remote["target.zarr.zip"] == b"original large archive"
    assert api.uploads[-2:] == ["dataset_infos.json", "README.md"]
    assert CSV_PATH in api.uploads
    assert "Existing user card" in api.remote["README.md"].decode()
    infos = json.loads(api.remote["dataset_infos.json"])
    assert infos["old"] == previous_info["old"]
    assert CONFIG_NAME in infos
    assert bundle.rows == 4


def test_preview_only_checks_existing_archive_before_upload(source):
    """错仓库、分支或 ZIP 路径时不上传任何预览文件。"""
    api = FakeApi({})
    with pytest.raises(ValueError, match="远端未找到完整 ZIP"):
        publish_preview(api, source, "target.zarr.zip", "owner/data", "master",
                        require_archive=True)
    assert not api.uploads
    assert not source.with_name(source.name + ".modelscope-preview").exists()


def test_cli_preview_only_skips_packaging(source, monkeypatch):
    """通过 CLI 修复已有上传时不会创建、重建或提交大 ZIP。"""
    api = FakeApi({"target.zarr.zip": b"already uploaded"})
    monkeypatch.setattr(upload, "create_api", lambda token: api)
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "test-secret")

    def forbidden(*args, **kwargs):
        raise AssertionError("不应重新打包")

    monkeypatch.setattr(upload, "pack_directory", forbidden)
    assert upload.main(["--input", str(source), "--repo-id", "owner/data",
                        "--preview-only", "--preview-rows", "4"]) == 0
    assert "target.zarr.zip" not in api.uploads
    assert not source.with_name("target.zarr.zip").exists()


def test_preview_dry_run_has_no_network_or_outputs(source, monkeypatch):
    """带网页预览参数的 dry-run 仍保持只读。"""
    monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
    assert upload.main(["--input", str(source), "--repo-id", "owner/data",
                        "--preview-only", "--dry-run"]) == 0
    assert not source.with_name(source.name + ".modelscope-preview").exists()


@pytest.mark.parametrize("ends", [[0, 7], [4, 3], [3.0, 7.0], []])
def test_invalid_episode_boundaries_do_not_publish(source, tmp_path, ends):
    """错误 episode 边界不得生成看似有效的预览。"""
    root = zarr.open_group(str(source), mode="a")
    del root["meta/episode_ends"]
    root.create_dataset("meta/episode_ends", data=np.asarray(ends))
    with pytest.raises(ValueError, match="episode_ends"):
        build_preview(source, tmp_path / "preview", "target.zip", "data")
    assert not (tmp_path / "preview").exists()


def test_nonfinite_sample_and_output_inside_source_are_rejected(source, tmp_path):
    """非法样本或源目录内输出不能污染数据集。"""
    with pytest.raises(ValueError, match="源数据内部"):
        build_preview(source, source / "preview", "target.zip", "data")
    root = zarr.open_group(str(source), mode="a")
    root["data/action"][0, 0] = np.nan
    with pytest.raises(ValueError, match="非有限"):
        build_preview(source, tmp_path / "preview", "target.zip", "data")


def test_sampling_stays_in_bounds():
    """短数据集和单帧数据集不会重复抽样或越界。"""
    assert sample_indices(1, 1000).tolist() == [0]
    assert sample_indices(3, 1000).tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        sample_indices(0, 1000)
