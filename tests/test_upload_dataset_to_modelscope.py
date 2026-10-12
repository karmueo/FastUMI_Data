"""离线验证数据集打包的完整性和 ModelScope 上传边界。"""

import json
import os
import sys
from types import SimpleNamespace
from urllib.parse import quote
import zipfile

import numpy as np
import pytest
import zarr

import upload_dataset_to_modelscope as upload


@pytest.fixture
def dataset(tmp_path):
    """创建包含隐藏元数据及数组分块的小型 Zarr。"""
    source = tmp_path / "target.zarr"
    root = zarr.open_group(str(source), mode="w")
    root.attrs["complete"] = True
    root.create_dataset("data/action", data=np.arange(21).reshape(7, 3),
                        chunks=(2, 3))
    root.create_dataset("meta/episode_ends", data=np.array([3, 7]))
    return source


@pytest.fixture
def fake_sdk(monkeypatch):
    """模拟 SDK 的隐式创建行为及认证、上传失败，不访问网络。"""
    state = SimpleNamespace(calls=[], token=None, endpoint=None,
                            repo_error=None, upload_error=None, creates=0)

    class FakeHubApi:
        def __init__(self, *, token, endpoint):
            state.token, state.endpoint = token, endpoint

        def get_repo(self, **kwargs):
            state.calls.append(("get_repo", kwargs))
            if state.repo_error:
                raise state.repo_error
            return {"repo_id": kwargs["repo_id"]}

        def create_repo(self, **kwargs):
            state.creates += 1
            raise AssertionError("不允许创建仓库")

        def upload_file(self, **kwargs):
            # 官方 SDK 会在上传前调用 create_repo；适配器必须只读取仓库。
            self.create_repo(repo_id=kwargs["repo_id"], repo_type=kwargs["repo_type"])
            state.calls.append(("upload_file", kwargs))
            if state.upload_error:
                raise state.upload_error
            return {"success": True}

    monkeypatch.setitem(sys.modules, "modelscope_hub", SimpleNamespace(HubApi=FakeHubApi))
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "ms-test-secret")
    return state


def args(source, *extra):
    """构造指向已有测试仓库的命令行参数。"""
    return ["--input", str(source), "--repo-id", "test-owner/test-data", *extra]


def test_zip_preserves_zarr_and_uses_zip64(dataset, tmp_path, monkeypatch):
    """Zarr 内容及元数据逐字节保留，ZIP64 条目可解压使用。"""
    archive = tmp_path / "target.zarr.zip"
    monkeypatch.setattr(upload, "COPY_BLOCK_SIZE", 7)
    # 降低标准库的 ZIP64 阈值，离线覆盖大文件和大中央目录的编码路径。
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 1)
    files = upload.snapshot_directory(dataset)
    upload.pack_directory(dataset, archive, files)
    with zipfile.ZipFile(archive) as packed:
        assert set(packed.namelist()) == {
            f"target.zarr/{item.relative_path}" for item in files}
        for member in packed.infolist():
            relative = member.filename.removeprefix("target.zarr/")
            assert packed.read(member) == (dataset / relative).read_bytes()
            assert member.compress_type == zipfile.ZIP_STORED
            assert member.extract_version >= 45
        assert packed.testzip() is None
        packed.extractall(tmp_path / "restored")
    restored = zarr.open_group(str(tmp_path / "restored/target.zarr"), mode="r")
    assert restored.attrs["complete"] is True
    np.testing.assert_array_equal(restored["data/action"][:], np.arange(21).reshape(7, 3))


def test_cache_reuses_and_rebuilds_for_source_changes(dataset, tmp_path, capsys):
    """未变化时复用 ZIP；同大小内容修改、文件增删均触发重建。"""
    archive = tmp_path / "target.zarr.zip"
    upload.pack_directory(dataset, archive, upload.snapshot_directory(dataset))
    initial = upload.archive_state(archive)
    upload.pack_directory(dataset, archive, upload.snapshot_directory(dataset))
    assert upload.archive_state(archive) == initial
    assert "复用 ZIP" in capsys.readouterr().out

    metadata = dataset / ".zattrs"
    info = metadata.stat()
    metadata.write_bytes(metadata.read_bytes().replace(b"true", b"null"))
    os.utime(metadata, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    added = dataset / "new-file"
    added.write_bytes(b"new")
    upload.pack_directory(dataset, archive, upload.snapshot_directory(dataset))
    with zipfile.ZipFile(archive) as packed:
        assert packed.read("target.zarr/.zattrs") == metadata.read_bytes()
        assert packed.read("target.zarr/new-file") == b"new"
    added.unlink()
    upload.pack_directory(dataset, archive, upload.snapshot_directory(dataset))
    with zipfile.ZipFile(archive) as packed:
        assert "target.zarr/new-file" not in packed.namelist()


def test_modified_archive_and_corrupt_cache_are_rebuilt(dataset, tmp_path):
    """ZIP 被修改或缓存损坏后不会复用旧文件。"""
    archive = tmp_path / "target.zarr.zip"
    files = upload.snapshot_directory(dataset)
    upload.pack_directory(dataset, archive, files)
    archive.write_bytes(b"broken archive")
    upload.pack_directory(dataset, archive, files)
    assert zipfile.is_zipfile(archive)
    cache = archive.with_name(archive.name + ".upload-cache.json")
    cache.write_text("invalid JSON", encoding="utf-8")
    upload.pack_directory(dataset, archive, files)
    assert json.loads(cache.read_text())["archive"] == upload.archive_state(archive)


def test_changed_source_aborts_without_replacing_previous_archive(
        dataset, tmp_path, monkeypatch):
    """打包期间源目录变化时保留原 ZIP，不遗留半成品。"""
    archive = tmp_path / "target.zarr.zip"
    archive.write_bytes(b"previous archive")
    files = upload.snapshot_directory(dataset)
    original_snapshot = upload.snapshot_directory

    def changed_snapshot(source):
        (source / "new-data").write_bytes(b"changed while packing")
        return original_snapshot(source)

    monkeypatch.setattr(upload, "snapshot_directory", changed_snapshot)
    with pytest.raises(ValueError, match="发生变化"):
        upload.pack_directory(dataset, archive, files)
    assert archive.read_bytes() == b"previous archive"
    assert not list(tmp_path.glob("*.part"))
    assert not list(tmp_path.glob(".*.part"))
    assert not archive.with_name(archive.name + ".upload-cache.json").exists()


def test_changed_file_before_copy_aborts(dataset, tmp_path):
    """扫描后被修改的文件不能进入新 ZIP。"""
    archive = tmp_path / "target.zarr.zip"
    files = upload.snapshot_directory(dataset)
    (dataset / ".zattrs").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="发生变化"):
        upload.pack_directory(dataset, archive, files)
    assert not archive.exists()
    assert not list(tmp_path.glob(".*.part"))


def test_insufficient_disk_space_does_not_create_output(dataset, tmp_path, monkeypatch):
    """空间不足时在创建 ZIP 或输出目录前拒绝打包。"""
    archive = tmp_path / "output/data.zip"
    monkeypatch.setattr(upload.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="空间不足"):
        upload.pack_directory(dataset, archive, upload.snapshot_directory(dataset))
    assert not archive.parent.exists()


def test_dry_run_needs_no_token_sdk_or_output(dataset, tmp_path, monkeypatch, capsys):
    """预览不读取凭据、不打包、不联网，不创建输出目录。"""
    archive = tmp_path / "output/target.zarr.zip"
    monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run 不允许此操作")

    for name in ("read_token", "create_api", "pack_directory"):
        monkeypatch.setattr(upload, name, forbidden)
    before = upload.snapshot_directory(dataset)
    assert upload.main(args(dataset, "--dry-run", "--archive-path", str(archive))) == 0
    assert not archive.parent.exists()
    assert upload.snapshot_directory(dataset) == before
    output = capsys.readouterr().out
    assert "test-owner/test-data/target.zarr.zip" in output
    assert "master" in output


@pytest.mark.parametrize("kind", ["missing", "empty", "root_link", "file_link",
                                  "directory_link", "parent_link", "fifo"])
def test_invalid_sources_are_rejected(tmp_path, kind, capsys):
    """输入缺失、空目录、各类链接及 FIFO 均不能上传。"""
    source = tmp_path / "source"
    if kind == "empty":
        source.mkdir()
    elif kind == "root_link":
        source.symlink_to(tmp_path / "missing")
    elif kind in ("file_link", "directory_link"):
        source.mkdir()
        (source / "real").write_bytes(b"real")
        (source / "link").symlink_to(tmp_path, target_is_directory=kind == "directory_link")
    elif kind == "parent_link":
        real = tmp_path / "real"
        real.mkdir()
        (real / "file").write_bytes(b"real")
        source.symlink_to(real, target_is_directory=True)
        source = source / "file"
    elif kind == "fifo":
        source.mkdir()
        os.mkfifo(source / "fifo")
    assert upload.main(args(source, "--dry-run")) == 2
    assert "失败" in capsys.readouterr().err


@pytest.mark.parametrize("target", ["../data.zip", "/data.zip", "a/../data.zip",
                                    "a//data.zip", "a\\data.zip", "data.zip\n"])
def test_invalid_remote_path(dataset, target):
    """远端路径穿越、绝对路径和控制字符被拒绝。"""
    assert upload.main(args(dataset, "--path-in-repo", target, "--dry-run")) == 2


def test_archive_inside_source_or_symlink_is_rejected(dataset, tmp_path):
    """禁止 ZIP 写回源目录或写入链接目标。"""
    assert upload.main(args(dataset, "--archive-path", str(dataset / "data.zip"),
                            "--dry-run")) == 2
    archive = tmp_path / "linked.zip"
    archive.symlink_to(tmp_path / "missing.zip")
    assert upload.main(args(dataset, "--archive-path", str(archive), "--dry-run")) == 2
    cache_archive = tmp_path / "cached.zip"
    cache_archive.with_name(cache_archive.name + ".upload-cache.json").symlink_to(
        tmp_path / "missing.json")
    assert upload.main(args(dataset, "--archive-path", str(cache_archive), "--dry-run")) == 2


@pytest.mark.parametrize("repo", ["data", "/data", "owner/", "owner/data/extra",
                                  "../data", "owner/data\n"])
def test_invalid_repo_id(dataset, repo):
    """仓库标识必须是两个非空路径分量。"""
    assert upload.main(["--input", str(dataset), "--repo-id", repo, "--dry-run"]) == 2


def test_directory_upload_to_existing_repo(dataset, fake_sdk, capsys):
    """默认目录上传保留 ZIP，目标类型、分支及 URL 正确，无仓库创建。"""
    assert upload.main(args(dataset)) == 0
    archive = dataset.with_name("target.zarr.zip")
    assert archive.is_file()
    assert fake_sdk.token == "ms-test-secret"
    assert fake_sdk.endpoint == "https://modelscope.cn"
    assert fake_sdk.creates == 0
    method, options = fake_sdk.calls[-1]
    assert method == "upload_file"
    assert options["path_or_fileobj"] == str(archive)
    assert options["path_in_repo"] == "target.zarr.zip"
    assert options["repo_type"] == "dataset"
    assert options["revision"] == "master"
    assert options["disable_tqdm"] is False
    captured = capsys.readouterr()
    assert "上传成功：https://modelscope.cn/datasets/test-owner/test-data" in captured.out
    assert fake_sdk.token not in captured.out + captured.err
    cache = archive.with_name(archive.name + ".upload-cache.json")
    assert fake_sdk.token not in cache.read_text()


def test_existing_file_is_uploaded_without_packaging(tmp_path, fake_sdk, monkeypatch):
    """文件直接上传，允许自定义仓库内路径和分支。"""
    source = tmp_path / "existing.zip"
    source.write_bytes(b"already packaged")

    def forbidden(*args, **kwargs):
        raise AssertionError("文件不应重新打包")

    monkeypatch.setattr(upload, "pack_directory", forbidden)
    assert upload.main(args(source, "--path-in-repo", "train/custom.zip",
                            "--revision", "dev")) == 0
    options = fake_sdk.calls[-1][1]
    assert options["path_or_fileobj"] == str(source)
    assert options["path_in_repo"] == "train/custom.zip"
    assert options["revision"] == "dev"
    assert list(tmp_path.iterdir()) == [source]
    assert upload.main(args(source, "--archive-path", str(tmp_path / "new.zip"),
                            "--dry-run")) == 2


@pytest.mark.parametrize("failure", ["auth", "missing_repo", "upload"])
def test_sdk_errors_do_not_leak_token(dataset, fake_sdk, capsys, failure):
    """认证、仓库及上传失败输出清除 Token，上传失败保留可复用 ZIP。"""
    token = "ms-secret/with+encoding"
    os.environ["MODELSCOPE_API_TOKEN"] = token
    error = RuntimeError(f"request failed: {token}, {quote(token, safe='')}")
    if failure == "upload":
        fake_sdk.upload_error = error
    else:
        fake_sdk.repo_error = error
    assert upload.main(args(dataset)) == 1
    captured = capsys.readouterr()
    assert token not in captured.out + captured.err
    assert quote(token, safe="") not in captured.out + captured.err
    assert "[REDACTED]" in captured.err
    archive = dataset.with_name("target.zarr.zip")
    assert archive.exists() == (failure == "upload")
    assert fake_sdk.creates == 0
    if failure == "upload":
        initial = upload.archive_state(archive)
        fake_sdk.upload_error = None
        assert upload.main(args(dataset)) == 0
        assert upload.archive_state(archive) == initial


def test_noninteractive_missing_token_is_clear(dataset, monkeypatch, capsys):
    """非交互环境缺少 Token 时不等待输入。"""
    monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert upload.main(args(dataset)) == 2
    assert "MODELSCOPE_API_TOKEN" in capsys.readouterr().err
    assert not dataset.with_name("target.zarr.zip").exists()


def test_interactive_hidden_token(monkeypatch):
    """终端环境使用 getpass 隐藏输入。"""
    monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(upload.getpass, "getpass", lambda prompt: "  ms-hidden  ")
    assert upload.read_token() == "ms-hidden"


def test_missing_sdk_error_is_actionable(monkeypatch):
    """缺少 SDK 时给出当前仓库环境的安装命令。"""
    monkeypatch.setitem(sys.modules, "modelscope_hub", None)
    with pytest.raises(RuntimeError, match="requirements-uv.txt"):
        upload.create_api("test-token")
