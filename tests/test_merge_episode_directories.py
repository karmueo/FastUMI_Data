"""Tests for copying and renumbering recording episode directories."""

import json
from pathlib import Path
import uuid

import pytest

import merge_episode_directories as merger


def _episode(parent: Path, name: str, payload: bytes, *, recording: bool = False) -> Path:
    """Create a small episode with an opaque payload and optional metadata."""
    episode = parent / name
    (episode / "bag").mkdir(parents=True)
    (episode / "bag" / "data.mcap").write_bytes(payload)
    if recording:
        (episode / "recording.json").write_text(json.dumps({
            "dir_name": parent.parent.name,
            "name": parent.name,
            "relative_path": f"{parent.parent.name}/{parent.name}/{name}",
            "recording_id": str(uuid.uuid4()),
            "size_bytes": 0,
            "started_at": 123.0,
        }), encoding="utf-8")
    return episode


def test_merge_orders_copies_and_updates_metadata(tmp_path: Path) -> None:
    """CLI input order wins; numeric ties use names and sources stay intact."""
    group = tmp_path / "dataset" / "rm75"
    first = group / "first"
    second = group / "second"
    _episode(first, "episode_10", b"ten", recording=True)
    _episode(first, "episode_02", b"two", recording=True)
    _episode(first, "episode_2", b"also-two")
    _episode(second, "episode_1", b"one", recording=True)
    _episode(second / "nested", "episode_0", b"ignore")
    (first / "notes.txt").write_text("not an episode")
    output = group / "merged"

    manifest = merger.merge_episode_directories([first, second], output)

    assert sorted(path.name for path in output.glob("episode_*")) == [
        "episode_0", "episode_1", "episode_2", "episode_3",
    ]
    assert [(output / f"episode_{index}" / "bag" / "data.mcap").read_bytes()
            for index in range(4)] == [b"two", b"also-two", b"ten", b"one"]
    assert (first / "episode_02" / "bag" / "data.mcap").read_bytes() == b"two"
    assert (second / "episode_1" / "bag" / "data.mcap").read_bytes() == b"one"
    assert not (output / "notes.txt").exists()
    assert len(manifest["episodes"]) == 4
    assert [item["source"] for item in manifest["episodes"]] == [
        "rm75/first/episode_02", "rm75/first/episode_2",
        "rm75/first/episode_10", "rm75/second/episode_1",
    ]
    assert json.loads((output / "merge_manifest.json").read_text()) == manifest
    assert manifest["episodes"][1]["new_recording_id"] is None

    for index in (0, 2, 3):
        copied = output / f"episode_{index}"
        info = json.loads((copied / "recording.json").read_text())
        source = tmp_path / "dataset" / manifest["episodes"][index]["source"]
        original = json.loads((source / "recording.json").read_text())
        assert info["dir_name"] == "rm75"
        assert info["name"] == "merged"
        assert info["relative_path"] == f"rm75/merged/episode_{index}"
        assert info["started_at"] == original["started_at"]
        assert info["recording_id"] != original["recording_id"]
        assert uuid.UUID(info["recording_id"])
        assert info["size_bytes"] == sum(
            path.stat().st_size for path in copied.rglob("*") if path.is_file()
        )
        assert manifest["episodes"][index]["original_recording_id"] == original["recording_id"]
        assert manifest["episodes"][index]["new_recording_id"] == info["recording_id"]


def test_rejects_existing_output_and_empty_source(tmp_path: Path) -> None:
    """Invalid requests leave source and preexisting output untouched."""
    group = tmp_path / "dataset" / "rm75"
    first = group / "first"
    first.mkdir(parents=True)
    output = group / "merged"
    with pytest.raises(ValueError, match="没有直属"):
        merger.merge_episode_directories([first], output)
    _episode(first, "episode_4", b"original")
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError, match="已存在"):
        merger.merge_episode_directories([first], output)
    assert marker.read_text() == "keep"
    assert (first / "episode_4" / "bag" / "data.mcap").read_bytes() == b"original"


def test_cli_accepts_ordered_input_directories(tmp_path: Path, capsys) -> None:
    """The documented --input/--output interface keeps argument order."""
    group = tmp_path / "dataset" / "rm75"
    first = group / "first"
    second = group / "second"
    _episode(first, "episode_8", b"first")
    _episode(second, "episode_0", b"second")
    output = group / "merged"

    assert merger.main(["--input", str(first), str(second), "--output", str(output)]) == 0
    assert (output / "episode_0" / "bag" / "data.mcap").read_bytes() == b"first"
    assert (output / "episode_1" / "bag" / "data.mcap").read_bytes() == b"second"
    assert "完成: 2 个 episode" in capsys.readouterr().out


def test_copy_failure_removes_staging_directory(tmp_path: Path, monkeypatch) -> None:
    """A mid-merge failure never publishes a partial result."""
    group = tmp_path / "dataset" / "rm75"
    first = group / "first"
    _episode(first, "episode_0", b"zero")
    _episode(first, "episode_1", b"one")
    output = group / "merged"
    original_copytree = merger.shutil.copytree
    calls = 0

    def fail_second_copy(source: Path, destination: Path, *args, **kwargs) -> Path:
        nonlocal calls
        if Path(source).name.startswith("episode_"):
            calls += 1
            if calls == 2:
                raise OSError("simulated copy failure")
        return original_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(merger.shutil, "copytree", fail_second_copy)
    with pytest.raises(OSError, match="simulated"):
        merger.merge_episode_directories([first], output)
    assert not output.exists()
    assert not list(group.glob(".merged.tmp-*"))
    assert (first / "episode_0").is_dir()


def test_rejects_duplicate_sources_and_episode_symlinks(tmp_path: Path) -> None:
    """Aliases and links cannot change the copied input set."""
    group = tmp_path / "dataset" / "rm75"
    first = group / "first"
    episode = _episode(first, "episode_0", b"zero")
    output = group / "merged"
    with pytest.raises(ValueError, match="不能重复"):
        merger.merge_episode_directories([first, first], output)
    (episode / "outside").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="符号链接"):
        merger.merge_episode_directories([first], output)
    assert not output.exists()
