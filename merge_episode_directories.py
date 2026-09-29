"""Copy episodes from ordered task directories into one renumbered directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
from typing import Sequence
import uuid


EPISODE_NAME = re.compile(r"episode_([0-9]+)\Z")


def _normalized_path(path: Path) -> Path:
    """Resolve a path after rejecting symbolic links in its existing components."""
    path = path.expanduser().absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"路径包含符号链接: {component}")
    return path.resolve()


def _episode_directories(source: Path) -> list[Path]:
    """Return only direct episode children, sorted by number and then name."""
    episodes = []
    for child in source.iterdir():
        match = EPISODE_NAME.fullmatch(child.name)
        if match is None:
            continue
        if child.is_symlink():
            raise ValueError(f"episode 是符号链接: {child}")
        if child.is_dir():
            episodes.append((int(match.group(1)), child.name, child))
    if not episodes:
        raise ValueError(f"源目录没有直属 episode_N: {source}")
    return [path for _, _, path in sorted(episodes)]


def _check_episode_tree(episode: Path) -> None:
    """Reject links so copying cannot pull data from outside an episode."""
    for parent, directories, files in os.walk(episode, followlinks=False):
        for name in directories + files:
            child = Path(parent) / name
            if child.is_symlink():
                raise ValueError(f"episode 内含符号链接: {child}")


def _prepare(inputs: Sequence[Path], output: Path) -> tuple[list[Path], Path, list[Path]]:
    """Validate the shared dataset root and build the complete ordered input."""
    if not inputs:
        raise ValueError("至少需要一个源目录")
    sources = [_normalized_path(Path(item)) for item in inputs]
    output = _normalized_path(Path(output))
    if len(set(sources)) != len(sources):
        raise ValueError("源目录不能重复")
    if any(not source.is_dir() for source in sources):
        raise ValueError("所有源路径都必须是目录")
    if len(sources[0].parents) < 2 or len(output.parents) < 2:
        raise ValueError("路径必须采用 数据根目录/组别/任务名 结构")
    data_root = sources[0].parents[1]
    if any(source.parents[1] != data_root for source in sources):
        raise ValueError("所有源目录必须位于同一数据根目录")
    if output.parents[1] != data_root:
        raise ValueError("输出目录必须位于同一数据根目录的组别下")
    if not output.parent.is_dir():
        raise ValueError(f"输出的组别目录不存在: {output.parent}")
    if os.path.lexists(output):
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output}")

    episodes = []
    for source in sources:
        children = _episode_directories(source)
        for episode in children:
            _check_episode_tree(episode)
        episodes.extend(children)
    return sources, output, episodes


def _update_recording(episode: Path, relative_path: Path, output: Path) -> tuple[str | None, str | None]:
    """Give a copied recording its new path, identity, and exact directory size."""
    recording_path = episode / "recording.json"
    if not recording_path.exists():
        return None, None
    recording = json.loads(recording_path.read_text(encoding="utf-8"))
    if not isinstance(recording, dict):
        raise ValueError(f"recording.json 必须是 JSON 对象: {recording_path}")
    original_id = recording.get("recording_id")
    new_id = str(uuid.uuid4())
    recording.update({
        "dir_name": output.parent.name,
        "name": output.name,
        "relative_path": relative_path.as_posix(),
        "recording_id": new_id,
        "size_bytes": 0,
    })
    # size_bytes includes recording.json itself, so write until its value and
    # the serialized file size agree (normally two or three iterations).
    for _ in range(10):
        recording_path.write_text(
            json.dumps(recording, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        size = sum(path.stat().st_size for path in episode.rglob("*") if path.is_file())
        if recording["size_bytes"] == size:
            return original_id, new_id
        recording["size_bytes"] = size
    raise ValueError(f"无法稳定计算 episode 大小: {episode}")


def merge_episode_directories(inputs: Sequence[Path], output: Path) -> dict:
    """Copy and renumber episodes, publishing the result only after success."""
    sources, output, episodes = _prepare(inputs, output)
    data_root = sources[0].parents[1]
    stage = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    manifest = {
        "source_directories": [source.relative_to(data_root).as_posix() for source in sources],
        "output_directory": output.relative_to(data_root).as_posix(),
        "episodes": [],
    }
    try:
        stage.mkdir()
        for index, source_episode in enumerate(episodes):
            target_name = f"episode_{index}"
            copied_episode = stage / target_name
            shutil.copytree(source_episode, copied_episode)
            target_relative = output.relative_to(data_root) / target_name
            original_id, new_id = _update_recording(
                copied_episode, target_relative, output,
            )
            manifest["episodes"].append({
                "source": source_episode.relative_to(data_root).as_posix(),
                "target": target_relative.as_posix(),
                "original_recording_id": original_id,
                "new_recording_id": new_id,
            })
        (stage / "merge_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if os.path.lexists(output):
            raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output}")
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ordered source directories and publish their copied episodes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+", type=Path,
                        help="按给定顺序合并的源任务目录，每个目录直属包含 episode_N")
    parser.add_argument("--output", required=True, type=Path,
                        help="新的输出任务目录；其组别目录必须已存在")
    arguments = parser.parse_args(argv)
    try:
        manifest = merge_episode_directories(arguments.input, arguments.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"完成: {len(manifest['episodes'])} 个 episode -> {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
