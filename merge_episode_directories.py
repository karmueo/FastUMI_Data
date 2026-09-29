"""按源任务目录顺序复制并重编号直属的 episode 目录。

依赖标准库处理路径、文件复制及 JSON 元数据；源目录保持不变，成功后才公布输出目录。
启动：python merge_episode_directories.py --input SRC [SRC ...] --output DST
启动参数：
    --input SRC [SRC ...]：必填，至少一个源任务目录，按给定顺序合并。
    --output DST：必填，尚不存在的输出任务目录；其组别目录须已存在。
    -h, --help：可选，显示帮助并退出。
输入：同一数据根目录下的源任务目录及其直属 episode_N，路径结构为
    数据根目录/组别/任务名；存在时读取各 episode 的 recording.json。
输出：在 DST 中写入重编号的 episode_N、原有 recording.json 的更新副本和
    merge_manifest.json；成功信息写入标准输出，参数及处理错误写入标准错误。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
from typing import Sequence
import uuid


# 仅识别名称完全符合 episode_N 的直属目录；N 为十进制非负整数。
EPISODE_NAME = re.compile(r"episode_([0-9]+)\Z")


def _normalized_path(path: Path) -> Path:
    """展开用户目录并规范化绝对路径，拒绝已有路径组件中的符号链接。

    Args:
        path: 待检查的源目录或输出目录路径。

    Returns:
        规范化后的绝对路径。

    Raises:
        ValueError: 路径或其已有父级包含符号链接。
    """
    path = path.expanduser().absolute()
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"路径包含符号链接: {component}")
    return path.resolve()


def _episode_directories(source: Path) -> list[Path]:
    """收集源目录直属的 episode_N，并按编号、名称排序。

    Args:
        source: 已确认存在的源任务目录。

    Returns:
        排序后的 episode 目录路径列表；忽略其他名称及非目录条目。

    Raises:
        ValueError: 匹配名称的条目是符号链接，或没有直属 episode_N 目录。
    """
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
    """遍历 episode 内容，拒绝可能指向目录外数据的符号链接。

    Args:
        episode: 待复制的 episode 目录；遍历时不跟随目录符号链接。

    Raises:
        ValueError: episode 中包含文件或目录符号链接。
    """
    for parent, directories, files in os.walk(episode, followlinks=False):
        for name in directories + files:
            child = Path(parent) / name
            if child.is_symlink():
                raise ValueError(f"episode 内含符号链接: {child}")


def _prepare(inputs: Sequence[Path], output: Path) -> tuple[list[Path], Path, list[Path]]:
    """验证源和目标路径，并按源目录顺序整理全部 episode。

    Args:
        inputs: 至少一个源任务目录，列表顺序决定合并顺序。
        output: 尚不存在的输出任务目录，其组别目录须已存在。

    Returns:
        规范化的源路径、规范化的输出路径、按合并顺序排列的 episode 路径。

    Raises:
        ValueError: 源路径重复、不存在、目录结构不符，或 episode 无效、含符号链接。
        FileExistsError: 输出路径已经存在。
    """
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
    # 所有任务目录必须共享的“数据根目录/组别/任务名”中的数据根目录。
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
    """更新副本的 recording.json 中的归属路径、标识和目录字节数。

    Args:
        episode: 已复制到暂存目录的 episode；存在元数据时会原地改写。
        relative_path: 目标 episode 相对于数据根目录的路径。
        output: 最终输出任务目录，用于填写组别名和任务名。

    Returns:
        原 recording_id 与新 UUID 字符串；没有 recording.json 时均为 None。

    Raises:
        ValueError: 元数据不是 JSON 对象，或 size_bytes 无法收敛。
        json.JSONDecodeError: recording.json 不是有效 JSON。
    """
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
    # size_bytes 包含 recording.json 本身；反复写入，直到记录值与实际文件总字节数一致。
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
    """复制并连续重编号 episode，完成后一次性公布输出任务目录。

    Args:
        inputs: 按合并顺序排列的源任务目录。
        output: 不得已存在的目标任务目录。

    Returns:
        与 merge_manifest.json 一致的字典，记录源、目标路径及新旧 recording_id。

    Raises:
        ValueError: 路径、episode 或元数据不符合要求。
        FileExistsError: 校验时或发布前检查时目标目录已存在。
        OSError: 读取、复制、写入或发布文件时失败。

    源目录不会改写；失败时删除暂存目录，不公布未完成的结果。
    """
    sources, output, episodes = _prepare(inputs, output)
    data_root = sources[0].parents[1]
    # 在目标组别目录中暂存副本，全部写入成功后再重命名为正式目录。
    stage = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    # 清单中的路径均相对于共享数据根目录，episode 顺序对应最终编号。
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
    """解析命令行参数，执行合并并向标准输出报告 episode 数量。

    Args:
        argv: 可选的参数序列；为 None 时读取进程命令行。

    Returns:
        合并成功时返回 0；参数或处理错误由 argparse 写入标准错误并退出。

    Raises:
        SystemExit: 参数无效或合并期间发生 OSError、ValueError。
    """
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
