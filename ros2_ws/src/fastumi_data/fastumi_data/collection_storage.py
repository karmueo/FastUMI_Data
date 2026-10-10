"""管理 UMI 采集的数据集根目录、暂存区、正式发布和回收区。

目录布局（均位于数据集根目录内，保证重命名发生在同一文件系统）::

    <root>/.fastumi_collection.lock      根目录独占锁
    <root>/.staging/<uuid>/              未保存采集，含归属标记
    <root>/.trash/<task>__<dir_name>/    已删除的正式记录
    <root>/<task>/<UTC时间>_<uuid>/      已保存的正式记录

清理只会触及带有效归属标记的暂存目录，因此不会误删已保存记录。
"""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
from typing import Any, Callable, List, Mapping, Optional, Tuple
import uuid as uuid_module

import yaml


# 根目录独占锁文件名。
LOCK_NAME = ".fastumi_collection.lock"
# 未保存采集的暂存目录名。
STAGING_NAME = ".staging"
# 已删除记录的回收区目录名。
TRASH_NAME = ".trash"
# 暂存目录内表明归属于本采集服务的标记文件名。
OWNER_MARKER = ".fastumi_staging.json"
# 任务名允许的字符：字母数字（含中文）、下划线、点和连字符，且首字符不为点。
_TASK_NAME_PATTERN = re.compile(r"^\w[\w.\-]{0,63}$")
# 采集名称的最大长度。
MAX_NAME_LENGTH = 128
# 正式目录名中的 UTC 时间格式。
DIR_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
# 预留的任务目录名，避免与内部目录冲突。
_RESERVED_NAMES = frozenset({STAGING_NAME, TRASH_NAME})


class StorageError(Exception):
    """带机器可读错误码的存储层异常。"""

    def __init__(self, code: str, message: str) -> None:
        """保存错误码和面向操作员的说明。"""
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class StagedCollection:
    """保存一条暂存采集的路径信息。"""

    uuid: str
    task_name: str
    dir_name: str
    """发布后的目录名 ``<UTC时间>_<uuid>``。"""
    root: Path
    """暂存目录 ``<root>/.staging/<uuid>``。"""

    @property
    def raw_dir(self) -> Path:
        """``raw/`` 目录，rosbag2 在其下创建 ``bag``。"""
        return self.root / "raw"

    @property
    def bag_uri(self) -> Path:
        """rosbag2 MCAP 存储目录。"""
        return self.raw_dir / "bag"

    @property
    def snapshot_dir(self) -> Path:
        """标定与处理配置快照目录。"""
        return self.root / "calibration_snapshot"


@dataclass(frozen=True)
class SavedRecord:
    """保存已发布记录列表所需的摘要。"""

    uuid: str
    task_name: str
    name: str
    dir_name: str
    relative_path: str
    created_at: float
    saved_at: float
    duration_s: float
    image_messages: int
    tracker_messages: int
    gripper_messages: int
    size_bytes: int
    calibrated: bool
    has_alarms: bool


def validate_task_name(task_name: str) -> str:
    """校验任务名可安全用作单级目录名。

    Args:
        task_name: 操作员提供的任务名。

    Returns:
        去除首尾空白后的任务名。

    Raises:
        StorageError: 为空、过长、含路径分隔符或为保留名称。
    """
    cleaned = task_name.strip()
    if not cleaned:
        raise StorageError("INVALID_TASK_NAME", "任务名不能为空")
    if cleaned in _RESERVED_NAMES or cleaned.startswith("."):
        raise StorageError("INVALID_TASK_NAME", f"任务名不能以点开头: {cleaned}")
    if not _TASK_NAME_PATTERN.match(cleaned):
        raise StorageError(
            "INVALID_TASK_NAME",
            "任务名只能包含字母、数字、下划线、点和连字符，且不超过 64 个字符",
        )
    return cleaned


def validate_collection_name(name: str) -> str:
    """校验可选采集名称，名称只写入元数据，不参与路径。"""
    cleaned = name.strip()
    if len(cleaned) > MAX_NAME_LENGTH:
        raise StorageError(
            "INVALID_NAME", f"采集名称不能超过 {MAX_NAME_LENGTH} 个字符"
        )
    if any(ord(char) < 32 for char in cleaned):
        raise StorageError("INVALID_NAME", "采集名称不能包含控制字符")
    return cleaned


def parse_uuid(text: str) -> str:
    """把任意 UUID 文本规范化为小写带连字符形式。

    Raises:
        StorageError: 文本不是合法 UUID。
    """
    try:
        return str(uuid_module.UUID(text.strip()))
    except (ValueError, AttributeError) as error:
        raise StorageError("INVALID_UUID", f"不是合法的采集 UUID: {text!r}") from error


def atomic_write_yaml(path: Path, document: Mapping[str, Any]) -> None:
    """先写同目录临时文件再原子替换，避免留下半个 YAML。"""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(dict(document), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def directory_size_bytes(path: Path) -> int:
    """统计目录内常规文件字节数，忽略符号链接。"""
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            total += child.stat().st_size
    return total


class CollectionStorage:
    """持有数据集根目录锁，并提供暂存、发布、列表和回收区操作。"""

    def __init__(
        self,
        root: Path,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        """保存根目录和 UTC 时钟，不触碰文件系统。

        Args:
            root: 数据集根目录，可以尚不存在。
            clock: 返回带时区当前时间的函数，测试中可注入固定时钟。
        """
        self._root = Path(root).expanduser().absolute()
        self._clock = clock
        self._lock_fd: Optional[int] = None

    @property
    def root(self) -> Path:
        """数据集根目录的绝对路径。"""
        return self._root

    @property
    def staging_root(self) -> Path:
        """未保存采集的暂存区目录。"""
        return self._root / STAGING_NAME

    @property
    def trash_root(self) -> Path:
        """已删除记录的回收区目录。"""
        return self._root / TRASH_NAME

    def acquire(self) -> List[str]:
        """创建根目录、取得独占锁并清理上次遗留的暂存数据。

        Returns:
            被清理的暂存目录名；无遗留时为空。

        Raises:
            StorageError: 根目录已被其他采集服务占用或无法创建。
        """
        if self._lock_fd is not None:
            raise StorageError("ROOT_LOCKED", "数据集根目录已被本对象锁定")
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self._root / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644
            )
        except OSError as error:
            raise StorageError(
                "ROOT_UNAVAILABLE", f"无法使用数据集根目录 {self._root}: {error}"
            ) from error
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(descriptor)
            raise StorageError(
                "ROOT_LOCKED",
                f"数据集根目录 {self._root} 已被另一个采集服务占用",
            ) from error
        self._lock_fd = descriptor
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode())
        self.staging_root.mkdir(exist_ok=True)
        self.trash_root.mkdir(exist_ok=True)
        return self.cleanup_staging()

    def release(self) -> List[str]:
        """清理本服务遗留的暂存数据并释放根目录锁。"""
        removed: List[str] = []
        if self._lock_fd is None:
            return removed
        try:
            removed = self.cleanup_staging()
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None
        return removed

    def _require_lock(self) -> None:
        """未持有根目录锁时拒绝任何写操作。"""
        if self._lock_fd is None:
            raise StorageError("ROOT_NOT_LOCKED", "尚未取得数据集根目录锁")

    def cleanup_staging(self) -> List[str]:
        """删除全部带有效归属标记的暂存目录。

        只处理 ``.staging`` 下名称为合法 UUID、非符号链接且标记中 UUID
        一致的目录；其余条目原样保留，避免误删。
        """
        self._require_lock()
        removed: List[str] = []
        for entry in sorted(self.staging_root.iterdir()):
            if not self._is_owned_staging(entry):
                continue
            shutil.rmtree(entry)
            removed.append(entry.name)
        return removed

    def _is_owned_staging(self, entry: Path) -> bool:
        """判断暂存条目是否确由本服务创建。"""
        if entry.is_symlink() or not entry.is_dir():
            return False
        try:
            if str(uuid_module.UUID(entry.name)) != entry.name:
                return False
            marker = json.loads(
                (entry / OWNER_MARKER).read_text(encoding="utf-8")
            )
        except (ValueError, OSError):
            return False
        return isinstance(marker, dict) and marker.get("uuid") == entry.name

    def create_staging(
        self, collection_uuid: str, task_name: str
    ) -> StagedCollection:
        """创建带归属标记的暂存目录。

        Args:
            collection_uuid: 已规范化的后端生成 UUID。
            task_name: 已校验的任务名。

        Raises:
            StorageError: 目录已存在或无法创建。
        """
        self._require_lock()
        task = validate_task_name(task_name)
        identifier = parse_uuid(collection_uuid)
        started = self._clock()
        dir_name = f"{started.strftime(DIR_TIME_FORMAT)}_{identifier}"
        root = self.staging_root / identifier
        try:
            root.mkdir()
            (root / "raw").mkdir()
            (root / "calibration_snapshot").mkdir()
            (root / OWNER_MARKER).write_text(
                json.dumps(
                    {
                        "uuid": identifier,
                        "task_name": task,
                        "pid": os.getpid(),
                        "created_at": started.isoformat(),
                    }
                ),
                encoding="utf-8",
            )
        except OSError as error:
            shutil.rmtree(root, ignore_errors=True)
            raise StorageError(
                "STAGING_FAILED", f"无法创建暂存目录: {error}"
            ) from error
        return StagedCollection(identifier, task, dir_name, root)

    def discard(self, staged: StagedCollection) -> None:
        """删除一条暂存采集；目录不属于本服务时拒绝。"""
        self._require_lock()
        if staged.root.parent != self.staging_root:
            raise StorageError("PATH_ESCAPE", "暂存目录不在暂存区内")
        if not self._is_owned_staging(staged.root):
            if staged.root.exists():
                raise StorageError(
                    "STAGING_NOT_OWNED", f"拒绝清理非本服务暂存目录: {staged.root}"
                )
            return
        shutil.rmtree(staged.root)

    def publish(self, staged: StagedCollection) -> Path:
        """把暂存目录原子重命名为正式记录目录。

        Returns:
            正式目录的绝对路径。

        Raises:
            StorageError: 暂存目录无效、目标已存在或重命名失败。
        """
        self._require_lock()
        if not self._is_owned_staging(staged.root):
            raise StorageError("STAGING_NOT_OWNED", "暂存目录缺失或归属标记无效")
        task_dir = self._root / validate_task_name(staged.task_name)
        destination = task_dir / staged.dir_name
        if destination.exists():
            raise StorageError("PUBLISH_EXISTS", f"正式目录已存在: {destination}")
        try:
            task_dir.mkdir(exist_ok=True)
            os.rename(staged.root, destination)
        except OSError as error:
            raise StorageError(
                "PUBLISH_FAILED", f"无法发布采集目录: {error}"
            ) from error
        # 正式目录已脱离暂存区，不会再被清理；残留标记仅是多余文件。
        (destination / OWNER_MARKER).unlink(missing_ok=True)
        return destination

    def find_saved(self, collection_uuid: str) -> Optional[Path]:
        """按 UUID 查找已保存记录目录；不存在时返回 ``None``。"""
        identifier = parse_uuid(collection_uuid)
        for task_dir in self._task_dirs():
            for candidate in task_dir.iterdir():
                if (
                    candidate.name.endswith(f"_{identifier}")
                    and candidate.is_dir()
                    and not candidate.is_symlink()
                    and (candidate / "session.yaml").is_file()
                ):
                    return self._inside_root(candidate)
        return None

    def _inside_root(self, path: Path) -> Path:
        """确认解析后的路径位于根目录内，防止符号链接越界。"""
        resolved = path.resolve()
        if not resolved.is_relative_to(self._root.resolve()):
            raise StorageError("PATH_ESCAPE", f"路径越出数据集根目录: {path}")
        return path

    def _task_dirs(self) -> List[Path]:
        """返回根目录下全部任务目录，跳过内部目录和符号链接。"""
        if not self._root.is_dir():
            return []
        return sorted(
            entry
            for entry in self._root.iterdir()
            if entry.is_dir()
            and not entry.is_symlink()
            and not entry.name.startswith(".")
        )

    def delete(self, collection_uuid: str) -> Path:
        """把已保存记录完整移入回收区。

        Returns:
            回收区中的目录路径。

        Raises:
            StorageError: 记录不存在或不是已保存记录。
        """
        self._require_lock()
        located = self.find_saved(collection_uuid)
        if located is None:
            raise StorageError("NOT_FOUND", f"未找到已保存记录 {collection_uuid}")
        destination = self.trash_root / f"{located.parent.name}__{located.name}"
        if destination.exists():
            raise StorageError("TRASH_EXISTS", f"回收区已存在同名目录: {destination}")
        try:
            os.rename(located, destination)
            try:
                located.parent.rmdir()
            except OSError:
                pass  # 任务目录仍有其他记录。
        except OSError as error:
            raise StorageError(
                "DELETE_FAILED", f"无法移入回收区: {error}"
            ) from error
        return destination

    def list_records(
        self, task_name: str = "", offset: int = 0, limit: int = 50
    ) -> Tuple[List[SavedRecord], int]:
        """按创建时间倒序列出已保存记录。

        Args:
            task_name: 非空时只列出该任务。
            offset: 跳过的记录数。
            limit: 返回的最大记录数。

        Returns:
            当前页记录和筛选后的总数。
        """
        task_filter = validate_task_name(task_name) if task_name.strip() else ""
        records: List[SavedRecord] = []
        for task_dir in self._task_dirs():
            if task_filter and task_dir.name != task_filter:
                continue
            for candidate in task_dir.iterdir():
                record = self._read_record(candidate)
                if record is not None:
                    records.append(record)
        records.sort(key=lambda item: (item.created_at, item.uuid), reverse=True)
        return records[offset:offset + limit], len(records)

    def _read_record(self, directory: Path) -> Optional[SavedRecord]:
        """从 ``session.yaml`` 读取记录摘要，无法解析的目录被忽略。"""
        if directory.is_symlink() or not directory.is_dir():
            return None
        manifest_path = directory / "session.yaml"
        try:
            document = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                return None
            identifier = parse_uuid(str(document["collection_uuid"]))
            counts = document.get("message_counts") or {}
            quality = document.get("quality") or {}
            return SavedRecord(
                uuid=identifier,
                task_name=str(document["task_name"]),
                name=str(document.get("name", "")),
                dir_name=directory.name,
                relative_path=str(directory.relative_to(self._root)),
                created_at=float(document.get("created_at_unix", 0.0)),
                saved_at=float(document.get("saved_at_unix", 0.0)),
                duration_s=float(document.get("duration_s", 0.0)),
                image_messages=int(counts.get("image", 0)),
                tracker_messages=int(counts.get("tracker_pose", 0)),
                gripper_messages=int(counts.get("gripper_state", 0)),
                size_bytes=int(document.get("size_bytes", 0)),
                calibrated=document.get("calibration_status") == "calibrated",
                has_alarms=bool(quality.get("has_alarms", False)),
            )
        except (OSError, KeyError, ValueError, TypeError, StorageError, yaml.YAMLError):
            return None

    def free_disk_bytes(self) -> int:
        """返回数据集根目录所在文件系统的可用字节数。"""
        target = self._root if self._root.exists() else self._root.parent
        return shutil.disk_usage(target).free
