"""验证采集存储的根目录锁、暂存发布、清理、回收区和路径安全。"""

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import uuid

import pytest
import yaml

from fastumi_data.collection_storage import (
    OWNER_MARKER,
    CollectionStorage,
    StorageError,
    parse_uuid,
    validate_collection_name,
    validate_task_name,
)


class _Clock:
    """每次调用前进一秒的固定 UTC 时钟。"""

    def __init__(self) -> None:
        """从固定时刻开始计时。"""
        self._now = datetime(2026, 5, 6, 7, 8, 9, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        """返回当前时刻并前进一秒。"""
        current = self._now
        self._now += timedelta(seconds=1)
        return current


def _storage(tmp_path: Path) -> CollectionStorage:
    """创建已取得根目录锁的存储。"""
    storage = CollectionStorage(tmp_path / "dataset", clock=_Clock())
    storage.acquire()
    return storage


def _save(storage: CollectionStorage, task: str, name: str = "") -> str:
    """创建、写入最小清单并发布一条记录，返回 UUID。"""
    identifier = str(uuid.uuid4())
    staged = storage.create_staging(identifier, task)
    (staged.bag_uri).mkdir(parents=True)
    (staged.bag_uri / "bag_0.mcap").write_bytes(b"mcap")
    created = storage._clock().timestamp()  # noqa: SLF001
    (staged.root / "session.yaml").write_text(
        yaml.safe_dump(
            {
                "collection_uuid": identifier,
                "task_name": task,
                "name": name,
                "created_at_unix": created,
                "saved_at_unix": created + 1,
                "duration_s": 3.5,
                "message_counts": {"image": 7, "tracker_pose": 9, "gripper_state": 7},
                "size_bytes": 4,
                "calibration_status": "calibrated",
                "quality": {"has_alarms": True},
            }
        ),
        encoding="utf-8",
    )
    storage.publish(staged)
    return identifier


def test_second_storage_on_same_root_is_rejected_until_released(
    tmp_path: Path,
) -> None:
    """验证根目录独占锁阻止第二个采集服务，释放后可重新获取。"""
    first = _storage(tmp_path)
    second = CollectionStorage(tmp_path / "dataset")

    with pytest.raises(StorageError) as raised:
        second.acquire()
    assert raised.value.code == "ROOT_LOCKED"

    first.release()
    second.acquire()
    second.release()


def test_publish_moves_staging_to_task_uuid_directory(tmp_path: Path) -> None:
    """验证发布后目录为 <task>/<UTC>_<uuid>，暂存区被清空。"""
    storage = _storage(tmp_path)
    identifier = str(uuid.uuid4())
    staged = storage.create_staging(identifier, "pick_place")
    (staged.root / "session.yaml").write_text("{}", encoding="utf-8")

    destination = storage.publish(staged)

    assert destination == (
        storage.root / "pick_place" / f"20260506T070809Z_{identifier}"
    )
    assert (destination / "raw").is_dir()
    assert not (destination / OWNER_MARKER).exists()
    assert not staged.root.exists()
    assert list(storage.staging_root.iterdir()) == []


def test_cleanup_removes_only_owned_staging_and_keeps_saved_records(
    tmp_path: Path,
) -> None:
    """验证启动清理只删除带归属标记的暂存，已保存记录与未知条目保留。"""
    storage = _storage(tmp_path)
    saved = _save(storage, "task_a")
    leftover = storage.create_staging(str(uuid.uuid4()), "task_a")
    foreign = storage.staging_root / "not-a-uuid"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("x", encoding="utf-8")
    unmarked = storage.staging_root / str(uuid.uuid4())
    unmarked.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "important.txt").write_text("x", encoding="utf-8")
    (storage.staging_root / str(uuid.uuid4())).symlink_to(outside)
    storage.release()

    reopened = CollectionStorage(tmp_path / "dataset")
    removed = reopened.acquire()

    assert removed == []  # release 已清理 leftover，重启时无遗留。
    assert not leftover.root.exists()
    assert foreign.exists() and unmarked.exists()
    assert (outside / "important.txt").exists()
    assert reopened.find_saved(saved) is not None
    reopened.release()


def test_acquire_cleans_leftover_after_crash(tmp_path: Path) -> None:
    """验证进程崩溃（未 release）后下次启动清理遗留暂存。"""
    root = tmp_path / "dataset"
    crashed = CollectionStorage(root)
    crashed.acquire()
    leftover = crashed.create_staging(str(uuid.uuid4()), "task_a")
    saved = _save(crashed, "task_a")
    os.close(crashed._lock_fd)  # noqa: SLF001  模拟进程退出释放锁。
    crashed._lock_fd = None  # noqa: SLF001

    restarted = CollectionStorage(root)
    removed = restarted.acquire()

    assert removed == [leftover.uuid]
    assert not leftover.root.exists()
    assert restarted.find_saved(saved) is not None
    restarted.release()


def test_discard_refuses_directory_outside_staging(tmp_path: Path) -> None:
    """验证丢弃暂存时拒绝不在暂存区内的目录。"""
    storage = _storage(tmp_path)
    staged = storage.create_staging(str(uuid.uuid4()), "task_a")
    foreign = type(staged)(
        staged.uuid, "task_a", staged.dir_name, tmp_path / "elsewhere"
    )
    foreign.root.mkdir()

    with pytest.raises(StorageError) as raised:
        storage.discard(foreign)

    assert raised.value.code == "PATH_ESCAPE"
    assert foreign.root.exists()
    storage.discard(staged)
    assert not staged.root.exists()


def test_delete_moves_whole_directory_to_trash(tmp_path: Path) -> None:
    """验证删除把完整目录移入回收区，任务目录为空时被移除。"""
    storage = _storage(tmp_path)
    identifier = _save(storage, "task_a")
    located = storage.find_saved(identifier)
    assert located is not None

    trashed = storage.delete(identifier)

    assert trashed.parent == storage.trash_root
    assert (trashed / "raw" / "bag" / "bag_0.mcap").read_bytes() == b"mcap"
    assert storage.find_saved(identifier) is None
    assert not (storage.root / "task_a").exists()


def test_delete_rejects_unsaved_and_unknown_uuid(tmp_path: Path) -> None:
    """验证暂存中的采集、未知 UUID 和非法文本都不能被删除。"""
    storage = _storage(tmp_path)
    staged = storage.create_staging(str(uuid.uuid4()), "task_a")

    for target in (staged.uuid, str(uuid.uuid4())):
        with pytest.raises(StorageError) as raised:
            storage.delete(target)
        assert raised.value.code == "NOT_FOUND"
    with pytest.raises(StorageError) as raised:
        storage.delete("../../etc")
    assert raised.value.code == "INVALID_UUID"
    assert staged.root.exists()


def test_symlinked_task_directory_is_ignored(tmp_path: Path) -> None:
    """验证指向根目录外的符号链接任务目录既不列出也不可删除。"""
    storage = _storage(tmp_path)
    outside = tmp_path / "outside"
    identifier = str(uuid.uuid4())
    record = outside / f"20260101T000000Z_{identifier}"
    record.mkdir(parents=True)
    (record / "session.yaml").write_text(
        yaml.safe_dump({"collection_uuid": identifier, "task_name": "x"}),
        encoding="utf-8",
    )
    (storage.root / "linked").symlink_to(outside)

    records, total = storage.list_records()

    assert (records, total) == ([], 0)
    assert storage.find_saved(identifier) is None
    assert record.exists()


def test_list_records_filters_orders_and_paginates(tmp_path: Path) -> None:
    """验证列表按创建时间倒序，支持任务筛选与分页。"""
    storage = _storage(tmp_path)
    first = _save(storage, "task_a", "first")
    second = _save(storage, "task_b", "second")
    third = _save(storage, "task_a", "third")

    everything, total = storage.list_records()
    only_a, total_a = storage.list_records("task_a")
    page, total_page = storage.list_records(offset=1, limit=1)

    assert [record.uuid for record in everything] == [third, second, first]
    assert total == 3
    assert [record.name for record in only_a] == ["third", "first"]
    assert total_a == 2
    assert [record.uuid for record in page] == [second]
    assert total_page == 3
    summary = everything[0]
    assert summary.relative_path.startswith("task_a/")
    assert (summary.image_messages, summary.tracker_messages) == (7, 9)
    assert summary.calibrated and summary.has_alarms
    assert summary.size_bytes == 4


@pytest.mark.parametrize(
    "task",
    ["", "   ", "..", "../x", "a/b", ".hidden", ".staging", "a" * 65, "a b", "x\0y"],
)
def test_invalid_task_names_are_rejected(task: str) -> None:
    """验证任务名不能越出单级目录或占用内部目录。"""
    with pytest.raises(StorageError) as raised:
        validate_task_name(task)
    assert raised.value.code == "INVALID_TASK_NAME"


@pytest.mark.parametrize("task", ["pick_place", "抓取-任务.v2", "_x", "A1"])
def test_valid_task_names_are_accepted(task: str) -> None:
    """验证字母数字、中文、下划线、点和连字符可作为任务名。"""
    assert validate_task_name(f"  {task} ") == task


def test_collection_name_and_uuid_validation() -> None:
    """验证名称长度与控制字符、UUID 规范化。"""
    assert validate_collection_name("  第一次 ") == "第一次"
    with pytest.raises(StorageError):
        validate_collection_name("x" * 129)
    with pytest.raises(StorageError):
        validate_collection_name("a\nb")
    identifier = uuid.uuid4()
    assert parse_uuid(str(identifier).upper()) == str(identifier)
    with pytest.raises(StorageError):
        parse_uuid("not-a-uuid")


def test_operations_require_root_lock(tmp_path: Path) -> None:
    """验证未取得根目录锁时不能创建暂存目录。"""
    storage = CollectionStorage(tmp_path / "dataset")

    with pytest.raises(StorageError) as raised:
        storage.create_staging(str(uuid.uuid4()), "task_a")

    assert raised.value.code == "ROOT_NOT_LOCKED"
