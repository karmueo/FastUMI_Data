"""验证采集生命周期状态机：独立与组合操作、幂等、冲突和故障。"""

import threading
import uuid

import pytest
import yaml

from collection_harness import MemoryBackend, make_harness
from fastumi_data.collection_state import CollectionConfig, CollectionState
from fastumi_data.collection_storage import CollectionStorage, StorageError
from fastumi_data.collection_writer import QueuedBagWriter


# 测试使用的固定 UUID 序列。
UUIDS = [str(uuid.uuid4()) for _ in range(8)]
IMAGE_TOPIC = "/umi_camera/image_raw"
POSE_TOPIC = "/vive_tracker/pose"
EVENT_TOPIC = "/fastumi/episode/events"


class _BlockedCloseBackend(MemoryBackend):
    """用事件阻塞关闭，验证超时后的采集归属和清理边界。"""

    def __init__(self, fail_on_close=False):
        """配置关闭失败，并创建进入关闭与允许退出的同步事件。"""
        super().__init__(fail_on_close=fail_on_close)
        # 允许测试确认线程已进入关闭阶段。
        self.close_started = threading.Event()
        # 测试主动放行前，写入线程不能退出。
        self.release_close = threading.Event()

    def close(self):
        """等待测试放行后调用后端关闭；意外未放行时抛出超时。"""
        self.close_started.set()
        if not self.release_close.wait(timeout=5):
            raise TimeoutError("测试未放行关闭")
        super().close()


def _start(harness, request_id="r-start", task="pick_place", name=""):
    """开始采集并断言成功，返回 UUID。"""
    result = harness.controller.start(request_id, task, name)
    assert result.accepted and result.completed, result
    assert result.state == "recording"
    return result.collection_uuid


def _push_image(harness, payload: bytes) -> None:
    """绕过 CDR 解析直接投递图像字节：测试专用的入队入口。"""
    harness.controller._submit(IMAGE_TOPIC, payload)  # noqa: SLF001


def test_start_stop_save_publishes_record_with_ordered_mcap(tmp_path) -> None:
    """验证完整流程发布正式目录，MCAP 以 START 开头、STOP 结尾，计数一致。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    identifier = _start(harness, name="第一次")
    _push_image(harness, b"image-0")
    harness.controller.on_tracker_pose(b"pose-0", harness.clock.wall, True)
    harness.controller.on_gripper_state(b"gripper-0", harness.clock.wall, True, 0.5)
    _push_image(harness, b"image-1")

    stopped = harness.controller.stop("r-stop", identifier)
    saved = harness.controller.save("r-save", identifier)

    assert stopped.completed and stopped.state == "pending"
    assert saved.accepted and saved.completed and saved.state == "idle"
    backend = harness.backends[0]
    topics = [topic for topic, _, _ in backend.writes]
    assert topics[0] == EVENT_TOPIC and topics[-1] == EVENT_TOPIC
    assert backend.writes[0][1].startswith(b"START|" + identifier.encode())
    assert backend.writes[-1][1].startswith(b"STOP|" + identifier.encode())
    assert topics.count(IMAGE_TOPIC) == 2
    assert topics.count(POSE_TOPIC) == 1
    assert backend.closed
    destination = harness.storage.find_saved(identifier)
    assert destination is not None
    assert destination.parent.name == "pick_place"
    assert destination.name.endswith(f"_{identifier}")
    manifest = yaml.safe_load((destination / "session.yaml").read_text("utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["collection_uuid"] == manifest["session_id"] == identifier
    assert manifest["episode_index"] == 0
    assert manifest["name"] == "第一次"
    assert manifest["bag_uri"] == "raw/bag"
    assert manifest["calibration_status"] == "uncalibrated"
    assert manifest["tracker_to_tcp"] is None
    assert manifest["message_counts"]["image"] == 2
    assert manifest["state"] == "saved"
    assert manifest["size_bytes"] > 0
    processing = yaml.safe_load(
        (destination / "calibration_snapshot" / "processing.yaml").read_text("utf-8")
    )
    assert processing["topics"]["image"] == IMAGE_TOPIC
    quality = yaml.safe_load((destination / "quality.yaml").read_text("utf-8"))
    assert quality["message_counts"]["image"] == 2
    assert "timing_note" in quality
    assert list(harness.storage.staging_root.iterdir()) == []


def test_messages_outside_recording_window_are_not_written(tmp_path) -> None:
    """验证开始前和停止后的消息不会进入 MCAP，START 总在传感器数据之前。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    _push_image(harness, b"before")
    identifier = _start(harness)
    _push_image(harness, b"during")
    harness.controller.stop("r-stop", identifier)
    _push_image(harness, b"after")
    harness.controller.cancel("r-cancel", identifier)

    payloads = [data for _, data, _ in harness.backends[0].writes]

    assert b"before" not in payloads and b"after" not in payloads
    assert payloads[1] == b"during"


def test_log_times_are_monotonic_and_stop_is_last(tmp_path) -> None:
    """验证 MCAP 日志时间不回退，STOP 不早于任何传感器消息。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    identifier = _start(harness)
    for _ in range(5):
        harness.clock.advance(10)
        _push_image(harness, b"x")
    harness.controller.stop("r-stop", identifier)

    stamps = [stamp for _, _, stamp in harness.backends[0].writes]

    assert stamps == sorted(stamps)


def test_stop_and_save_skips_pending_state(tmp_path) -> None:
    """验证组合保存直接 stopping -> saving -> idle，不暴露 pending。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    identifier = _start(harness)
    harness.states.clear()

    result = harness.controller.stop_and_save("r-both", identifier)

    assert result.completed and result.state == "idle"
    assert "pending" not in harness.states
    assert "stopping" in harness.states and "saving" in harness.states
    assert harness.storage.find_saved(identifier) is not None


def test_pending_record_blocks_new_start_until_saved_or_cancelled(tmp_path) -> None:
    """验证待保存阻止再次开始，保存后可以开始下一条。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    identifier = _start(harness)
    harness.controller.stop("r-stop", identifier)

    blocked = harness.controller.start("r-second", "pick_place", "")
    assert not blocked.accepted and blocked.code == "PENDING_RECORD"
    assert blocked.collection_uuid == identifier

    harness.controller.save("r-save", identifier)
    harness.feed()
    second = harness.controller.start("r-third", "pick_place", "")
    assert second.completed and second.collection_uuid != identifier


def test_stop_and_cancel_discards_recording(tmp_path) -> None:
    """验证停止并取消清理暂存，不产生正式记录。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    identifier = _start(harness)
    _push_image(harness, b"x")
    harness.states.clear()

    result = harness.controller.stop_and_cancel("r-cancel", identifier)

    assert result.completed and result.state == "idle"
    assert harness.states[0] == "cancelling" and harness.states[-1] == "idle"
    assert harness.storage.find_saved(identifier) is None
    assert list(harness.storage.staging_root.iterdir()) == []
    assert harness.backends[0].closed


def test_cancel_pending_discards_stopped_recording(tmp_path) -> None:
    """验证独立停止后取消同样清理全部数据。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    identifier = _start(harness)
    harness.controller.stop("r-stop", identifier)

    result = harness.controller.cancel("r-cancel", identifier)

    assert result.completed and result.state == "idle"
    assert list(harness.storage.staging_root.iterdir()) == []
    assert harness.controller.status_snapshot().collection_uuid == ""


def test_duplicate_request_returns_existing_result_without_side_effects(
    tmp_path,
) -> None:
    """验证重复 start 请求返回首次结果，不会创建第二条采集。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    first = harness.controller.start("same", "pick_place", "n")
    second = harness.controller.start("same", "pick_place", "n")

    assert second == first
    assert len(harness.backends) == 1
    assert len(list(harness.storage.staging_root.iterdir())) == 1
    stop_one = harness.controller.stop("stop-1", first.collection_uuid)
    stop_two = harness.controller.stop("stop-1", first.collection_uuid)
    assert stop_two == stop_one and stop_one.completed
    assert harness.backends[0].writes[-1][1].startswith(b"STOP")
    assert [w[0] for w in harness.backends[0].writes].count(EVENT_TOPIC) == 2


def test_duplicate_of_completed_rejection_is_stable(tmp_path) -> None:
    """验证被拒绝的请求在重试时返回同一拒绝，而不是随状态变化。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    wrong = str(uuid.uuid4())
    identifier = _start(harness)
    first = harness.controller.stop("r1", wrong)
    harness.controller.stop("r2", identifier)

    again = harness.controller.stop("r1", wrong)

    assert again == first and again.code == "UUID_MISMATCH"


def test_request_id_reused_for_different_request_is_rejected(tmp_path) -> None:
    """验证同一 request_id 用于不同操作或参数会被拒绝。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    identifier = _start(harness, request_id="shared")

    other_operation = harness.controller.stop("shared", identifier)
    other_task = harness.controller.start("shared", "another_task", "")

    assert other_operation.code == "REQUEST_ID_CONFLICT"
    assert other_task.code == "REQUEST_ID_CONFLICT"
    assert harness.controller.status_snapshot().state == "recording"


def test_empty_request_id_is_rejected(tmp_path) -> None:
    """验证空 request_id 不被受理，也不缓存。"""
    harness = make_harness(tmp_path, uuids=UUIDS)

    result = harness.controller.start("  ", "pick_place", "")

    assert not result.accepted and result.code == "INVALID_REQUEST_ID"
    assert harness.controller.status_snapshot().state == "idle"


@pytest.mark.parametrize(
    ("operation", "needs_pending"),
    [("stop", False), ("save", True), ("cancel", True), ("stop_and_save", False),
     ("stop_and_cancel", False)],
)
def test_wrong_or_malformed_uuid_is_rejected_without_state_change(
    tmp_path, operation, needs_pending
) -> None:
    """验证错误 UUID、非法 UUID 不改变状态且不触碰数据。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    identifier = _start(harness)
    if needs_pending:
        harness.controller.stop("r-stop", identifier)
    expected_state = harness.controller.status_snapshot().state
    method = getattr(harness.controller, operation)

    mismatch = method("r-wrong", str(uuid.uuid4()))
    malformed = method("r-bad", "../../etc/passwd")

    assert (mismatch.accepted, mismatch.code) == (False, "UUID_MISMATCH")
    assert (malformed.accepted, malformed.code) == (False, "INVALID_UUID")
    assert harness.controller.status_snapshot().state == expected_state
    assert len(list(harness.storage.staging_root.iterdir())) == 1


def test_operations_without_active_collection_are_rejected(tmp_path) -> None:
    """验证 idle 时停止、保存、取消都返回无活动采集。"""
    harness = make_harness(tmp_path)

    for index, operation in enumerate(
        ("stop", "save", "cancel", "stop_and_save", "stop_and_cancel")
    ):
        result = getattr(harness.controller, operation)(f"r{index}", str(uuid.uuid4()))
        assert result.code == "NO_ACTIVE_COLLECTION" and not result.accepted


def test_state_conflicts_are_rejected(tmp_path) -> None:
    """验证录制中不能保存或取消，待保存时不能再停止。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    identifier = _start(harness)

    assert harness.controller.save("a", identifier).code == "STATE_CONFLICT"
    assert harness.controller.cancel("b", identifier).code == "STATE_CONFLICT"
    assert harness.controller.start("c", "pick_place", "").code == "STATE_CONFLICT"
    harness.controller.stop("d", identifier)
    assert harness.controller.stop("e", identifier).code == "STATE_CONFLICT"
    assert harness.controller.stop_and_save("f", identifier).code == "STATE_CONFLICT"
    assert harness.controller.stop_and_cancel("g", identifier).code == "STATE_CONFLICT"


def test_concurrent_starts_admit_exactly_one(tmp_path) -> None:
    """验证并发开始请求只有一个成功，其余因状态冲突被拒绝。"""
    gate = threading.Event()
    harness = make_harness(tmp_path, uuids=UUIDS)
    harness.pending_backends.append(MemoryBackend(write_gate=gate))
    results = []
    barrier = threading.Barrier(4)

    def worker(index: int) -> None:
        """同时发起开始请求。"""
        barrier.wait()
        results.append(harness.controller.start(f"r{index}", "pick_place", ""))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    gate.set()

    completed = [r for r in results if r.completed]
    rejected = [r for r in results if not r.accepted]
    assert len(completed) == 1 and len(rejected) == 3
    assert {r.code for r in rejected} <= {"STATE_CONFLICT", "PENDING_RECORD"}
    assert len(harness.backends) == 1


def test_operations_are_rejected_while_stop_is_in_progress(tmp_path) -> None:
    """验证关闭 MCAP 期间的保存、取消、再次停止请求都被拒绝。"""
    close_gate = threading.Event()
    harness = make_harness(tmp_path, uuids=UUIDS)
    harness.pending_backends.append(MemoryBackend(close_gate=close_gate))
    identifier = _start(harness)
    outcome = []
    worker = threading.Thread(
        target=lambda: outcome.append(harness.controller.stop("r-stop", identifier))
    )
    worker.start()
    while harness.controller.status_snapshot().state != "stopping":
        threading.Event().wait(0.01)

    save = harness.controller.save("r-save", identifier)
    cancel = harness.controller.cancel("r-cancel", identifier)
    again = harness.controller.stop("r-stop2", identifier)
    duplicate = harness.controller.stop("r-stop", identifier)
    close_gate.set()
    worker.join(timeout=10)

    assert save.code == cancel.code == again.code == "STATE_CONFLICT"
    assert duplicate.code == "IN_PROGRESS" and duplicate.accepted
    assert not duplicate.completed
    assert outcome[0].completed and outcome[0].state == "pending"


def test_preflight_blocks_start_with_reasons(tmp_path) -> None:
    """验证缺失发布者、过期数据会阻止开始并保持 idle，不创建暂存。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    harness.discovered["gripper"] = False
    harness.clock.advance(400)  # Tracker 超过 0.25 s 未更新。

    result = harness.controller.start("r1", "pick_place", "")
    status = harness.controller.status_snapshot()

    assert not result.accepted and result.code == "PREFLIGHT_FAILED"
    assert "夹爪" in result.message and "Tracker" in result.message
    assert status.state == "idle" and not status.can_start
    assert len(status.start_blockers) >= 2
    assert list(harness.storage.staging_root.iterdir()) == []
    assert harness.backends == []


def test_low_disk_space_blocks_start(tmp_path) -> None:
    """验证磁盘可用空间低于门限时拒绝开始。"""
    harness = make_harness(
        tmp_path, config=CollectionConfig(min_free_disk_bytes=1 << 62)
    )

    result = harness.controller.start("r1", "pick_place", "")

    assert result.code == "PREFLIGHT_FAILED" and "磁盘" in result.message
    assert "DISK_LOW" in harness.controller.status_snapshot().alarms


def test_invalid_task_and_name_are_rejected(tmp_path) -> None:
    """验证非法任务名和名称在开始前被拒绝。"""
    harness = make_harness(tmp_path)

    traversal = harness.controller.start("r1", "../escape", "")
    long_name = harness.controller.start("r2", "pick_place", "x" * 200)

    assert traversal.code == "INVALID_TASK_NAME"
    assert long_name.code == "INVALID_NAME"
    assert not (tmp_path / "escape").exists()


def test_write_failure_enters_error_and_data_is_never_saved(tmp_path) -> None:
    """验证写入故障进入 error；保存与开始被拒绝，只能取消清理。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    harness.pending_backends.append(MemoryBackend(fail_on_write=2))
    identifier = _start(harness)
    for _ in range(5):
        _push_image(harness, b"x")
    # 写入线程异步失败，轮询等待状态切换。
    for _ in range(200):
        if harness.controller.tick().state == "error":
            break
        threading.Event().wait(0.01)

    status = harness.controller.status_snapshot()
    assert status.state == "error"
    assert status.last_result_code == "WRITE_FAILED" and "磁盘已满" in status.last_error
    assert status.can_cancel and not (status.can_start or status.can_save)
    assert harness.controller.save("r-save", identifier).code == "ERROR_STATE"
    assert harness.controller.stop("r-stop", identifier).code == "ERROR_STATE"
    assert harness.controller.start("r-new", "pick_place", "").code == "ERROR_STATE"
    assert harness.storage.find_saved(identifier) is None

    cancelled = harness.controller.cancel("r-cancel", identifier)

    assert cancelled.completed and cancelled.state == "idle"
    assert list(harness.storage.staging_root.iterdir()) == []


def test_queue_overflow_enters_error(tmp_path) -> None:
    """验证写入队列溢出被明确报告并使采集失败。"""
    gate = threading.Event()
    harness = make_harness(
        tmp_path,
        config=CollectionConfig(min_free_disk_bytes=0, queue_max_messages=3),
        uuids=UUIDS,
    )
    harness.pending_backends.append(MemoryBackend(write_gate=gate))
    identifier = _start(harness)

    for _ in range(10):
        _push_image(harness, b"x")
    status = harness.controller.tick()
    gate.set()

    assert status.state == "error"
    assert status.last_result_code == "QUEUE_OVERFLOW"
    assert harness.controller.stop_and_save("r1", identifier).code == "ERROR_STATE"
    assert harness.storage.find_saved(identifier) is None


def test_close_failure_during_stop_is_error_not_pending(tmp_path) -> None:
    """验证关闭 MCAP 失败时停止不会进入 pending，也无法保存。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    harness.pending_backends.append(MemoryBackend(fail_on_close=True))
    identifier = _start(harness)

    stopped = harness.controller.stop("r-stop", identifier)

    assert stopped.accepted and not stopped.completed
    assert stopped.code == "WRITE_FAILED" and stopped.state == "error"
    assert harness.controller.save("r-save", identifier).code == "ERROR_STATE"
    assert harness.storage.find_saved(identifier) is None
    assert harness.controller.cancel("r-cancel", identifier).completed


def test_stop_and_save_with_failure_does_not_publish(tmp_path) -> None:
    """验证组合保存遇到写入故障不会发布正式目录。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    harness.pending_backends.append(MemoryBackend(fail_on_close=True))
    identifier = _start(harness)

    result = harness.controller.stop_and_save("r1", identifier)

    assert not result.completed and result.state == "error"
    assert harness.storage.find_saved(identifier) is None
    assert harness.storage.list_records()[1] == 0


def test_start_failure_cleans_up_and_returns_to_idle(tmp_path) -> None:
    """验证开始阶段后端打开失败时清理暂存并回到 idle。"""

    def failing_factory():
        """返回打开即失败的后端。"""
        backend = MemoryBackend()
        backend.open = lambda uri, topics: (_ for _ in ()).throw(OSError("无法创建"))
        return backend

    harness = make_harness(tmp_path, backend_factory=failing_factory)

    result = harness.controller.start("r1", "pick_place", "")

    assert result.accepted and not result.completed
    assert result.code == "START_FAILED" and result.state == "idle"
    assert list(harness.storage.staging_root.iterdir()) == []
    assert harness.controller.status_snapshot().can_start


def test_delete_saved_record_and_reject_active_or_unknown(tmp_path) -> None:
    """验证删除只接受已保存 UUID：活动采集和未知 UUID 被拒绝。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    saved = _start(harness, request_id="s1")
    harness.controller.stop_and_save("s2", saved)
    harness.feed()
    active = _start(harness, request_id="s3")

    on_active = harness.controller.delete("d1", active)
    unknown = harness.controller.delete("d2", str(uuid.uuid4()))
    deleted = harness.controller.delete("d3", saved)
    again = harness.controller.delete("d4", saved)

    assert on_active.code == "NOT_SAVED" and not on_active.accepted
    assert unknown.code == "NOT_FOUND"
    assert deleted.completed and deleted.code == "OK"
    assert again.code == "NOT_FOUND"
    assert harness.storage.find_saved(saved) is None
    assert harness.controller.status_snapshot().state == "recording"
    assert (harness.storage.trash_root / next(harness.storage.trash_root.iterdir()).name).is_dir()


def test_list_records_through_controller(tmp_path) -> None:
    """验证控制器列表支持任务筛选、分页，并约束非法任务名与上限。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    for index, task in enumerate(("task_a", "task_b", "task_a")):
        harness.feed()
        identifier = _start(harness, request_id=f"s{index}", task=task)
        harness.controller.stop_and_save(f"p{index}", identifier)

    everything = harness.controller.list_records()
    only_a = harness.controller.list_records("task_a", 0, 1)
    invalid = harness.controller.list_records("../x")

    assert everything.success and everything.total == 3
    assert [r.task_name for r in only_a.records] == ["task_a"]
    assert only_a.total == 2
    assert not invalid.success and invalid.code == "INVALID_TASK_NAME"


def test_button_availability_follows_backend_state(tmp_path) -> None:
    """验证各状态下 can_* 由后端计算。"""
    harness = make_harness(tmp_path, uuids=UUIDS)

    def flags():
        """返回当前按钮可用性元组。"""
        s = harness.controller.status_snapshot()
        return (s.can_start, s.can_stop, s.can_save, s.can_cancel,
                s.can_stop_and_save, s.can_stop_and_cancel)

    assert flags() == (True, False, False, False, False, False)
    identifier = _start(harness)
    assert flags() == (False, True, False, False, True, True)
    harness.controller.stop("r1", identifier)
    assert flags() == (False, False, True, True, False, False)
    harness.controller.cancel("r2", identifier)
    assert flags() == (True, False, False, False, False, False)


def test_state_version_increases_and_last_result_is_reported(tmp_path) -> None:
    """验证状态版本单调递增，最近请求结果可供面板匹配。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    versions = [harness.controller.status_snapshot().state_version]
    identifier = _start(harness, request_id="abc")
    versions.append(harness.controller.status_snapshot().state_version)
    rejected = harness.controller.save("def", identifier)
    status = harness.controller.status_snapshot()
    versions.append(status.state_version)

    assert versions == sorted(set(versions))
    assert status.last_request_id == "def"
    assert status.last_operation == "save"
    assert status.last_result_code == rejected.code == "STATE_CONFLICT"
    assert rejected.state_version == status.state_version


def test_shutdown_discards_unsaved_and_keeps_saved(tmp_path) -> None:
    """验证有序退出清理未保存数据，已保存记录保留，根目录锁被释放。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    saved = _start(harness, request_id="s1")
    harness.controller.stop_and_save("s2", saved)
    harness.feed()
    active = _start(harness, request_id="s3")
    _push_image(harness, b"x")

    removed = harness.controller.shutdown()

    assert removed == [active]
    assert harness.storage.find_saved(saved) is not None
    assert harness.backends[1].closed
    from fastumi_data.collection_storage import CollectionStorage

    CollectionStorage(tmp_path / "dataset").acquire()


@pytest.mark.parametrize("fail_on_close", [False, True])
def test_cancel_timeout_keeps_collection_until_writer_exits(
    tmp_path, monkeypatch, fail_on_close
) -> None:
    """取消超时保留数据和锁，关闭完成后可重试，旧故障不影响下一条采集。"""
    harness = make_harness(tmp_path)
    backend = _BlockedCloseBackend(fail_on_close)
    harness.pending_backends.append(backend)
    identifier = _start(harness)
    writer = harness.controller._active.writer
    staged = harness.controller._active.staged.root
    original_abort = writer.abort
    # 缩短真实线程等待，保持超时路径的执行行为。
    monkeypatch.setattr(writer, "abort", lambda: original_abort(0.01))
    try:
        cancelled = harness.controller.stop_and_cancel("cancel-1", identifier)

        assert backend.close_started.wait(timeout=1)
        assert cancelled.accepted and not cancelled.completed
        assert cancelled.code == "CANCEL_TIMEOUT" and cancelled.state == "error"
        assert staged.is_dir()
        status = harness.controller.status_snapshot()
        assert status.collection_uuid == identifier
        assert status.can_cancel and not status.can_start
        assert harness.controller.start("blocked", "demo", "").code == "ERROR_STATE"
        with pytest.raises(StorageError, match="已被另一个采集服务占用"):
            CollectionStorage(harness.storage.root).acquire()

        backend.release_close.set()
        assert writer.finish(1).thread_stopped
        retried = harness.controller.cancel("cancel-2", identifier)
        assert retried.completed and retried.state == "idle"
        assert not staged.exists()
        harness.feed()
        next_id = _start(harness, request_id="next")
        saved = harness.controller.stop_and_save("save-next", next_id)
        assert saved.completed and saved.state == "idle"
        assert harness.storage.find_saved(next_id) is not None
    finally:
        backend.release_close.set()
        original_abort(1)
        harness.controller.shutdown()


def test_shutdown_timeout_preserves_staging_and_root_lock(tmp_path, monkeypatch) -> None:
    """退出超时仍保留活动采集和根目录锁，放行写入线程后可重试退出。"""
    harness = make_harness(tmp_path)
    backend = _BlockedCloseBackend()
    harness.pending_backends.append(backend)
    identifier = _start(harness)
    writer = harness.controller._active.writer
    staged = harness.controller._active.staged.root
    original_abort = writer.abort
    monkeypatch.setattr(writer, "abort", lambda: original_abort(0.01))
    try:
        with pytest.raises(StorageError) as caught:
            harness.controller.shutdown()
        assert caught.value.code == "CANCEL_TIMEOUT"
        assert staged.is_dir()
        assert harness.controller.status_snapshot().collection_uuid == identifier
        with pytest.raises(StorageError):
            CollectionStorage(harness.storage.root).acquire()
        backend.release_close.set()
        assert writer.finish(1).thread_stopped
        assert harness.controller.shutdown() == [identifier]
        assert not staged.exists()
    finally:
        backend.release_close.set()
        original_abort(1)
        harness.controller.shutdown()


def test_start_failure_keeps_staging_when_abort_times_out(tmp_path, monkeypatch) -> None:
    """开始事件编码失败且关闭阻塞时保留采集，允许通过 UUID 重试取消。"""
    harness = make_harness(tmp_path)
    backend = _BlockedCloseBackend()
    harness.pending_backends.append(backend)
    original_abort = QueuedBagWriter.abort
    monkeypatch.setattr(
        QueuedBagWriter, "abort", lambda self: original_abort(self, 0.01)
    )

    def fail_encoding(*args):
        """注入开始事件编码异常，触发启动失败清理。"""
        raise ValueError("事件编码失败")

    monkeypatch.setattr(harness.controller, "_encode_event", fail_encoding)
    try:
        failed = harness.controller.start("failed", "demo", "")
        assert failed.code == "START_FAILED" and failed.state == "error"
        assert failed.collection_uuid
        assert harness.controller._active.staged.root.is_dir()
        backend.release_close.set()
        writer = harness.controller._active.writer
        assert writer.finish(1).thread_stopped
        assert harness.controller.cancel("cancel", failed.collection_uuid).completed
    finally:
        backend.release_close.set()
        harness.controller.shutdown()


@pytest.mark.parametrize("operation", ["stop", "stop_and_save"])
def test_recording_duration_and_rates_exclude_close_time(tmp_path, operation) -> None:
    """验证关闭耗时不改变录制边界、清单时长和质量文件平均频率。"""
    harness = make_harness(tmp_path)

    class SlowCloseBackend(MemoryBackend):
        """关闭时推进可控时钟，模拟五秒的 MCAP 收尾耗时。"""

        def close(self):
            """推进时钟再完成关闭。"""
            harness.clock.advance(5000)
            super().close()

    harness.pending_backends.append(SlowCloseBackend())
    try:
        identifier = _start(harness)
        _push_image(harness, b"first")
        _push_image(harness, b"second")
        harness.clock.advance(1000)
        result = getattr(harness.controller, operation)("stop", identifier)
        assert result.completed
        if operation == "stop":
            assert harness.controller.status_snapshot().duration_s == 1.0
            assert harness.controller.save("save", identifier).completed
        directory = harness.storage.find_saved(identifier)
        manifest = yaml.safe_load((directory / "session.yaml").read_text("utf-8"))
        quality = yaml.safe_load((directory / "quality.yaml").read_text("utf-8"))
        assert manifest["duration_s"] == 1.0
        assert quality["mean_rate_hz"]["image"] == 2.0
        writes = harness.backends[0].writes
        assert writes[-1][2] - writes[0][2] == 1_000_000_000
    finally:
        harness.controller.shutdown()


def test_static_tf_is_replayed_at_start_of_each_collection(tmp_path) -> None:
    """验证缓存的静态 TF 在每条新采集开始时写入，且早于传感器数据。"""
    harness = make_harness(tmp_path, uuids=UUIDS)
    harness.controller.on_tf_static(b"tf-1")
    first = _start(harness, request_id="s1")
    harness.controller.stop_and_save("s2", first)
    harness.feed()
    second = _start(harness, request_id="s3")
    harness.controller.stop_and_save("s4", second)  # 排空并关闭后再检查写入。

    topics = [t for t, _, _ in harness.backends[1].writes]
    assert topics[:2] == [EVENT_TOPIC, "/tf_static"]
    assert harness.backends[1].writes[1][1] == b"tf-1"


def test_state_enum_values_match_interface_constants() -> None:
    """验证状态枚举值与 CollectionStatus 常量一致。"""
    from fastumi_interfaces.msg import CollectionStatus

    for state in CollectionState:
        assert getattr(CollectionStatus, f"STATE_{state.name}") == state.value
