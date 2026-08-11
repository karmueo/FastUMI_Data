"""验证 EpisodeManager 的观察器线性化和冻结语义。"""

from __future__ import annotations

import pytest

rclpy = pytest.importorskip("rclpy")
from fastumi_interfaces.msg import EpisodeEvent
from std_srvs.srv import Trigger

from fastumi_data.episode_manager import EpisodeManager


@pytest.fixture
def manager() -> EpisodeManager:
    """创建并在测试结束后销毁一个 episode 管理节点。"""
    if not rclpy.ok():
        rclpy.init()
    node = EpisodeManager()
    yield node
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def test_observer_precedes_state_commit_and_freeze_rejects(manager: EpisodeManager) -> None:
    """验证观察器看到旧状态，冻结后不接受排队或新转换。"""
    observed = []

    def observer(event: EpisodeEvent) -> None:
        """记录观察顺序和当前管理器状态。"""
        observed.append((event.event_type, manager._active, manager._episode_index))

    manager._event_observer = observer
    start = manager._start(Trigger.Request(), Trigger.Response())
    stop = manager._stop(Trigger.Request(), Trigger.Response())
    snapshot = manager.freeze()
    rejected = manager._start(Trigger.Request(), Trigger.Response())

    assert start.success and stop.success
    assert observed == [(EpisodeEvent.START, False, 0), (EpisodeEvent.STOP, True, 0)]
    assert snapshot.active is False and snapshot.episode_index == 1
    assert snapshot.revision == 2
    assert not rejected.success


def test_observer_failure_does_not_commit_state(manager: EpisodeManager) -> None:
    """验证同步缓存失败会向服务返回失败且索引、状态保持不变。"""
    manager._event_observer = lambda event: (_ for _ in ()).throw(RuntimeError("缓存失败"))

    response = manager._start(Trigger.Request(), Trigger.Response())

    assert not response.success
    assert manager._active is False
    assert manager._episode_index == 0
    assert manager.state_snapshot().revision == 0


def test_try_freeze_idle_rejects_active_without_freezing(manager: EpisodeManager) -> None:
    """验证活动 episode 会拒绝手动保存，结束后仍可原子冻结。"""
    started = manager._start(Trigger.Request(), Trigger.Response())

    active_snapshot = manager.try_freeze_idle()
    stopped = manager._stop(Trigger.Request(), Trigger.Response())
    idle_snapshot = manager.try_freeze_idle()
    rejected = manager._start(Trigger.Request(), Trigger.Response())

    assert started.success
    assert active_snapshot is None
    assert stopped.success
    assert idle_snapshot is not None
    assert idle_snapshot.active is False and idle_snapshot.episode_index == 1
    assert idle_snapshot.revision == 2
    assert not rejected.success


def test_clear_annotations_resets_active_state_and_transient_writer(
    manager: EpisodeManager,
) -> None:
    """验证清空会删除活动边界、重置编号并刷新瞬态发布器。"""
    observed = []
    manager._event_observer = observed.append
    original_publisher = manager._event_publisher
    started = manager._start(Trigger.Request(), Trigger.Response())

    cleared = manager.try_clear_annotations(observed.clear)
    restarted = manager._start(Trigger.Request(), Trigger.Response())

    assert started.success and cleared and restarted.success
    assert manager._active is True
    assert manager._episode_index == 0
    assert manager.state_snapshot().revision == 3
    assert len(observed) == 1
    assert observed[0].episode_index == 0
    assert manager._event_publisher is not original_publisher


def test_clear_annotations_rejects_frozen_manager(manager: EpisodeManager) -> None:
    """验证结束保存冻结后不能再清空标记。"""
    manager.freeze()

    assert not manager.try_clear_annotations(lambda: None)


def test_replace_annotations_reindexes_idle_manager_and_refreshes_publisher(
    manager: EpisodeManager,
) -> None:
    """验证单条删除使用的新事件集会原子替换编号和清空瞬态历史。"""
    observed = []
    manager._event_observer = observed.append
    manager._start(Trigger.Request(), Trigger.Response())
    manager._stop(Trigger.Request(), Trigger.Response())
    manager._start(Trigger.Request(), Trigger.Response())
    manager._stop(Trigger.Request(), Trigger.Response())
    original_publisher = manager._event_publisher

    snapshot = manager.try_replace_annotations(lambda: (tuple(observed[:2]), 1))

    assert snapshot == manager.state_snapshot()
    assert snapshot is not None
    assert snapshot.active is False and snapshot.episode_index == 1
    assert snapshot.revision == 5
    assert manager._event_publisher is not original_publisher


def test_consistent_state_reader_returns_revision_and_locked_payload(
    manager: EpisodeManager,
) -> None:
    """验证权威 payload 与 active、index、revision 来自同一状态锁区间。"""
    manager._start(Trigger.Request(), Trigger.Response())

    snapshot, payload = manager.read_consistent_state(
        lambda: (manager._active, manager._episode_index)
    )

    assert snapshot.active is True
    assert snapshot.episode_index == 0
    assert snapshot.revision == 1
    assert payload == (True, 0)


def test_replace_annotations_rejects_active_or_frozen_manager(
    manager: EpisodeManager,
) -> None:
    """验证活动 episode 和结束保存冻结状态都不能执行单条删除。"""
    callback_calls = []
    manager._start(Trigger.Request(), Trigger.Response())

    active_result = manager.try_replace_annotations(
        lambda: callback_calls.append(True)
    )
    manager._stop(Trigger.Request(), Trigger.Response())
    manager.freeze()
    frozen_result = manager.try_replace_annotations(
        lambda: callback_calls.append(True)
    )

    assert active_result is None
    assert frozen_result is None
    assert callback_calls == []
