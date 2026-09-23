"""Check generation ordering and recovery independently of robot hardware."""

from types import SimpleNamespace

import pytest

from fastumi_rm75.teleop_generation import TeleopGenerationStore


def test_late_enable_is_rejected_after_disable(tmp_path):
    path = tmp_path / 'generation.json'
    store = TeleopGenerationStore(path)
    try:
        assert store.begin(2, 'disable') == 'new'
        store.finish(True, 'DISABLED', '已禁用')
        assert store.begin(1, 'enable') == 'stale'
        assert store.begin(2, 'enable') == 'stale'
        assert store.begin(2, 'disable') == 'duplicate'
    finally:
        store.close()
    restarted = TeleopGenerationStore(path)
    try:
        assert restarted.record['generation'] == 2
        assert restarted.begin(1, 'enable') == 'stale'
        assert restarted.begin(3, 'enable') == 'new'
    finally:
        restarted.close()


def test_unfinished_operation_is_not_replayed(tmp_path):
    path = tmp_path / 'generation.json'
    store = TeleopGenerationStore(path)
    store.begin(4, 'enable')
    store.close()
    restarted = TeleopGenerationStore(path)
    try:
        assert restarted.record['code'] == 'INTERRUPTED'
        assert restarted.begin(4, 'enable') == 'duplicate'
        assert restarted.begin(3, 'enable') == 'stale'
    finally:
        restarted.close()


def test_failed_persistence_does_not_accept_generation(tmp_path, monkeypatch):
    store = TeleopGenerationStore(tmp_path / 'generation.json')
    try:
        def fail():
            raise OSError('disk full')
        monkeypatch.setattr(store, '_write', fail)
        with pytest.raises(OSError, match='disk full'):
            store.begin(7, 'enable')
        assert store.record['generation'] == 0
    finally:
        store.close()


def test_bridge_rejects_late_enable_without_running_it(tmp_path):
    from fastumi_rm75.rm75_policy_bridge import Rm75PolicyBridge

    calls = []
    bridge = SimpleNamespace(
        _generation_store=TeleopGenerationStore(tmp_path / 'generation.json'),
        _enabled=False,
    )
    def enable(response):
        calls.append('enable')
        bridge._enabled = True
        response.success, response.message = True, 'enabled'
        return response
    def disable(response):
        calls.append('disable')
        bridge._enabled = False
        response.success, response.message = True, 'disabled'
        return response
    bridge._enable_operation = enable
    bridge._disable_operation = disable
    bridge._stop_motion = lambda reason: calls.append('stop')
    try:
        response = SimpleNamespace(success=False, code='', message='')
        Rm75PolicyBridge._generation_operation(
            bridge, SimpleNamespace(operation_generation=2), response, 'disable')
        assert response.success and response.code == 'DISABLED'
        response = SimpleNamespace(success=False, code='', message='')
        Rm75PolicyBridge._generation_operation(
            bridge, SimpleNamespace(operation_generation=1), response, 'enable')
        assert response.code == 'STALE_GENERATION'
        assert calls == ['disable']
    finally:
        bridge._generation_store.close()


def test_bridge_does_not_enable_when_generation_write_fails(tmp_path, monkeypatch):
    from fastumi_rm75.rm75_policy_bridge import Rm75PolicyBridge

    store = TeleopGenerationStore(tmp_path / 'generation.json')
    calls = []
    bridge = SimpleNamespace(
        _generation_store=store, _enabled=False,
        _enable_operation=lambda response: calls.append('enable'),
    )
    monkeypatch.setattr(store, '_write', lambda: (_ for _ in ()).throw(OSError('disk full')))
    try:
        response = SimpleNamespace(success=False, code='', message='')
        Rm75PolicyBridge._generation_operation(
            bridge, SimpleNamespace(operation_generation=1), response, 'enable')
        assert response.code == 'IO_ERROR'
        assert not response.enabled
        assert calls == []
    finally:
        store.close()


def test_bridge_stops_motion_when_disable_persistence_fails(tmp_path, monkeypatch):
    from fastumi_rm75.rm75_policy_bridge import Rm75PolicyBridge

    store = TeleopGenerationStore(tmp_path / 'generation.json')
    stopped = []
    bridge = SimpleNamespace(_generation_store=store, _enabled=True)
    def stop(reason):
        stopped.append(reason)
        bridge._enabled = False
    bridge._stop_motion = stop
    monkeypatch.setattr(store, '_write', lambda: (_ for _ in ()).throw(OSError('disk full')))
    try:
        response = SimpleNamespace(success=False, code='', message='')
        Rm75PolicyBridge._generation_operation(
            bridge, SimpleNamespace(operation_generation=1), response, 'disable')
        assert response.code == 'IO_ERROR'
        assert not response.enabled
        assert stopped
    finally:
        store.close()
