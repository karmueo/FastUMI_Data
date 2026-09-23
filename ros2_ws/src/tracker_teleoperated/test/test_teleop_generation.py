"""验证本机遥操启停代次持久化、迟到请求门控及旧暂停接口。"""

from types import SimpleNamespace

import numpy as np
from fastumi_interfaces.srv import SetTeleopGeneration
from std_srvs.srv import SetBool

from tracker_teleoperated.node import TrackerTeleopNode
from tracker_teleoperated.teleop_generation import TeleopGenerationStore


def control_state(store):
    """创建只记录启停副作用的轻量控制节点状态。"""
    state = SimpleNamespace(
        _generation_store=store,
        _generation_fault=False,
        _enabled=False,
        _homing=False,
        _workspace_calibration_samples=[],
        _mapping_mode="workspace",
        _mapping_basis=np.eye(3),
    )
    state._capture_reference = lambda _initialize_mapping: None
    state._publish_enabled = lambda: None
    state._publish_status = lambda _message: None
    state.get_logger = lambda: SimpleNamespace(warning=lambda _message: None)
    state._stop_homing = lambda _reason: False
    state._cancel_workspace_calibration = lambda: False
    state._disable = lambda _reason, publish_hold, **_kwargs: setattr(
        state, "_enabled", False
    )
    return state


def generation_request(state, generation, operation):
    """调用真实节点代次处理逻辑。"""
    request = SetTeleopGeneration.Request(operation_generation=generation)
    return TrackerTeleopNode._generation_operation(
        state, request, SetTeleopGeneration.Response(), operation
    )


def test_late_enable_is_rejected_after_pause_and_restart(tmp_path):
    """暂停先到时，迟到启用及节点重启后的旧请求均被拒绝。"""
    path = tmp_path / "generation.json"
    store = TeleopGenerationStore(path)
    state = control_state(store)
    try:
        paused = generation_request(state, 1, "disable")
        late = generation_request(state, 1, "enable")
        assert paused.success and not paused.enabled
        assert late.code == "STALE_GENERATION" and not state._enabled
    finally:
        store.close()
    restarted = TeleopGenerationStore(path)
    try:
        state = control_state(restarted)
        assert generation_request(state, 1, "enable").code == "STALE_GENERATION"
        assert generation_request(state, 2, "enable").success
    finally:
        restarted.close()


def test_legacy_interface_only_pauses_and_fences_pending_enable(tmp_path):
    """无代次启用被拒绝，旧暂停消耗原启用请求的代次。"""
    store = TeleopGenerationStore(tmp_path / "generation.json")
    state = control_state(store)
    try:
        rejected = TrackerTeleopNode._set_enabled_callback(
            state, SetBool.Request(data=True), SetBool.Response()
        )
        paused = TrackerTeleopNode._set_enabled_callback(
            state, SetBool.Request(data=False), SetBool.Response()
        )
        late = generation_request(state, 1, "enable")
        assert not rejected.success and "带代次" in rejected.message
        assert paused.success and store.record["generation"] == 1
        assert late.code == "STALE_GENERATION" and not state._enabled
    finally:
        store.close()


def test_home_fences_enable_even_when_already_paused(tmp_path):
    """回位开始时已暂停，仍使尚未到达的启用请求失效。"""
    store = TeleopGenerationStore(tmp_path / "generation.json")
    state = control_state(store)
    try:
        TrackerTeleopNode._disable(
            state, "收到回位请求", publish_hold=False, force_fence=True
        )
        assert store.record["generation"] == 1
        assert generation_request(state, 1, "enable").code == "STALE_GENERATION"
    finally:
        store.close()


def test_safety_pause_advances_generation(tmp_path):
    """运行中的安全暂停会持久化新代次并阻止旧启用请求。"""
    store = TeleopGenerationStore(tmp_path / "generation.json")
    state = control_state(store)
    state._enabled = True
    state._update_gripper = lambda _enabled, _now: None
    state._latest_joint_positions = None
    state._latest_joint_monotonic = 0.0
    state._feedback_timeout_s = 1.0
    state._clear_control_reference = lambda: None
    state._mapping_recovery_instruction = lambda: "等待输入恢复"
    state.get_logger = lambda: SimpleNamespace(
        warning=lambda _message: None, error=lambda _message: None
    )
    try:
        TrackerTeleopNode._disable(state, "跟踪状态无效", publish_hold=True)
        assert not state._enabled
        assert store.record["generation"] == 1
        assert generation_request(state, 1, "enable").code == "STALE_GENERATION"
    finally:
        store.close()


def test_persistence_failure_rejects_enable(tmp_path, monkeypatch):
    """代次无法落盘时不执行启用。"""
    store = TeleopGenerationStore(tmp_path / "generation.json")
    state = control_state(store)
    monkeypatch.setattr(store, "_write", lambda: (_ for _ in ()).throw(OSError("disk full")))
    try:
        result = generation_request(state, 1, "enable")
        assert result.code == "IO_ERROR"
        assert not state._enabled and state._generation_fault
    finally:
        store.close()
