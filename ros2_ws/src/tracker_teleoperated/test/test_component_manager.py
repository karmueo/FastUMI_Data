"""验证本机管理边界和远端录制退出状态机。"""

from concurrent.futures import Future
from pathlib import Path
import time

from fastumi_interfaces.msg import RecordingInfo, RecordingStatus
from fastumi_interfaces.srv import GetRecordingStatus, StopRecording
from rclpy.parameter import Parameter
from std_srvs.srv import SetBool
import pytest
import rclpy
import yaml

from tracker_teleoperated.component_manager import ComponentManager, LOCAL_COMPONENT_IDS
from tracker_teleoperated.component_runtime import Health


ROOT = Path(__file__).resolve().parents[1]


class FakeProcess:
    """模拟由管理器拥有的本机进程。"""

    def __init__(self, *_args):
        """创建存活进程。"""
        self.code = None
        self.stop_started = None

    def poll(self, **_kwargs):
        """返回当前退出码。"""
        return self.code

    def stop(self):
        """记录停止并立即退出。"""
        self.stop_started = time.monotonic()
        self.code = 0


class FakeClient:
    """提供确定性异步服务响应。"""

    def __init__(self, result=None):
        """创建可预置结果的 Future。"""
        self.requests = []
        self.future = Future()
        if result is not None:
            self.future.set_result(result)

    def service_is_ready(self):
        """模拟服务已发现。"""
        return True

    def call_async(self, request):
        """保存请求并返回 Future。"""
        self.requests.append(request)
        return self.future

    def remove_pending_request(self, future):
        """取消超时请求。"""
        future.cancel()


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """创建不接触硬件和真实子进程的管理器。"""
    import tracker_teleoperated.component_manager as module

    original = module.get_package_share_directory
    monkeypatch.setattr(
        module, "get_package_share_directory",
        lambda package: str(ROOT) if package == "tracker_teleoperated" else original(package))
    monkeypatch.setenv("ROS_HOME", str(tmp_path / "ros"))
    config = yaml.safe_load((ROOT / "config/component_manager.yaml").read_text())
    config["workspace_root"] = str(ROOT.parents[1])
    config_path = tmp_path / "manager.yaml"
    config_path.write_text(yaml.safe_dump(config))
    rclpy.init()
    node = ComponentManager(parameter_overrides=[
        Parameter("manager_config", value=str(config_path)),
        Parameter("config_file", value=str(ROOT / "config/tracker_teleoperated.yaml")),
        Parameter("autostart", value=False),
    ], process_factory=FakeProcess)
    monkeypatch.setattr(node, "_command", lambda key: ([key], {}))
    monkeypatch.setattr(node, "_external_present", lambda _component: False)
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()


def status(state, recording_id="", completed="", error=""):
    """构造远端录制状态。"""
    return RecordingStatus(
        state=state, recording_id=recording_id,
        last_completed=RecordingInfo(recording_id=completed), last_error=error)


def test_health_distinguishes_missing_invalid_and_stale():
    """数据健康必须区分尚无数据、无效消息和超时。"""
    now = [10.0]
    health = Health({"pose": 0.25}, clock=lambda: now[0])
    assert "尚无数据" in health.check()[1]
    health.receive("pose", False)
    assert "无效" in health.check()[1]
    health.receive("pose", True)
    now[0] += 0.3
    assert "超时" in health.check()[1]


def test_only_five_local_components_can_start(manager):
    """硬件和远端录制组件不会被本机入口接管。"""
    assert set(LOCAL_COMPONENT_IDS) == {
        "tracker", "umi_camera", "estimator", "wrist_decoder", "teleop"}
    for key in LOCAL_COMPONENT_IDS:
        assert manager._start(key)[0]
        assert manager.components[key].ownership == "local"
    for key in ("arm", "gripper", "wrist_encoder", "recorder"):
        accepted, message = manager._start(key)
        assert not accepted
        assert "只监测" in message
        assert manager.components[key].process is None


def test_local_external_discovery_ignores_stale_publisher_endpoint(manager):
    """快速重启遗留的发布端没有新数据时，不阻止本机组件重新启动。"""
    component = manager.components["wrist_decoder"]
    topic = manager.probes["wrist_decoder"][0]
    manager._external_present = ComponentManager._external_present.__get__(manager)
    manager.count_publishers = lambda candidate: int(candidate == topic)
    assert not manager._external_present(component)
    component.health.receive(topic, True)
    assert manager._external_present(component)


def test_camera_switch_rechecks_external_teleop_before_manager_tick(manager, monkeypatch):
    """外部遥操在归属刷新前出现时，相机切换仍必须被拒绝。"""
    manager.recording_enabled = False
    teleop = manager.components["teleop"]
    assert teleop.ownership == "none"
    monkeypatch.setattr(
        manager, "_external_present", lambda component: component is teleop)

    assert "外部遥操" in manager.video_sources.reason()


def test_shutdown_stops_remote_recording_and_waits_for_matching_completion(manager):
    """退出先暂停，再按 UUID 停止远端录制并等待保存完成。"""
    manager._start("teleop")
    manager.pause_client = FakeClient(SetBool.Response(success=True))
    manager.recording_status_client = FakeClient(
        GetRecordingStatus.Response(status=status("recording", "recording-1")))
    manager.recording_stop_client = FakeClient(
        StopRecording.Response(success=True, code="STOP_ACCEPTED", recording_id="recording-1"))
    manager.request_shutdown()
    manager._advance_stop(time.monotonic())
    manager._advance_stop(time.monotonic())
    manager._advance_stop(time.monotonic())
    assert manager.operation["phase"] == "record_wait"
    assert manager.recording_stop_client.requests[0].recording_id == "recording-1"
    assert manager.components["teleop"].process.stop_started is None
    manager._recording_status_callback(status("idle", completed="recording-1"))
    manager._advance_stop(time.monotonic())
    assert manager.components["teleop"].process.stop_started is not None
    manager._tick()
    assert manager.shutdown_complete


def test_remote_save_error_is_reported_then_local_session_exits(manager):
    """远端保存失败仍回收本机节点，且诊断不报告成功。"""
    manager._start("wrist_decoder")
    manager.pause_client = FakeClient(SetBool.Response(success=True))
    manager.recording_status_client = FakeClient(
        GetRecordingStatus.Response(status=status("error", error="磁盘写满")))
    manager.request_shutdown()
    manager._advance_stop(time.monotonic())
    manager._advance_stop(time.monotonic())
    manager._tick()
    assert manager.shutdown_complete
    assert "磁盘写满" in manager.last_operation


def test_unavailable_recording_service_uses_request_timeout(manager):
    """录制服务未启动时三秒结束查询，不占用完整保存超时。"""
    manager.request_shutdown()
    manager._advance_stop(time.monotonic())
    assert manager.operation["phase"] == "record_query"
    query_deadline = manager.operation["deadline"]
    save_deadline = manager.operation["save_deadline"]
    assert save_deadline - query_deadline > 290
    manager._advance_stop(query_deadline + 0.1)
    assert manager.shutdown_complete
    assert "录制状态服务不可用" in manager.last_operation
