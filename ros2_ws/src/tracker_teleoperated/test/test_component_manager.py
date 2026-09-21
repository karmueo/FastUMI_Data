"""验证组件健康、外部归属、异步收尾以及真实进程树回收。"""

from concurrent.futures import Future
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import rclpy
from rclpy.parameter import Parameter
from std_srvs.srv import SetBool, Trigger
import yaml

from tracker_teleoperated.component_runtime import Health, ManagedProcess, process_table
from tracker_teleoperated.component_manager import ComponentManager

# 源码配置用于创建临时测试会话。
ROOT = Path(__file__).resolve().parents[1]


def test_health():
    """持续收到静止位姿仍健康，区分无数据、无效和超时。"""
    now = [10.0]
    health = Health({"pose": 0.25}, clock=lambda: now[0])
    assert "尚无数据" in health.check()[1]
    health.receive("pose", False)
    assert "无效" in health.check()[1]
    health.receive("pose")
    assert health.check()[0]
    now[0] += 0.3
    assert "超时" in health.check()[1]
    health.receive("pose")
    assert health.check()[0]


class FakeProcess:
    """模拟进程，测试不会启动硬件。"""

    def __init__(self, *args):
        """初始化存活状态。"""
        self.code = None
        self.stop_started = None

    def poll(self, **kwargs):
        """返回测试设定的退出码。"""
        return self.code

    def stop(self):
        """模拟优雅退出。"""
        self.stop_started = time.monotonic()
        self.code = 0


class FakeClient:
    """提供可控的异步服务结果。"""

    def __init__(self, result=None):
        """创建 Future 并保存请求列表。"""
        self.requests = []
        self.future = Future()
        if result is not None:
            self.future.set_result(result)

    def remove_pending_request(self, future):
        """模拟释放超时请求。"""
        future.cancel()

    def service_is_ready(self):
        """模拟服务已经发现。"""
        return True

    def call_async(self, request):
        """记录请求并返回 Future。"""
        self.requests.append(request)
        return self.future


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """创建只使用假进程的管理器，并隔离锁和日志。"""
    import tracker_teleoperated.component_manager as module
    original = module.get_package_share_directory
    monkeypatch.setattr(module, "get_package_share_directory", lambda package: str(ROOT) if package == "tracker_teleoperated" else original(package))
    monkeypatch.setenv("ROS_HOME", str(tmp_path / "ros"))
    config = yaml.safe_load((ROOT / "config/component_manager.yaml").read_text())
    config["workspace_root"] = str(ROOT.parents[1])
    path = tmp_path / "manager.yaml"
    path.write_text(yaml.safe_dump(config))
    rclpy.init()
    node = ComponentManager(parameter_overrides=[
        Parameter("manager_config", value=str(path)),
        Parameter("config_file", value=str(ROOT / "config/tracker_teleoperated.yaml")),
        Parameter("autostart", value=False),
    ], process_factory=FakeProcess)
    monkeypatch.setattr(node, "_command", lambda key: ([key], {}))
    monkeypatch.setattr(node, "_external_present", lambda component: False)
    try:
        yield node
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_idempotent_start_and_external_read_only(manager, monkeypatch):
    """重复启动不创建新进程，外部节点拒绝停止。"""
    assert manager._start("arm")[0]
    process = manager.components["arm"].process
    assert manager._start("arm")[0]
    assert manager.components["arm"].process is process
    monkeypatch.setattr(manager, "_external_present", lambda component: component.key == "tracker")
    assert manager._start("tracker")[0]
    assert manager.components["tracker"].ownership == "external"
    assert manager.components["tracker"].process is None
    assert not manager._set_running("tracker", SetBool.Request(data=False), SetBool.Response()).success


def test_stop_waits_for_save(manager):
    """保存之前不关闭依赖，并拒绝并发启动请求。"""
    for key in ("arm", "teleop", "recorder"):
        manager._start(key)
    manager.pause_client = FakeClient(SetBool.Response(success=True))
    manager.save_client = FakeClient()
    manager._begin_stop(["arm"])
    manager._advance_stop(time.monotonic())
    manager._advance_stop(time.monotonic())
    assert manager.components["arm"].process.stop_started is None
    assert manager.operation["phase"] == "save"
    assert manager.pause_client.requests[0].data is False
    assert not manager._set_running("tracker", SetBool.Request(data=True), SetBool.Response()).success
    manager.save_client.future.set_result(Trigger.Response(success=True, message="idle"))
    manager._advance_stop(time.monotonic())
    assert manager.components["arm"].process.stop_started is not None
    assert manager.components["recorder"].process.stop_started is None


def test_save_failure_and_external_shutdown(manager):
    """写盘错误不报告成功，退出时保持外部组件运行。"""
    manager._start("recorder")
    manager.components["arm"].ownership = "external"
    manager.save_client = FakeClient(Trigger.Response(success=False, message="磁盘写满"))
    manager.request_shutdown()
    manager._advance_stop(time.monotonic())
    manager._tick()
    assert manager.shutdown_complete
    assert "磁盘写满" in manager.last_operation
    assert manager.components["arm"].ownership == "external"


def test_recorder_crash_pauses_control(manager):
    """记录节点退出后暂停遥操，同时保留诊断与重试入口。"""
    manager._start("teleop")
    manager._start("recorder")
    manager.pause_client = FakeClient(SetBool.Response(success=True))
    manager.components["recorder"].process.code = 7
    manager._tick()
    assert manager.components["recorder"].state == "exited"
    assert manager.pause_client.requests[0].data is False
    assert not manager.shutdown_complete
    assert manager._start("recorder")[0]


def test_start_failure(manager):
    """缺失环境只使对应组件报错，管理器继续运行。"""
    def fail(*args):
        """模拟启动失败。"""
        raise OSError("缺少解释器")
    manager.process_factory = fail
    assert not manager._start("umi_camera")[0]
    assert manager.components["umi_camera"].state == "failed"


@pytest.mark.parametrize("crash", [False, True])
def test_detached_process_cleanup(tmp_path, crash):
    """setsid 后代在父进程崩溃和正常退出后回收，不影响无关进程。"""
    marker = tmp_path / "child.pid"
    script = (
        "import subprocess,sys,time,pathlib\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(child.pid))\n"
        + ("raise RuntimeError('simulated crash')\n" if crash else "time.sleep(60)\n")
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    owned = ManagedProcess([sys.executable, "-c", script], dict(os.environ), tmp_path / "log", tmp_path)
    try:
        deadline = time.monotonic() + 12
        while not marker.exists() and time.monotonic() < deadline:
            owned.poll()
            time.sleep(0.02)
        assert marker.exists()
        child_pid = int(marker.read_text())
        if not crash:
            owned.stop()
        while owned.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert owned.poll() is not None
        assert child_pid not in process_table()
        assert unrelated.poll() is None
    finally:
        owned.signal_owned(9)
        owned.process.wait(timeout=5)
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_real_recorder_save_before_managed_shutdown(tmp_path, monkeypatch):
    """真实管理器启动记录子进程，关闭会话时自动写入 HDF5 并回收进程。"""
    import h5py
    from rm_ros_interfaces.msg import Jointpos
    from std_msgs.msg import String
    from rclpy.qos import QoSProfile, DurabilityPolicy

    monkeypatch.setenv("ROS_HOME", str(tmp_path / "ros"))
    configuration = yaml.safe_load((ROOT / "config/component_manager.yaml").read_text())
    configuration["workspace_root"] = str(ROOT.parents[1])
    configuration["save_timeout_s"] = 10.0
    for key, item in configuration["components"].items():
        item["mode"] = "auto" if key == "recorder" else "disabled"
    manager_file = tmp_path / "manager.yaml"
    manager_file.write_text(yaml.safe_dump(configuration))
    parameters = yaml.safe_load((ROOT / "config/tracker_teleoperated.yaml").read_text())
    parameters["tracker_teleop_recorder"]["ros__parameters"].update(
        dataset_root=str(tmp_path / "data"), dir_name="test", name="session", record_camera=False)
    config_file = tmp_path / "control.yaml"
    config_file.write_text(yaml.safe_dump(parameters))
    rclpy.init()
    node = ComponentManager(parameter_overrides=[
        Parameter("manager_config", value=str(manager_file)),
        Parameter("config_file", value=str(config_file)), Parameter("autostart", value=False),
    ])
    state = []
    node.create_subscription(String, "/tracker_teleoperated/record_state", lambda m: state.append(m.data),
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    command = node.create_publisher(String, "/tracker_teleoperated/record_command", 10)
    actions = node.create_publisher(Jointpos, parameters["tracker_teleop_recorder"]["ros__parameters"]["joint_action_topic"], 10)

    def until(predicate, seconds=10):
        """等待真实 DDS 回调与后台进程的异步结果。"""
        deadline = time.monotonic() + seconds
        while not predicate() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.03)
        return predicate()

    try:
        assert node._start("recorder")[0]
        assert until(lambda: command.get_subscription_count() > 0 and state)
        command.publish(String(data="a"))
        assert until(lambda: state[-1] == "recording")
        for _ in range(10):
            actions.publish(Jointpos(dof=7, joint=[0.1] * 7))
            rclpy.spin_once(node, timeout_sec=0.03)
        node.request_shutdown()
        assert until(lambda: node.shutdown_complete, 15)
        saved = tmp_path / "data/test/session/episode_0/proprio.hdf5"
        assert saved.is_file()
        with h5py.File(saved) as data:
            assert data["action/joint_action/position"].shape[0] >= 1
        assert node.components["recorder"].process is None
        assert node.last_operation == "组件已停止"
    finally:
        for component in node.components.values():
            if component.process:
                component.process.signal_owned(9)
                component.process.process.wait(timeout=5)
        node.destroy_node()
        rclpy.shutdown()


def test_save_timeout_is_reported_and_pending_request_released(manager):
    """保存服务无响应时有明确错误，不能无限积累在途请求。"""
    manager._start("recorder")
    manager.save_client = FakeClient()
    manager.request_shutdown()
    manager._advance_stop(time.monotonic())
    manager._advance_stop(manager.operation["deadline"] + 1)
    assert manager.save_client.future.cancelled()
    manager._tick()
    assert manager.shutdown_complete
    assert "保存超时" in manager.last_operation


def test_pause_timeout_terminates_owned_controller(manager):
    """暂停服务失效时回收本会话控制进程，避免继续发出控制目标。"""
    manager._start("teleop")
    manager._start("arm")
    manager.pause_client = FakeClient()
    manager._begin_stop(["arm"])
    manager._advance_stop(time.monotonic())
    manager._advance_stop(manager.operation["deadline"] + 1)
    assert manager.pause_client.future.cancelled()
    assert manager.components["teleop"].state == "stopping"
    assert manager.components["teleop"].process.stop_started is not None


@pytest.mark.parametrize("override", [
    {"components": {"arm": {"mode": "unknown"}}},
    {"components": {"arm": {"nodes": "rm_driver"}}},
    {"components": {"arm": {"health_timeout_s": float("nan")}}},
    {"components": {}, "save_timeout_s": 0},
])
def test_invalid_config_rejected_before_launch(tmp_path, override):
    """错误配置在启动任何组件前被拒绝。"""
    from tracker_teleoperated.component_runtime import read_configuration
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(override))
    with pytest.raises(ValueError):
        read_configuration(str(path))


def test_recording_directory_while_stopped(manager):
    """记录节点未启动时，诊断仍发布与记录节点一致的绝对任务目录。"""
    from tracker_teleoperated.recording_paths import output_directory

    class CapturePublisher:
        """捕获诊断结果，避免测试依赖 DDS 发现延迟。"""

        def publish(self, message):
            """保存本轮发布的诊断快照。"""
            self.message = message

    capture = CapturePublisher()
    manager.status_publisher = capture
    manager._publish_status(time.monotonic())
    status = next(item for item in capture.message.status if item.name == "recorder")
    values = {item.key: item.value for item in status.values}
    assert values["process_state"] == "stopped"
    assert values["output_directory"] == str(output_directory(
        manager.record["dataset_root"], manager.record["dir_name"], manager.record["name"],
    ))
    assert Path(values["output_directory"]).is_absolute()


@pytest.mark.parametrize("discovery", ["node", "invalid_message"])
def test_estimator_running_independent_of_prediction_validity(manager, monkeypatch, discovery):
    """预测节点已出现或已返回无效结果时显示运行中，数据健康仍独立报错。"""
    assert manager._start("estimator")[0]
    component = manager.components["estimator"]
    monkeypatch.setattr(manager, "get_node_names_and_namespaces", lambda: [])
    manager._tick()
    assert component.state == "starting"
    if discovery == "node":
        monkeypatch.setattr(manager, "get_node_names_and_namespaces", lambda: [
            ("gripper_openness_estimator", "/"),
        ])
        manager._next_graph = 0
    else:
        component.health.receive("/gripper/state", False)
    manager._tick()
    assert component.state == "running"
    assert not component.health.check()[0]
    component.health.receive("/gripper/state", True)
    manager._tick()
    assert component.state == "running"
    assert component.health.check()[0]
    component.health.receive("/gripper/state", False)
    manager._tick()
    assert component.state == "running"
    assert not component.health.check()[0]
    component.process.code = 2
    manager._tick()
    assert component.state == "exited"
