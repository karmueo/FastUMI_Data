"""验证一键启动的冲突策略、就绪判断和退出处理。"""

import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import EmitEvent, ExecuteProcess
from launch.events import Shutdown
from launch.events.process import ProcessExited
from rclpy.node import Node
from rclpy.qos import QoSReliabilityPolicy

from tracker_teleoperated.bringup_checks import (
    FRESHNESS_SECONDS,
    INITIAL_JOINT_POSITIONS,
    INPUT_TOPICS,
    ReadinessNode,
    TELEOP_TOPICS,
    conflict_topics,
    missing_inputs,
)


# 对应被测的一键启动文件。
LAUNCH_FILE = (
    Path(__file__).resolve().parents[1] / "launch/full_teleop.launch.py"
)


def _load_launch():
    """按路径加载 launch 文件供纯逻辑测试使用。"""
    spec = importlib.util.spec_from_file_location(
        "full_teleop_launch", LAUNCH_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conflicts_abort_all_managed_topics():
    """本机模式汇总设备和遥操节点的重复发布者。"""
    counts = {topic: 1 for topic in INPUT_TOPICS.values()}
    counts[TELEOP_TOPICS[0]] = 1
    assert conflict_topics(counts, start_arm=True) == [
        *INPUT_TOPICS.values(), TELEOP_TOPICS[0]
    ]


def test_remote_arm_allows_existing_joint_feedback():
    """外部机械臂模式允许关节反馈，但仍拒绝其他重复设备。"""
    counts = {
        INPUT_TOPICS["机械臂"]: 1,
        INPUT_TOPICS["USB 相机"]: 1,
    }
    assert conflict_topics(counts, start_arm=False) == [
        INPUT_TOPICS["USB 相机"]
    ]


def test_readiness_requires_fresh_data_and_initial_pose():
    """五路数据和初始姿态都满足后才放行遥操。"""
    now = 100.0
    recent = {name: now - 0.5 for name in INPUT_TOPICS}
    assert missing_inputs(recent, now, True, INITIAL_JOINT_POSITIONS) == []
    assert missing_inputs(recent, now, False, None) == []
    stale = dict(recent, **{"USB 相机": now - FRESHNESS_SECONDS - 0.1})
    assert missing_inputs(stale, now, True, INITIAL_JOINT_POSITIONS) == [
        "USB 相机"
    ]
    off_target = tuple(value + 0.02 for value in INITIAL_JOINT_POSITIONS)
    assert missing_inputs(recent, now, True, off_target) == ["机械臂初始姿态"]


def test_camera_readiness_uses_compatible_sensor_qos(monkeypatch):
    """相机 BEST_EFFORT 发布者可被就绪检查器接收。"""
    subscriptions = []
    monkeypatch.setattr(Node, "__init__", lambda self, name: None)
    monkeypatch.setattr(
        Node, "create_subscription",
        lambda self, message_type, topic, callback, qos: subscriptions.append(
            (topic, qos)
        ),
    )
    ReadinessNode(require_initial_pose=False)
    camera_qos = next(
        qos for topic, qos in subscriptions
        if topic == INPUT_TOPICS["USB 相机"]
    )
    assert camera_qos.reliability == QoSReliabilityPolicy.BEST_EFFORT


def test_readiness_failure_requests_shutdown():
    """就绪失败不会创建遥操进程，并向 launch 发送 Shutdown。"""
    module = _load_launch()
    teleop = ExecuteProcess(cmd=["true"])
    assert module._readiness_exit_actions(0, teleop) == [teleop]
    failed = module._readiness_exit_actions(1, teleop)
    assert len(failed) == 1
    assert isinstance(failed[0], EmitEvent)
    assert isinstance(failed[0].event, Shutdown)


def test_managed_component_exit_requests_shutdown():
    """受管理设备进程退出时，事件处理器结束整个 launch。"""
    module = _load_launch()
    process = ExecuteProcess(cmd=["true"])
    handler = module._shutdown_on_exit(process, "设备").event_handler
    event = ProcessExited(
        action=process, returncode=1, name="设备", cmd=["true"],
        cwd=None, env=None, pid=123,
    )
    assert handler.matches(event)
    actions = handler.handle(event, LaunchContext())
    assert len(actions) == 1
    assert isinstance(actions[0].event, Shutdown)


def test_remote_arm_launch_skips_local_driver(tmp_path, monkeypatch):
    """外部机械臂模式只启动四个本地前置组件和就绪检查。"""
    module = _load_launch()
    for relative in (
        ".venv-numpy1/bin/activate",
        "install/setup.bash",
        "src/unitree_gripper/run_gripper.sh",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    checked = []
    monkeypatch.setattr(
        module, "check_for_existing_publishers", checked.append
    )
    context = LaunchContext()
    context.launch_configurations.update({
        "workspace_root": str(tmp_path),
        "start_arm": "false",
        "move_to_initial_pose": "true",
        "startup_timeout_s": "180",
        "config_file": "/tmp/teleop.yaml",
        "use_keyboard": "false",
        "keyboard_prefix": "xterm -e",
    })
    actions = module._start_components(context)
    processes = [
        action for action in actions if isinstance(action, ExecuteProcess)
    ]
    commands = [
        " ".join(
            "".join(part.perform(context) for part in word)
            for word in process.cmd
        )
        for process in processes
    ]
    assert checked == [False]
    assert len(processes) == 5
    assert not any("rm_bringup" in command for command in commands)
    assert any("tracker_teleop_wait_ready" in command for command in commands)
