"""验证实机执行器的 dry-run、预热门控、单序列和停止行为。"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("fastumi_interfaces.msg")
pytest.importorskip("rm_ros_interfaces.msg")
import rclpy

from vr_umi_ros.core import ActionSequence, InferenceContext
from vr_umi_ros.node import sequence_message
from vr_umi_ros.real_executor import RealRobotExecutor


class Publisher:
    """记录消息并报告单个订阅者。"""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)

    def get_subscription_count(self):
        return 1


def policy_message(node, sequence_id):
    """构造起始姿态附近且仍含未来点的有效策略序列。"""
    now_ns = node.get_clock().now().nanoseconds
    positions = np.tile(np.array([0.30, 0.0, 0.30]), (16, 1))
    positions[:, 2] += np.linspace(0.0, 0.002, 16)
    sequence = ActionSequence(
        positions=positions,
        quaternions=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (16, 1)),
        gripper_openness=np.linspace(0.5, 0.55, 16),
        time_from_start=np.arange(16) / 30.0,
    )
    return sequence_message(
        sequence, InferenceContext(np.eye(4), now_ns - 100_000_000, 0, sequence_id))


@pytest.fixture
def executor():
    rclpy.init()
    from infer_real import DEFAULT_URDF

    node = RealRobotExecutor(DEFAULT_URDF, "digest", report_root=None)
    # 单元测试可能运行在连接真实驱动的 ROS_DOMAIN_ID 中，任何许可模式测试都
    # 必须先断开真实硬件 publisher。
    node.arm_publisher = Publisher()
    node.gripper_publisher = Publisher()
    node.stop_publisher = Publisher()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_dry_run_cannot_arm_or_publish(executor):
    """默认模式即使收到操作请求也不能进入硬件执行态。"""
    arm, gripper, stop = Publisher(), Publisher(), Publisher()
    executor.arm_publisher, executor.gripper_publisher = arm, gripper
    executor.stop_publisher = stop
    success, reason = executor.arm_policy()
    assert not success and "--enable-motion" in reason
    executor.tick()
    assert not arm.messages and not gripper.messages and not stop.messages


def test_next_sequence_executes_once_and_operator_stop_holds_gripper(executor):
    """实机许可后只接收按键后的新序列，并在停止时发送 RM stop 与夹爪保持。"""
    executor.enable_motion = True
    executor.self_test_completed = True
    executor.arm_position = np.array([0.30, 0.0, 0.30])
    executor.arm_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    executor.gripper = 0.5
    executor._inputs_ready_reason = lambda *args, **kwargs: None
    arm, gripper, stop = Publisher(), Publisher(), Publisher()
    executor.arm_publisher, executor.gripper_publisher = arm, gripper
    executor.stop_publisher = stop

    for sequence_id in (1, 2):
        message = policy_message(executor, sequence_id)
        executor.record_inference(message, 0.3)
        executor.on_sequence(message)
    success, _ = executor.arm_policy()
    assert success and executor.phase == "armed"
    old = policy_message(executor, 2)
    executor.on_sequence(old)
    assert executor.phase == "armed"
    message = policy_message(executor, 3)
    executor.record_inference(message, 0.3)
    executor.on_sequence(message)
    assert executor.phase == "armed" and executor.plan is not None
    executor.tick()
    assert executor.phase == "executing" and not arm.messages
    executor.tick()
    assert len(arm.messages) == len(gripper.messages) == 1

    executor.emergency_stop("test stop")
    assert executor.phase == "locked" and len(stop.messages) == 1
    assert gripper.messages[-1].data == pytest.approx(0.5)


def test_workspace_self_test_and_feedback_watchdog(executor):
    """工作区越界被拒绝，自检幅度固定，反馈过期会停止并保持夹爪。"""
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    start = np.array([0.30, 0.0, 0.30])
    with pytest.raises(ValueError, match="outside workspace"):
        executor._validate_targets(
            np.array([[1.10, 0.0, 0.30]]), identity[None], start, identity)

    executor.enable_motion = True
    executor.arm_position, executor.arm_quaternion = start, identity
    executor.gripper = 0.5
    executor._inputs_ready_reason = lambda *args, **kwargs: None
    success, _ = executor.start_self_test()
    assert success
    np.testing.assert_allclose(executor.plan.positions[0] - start, [0, 0, 0.001])
    np.testing.assert_allclose(executor.plan.grippers, [0.55, 0.5])

    arm, gripper, stop = Publisher(), Publisher(), Publisher()
    executor.arm_publisher, executor.gripper_publisher = arm, gripper
    executor.stop_publisher = stop
    del executor._inputs_ready_reason
    executor.joint_received = executor.arm_received = 0.0
    executor.tick()
    assert executor.phase == "locked"
    assert len(stop.messages) == 1
    assert gripper.messages[-1].data == pytest.approx(0.5)


def test_duplicate_command_publisher_refuses_unlock(executor, monkeypatch):
    """所有其他预热条件满足时，重复 Cartesian 发布者仍阻止解锁。"""
    import time

    now = time.monotonic()
    executor.enable_motion = True
    executor.self_test_completed = True
    executor.joint_received = executor.arm_received = now
    executor.gripper_received = executor.error_received = now
    executor.robot_error_clear = True
    executor.fk_match_count = 5
    executor.valid_predictions = [(now, 0.3), (now, 0.3)]
    executor.arm_publisher = Publisher()
    executor.gripper_publisher = Publisher()
    monkeypatch.setattr(
        executor, "count_publishers",
        lambda topic: 2 if topic == "/rm_driver/movep_canfd_custom_cmd" else 1)
    success, reason = executor.arm_policy()
    assert not success and "multiple RM75" in reason


def test_keyboard_request_waits_for_fresh_feedback(executor):
    """推理期间按下 t 只排队，直到 ROS 线程看到新鲜反馈才开始运动。"""
    executor.enable_motion = True
    executor.arm_position = np.array([0.30, 0.0, 0.30])
    executor.arm_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    executor.gripper = 0.5
    state = {"ready": False, "policy_fresh": True}

    def readiness(require_policy=True):
        if not state["ready"]:
            return "joint feedback stale"
        if require_policy and not state["policy_fresh"]:
            return "latest policy prediction is stale"
        return None

    executor._inputs_ready_reason = readiness

    success, message = executor.queue_operator_command("t")
    assert success and "queued" in message
    executor.tick()
    assert executor.phase == "locked" and executor.pending_operator_command == "t"

    state["ready"] = True
    executor.tick()
    assert executor.phase == "executing" and executor.pending_operator_command is None
    state["policy_fresh"] = False
    executor.tick()
    assert executor.phase == "executing"
    executor.plan.times_ns[:] = executor.get_clock().now().nanoseconds - 1
    executor.tick()
    assert executor.phase == "locked" and executor.self_test_completed


def test_policy_requires_completed_self_test(executor):
    """分级验收顺序由状态机强制执行，不能跳过 t 直接按 a。"""
    executor.enable_motion = True
    success, message = executor.queue_operator_command("a")
    assert not success and "self-test" in message


def test_armed_policy_waits_for_feedback_refresh(executor):
    """armed 阶段的同步推理假过期只等待；截止时间后仍无反馈才锁定。"""
    import time

    executor.enable_motion = True
    executor.phase = "armed"
    executor.armed_deadline = time.monotonic() + 1.0
    executor._inputs_ready_reason = lambda *args, **kwargs: "joint feedback missing or stale"
    executor.tick()
    assert executor.phase == "armed"
    assert not executor.stop_publisher.messages

    executor.armed_deadline = time.monotonic() - 1.0
    executor.tick()
    assert executor.phase == "locked"
    assert len(executor.stop_publisher.messages) == 1
