"""无硬件验证 RM75 急停、回位与关节运动服务的状态逻辑。"""

import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from fastumi_interfaces.srv import MoveJoints

from fastumi_bringup.arm_control_service import ArmControlService
from fastumi_bringup.kinematics import ArmKinematics
from fastumi_bringup.initial_pose import INITIAL_JOINT_POSITIONS


URDF = Path(__file__).parents[2] / "fastumi_rm75/assets/rm_75_kinematic.urdf"


@pytest.fixture
def service(monkeypatch):
    """仅构造服务逻辑状态，避免创建 ROS 网络参与者或真实发布器。"""
    node = ArmControlService.__new__(ArmControlService)
    node.lock = threading.Lock()
    node.current_positions = None
    node.last_feedback_at = None
    node.last_valid_feedback_at = None
    node.motion_call = None
    node.stop_waiters = []
    node.last_stop_success = True
    node.kinematics = ArmKinematics.from_urdf(URDF)
    node.home_speed = 20
    node.home_timeout = 2.0
    node.stop_timeout = 0.2
    node.movej_publisher = Mock()
    node.movej_publisher.get_subscription_count.return_value = 1
    node.stop_publisher = Mock()
    node.stop_publisher.get_subscription_count.return_value = 1
    monkeypatch.setattr(ArmControlService, "get_logger", lambda self: Mock())
    return node


def provide_feedback(node, positions=None):
    """发送按关节名称排列的当前弧度位置和有效反馈。"""
    node._joint_state_callback(JointState(
        name=[f"joint{index}" for index in range(1, 8)],
        position=positions if positions is not None else [1.0] * 7))
    node._feedback_valid_callback(Bool(data=True))


def call(handler):
    return handler(Trigger.Request(), Trigger.Response())


def move(service, degrees, speed=30):
    return service._move_joints(
        MoveJoints.Request(joint_degrees=degrees, speed=speed), MoveJoints.Response())


def call_in_thread(handler):
    """在线程中调用阻塞服务，返回线程和结果容器。"""
    holder = SimpleNamespace(response=None)
    thread = threading.Thread(target=lambda: setattr(holder, "response", handler()))
    thread.start()
    return thread, holder


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "等待条件超时"
        time.sleep(0.005)


def test_home_rejected_without_fresh_feedback(service):
    response = call(service._return_home)
    assert not response.success
    assert "joint_states" in response.message
    provide_feedback(service)
    service.last_valid_feedback_at = None
    assert "UDP" in call(service._return_home).message
    provide_feedback(service)
    service.last_feedback_at -= 1.0
    assert not call(service._return_home).success
    service.movej_publisher.publish.assert_not_called()


def test_home_rejected_without_driver(service):
    provide_feedback(service)
    service.movej_publisher.get_subscription_count.return_value = 0
    response = call(service._return_home)
    assert not response.success
    assert "movej_cmd" in response.message


def test_home_already_at_target_skips_motion(service):
    provide_feedback(service, list(INITIAL_JOINT_POSITIONS))
    response = call(service._return_home)
    assert response.success
    service.movej_publisher.publish.assert_not_called()


def test_home_succeeds_after_movej_result(service):
    provide_feedback(service)
    thread, holder = call_in_thread(lambda: call(service._return_home))
    wait_until(lambda: service.movej_publisher.publish.called)
    command = service.movej_publisher.publish.call_args.args[0]
    assert list(command.joint) == pytest.approx(INITIAL_JOINT_POSITIONS)
    assert command.speed == 20 and command.block and command.dof == 7

    duplicate = call(service._return_home)
    assert not duplicate.success and "尚未结束" in duplicate.message

    service._movej_result_callback(Bool(data=True))
    thread.join(1.0)
    assert holder.response.success
    assert service.motion_call is None


def test_movej_failure_reported(service):
    provide_feedback(service)
    thread, holder = call_in_thread(lambda: call(service._return_home))
    wait_until(lambda: service.movej_publisher.publish.called)
    service._movej_result_callback(Bool(data=False))
    thread.join(1.0)
    assert not holder.response.success


def test_unsolicited_movej_result_ignored(service):
    service._movej_result_callback(Bool(data=True))
    assert service.motion_call is None


def test_emergency_stop_interrupts_home(service):
    provide_feedback(service)
    home_thread, home = call_in_thread(lambda: call(service._return_home))
    wait_until(lambda: service.movej_publisher.publish.called)

    stop_thread, stop = call_in_thread(lambda: call(service._emergency_stop))
    wait_until(lambda: service.stop_waiters)
    service._stop_result_callback(Bool(data=True))
    stop_thread.join(1.0)
    home_thread.join(1.0)

    service.stop_publisher.publish.assert_called_once()
    assert stop.response.success
    assert not home.response.success
    assert "急停" in home.response.message
    assert service.motion_call is not None
    # 迟到的 MoveJ 结果不能把已中断的回位改成成功。
    service._movej_result_callback(Bool(data=True))
    assert service.motion_call is None


def test_emergency_stop_reports_driver_failure(service):
    thread, holder = call_in_thread(lambda: call(service._emergency_stop))
    wait_until(lambda: service.stop_waiters)
    service._stop_result_callback(Bool(data=False))
    thread.join(1.0)
    assert not holder.response.success
    assert "失败" in holder.response.message
    provide_feedback(service)
    assert "停止" in call(service._return_home).message
    service.movej_publisher.publish.assert_not_called()


@pytest.mark.parametrize("listening, text", [(1, "未确认"), (0, "未订阅")])
def test_emergency_stop_timeout_still_publishes(service, listening, text):
    service.stop_publisher.get_subscription_count.return_value = listening
    response = call(service._emergency_stop)
    service.stop_publisher.publish.assert_called_once()
    assert not response.success
    assert text in response.message
    assert len(service.stop_waiters) == 1
    provide_feedback(service)
    assert "停止" in call(service._return_home).message
    service._stop_result_callback(Bool(data=True))
    assert service.stop_waiters == []


def test_home_timeout_sends_stop(service):
    service.home_timeout = 0.05
    provide_feedback(service)
    response = call(service._return_home)
    assert not response.success
    assert "超时" in response.message
    service.stop_publisher.publish.assert_called_once()
    assert service.motion_call is not None
    assert len(service.stop_waiters) == 1


@pytest.mark.parametrize("interruption", ["stop", "timeout"])
@pytest.mark.parametrize("stop_first", [True, False])
@pytest.mark.parametrize("late_result", [True, False])
def test_interrupted_motion_blocks_retry_until_both_results(
        service, interruption, stop_first, late_result):
    """停止确认和旧 MoveJ 结果任意先后到达，都不能串给下一条运动。"""
    service.home_timeout = 0.03 if interruption == "timeout" else 2.0
    provide_feedback(service)
    motion_thread, old = call_in_thread(lambda: call(service._return_home))
    wait_until(lambda: service.movej_publisher.publish.call_count == 1)
    if interruption == "stop":
        stop_thread, stopped = call_in_thread(lambda: call(service._emergency_stop))
        wait_until(lambda: service.stop_waiters)
    motion_thread.join(1.0)
    assert old.response is not None and not old.response.success

    def retry_is_rejected():
        provide_feedback(service)
        assert not call(service._return_home).success
        assert not move(service, TARGET_DEGREES).success
        assert service.movej_publisher.publish.call_count == 1

    retry_is_rejected()
    callbacks = [lambda: service._stop_result_callback(Bool(data=True)),
                 lambda: service._movej_result_callback(Bool(data=late_result))]
    if not stop_first:
        callbacks.reverse()
    callbacks[0]()
    retry_is_rejected()
    callbacks[1]()
    if interruption == "stop":
        stop_thread.join(1.0)
        assert stopped.response.success
    assert service.motion_call is None

    service.home_timeout = 2.0
    provide_feedback(service)
    next_thread, next_call = call_in_thread(lambda: move(service, TARGET_DEGREES))
    wait_until(lambda: service.movej_publisher.publish.call_count == 2)
    assert next_call.response is None and not service.motion_call.done.is_set()
    service._movej_result_callback(Bool(data=True))
    next_thread.join(1.0)
    assert next_call.response.success
    assert not old.response.success


def test_failed_stop_requires_a_successful_retry(service):
    """停止明确失败后，只有再次停止成功才允许新运动。"""
    provide_feedback(service, list(INITIAL_JOINT_POSITIONS))
    first_thread, first = call_in_thread(lambda: call(service._emergency_stop))
    wait_until(lambda: service.stop_waiters)
    service._stop_result_callback(Bool(data=False))
    first_thread.join(1.0)
    assert not first.response.success
    assert not call(service._return_home).success
    retry_thread, retry = call_in_thread(lambda: call(service._emergency_stop))
    wait_until(lambda: service.stop_waiters)
    service._stop_result_callback(Bool(data=True))
    retry_thread.join(1.0)
    assert retry.response.success
    assert call(service._return_home).success


def test_each_stop_command_requires_its_own_result(service):
    """连续急停的第一个结果不能提前解锁仍在等待确认的第二次停止。"""
    provide_feedback(service, list(INITIAL_JOINT_POSITIONS))
    first_thread, first = call_in_thread(lambda: call(service._emergency_stop))
    wait_until(lambda: len(service.stop_waiters) == 1)
    second_thread, second = call_in_thread(lambda: call(service._emergency_stop))
    wait_until(lambda: len(service.stop_waiters) == 2)
    service._stop_result_callback(Bool(data=True))
    first_thread.join(1.0)
    assert first.response.success and second.response is None
    assert not call(service._return_home).success
    service._stop_result_callback(Bool(data=True))
    second_thread.join(1.0)
    assert second.response.success
    assert call(service._return_home).success


TARGET_DEGREES = [10.0, 30.0, -5.0, 60.0, 0.0, 80.0, 45.0]


def test_move_joints_converts_degrees_and_speed(service):
    provide_feedback(service)
    thread, holder = call_in_thread(lambda: move(service, TARGET_DEGREES, speed=35))
    wait_until(lambda: service.movej_publisher.publish.called)
    command = service.movej_publisher.publish.call_args.args[0]
    assert list(command.joint) == pytest.approx([math.radians(v) for v in TARGET_DEGREES])
    assert command.speed == 35 and command.block and command.dof == 7

    assert not call(service._return_home).success
    assert "尚未结束" in move(service, TARGET_DEGREES).message

    service._movej_result_callback(Bool(data=True))
    thread.join(1.0)
    assert holder.response.success
    assert "目标关节角" in holder.response.message
    assert service.motion_call is None


@pytest.mark.parametrize("degrees, speed, text", [
    (TARGET_DEGREES[:6], 30, "7 个值"),
    (TARGET_DEGREES[:6] + [float("nan")], 30, "NaN"),
    (TARGET_DEGREES, 0, "speed"),
    (TARGET_DEGREES, 101, "speed"),
    ([0.0, 140.0, 0.0, 0.0, 0.0, 0.0, 0.0], 30, "joint2"),
    ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -400.0], 30, "joint7"),
])
def test_move_joints_rejects_invalid_request(service, degrees, speed, text):
    provide_feedback(service)
    response = move(service, degrees, speed)
    assert not response.success
    assert text in response.message
    service.movej_publisher.publish.assert_not_called()
    assert service.motion_call is None


def test_move_joints_already_at_target(service):
    provide_feedback(service, [math.radians(v) for v in TARGET_DEGREES])
    response = move(service, TARGET_DEGREES)
    assert response.success
    service.movej_publisher.publish.assert_not_called()


def test_move_joints_requires_ready_arm(service):
    response = move(service, TARGET_DEGREES)
    assert not response.success and "未就绪" in response.message
    service.movej_publisher.publish.assert_not_called()


def test_emergency_stop_interrupts_move_joints(service):
    provide_feedback(service)
    move_thread, moving = call_in_thread(lambda: move(service, TARGET_DEGREES))
    wait_until(lambda: service.movej_publisher.publish.called)
    stop_thread, stop = call_in_thread(lambda: call(service._emergency_stop))
    wait_until(lambda: service.stop_waiters)
    service._stop_result_callback(Bool(data=True))
    stop_thread.join(1.0)
    move_thread.join(1.0)
    assert stop.response.success
    assert not moving.response.success and "急停" in moving.response.message
