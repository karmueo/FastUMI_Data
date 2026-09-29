"""无硬件检查任务门控、回位及停止抢占。"""

import asyncio
import time
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np

from fastumi_interfaces.srv import ResetPolicyController
from fastumi_rm75 import rm75_placo_controller
from fastumi_rm75.rm75_placo_controller import Rm75PlacoController
from rclpy.task import Future
from std_msgs.msg import Bool


HOME = np.deg2rad([0, 20, 0, 70, 0, 90, 90])


def controller(joints=None):
    """用内存发布器构造控制器状态，不连接 ROS 图。"""
    now = time.monotonic()
    node = SimpleNamespace(
        _task_control_enabled=True, _task_enabled=False, _enabled_episode=None,
        _dry_run=False, _minimum_episode=1, _latest_episode=None,
        _latest_sequence=None, _faulted_episode=None, _episode_started_at=None,
        _home_since=now - 1, _start_verified_episode=None,
        _feedback_positions=np.asarray(joints if joints is not None else HOME),
        _feedback_monotonic=now, _feedback_timeout_s=0.25,
        _start_joints=HOME, _start_joint_tolerance=0.035,
        _start_settle_s=0.5, _start_gripper_min=0.95,
        _gripper_state=1.0, _gripper_monotonic=now, _gripper_timeout_s=0.25,
        _valid_feedback=True, _valid_feedback_at=now,
        _home_future=None, _home_episode=None, _home_started_at=None,
        _home_command_sent_at=None, _home_accepted=False,
        _home_settled_since=None, _stop_future=None, _close_allowed=False,
        _task_io_group=object(),
        _stop_control=Mock(), _movej_publisher=Mock(),
        _gripper_publisher=Mock(), _stop_publisher=Mock(),
        get_logger=lambda: Mock(),
    )
    node._feedback_is_fresh = MethodType(Rm75PlacoController._feedback_is_fresh, node)
    node._hold_gripper = MethodType(Rm75PlacoController._hold_gripper, node)
    node._finish_home = MethodType(Rm75PlacoController._finish_home, node)
    node._on_reset = MethodType(Rm75PlacoController._on_reset, node)
    node._start_state_ready = Mock(return_value=True)
    node._movej_publisher.get_subscription_count.return_value = 1
    node._gripper_publisher.get_subscription_count.return_value = 1
    return node


def request(episode):
    value = ResetPolicyController.Request()
    value.episode_id = episode
    return value


def test_start_requires_fresh_home_and_rejects_old_policy():
    node = controller()
    node._feedback_monotonic -= 1
    rejected = Rm75PlacoController._on_start_task(
        node, request(2), ResetPolicyController.Response())
    assert not rejected.success and not node._task_enabled
    node._feedback_monotonic = time.monotonic()
    accepted = Rm75PlacoController._on_start_task(
        node, request(2), ResetPolicyController.Response())
    assert accepted.success and node._task_enabled and node._enabled_episode == 2
    Rm75PlacoController._on_policy(node, SimpleNamespace(episode_id=1, sequence_id=99))
    node._stop_control.assert_not_called()


def test_return_opens_gripper_even_when_arm_is_home():
    node = controller()
    node._home_future = Future()
    node._home_started_at = time.monotonic()
    Rm75PlacoController._home_tick(node)
    node._gripper_publisher.publish.assert_called_once()
    assert node._gripper_publisher.publish.call_args.args[0].data == 1.0
    node._movej_publisher.publish.assert_not_called()
    node._home_settled_since = time.monotonic() - 0.6
    Rm75PlacoController._home_tick(node)
    assert node._home_future is None


def test_return_movej_and_stop_interrupt():
    node = controller(HOME + 0.1)
    node._home_future = Future()
    pending = node._home_future
    node._home_started_at = time.monotonic()
    Rm75PlacoController._home_tick(node)
    command = node._movej_publisher.publish.call_args.args[0]
    assert command.speed == 20 and command.block and command.dof == 7
    stop = Rm75PlacoController._on_reset(
        node, request(2), ResetPolicyController.Response())
    assert stop.success and pending.done() and pending.result()[0] is False
    assert not node._task_enabled and node._minimum_episode == 2
    node._stop_publisher.publish.assert_called_once()
    Rm75PlacoController._on_movej_result(node, Bool(data=True))
    assert node._home_future is None


def test_return_requires_result_and_fresh_settled_feedback():
    node = controller(HOME + 0.1)
    node._home_future = Future()
    pending = node._home_future
    node._home_started_at = time.monotonic()
    Rm75PlacoController._home_tick(node)
    Rm75PlacoController._on_movej_result(node, Bool(data=True))
    node._feedback_positions = HOME.copy()
    node._gripper_state = 0.8
    Rm75PlacoController._home_tick(node)
    assert not pending.done()
    node._gripper_state = 1.0
    Rm75PlacoController._home_tick(node)
    node._home_settled_since = time.monotonic() - 0.6
    Rm75PlacoController._home_tick(node)
    assert pending.done() and pending.result()[0] is True


def test_return_feedback_loss_and_timeout_stop_motion():
    for timeout in (False, True):
        node = controller(HOME + 0.1)
        node._home_future = Future()
        pending = node._home_future
        node._home_started_at = time.monotonic()
        Rm75PlacoController._home_tick(node)
        if timeout:
            node._home_command_sent_at = time.monotonic() - 121
        else:
            node._valid_feedback = False
        Rm75PlacoController._home_tick(node)
        assert pending.done() and pending.result()[0] is False
        node._stop_publisher.publish.assert_called_once()


def test_stop_response_waits_for_driver_ack(monkeypatch):
    monkeypatch.setattr(rm75_placo_controller, "Future", asyncio.Future)
    async def exercise():
        node = controller()
        node.create_timer = Mock(return_value=object())
        node.destroy_timer = Mock()
        pending = asyncio.create_task(Rm75PlacoController._on_reset_service(
            node, request(2), ResetPolicyController.Response()))
        await asyncio.sleep(0)
        assert not pending.done()
        Rm75PlacoController._on_move_stop_result(node, Bool(data=True))
        response = await pending
        assert response.success
        node._stop_publisher.publish.assert_called_once()

    asyncio.run(exercise())


def test_return_rejects_dry_run():
    node = controller()
    node._dry_run = True
    result = asyncio.run(Rm75PlacoController._on_return_to_start(
        node, request(1), ResetPolicyController.Response()))
    assert not result.success
    node._movej_publisher.publish.assert_not_called()
    node._gripper_publisher.publish.assert_not_called()


def test_dry_run_start_stop_never_publish_hardware_commands():
    node = controller()
    node._dry_run = True
    started = Rm75PlacoController._on_start_task(
        node, request(2), ResetPolicyController.Response())
    stopped = Rm75PlacoController._on_reset(
        node, request(3), ResetPolicyController.Response())
    assert started.success and stopped.success and not node._task_enabled
    node._movej_publisher.publish.assert_not_called()
    node._gripper_publisher.publish.assert_not_called()
    node._stop_publisher.publish.assert_not_called()
