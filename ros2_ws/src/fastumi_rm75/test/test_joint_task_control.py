"""内存发布器验证新执行器生命周期及 dry-run 无硬件副作用。"""

import asyncio
import json
import time
from types import MethodType, SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
from fastumi_interfaces.srv import ResetPolicyController
from rclpy.task import Future

from fastumi_rm75.joint_control import JointTrajectory
from fastumi_rm75.rm75_joint_controller import Rm75JointController


def controller(dry=True):
    node = Rm75JointController.__new__(Rm75JointController)
    now = time.monotonic()
    values = dict(
        _task_control_enabled=True, _require_start_state=True, _task_enabled=False, _enabled_episode=None,
        _dry_run=dry, _minimum_episode=1, _latest_episode=None, _latest_sequence=-1,
        _faulted_episode=None, _episode_started_at=now, _home_since=now - 1,
        _start_verified_episode=None, _feedback_positions=np.zeros(7), _feedback_monotonic=now,
        _feedback_timeout_s=0.25, _start_joints=np.zeros(7), _start_joint_tolerance=0.035,
        _start_settle_s=0.5, _start_gripper_min=0.95, _gripper_state=1.0,
        _gripper_monotonic=now, _gripper_timeout_s=0.25, _valid_feedback=True, _valid_feedback_at=now,
        _home_future=None, _home_episode=None, _home_started_at=None, _home_command_sent_at=None,
        _home_accepted=False, _home_settled_since=None, _stop_future=None, _close_allowed=False,
        _task_io_group=object(), _movej_publisher=Mock(), _gripper_publisher=Mock(), _stop_publisher=Mock(),
        _joint_publisher=Mock(), _debug_publisher=Mock(), _trajectory=None, _command_positions=None,
        _last_gripper=None, _last_tick_ns=None, _last_clock_ns=None, _period=0.02,
        _velocities=np.ones(7), _joint_lower=-np.ones(7), _joint_upper=np.ones(7),
        _workspace_min=np.array([-0.6, -0.5, 0.16]), _workspace_max=np.array([0.6, 0.5, 0.5]),
        _reason="idle", _robot=Mock(),
    )
    for key, value in values.items():
        setattr(node, key, value)
    node.get_logger = lambda: Mock()
    node.get_clock = lambda: NS(now=lambda: NS(nanoseconds=2_000_000_000, to_msg=lambda: NS(sec=2, nanosec=0)))
    node.get_parameter = lambda name: NS(value=2.0)
    node._set_robot_joints = Mock()
    node._robot.get_T_world_frame.return_value = np.array([[1, 0, 0, 0.3], [0, 1, 0, 0], [0, 0, 1, 0.3], [0, 0, 0, 1]])
    return node


def req(episode):
    value = ResetPolicyController.Request()
    value.episode_id = episode
    return value


def test_dry_run_still_requires_start_and_fresh_gripper():
    node = controller()
    node._gripper_monotonic -= 1
    response = node._on_start_task(req(2), ResetPolicyController.Response())
    assert not response.success
    node._gripper_monotonic = time.monotonic()
    response = node._on_start_task(req(2), ResetPolicyController.Response())
    assert response.success and node._task_enabled


def test_dry_run_start_stop_home_and_shutdown_never_publish_hardware():
    node = controller()
    assert node._on_start_task(req(2), ResetPolicyController.Response()).success
    assert node._on_reset(req(3), ResetPolicyController.Response()).success
    result = asyncio.run(node._on_return_to_start(req(3), ResetPolicyController.Response()))
    assert not result.success  # Physical return is intentionally unavailable in dry-run.
    node._stop_control("shutdown")
    for publisher in (node._joint_publisher, node._gripper_publisher, node._stop_publisher, node._movej_publisher):
        publisher.publish.assert_not_called()


def test_stop_interrupts_home_and_waits_for_driver_ack_state():
    node = controller(dry=False)
    node._home_future = Future()
    pending = node._home_future
    result = node._on_reset(req(2), ResetPolicyController.Response())
    assert result.success and pending.done() and pending.result()[0] is False
    assert node._stop_future is not None and not node._stop_future.done()
    node._on_move_stop_result(NS(data=True))
    assert node._stop_future.result() is True
    assert not node._task_enabled


def test_feedback_watchdog_disables_episode_and_old_policy():
    node = controller(dry=False)
    node._task_enabled, node._enabled_episode = True, 2
    node._gripper_monotonic -= 1
    node._tick()
    assert not node._task_enabled
    node._stop_publisher.publish.assert_called_once()
    node._on_policy(NS(episode_id=2, sequence_id=99))
    assert node._trajectory is None


def test_interpolated_fk_workspace_and_clock_regression_stop():
    node = controller()
    node._task_enabled, node._enabled_episode = True, 2
    node._command_positions = np.zeros(7)
    node._trajectory = JointTrajectory(np.array([2_020_000_000]), np.ones((1, 7)) * 0.2,
                                       np.ones(1), 1_980_000_000, np.zeros(7), 1)
    node._robot.get_T_world_frame.return_value[0, 3] = 0.7
    node._tick()
    assert not node._task_enabled and "workspace" in node._reason
    node._task_enabled = True
    node._last_clock_ns = 3_000_000_000
    node._tick()
    assert not node._task_enabled and "backwards" in node._reason


def test_exhaustion_stops_hardware_and_logs_last_prediction_timing():
    node = controller(dry=False)
    logger = Mock()
    node.get_logger = lambda: logger
    node._task_enabled, node._enabled_episode = True, 2
    node._command_positions = np.zeros(7)
    node._trajectory = JointTrajectory(np.array([1_990_000_000]), np.zeros((1, 7)),
                                       np.ones(1), 1_500_000_000, np.zeros(7), 1)
    node._sequence_timing = dict(source_ns=730_000_000, received_ns=1_500_000_000,
                                end_ns=1_990_000_000, sequence_id=9)
    node._tick()
    assert not node._task_enabled and node._trajectory is None
    assert node._reason == "trajectory horizon exhausted"
    node._stop_publisher.publish.assert_called_once()
    log = logger.warning.call_args_list[-1].args[0]
    timing = json.loads(log.split(": ", 1)[1])
    assert timing["sequence_id"] == 9
    assert timing["source_age_s"] == 1.27
    assert timing["since_receive_s"] == 0.5
    assert timing["horizon_remaining_s"] == -0.01
