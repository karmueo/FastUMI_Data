"""验证实机首条策略必须等待机械臂回位及夹爪张开。"""

from types import SimpleNamespace
import time

import numpy as np

from fastumi_rm75.rm75_placo_controller import Rm75PlacoController
from fastumi_interfaces.srv import ResetPolicyController


def _gate(joints, gripper, *, joint_age=1.0, gripper_age=0.01):
    """构造不连接 ROS 图的控制器起始状态。"""
    warnings = []
    logger = SimpleNamespace(warning=lambda *args, **kwargs: warnings.append(args[0]))
    now = time.monotonic()
    node = SimpleNamespace(
        _dry_run=False,
        _require_start_state=True,
        _start_verified_episode=None,
        _feedback_positions=np.asarray(joints, dtype=np.float64),
        _start_joints=np.array([0.0, 0.3491, 0.0, 1.2217, 0.0, 1.5708, 1.5708]),
        _start_joint_tolerance=0.035,
        _home_since=now - joint_age,
        _start_settle_s=0.5,
        _gripper_state=gripper,
        _gripper_monotonic=now - gripper_age,
        _gripper_timeout_s=0.25,
        _start_gripper_min=0.95,
        get_logger=lambda: logger,
    )
    return node, warnings


def test_real_policy_waits_for_home_and_open_gripper():
    home = [0.0, 0.3491, 0.0, 1.2217, 0.0, 1.5708, 1.5708]
    for joints, gripper, joint_age, gripper_age in (
        ([0.4, *home[1:]], 1.0, 1.0, 0.01),
        (home, 0.5, 1.0, 0.01),
        (home, 1.0, 0.1, 0.01),
        (home, 1.0, 1.0, 0.5),
    ):
        node, warnings = _gate(
            joints, gripper, joint_age=joint_age, gripper_age=gripper_age)
        assert not Rm75PlacoController._start_state_ready(node, 0)
        assert node._start_verified_episode is None
        assert warnings

    node, _ = _gate(home, 0.99)
    assert Rm75PlacoController._start_state_ready(node, 0)
    assert node._start_verified_episode == 0
    assert Rm75PlacoController._start_state_ready(node, 0)
    node._start_verified_episode = None
    node._home_since = None
    assert not Rm75PlacoController._start_state_ready(node, 1)


def test_gripper_close_requires_fresh_vision_and_episode_elapsed_time():
    now = time.monotonic()
    node = SimpleNamespace(
        _require_close_approval=True,
        _close_allowed=True,
        _close_approval_monotonic=now - 0.05,
        _close_approval_timeout_s=0.25,
        _episode_started_at=now - 2.0,
        _close_min_episode_time_s=1.5,
        _gripper_state=0.99,
        _gripper_monotonic=now - 0.05,
        _gripper_timeout_s=0.25,
    )
    assert Rm75PlacoController._close_is_allowed(node)
    node._episode_started_at = now - 0.5
    assert not Rm75PlacoController._close_is_allowed(node)
    node._episode_started_at = now - 2.0
    node._close_approval_monotonic = now - 0.4
    assert not Rm75PlacoController._close_is_allowed(node)
    node._close_approval_monotonic = now - 0.05
    node._close_allowed = False
    assert not Rm75PlacoController._close_is_allowed(node)


def test_reset_stops_old_trajectory_and_rejects_queued_old_episode():
    """控制器确认重置前停机，并阻止旧 episode 的排队预测重新启动。"""
    events = []
    node = SimpleNamespace(
        _minimum_episode=0,
        _latest_episode=2,
        _start_verified_episode=2,
        _home_since=1.0,
        _faulted_episode=2,
        _episode_started_at=1.0,
        _close_allowed=True,
        _stop_control=lambda reason: events.append(reason),
    )
    request = ResetPolicyController.Request()
    request.episode_id = 3
    response = Rm75PlacoController._on_reset(
        node, request, ResetPolicyController.Response())
    assert response.success and events == ["episode reset"]
    assert node._minimum_episode == 3
    assert node._start_verified_episode is None and node._home_since is None
    assert node._faulted_episode is None and not node._close_allowed
    old_request = ResetPolicyController.Request()
    old_request.episode_id = 2
    old_response = Rm75PlacoController._on_reset(
        node, old_request, ResetPolicyController.Response())
    assert not old_response.success and node._minimum_episode == 3
    Rm75PlacoController._on_policy(
        node, SimpleNamespace(episode_id=2, sequence_id=99))
    assert events == ["episode reset", "episode reset"]
