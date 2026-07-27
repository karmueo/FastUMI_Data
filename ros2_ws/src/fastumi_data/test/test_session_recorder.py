"""测试 FastUMI 会话录制子进程的联动监管。"""

import subprocess
from unittest.mock import Mock

import pytest

from fastumi_data.session_recorder import _wait_for_recording_processes


def test_reports_episode_manager_exit() -> None:
    """验证事件管理器退出会立即使整场录制失败。"""
    manager_process = Mock()
    manager_process.poll.return_value = 2
    bag_process = Mock()

    with pytest.raises(RuntimeError, match="episode_manager"):
        _wait_for_recording_processes(manager_process, bag_process)

    bag_process.wait.assert_not_called()


def test_returns_when_bag_finishes_cleanly() -> None:
    """验证录包进程正常结束时监管函数正常返回。"""
    manager_process = Mock()
    manager_process.poll.return_value = None
    bag_process = Mock()
    bag_process.wait.return_value = 0

    _wait_for_recording_processes(manager_process, bag_process)

    bag_process.wait.assert_called_once_with(timeout=0.2)


def test_retries_while_both_processes_are_running() -> None:
    """验证录包等待超时后会继续检查管理器状态。"""
    manager_process = Mock()
    manager_process.poll.side_effect = [None, 3]
    bag_process = Mock()
    bag_process.wait.side_effect = subprocess.TimeoutExpired("bag", 0.2)

    with pytest.raises(RuntimeError, match="episode_manager"):
        _wait_for_recording_processes(manager_process, bag_process)
