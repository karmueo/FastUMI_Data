"""无硬件验证键盘回位命令与终端输入。"""

import os
import pty
import sys
import termios
import time
from unittest.mock import Mock

import pytest
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from fastumi_bringup.initial_pose import INITIAL_JOINT_POSITIONS
from fastumi_bringup.keyboard_home import KeyboardHome, TerminalKeyReader, main


@pytest.fixture
def home_node(monkeypatch):
    """仅构造节点逻辑状态，避免创建 ROS 网络参与者或真实发布器。"""
    node = KeyboardHome.__new__(KeyboardHome)
    node.current_positions = None
    node.last_feedback_at = None
    node.last_valid_feedback_at = None
    node.command_sent_at = None
    node.disabled = False
    node.command_publisher = Mock()
    node.command_publisher.get_subscription_count.return_value = 1
    node.gripper_publisher = Mock()
    node.gripper_publisher.get_subscription_count.return_value = 1
    monkeypatch.setattr(KeyboardHome, "get_logger", lambda self: Mock())
    return node


def provide_feedback(node, positions=None):
    """发送按关节名称排列的当前弧度位置和有效反馈。"""
    if positions is None:
        positions = [1.0] * 7
    node._joint_state_callback(JointState(
        name=[f"joint{index}" for index in range(1, 8)], position=positions
    ))
    node._feedback_valid_callback(Bool(data=True))


def test_space_publishes_home_and_full_open_once(home_node):
    provide_feedback(home_node)
    home_node.handle_key("x")
    home_node.handle_key(" ")
    home_node.handle_key(" ")
    home_node.command_publisher.publish.assert_called_once()
    home_node.gripper_publisher.publish.assert_called_once()
    assert home_node.gripper_publisher.publish.call_args.args[0].data == 1.0
    command = home_node.command_publisher.publish.call_args.args[0]
    assert list(command.joint) == pytest.approx(INITIAL_JOINT_POSITIONS, abs=1e-6)
    assert (command.speed, command.block, command.trajectory_connect, command.dof) == (
        20, True, 0, 7
    )
    home_node._result_callback(Bool(data=True))
    assert home_node.command_sent_at is None
    assert not home_node.disabled


def test_space_without_valid_feedback_is_ignored(home_node):
    home_node.handle_key(" ")
    provide_feedback(home_node)
    home_node._feedback_valid_callback(Bool(data=False))
    home_node.handle_key(" ")
    provide_feedback(home_node)
    home_node.last_feedback_at = time.monotonic() - 1.0
    home_node.handle_key(" ")
    home_node.command_publisher.get_subscription_count.return_value = 0
    provide_feedback(home_node)
    home_node.handle_key(" ")
    home_node.command_publisher.publish.assert_not_called()
    home_node.gripper_publisher.publish.assert_not_called()


def test_already_home_still_opens_gripper(home_node):
    provide_feedback(home_node, INITIAL_JOINT_POSITIONS)
    home_node.handle_key(" ")
    home_node.command_publisher.publish.assert_not_called()
    home_node.gripper_publisher.publish.assert_called_once()
    assert home_node.gripper_publisher.publish.call_args.args[0].data == 1.0


def test_missing_gripper_subscriber_skips_both_commands(home_node):
    provide_feedback(home_node)
    home_node.gripper_publisher.get_subscription_count.return_value = 0
    home_node.handle_key(" ")
    home_node.command_publisher.publish.assert_not_called()
    home_node.gripper_publisher.publish.assert_not_called()


@pytest.mark.parametrize("failure", ["result", "timeout"])
def test_failure_locks_out_later_commands(home_node, failure):
    provide_feedback(home_node)
    home_node.handle_key(" ")
    if failure == "result":
        home_node._result_callback(Bool(data=False))
    else:
        home_node.command_sent_at = time.monotonic() - 121
        home_node._check_timeout()
    assert home_node.disabled
    home_node.handle_key(" ")
    home_node.command_publisher.publish.assert_called_once()
    home_node.gripper_publisher.publish.assert_called_once()


def test_terminal_reader_restores_settings_and_reads_space():
    master, slave = pty.openpty()
    try:
        with os.fdopen(slave, "r", buffering=1) as stream:
            original = termios.tcgetattr(stream.fileno())
            with TerminalKeyReader(stream) as reader:
                os.write(master, b" ")
                assert reader.read_key(0.2) == " "
            assert termios.tcgetattr(stream.fileno()) == original
    finally:
        os.close(master)


def test_main_rejects_noninteractive_input(monkeypatch, capsys):
    stream = Mock()
    stream.isatty.return_value = False
    monkeypatch.setattr(sys, "stdin", stream)
    assert main() == 2
    assert "交互终端" in capsys.readouterr().err
