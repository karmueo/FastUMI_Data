"""验证 agx 回位程序的单次命令与超时语义，不发布实际运动命令。"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import Mock
import time

import pytest
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


@pytest.fixture
def mover():
    """将原程序命令发布器替换为内存 mock。"""
    script = Path(__file__).parents[2] / 'ros2_rm_robot/rm_bringup/scripts/rm_75_initial_pose.py'
    spec = spec_from_file_location('agx_initial_pose', script)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    rclpy.init()
    node = module.InitialPoseMover()
    node.command_publisher = Mock()
    node.command_publisher.get_subscription_count.return_value = 1
    yield node
    node.destroy_node()
    rclpy.shutdown()


def feedback(node, positions):
    """直接提供有效反馈，避免通过真实驱动。"""
    node._joint_state_callback(JointState(name=[f'joint{i}' for i in range(1, 8)], position=positions))
    node._feedback_valid_callback(Bool(data=True))


def test_home_sends_only_once(mover):
    """有效反馈后最多发送一次阻塞 MoveJ，并接受成功结果。"""
    feedback(mover, [1.]*7)
    mover._timer_callback()
    mover._timer_callback()
    mover.command_publisher.publish.assert_called_once()
    command = mover.command_publisher.publish.call_args.args[0]
    assert command.speed == 20 and command.block and command.dof == 7
    mover._result_callback(Bool(data=True))
    assert mover.finished and not mover.failed


def test_already_home_no_command(mover):
    """已在目标容差内时立即完成，不发运动指令。"""
    feedback(mover, [0.]*7)
    mover._timer_callback()
    assert mover.finished and not mover.failed
    mover.command_publisher.publish.assert_not_called()


def test_feedback_timeout(mover):
    """未收到有效反馈时超时失败。"""
    mover.started_at = time.monotonic()-31
    mover._timer_callback()
    assert mover.finished and mover.failed
    mover.command_publisher.publish.assert_not_called()


def test_result_timeout(mover):
    """已发命令等待结果超时后失败，不自动重试。"""
    mover.command_sent_at = time.monotonic()-121
    mover._timer_callback()
    assert mover.finished and mover.failed
    mover.command_publisher.publish.assert_not_called()
