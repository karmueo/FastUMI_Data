"""按空格键将 RM75 移至初始关节位姿，并将夹爪完全打开。"""

import math
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from rm_ros_interfaces.msg import Movej
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32

from fastumi_bringup.initial_pose import INITIAL_JOINT_POSITIONS


JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
FEEDBACK_FRESHNESS_SECONDS = 0.5
RESULT_TIMEOUT_SECONDS = 120.0
ALREADY_AT_TARGET_TOLERANCE = 0.01


class TerminalKeyReader:
    """临时启用逐键输入，并在退出时恢复终端配置。"""

    def __init__(self, stream):
        self.stream = stream
        self.original_settings = None

    def __enter__(self):
        if not self.stream.isatty():
            raise RuntimeError("键盘回位需要交互终端；请在第二个终端运行 ros2 run")
        fd = self.stream.fileno()
        self.original_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
        except BaseException:
            termios.tcsetattr(fd, termios.TCSADRAIN, self.original_settings)
            self.original_settings = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.original_settings is not None:
            termios.tcsetattr(
                self.stream.fileno(), termios.TCSADRAIN, self.original_settings
            )
            self.original_settings = None

    def read_key(self, timeout_seconds):
        """等待单个按键；超时返回 None。"""
        readable, _, _ = select.select([self.stream], [], [], timeout_seconds)
        return self.stream.read(1) if readable else None


class KeyboardHome(Node):
    """检查硬件就绪后，同时请求夹爪全开和单次阻塞 MoveJ。"""

    def __init__(self):
        super().__init__("keyboard_home")
        self.current_positions = None
        self.last_feedback_at = None
        self.last_valid_feedback_at = None
        self.command_sent_at = None
        self.disabled = False

        self.command_publisher = self.create_publisher(
            Movej, "/rm_driver/movej_cmd", 10
        )
        self.gripper_publisher = self.create_publisher(
            Float32, "/motion_control/gripper_command", 10
        )
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10
        )
        self.create_subscription(
            Bool, "/rm_driver/udp_feedback_valid", self._feedback_valid_callback, 10
        )
        self.create_subscription(
            Bool, "/rm_driver/movej_result", self._result_callback, 10
        )
        self.create_timer(0.1, self._check_timeout)
        self.get_logger().info("键盘回位已就绪：按空格键回位 RM75 并完全打开夹爪")

    def _joint_state_callback(self, message):
        """按名称读取七关节弧度反馈，忽略不完整或非有限值。"""
        if len(message.name) != len(message.position):
            return
        positions_by_name = dict(zip(message.name, message.position))
        if any(name not in positions_by_name for name in JOINT_NAMES):
            return
        positions = [float(positions_by_name[name]) for name in JOINT_NAMES]
        if not all(math.isfinite(position) for position in positions):
            return
        self.current_positions = positions
        self.last_feedback_at = time.monotonic()

    def _feedback_valid_callback(self, message):
        """仅接受当前仍有效的驱动 UDP 反馈。"""
        self.last_valid_feedback_at = time.monotonic() if message.data else None

    def _result_callback(self, message):
        """处理本次 MoveJ 结果；失败后需人工重启监听节点。"""
        if self.command_sent_at is None or self.disabled:
            return
        self.command_sent_at = None
        if message.data:
            self.get_logger().info("RM75 已完成初始位姿回位")
        else:
            self.disabled = True
            self.get_logger().error("MoveJ 回位失败；停止接受按键，请检查机械臂后重启节点")

    def _check_timeout(self):
        """超时后锁定命令，避免未知运动状态下再次发起回位。"""
        if (self.command_sent_at is not None and
                time.monotonic() - self.command_sent_at > RESULT_TIMEOUT_SECONDS):
            self.command_sent_at = None
            self.disabled = True
            self.get_logger().error("等待 MoveJ 结果超时；停止接受按键，请检查机械臂后重启节点")

    def handle_key(self, key):
        """空格键请求夹爪全开及回位；硬件未就绪时不下发命令。"""
        if key != " ":
            return
        if self.disabled:
            self.get_logger().warning("回位节点已锁定；请检查机械臂后重启节点")
            return
        if self.command_sent_at is not None:
            self.get_logger().warning("回位运动尚未结束，忽略重复按键")
            return

        now = time.monotonic()
        if (self.current_positions is None or
                self.last_feedback_at is None or
                self.last_valid_feedback_at is None or
                now - self.last_feedback_at > FEEDBACK_FRESHNESS_SECONDS or
                now - self.last_valid_feedback_at > FEEDBACK_FRESHNESS_SECONDS or
                self.command_publisher.get_subscription_count() == 0 or
                self.gripper_publisher.get_subscription_count() == 0):
            self.get_logger().warning("机械臂或夹爪尚未就绪，未发送回位与全开命令")
            return

        self.gripper_publisher.publish(Float32(data=1.0))
        self.get_logger().info("已发送夹爪全开命令")

        maximum_error = max(
            abs(current - target)
            for current, target in zip(self.current_positions, INITIAL_JOINT_POSITIONS)
        )
        if maximum_error <= ALREADY_AT_TARGET_TOLERANCE:
            self.get_logger().info("RM75 已在初始位姿，无需移动")
            return

        command = Movej()
        command.joint = list(INITIAL_JOINT_POSITIONS)
        command.speed = 20
        command.block = True
        command.trajectory_connect = 0
        command.dof = 7
        self.command_sent_at = now
        self.command_publisher.publish(command)
        self.get_logger().warning("以 20% 速度将 RM75 移至初始位姿")


def main(args=None):
    """运行 ROS 回位节点及当前终端的逐键读取循环。"""
    if not sys.stdin.isatty():
        print("键盘回位需要交互终端；请在第二个终端运行 ros2 run", file=sys.stderr)
        return 2

    rclpy.init(args=args)
    node = None
    try:
        node = KeyboardHome()
        with TerminalKeyReader(sys.stdin) as reader:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
                node.handle_key(reader.read_key(0.05))
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError) as error:
        print(f"键盘回位无法读取终端：{error}", file=sys.stderr)
        return 2
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
