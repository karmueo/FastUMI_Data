"""Unitree Dex1-1 DDS 与 ROS 2 归一化夹爪话题的适配节点。"""

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32

from unitree_gripper.control import GripperControl
from unitree_gripper._vendor.unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_gripper._vendor.unitree_sdk2py.idl.unitree_go.msg.dds_ import (
    MotorCmd_,
    MotorCmds_,
    MotorStates_,
)


class Dex1GripperNode(Node):
    """以 50 Hz 控制电机，并发布基于真实反馈的开度。"""

    def __init__(self):
        super().__init__("dex1_gripper_node")
        defaults = {
            "network_interface": "wlp131s0",
            "cmd_topic_name": "/motion_control/gripper_command",
            "state_topic_name": "/motion_control/gripper_state",
            "pos_close_rad": 0.02,
            "pos_open_rad": 5.0,
            "kp": 5.0,
            "kd": 0.05,
            "max_speed_rad_s": 6.0,
            "max_hold_error_rad": 0.30,
            "feedback_timeout_s": 0.5,
            "startup_openness": 1.0,
            "control_rate_hz": 50.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        config = {name: self.get_parameter(name).value for name in defaults}
        rate = float(config["control_rate_hz"])
        kp = float(config["kp"])
        kd = float(config["kd"])
        if not all(math.isfinite(value) for value in (rate, kp, kd)):
            raise ValueError("控制频率及增益必须是有限数值")
        if rate <= 0 or kp < 0 or kd < 0:
            raise ValueError("控制频率必须大于零，增益不能为负")
        self._period = 1.0 / rate
        self._kp = kp
        self._kd = kd
        self._last_tick = time.monotonic()
        self._controller = GripperControl(
            pos_close=float(config["pos_close_rad"]),
            pos_open=float(config["pos_open_rad"]),
            max_speed_rad_s=float(config["max_speed_rad_s"]),
            max_hold_error_rad=float(config["max_hold_error_rad"]),
            feedback_timeout_s=float(config["feedback_timeout_s"]),
            target_ratio=float(config["startup_openness"]),
        )
        self._dds_publishers = {}
        self._dds_subscribers = {}
        self._dds_commands = {}
        self._last_error_log = 0.0

        network = str(config["network_interface"])
        if not network:
            raise ValueError("network_interface 不能为空")
        ChannelFactoryInitialize(0, network)
        time.sleep(0.3)
        try:
            for side in self._controller.sides:
                publisher = ChannelPublisher(f"rt/dex1/{side}/cmd", MotorCmds_)
                publisher.Init()
                self._dds_publishers[side] = publisher
                subscriber = ChannelSubscriber(f"rt/dex1/{side}/state", MotorStates_)
                subscriber.Init(None, 1)
                self._dds_subscribers[side] = subscriber
                self._dds_commands[side] = MotorCmds_(cmds=[
                    MotorCmd_(1, 0.0, 0.0, 0.0, kp, kd, [0, 0, 0])
                ])
        except Exception:
            self._close_channels()
            raise

        self._subscription = self.create_subscription(
            Float32, str(config["cmd_topic_name"]), self._on_command, 10
        )
        self._state_publisher = self.create_publisher(
            Float32, str(config["state_topic_name"]), 10
        )
        self._timer = self.create_timer(self._period, self._control_loop)
        self.get_logger().info(
            f"Unitree 夹爪节点已启动，网卡 {network}，初始开度 "
            f"{self._controller.target_ratio:.2f}"
        )

    def _on_command(self, message: Float32) -> None:
        """忽略越界及 NaN/Inf 指令，保持最后一个有效目标。"""
        if not self._controller.set_target(float(message.data)):
            self.get_logger().warn("忽略非法夹爪开度，要求有限值且位于 [0, 1]")

    def _control_loop(self) -> None:
        """读取真实反馈、限速下发并发布当前开度。"""
        now = time.monotonic()
        dt = min(max(now - self._last_tick, 1e-6), self._period)
        self._last_tick = now
        for side, subscriber in self._dds_subscribers.items():
            try:
                feedback = subscriber.Read(0.001)
                states = getattr(feedback, "states", None)
                if states:
                    self._controller.update_feedback(side, float(states[0].q), now)
            except Exception as exc:
                if now - self._last_error_log > 2.0:
                    self.get_logger().error(f"读取 {side} 电机反馈失败: {exc}")
                    self._last_error_log = now

        for side, position in self._controller.next_commands(now, dt).items():
            command = self._dds_commands[side].cmds[0]
            command.q = position
            command.kp = self._kp
            command.kd = self._kd
            command.mode = 1
            if not self._dds_publishers[side].Write(self._dds_commands[side]):
                if now - self._last_error_log > 2.0:
                    self.get_logger().error(f"发送 {side} 电机命令失败")
                    self._last_error_log = now

        ratio = self._controller.current_ratio(now)
        if ratio is not None:
            self._state_publisher.publish(Float32(data=float(ratio)))

    def _close_channels(self) -> None:
        for subscriber in self._dds_subscribers.values():
            try:
                subscriber.Close()
            except Exception:
                pass
        for publisher in self._dds_publishers.values():
            try:
                publisher.Close()
            except Exception:
                pass

    def destroy_node(self):
        """先释放 DDS 端点，再销毁 ROS 节点。"""
        self._close_channels()
        return super().destroy_node()


def main(args=None):
    """运行夹爪控制节点。"""
    rclpy.init(args=args)
    node = None
    try:
        node = Dex1GripperNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
