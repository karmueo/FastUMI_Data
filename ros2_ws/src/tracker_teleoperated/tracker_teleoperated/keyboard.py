"""提供遥操启停服务的交互式单键键盘客户端。"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
from typing import Optional, TextIO

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool, Empty, String
from std_srvs.srv import SetBool, Trigger


class TerminalKeyReader:
    """以 cbreak 模式从交互终端读取无需回车的单个按键。"""

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        """保存输入流并初始化可恢复的终端状态。"""
        self._stream = stream or sys.stdin
        self._file_descriptor: Optional[int] = None
        self._original_settings: Optional[list] = None
        self.enabled = False

    def __enter__(self) -> "TerminalKeyReader":
        """在交互终端中进入 cbreak 模式。"""
        if not self._stream.isatty():
            return self
        self._file_descriptor = self._stream.fileno()
        self._original_settings = termios.tcgetattr(self._file_descriptor)
        tty.setcbreak(self._file_descriptor)
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """恢复进入读取器之前的终端设置。"""
        del exc_type, exc_value, traceback
        if (
            self._file_descriptor is not None
            and self._original_settings is not None
        ):
            termios.tcsetattr(
                self._file_descriptor,
                termios.TCSADRAIN,
                self._original_settings,
            )
        self.enabled = False

    def read_key(self, timeout_s: float) -> Optional[str]:
        """在超时内读取一个字符，非交互输入时安全等待。"""
        if not self.enabled:
            time.sleep(timeout_s)
            return None
        readable, _, _ = select.select([self._stream], [], [], timeout_s)
        if not readable:
            return None
        return self._stream.read(1)


def requested_state_for_key(
    key: Optional[str], enabled: bool
) -> Optional[bool]:
    """把空格和 s 键映射为目标启用状态。"""
    if key == " ":
        return not enabled
    if key in ("s", "S"):
        return False
    return None


def is_calibrate_key(key: Optional[str]) -> bool:
    """判断按键是否请求记录下一项工作空间标定样本。"""
    return key in ("c", "C")


def is_home_key(key: Optional[str]) -> bool:
    """判断按键是否请求回到配置或启动时记录的关节位姿。"""
    return key in ("h", "H")


def keyboard_instructions(auto_mapping_enabled: bool) -> str:
    """按实际映射模式生成键盘快捷键和下一步操作提示。"""
    common_keys = (
        "[h] 回到配置的关节位姿，"
        "[空格] 启用/暂停，[s] 暂停并取消当前操作，"
        "[q] 暂停并退出"
    )
    if auto_mapping_enabled:
        return (
            "当前为自动映射模式，无需执行三点工作空间标定。\n"
            "请确认 Tracker odom 的 +X/+Y/+Z 与机械臂 Base "
            "+X/+Y/+Z 对齐，待跟踪稳定后将 UMI 移到操作起点，"
            "然后按 [空格] 启用遥操。\n"
            f"快捷键: {common_keys}"
        )
    return (
        "当前为工作空间标定模式。\n"
        "请依次在起点、大致向上移动后、从当前位置大致向前移动后按 [c]；"
        "标定会平均修正两段方向偏差并保存；后续启停和回位会保留标定方向。"
        "完成后按 [空格] 启用遥操。\n"
        f"快捷键: [c] 记录工作空间标定点，{common_keys}"
    )


class TrackerTeleopKeyboard(Node):
    """发布存活心跳，并根据控制节点状态调用启停服务。"""

    def __init__(self) -> None:
        """创建心跳、状态订阅和启停服务客户端。"""
        super().__init__("tracker_teleop_keyboard")
        self.enabled = False
        self.auto_mapping_enabled: Optional[bool] = None
        # 最近显示的状态说明，用于避免相同提示重复刷屏。
        self._last_status = ""
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._heartbeat_publisher = self.create_publisher(
            Empty, "/tracker_teleoperated/keyboard_heartbeat", 10
        )
        self.create_subscription(
            Bool,
            "/tracker_teleoperated/enabled",
            self._enabled_callback,
            state_qos,
        )
        self.create_subscription(
            String,
            "/tracker_teleoperated/status",
            self._status_callback,
            state_qos,
        )
        self.create_subscription(
            Bool,
            "/tracker_teleoperated/auto_mapping_enabled",
            self._auto_mapping_callback,
            state_qos,
        )
        self._service_client = self.create_client(
            SetBool, "/tracker_teleoperated/set_enabled"
        )
        self._calibration_client = self.create_client(
            Trigger, "/tracker_teleoperated/calibrate_workspace"
        )
        self._home_client = self.create_client(
            Trigger, "/tracker_teleoperated/return_home"
        )
        self.create_timer(0.1, self._publish_heartbeat)

    def _enabled_callback(self, message: Bool) -> None:
        """同步控制节点报告的实际启用状态。"""
        self.enabled = bool(message.data)

    def _status_callback(self, message: String) -> None:
        """即时显示控制端的异步状态变化、暂停原因和恢复步骤。"""
        if message.data and message.data != self._last_status:
            self._last_status = message.data
            print(f"\n[控制状态] {message.data}", flush=True)

    def _auto_mapping_callback(self, message: Bool) -> None:
        """同步控制节点实际生效的自动映射状态。"""
        self.auto_mapping_enabled = bool(message.data)

    def wait_for_configuration(self, timeout_s: float = 3.0) -> None:
        """等待控制节点的映射状态，避免显示错误操作提示。"""
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and self.auto_mapping_enabled is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError("未收到控制节点的映射模式状态")
            rclpy.spin_once(self, timeout_sec=min(0.1, remaining))

    def _publish_heartbeat(self) -> None:
        """以 10 Hz 告知控制节点键盘进程仍然存活。"""
        self._heartbeat_publisher.publish(Empty())

    def request_enabled(self, enabled: bool, timeout_s: float = 2.0) -> str:
        """同步调用启停服务并返回服务端说明。"""
        if not self._service_client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError("遥操启停服务未就绪")
        request = SetBool.Request()
        request.data = bool(enabled)
        future = self._service_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        response = future.result()
        if response is None:
            raise RuntimeError("遥操启停服务调用超时")
        if not response.success:
            raise RuntimeError(response.message)
        return str(response.message)

    def request_workspace_calibration(self, timeout_s: float = 2.0) -> str:
        """同步请求记录工作空间标定的下一项 Tracker 样本。"""
        if not self._calibration_client.wait_for_service(
            timeout_sec=timeout_s
        ):
            raise RuntimeError("工作空间标定服务未就绪")
        future = self._calibration_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        response = future.result()
        if response is None:
            raise RuntimeError("工作空间标定服务调用超时")
        if not response.success:
            raise RuntimeError(response.message)
        return str(response.message)

    def request_return_home(self, timeout_s: float = 2.0) -> str:
        """同步请求机械臂回到配置或启动时记录的关节位姿。"""
        if not self._home_client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError("机械臂回位服务未就绪")
        future = self._home_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        response = future.result()
        if response is None:
            raise RuntimeError("机械臂回位服务调用超时")
        if not response.success:
            raise RuntimeError(response.message)
        return str(response.message)


def main(args=None) -> None:
    """运行键盘循环；退出前始终请求暂停机械臂。"""
    rclpy.init(args=args)
    node = TrackerTeleopKeyboard()
    try:
        node.wait_for_configuration()
        assert node.auto_mapping_enabled is not None
        print(keyboard_instructions(node.auto_mapping_enabled))
        with TerminalKeyReader() as reader:
            if not reader.enabled:
                raise RuntimeError("键盘节点需要交互式终端")
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.0)
                key = reader.read_key(0.02)
                if key in ("q", "Q"):
                    break
                if is_calibrate_key(key):
                    if node.auto_mapping_enabled:
                        print(
                            "自动映射已启用，无需标定；请对齐坐标轴，"
                            "将 UMI 移到操作起点后按空格启用。"
                        )
                        continue
                    try:
                        print(node.request_workspace_calibration())
                    except RuntimeError as error:
                        print(f"操作失败: {error}")
                    continue
                if is_home_key(key):
                    try:
                        print(node.request_return_home())
                    except RuntimeError as error:
                        print(f"操作失败: {error}")
                    continue
                requested = requested_state_for_key(key, node.enabled)
                if requested is None:
                    continue
                try:
                    print(node.request_enabled(requested))
                except RuntimeError as error:
                    print(f"操作失败: {error}")
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        print(f"键盘节点退出: {error}")
    finally:
        try:
            print(node.request_enabled(False, timeout_s=1.0))
        except RuntimeError as error:
            print(f"退出时暂停请求失败: {error}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
