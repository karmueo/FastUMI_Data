"""在交互终端中按空格急停、回车回位、J 键输入关节角，调用 RM75 控制服务。"""

from dataclasses import dataclass
import math
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from fastumi_interfaces.srv import MoveJoints


OPERATION_LABELS = {"stop": "紧急停止", "home": "回到初始位姿", "move": "关节运动"}
KEY_OPERATIONS = {" ": "stop", "\r": "home", "\n": "home", "j": "move", "J": "move"}
RESPONSE_TIMEOUT_SECONDS = {"stop": 5.0, "home": 170.0, "move": 170.0}
MOTION_OPERATIONS = ("home", "move")
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
JOINT_STATE_FRESHNESS_SECONDS = 1.0
CANCEL_KEYS = ("\x1b", "\x03")
BACKSPACE_KEYS = ("\x7f", "\b")


class TerminalKeyReader:
    """临时启用逐键输入，并在退出时恢复终端配置。"""

    def __init__(self, stream):
        self.stream = stream
        self.original_settings = None

    def __enter__(self):
        if not self.stream.isatty():
            raise RuntimeError("键盘控制需要交互终端；请在单独终端运行 ros2 run")
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


@dataclass
class PendingCall:
    """记录一个服务请求及其本地等待期限。"""

    future: object
    deadline: float
    superseded: bool = False


def parse_joint_input(text, current):
    """解析单个关节的目标角（度）；留空保持 current，格式不对时抛出 ValueError。"""
    text = text.strip()
    if not text:
        return current
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("关节角不能是 NaN 或无穷值")
    return value


def parse_speed_input(text, current):
    """解析 1～100 的速度百分比；留空保持 current，格式或范围不对时抛出 ValueError。"""
    text = text.strip()
    if not text:
        return current
    speed = int(text)
    if not 1 <= speed <= 100:
        raise ValueError(f"速度必须在 1～100 之间，收到 {speed}")
    return speed


def format_degrees(values):
    return "  ".join(f"{name}={value:.2f}" for name, value in zip(JOINT_NAMES, values))


class ArmKeyboardControl(Node):
    """异步调用急停、回位与关节运动服务；运动等待期间急停键仍可立即发送。

    J 键进入逐关节输入：显示当前关节角（度）→ 从 joint1 到 joint7 依次输入目标角，
    直接回车保持该关节不变 → 显示当前速度并可输入新速度（回车保持）→ 显示汇总后按 y 执行、
    n 取消。速度初值取参数 move_speed，输入新速度后在本节点后续运动中沿用。
    输入期间按键用于编辑而不是急停，因此只允许在没有运动请求等待时进入输入；Esc 随时取消。
    """

    def __init__(self):
        super().__init__("arm_keyboard_control")
        self.declare_parameter("emergency_stop_service", "/fastumi/arm/emergency_stop")
        self.declare_parameter("return_home_service", "/fastumi/arm/return_home")
        self.declare_parameter("move_joints_service", "/fastumi/arm/move_joints")
        self.declare_parameter("move_speed", 20)
        self.move_speed = int(self.get_parameter("move_speed").value)
        if not 1 <= self.move_speed <= 100:
            raise ValueError("move_speed 必须在 1～100 之间（百分比）")
        self.service_names = {
            "stop": self.get_parameter("emergency_stop_service").value,
            "home": self.get_parameter("return_home_service").value,
            "move": self.get_parameter("move_joints_service").value,
        }
        self.service_clients = {
            "stop": self.create_client(Trigger, self.service_names["stop"]),
            "home": self.create_client(Trigger, self.service_names["home"]),
            "move": self.create_client(MoveJoints, self.service_names["move"]),
        }
        self.create_subscription(JointState, "/joint_states", self._joint_state_callback, 10)
        self.pending = {}
        self.current_degrees = None
        self.joint_state_at = None
        self._reset_input()
        self._emit("键盘控制已就绪：空格紧急停止，回车回到初始位姿，J 输入关节角运动；Ctrl+C 退出")

    @staticmethod
    def _emit(message):
        print(message, flush=True)

    @staticmethod
    def _echo(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    def _joint_state_callback(self, message):
        """按名称读取七关节弧度反馈并转换为度，忽略不完整或非有限值。"""
        if len(message.name) != len(message.position):
            return
        positions_by_name = dict(zip(message.name, message.position))
        if any(name not in positions_by_name for name in JOINT_NAMES):
            return
        positions = [float(positions_by_name[name]) for name in JOINT_NAMES]
        if not all(math.isfinite(position) for position in positions):
            return
        self.current_degrees = [math.degrees(position) for position in positions]
        self.joint_state_at = time.monotonic()

    def _reset_input(self):
        self.input_stage = None
        self.input_buffer = ""
        self.input_start = None
        self.input_degrees = []

    def _motion_pending(self):
        return any(operation in self.pending for operation in MOTION_OPERATIONS)

    def handle_key(self, key):
        """输入模式下按键交给行编辑；否则空格急停、回车回位、J 进入关节角输入。"""
        if key is None:
            return
        if self.input_stage is not None:
            self._handle_input_key(key)
            return
        operation = KEY_OPERATIONS.get(key)
        if operation is None:
            return
        label = OPERATION_LABELS[operation]
        if operation in MOTION_OPERATIONS and self._motion_pending():
            self._emit(f"{label}：已有运动请求在执行，忽略按键（空格可急停）")
            return
        if operation == "move":
            self._begin_input()
            return
        self._send(operation, Trigger.Request())

    def _send(self, operation, request):
        """异步调用服务；急停会让正在等待的运动请求标记为被取代。"""
        label = OPERATION_LABELS[operation]
        client = self.service_clients[operation]
        if not client.service_is_ready():
            self._emit(f"{label}：服务 {self.service_names[operation]} 不可用，未发送请求")
            return False
        try:
            future = client.call_async(request)
        except Exception as error:
            self._emit(f"{label}：服务调用失败：{error}")
            return False

        if operation == "stop":
            for motion in MOTION_OPERATIONS:
                if motion in self.pending:
                    self.pending[motion].superseded = True
        # 重复急停覆盖旧的等待记录，只报告最新一次结果。
        self.pending[operation] = PendingCall(
            future, time.monotonic() + RESPONSE_TIMEOUT_SECONDS[operation])
        self._emit(f"{label}：已发送请求，等待服务执行结果…")
        return True

    def _begin_input(self):
        """显示当前关节角并从 joint1 开始提示；没有新鲜反馈时不进入输入。"""
        if (self.current_degrees is None or self.joint_state_at is None or
                time.monotonic() - self.joint_state_at > JOINT_STATE_FRESHNESS_SECONDS):
            self._emit("关节运动：未收到新鲜的 /joint_states，无法显示当前关节角，未进入输入")
            return
        # 固定进入输入时的关节角作为“保持不变”的值，避免输入期间反馈抖动改变目标。
        self.input_start = list(self.current_degrees)
        self.input_degrees = []
        self.input_buffer = ""
        self.input_stage = "joints"
        self._emit("—— 关节运动输入（逐个输入目标角，单位度；直接回车保持不变；Esc 取消）——")
        self._emit(f"当前关节角（度）：{format_degrees(self.input_start)}")
        self._prompt_joint()

    def _prompt_joint(self):
        index = len(self.input_degrees)
        self._echo(f"{JOINT_NAMES[index]}（当前 {self.input_start[index]:.2f}）：")

    def _handle_input_key(self, key):
        """逐键行编辑：回车提交当前关节，退格删除，Esc 取消；确认阶段只认 y/n。"""
        if key in CANCEL_KEYS:
            self._emit("")
            self._emit("关节运动：已取消输入")
            self._reset_input()
            return
        if self.input_stage == "confirm":
            if key in ("y", "Y"):
                self._emit(key)
                request = MoveJoints.Request(
                    joint_degrees=self.input_degrees, speed=self.move_speed)
                self._reset_input()
                self._send("move", request)
            elif key in ("n", "N"):
                self._emit(key)
                self._emit("关节运动：已取消本次执行")
                self._reset_input()
            return
        if key in ("\r", "\n"):
            self._emit("")
            line, self.input_buffer = self.input_buffer, ""
            if self.input_stage == "speed":
                self._submit_speed(line)
            else:
                self._submit_joint(line)
            return
        if key in BACKSPACE_KEYS:
            if self.input_buffer:
                self.input_buffer = self.input_buffer[:-1]
                self._echo("\b \b")
            return
        if key.isprintable():
            self.input_buffer += key
            self._echo(key)

    def _submit_joint(self, line):
        """记录当前关节目标；格式不对时提示并重新输入该关节。"""
        index = len(self.input_degrees)
        try:
            value = parse_joint_input(line, self.input_start[index])
        except ValueError:
            self._emit(f"输入无效：{line.strip()!r} 不是有效角度，请重新输入")
            self._prompt_joint()
            return
        self.input_degrees.append(value)
        if len(self.input_degrees) < len(JOINT_NAMES):
            self._prompt_joint()
            return
        self.input_stage = "speed"
        self._prompt_speed()

    def _prompt_speed(self):
        self._echo(f"速度百分比 1～100（当前 {self.move_speed}%，直接回车保持）：")

    def _submit_speed(self, line):
        """更新运动速度；格式或范围不对时提示并重新输入速度。"""
        try:
            self.move_speed = parse_speed_input(line, self.move_speed)
        except ValueError:
            self._emit(f"输入无效：{line.strip()!r} 不是 1～100 的整数，请重新输入")
            self._prompt_speed()
            return
        self.input_stage = "confirm"
        self._emit(f"目标关节角（度）：{format_degrees(self.input_degrees)}")
        deltas = [target - start for target, start in zip(self.input_degrees, self.input_start)]
        self._emit("相对当前变化（度）：" + "  ".join(
            f"{name}={delta:+.2f}" for name, delta in zip(JOINT_NAMES, deltas)))
        self._emit(f"速度：{self.move_speed}%")
        self._echo("执行本次运动？[y/n]：")

    def check_responses(self):
        """报告已完成或超时的请求；被急停取代的运动结果只提示不算成功。"""
        now = time.monotonic()
        for operation in ("stop", *MOTION_OPERATIONS):
            call = self.pending.get(operation)
            if call is None:
                continue
            label = OPERATION_LABELS[operation]
            if call.future.done():
                del self.pending[operation]
                try:
                    result = call.future.result()
                    if result is None:
                        raise RuntimeError("服务未返回结果")
                    success, message = bool(result.success), str(result.message)
                except Exception as error:
                    success, message = False, f"服务调用异常：{error}"
                if call.superseded:
                    self._emit(f"{label}：已被紧急停止取代（服务返回：{message}）")
                    continue
                self._emit(f"{label}：{'成功' if success else '失败'}：{message}")
                continue
            if now < call.deadline:
                continue
            del self.pending[operation]
            call.future.cancel()
            self._emit(f"{label}：等待服务响应超时，执行状态尚未确认")


def main(args=None):
    """在单独终端运行 ROS 节点，并保持逐键监听。"""
    if not sys.stdin.isatty():
        print("键盘控制需要交互终端；请在单独终端运行 "
              "ros2 run fastumi_arm_keyboard keyboard_control", file=sys.stderr)
        return 2

    rclpy.init(args=args)
    node = None
    try:
        node = ArmKeyboardControl()
        with TerminalKeyReader(sys.stdin) as reader:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
                node.check_responses()
                node.handle_key(reader.read_key(0.05))
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError) as error:
        print(f"键盘控制无法读取终端：{error}", file=sys.stderr)
        return 2
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
