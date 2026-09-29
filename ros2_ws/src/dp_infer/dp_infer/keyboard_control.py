"""在交互终端中通过按键调用 DP 任务控制与推理配置服务。"""

from collections import deque
from dataclasses import dataclass
import os
import select
import sys
import termios
import time
import tty

import rclpy
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node
from std_srvs.srv import Trigger

from fastumi_interfaces.srv import SetNumInferenceSteps


SERVICE_NAMES = {
    "start": "/fastumi/policy/start_task",
    "stop": "/fastumi/policy/stop_task",
    "home": "/fastumi/policy/return_to_start",
}
OPERATION_LABELS = {"start": "开始任务", "stop": "停止任务", "home": "回到初始状态"}
KEY_OPERATIONS = {" ": "home", "\r": "start", "\n": "start", "\x7f": "stop", "\b": "stop"}
RESPONSE_TIMEOUT_SECONDS = {"start": 5.0, "stop": 5.0, "home": 170.0}
STEPS_RESPONSE_TIMEOUT_SECONDS = 5.0
STEP_KEYS = {"\x1b[C": "right", "\x1bOC": "right",
             "\x1b[D": "left", "\x1bOD": "left"}


class TerminalKeyReader:
    """逐字节读取终端，退出时恢复原有终端设置。"""

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
            termios.tcsetattr(self.stream.fileno(), termios.TCSADRAIN,
                              self.original_settings)
            self.original_settings = None

    def read_key(self, timeout_seconds):
        """读取普通键或完整的三字节方向键序列。"""
        fd = self.stream.fileno()
        readable, _, _ = select.select([fd], [], [], timeout_seconds)
        if not readable:
            return None
        data = os.read(fd, 1)
        if not data:
            raise RuntimeError("交互终端已关闭")
        if data == b"\x1b":
            for _ in range(2):
                more, _, _ = select.select([fd], [], [], 0.03)
                if not more:
                    break
                fragment = os.read(fd, 1)
                if not fragment:
                    raise RuntimeError("交互终端已关闭")
                data += fragment
        return data.decode("ascii", errors="ignore")


@dataclass
class PendingCall:
    """记录一个服务请求及其本地等待期限。"""

    future: object
    deadline: float
    superseded: bool = False


@dataclass
class PendingStepsCall:
    """记录启动读取、设置前读取、设置调用或设置后读取。"""

    phase: str
    future: object
    deadline: float
    direction: str = ""
    target: int = 0
    success: bool = False
    message: str = ""


class KeyboardControl(Node):
    """异步调用任务与配置服务，使停止键保持可响应。"""

    def __init__(self):
        super().__init__("dp_keyboard_control")
        self.declare_parameter("inference_parameter_node", "/dp_infer")
        self.declare_parameter(
            "set_inference_steps_service", "/fastumi/policy/set_inference_steps")
        self.task_clients = {
            operation: self.create_client(Trigger, name)
            for operation, name in SERVICE_NAMES.items()
        }
        parameter_node = self.get_parameter("inference_parameter_node").value.rstrip("/")
        self.steps_parameter_client = self.create_client(
            GetParameters, f"{parameter_node}/get_parameters")
        self.steps_service_client = self.create_client(
            SetNumInferenceSteps,
            self.get_parameter("set_inference_steps_service").value)
        self.pending = {}
        self.motion_locked = False
        self.step_directions = deque()
        self.steps_call = None
        self.startup_steps_pending = True
        self.startup_steps_deadline = time.monotonic() + STEPS_RESPONSE_TIMEOUT_SECONDS
        self._emit("键盘控制已就绪：回车开始任务，Backspace 停止任务，空格回到初始状态，"
                   "左右方向键调整去噪步数；Ctrl+C 退出")

    @staticmethod
    def _emit(message):
        print(message, flush=True)

    def handle_key(self, key):
        """任务键调用任务服务；方向键进入独立的步数请求队列。"""
        direction = STEP_KEYS.get(key)
        if direction is not None:
            self.step_directions.append(direction)
            self._pump_steps()
            return
        operation = KEY_OPERATIONS.get(key)
        if operation is None:
            return
        if operation == "stop" and self.step_directions:
            count = len(self.step_directions)
            self.step_directions.clear()
            self._emit(f"停止任务：已清除 {count} 个尚未开始的方向键请求")
        label = OPERATION_LABELS[operation]
        if operation in self.pending:
            self._emit(f"{label}：请求仍在执行，忽略重复按键")
            return
        if operation != "stop":
            if self.motion_locked:
                self._emit(f"{label}：上次回位或停机状态未确认，请先按 Backspace 停止任务")
                return
            if self.pending:
                self._emit(f"{label}：已有请求正在执行，请等待结果或按 Backspace 停止")
                return

        client = self.task_clients[operation]
        if not client.service_is_ready():
            self._emit(f"{label}：服务 {SERVICE_NAMES[operation]} 不可用，未发送请求")
            return
        try:
            future = client.call_async(Trigger.Request())
        except Exception as error:
            self._emit(f"{label}：服务调用失败：{error}")
            return

        if operation == "stop":
            for other_operation, call in self.pending.items():
                if other_operation != "stop":
                    call.superseded = True
        self.pending[operation] = PendingCall(
            future, time.monotonic() + RESPONSE_TIMEOUT_SECONDS[operation]
        )
        self._emit(f"{label}：已发送请求，等待服务执行结果…")

    def _request_steps(self, phase, direction="", target=0, success=False, message=""):
        """异步发起一次参数读取或设置请求，返回是否已发出。"""
        client = (self.steps_service_client if phase == "set"
                  else self.steps_parameter_client)
        if not client.service_is_ready():
            return False
        request = (SetNumInferenceSteps.Request(num_inference_steps=target)
                   if phase == "set" else GetParameters.Request(names=["num_inference_steps"]))
        try:
            future = client.call_async(request)
        except Exception as error:
            self._emit(f"去噪步数：服务调用异常：{error}")
            return False
        self.steps_call = PendingStepsCall(
            phase, future, time.monotonic() + STEPS_RESPONSE_TIMEOUT_SECONDS,
            direction, target, success, message)
        return True

    def _pump_steps(self):
        """只启动一个待处理请求；每个方向键都重新读取参数。"""
        if self.steps_call is not None:
            return
        if self.startup_steps_pending:
            if self._request_steps("startup"):
                self.startup_steps_pending = False
            elif time.monotonic() >= self.startup_steps_deadline:
                self.startup_steps_pending = False
                self._emit("当前 num_inference_steps：未确认（参数服务不可用）")
            return
        if not self.step_directions:
            return
        direction = self.step_directions.popleft()
        if not self._request_steps("before", direction=direction):
            self._emit("调整去噪步数：失败，当前值未确认；未发送设置请求")

    @staticmethod
    def _steps_from_response(result):
        """只接受推理节点返回的单个整数参数值。"""
        if result is None or len(result.values) != 1:
            raise ValueError("参数服务未返回 num_inference_steps")
        value = result.values[0]
        if value.type != ParameterType.PARAMETER_INTEGER or not 1 <= value.integer_value <= 50:
            raise ValueError("num_inference_steps 不是有效的 1～50 整数")
        return value.integer_value

    def _finish_steps(self, call, actual=None):
        """显示设置服务结果和重新读取的权威参数值。"""
        current = (f"当前 num_inference_steps={actual}" if actual is not None
                   else "当前值未确认")
        mismatch = ("；读回值与目标不同" if call.success and actual is not None
                    and actual != call.target else "")
        self._emit(f"调整去噪步数：{'成功' if call.success else '失败'}"
                   f"（success={call.success}，目标={call.target}，{current}）"
                   f"：{call.message}{mismatch}")

    def _handle_steps_result(self, call, result):
        """按读取、设置、复读的顺序推进一个方向键请求。"""
        if call.phase == "startup":
            actual = self._steps_from_response(result)
            self._emit(f"当前 num_inference_steps={actual}")
            return
        if call.phase == "before":
            actual = self._steps_from_response(result)
            target = min(32, max(2, actual * 2 if call.direction == "right"
                                  else actual // 2))
            if not self._request_steps("set", call.direction, target):
                message = "设置服务不可用或请求发送失败"
                if not self._request_steps("after", call.direction, target,
                                           False, message):
                    self._finish_steps(PendingStepsCall(
                        "after", None, 0, call.direction, target, False, message))
            return
        if call.phase == "set":
            success = bool(result.success) if result is not None else False
            message = str(result.message) if result is not None else "服务未返回结果"
            if not self._request_steps("after", call.direction, call.target,
                                       success, message):
                self._finish_steps(PendingStepsCall(
                    "after", None, 0, call.direction, call.target,
                    success, message))
            return
        actual = self._steps_from_response(result)
        self._finish_steps(call, actual)

    def _check_steps_response(self):
        """处理步数请求的答复与超时，不阻塞任务停止服务。"""
        call = self.steps_call
        if call is None:
            self._pump_steps()
            return
        if not call.future.done() and time.monotonic() < call.deadline:
            return
        self.steps_call = None
        if not call.future.done():
            call.future.cancel()
            error = "等待服务响应超时"
        else:
            try:
                self._handle_steps_result(call, call.future.result())
                self._pump_steps()
                return
            except Exception as exception:
                error = str(exception)
        if call.phase == "startup":
            self._emit(f"当前 num_inference_steps：未确认（{error}）")
        elif call.phase == "before":
            self._emit(f"调整去噪步数：失败，当前值未确认；未发送设置请求（{error}）")
        elif call.phase == "set":
            if not self._request_steps("after", call.direction, call.target,
                                       False, error):
                self._finish_steps(PendingStepsCall(
                    "after", None, 0, call.direction, call.target, False, error))
        else:
            call.message = f"{call.message}；读回失败：{error}"
            self._finish_steps(call, None)
        self._pump_steps()

    def check_responses(self):
        """报告已完成或超时的请求，忽略被停止请求取代的旧结果。"""
        now = time.monotonic()
        # 停止结果优先处理，避免同一轮到达的旧回位结果显示为当前成功。
        operations = ("stop", "start", "home")
        for operation in operations:
            call = self.pending.get(operation)
            if call is None:
                continue
            label = OPERATION_LABELS[operation]
            if call.future.done():
                del self.pending[operation]
                if call.superseded:
                    self._emit(f"{label}：旧请求已被停止请求取代，忽略迟到结果")
                    continue
                try:
                    result = call.future.result()
                    if result is None:
                        raise RuntimeError("服务未返回结果")
                    success, message = bool(result.success), str(result.message)
                except Exception as error:
                    success, message = False, f"服务调用异常：{error}"
                self._emit(f"{label}：{'成功' if success else '失败'}（success={success}）：{message}")
                if operation == "stop":
                    self.motion_locked = not success
                    if success:
                        for old_operation, old_call in list(self.pending.items()):
                            if old_operation != "stop" and old_call.superseded:
                                old_call.future.cancel()
                                del self.pending[old_operation]
                continue
            if now < call.deadline:
                continue
            del self.pending[operation]
            call.future.cancel()
            if call.superseded:
                self._emit(f"{label}：旧请求已被停止请求取代")
                continue
            self._emit(f"{label}：等待服务响应超时，执行状态尚未确认")
            if operation in ("home", "stop"):
                self.motion_locked = True
                self._emit("已禁止开始与回位请求；请按 Backspace 请求停止任务并确认结果")
        self._check_steps_response()


def main(args=None):
    """在单独终端运行 ROS 节点，并保持逐键监听。"""
    if not sys.stdin.isatty():
        print("键盘控制需要交互终端；请在单独终端运行 ros2 run dp_infer keyboard_control",
              file=sys.stderr)
        return 2

    rclpy.init(args=args)
    node = None
    try:
        node = KeyboardControl()
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
