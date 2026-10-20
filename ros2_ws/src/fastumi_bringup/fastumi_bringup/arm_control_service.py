"""提供 RM75 轨迹急停、回到初始关节位姿和指定关节角 MoveJ 服务。"""

import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from ament_index_python.packages import get_package_share_directory
from fastumi_interfaces.srv import MoveJoints
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rm_ros_interfaces.msg import Movej
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty
from std_srvs.srv import Trigger

from fastumi_bringup.hand_eye.kinematics import ArmKinematics
from fastumi_bringup.initial_pose import INITIAL_JOINT_POSITIONS
from fastumi_bringup.keyboard_home import (
    ALREADY_AT_TARGET_TOLERANCE, FEEDBACK_FRESHNESS_SECONDS, JOINT_NAMES,
)


class ArmControlService(Node):
    """急停服务随时可用；回位与关节运动互斥，阻塞等待 MoveJ 结果，可被急停中断。"""

    def __init__(self):
        super().__init__("fastumi_arm_control")
        self.declare_parameter("emergency_stop_service", "/fastumi/arm/emergency_stop")
        self.declare_parameter("return_home_service", "/fastumi/arm/return_home")
        self.declare_parameter("move_joints_service", "/fastumi/arm/move_joints")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("home_speed", 20)
        self.declare_parameter("home_timeout", 120.0)
        self.declare_parameter("stop_timeout", 2.0)
        self.home_speed = int(self.get_parameter("home_speed").value)
        self.home_timeout = float(self.get_parameter("home_timeout").value)
        self.stop_timeout = float(self.get_parameter("stop_timeout").value)
        if not 1 <= self.home_speed <= 100:
            raise ValueError("home_speed 必须在 1～100 之间（百分比）")
        if self.home_timeout <= 0.0 or self.stop_timeout <= 0.0:
            raise ValueError("home_timeout 与 stop_timeout 必须为正数（秒）")
        urdf_path = self.get_parameter("urdf_path").value
        if not urdf_path:
            try:
                urdf_path = str(Path(get_package_share_directory("fastumi_bringup"))
                                / "assets/rm_75_kinematic.urdf")
            except LookupError as error:
                raise ValueError("未找到已安装的 fastumi_bringup；请构建该包或设置 urdf_path") from error
        # 仅用 URDF 的关节限位（弧度）在下发前检查目标角。
        self.kinematics = ArmKinematics.from_urdf(urdf_path)

        self.lock = threading.Lock()
        self.current_positions = None
        self.last_feedback_at = None
        self.last_valid_feedback_at = None
        self.motion_call = None
        self.stop_waiters = []
        self.last_stop_success = True

        topics = MutuallyExclusiveCallbackGroup()
        services = ReentrantCallbackGroup()
        self.movej_publisher = self.create_publisher(Movej, "/rm_driver/movej_cmd", 10)
        self.stop_publisher = self.create_publisher(Empty, "/rm_driver/move_stop_cmd", 10)
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10,
            callback_group=topics)
        self.create_subscription(
            Bool, "/rm_driver/udp_feedback_valid", self._feedback_valid_callback, 10,
            callback_group=topics)
        self.create_subscription(
            Bool, "/rm_driver/movej_result", self._movej_result_callback, 10,
            callback_group=topics)
        self.create_subscription(
            Bool, "/rm_driver/move_stop_result", self._stop_result_callback, 10,
            callback_group=topics)
        self.create_service(
            Trigger, self.get_parameter("emergency_stop_service").value,
            self._emergency_stop, callback_group=services)
        self.create_service(
            Trigger, self.get_parameter("return_home_service").value,
            self._return_home, callback_group=services)
        self.create_service(
            MoveJoints, self.get_parameter("move_joints_service").value,
            self._move_joints, callback_group=services)
        self.get_logger().info("RM75 急停、回位与关节运动服务已就绪")

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
        with self.lock:
            self.current_positions = positions
            self.last_feedback_at = time.monotonic()

    def _feedback_valid_callback(self, message):
        """仅接受当前仍有效的驱动 UDP 反馈。"""
        with self.lock:
            self.last_valid_feedback_at = time.monotonic() if message.data else None

    def _movej_result_callback(self, message):
        """消费当前运动的唯一结果；中断后仍保留归属，直到迟到结果收到。"""
        with self.lock:
            call = self.motion_call
            if call is None or call.result_received:
                return
            call.result_received = True
            if not call.done.is_set():
                call.success = bool(message.data)
                call.message = call.done_message if message.data else "MoveJ 执行失败，请检查机械臂"
                call.done.set()
            if call.returned:
                self.motion_call = None

    def _stop_result_callback(self, message):
        """按下发顺序消费停止结果，全部确认前不允许开始新运动。"""
        with self.lock:
            if not self.stop_waiters:
                return
            waiter = self.stop_waiters.pop(0)
            waiter.success = bool(message.data)
            self.last_stop_success = waiter.success
            waiter.done.set()

    def _request_stop(self):
        """持有 lock 时发布停止命令，保留等待记录直到驱动回报结果。"""
        waiter = SimpleNamespace(done=threading.Event(), success=False)
        self.stop_waiters.append(waiter)
        self.stop_publisher.publish(Empty())
        return waiter

    def _not_ready_reason(self, now):
        """返回回位前置条件不满足的原因；就绪时返回 None。"""
        if (self.current_positions is None or self.last_feedback_at is None or
                now - self.last_feedback_at > FEEDBACK_FRESHNESS_SECONDS):
            return "未收到新鲜的 /joint_states 反馈"
        if (self.last_valid_feedback_at is None or
                now - self.last_valid_feedback_at > FEEDBACK_FRESHNESS_SECONDS):
            return "rm_driver UDP 反馈无效"
        if self.movej_publisher.get_subscription_count() == 0:
            return "rm_driver 未订阅 /rm_driver/movej_cmd"
        return None

    def _emergency_stop(self, request, response):
        """立即发送轨迹停止并中断进行中的运动，再等待驱动确认。"""
        del request
        with self.lock:
            driver_listening = self.stop_publisher.get_subscription_count() > 0
            waiter = self._request_stop()
            call = self.motion_call
            if call is not None and not call.done.is_set():
                call.success = False
                call.message = "运动被急停中断"
                call.done.set()
        self.get_logger().error("急停服务触发：已发送 RM75 轨迹停止")

        if not waiter.done.wait(self.stop_timeout):
            # 超时不删除记录，防止迟到的停止结果确认下一条停止命令。
            response.success = False
            response.message = ("已发送轨迹停止，但 rm_driver 未订阅停止话题" if not driver_listening
                                else "已发送轨迹停止，停止结果未确认")
            return response
        response.success = waiter.success
        response.message = "RM75 轨迹已停止" if waiter.success else "rm_driver 报告轨迹停止失败"
        return response

    def _return_home(self, request, response):
        """以 home_speed 回到初始关节位姿。"""
        del request
        return self._execute_movej(
            INITIAL_JOINT_POSITIONS, self.home_speed, "RM75 已回到初始位姿", response)

    def _move_joints(self, request, response):
        """校验度数目标与速度后执行 MoveJ；任一校验失败都不下发命令。"""
        degrees = list(request.joint_degrees)
        if len(degrees) != len(JOINT_NAMES):
            response.success = False
            response.message = f"joint_degrees 需要 {len(JOINT_NAMES)} 个值，收到 {len(degrees)} 个"
            return response
        if not all(math.isfinite(value) for value in degrees):
            response.success = False
            response.message = "joint_degrees 含 NaN 或无穷值"
            return response
        if not 1 <= request.speed <= 100:
            response.success = False
            response.message = f"speed 必须在 1～100 之间（百分比），收到 {request.speed}"
            return response
        target = [math.radians(value) for value in degrees]
        for name, value, lower, upper in zip(
                JOINT_NAMES, degrees, self.kinematics.lower, self.kinematics.upper):
            if not math.radians(value) >= lower or not math.radians(value) <= upper:
                response.success = False
                response.message = (
                    f"{name}={value:.2f}° 超出限位 "
                    f"[{math.degrees(lower):.2f}°, {math.degrees(upper):.2f}°]")
                return response
        return self._execute_movej(
            target, int(request.speed), "RM75 已到达目标关节角", response)

    def _execute_movej(self, target, speed, done_message, response):
        """就绪时发送阻塞 MoveJ（目标为弧度），等待结果、急停中断或超时。"""
        with self.lock:
            if self.motion_call is not None:
                response.success = False
                response.message = "上一次运动尚未结束或 MoveJ 结果未收到，忽略新请求"
                return response
            if self.stop_waiters or not self.last_stop_success:
                response.success = False
                response.message = "轨迹停止尚未确认成功，忽略新请求；请检查机械臂和驱动"
                return response
            now = time.monotonic()
            reason = self._not_ready_reason(now)
            if reason is not None:
                response.success = False
                response.message = f"机械臂未就绪，未发送运动命令：{reason}"
                return response
            maximum_error = max(
                abs(current - goal) for current, goal in zip(self.current_positions, target))
            if maximum_error <= ALREADY_AT_TARGET_TOLERANCE:
                response.success = True
                response.message = "RM75 已在目标关节角，无需移动"
                return response

            command = Movej()
            command.joint = [float(value) for value in target]
            command.speed = speed
            command.block = True
            command.trajectory_connect = 0
            command.dof = 7
            call = SimpleNamespace(
                done=threading.Event(), success=False, message="", done_message=done_message,
                result_received=False, returned=False)
            self.motion_call = call
            self.movej_publisher.publish(command)
        self.get_logger().warning(
            "以 {}% 速度执行 MoveJ，目标 [{}] 度".format(
                speed, ", ".join(f"{math.degrees(value):.1f}" for value in target)))

        call.done.wait(self.home_timeout)
        with self.lock:
            # 等待超时与回调可能同时发生，以锁内状态为准。
            finished = call.done.is_set()
            if not finished:
                call.success = False
                call.message = "等待 MoveJ 结果超时，已发送轨迹停止"
                call.done.set()
                self._request_stop()
            call.returned = True
            if call.result_received and self.motion_call is call:
                self.motion_call = None
        if not finished:
            self.get_logger().error(call.message)
        response.success = call.success
        response.message = call.message
        return response


def main(args=None):
    """用多线程执行器运行服务，保证回位等待期间急停仍可响应。"""
    rclpy.init(args=args)
    node = None
    try:
        node = ArmControlService()
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
