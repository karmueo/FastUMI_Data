"""把 episode 相对策略目标安全转换为 RM75 基座下的连续笛卡尔透传。"""

from __future__ import annotations

from typing import Optional

from fastumi_data.pose_math import (
    interpolate_pose,
    matrix_to_pose,
    pose_to_matrix,
    relative_transform,
)
from geometry_msgs.msg import PoseStamped, TransformStamped
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Empty
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from fastumi_rm75.safety import SafetyGate, SafetyLimits


try:
    # 睿尔曼官方 ROS 2 驱动安装后提供该连续位姿透传消息。
    from rm_ros_interfaces.msg import Carteposcustom, Rmerr
except ImportError:
    Carteposcustom = None
    Rmerr = None


def _stamp_to_ns(stamp) -> int:
    """把 ROS Time 消息转换为纳秒；零时间戳返回 0。"""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _transform_message_to_matrix(message: TransformStamped) -> np.ndarray:
    """把 TF 消息转换为 4×4 齐次变换。"""
    translation = message.transform.translation
    rotation = message.transform.rotation
    return pose_to_matrix(
        np.asarray(
            [translation.x, translation.y, translation.z], dtype=np.float64
        ),
        np.asarray(
            [rotation.x, rotation.y, rotation.z, rotation.w],
            dtype=np.float64,
        ),
    )


def _fill_pose_message(transform: np.ndarray, message: PoseStamped) -> None:
    """把齐次变换写入 PoseStamped 的 pose 字段。"""
    position, quaternion = matrix_to_pose(transform)
    message.pose.position.x = float(position[0])
    message.pose.position.y = float(position[1])
    message.pose.position.z = float(position[2])
    message.pose.orientation.x = float(quaternion[0])
    message.pose.orientation.y = float(quaternion[1])
    message.pose.orientation.z = float(quaternion[2])
    message.pose.orientation.w = float(quaternion[3])


class Rm75PolicyBridge(Node):
    """执行相对坐标反变换、100 Hz 插值、安全检查和 RM75 透传。"""

    def __init__(self) -> None:
        """加载严格安全配置并创建 ROS 接口。"""
        super().__init__("rm75_policy_bridge")
        self._declare_parameters()
        self._dry_run = bool(self.get_parameter("dry_run").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._tcp_frame = str(self.get_parameter("tcp_frame").value)
        self._relative_frame = str(
            self.get_parameter("relative_frame").value
        )
        self._joint_names = [
            str(name) for name in self.get_parameter("joint_names").value
        ]
        if len(self._joint_names) != 7 or len(set(self._joint_names)) != 7:
            raise ValueError("joint_names 必须包含七个唯一 RM75 关节名")
        self._require_robot_error_state = bool(
            self.get_parameter("require_robot_error_state").value
        )
        policy_rate_hz = float(self.get_parameter("policy_rate_hz").value)
        servo_rate_hz = float(self.get_parameter("servo_rate_hz").value)
        if policy_rate_hz <= 0.0 or servo_rate_hz <= 0.0:
            raise ValueError("policy_rate_hz 和 servo_rate_hz 必须为正数")
        self._interpolation_duration_ns = int(1.0e9 / policy_rate_hz)
        limits = SafetyLimits(
            workspace_min_m=np.asarray(
                self.get_parameter("workspace_min_m").value,
                dtype=np.float64,
            ),
            workspace_max_m=np.asarray(
                self.get_parameter("workspace_max_m").value,
                dtype=np.float64,
            ),
            joint_min_rad=np.deg2rad(
                np.asarray(
                    self.get_parameter("joint_min_deg").value,
                    dtype=np.float64,
                )
            ),
            joint_max_rad=np.deg2rad(
                np.asarray(
                    self.get_parameter("joint_max_deg").value,
                    dtype=np.float64,
                )
            ),
            maximum_step_translation_m=float(
                self.get_parameter("maximum_step_translation_m").value
            ),
            maximum_step_rotation_rad=np.deg2rad(
                float(self.get_parameter("maximum_step_rotation_deg").value)
            ),
            state_timeout_s=float(
                self.get_parameter("state_timeout_s").value
            ),
            target_timeout_s=float(
                self.get_parameter("target_timeout_s").value
            ),
        )
        self._safety = SafetyGate(limits)
        if not self._dry_run and (
            Carteposcustom is None
            or (self._require_robot_error_state and Rmerr is None)
        ):
            raise RuntimeError(
                "非 dry-run 模式需要安装睿尔曼 rm_ros_interfaces"
            )

        self._tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._debug_publisher = self.create_publisher(
            PoseStamped, "/fastumi/rm75/commanded_pose", 10
        )
        self._relative_state_publisher = self.create_publisher(
            PoseStamped, "/fastumi/rm75/relative_tcp", 10
        )
        self._image_state_publisher = self.create_publisher(
            PoseStamped, "/fastumi/rm75/relative_tcp_at_image", 10
        )
        self._stop_publisher = self.create_publisher(
            Empty, "/rm_driver/move_stop_cmd", 10
        )
        self._rm_publisher = None
        if Carteposcustom is not None:
            self._rm_publisher = self.create_publisher(
                Carteposcustom,
                "/rm_driver/movep_canfd_custom_cmd",
                10,
            )
        self._target_subscription = self.create_subscription(
            PoseStamped,
            "/fastumi/policy/relative_target",
            self._target_callback,
            10,
        )
        self._joint_subscription = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 20
        )
        self._image_subscription = self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self._image_callback,
            qos_profile_sensor_data,
        )
        self._operator_estop_subscription = self.create_subscription(
            Bool,
            "/fastumi/rm75/operator_estop",
            self._operator_estop_callback,
            10,
        )
        self._robot_error_subscription = None
        if Rmerr is not None:
            self._robot_error_subscription = self.create_subscription(
                Rmerr,
                "/rm_driver/udp_rm_err",
                self._robot_error_callback,
                10,
            )
        self._enable_service = self.create_service(
            Trigger, "/fastumi/rm75/enable", self._enable
        )
        self._disable_service = self.create_service(
            Trigger, "/fastumi/rm75/disable", self._disable
        )
        self._emergency_stop_service = self.create_service(
            Trigger,
            "/fastumi/rm75/emergency_stop",
            self._emergency_stop,
        )
        self._timer = self.create_timer(
            1.0 / servo_rate_hz, self._servo_tick
        )

        # 运行状态和变换只在显式 enable 后建立。
        self._enabled = False
        self._enabled_at_ns: Optional[int] = None
        self._latest_joint_state_ns: Optional[int] = None
        self._latest_joint_state_safe = False
        self._latest_robot_error_ns: Optional[int] = None
        self._robot_error_clear: Optional[bool] = None
        self._latest_target_ns: Optional[int] = None
        self._episode_start: Optional[np.ndarray] = None
        self._previous_policy_target: Optional[np.ndarray] = None
        self._commanded_target: Optional[np.ndarray] = None
        self._interpolation_start: Optional[np.ndarray] = None
        self._interpolation_goal: Optional[np.ndarray] = None
        self._interpolation_start_ns: Optional[int] = None
        self.get_logger().info(
            f"RM75 策略桥已启动，dry_run={self._dry_run}，"
            f"基座 {self._base_frame}，TCP {self._tcp_frame}"
        )

    def _declare_parameters(self) -> None:
        """声明运行、坐标系和安全门限参数。"""
        self.declare_parameter("dry_run", True)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tcp_frame", "fastumi_tcp")
        self.declare_parameter("relative_frame", "episode_start_tcp")
        self.declare_parameter("policy_rate_hz", 20.0)
        self.declare_parameter("servo_rate_hz", 100.0)
        self.declare_parameter(
            "workspace_min_m", [0.05, -0.70, 0.05]
        )
        self.declare_parameter(
            "workspace_max_m", [0.90, 0.70, 1.00]
        )
        self.declare_parameter(
            "joint_names",
            [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
            ],
        )
        self.declare_parameter(
            "joint_min_deg",
            [-178.0, -130.0, -178.0, -135.0, -178.0, -128.0, -360.0],
        )
        self.declare_parameter(
            "joint_max_deg",
            [178.0, 130.0, 178.0, 135.0, 178.0, 128.0, 360.0],
        )
        self.declare_parameter("maximum_step_translation_m", 0.02)
        self.declare_parameter("maximum_step_rotation_deg", 10.0)
        self.declare_parameter("state_timeout_s", 0.10)
        self.declare_parameter("target_timeout_s", 0.20)
        self.declare_parameter("require_robot_error_state", True)
        self.declare_parameter(
            "image_topic",
            "/xv_sdk/SN250801DR48FB26001253/rgb/image",
        )

    def _lookup_tcp(self, timestamp: Optional[Time] = None) -> np.ndarray:
        """查询 URDF FK 在指定时间或最新时刻生成的 TCP。"""
        transform = self._tf_buffer.lookup_transform(
            self._base_frame,
            self._tcp_frame,
            timestamp or Time(),
            timeout=Duration(seconds=0.05),
        )
        return _transform_message_to_matrix(transform)

    def _joint_state_callback(self, message: JointState) -> None:
        """记录带时间戳关节状态并实时检查七轴机械限位。"""
        timestamp_ns = _stamp_to_ns(message.header.stamp)
        self._latest_joint_state_ns = (
            timestamp_ns
            if timestamp_ns > 0
            else self.get_clock().now().nanoseconds
        )
        position_by_name = dict(zip(message.name, message.position))
        if all(name in position_by_name for name in self._joint_names):
            positions = np.asarray(
                [position_by_name[name] for name in self._joint_names],
                dtype=np.float64,
            )
        elif len(message.position) == 7 and not message.name:
            positions = np.asarray(message.position, dtype=np.float64)
        else:
            self._latest_joint_state_safe = False
            if self._enabled:
                self._stop_motion("joint_states 缺少配置的 RM75 七个关节")
            return
        unsafe_reason = self._safety.validate_joint_positions(positions)
        self._latest_joint_state_safe = unsafe_reason is None
        if unsafe_reason is not None and self._enabled:
            self._stop_motion(unsafe_reason)

    def _robot_error_callback(self, message) -> None:
        """监控官方 UDP 错误列表，任一非零错误立即停止轨迹。"""
        errors = [int(value) for value in message.err]
        declared_length = int(message.err_len)
        self._latest_robot_error_ns = self.get_clock().now().nanoseconds
        self._robot_error_clear = (
            declared_length == 0 and not any(value != 0 for value in errors)
        )
        if self._enabled and not self._robot_error_clear:
            self._stop_motion(f"RM75 驱动报告错误: {errors}")

    def _operator_estop_callback(self, message: Bool) -> None:
        """响应人工急停布尔话题。"""
        if message.data:
            self._stop_motion("人工急停话题触发", force_stop=True)

    def _image_callback(self, message: Image) -> None:
        """发布图像原始时间戳对应的真实 TCP 相对状态。"""
        if not self._enabled or self._episode_start is None:
            return
        try:
            current = self._lookup_tcp(Time.from_msg(message.header.stamp))
        except TransformException as error:
            self.get_logger().warning(
                f"图像时刻 FK 不可用: {error}",
                throttle_duration_sec=1.0,
            )
            return
        relative = relative_transform(self._episode_start, current)
        state_message = PoseStamped()
        state_message.header.stamp = message.header.stamp
        state_message.header.frame_id = self._relative_frame
        _fill_pose_message(relative, state_message)
        self._image_state_publisher.publish(state_message)

    def _target_callback(self, message: PoseStamped) -> None:
        """接收 episode 相对目标，转换到 RM75 基座并执行安全检查。"""
        if not self._enabled or self._episode_start is None:
            self.get_logger().warning(
                "忽略策略目标：RM75 桥尚未 enable",
                throttle_duration_sec=1.0,
            )
            return
        if message.header.frame_id not in ("", self._relative_frame):
            self._stop_motion(
                f"策略目标 frame_id 必须为 {self._relative_frame}"
            )
            return
        relative_target = pose_to_matrix(
            np.asarray(
                [
                    message.pose.position.x,
                    message.pose.position.y,
                    message.pose.position.z,
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    message.pose.orientation.x,
                    message.pose.orientation.y,
                    message.pose.orientation.z,
                    message.pose.orientation.w,
                ],
                dtype=np.float64,
            ),
        )
        absolute_target = self._episode_start @ relative_target
        previous = (
            self._previous_policy_target
            if self._previous_policy_target is not None
            else self._episode_start
        )
        unsafe_reason = self._safety.validate_target(
            absolute_target, previous
        )
        if unsafe_reason is not None:
            self._stop_motion(unsafe_reason)
            return
        now_ns = self.get_clock().now().nanoseconds
        source_timestamp_ns = _stamp_to_ns(message.header.stamp)
        self._latest_target_ns = source_timestamp_ns or now_ns
        self._interpolation_start = (
            self._commanded_target.copy()
            if self._commanded_target is not None
            else previous.copy()
        )
        self._interpolation_goal = absolute_target
        self._interpolation_start_ns = now_ns
        self._previous_policy_target = absolute_target

    def _enable(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """记录当前 RM75 TCP 为 episode 起点并允许发送命令。"""
        del request
        now_ns = self.get_clock().now().nanoseconds
        if self._latest_joint_state_ns is None or (
            now_ns - self._latest_joint_state_ns
        ) / 1.0e9 > self._safety.limits.state_timeout_s:
            response.success = False
            response.message = "joint_states 不可用或已过期"
            return response
        if not self._latest_joint_state_safe:
            response.success = False
            response.message = "joint_states 缺失或已超出配置关节限位"
            return response
        if not self._dry_run and self._require_robot_error_state:
            robot_status_age_s = (
                float("inf")
                if self._latest_robot_error_ns is None
                else (now_ns - self._latest_robot_error_ns) / 1.0e9
            )
            if (
                self._robot_error_clear is not True
                or robot_status_age_s > self._safety.limits.state_timeout_s
            ):
                response.success = False
                response.message = "RM75 错误状态不可用、已过期或存在错误"
                return response
        try:
            current_tcp = self._lookup_tcp()
        except TransformException as error:
            response.success = False
            response.message = f"无法查询 RM75 TCP: {error}"
            return response
        self._episode_start = current_tcp
        self._commanded_target = current_tcp.copy()
        self._previous_policy_target = current_tcp.copy()
        self._interpolation_start = current_tcp.copy()
        self._interpolation_goal = current_tcp.copy()
        self._interpolation_start_ns = now_ns
        self._enabled_at_ns = now_ns
        self._latest_target_ns = None
        self._enabled = True
        response.success = True
        response.message = "RM75 策略桥已启用并记录起始 TCP"
        self.get_logger().warning(response.message)
        return response

    def _disable(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """停止轨迹并关闭策略桥。"""
        del request
        self._stop_motion("操作员禁用")
        response.success = True
        response.message = "RM75 策略桥已禁用"
        return response

    def _emergency_stop(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """立即触发官方轨迹停止接口并禁用策略桥。"""
        del request
        self._stop_motion("人工急停服务触发", force_stop=True)
        response.success = True
        response.message = "已触发 RM75 轨迹停止"
        return response

    def _stop_motion(self, reason: str, force_stop: bool = False) -> None:
        """停止 RM75 轨迹、禁用桥接器并记录原因。"""
        if not self._dry_run and (self._enabled or force_stop):
            self._stop_publisher.publish(Empty())
        self._enabled = False
        self.get_logger().error(f"RM75 策略桥停止: {reason}")

    def _publish_command(self, transform: np.ndarray) -> None:
        """发布可视化命令，并在非 dry-run 时发送 RM75 CANFD 目标。"""
        debug_message = PoseStamped()
        debug_message.header.stamp = self.get_clock().now().to_msg()
        debug_message.header.frame_id = self._base_frame
        _fill_pose_message(transform, debug_message)
        self._debug_publisher.publish(debug_message)
        if self._dry_run:
            return
        command = Carteposcustom()
        command.pose = debug_message.pose
        command.follow = True
        command.trajectory_mode = 0
        command.radio = 0
        self._rm_publisher.publish(command)

    def _publish_relative_state(self) -> None:
        """发布当前 RM75 TCP 在 episode 起始坐标系中的相对位姿。"""
        if self._episode_start is None:
            return
        try:
            current = self._lookup_tcp()
        except TransformException:
            return
        relative = relative_transform(self._episode_start, current)
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._relative_frame
        _fill_pose_message(relative, message)
        self._relative_state_publisher.publish(message)

    def _servo_tick(self) -> None:
        """执行 watchdog、SE(3) 插值和 100 Hz 命令发布。"""
        if not self._enabled:
            return
        now_ns = self.get_clock().now().nanoseconds
        if self._latest_target_ns is None:
            if (
                self._enabled_at_ns is not None
                and (now_ns - self._enabled_at_ns) / 1.0e9
                > self._safety.limits.target_timeout_s
            ):
                self._stop_motion("启用后未及时收到策略目标")
            return
        stale_reason = self._safety.stale_reason(
            now_ns, self._latest_joint_state_ns, self._latest_target_ns
        )
        if stale_reason is not None:
            self._stop_motion(stale_reason)
            return
        if not self._latest_joint_state_safe:
            self._stop_motion("RM75 关节状态越界或不完整")
            return
        if not self._dry_run and self._require_robot_error_state:
            robot_status_age_s = (
                float("inf")
                if self._latest_robot_error_ns is None
                else (now_ns - self._latest_robot_error_ns) / 1.0e9
            )
            if (
                self._robot_error_clear is not True
                or robot_status_age_s > self._safety.limits.state_timeout_s
            ):
                self._stop_motion("RM75 驱动错误状态异常或已过期")
                return
        if (
            self._interpolation_start is None
            or self._interpolation_goal is None
            or self._interpolation_start_ns is None
        ):
            self._stop_motion("内部插值状态不完整")
            return
        ratio = (now_ns - self._interpolation_start_ns) / (
            self._interpolation_duration_ns
        )
        start_position, start_quaternion = matrix_to_pose(
            self._interpolation_start
        )
        goal_position, goal_quaternion = matrix_to_pose(
            self._interpolation_goal
        )
        position, quaternion = interpolate_pose(
            start_position,
            start_quaternion,
            goal_position,
            goal_quaternion,
            ratio,
        )
        self._commanded_target = pose_to_matrix(position, quaternion)
        self._publish_command(self._commanded_target)
        self._publish_relative_state()


def main(args: Optional[list[str]] = None) -> None:
    """运行 RM75 策略桥节点。"""
    rclpy.init(args=args)
    node = Rm75PolicyBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node._enabled:
                node._stop_motion("节点退出")
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            # ros2 launch 与外层进程管理器可能同时转发 SIGINT。
            pass


if __name__ == "__main__":
    main()
