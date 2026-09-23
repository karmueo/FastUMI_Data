"""实现 VIVE Tracker 输入到 RM75 CANFD 关节指令的 ROS 2 控制节点。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Protocol

from fastumi_interfaces.msg import GripperState, TrackerStatus
from fastumi_interfaces.srv import GetTeleopGeneration, SetTeleopGeneration
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rm_ros_interfaces.msg import Jointpos, Movej
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float32, String
from std_srvs.srv import SetBool, Trigger

from tracker_teleoperated.calibration_store import (
    CalibrationStoreError,
    load_workspace_calibration,
    resolve_workspace_calibration_path,
    save_workspace_calibration,
)
from tracker_teleoperated.core import (
    PoseStreamValidator,
    advance_joint_command,
    axis_mapping_from_rpy,
    low_pass_pose,
    map_tracker_target,
    mapping_basis_from_reference,
    parse_home_joint_positions,
    pose_to_matrix,
    validate_scale,
    validate_transform,
    workspace_mapping_from_samples,
)
from tracker_teleoperated.gripper_follow import GripperFollower
from tracker_teleoperated.kinematics import (
    JOINT_NAMES,
    PlacoRm75Kinematics,
    load_joint_velocity_limits,
    resolve_rm75_urdf,
)
from tracker_teleoperated.teleop_generation import TeleopGenerationStore


class Kinematics(Protocol):
    """定义控制节点依赖的最小运动学接口。"""

    def end_effector_pose(self, positions: np.ndarray) -> np.ndarray:
        """计算七轴关节角对应的末端位姿。"""
        ...

    def solve(
        self, target_pose: np.ndarray, seed_positions: np.ndarray
    ) -> np.ndarray:
        """求解末端目标对应的七轴关节角。"""
        ...


def stamp_to_ns(stamp) -> int:
    """将 ROS 时间消息转换为整数纳秒。"""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def reorder_joint_state(message: JointState) -> np.ndarray:
    """按 joint1 至 joint7 重排关节反馈并拒绝不完整输入。"""
    if len(message.name) != len(message.position):
        raise ValueError("关节名称和位置数量不一致")
    positions_by_name = dict(zip(message.name, message.position))
    missing = [name for name in JOINT_NAMES if name not in positions_by_name]
    if missing:
        raise ValueError(f"关节反馈缺少: {', '.join(missing)}")
    positions = np.asarray(
        [positions_by_name[name] for name in JOINT_NAMES], dtype=np.float64
    )
    if not np.all(np.isfinite(positions)):
        raise ValueError("关节反馈包含非有限数值")
    return positions


def build_joint_command(positions: np.ndarray) -> Jointpos:
    """构造 RM75 七轴低跟随 CANFD 指令。"""
    joints = np.asarray(positions, dtype=np.float64)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise ValueError("CANFD 指令必须包含七个有限关节角")
    message = Jointpos()
    message.joint = joints.astype(np.float32).tolist()
    message.follow = False
    message.expand = 0.0
    message.dof = 7
    return message


def build_home_command(positions: np.ndarray, speed_percent: int) -> Movej:
    """构造 RM75 七轴阻塞 MoveJ 回位指令。"""
    joints = np.asarray(positions, dtype=np.float64)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise ValueError("MoveJ 回位指令必须包含七个有限关节角")
    speed = int(speed_percent)
    if speed != speed_percent or not 1 <= speed <= 100:
        raise ValueError("MoveJ 回位速度必须是 [1, 100] 范围内的整数")
    message = Movej()
    message.joint = joints.astype(np.float32).tolist()
    message.speed = speed
    message.block = True
    message.trajectory_connect = 0
    message.dof = 7
    return message


def fill_pose_message(transform: np.ndarray, message: PoseStamped) -> None:
    """将 4×4 齐次变换写入 PoseStamped。"""
    translation = transform[:3, 3]
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    message.pose.position.x = float(translation[0])
    message.pose.position.y = float(translation[1])
    message.pose.position.z = float(translation[2])
    message.pose.orientation.x = float(quaternion[0])
    message.pose.orientation.y = float(quaternion[1])
    message.pose.orientation.z = float(quaternion[2])
    message.pose.orientation.w = float(quaternion[3])


class TrackerTeleopNode(Node):
    """管理遥操状态、安全门限、逆运动学和 RM75 指令发布。"""

    def __init__(self, kinematics: Optional[Kinematics] = None) -> None:
        """加载参数、创建 ROS 接口并保持初始暂停状态。"""
        super().__init__("tracker_teleop")
        self._declare_parameters()
        generation_path = str(self.get_parameter("teleop_generation_file").value)
        self._generation_store = TeleopGenerationStore(
            generation_path or Path.home() / ".ros/tracker_teleoperated/teleop_generation.json"
        )
        self._generation_fault = False
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._odom_frame = str(self.get_parameter("odom_frame").value)
        self._home_speed_percent = int(
            self.get_parameter("home_speed_percent").value
        )
        if not 1 <= self._home_speed_percent <= 100:
            raise ValueError("home_speed_percent 必须位于 [1, 100] 范围")
        self._home_timeout_s = float(
            self.get_parameter("home_timeout_s").value
        )
        if (
            not np.isfinite(self._home_timeout_s)
            or self._home_timeout_s <= 0.0
        ):
            raise ValueError("home_timeout_s 必须为正有限数")
        self._home_command_quiet_period_s = float(
            self.get_parameter("home_command_quiet_period_s").value
        )
        if (
            not np.isfinite(self._home_command_quiet_period_s)
            or self._home_command_quiet_period_s <= 0.0
        ):
            raise ValueError("home_command_quiet_period_s 必须为正有限数")
        # 配置为空时延迟到首帧完整关节反馈，非空时锁定指定的回位目标。
        self._home_joint_positions = parse_home_joint_positions(
            self.get_parameter("home_joint_positions_rad").value
        )
        self._control_rate_hz = float(
            self.get_parameter("control_rate_hz").value
        )
        if self._control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz 必须为正数")
        self._translation_scale = validate_scale(
            self.get_parameter("translation_scale").value,
            "translation_scale",
        )
        self._rotation_scale = validate_scale(
            self.get_parameter("rotation_scale").value,
            "rotation_scale",
        )
        self._mapping_mode = str(self.get_parameter("mapping_mode").value)
        if self._mapping_mode not in ("workspace", "reference_eef"):
            raise ValueError(
                "mapping_mode 必须为 workspace 或 reference_eef"
            )
        self._workspace_minimum_angle_deg = float(
            self.get_parameter("workspace_minimum_angle_deg").value
        )
        if (
            not np.isfinite(self._workspace_minimum_angle_deg)
            or not 0.0 < self._workspace_minimum_angle_deg < 90.0
        ):
            raise ValueError("workspace_minimum_angle_deg 必须位于 (0, 90) 度")
        self._workspace_calibration_path = (
            resolve_workspace_calibration_path(
                str(self.get_parameter("workspace_calibration_file").value)
            )
        )
        loaded_workspace_mapping: Optional[np.ndarray] = None
        workspace_load_message = ""
        if self._mapping_mode == "workspace":
            try:
                loaded_workspace_mapping = load_workspace_calibration(
                    self._workspace_calibration_path,
                    self._odom_frame,
                    self._base_frame,
                )
                workspace_load_message = (
                    "已加载工作空间标定方向，遥操保持暂停；"
                    "按空格以当前 UMI 和机械臂位姿建立运动零点"
                )
            except FileNotFoundError:
                workspace_load_message = (
                    "未找到工作空间标定文件，遥操保持暂停；"
                    "请按 c 完成三点标定"
                )
            except CalibrationStoreError as error:
                workspace_load_message = (
                    f"工作空间标定文件未加载: {error}；"
                    "遥操保持暂停，请按 c 重新标定"
                )
        self._axis_mapping = axis_mapping_from_rpy(
            self.get_parameter("axis_mapping_rpy_deg").value
        )
        self._pose_smoothing_enabled = bool(
            self.get_parameter("pose_smoothing_enabled").value
        )
        self._joint_smoothing_enabled = bool(
            self.get_parameter("joint_smoothing_enabled").value
        )
        self._pose_filter_cutoff_hz = float(
            self.get_parameter("pose_filter_cutoff_hz").value
        )
        self._joint_acceleration_limit = float(
            self.get_parameter("joint_acceleration_limit_rad_s2").value
        )
        self._freeze_timeout_s = float(
            self.get_parameter("pose_freeze_timeout_s").value
        )
        self._pose_timeout_s = float(
            self.get_parameter("pose_timeout_s").value
        )
        self._feedback_timeout_s = float(
            self.get_parameter("feedback_timeout_s").value
        )
        self._heartbeat_timeout_s = float(
            self.get_parameter("heartbeat_timeout_s").value
        )
        self._status_sync_tolerance_ns = int(
            float(self.get_parameter("status_sync_tolerance_s").value) * 1.0e9
        )
        if not 0.0 < self._freeze_timeout_s < self._pose_timeout_s:
            raise ValueError("位姿冻结阈值必须为正数且小于位姿超时阈值")
        if self._feedback_timeout_s <= 0.0 or self._heartbeat_timeout_s <= 0.0:
            raise ValueError("反馈和心跳超时必须为正数")

        urdf_path = resolve_rm75_urdf(
            str(self.get_parameter("urdf_path").value)
        )
        self._kinematics = kinematics or PlacoRm75Kinematics(
            urdf_path,
            base_frame=self._base_frame,
            eef_frame=str(self.get_parameter("eef_frame").value),
            control_rate_hz=self._control_rate_hz,
        )
        velocity_ratio = float(
            self.get_parameter("joint_velocity_ratio").value
        )
        if not 0.0 < velocity_ratio <= 1.0:
            raise ValueError("joint_velocity_ratio 必须位于 (0, 1] 范围")
        self._joint_velocity_limits = (
            load_joint_velocity_limits(urdf_path) * velocity_ratio
        )
        self._pose_validator = PoseStreamValidator(
            position_jump_m=float(
                self.get_parameter("position_jump_threshold_m").value
            ),
            rotation_jump_deg=float(
                self.get_parameter("rotation_jump_threshold_deg").value
            ),
            recovery_samples=int(
                self.get_parameter("recovery_samples").value
            ),
        )
        # 夹爪跟随决策与机械臂运动学独立，使用单调时钟判断输入是否新鲜。
        self._gripper_follower = GripperFollower(
            estimate_timeout_s=float(
                self.get_parameter("gripper_estimate_timeout_s").value
            ),
            feedback_timeout_s=float(
                self.get_parameter("gripper_feedback_timeout_s").value
            ),
        )

        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._command_publisher = self.create_publisher(
            Jointpos,
            str(self.get_parameter("command_topic").value),
            command_qos,
        )
        self._home_publisher = self.create_publisher(
            Movej,
            str(self.get_parameter("home_command_topic").value),
            command_qos,
        )
        self._move_stop_publisher = self.create_publisher(
            Empty,
            str(self.get_parameter("move_stop_topic").value),
            command_qos,
        )
        self._gripper_command_publisher = self.create_publisher(
            Float32,
            str(self.get_parameter("gripper_command_topic").value),
            command_qos,
        )
        self._target_publisher = self.create_publisher(
            PoseStamped, "/tracker_teleoperated/target_pose", 10
        )
        self._joint_target_publisher = self.create_publisher(
            JointState, "/tracker_teleoperated/joint_target", 10
        )
        self._enabled_publisher = self.create_publisher(
            Bool, "/tracker_teleoperated/enabled", state_qos
        )
        # 锁存最近的状态说明，让 RViz 面板能看到异步暂停原因和恢复步骤。
        self._status_publisher = self.create_publisher(
            String, "/tracker_teleoperated/status", state_qos
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("tracker_odom_topic").value),
            self._odom_callback,
            20,
        )
        self.create_subscription(
            TrackerStatus,
            str(self.get_parameter("tracker_status_topic").value),
            self._status_callback,
            20,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._joint_state_callback,
            20,
        )
        self.create_subscription(
            GripperState,
            str(self.get_parameter("gripper_estimate_topic").value),
            self._gripper_estimate_callback,
            1,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("gripper_feedback_topic").value),
            self._gripper_feedback_callback,
            1,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("home_result_topic").value),
            self._home_result_callback,
            10,
        )
        self.create_subscription(
            Empty,
            "/tracker_teleoperated/panel_heartbeat",
            self._heartbeat_callback,
            10,
        )
        self.create_service(
            SetBool,
            "/tracker_teleoperated/set_enabled",
            self._set_enabled_callback,
        )
        self.create_service(
            SetTeleopGeneration,
            "/tracker_teleoperated/enable",
            self._enable_generation_callback,
        )
        self.create_service(
            SetTeleopGeneration,
            "/tracker_teleoperated/disable",
            self._disable_generation_callback,
        )
        self.create_service(
            GetTeleopGeneration,
            "/tracker_teleoperated/get_generation",
            self._get_generation_callback,
        )
        self.create_service(
            Trigger,
            "/tracker_teleoperated/initialize",
            self._initialize_callback,
        )
        self.create_service(
            Trigger,
            "/tracker_teleoperated/calibrate_workspace",
            self._calibrate_workspace_callback,
        )
        self.create_service(
            Trigger,
            "/tracker_teleoperated/return_home",
            self._return_home_callback,
        )
        self.create_service(
            Trigger,
            "/tracker_teleoperated/shutdown",
            self._shutdown_callback,
        )

        self._enabled = False
        # 面板请求退出后，由主循环完成安全收尾并结束控制进程。
        self._shutdown_requested = False
        self._latest_status_valid = False
        self._latest_status_stamp_ns: Optional[int] = None
        self._pending_tracker_sample: Optional[tuple[int, np.ndarray]] = None
        self._latest_tracker_pose: Optional[np.ndarray] = None
        self._latest_tracker_stamp_ns: Optional[int] = None
        self._latest_tracker_monotonic = 0.0
        self._latest_joint_positions: Optional[np.ndarray] = None
        self._latest_joint_monotonic = 0.0
        self._homing = False
        # 待发命令存在时处于 CANFD 静默期，实际发布后才开始计算 MoveJ 超时。
        self._home_started_monotonic = 0.0
        self._home_command_due_monotonic = 0.0
        self._pending_home_command: Optional[Movej] = None
        self._latest_heartbeat_monotonic = 0.0
        self._mapping_basis: Optional[np.ndarray] = loaded_workspace_mapping
        self._workspace_calibration_samples: list[np.ndarray] = []
        self._reference_tracker_pose: Optional[np.ndarray] = None
        self._reference_eef_pose: Optional[np.ndarray] = None
        self._filtered_tracker_pose: Optional[np.ndarray] = None
        self._command_positions: Optional[np.ndarray] = None
        self._command_velocity = np.zeros(7, dtype=np.float64)
        self._last_tick_monotonic = time.monotonic()
        self._timer = self.create_timer(
            1.0 / self._control_rate_hz, self._control_tick
        )
        self.create_timer(0.5, self._publish_enabled)
        self._publish_enabled()
        self._publish_status(
            workspace_load_message
            or "遥操已暂停，等待标定或人工启用"
        )
        self.get_logger().info(
            "Tracker 遥操节点已启动，默认暂停，"
            f"mapping_mode={self._mapping_mode}，"
            f"控制频率={self._control_rate_hz:g} Hz"
        )
        if self._home_joint_positions is not None:
            self.get_logger().info(
                "已使用配置的回位关节角（joint1 至 joint7，rad）: "
                f"{self._home_joint_positions.tolist()}"
            )
        if workspace_load_message:
            if loaded_workspace_mapping is None:
                self.get_logger().warning(workspace_load_message)
            else:
                self.get_logger().info(
                    f"{workspace_load_message}，文件="
                    f"{self._workspace_calibration_path}"
                )

    def _declare_parameters(self) -> None:
        """声明遥操接口、坐标映射、平滑和安全参数。"""
        self.declare_parameter("tracker_odom_topic", "/vive_tracker/odom")
        self.declare_parameter("tracker_status_topic", "/vive_tracker/status")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("command_topic", "/rm_driver/movej_canfd_cmd")
        self.declare_parameter("home_command_topic", "/rm_driver/movej_cmd")
        self.declare_parameter("home_result_topic", "/rm_driver/movej_result")
        self.declare_parameter("move_stop_topic", "/rm_driver/move_stop_cmd")
        self.declare_parameter("gripper_estimate_topic", "/gripper/state")
        self.declare_parameter(
            "gripper_feedback_topic", "/motion_control/gripper_state"
        )
        self.declare_parameter(
            "gripper_command_topic", "/motion_control/gripper_command"
        )
        self.declare_parameter("gripper_estimate_timeout_s", 0.25)
        self.declare_parameter("gripper_feedback_timeout_s", 0.25)
        self.declare_parameter("home_speed_percent", 20)
        self.declare_parameter("home_timeout_s", 30.0)
        self.declare_parameter("home_command_quiet_period_s", 0.20)
        self.declare_parameter("teleop_generation_file", "")
        # ROS 的空列表可能为未设置值或字节数组，动态类型允许数值数组覆盖。
        # 参数只在启动时读取，因此禁止运行期间修改后产生配置与目标不一致。
        self.declare_parameter(
            "home_joint_positions_rad",
            [],
            ParameterDescriptor(
                dynamic_typing=True,
                read_only=True,
                description="回位七轴关节角（rad）；空列表使用启动时当前位姿",
            ),
        )
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("eef_frame", "Link7")
        self.declare_parameter("odom_frame", "vive_tracker_odom")
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("translation_scale", 0.5)
        self.declare_parameter("rotation_scale", 1.0)
        self.declare_parameter("mapping_mode", "workspace")
        self.declare_parameter("workspace_minimum_angle_deg", 60.0)
        self.declare_parameter("workspace_calibration_file", "")
        self.declare_parameter("axis_mapping_rpy_deg", [0.0, 0.0, 0.0])
        self.declare_parameter("pose_smoothing_enabled", False)
        self.declare_parameter("joint_smoothing_enabled", False)
        self.declare_parameter("pose_filter_cutoff_hz", 8.0)
        self.declare_parameter("joint_velocity_ratio", 0.25)
        self.declare_parameter("joint_acceleration_limit_rad_s2", 2.0)
        self.declare_parameter("pose_freeze_timeout_s", 0.10)
        self.declare_parameter("pose_timeout_s", 0.25)
        self.declare_parameter("feedback_timeout_s", 0.25)
        self.declare_parameter("heartbeat_timeout_s", 0.50)
        self.declare_parameter("status_sync_tolerance_s", 0.05)
        self.declare_parameter("position_jump_threshold_m", 0.30)
        self.declare_parameter("rotation_jump_threshold_deg", 45.0)
        self.declare_parameter("recovery_samples", 3)

    def _publish_enabled(self) -> None:
        """发布当前真实启用状态。"""
        self._enabled_publisher.publish(Bool(data=self._enabled))

    def _publish_status(self, message: str) -> None:
        """发布可供面板和晚加入订阅者读取的状态及恢复说明。"""
        self._status_publisher.publish(String(data=message))

    def _heartbeat_callback(self, _message: Empty) -> None:
        """记录 RViz 面板最近一次存活心跳。"""
        self._latest_heartbeat_monotonic = time.monotonic()

    def _gripper_estimate_callback(self, message: GripperState) -> None:
        """接收预测状态；无效帧立即使夹爪退出跟随。"""
        now = time.monotonic()
        valid = self._gripper_follower.update_estimate(
            bool(message.valid), float(message.filtered_openness), now
        )
        if not valid and self._enabled:
            self._update_gripper(True, now)

    def _gripper_feedback_callback(self, message: Float32) -> None:
        """接收真实夹爪开度，暂停时尽快建立保持目标。"""
        now = time.monotonic()
        valid = self._gripper_follower.update_feedback(float(message.data), now)
        if not valid or not self._enabled:
            self._update_gripper(self._enabled, now)

    def _update_gripper(self, enabled: bool, now: float) -> None:
        """根据遥操与夹爪输入状态发布预测目标或一次性保持目标。"""
        decision = self._gripper_follower.step(enabled, now)
        if decision.command is not None:
            self._gripper_command_publisher.publish(
                Float32(data=float(decision.command))
            )
        if not enabled or not decision.changed:
            return
        # 夹爪输入问题只更改夹爪状态说明，不改变机械臂启用状态。
        status_messages = {
            "following": "夹爪正在跟随预测开度；机械臂继续跟随",
            "feedback_missing": "夹爪实测反馈未就绪，暂不发送夹爪预测命令；机械臂继续跟随",
            "feedback_invalid": "夹爪实测反馈无效，保持最近实测开度；机械臂继续跟随",
            "feedback_timeout": "夹爪实测反馈超时，保持最近实测开度；机械臂继续跟随",
            "estimate_missing": "夹爪预测未就绪，保持最近实测开度；机械臂继续跟随",
            "estimate_invalid": "夹爪预测无效，保持最近实测开度；机械臂继续跟随",
            "estimate_timeout": "夹爪预测超时，保持最近实测开度；机械臂继续跟随",
        }
        self._publish_status(status_messages[decision.mode])

    def _cancel_workspace_calibration(self) -> bool:
        """清除未完成的工作空间标定样本并返回此前是否正在标定。"""
        was_active = bool(self._workspace_calibration_samples)
        self._workspace_calibration_samples.clear()
        if was_active:
            self.get_logger().warning("未完成的工作空间标定已取消")
        return was_active

    def _mapping_recovery_instruction(self) -> str:
        """根据当前是否已有标定方向返回人工恢复说明。"""
        if self._mapping_basis is not None:
            return "标定方向已保留，待输入恢复稳定后按空格重新启用"
        return "尚未建立标定方向，待输入稳定后按 c 完成三点标定"

    def _status_callback(self, message: TrackerStatus) -> None:
        """记录 Tracker 质量状态，并在无效时立即停止跟随。"""
        stamp_ns = stamp_to_ns(message.header.stamp)
        if (
            self._latest_status_stamp_ns is not None
            and stamp_ns < self._latest_status_stamp_ns
        ):
            self.get_logger().warning("忽略时间戳倒退的 Tracker 状态")
            return
        self._latest_status_stamp_ns = stamp_ns
        self._latest_status_valid = bool(
            message.device_connected
            and message.pose_valid
            and message.tracking_state == TrackerStatus.TRACKING_RUNNING_OK
        )
        if not self._latest_status_valid:
            self._pending_tracker_sample = None
            self._pose_validator.invalidate()
            self._cancel_workspace_calibration()
            self._disable("Tracker 跟踪状态无效", publish_hold=True)
            return
        if self._pending_tracker_sample is not None:
            pending_stamp_ns, pending_pose = self._pending_tracker_sample
            if (
                abs(stamp_ns - pending_stamp_ns)
                <= self._status_sync_tolerance_ns
            ):
                self._pending_tracker_sample = None
                self._accept_tracker_pose(pending_pose, pending_stamp_ns)

    def _status_matches(self, stamp_ns: int) -> bool:
        """判断最近状态是否有效且与当前里程计采样时间相符。"""
        return bool(
            self._latest_status_valid
            and self._latest_status_stamp_ns is not None
            and abs(stamp_ns - self._latest_status_stamp_ns)
            <= self._status_sync_tolerance_ns
        )

    def _odom_callback(self, message: Odometry) -> None:
        """校验、关联并保存最新 Tracker 首帧归零位姿。"""
        if message.header.frame_id != self._odom_frame:
            self.get_logger().warning(
                f"忽略 frame_id={message.header.frame_id!r} 的 Tracker 里程计，"
                f"期望 {self._odom_frame!r}"
            )
            self._pose_validator.invalidate()
            self._cancel_workspace_calibration()
            self._disable("Tracker 里程计坐标系不匹配", publish_hold=True)
            return
        stamp_ns = stamp_to_ns(message.header.stamp)
        if stamp_ns <= 0:
            self.get_logger().warning("忽略零时间戳 Tracker 里程计")
            return
        if (
            self._latest_tracker_stamp_ns is not None
            and stamp_ns <= self._latest_tracker_stamp_ns
        ):
            self.get_logger().warning("忽略时间戳重复或倒退的 Tracker 里程计")
            return
        pose = message.pose.pose
        try:
            transform = pose_to_matrix(
                np.array(
                    [pose.position.x, pose.position.y, pose.position.z],
                    dtype=np.float64,
                ),
                np.array(
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ],
                    dtype=np.float64,
                ),
            )
        except ValueError as error:
            self._pose_validator.invalidate()
            self._cancel_workspace_calibration()
            self._disable(f"Tracker 位姿无效: {error}", publish_hold=True)
            return
        if not self._status_matches(stamp_ns):
            status_is_older = bool(
                self._latest_status_stamp_ns is None
                or self._latest_status_stamp_ns
                < stamp_ns - self._status_sync_tolerance_ns
            )
            if status_is_older:
                self._pending_tracker_sample = (stamp_ns, transform)
                return
            self._pose_validator.invalidate()
            self._cancel_workspace_calibration()
            self._disable("Tracker 状态与里程计未有效关联", publish_hold=True)
            return
        self._pending_tracker_sample = None
        self._accept_tracker_pose(transform, stamp_ns)

    def _accept_tracker_pose(
        self, transform: np.ndarray, stamp_ns: int
    ) -> None:
        """对已经和有效状态关联的 Tracker 位姿执行连续性检查。"""
        try:
            validation = self._pose_validator.update(transform)
        except ValueError as error:
            self._pose_validator.invalidate()
            self._cancel_workspace_calibration()
            self._disable(f"Tracker 位姿无效: {error}", publish_hold=True)
            return
        if not validation.accepted:
            self._cancel_workspace_calibration()
            self._disable(validation.reason, publish_hold=True)
            return
        if validation.relocalized:
            self._disable(validation.reason, publish_hold=True)
            message = (
                f"{validation.reason}，遥操保持暂停；"
                f"{self._mapping_recovery_instruction()}"
            )
            self.get_logger().warning(message)
            self._publish_status(message)
        elif validation.ready and validation.reason:
            self._publish_status(
                f"{validation.reason}，遥操保持暂停；"
                f"{self._mapping_recovery_instruction()}"
            )
        self._latest_tracker_pose = transform
        self._latest_tracker_stamp_ns = stamp_ns
        self._latest_tracker_monotonic = time.monotonic()

    def _joint_state_callback(self, message: JointState) -> None:
        """保存七轴反馈，未配置回位目标时锁定启动后的首帧位姿。"""
        try:
            positions = reorder_joint_state(message)
        except ValueError as error:
            self.get_logger().warning(str(error))
            return
        if self._home_joint_positions is None:
            self._home_joint_positions = positions.copy()
            self.get_logger().info(
                "已将启动时当前关节位姿记录为回位目标，可按 h 执行回位"
            )
        self._latest_joint_positions = positions
        self._latest_joint_monotonic = time.monotonic()

    def _home_result_callback(self, message: Bool) -> None:
        """处理 RM75 MoveJ 回位执行结果并解除回位状态。"""
        if (
            not self._homing
            or self._pending_home_command is not None
            or self._home_started_monotonic <= 0.0
        ):
            return
        self._homing = False
        self._home_started_monotonic = 0.0
        self._home_command_due_monotonic = 0.0
        if message.data:
            self.get_logger().info("机械臂已到达回位目标")
            self._publish_status("机械臂已到达回位目标，遥操保持暂停")
        else:
            self.get_logger().error("机械臂回位失败")
            self._publish_status("机械臂回位失败，遥操保持暂停")

    def _clear_control_reference(self) -> None:
        """清除当前遥操运动零点和关节指令状态。"""
        self._reference_tracker_pose = None
        self._reference_eef_pose = None
        self._filtered_tracker_pose = None
        self._command_positions = None
        self._command_velocity.fill(0.0)

    def _stop_homing(self, reason: str, publish_stop: bool = True) -> bool:
        """退出回位状态，并按需请求 RM75 停止当前规划轨迹。"""
        if not self._homing:
            return False
        if publish_stop:
            self._move_stop_publisher.publish(Empty())
        self._homing = False
        self._home_started_monotonic = 0.0
        self._home_command_due_monotonic = 0.0
        self._pending_home_command = None
        self.get_logger().warning(f"机械臂回位已停止: {reason}")
        self._publish_status(f"机械臂回位已停止: {reason}；遥操保持暂停")
        return True

    def _tracker_freshness_error(self, now: float) -> Optional[str]:
        """返回阻止工作空间采样的首个 Tracker 健康问题。"""
        if self._latest_tracker_pose is None:
            return "尚未收到有效 Tracker 里程计"
        if now - self._latest_tracker_monotonic > self._pose_timeout_s:
            return "Tracker 里程计已超时"
        if not self._latest_status_valid:
            return "Tracker 跟踪状态无效"
        if (
            self._pose_validator.stable_samples
            < self._pose_validator.recovery_samples
        ):
            return "Tracker 尚未积累足够的连续稳定帧"
        return None

    def _freshness_error(self, now: float) -> Optional[str]:
        """返回阻止初始化或启用的首个完整输入健康问题。"""
        if now - self._latest_heartbeat_monotonic > self._heartbeat_timeout_s:
            return "面板心跳未就绪或已超时"
        tracker_error = self._tracker_freshness_error(now)
        if tracker_error is not None:
            return tracker_error
        if self._latest_joint_positions is None:
            return "尚未收到完整七轴反馈"
        if now - self._latest_joint_monotonic > self._feedback_timeout_s:
            return "七轴反馈已超时"
        return None

    def _calibrate_workspace_callback(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """依次记录起点、上方点和前方点并生成固定工作空间映射。"""
        if self._homing:
            response.success = False
            response.message = "机械臂正在回位，不能采集工作空间标定点"
            return response
        if self._mapping_mode != "workspace":
            response.success = False
            response.message = (
                "当前 mapping_mode=reference_eef，不使用工作空间标定"
            )
            return response
        if self._enabled:
            response.success = False
            response.message = "遥操启用期间不能标定工作空间，请先暂停"
            return response
        health_error = self._tracker_freshness_error(time.monotonic())
        if health_error is not None:
            self._cancel_workspace_calibration()
            response.success = False
            response.message = f"工作空间标定已取消: {health_error}"
            return response
        assert self._latest_tracker_pose is not None
        self._workspace_calibration_samples.append(
            self._latest_tracker_pose.copy()
        )
        sample_count = len(self._workspace_calibration_samples)
        response.success = True
        if sample_count == 1:
            response.message = (
                "已记录工作空间起点；大致向上移动至少 5 cm 后再次按 c"
            )
            return response
        if sample_count == 2:
            response.message = (
                "已记录上方点；从当前位置大致向前移动至少 5 cm 后再次按 c"
            )
            return response

        samples = tuple(self._workspace_calibration_samples)
        self._workspace_calibration_samples.clear()
        try:
            mapping_basis = workspace_mapping_from_samples(
                *samples, minimum_angle_deg=self._workspace_minimum_angle_deg
            )
        except ValueError as error:
            response.success = False
            response.message = f"工作空间标定失败，本轮样本已清除: {error}"
            return response
        save_error = self._save_workspace_mapping(mapping_basis)
        if save_error is not None:
            response.success = False
            response.message = (
                "工作空间标定未生效，旧标定方向保持不变: "
                f"{save_error}"
            )
            self.get_logger().error(response.message)
            self._publish_status(f"遥操保持暂停；{response.message}")
            return response
        self._mapping_basis = mapping_basis
        self._reference_tracker_pose = None
        self._reference_eef_pose = None
        self._filtered_tracker_pose = None
        response.message = (
            "工作空间标定完成：前/左/上已映射到 Base +X/+Y/+Z；"
            "请让两端夹爪朝下并对齐夹指方向后启用遥操"
        )
        self.get_logger().warning(response.message)
        self._publish_status(f"遥操已暂停；{response.message}")
        return response

    def _save_workspace_mapping(
        self, mapping_basis: np.ndarray
    ) -> Optional[str]:
        """保存候选标定方向，返回错误文本或 None。"""
        try:
            save_workspace_calibration(
                self._workspace_calibration_path,
                self._odom_frame,
                self._base_frame,
                mapping_basis,
            )
        except CalibrationStoreError as error:
            return str(error)
        return None

    def _capture_reference(self, initialize_mapping: bool) -> Optional[str]:
        """校验输入并原子更新固定映射轴向及本次跟随零点。"""
        health_error = self._freshness_error(time.monotonic())
        if health_error is not None:
            return health_error
        assert self._latest_tracker_pose is not None
        assert self._latest_joint_positions is not None
        tracker_reference = self._latest_tracker_pose.copy()
        joint_reference = self._latest_joint_positions.copy()
        try:
            eef_reference = validate_transform(
                self._kinematics.end_effector_pose(joint_reference),
                "末端参考位姿",
            )
            if initialize_mapping and self._mapping_mode == "reference_eef":
                mapping_basis = mapping_basis_from_reference(
                    tracker_reference,
                    eef_reference,
                    self._axis_mapping,
                )
            elif self._mapping_basis is None:
                return "尚未建立 Tracker 到 Base 的坐标映射"
            else:
                mapping_basis = self._mapping_basis.copy()
        except Exception as error:
            return f"无法建立末端参考位姿: {error}"

        self._mapping_basis = mapping_basis
        self._reference_tracker_pose = tracker_reference
        self._reference_eef_pose = eef_reference.copy()
        self._filtered_tracker_pose = tracker_reference.copy()
        self._command_positions = joint_reference
        self._command_velocity.fill(0.0)
        self._last_tick_monotonic = time.monotonic()
        return None

    def _initialize_callback(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """以当前 Tracker 和末端位姿重建运动零点并保持启停状态。"""
        if self._homing:
            response.success = False
            response.message = "机械臂正在回位，不能重置运动零点"
            return response
        if self._workspace_calibration_samples:
            response.success = False
            response.message = "工作空间标定尚未完成，请完成标定或按 s 取消"
            return response
        initialize_mapping = self._mapping_mode == "reference_eef"
        error = self._capture_reference(initialize_mapping=initialize_mapping)
        if error is not None:
            response.success = False
            response.message = f"无法初始化运动零点: {error}"
            return response
        response.success = True
        state = "继续跟随" if self._enabled else "保持暂停"
        response.message = f"运动零点已重置，遥操{state}"
        self.get_logger().warning(response.message)
        self._publish_status(response.message)
        return response

    def _return_home_callback(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """暂停遥操并回到配置目标，默认使用启动时记录的七轴位姿。"""
        if self._homing:
            response.success = False
            response.message = "机械臂正在回位"
            return response
        if self._home_joint_positions is None:
            response.success = False
            response.message = "尚未收到完整七轴反馈，无法确定默认回位目标"
            return response
        now = time.monotonic()
        if (
            self._latest_joint_positions is None
            or now - self._latest_joint_monotonic > self._feedback_timeout_s
        ):
            response.success = False
            response.message = "七轴反馈未就绪或已超时，拒绝执行回位"
            return response

        if now - self._latest_heartbeat_monotonic > self._heartbeat_timeout_s:
            response.success = False
            response.message = "面板心跳未就绪或已超时"
            return response

        self._cancel_workspace_calibration()
        # 回位前禁止补发旧保持点，避免它滞留到阻塞式 MoveJ 完成后执行。
        self._disable("收到回位请求", publish_hold=False, force_fence=True)
        self._clear_control_reference()
        home_positions = self._home_joint_positions.copy()
        self._pending_home_command = build_home_command(
            home_positions, self._home_speed_percent
        )
        self._homing = True
        self._home_started_monotonic = 0.0
        self._home_command_due_monotonic = (
            now + self._home_command_quiet_period_s
        )
        response.success = True
        response.message = (
            "已暂停遥操，等待旧透传命令排空后回位，"
            f"静默时间={self._home_command_quiet_period_s:g} 秒"
        )
        self.get_logger().warning(response.message)
        self._publish_status(response.message)
        return response

    def _shutdown_callback(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """确认面板退出请求，让主循环安全停止控制节点。"""
        self._shutdown_requested = True
        response.success = True
        response.message = "控制节点已收到退出请求"
        return response

    def _set_enabled_callback(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        """兼容旧暂停接口，并拒绝无代次的启用请求。"""
        if request.data:
            response.success = False
            response.message = "启用遥操必须使用带代次的 enable 服务"
            return response
        try:
            generation = self._generation_store.record["generation"] + 1
            self._generation_store.begin(generation, "disable")
            result = TrackerTeleopNode._apply_enabled_callback(self, request, response)
            self._generation_store.finish(result.success, "DISABLED", result.message)
            return result
        except (OSError, ValueError) as error:
            self._generation_fault = True
            TrackerTeleopNode._apply_enabled_callback(self, request, response)
            response.success = False
            response.message = f"遥操已暂停，但代次持久化失败: {error}"
            return response

    def _generation_operation(self, request, response, operation: str):
        """持久化请求代次后执行启停，并返回节点权威状态。"""
        response.operation_generation = self._generation_store.record["generation"]
        response.enabled = self._enabled
        generation = int(request.operation_generation)
        if generation == 0:
            response.code, response.message = "INVALID_ARGUMENT", "操作代次必须大于零"
            return response
        if operation == "enable" and self._generation_fault:
            response.code, response.message = "IO_ERROR", "代次持久化故障，拒绝启用"
            return response
        try:
            decision = self._generation_store.begin(generation, operation)
        except (OSError, ValueError) as error:
            self._generation_fault = True
            if operation == "disable":
                pause = SetBool.Request(data=False)
                TrackerTeleopNode._apply_enabled_callback(self, pause, SetBool.Response())
            response.code, response.message = "IO_ERROR", str(error)
            response.enabled = self._enabled
            return response
        response.operation_generation = self._generation_store.record["generation"]
        if decision == "stale":
            response.code, response.message = "STALE_GENERATION", "操作代次已过期"
        elif decision == "duplicate":
            record = self._generation_store.record
            response.success = record["success"]
            response.code, response.message = record["code"], record["message"]
        else:
            action = SetBool.Request(data=operation == "enable")
            result = TrackerTeleopNode._apply_enabled_callback(self, action, SetBool.Response())
            response.success, response.message = result.success, result.message
            code = ("ENABLED" if operation == "enable" else "DISABLED") if result.success else "REJECTED"
            try:
                self._generation_store.finish(result.success, code, result.message)
                response.code = code
            except (OSError, ValueError) as error:
                self._generation_fault = True
                pause = SetBool.Request(data=False)
                TrackerTeleopNode._apply_enabled_callback(self, pause, SetBool.Response())
                response.success, response.code, response.message = False, "IO_ERROR", str(error)
        response.enabled = self._enabled
        return response

    def _enable_generation_callback(self, request, response):
        """执行带持久化代次的遥操启用。"""
        return TrackerTeleopNode._generation_operation(self, request, response, "enable")

    def _disable_generation_callback(self, request, response):
        """执行带持久化代次的遥操暂停。"""
        return TrackerTeleopNode._generation_operation(self, request, response, "disable")

    def _get_generation_callback(self, _request, response):
        """返回最高代次和当前启用状态。"""
        response.operation_generation = self._generation_store.record["generation"]
        response.enabled = self._enabled
        return response

    def _apply_enabled_callback(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        """应用已通过代次门控的启停请求。"""
        if not request.data:
            homing_stopped = self._stop_homing("收到人工暂停请求")
            calibration_cancelled = self._cancel_workspace_calibration()
            self._disable("收到人工暂停请求", publish_hold=True, fenced=True)
            response.success = True
            details = []
            if homing_stopped:
                details.append("回位已停止")
            if calibration_cancelled:
                details.append("工作空间标定已取消")
            response.message = "遥操已暂停"
            if details:
                response.message += "，" + "，".join(details)
            return response
        if self._homing:
            response.success = False
            response.message = "机械臂正在回位，完成或停止后才能启用遥操"
            return response
        if self._enabled:
            response.success = True
            response.message = "遥操已经启用，控制零点保持不变"
            return response
        if self._workspace_calibration_samples:
            response.success = False
            response.message = "工作空间标定尚未完成，请继续按 c 或按 s 取消"
            return response
        if self._mapping_mode == "workspace" and self._mapping_basis is None:
            response.success = False
            response.message = "尚未标定工作空间，请暂停状态下按 c 完成三点标定"
            return response
        initialize_mapping = (
            self._mapping_mode == "reference_eef"
            and self._mapping_basis is None
        )
        reference_error = self._capture_reference(initialize_mapping)
        if reference_error is not None:
            response.success = False
            response.message = f"无法启用遥操: {reference_error}"
            return response
        self._enabled = True
        self._publish_enabled()
        response.success = True
        initialization = (
            "并自动初始化参考轴向" if initialize_mapping else ""
        )
        response.message = f"遥操已启用{initialization}，当前姿态已设为运动零点"
        self.get_logger().warning(response.message)
        self._publish_status(response.message)
        return response

    def _publish_joint_target(self, positions: np.ndarray) -> None:
        """发布七轴调试关节目标。"""
        joint_message = JointState()
        joint_message.header.stamp = self.get_clock().now().to_msg()
        joint_message.name = list(JOINT_NAMES)
        joint_message.position = np.asarray(positions).tolist()
        self._joint_target_publisher.publish(joint_message)

    def _publish_command(self, positions: np.ndarray) -> None:
        """发布调试关节目标和 RM75 CANFD 指令。"""
        self._publish_joint_target(positions)
        self._command_publisher.publish(build_joint_command(positions))

    def _disable(
        self, reason: str, publish_hold: bool, *, force_fence: bool = False,
        fenced: bool = False,
    ) -> None:
        """停止当前跟随、按条件发送一次保持点并清除控制参考。"""
        if not fenced and (self._enabled or force_fence):
            try:
                generation = self._generation_store.record["generation"] + 1
                self._generation_store.begin(generation, "disable")
                self._generation_store.finish(True, "DISABLED", reason)
            except (OSError, ValueError) as error:
                self._generation_fault = True
                self.get_logger().error(f"遥操代次持久化失败，保持禁用: {error}")
        if not self._enabled:
            return
        now = time.monotonic()
        self._update_gripper(False, now)
        feedback_fresh = bool(
            self._latest_joint_positions is not None
            and now - self._latest_joint_monotonic <= self._feedback_timeout_s
        )
        if (
            publish_hold
            and feedback_fresh
            and self._command_positions is not None
        ):
            self._publish_command(self._command_positions)
        self._enabled = False
        self._clear_control_reference()
        self._publish_enabled()
        self.get_logger().warning(f"遥操已暂停: {reason}")
        self._publish_status(
            f"遥操已暂停: {reason}；{self._mapping_recovery_instruction()}"
        )

    def _publish_target_pose(self, target: np.ndarray) -> None:
        """发布基座坐标系中的 Link7 调试目标。"""
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._base_frame
        fill_pose_message(target, message)
        self._target_publisher.publish(message)

    def _control_tick(self) -> None:
        """在固定频率下检查安全条件、运行 IK 并发布关节目标。"""
        now = time.monotonic()
        dt = min(max(now - self._last_tick_monotonic, 1.0e-4), 0.1)
        self._last_tick_monotonic = now
        if self._homing:
            if now - self._latest_heartbeat_monotonic > self._heartbeat_timeout_s:
                self._stop_homing("面板心跳超时")
            elif now - self._latest_joint_monotonic > self._feedback_timeout_s:
                self._stop_homing("七轴反馈超时")
            elif self._pending_home_command is not None:
                if now >= self._home_command_due_monotonic:
                    command = self._pending_home_command
                    self._pending_home_command = None
                    self._home_command_due_monotonic = 0.0
                    self._home_started_monotonic = now
                    self._publish_joint_target(
                        np.asarray(command.joint, dtype=np.float64)
                    )
                    self._home_publisher.publish(command)
                    message = (
                        "旧透传命令静默期结束，已开始回到目标位姿，"
                        f"MoveJ 速度={self._home_speed_percent}%"
                    )
                    self.get_logger().warning(message)
                    self._publish_status(message)
            elif now - self._home_started_monotonic > self._home_timeout_s:
                self._stop_homing("超过回位时间上限")
            return
        if not self._enabled:
            self._update_gripper(False, now)
            return
        if now - self._latest_heartbeat_monotonic > self._heartbeat_timeout_s:
            self._disable("面板心跳超时", publish_hold=True)
            return
        if now - self._latest_joint_monotonic > self._feedback_timeout_s:
            self._disable("七轴反馈超时", publish_hold=False)
            return
        tracker_age = now - self._latest_tracker_monotonic
        if tracker_age > self._pose_timeout_s:
            self._disable("Tracker 里程计超时", publish_hold=True)
            return
        self._update_gripper(True, now)
        assert self._command_positions is not None
        if tracker_age > self._freeze_timeout_s:
            self._publish_command(self._command_positions)
            return
        assert self._latest_tracker_pose is not None
        assert self._filtered_tracker_pose is not None
        assert self._mapping_basis is not None
        assert self._reference_tracker_pose is not None
        assert self._reference_eef_pose is not None
        try:
            tracker_pose = self._latest_tracker_pose
            if self._pose_smoothing_enabled:
                tracker_pose = low_pass_pose(
                    self._filtered_tracker_pose,
                    tracker_pose,
                    self._pose_filter_cutoff_hz,
                    dt,
                )
            self._filtered_tracker_pose = tracker_pose.copy()
            target_pose = map_tracker_target(
                self._reference_tracker_pose,
                tracker_pose,
                self._reference_eef_pose,
                self._mapping_basis,
                self._translation_scale,
                self._rotation_scale,
            )
            ik_positions = self._kinematics.solve(
                target_pose, self._command_positions
            )
            if self._joint_smoothing_enabled:
                next_positions, next_velocity = advance_joint_command(
                    self._command_positions,
                    self._command_velocity,
                    ik_positions,
                    self._joint_velocity_limits,
                    self._joint_acceleration_limit,
                    dt,
                )
            else:
                next_positions = np.asarray(ik_positions, dtype=np.float64)
                next_velocity = np.zeros(7, dtype=np.float64)
            if next_positions.shape != (7,) or not np.all(
                np.isfinite(next_positions)
            ):
                raise RuntimeError("IK 关节输出无效")
        except Exception as error:
            self._disable(f"IK 或目标计算失败: {error}", publish_hold=True)
            return
        self._command_positions = next_positions.copy()
        self._command_velocity = next_velocity.copy()
        self._publish_target_pose(target_pose)
        self._publish_command(next_positions)


def main(args=None) -> None:
    """初始化 ROS 2 并运行 Tracker 遥操控制节点。"""
    rclpy.init(args=args)
    node: Optional[TrackerTeleopNode] = None
    try:
        node = TrackerTeleopNode()
        while rclpy.ok() and not node._shutdown_requested:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node._stop_homing("控制节点退出")
            node._disable("控制节点退出", publish_hold=True)
            node._generation_store.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
