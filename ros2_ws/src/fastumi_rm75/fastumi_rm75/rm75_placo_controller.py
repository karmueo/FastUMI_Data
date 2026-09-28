"""使用 Placo 把绝对 Link7 策略序列转换成 RM75 七轴透传命令。"""

from __future__ import annotations

from pathlib import Path
import signal
import time

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from fastumi_interfaces.msg import PolicyActionSequence
from fastumi_interfaces.srv import ResetPolicyController
import numpy as np
import placo
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rm_ros_interfaces.msg import Jointpos
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float32

from fastumi_rm75.placo_control import (
    JOINT_NAMES,
    decode_trajectory,
    fill_joint_command,
    gate_gripper_target,
    ordered_joint_positions,
    pose_matrix,
    sample_trajectory,
    sequence_is_new,
    urdf_limits,
    validate_gripper_targets,
    validate_start_envelope,
    validate_workspace,
)


class Rm75PlacoController(Node):
    """以 50 Hz 连续求解最新策略目标并发布低跟随关节命令。"""

    def __init__(self) -> None:
        super().__init__("rm75_placo_controller")
        try:
            package_root = Path(get_package_share_directory("fastumi_rm75"))
        except PackageNotFoundError:
            package_root = Path(__file__).resolve().parents[1]
        default_urdf = package_root / "assets" / "rm_75_kinematic.urdf"
        self.declare_parameter("dry_run", False)
        self.declare_parameter("urdf_path", str(default_urdf))
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("velocity_scale", 0.25)
        self.declare_parameter("joint_state_timeout_s", 0.25)
        self.declare_parameter("require_start_state", False)
        self.declare_parameter("start_joint_positions", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("start_joint_tolerance_rad", Parameter.Type.DOUBLE)
        self.declare_parameter("start_joint_settle_s", Parameter.Type.DOUBLE)
        self.declare_parameter("start_gripper_min", Parameter.Type.DOUBLE)
        self.declare_parameter("gripper_state_timeout_s", 0.25)
        self.declare_parameter("gripper_command_min", 0.0)
        self.declare_parameter("require_close_approval", True)
        self.declare_parameter("close_approval_topic", "/fastumi/policy/gripper_close_allowed")
        self.declare_parameter("close_approval_timeout_s", 0.25)
        self.declare_parameter("close_min_episode_time_s", 1.5)
        self.declare_parameter("workspace_min", [-10.0] * 3)
        self.declare_parameter("workspace_max", [10.0] * 3)
        self.declare_parameter("max_start_displacement_m", 0.0)
        self.declare_parameter("max_start_rise_m", 0.0)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("end_frame", "Link7")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("gripper_state_topic", "/motion_control/gripper_state")
        self.declare_parameter(
            "policy_topic", "/fastumi/policy/action_sequence"
        )
        self.declare_parameter(
            "joint_command_topic", "/rm_driver/movej_canfd_cmd"
        )
        self.declare_parameter(
            "gripper_command_topic", "/motion_control/gripper_command"
        )
        self.declare_parameter(
            "debug_joint_topic", "/fastumi/rm75/placo/joint_command"
        )
        self.declare_parameter("stop_topic", "/rm_driver/move_stop_cmd")
        self.declare_parameter("reset_service", "/fastumi/rm75/placo/reset_episode")

        self._dry_run = bool(self.get_parameter("dry_run").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._end_frame = str(self.get_parameter("end_frame").value)
        rate_hz = float(self.get_parameter("control_rate_hz").value)
        velocity_scale = float(self.get_parameter("velocity_scale").value)
        self._feedback_timeout_s = float(
            self.get_parameter("joint_state_timeout_s").value
        )
        start_defaults = {
            "start_joint_positions": [0.0] * 7,
            "start_joint_tolerance_rad": 0.035,
            "start_joint_settle_s": 0.5,
            "start_gripper_min": 0.95,
        }
        missing_start = [name for name in start_defaults
                         if self.get_parameter_or(name).type_ == Parameter.Type.NOT_SET]
        if not self._dry_run and missing_start:
            raise ValueError(
                "real control requires explicit start-state parameters: "
                + ", ".join(missing_start))

        def start_value(name):
            parameter = self.get_parameter_or(name)
            return (start_defaults[name] if parameter.type_ == Parameter.Type.NOT_SET
                    else parameter.value)

        self._require_start_state = bool(self.get_parameter("require_start_state").value)
        self._start_joints = np.asarray(
            start_value("start_joint_positions"), dtype=np.float64)
        self._start_joint_tolerance = float(start_value("start_joint_tolerance_rad"))
        self._start_settle_s = float(start_value("start_joint_settle_s"))
        self._start_gripper_min = float(start_value("start_gripper_min"))
        self._gripper_timeout_s = float(
            self.get_parameter("gripper_state_timeout_s").value)
        self._gripper_command_min = float(
            self.get_parameter("gripper_command_min").value)
        self._require_close_approval = bool(
            self.get_parameter("require_close_approval").value)
        self._close_approval_timeout_s = float(
            self.get_parameter("close_approval_timeout_s").value)
        self._close_min_episode_time_s = float(
            self.get_parameter("close_min_episode_time_s").value)
        self._workspace_min = np.asarray(
            self.get_parameter("workspace_min").value, dtype=np.float64)
        self._workspace_max = np.asarray(
            self.get_parameter("workspace_max").value, dtype=np.float64)
        self._max_start_displacement_m = float(
            self.get_parameter("max_start_displacement_m").value)
        self._max_start_rise_m = float(
            self.get_parameter("max_start_rise_m").value)
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be positive and finite")
        if not np.isfinite(velocity_scale) or not 0.0 < velocity_scale <= 1.0:
            raise ValueError("velocity_scale must be in (0, 1]")
        if not np.isfinite(self._feedback_timeout_s) or self._feedback_timeout_s <= 0.0:
            raise ValueError("joint_state_timeout_s must be positive and finite")
        if not self._dry_run and not self._require_start_state:
            raise ValueError("real control requires require_start_state=true")
        if self._start_joints.shape != (7,) or not np.isfinite(self._start_joints).all():
            raise ValueError("start_joint_positions must contain seven finite radians")
        if (not np.isfinite(self._start_joint_tolerance)
                or self._start_joint_tolerance <= 0.0):
            raise ValueError("start_joint_tolerance_rad must be positive and finite")
        if not np.isfinite(self._start_settle_s) or self._start_settle_s <= 0.0:
            raise ValueError("start_joint_settle_s must be positive and finite")
        if not np.isfinite(self._start_gripper_min) or not 0.0 <= self._start_gripper_min <= 1.0:
            raise ValueError("start_gripper_min must be in [0, 1]")
        if not np.isfinite(self._gripper_timeout_s) or self._gripper_timeout_s <= 0.0:
            raise ValueError("gripper_state_timeout_s must be positive and finite")
        if not np.isfinite(self._gripper_command_min) or not 0.0 <= self._gripper_command_min <= 1.0:
            raise ValueError("gripper_command_min must be in [0, 1]")
        if not np.isfinite(self._close_approval_timeout_s) or self._close_approval_timeout_s <= 0.0:
            raise ValueError("close_approval_timeout_s must be positive and finite")
        if not np.isfinite(self._close_min_episode_time_s) or self._close_min_episode_time_s < 0.0:
            raise ValueError("close_min_episode_time_s must be nonnegative and finite")
        if (self._workspace_min.shape != (3,) or self._workspace_max.shape != (3,)
                or not np.isfinite(self._workspace_min).all()
                or not np.isfinite(self._workspace_max).all()
                or np.any(self._workspace_min >= self._workspace_max)):
            raise ValueError("workspace bounds must be finite ordered xyz triples")
        if (not np.isfinite(self._max_start_displacement_m)
                or self._max_start_displacement_m < 0
                or not np.isfinite(self._max_start_rise_m)
                or self._max_start_rise_m < 0):
            raise ValueError("start displacement and rise limits must be nonnegative")
        configured_urdf = str(self.get_parameter("urdf_path").value).strip()
        urdf_path = (
            Path(configured_urdf).expanduser() if configured_urdf else default_urdf
        )
        if not urdf_path.is_file():
            raise FileNotFoundError(f"RM75 URDF does not exist: {urdf_path}")
        self._joint_lower, self._joint_upper, velocities = urdf_limits(urdf_path)

        self._robot = placo.RobotWrapper(
            str(urdf_path), placo.Flags.ignore_collisions
        )
        for name, position in zip(JOINT_NAMES, self._start_joints):
            self._robot.set_joint(name, float(position))
        self._robot.update_kinematics()
        self._start_xyz = np.asarray(
            self._robot.get_T_world_frame(self._end_frame)[:3, 3],
            dtype=np.float64).copy()
        for name, velocity in zip(JOINT_NAMES, velocities):
            self._robot.set_velocity_limit(name, float(velocity * velocity_scale))
        self._solver = placo.KinematicsSolver(self._robot)
        self._solver.dt = 1.0 / rate_hz
        self._solver.mask_fbase(True)
        self._solver.enable_joint_limits(True)
        self._solver.enable_velocity_limits(True)
        self._robot.update_kinematics()
        self._effector_task = self._solver.add_frame_task(
            self._end_frame, self._robot.get_T_world_frame(self._end_frame)
        )
        self._effector_task.configure("policy_link7_target", "soft", 1.0)

        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._joint_publisher = self.create_publisher(
            Jointpos,
            str(self.get_parameter("joint_command_topic").value),
            command_qos,
        )
        self._gripper_publisher = self.create_publisher(
            Float32,
            str(self.get_parameter("gripper_command_topic").value),
            command_qos,
        )
        self._debug_publisher = self.create_publisher(
            JointState,
            str(self.get_parameter("debug_joint_topic").value),
            command_qos,
        )
        self._stop_publisher = self.create_publisher(
            Empty, str(self.get_parameter("stop_topic").value), command_qos
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._on_joint_state,
            command_qos,
        )
        self.create_subscription(
            Float32,
            str(self.get_parameter("gripper_state_topic").value),
            self._on_gripper_state,
            command_qos,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("close_approval_topic").value),
            self._on_close_approval,
            command_qos,
        )
        self.create_subscription(
            PolicyActionSequence,
            str(self.get_parameter("policy_topic").value),
            self._on_policy,
            command_qos,
        )
        self.create_service(
            ResetPolicyController,
            str(self.get_parameter("reset_service").value),
            self._on_reset,
        )
        self.create_timer(1.0 / rate_hz, self._tick)

        self._feedback_positions = None
        self._feedback_monotonic = float("-inf")
        self._gripper_state = None
        self._gripper_monotonic = float("-inf")
        self._close_allowed = False
        self._close_approval_monotonic = float("-inf")
        self._episode_started_at = None
        self._home_since = None
        self._start_verified_episode = None
        self._faulted_episode = None
        self._command_positions = None
        self._trajectory = None
        self._latest_episode = None
        self._latest_sequence = None
        self._minimum_episode = 0
        self._last_gripper = None
        self.get_logger().warning(
            "RM75 Placo controller ready: "
            f"dry_run={self._dry_run}, rate={rate_hz:g} Hz, "
            f"velocity_scale={velocity_scale:g}, "
            f"require_start_state={self._require_start_state}"
        )

    def _feedback_is_fresh(self) -> bool:
        return self._feedback_positions is not None and (
            time.monotonic() - self._feedback_monotonic
            <= self._feedback_timeout_s
        )

    def _set_robot_joints(self, positions: np.ndarray) -> None:
        for name, position in zip(JOINT_NAMES, positions):
            self._robot.set_joint(name, float(position))
        self._robot.update_kinematics()

    def _on_joint_state(self, message: JointState) -> None:
        try:
            positions = ordered_joint_positions(message.name, message.position)
            if np.any(positions < self._joint_lower) or np.any(
                positions > self._joint_upper
            ):
                raise ValueError("joint feedback is outside URDF limits")
        except (TypeError, ValueError) as error:
            self.get_logger().warning(
                f"Invalid /joint_states feedback: {error}",
                throttle_duration_sec=1.0,
            )
            if self._trajectory is not None:
                self._stop_control("invalid joint feedback")
            self._home_since = None
            return
        self._feedback_positions = positions
        self._feedback_monotonic = time.monotonic()
        if self._start_verified_episode is None:
            home_error = np.max(np.abs(positions - self._start_joints))
            if home_error <= self._start_joint_tolerance:
                if self._home_since is None:
                    self._home_since = self._feedback_monotonic
            else:
                self._home_since = None
        if self._trajectory is None and self._command_positions is None:
            self._set_robot_joints(positions)

    def _on_gripper_state(self, message: Float32) -> None:
        """保存归一化夹爪反馈，供实机首条策略的起始状态门控。"""
        openness = float(message.data)
        if np.isfinite(openness) and 0.0 <= openness <= 1.0:
            self._gripper_state = openness
            self._gripper_monotonic = time.monotonic()
        else:
            self._gripper_state = None

    def _on_close_approval(self, message: Bool) -> None:
        """保存视觉近端许可，离开夹爪中心或图像缺流时自动失效。"""
        self._close_allowed = bool(message.data)
        self._close_approval_monotonic = time.monotonic()

    def _close_is_allowed(self) -> bool:
        """闭合必须同时满足视觉许可、消息新鲜和本轮最短接近时间。"""
        if not self._require_close_approval:
            return True
        now = time.monotonic()
        return bool(
            self._close_allowed
            and now - self._close_approval_monotonic <= self._close_approval_timeout_s
            and self._episode_started_at is not None
            and now - self._episode_started_at >= self._close_min_episode_time_s
            and self._gripper_state is not None
            and now - self._gripper_monotonic <= self._gripper_timeout_s
        )

    def _start_state_ready(self, episode_id: int) -> bool:
        """每个 episode 仅在七轴已回位、静止且夹爪张开后允许首次实机控制。"""
        if self._dry_run or not self._require_start_state:
            return True
        if self._start_verified_episode == episode_id:
            return True
        now = time.monotonic()
        joint_error = np.max(np.abs(self._feedback_positions - self._start_joints))
        ready = (joint_error <= self._start_joint_tolerance
                 and self._home_since is not None
                 and now - self._home_since >= self._start_settle_s
                 and self._gripper_state is not None
                 and now - self._gripper_monotonic <= self._gripper_timeout_s
                 and self._gripper_state >= self._start_gripper_min)
        if not ready:
            self.get_logger().warning(
                "Policy ignored until RM75 and gripper return to the start state "
                f"(max joint error={np.rad2deg(joint_error):.1f} deg, "
                f"gripper={self._gripper_state})",
                throttle_duration_sec=1.0,
            )
            return False
        self._start_verified_episode = episode_id
        return True

    def _on_reset(self, request, response):
        """同步停止当前轨迹，并只接受本次重置后的 episode。"""
        self._stop_control("episode reset")
        episode_id = int(request.episode_id)
        if episode_id < self._minimum_episode or (
                self._latest_episode is not None and episode_id <= self._latest_episode):
            response.success = False
            response.message = "reset episode must exceed the last accepted episode"
            return response
        self._minimum_episode = episode_id
        self._start_verified_episode = None
        self._home_since = None
        self._faulted_episode = None
        self._episode_started_at = None
        self._close_allowed = False
        response.success = True
        response.message = f"Stopped controller before episode {episode_id}"
        return response

    def _on_policy(self, message: PolicyActionSequence) -> None:
        episode_id, sequence_id = int(message.episode_id), int(message.sequence_id)
        if episode_id < self._minimum_episode:
            return
        if self._latest_episode is not None and episode_id > self._latest_episode:
            self._stop_control("new episode requires return to start state")
            self._latest_episode = episode_id
            self._latest_sequence = -1
            self._start_verified_episode = None
            self._home_since = None
            self._faulted_episode = None
            self._episode_started_at = None
            self._close_allowed = False
        if not sequence_is_new(
            self._latest_episode,
            self._latest_sequence,
            episode_id,
            sequence_id,
        ):
            return
        if self._faulted_episode == episode_id:
            self.get_logger().warning(
                "Policy ignored after a workspace, gripper, or IK fault; reset the episode "
                "after returning RM75 and gripper to the start state",
                throttle_duration_sec=1.0,
            )
            return
        if not self._feedback_is_fresh():
            self.get_logger().warning(
                "Policy ignored because /joint_states feedback is missing or stale",
                throttle_duration_sec=1.0,
            )
            return
        if not self._start_state_ready(episode_id):
            return
        # 新序列以最新实测关节为锚点，避免连续预测时沿已发出但尚未到达的命令累积误差。
        self._set_robot_joints(self._feedback_positions)
        anchor_transform = self._robot.get_T_world_frame(self._end_frame).copy()
        try:
            trajectory = decode_trajectory(
                message,
                self.get_clock().now().nanoseconds,
                anchor_transform,
                self._last_gripper,
                base_frame=self._base_frame,
                end_frame=self._end_frame,
            )
        except (TypeError, ValueError) as error:
            self.get_logger().warning(f"Policy sequence rejected: {error}")
            return
        if not self._dry_run:
            try:
                validate_workspace(
                    np.vstack((anchor_transform[:3, 3], trajectory.positions)),
                    self._workspace_min, self._workspace_max)
                validate_gripper_targets(
                    trajectory.grippers, self._gripper_command_min)
            except ValueError as error:
                self.get_logger().error(f"Policy sequence rejected: {error}")
                self._stop_control("policy safety limit")
                self._start_verified_episode = None
                self._home_since = None
                self._faulted_episode = episode_id
                return
        self._trajectory = trajectory
        self._latest_episode = episode_id
        self._latest_sequence = sequence_id
        if self._episode_started_at is None:
            self._episode_started_at = time.monotonic()
        if self._command_positions is None:
            self._command_positions = self._feedback_positions.copy()

    def _publish(self, positions: np.ndarray, gripper: float) -> None:
        debug = JointState()
        debug.header.stamp = self.get_clock().now().to_msg()
        debug.name = list(JOINT_NAMES)
        debug.position = positions.tolist()
        self._debug_publisher.publish(debug)
        if self._dry_run:
            return
        joint_command = Jointpos()
        fill_joint_command(joint_command, positions)
        self._joint_publisher.publish(joint_command)
        self._gripper_publisher.publish(Float32(data=float(gripper)))

    def _stop_control(self, reason: str) -> None:
        if self._trajectory is None:
            return
        self._trajectory = None
        self._command_positions = None
        self._last_gripper = None
        if not self._dry_run:
            self._stop_publisher.publish(Empty())
        self.get_logger().warning(f"RM75 control stopped: {reason}")

    def _tick(self) -> None:
        if self._trajectory is None:
            return
        if not self._feedback_is_fresh():
            self._stop_control("/joint_states feedback timed out")
            return
        now_ns = self.get_clock().now().nanoseconds
        if now_ns > int(self._trajectory.times_ns[-1]):
            self._stop_control("policy sequence completed")
            return
        position, quaternion, gripper = sample_trajectory(
            self._trajectory, now_ns
        )
        gripper = gate_gripper_target(
            gripper, self._last_gripper, self._gripper_state,
            self._close_is_allowed())
        self._effector_task.T_world_frame = pose_matrix(position, quaternion)
        try:
            self._solver.solve(True)
            self._robot.update_kinematics()
            positions = np.asarray(
                [self._robot.get_joint(name) for name in JOINT_NAMES],
                dtype=np.float64,
            )
            if not np.isfinite(positions).all():
                raise ValueError("IK returned non-finite joints")
            tolerance = 1.0e-7
            if np.any(positions < self._joint_lower - tolerance) or np.any(
                positions > self._joint_upper + tolerance
            ):
                raise ValueError("IK returned joints outside URDF limits")
            solved_xyz = self._robot.get_T_world_frame(self._end_frame)[:3, 3]
            validate_start_envelope(
                solved_xyz, self._start_xyz,
                self._max_start_displacement_m, self._max_start_rise_m)
            if not self._dry_run:
                validate_workspace(
                    solved_xyz,
                    self._workspace_min, self._workspace_max)
        except Exception as error:
            self.get_logger().error(f"Placo command rejected: {error}")
            self._stop_control("IK or trial envelope failure")
            self._start_verified_episode = None
            self._home_since = None
            self._faulted_episode = self._latest_episode
            return
        self._command_positions = positions.copy()
        self._last_gripper = float(gripper)
        self._publish(positions, gripper)

    def destroy_node(self) -> bool:
        if self._trajectory is not None:
            self._stop_control("node shutdown")
            if not self._dry_run:
                # Give the driver time to acknowledge the final stop before DDS teardown.
                self._stop_publisher.wait_for_all_acked(Duration(seconds=0.2))
        return super().destroy_node()


def main(args=None) -> None:
    """启动 RM75 Placo 控制节点。"""
    # Keep ROS alive through destroy_node(): the default rclpy SIGINT handler
    # shuts its context down before the final move_stop command can be sent.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    previous_int_handler = signal.getsignal(signal.SIGINT)
    previous_term_handler = signal.getsignal(signal.SIGTERM)
    shutdown_requested = False

    def handle_shutdown(_signum, _frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    node = None
    try:
        node = Rm75PlacoController()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        # A signal can interrupt rclpy's C message conversion before it raises
        # KeyboardInterrupt. Only suppress that known teardown error on shutdown.
        if not shutdown_requested or "Unable to convert call argument to Python object" not in str(error):
            raise
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
            signal.signal(signal.SIGINT, previous_int_handler)
            signal.signal(signal.SIGTERM, previous_term_handler)


if __name__ == "__main__":
    main()
