"""使用 Placo 把绝对 Link7 策略序列转换成 RM75 七轴透传命令。"""

from __future__ import annotations

from pathlib import Path
import time

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
from fastumi_interfaces.msg import PolicyActionSequence
import numpy as np
import placo
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rm_ros_interfaces.msg import Jointpos
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float32

from fastumi_rm75.placo_control import (
    JOINT_NAMES,
    decode_trajectory,
    fill_joint_command,
    ordered_joint_positions,
    pose_matrix,
    sample_trajectory,
    sequence_is_new,
    urdf_limits,
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
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("end_frame", "Link7")
        self.declare_parameter("joint_state_topic", "/joint_states")
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

        self._dry_run = bool(self.get_parameter("dry_run").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._end_frame = str(self.get_parameter("end_frame").value)
        rate_hz = float(self.get_parameter("control_rate_hz").value)
        velocity_scale = float(self.get_parameter("velocity_scale").value)
        self._feedback_timeout_s = float(
            self.get_parameter("joint_state_timeout_s").value
        )
        if not np.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be positive and finite")
        if not np.isfinite(velocity_scale) or not 0.0 < velocity_scale <= 1.0:
            raise ValueError("velocity_scale must be in (0, 1]")
        if not np.isfinite(self._feedback_timeout_s) or self._feedback_timeout_s <= 0.0:
            raise ValueError("joint_state_timeout_s must be positive and finite")
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
            PolicyActionSequence,
            str(self.get_parameter("policy_topic").value),
            self._on_policy,
            command_qos,
        )
        self.create_timer(1.0 / rate_hz, self._tick)

        self._feedback_positions = None
        self._feedback_monotonic = float("-inf")
        self._command_positions = None
        self._trajectory = None
        self._latest_episode = None
        self._latest_sequence = None
        self._last_gripper = None
        self.get_logger().warning(
            "RM75 Placo controller ready: "
            f"dry_run={self._dry_run}, rate={rate_hz:g} Hz, "
            f"velocity_scale={velocity_scale:g}"
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
            return
        self._feedback_positions = positions
        self._feedback_monotonic = time.monotonic()
        if self._trajectory is None and self._command_positions is None:
            self._set_robot_joints(positions)

    def _on_policy(self, message: PolicyActionSequence) -> None:
        episode_id, sequence_id = int(message.episode_id), int(message.sequence_id)
        if not sequence_is_new(
            self._latest_episode,
            self._latest_sequence,
            episode_id,
            sequence_id,
        ):
            return
        if not self._feedback_is_fresh():
            self.get_logger().warning(
                "Policy ignored because /joint_states feedback is missing or stale",
                throttle_duration_sec=1.0,
            )
            return
        if self._command_positions is None:
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
        self._trajectory = trajectory
        self._latest_episode = episode_id
        self._latest_sequence = sequence_id
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
        except Exception as error:
            self.get_logger().error(f"Placo IK failed: {error}")
            self._stop_control("IK failure")
            return
        self._command_positions = positions.copy()
        self._last_gripper = float(gripper)
        self._publish(positions, gripper)

    def destroy_node(self) -> bool:
        if self._trajectory is not None:
            self._stop_control("node shutdown")
        return super().destroy_node()


def main(args=None) -> None:
    """启动 RM75 Placo 控制节点。"""
    rclpy.init(args=args)
    node = None
    try:
        node = Rm75PlacoController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
