"""RM75 原生关节序列执行器，复用既有驱动停车与回位服务协议。"""

import json
from pathlib import Path
import signal
import time

from ament_index_python.packages import get_package_share_directory
from fastumi_interfaces.msg import PolicyJointActionSequence
from fastumi_interfaces.srv import ResetPolicyController
import numpy as np
import placo
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rm_ros_interfaces.msg import Jointpos, Movej
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float32, String

from fastumi_rm75.joint_control import (
    decode_joint_trajectory, limit_joint_step, sample_joint_trajectory, validate_joints,
)
from fastumi_rm75.placo_control import (
    JOINT_NAMES, sequence_is_new, stamp_to_ns, urdf_limits, validate_workspace,
)
from fastumi_rm75.rm75_placo_controller import Rm75PlacoController


class Rm75JointController(Rm75PlacoController):
    """仅复用反馈与 start/stop/home 方法；不构造或调用 IK solver。"""

    def __init__(self):
        Node.__init__(self, "rm75_joint_controller")
        root = Path(get_package_share_directory("fastumi_rm75"))
        home = np.deg2rad([0, 20, 0, 70, 0, 90, 90]).tolist()
        defaults = {
            "dry_run": True, "urdf_path": str(root / "assets/rm_75_kinematic.urdf"),
            "control_rate_hz": 50.0, "velocity_scale": 0.25,
            "joint_state_timeout_s": 0.25, "gripper_state_timeout_s": 0.25,
            "start_joint_positions": home, "start_joint_tolerance_rad": 0.035,
            "start_joint_settle_s": 0.5, "start_gripper_min": 0.95,
            "workspace_min": [-0.6, -0.5, 0.16], "workspace_max": [0.6, 0.5, 0.50],
            "joint_state_topic": "/joint_states", "gripper_state_topic": "/motion_control/gripper_state",
            "policy_topic": "/fastumi/policy/joint_action_sequence",
            "joint_command_topic": "/rm_driver/movej_canfd_cmd",
            "gripper_command_topic": "/motion_control/gripper_command",
            "debug_joint_topic": "/fastumi/rm75/joint/joint_command",
            "first_sequence_timeout_s": 2.0,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        get = lambda name: self.get_parameter(name).value
        self._dry_run = get("dry_run")
        self._period = 1 / get("control_rate_hz")
        self._feedback_timeout_s, self._gripper_timeout_s = get("joint_state_timeout_s"), get("gripper_state_timeout_s")
        self._start_joints = np.array(get("start_joint_positions"))
        self._start_joint_tolerance = get("start_joint_tolerance_rad")
        self._start_settle_s, self._start_gripper_min = get("start_joint_settle_s"), get("start_gripper_min")
        self._workspace_min, self._workspace_max = np.array(get("workspace_min")), np.array(get("workspace_max"))
        for name in ("control_rate_hz", "velocity_scale", "joint_state_timeout_s", "gripper_state_timeout_s",
                     "start_joint_tolerance_rad", "start_joint_settle_s", "first_sequence_timeout_s"):
            if not np.isfinite(get(name)) or get(name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if get("velocity_scale") > 1 or not 0 <= self._start_gripper_min <= 1:
            raise ValueError("invalid speed scale or start gripper")
        if (self._workspace_min.shape != (3,) or self._workspace_max.shape != (3,)
                or not np.isfinite([self._workspace_min, self._workspace_max]).all()
                or np.any(self._workspace_min >= self._workspace_max)):
            raise ValueError("invalid workspace bounds")
        self._joint_lower, self._joint_upper, velocities = urdf_limits(Path(get("urdf_path")))
        validate_joints(self._start_joints, self._joint_lower, self._joint_upper)
        self._velocities = velocities * get("velocity_scale")
        self._robot = placo.RobotWrapper(get("urdf_path"), placo.Flags.ignore_collisions)
        self._end_frame = "Link7"
        self._base_frame = "base_link"
        self._set_robot_joints(self._start_joints)
        validate_workspace(self._robot.get_T_world_frame("Link7")[:3, 3], self._workspace_min, self._workspace_max)
        self._task_control_enabled, self._require_start_state = True, True
        self._task_enabled, self._enabled_episode = False, None
        self._task_io_group = ReentrantCallbackGroup()
        # Existing asynchronous stop/home handlers operate on these driver states.
        self._feedback_positions, self._gripper_state = None, None
        self._feedback_monotonic, self._gripper_monotonic = -np.inf, -np.inf
        self._home_since, self._start_verified_episode, self._faulted_episode = None, None, None
        self._command_positions, self._trajectory, self._last_gripper = None, None, None
        self._latest_episode, self._latest_sequence, self._minimum_episode = None, None, 0
        self._stop_future, self._hold_confirmed = None, True
        self._home_future, self._home_episode, self._home_started_at = None, None, None
        self._home_command_sent_at, self._home_accepted, self._home_settled_since = None, False, None
        self._valid_feedback, self._valid_feedback_at = False, -np.inf
        self._episode_started_at, self._close_allowed = None, False
        self._last_tick_ns, self._last_clock_ns = None, None
        self._sequence_timing = {}
        self._reason = "idle"
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self._debug_publisher = self.create_publisher(JointState, get("debug_joint_topic"), qos)
        self._status_publisher = self.create_publisher(String, "/fastumi/rm75/joint/status", qos)
        # Dry-run has no publishers to any driver motion or gripper command topic.
        self._joint_publisher = self._gripper_publisher = self._stop_publisher = self._movej_publisher = None
        if not self._dry_run:
            self._joint_publisher = self.create_publisher(Jointpos, get("joint_command_topic"), qos)
            self._gripper_publisher = self.create_publisher(Float32, get("gripper_command_topic"), qos)
            self._stop_publisher = self.create_publisher(Empty, "/rm_driver/move_stop_cmd", qos)
            self._movej_publisher = self.create_publisher(Movej, "/rm_driver/movej_cmd", qos)
        for message_type, topic, callback in (
            (JointState, get("joint_state_topic"), self._on_joint_state),
            (Float32, get("gripper_state_topic"), self._on_gripper_state),
            (PolicyJointActionSequence, get("policy_topic"), self._on_policy),
            (Bool, "/rm_driver/udp_feedback_valid", self._on_udp_valid),
            (Bool, "/rm_driver/movej_result", self._on_movej_result),
            (Bool, "/rm_driver/move_stop_result", self._on_move_stop_result),
        ):
            self.create_subscription(message_type, topic, callback, qos)
        for name, callback in (("start_task", self._on_start_task), ("reset_episode", self._on_reset_service),
                               ("return_to_start", self._on_return_to_start)):
            self.create_service(ResetPolicyController, "/fastumi/rm75/joint/" + name, callback,
                                callback_group=self._task_io_group)
        self.create_timer(self._period, self._tick)
        self.create_timer(0.05, self._home_tick)
        self.create_timer(0.1, self._status)

    def _feedback_is_fresh(self):
        now = time.monotonic()
        return (self._feedback_positions is not None and now - self._feedback_monotonic <= self._feedback_timeout_s
                and self._gripper_state is not None and now - self._gripper_monotonic <= self._gripper_timeout_s
                and self._valid_feedback and now - self._valid_feedback_at <= 0.5)

    def _start_state_ready(self, episode_id):
        if self._start_verified_episode == episode_id:
            return True
        ready = (self._feedback_is_fresh() and self._home_since is not None
                 and time.monotonic() - self._home_since >= self._start_settle_s
                 and np.max(np.abs(self._feedback_positions - self._start_joints)) <= self._start_joint_tolerance
                 and self._gripper_state >= self._start_gripper_min)
        if ready:
            self._start_verified_episode = episode_id
        return ready

    def _on_start_task(self, request, response):
        response = super()._on_start_task(request, response)
        if response.success:
            self._sequence_timing = {}
            self._episode_started_at = time.monotonic()
            self._reason = "waiting for first prediction"
        return response

    def _hold_gripper(self):
        return True if self._dry_run else super()._hold_gripper()

    def _on_policy(self, message):
        episode, sequence = int(message.episode_id), int(message.sequence_id)
        if not self._task_enabled or episode != self._enabled_episode:
            return
        if not sequence_is_new(self._latest_episode, self._latest_sequence, episode, sequence):
            return
        try:
            if not self._feedback_is_fresh():
                raise ValueError("feedback is stale")
            received_ns = self.get_clock().now().nanoseconds
            trajectory = decode_joint_trajectory(message, received_ns,
                self._feedback_positions, self._last_gripper if self._last_gripper is not None else self._gripper_state,
                self._joint_lower, self._joint_upper)
            for q in trajectory.positions:
                self._set_robot_joints(q)
                validate_workspace(self._robot.get_T_world_frame("Link7")[:3, 3], self._workspace_min, self._workspace_max)
        except (ValueError, TypeError) as error:
            self._stop_control(f"invalid policy sequence: {error}")
            return
        self._trajectory = trajectory
        previous = self._sequence_timing.get("received_ns")
        self._sequence_timing = {
            "episode_id": episode, "sequence_id": sequence,
            "source_ns": stamp_to_ns(message.header.stamp), "received_ns": received_ns,
            "end_ns": int(trajectory.times_ns[-1]),
            "age_on_receive_s": (received_ns - stamp_to_ns(message.header.stamp)) / 1e9,
            "receive_interval_s": None if previous is None else (received_ns - previous) / 1e9,
        }
        self._latest_episode, self._latest_sequence = episode, sequence
        if self._command_positions is None:
            self._command_positions = self._feedback_positions.copy()
        self._reason = "executing prediction"

    def _stop_control(self, reason):
        active = self._task_enabled or self._trajectory is not None
        self._trajectory, self._command_positions, self._last_gripper = None, None, None
        self._task_enabled, self._enabled_episode = False, None
        self._last_tick_ns = None
        self._start_verified_episode, self._home_since = None, None
        self._reason = reason
        if active and not self._dry_run:
            self._stop_publisher.publish(Empty())
            self._hold_gripper()
        if active:
            self.get_logger().warning(f"RM75 joint control stopped: {reason}")
            timing = getattr(self, "_sequence_timing", {})
            if timing:
                now = self.get_clock().now().nanoseconds
                self.get_logger().warning("Last joint prediction: " + json.dumps({
                    **timing, "source_age_s": (now - timing["source_ns"]) / 1e9,
                    "since_receive_s": (now - timing["received_ns"]) / 1e9,
                    "horizon_remaining_s": (timing["end_ns"] - now) / 1e9,
                }))

    def _tick(self):
        now = self.get_clock().now().nanoseconds
        if self._last_clock_ns is not None and now < self._last_clock_ns:
            self._stop_control("ROS clock moved backwards")
            self._finish_home(False, "Clock moved backwards")
        self._last_clock_ns = now
        if not self._task_enabled:
            return
        if not self._feedback_is_fresh():
            self._stop_control("joint, gripper or UDP feedback timed out")
            return
        if self._trajectory is None:
            if time.monotonic() - self._episode_started_at > self.get_parameter("first_sequence_timeout_s").value:
                self._stop_control("first prediction timed out")
            return
        try:
            target, gripper = sample_joint_trajectory(self._trajectory, now)
            dt = self._period if self._last_tick_ns is None else (now - self._last_tick_ns) / 1e9
            positions = limit_joint_step(target, self._command_positions, self._velocities, dt, self._period)
            validate_joints(positions, self._joint_lower, self._joint_upper)
            self._set_robot_joints(positions)
            validate_workspace(self._robot.get_T_world_frame("Link7")[:3, 3], self._workspace_min, self._workspace_max)
        except Exception as error:
            self._stop_control(str(error))
            return
        self._last_tick_ns = now
        self._command_positions, self._last_gripper = positions.copy(), gripper
        self._publish(positions, gripper)

    def _status(self):
        self._status_publisher.publish(String(data=json.dumps({
            "episode_id": self._minimum_episode, "enabled": self._task_enabled,
            "reason": self._reason, "dry_run": self._dry_run,
        })))

    def destroy_node(self):
        self._finish_home(False, "Controller shutdown")
        self._stop_control("controller shutdown")
        if not self._dry_run:
            self._stop_publisher.wait_for_all_acked(Duration(seconds=0.2))
        return Node.destroy_node(self)


def main(args=None):
    """保留 ROS 上下文到最终停车消息发送完毕。"""
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    def interrupt(_sig, _frame):
        raise KeyboardInterrupt
    for sig in old:
        signal.signal(sig, interrupt)
    node = None
    try:
        node = Rm75JointController()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for sig, handler in old.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
