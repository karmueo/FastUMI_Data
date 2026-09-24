"""验证遥操 ROS 接口、跳变恢复状态和面板提示契约。"""

from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from fastumi_interfaces.msg import GripperState, TrackerStatus
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool, Trigger

from tracker_teleoperated.core import (
    PoseStreamValidator,
    parse_home_joint_positions,
)
from tracker_teleoperated.gripper_follow import GripperFollower
from tracker_teleoperated.node import (
    TrackerTeleopNode,
    build_home_command,
    build_joint_command,
    reorder_joint_state,
)


def make_pose(position=(0.0, 0.0, 0.0), rpy_deg=(0.0, 0.0, 0.0)):
    """构造节点参考状态测试使用的有限位姿。"""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler(
        "xyz", rpy_deg, degrees=True
    ).as_matrix()
    pose[:3, 3] = position
    return pose


class FakeKinematics:
    """提供可控末端位姿或异常的测试运动学实现。"""

    def __init__(self, pose):
        """保存下一次正运动学调用的返回值。"""
        self.pose = pose
        self.error = None

    def end_effector_pose(self, _positions):
        """返回配置的末端位姿，或抛出配置的异常。"""
        if self.error is not None:
            raise self.error
        return self.pose.copy()

    def solve(self, target_pose, _seed_positions):
        """将目标的基座 X 位移映射成可观察的七轴测试指令。"""
        if self.error is not None:
            raise self.error
        return np.full(7, target_pose[0, 3])


class ReferenceState:
    """承载参考采集方法需要的最小节点状态。"""

    def __init__(self):
        """创建健康输入和空参考状态。"""
        self._latest_tracker_pose = make_pose(rpy_deg=(10.0, 20.0, 30.0))
        self._latest_joint_positions = np.arange(7, dtype=np.float64)
        self._kinematics = FakeKinematics(
            make_pose(rpy_deg=(5.0, 40.0, -20.0))
        )
        self._axis_mapping = np.eye(3)
        self._mapping_mode = "reference_eef"
        self._mapping_basis = None
        self._reference_tracker_pose = None
        self._reference_eef_pose = None
        self._filtered_tracker_pose = None
        self._command_positions = None
        self._command_velocity = np.ones(7)
        self._last_tick_monotonic = 0.0

    def _freshness_error(self, _now):
        """模拟通过全部输入健康检查。"""
        return None


class FakeLogger:
    """提供节点状态测试需要的最小日志接口。"""

    def warning(self, _message):
        """忽略测试中的告警文本。"""

    def info(self, _message):
        """忽略测试中的普通日志文本。"""

    def error(self, _message):
        """忽略测试中的错误日志文本。"""


class FakePublisher:
    """记录测试期间发布的 ROS 消息。"""

    def __init__(self):
        """创建空消息记录。"""
        self.messages = []

    def publish(self, message):
        """保存一条已发布消息。"""
        self.messages.append(message)


class CommandPublishingState:
    """承载指令发布测试所需的最小节点状态。"""

    def __init__(self):
        """创建可记录调试目标和驱动命令的发布状态。"""
        self._command_publisher = FakePublisher()
        self.joint_targets = []

    def _publish_joint_target(self, positions):
        """记录发布到调试话题的关节目标。"""
        self.joint_targets.append(np.asarray(positions).copy())


class FakeGenerationStore:
    """在原有控制回归测试中记录代次，不访问文件系统。"""

    def __init__(self):
        """创建初始代次。"""
        self.record = {"generation": 0, "success": False, "code": "", "message": ""}

    def begin(self, generation, _operation):
        """记录新代次。"""
        self.record["generation"] = generation
        return "new"

    def finish(self, success, code, message):
        """记录服务响应。"""
        self.record.update(success=success, code=code, message=message)


class WorkspaceCalibrationState:
    """承载三点工作空间标定回调需要的最小节点状态。"""

    _cancel_workspace_calibration = (
        TrackerTeleopNode._cancel_workspace_calibration
    )
    _stop_homing = TrackerTeleopNode._stop_homing
    _publish_status = TrackerTeleopNode._publish_status

    def __init__(self):
        """创建暂停、输入健康且尚未标定的节点状态。"""
        self._mapping_mode = "workspace"
        self._workspace_minimum_angle_deg = 60.0
        self._enabled = False
        self._generation_store = FakeGenerationStore()
        self._generation_fault = False
        self._homing = False
        self._latest_tracker_pose = make_pose()
        self._workspace_calibration_samples = []
        self._mapping_basis = None
        self._reference_tracker_pose = make_pose((1.0, 1.0, 1.0))
        self._reference_eef_pose = make_pose((2.0, 2.0, 2.0))
        self._filtered_tracker_pose = make_pose((3.0, 3.0, 3.0))
        self._logger = FakeLogger()
        self.health_error = None
        self.save_error = None
        self.saved_mapping = None
        self._status_publisher = FakePublisher()

    def _tracker_freshness_error(self, _now):
        """返回测试配置的输入健康错误。"""
        return self.health_error

    def get_logger(self):
        """返回空实现日志器。"""
        return self._logger

    def _disable(self, _reason, publish_hold, **_kwargs):
        """模拟暂停节点并记录是否请求保持指令。"""
        self._enabled = False
        self.publish_hold = publish_hold

    def _save_workspace_mapping(self, mapping_basis):
        """记录候选标定方向，或模拟持久化失败。"""
        if self.save_error is None:
            self.saved_mapping = np.asarray(mapping_basis).copy()
        return self.save_error


class HomeState:
    """承载启动姿态记录和 MoveJ 回位回调需要的最小节点状态。"""

    _cancel_workspace_calibration = (
        TrackerTeleopNode._cancel_workspace_calibration
    )
    _clear_control_reference = TrackerTeleopNode._clear_control_reference
    _stop_homing = TrackerTeleopNode._stop_homing
    _publish_status = TrackerTeleopNode._publish_status

    def __init__(self):
        """创建具有新鲜反馈和已记录初始位姿的暂停状态。"""
        self._enabled = False
        self._generation_store = FakeGenerationStore()
        self._generation_fault = False
        self._homing = False
        self._home_started_monotonic = 0.0
        self._home_command_due_monotonic = 0.0
        self._pending_home_command = None
        self._home_timeout_s = 30.0
        self._home_command_quiet_period_s = 0.20
        self._home_speed_percent = 20
        self._feedback_timeout_s = 0.25
        self._feedback_freeze_timeout_s = 0.25
        self._latest_joint_stamp_ns = None
        self._heartbeat_timeout_s = 0.50
        self._latest_heartbeat_monotonic = time.monotonic()
        self._latest_status_stamp_ns = None
        self._pending_tracker_sample = None
        self._latest_joint_monotonic = time.monotonic()
        self._latest_joint_positions = np.full(7, 0.2)
        self._home_joint_positions = np.linspace(-0.3, 0.3, 7)
        self._workspace_calibration_samples = []
        self._mapping_basis = Rotation.from_euler(
            "z", 20.0, degrees=True
        ).as_matrix()
        self._reference_tracker_pose = make_pose()
        self._reference_eef_pose = make_pose()
        self._filtered_tracker_pose = make_pose()
        self._command_positions = np.zeros(7)
        self._command_velocity = np.ones(7)
        self._last_tick_monotonic = time.monotonic()
        self._home_publisher = FakePublisher()
        self._move_stop_publisher = FakePublisher()
        self._logger = FakeLogger()
        self.joint_targets = []
        self.publish_hold = None
        self._status_publisher = FakePublisher()
        self.commands = []

    def get_logger(self):
        """返回空实现日志器。"""
        return self._logger

    def _disable(self, _reason, publish_hold, **_kwargs):
        """模拟暂停遥操。"""
        self._enabled = False
        self.publish_hold = publish_hold

    def _publish_joint_target(self, positions):
        """记录调试关节目标。"""
        self.joint_targets.append(np.asarray(positions).copy())

    def _publish_command(self, positions):
        """记录测试中意外发出的 CANFD 关节目标。"""
        self.commands.append(np.asarray(positions).copy())

    def _update_gripper(self, _enabled, _now):
        """忽略回位状态测试无关的夹爪更新。"""


class TrackerInputState(WorkspaceCalibrationState):
    """使用真实输入校验和暂停逻辑验证跳变后的控制与标定状态。"""

    _accept_tracker_pose = TrackerTeleopNode._accept_tracker_pose
    _disable = TrackerTeleopNode._disable
    _clear_control_reference = TrackerTeleopNode._clear_control_reference
    _publish_enabled = TrackerTeleopNode._publish_enabled
    _update_gripper = TrackerTeleopNode._update_gripper
    _capture_reference = TrackerTeleopNode._capture_reference
    _freshness_error = TrackerTeleopNode._freshness_error
    _mapping_recovery_instruction = (
        TrackerTeleopNode._mapping_recovery_instruction
    )

    def __init__(self):
        """创建已标定、已启用且关节反馈新鲜的遥操状态。"""
        super().__init__()
        self._mapping_basis = np.eye(3)
        self._kinematics = FakeKinematics(make_pose((0.4, 0.0, 0.2)))
        self._enabled = True
        self._pose_validator = PoseStreamValidator(recovery_samples=3)
        self._latest_joint_positions = np.zeros(7)
        self._latest_joint_monotonic = time.monotonic()
        self._last_tick_monotonic = time.monotonic()
        self._feedback_timeout_s = 0.50
        self._feedback_freeze_timeout_s = 0.25
        self._feedback_frozen = False
        self._heartbeat_timeout_s = 0.50
        self._latest_heartbeat_monotonic = time.monotonic()
        self._latest_status_stamp_ns = None
        self._pending_tracker_sample = None
        self._command_positions = np.zeros(7)
        self._command_velocity = np.zeros(7)
        self._enabled_publisher = FakePublisher()
        self._gripper_follower = GripperFollower(0.25, 0.25)
        self._gripper_command_publisher = FakePublisher()
        # 记录保持目标，验证恢复输入不会自动继续驱动机械臂。
        self.commands = []
        self.targets = []

    def _publish_command(self, positions):
        """记录指令，避免向实机发送数据。"""
        self.commands.append(np.asarray(positions).copy())

    def _publish_target_pose(self, target):
        """记录控制周期求出的末端目标。"""
        self.targets.append(np.asarray(target).copy())


class GripperBridgeState:
    """承载夹爪 ROS 回调和状态发布所需的最小节点状态。"""

    _update_gripper = TrackerTeleopNode._update_gripper
    _publish_status = TrackerTeleopNode._publish_status

    def __init__(self):
        """创建默认暂停且尚无夹爪输入的模拟节点。"""
        self._enabled = False
        self._feedback_frozen = False
        self._gripper_follower = GripperFollower(0.25, 0.25)
        self._gripper_command_publisher = FakePublisher()
        self._status_publisher = FakePublisher()


def test_gripper_ros_callbacks_hold_and_resume_without_pausing_arm():
    """确认状态接口输出实测保持值，预测恢复后机械臂保持启用。"""
    state = GripperBridgeState()
    TrackerTeleopNode._gripper_feedback_callback(state, Float32(data=0.3))
    assert state._gripper_command_publisher.messages[-1].data == pytest.approx(
        0.3
    )

    predicted = GripperState()
    predicted.valid = True
    predicted.filtered_openness = 0.8
    TrackerTeleopNode._gripper_estimate_callback(state, predicted)
    state._enabled = True
    TrackerTeleopNode._update_gripper(state, True, time.monotonic())
    assert state._gripper_command_publisher.messages[-1].data == pytest.approx(
        0.8
    )

    TrackerTeleopNode._gripper_feedback_callback(state, Float32(data=0.45))
    predicted.valid = False
    predicted.filtered_openness = float("nan")
    TrackerTeleopNode._gripper_estimate_callback(state, predicted)
    assert state._gripper_command_publisher.messages[-1].data == pytest.approx(
        0.45
    )
    assert state._enabled
    assert "夹爪预测无效" in state._status_publisher.messages[-1].data

    predicted.valid = True
    predicted.filtered_openness = 0.2
    TrackerTeleopNode._gripper_estimate_callback(state, predicted)
    TrackerTeleopNode._update_gripper(state, True, time.monotonic())
    assert state._gripper_command_publisher.messages[-1].data == pytest.approx(
        0.2
    )
    assert state._enabled


def test_gripper_prediction_timeout_does_not_pause_arm_control():
    """确认控制周期在预测超时时仅保持夹爪，机械臂继续发送原有目标。"""
    state = TrackerInputState()
    now = time.monotonic()
    state._latest_tracker_monotonic = now - 0.15
    state._pose_timeout_s = 0.25
    state._freeze_timeout_s = 0.10
    state._gripper_follower.update_feedback(0.4, now)
    state._gripper_follower.update_estimate(True, 0.8, now - 0.3)

    TrackerTeleopNode._control_tick(state)

    assert state._enabled
    assert state._gripper_command_publisher.messages[-1].data == pytest.approx(
        0.4
    )
    assert state.commands[-1] == pytest.approx(np.zeros(7))
    assert "夹爪预测超时" in state._status_publisher.messages[-1].data


def test_short_joint_feedback_gap_freezes_target_then_recovers():
    """验证短暂反馈断流后保留原有运动零点并继续遥操。"""
    state = TrackerInputState()
    now = time.monotonic()
    state._latest_tracker_monotonic = now - 0.15
    state._pose_timeout_s = 0.25
    state._freeze_timeout_s = 0.10
    state._latest_joint_monotonic = now - 0.30
    state._latest_tracker_pose = make_pose((0.2, 0.0, 0.0))

    TrackerTeleopNode._control_tick(state)

    assert state._enabled and state._feedback_frozen
    assert state.commands[-1] == pytest.approx(np.zeros(7))
    assert "七轴反馈短暂中断" in state._status_publisher.messages[-1].data

    state._latest_joint_monotonic = time.monotonic()
    TrackerTeleopNode._control_tick(state)

    assert state._enabled and not state._feedback_frozen
    assert state._reference_tracker_pose == pytest.approx(make_pose((1.0, 1.0, 1.0)))
    assert state._reference_eef_pose == pytest.approx(make_pose((2.0, 2.0, 2.0)))
    assert "七轴反馈已恢复" in state._status_publisher.messages[-1].data


def test_normal_joint_feedback_gap_keeps_following_tracker():
    """验证常见反馈间隔内仍计算 Tracker 目标并发送新关节指令。"""
    state = TrackerInputState()
    now = time.monotonic()
    state._latest_joint_monotonic = now - 0.15
    state._latest_tracker_monotonic = now
    state._pose_timeout_s = 0.25
    state._freeze_timeout_s = 0.10
    state._pose_smoothing_enabled = False
    state._joint_smoothing_enabled = False
    state._translation_scale = 1.0
    state._rotation_scale = 1.0
    state._reference_tracker_pose = make_pose()
    state._reference_eef_pose = make_pose()
    state._latest_tracker_pose = make_pose((0.1, 0.0, 0.0))

    TrackerTeleopNode._control_tick(state)

    assert state._enabled and not state._feedback_frozen
    assert state.targets[-1][0, 3] == pytest.approx(0.1)
    assert state.commands[-1] == pytest.approx([0.1] * 7)


def test_prolonged_joint_feedback_gap_pauses_control():
    """验证持续反馈失联仍停止遥操并清除运动零点。"""
    state = TrackerInputState()
    now = time.monotonic()
    state._latest_joint_monotonic = now - 0.60

    TrackerTeleopNode._control_tick(state)

    assert not state._enabled
    assert state._reference_tracker_pose is None
    assert "七轴反馈超时" in state._status_publisher.messages[-1].data


def test_joint_feedback_recovery_does_not_recompute_reference():
    """验证反馈恢复时无需正运动学计算，原有零点仍有效。"""
    state = TrackerInputState()
    now = time.monotonic()
    state._latest_tracker_monotonic = now - 0.15
    state._pose_timeout_s = 0.25
    state._freeze_timeout_s = 0.10
    state._feedback_frozen = True
    state._kinematics.error = RuntimeError("正运动学失败")

    TrackerTeleopNode._control_tick(state)

    assert state._enabled and not state._feedback_frozen
    assert state._reference_tracker_pose == pytest.approx(make_pose((1.0, 1.0, 1.0)))
    assert state.commands[-1] == pytest.approx(np.zeros(7))
    assert "七轴反馈已恢复" in state._status_publisher.messages[-1].data


def test_enable_requires_feedback_fresher_than_freeze_limit():
    """验证处于冻结区间的反馈不能用于建立新的遥操零点。"""
    now = time.monotonic()
    state = SimpleNamespace(
        _latest_heartbeat_monotonic=now,
        _heartbeat_timeout_s=0.50,
        _tracker_freshness_error=lambda _now: None,
        _latest_joint_positions=np.zeros(7),
        _latest_joint_monotonic=now - 0.30,
        _feedback_freeze_timeout_s=0.25,
    )

    assert "七轴反馈不够新鲜" in TrackerTeleopNode._freshness_error(state, now)


def test_invalid_tracker_status_pauses_and_preserves_mapping():
    """验证跟踪状态失效会暂停控制，但不清除成功标定方向。"""
    state = TrackerInputState()
    message = TrackerStatus()
    message.device_connected = True
    message.pose_valid = False

    TrackerTeleopNode._status_callback(state, message)

    assert not state._enabled
    assert state._mapping_basis == pytest.approx(np.eye(3))
    assert "标定方向已保留" in state._status_publisher.messages[-1].data


def test_tracker_pause_holds_latest_real_gripper_openness():
    """确认 Tracker 故障沿用机械臂暂停路径并保持真实夹爪开度。"""
    state = TrackerInputState()
    state._gripper_follower.update_feedback(0.4, time.monotonic())
    state._gripper_follower.update_estimate(True, 0.9, time.monotonic())
    state._update_gripper(True, time.monotonic())
    state._gripper_follower.update_feedback(0.5, time.monotonic())

    TrackerTeleopNode._disable(state, "Tracker 跟踪状态无效", True)

    assert not state._enabled
    assert [
        message.data for message in state._gripper_command_publisher.messages
    ] == pytest.approx([0.9, 0.5])


def test_heartbeat_timeout_pauses_and_preserves_mapping(monkeypatch):
    """验证输入超时会暂停控制，但不清除成功标定方向。"""
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    state = TrackerInputState()
    state._latest_heartbeat_monotonic = 99.0

    TrackerTeleopNode._control_tick(state)

    assert not state._enabled
    assert state._mapping_basis == pytest.approx(np.eye(3))
    assert "标定方向已保留" in state._status_publisher.messages[-1].data


@pytest.mark.parametrize("returns_to_original", [True, False])
def test_jump_pauses_and_always_preserves_mapping(
    returns_to_original, monkeypatch
):
    """验证坏帧和稳定重定位都会暂停控制并保留成功标定的方向。"""
    monkeypatch.setattr(time, "monotonic", lambda: 100.0)
    state = TrackerInputState()
    for stamp in range(1, 4):
        state._accept_tracker_pose(make_pose(), stamp)

    state._accept_tracker_pose(make_pose((1.0, 0.0, 0.0)), 4)

    assert not state._enabled
    assert not state._enabled_publisher.messages[-1].data
    assert state._reference_tracker_pose is None
    assert len(state.commands) == 1
    assert "位置差=1.000 m" in state._status_publisher.messages[-1].data

    recovered_pose = (
        make_pose() if returns_to_original else make_pose((1.0, 0.0, 0.0))
    )
    for stamp in range(5, 8):
        state._accept_tracker_pose(recovered_pose, stamp)
    TrackerTeleopNode._control_tick(state)

    assert not state._enabled
    assert len(state.commands) == 1
    assert state._mapping_basis == pytest.approx(np.eye(3))
    assert "标定方向已保留" in state._status_publisher.messages[-1].data
    assert "按空格" in state._status_publisher.messages[-1].data
    if returns_to_original:
        assert "原有轨迹" in state._status_publisher.messages[-1].data
    else:
        assert "新的稳定位置" in state._status_publisher.messages[-1].data


def test_joint_state_is_reordered_by_name():
    """验证乱序反馈按 RM75 驱动顺序输出。"""
    message = JointState()
    message.name = [f"joint{index}" for index in range(7, 0, -1)]
    message.position = [float(index) for index in range(7, 0, -1)]

    assert reorder_joint_state(message) == pytest.approx(np.arange(1.0, 8.0))


def test_incomplete_joint_state_is_rejected():
    """验证缺少任意 RM75 关节时拒绝反馈。"""
    message = JointState()
    message.name = ["joint1"]
    message.position = [0.0]

    with pytest.raises(ValueError, match="缺少"):
        reorder_joint_state(message)


def test_canfd_message_uses_radians_and_low_follow_contract():
    """验证 CANFD 七轴低跟随消息字段。"""
    positions = np.linspace(-0.3, 0.3, 7)
    message = build_joint_command(positions)

    assert message.joint == pytest.approx(positions)
    assert message.follow is False
    assert message.expand == pytest.approx(0.0)
    assert message.dof == 7


def test_home_message_uses_blocking_seven_axis_movej_contract():
    """验证回位使用弧度七轴阻塞 MoveJ 和配置的速度百分比。"""
    positions = np.linspace(-0.6, 0.6, 7)

    message = build_home_command(positions, speed_percent=20)

    assert message.joint == pytest.approx(positions)
    assert message.speed == 20
    assert message.block is True
    assert message.trajectory_connect == 0
    assert message.dof == 7


@pytest.mark.parametrize("speed", [0, 101, 20.5])
def test_home_message_rejects_invalid_speed(speed):
    """验证 MoveJ 回位速度必须为合法整数百分比。"""
    with pytest.raises(ValueError, match="速度"):
        build_home_command(np.zeros(7), speed)


def test_shutdown_request_marks_control_node_for_exit():
    """验证面板退出请求会让控制节点结束主循环。"""
    state = SimpleNamespace(_shutdown_requested=False)

    response = TrackerTeleopNode._shutdown_callback(
        state, Trigger.Request(), Trigger.Response()
    )

    assert response.success
    assert state._shutdown_requested


def make_joint_state(positions):
    """构造包含七轴有序反馈的 JointState。"""
    message = JointState()
    message.name = [f"joint{index}" for index in range(1, 8)]
    message.position = list(positions)
    return message


def test_first_complete_joint_feedback_is_kept_as_home_pose():
    """验证启动后首帧完整反馈被锁定且不会被后续反馈覆盖。"""
    state = HomeState()
    state._home_joint_positions = parse_home_joint_positions([])
    first = np.linspace(-0.3, 0.3, 7)
    second = np.linspace(0.4, 1.0, 7)

    TrackerTeleopNode._joint_state_callback(state, make_joint_state(first))
    TrackerTeleopNode._joint_state_callback(state, make_joint_state(second))

    assert state._home_joint_positions == pytest.approx(first)
    assert state._latest_joint_positions == pytest.approx(second)


def test_configured_home_survives_feedback_and_is_used_for_movej():
    """验证配置的回位姿态优先于启动反馈，按 h 时使用配置关节角。"""
    state = HomeState()
    configured = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    state._home_joint_positions = parse_home_joint_positions(configured)
    TrackerTeleopNode._joint_state_callback(state, make_joint_state(np.zeros(7)))
    TrackerTeleopNode._joint_state_callback(state, make_joint_state(np.ones(7)))

    response = call_return_home(state)
    send_pending_home(state)

    assert response.success
    assert state._home_joint_positions == pytest.approx(configured)
    assert state._home_publisher.messages[0].joint == pytest.approx(configured)
    assert state._latest_joint_positions == pytest.approx(np.ones(7))


def test_configured_home_still_requires_current_feedback():
    """验证指定目标后仍需完整的新鲜反馈，不能仅凭配置启动回位。"""
    state = HomeState()
    state._home_joint_positions = parse_home_joint_positions([0.1] * 7)
    state._latest_joint_positions = None

    response = call_return_home(state)

    assert not response.success and "反馈未就绪" in response.message
    assert state._home_publisher.messages == []


def test_default_yaml_uses_configured_home_target():
    """验证包内默认 YAML 为 h 提供指定的七轴弧度回位目标。"""
    config_path = Path(__file__).parents[1] / "config" / "tracker_teleoperated.yaml"
    parameters = yaml.safe_load(config_path.read_text())["tracker_teleop"][
        "ros__parameters"
    ]

    assert parse_home_joint_positions(
        parameters["home_joint_positions_rad"]
    ) == pytest.approx(
        [
            0.0,
            0.3490658503988659,
            0.0,
            1.2217304763960306,
            0.0,
            1.5707963267948966,
            1.5707963267948966,
        ]
    )
    assert parameters["workspace_minimum_angle_deg"] == pytest.approx(60.0)
    assert parameters["mapping_mode"] == "workspace"
    assert parameters["translation_scale"] == pytest.approx(1.0)
    assert parameters["gripper_estimate_topic"] == "/gripper/state"
    assert parameters["gripper_feedback_topic"] == "/motion_control/gripper_state"
    assert parameters["gripper_command_topic"] == "/motion_control/gripper_command"
    assert parameters["gripper_estimate_timeout_s"] == pytest.approx(0.25)
    assert parameters["gripper_feedback_timeout_s"] == pytest.approx(0.25)
    assert parameters["feedback_freeze_timeout_s"] == pytest.approx(0.25)
    assert parameters["feedback_timeout_s"] == pytest.approx(0.50)
    assert parameters["home_command_quiet_period_s"] == pytest.approx(0.20)


def call_return_home(state):
    """调用回位回调并返回 Trigger 响应。"""
    return TrackerTeleopNode._return_home_callback(
        state, Trigger.Request(), Trigger.Response()
    )


def send_pending_home(state):
    """越过透传静默期并执行一次控制周期。"""
    state._home_command_due_monotonic = time.monotonic() - 1.0
    TrackerTeleopNode._control_tick(state)


def test_return_home_pauses_then_publishes_one_movej_after_quiet_period():
    """验证回位先静默排空旧透传，再发送且只发送一次 MoveJ。"""
    state = HomeState()
    state._enabled = True
    state._workspace_calibration_samples.append(make_pose())
    original_mapping = state._mapping_basis.copy()

    response = call_return_home(state)

    assert response.success and "等待旧透传命令排空" in response.message
    assert not state._enabled
    assert state._homing
    assert state.publish_hold is False
    assert state._home_started_monotonic == 0.0
    assert state._pending_home_command is not None
    assert state._workspace_calibration_samples == []
    assert state.joint_targets == []
    assert state._home_publisher.messages == []

    TrackerTeleopNode._control_tick(state)
    assert state._home_publisher.messages == []

    send_pending_home(state)
    TrackerTeleopNode._control_tick(state)

    assert state._pending_home_command is None
    assert state._home_started_monotonic > 0.0
    assert state.joint_targets[0] == pytest.approx(state._home_joint_positions)
    assert len(state._home_publisher.messages) == 1
    command = state._home_publisher.messages[0]
    assert command.joint == pytest.approx(state._home_joint_positions)
    assert command.speed == 20 and command.block
    assert state._reference_tracker_pose is None
    assert state._reference_eef_pose is None
    assert state._mapping_basis == pytest.approx(original_mapping)


def test_stale_home_result_is_ignored_during_quiet_period():
    """验证静默排空阶段到达的旧 MoveJ 结果不会取消待发回位。"""
    state = HomeState()
    call_return_home(state)

    TrackerTeleopNode._home_result_callback(state, Bool(data=True))

    assert state._homing
    assert state._pending_home_command is not None
    assert state._home_publisher.messages == []


def test_completed_home_stays_paused_without_new_canfd_commands():
    """验证回位完成后控制周期保持暂停且不会恢复旧透传。"""
    state = HomeState()
    call_return_home(state)
    send_pending_home(state)

    TrackerTeleopNode._home_result_callback(state, Bool(data=True))
    TrackerTeleopNode._control_tick(state)

    assert not state._homing
    assert not state._enabled
    assert state.commands == []


def test_return_home_rejects_missing_or_stale_joint_feedback():
    """验证缺少初始姿态或实时反馈过期时拒绝回位。"""
    missing = HomeState()
    missing._home_joint_positions = None
    stale = HomeState()
    stale._latest_joint_monotonic = time.monotonic() - 1.0

    missing_response = call_return_home(missing)
    stale_response = call_return_home(stale)

    assert not missing_response.success and "尚未收到" in missing_response.message
    assert not stale_response.success and "已超时" in stale_response.message
    assert missing._home_publisher.messages == []
    assert stale._home_publisher.messages == []


def test_pause_stops_homing_and_enable_is_rejected_while_homing():
    """验证静默阶段禁止启用，暂停会取消待发 MoveJ 并停止运动。"""
    state = HomeState()
    call_return_home(state)
    enable = SetBool.Request()
    enable.data = True
    reject = TrackerTeleopNode._apply_enabled_callback(
        state, enable, SetBool.Response()
    )
    pause = SetBool.Request()
    pause.data = False
    stopped = TrackerTeleopNode._apply_enabled_callback(
        state, pause, SetBool.Response()
    )

    assert not reject.success and "正在回位" in reject.message
    assert stopped.success and "回位已停止" in stopped.message
    assert not state._homing
    assert state._pending_home_command is None
    assert state._home_publisher.messages == []
    assert len(state._move_stop_publisher.messages) == 1


def test_calibration_and_zero_reset_are_rejected_while_homing():
    """验证回位期间不能采集标定点或重置遥操运动零点。"""
    state = HomeState()
    state._homing = True

    calibration = TrackerTeleopNode._calibrate_workspace_callback(
        state, Trigger.Request(), Trigger.Response()
    )
    initialize = TrackerTeleopNode._initialize_callback(
        state, Trigger.Request(), Trigger.Response()
    )

    assert not calibration.success and "正在回位" in calibration.message
    assert not initialize.success and "正在回位" in initialize.message


def test_home_result_and_timeout_leave_homing_state():
    """验证驱动结果或超时都会解除回位状态，超时同时请求停止轨迹。"""
    completed = HomeState()
    completed._homing = True
    completed._home_started_monotonic = time.monotonic()
    TrackerTeleopNode._home_result_callback(completed, Bool(data=True))

    failed = HomeState()
    failed._homing = True
    failed._home_started_monotonic = time.monotonic()
    TrackerTeleopNode._home_result_callback(failed, Bool(data=False))

    timed_out = HomeState()
    timed_out._homing = True
    timed_out._home_started_monotonic = time.monotonic() - 31.0
    TrackerTeleopNode._control_tick(timed_out)

    assert not completed._homing
    assert not failed._homing
    assert not timed_out._homing
    assert len(timed_out._move_stop_publisher.messages) == 1


def test_home_timeout_starts_when_movej_is_actually_published(monkeypatch):
    """验证静默等待不占用 MoveJ 的执行超时时间。"""
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    state = HomeState()
    state._feedback_timeout_s = 100.0
    state._heartbeat_timeout_s = 100.0

    call_return_home(state)
    TrackerTeleopNode._control_tick(state)

    assert state._homing
    assert state._home_started_monotonic == 0.0
    assert state._home_publisher.messages == []

    now[0] = 100.21
    TrackerTeleopNode._control_tick(state)
    assert state._home_started_monotonic == pytest.approx(100.21)
    assert len(state._home_publisher.messages) == 1

    now[0] = 130.20
    TrackerTeleopNode._control_tick(state)
    assert state._homing

    now[0] = 130.22
    TrackerTeleopNode._control_tick(state)
    assert not state._homing
    assert len(state._move_stop_publisher.messages) == 1


def test_joint_feedback_loss_stops_homing():
    """验证静默阶段关节反馈中断会取消待发命令并请求停止。"""
    state = HomeState()
    call_return_home(state)
    state._latest_joint_monotonic = time.monotonic() - 1.0

    TrackerTeleopNode._control_tick(state)

    assert not state._homing
    assert state._pending_home_command is None
    assert state._home_publisher.messages == []
    assert len(state._move_stop_publisher.messages) == 1


def call_workspace_calibration(state):
    """调用工作空间标定回调并返回 Trigger 响应。"""
    response = Trigger.Response()
    return TrackerTeleopNode._calibrate_workspace_callback(
        state, Trigger.Request(), response
    )


def test_workspace_calibration_collects_three_samples_and_replaces_basis():
    """验证三次采样成功后原子替换工作空间映射并清除运动零点。"""
    state = WorkspaceCalibrationState()
    state._mapping_basis = Rotation.from_euler(
        "z", 30.0, degrees=True
    ).as_matrix()
    first = call_workspace_calibration(state)
    state._latest_tracker_pose = make_pose((0.0, 0.0, 0.10))
    second = call_workspace_calibration(state)
    state._latest_tracker_pose = make_pose((0.10, 0.0, 0.10))
    third = call_workspace_calibration(state)

    assert first.success and "起点" in first.message
    assert second.success and "上方点" in second.message
    assert "大致向上" in first.message
    assert "大致向前" in second.message
    assert third.success and "标定完成" in third.message
    assert state._mapping_basis == pytest.approx(np.eye(3))
    assert state.saved_mapping == pytest.approx(np.eye(3))
    assert state._workspace_calibration_samples == []
    assert state._reference_tracker_pose is None
    assert state._reference_eef_pose is None
    assert state._filtered_tracker_pose is None


def test_workspace_calibration_rejects_near_collinear_samples():
    """验证节点按配置夹角拒绝近乎共线的样本且保留旧映射。"""
    state = WorkspaceCalibrationState()
    state._mapping_basis = np.eye(3)
    call_workspace_calibration(state)
    state._latest_tracker_pose = make_pose((0.0, 0.0, 0.10))
    call_workspace_calibration(state)
    state._latest_tracker_pose = make_pose((0.01, 0.0, 0.20))

    response = call_workspace_calibration(state)

    assert not response.success and "夹角" in response.message
    assert state._mapping_basis == pytest.approx(np.eye(3))
    assert state.saved_mapping is None


def test_approximate_calibration_is_accepted():
    """验证粗略方向可完成工作空间标定。"""
    calibration_state = WorkspaceCalibrationState()
    calibration_state._workspace_minimum_angle_deg = 45.0
    call_workspace_calibration(calibration_state)
    calibration_state._latest_tracker_pose = make_pose(
        (0.0, 0.0, 0.12), (45.0, 10.0, -20.0)
    )
    call_workspace_calibration(calibration_state)
    calibration_state._latest_tracker_pose = make_pose(
        (0.15, 0.0, 0.20), (-35.0, 50.0, 80.0)
    )

    response = call_workspace_calibration(calibration_state)
    assert response.success and "标定完成" in response.message
    assert calibration_state._mapping_basis is not None


def test_publish_command_sends_debug_target_and_driver_command():
    """验证关节目标同时发布到调试和 RM75 CANFD 话题。"""
    publishing_state = CommandPublishingState()
    target_positions = np.linspace(-0.2, 0.2, 7)

    TrackerTeleopNode._publish_command(publishing_state, target_positions)

    assert publishing_state.joint_targets[0] == pytest.approx(target_positions)
    assert len(publishing_state._command_publisher.messages) == 1
    assert publishing_state._command_publisher.messages[0].joint == pytest.approx(
        target_positions
    )


def test_failed_workspace_calibration_keeps_last_successful_basis():
    """验证几何校验失败只清除本轮样本并保留上次成功映射。"""
    state = WorkspaceCalibrationState()
    old_basis = Rotation.from_euler("z", 20.0, degrees=True).as_matrix()
    state._mapping_basis = old_basis.copy()
    call_workspace_calibration(state)
    state._latest_tracker_pose = make_pose((0.0, 0.0, 0.10))
    call_workspace_calibration(state)
    state._latest_tracker_pose = make_pose((0.0, 0.0, 0.20))

    response = call_workspace_calibration(state)

    assert not response.success
    assert "标定失败" in response.message
    assert state._workspace_calibration_samples == []
    assert state._mapping_basis == pytest.approx(old_basis)


def test_workspace_calibration_save_failure_keeps_old_basis():
    """验证候选方向写入失败时不替换内存中的旧标定。"""
    state = WorkspaceCalibrationState()
    old_basis = Rotation.from_euler("z", 20.0, degrees=True).as_matrix()
    state._mapping_basis = old_basis.copy()
    state.save_error = "磁盘只读"
    call_workspace_calibration(state)
    state._latest_tracker_pose = make_pose((0.0, 0.0, 0.10))
    call_workspace_calibration(state)
    state._latest_tracker_pose = make_pose((0.10, 0.0, 0.10))

    response = call_workspace_calibration(state)

    assert not response.success and "旧标定方向保持不变" in response.message
    assert state._mapping_basis == pytest.approx(old_basis)
    assert state.saved_mapping is None


def test_workspace_calibration_input_failure_clears_only_pending_samples():
    """验证输入失效会取消本轮标定并保留既有工作空间映射。"""
    state = WorkspaceCalibrationState()
    state._mapping_basis = np.eye(3)
    call_workspace_calibration(state)
    state.health_error = "Tracker 里程计已超时"

    response = call_workspace_calibration(state)

    assert not response.success
    assert "已取消" in response.message
    assert state._workspace_calibration_samples == []
    assert state._mapping_basis == pytest.approx(np.eye(3))


def test_workspace_mode_rejects_enable_during_or_before_calibration():
    """验证工作空间标定未完成或正在采样时禁止启用。"""
    state = WorkspaceCalibrationState()
    request = SetBool.Request()
    request.data = True
    missing = TrackerTeleopNode._apply_enabled_callback(
        state, request, SetBool.Response()
    )
    state._workspace_calibration_samples.append(make_pose())
    in_progress = TrackerTeleopNode._apply_enabled_callback(
        state, request, SetBool.Response()
    )

    assert not missing.success and "尚未标定" in missing.message
    assert not in_progress.success and "尚未完成" in in_progress.message


def test_pause_cancels_pending_workspace_calibration():
    """验证 s 会取消未完成采样，同时保留此前成功标定方向。"""
    state = WorkspaceCalibrationState()
    old_basis = Rotation.from_euler("z", 25.0, degrees=True).as_matrix()
    state._mapping_basis = old_basis.copy()
    state._workspace_calibration_samples.append(make_pose())
    request = SetBool.Request()
    request.data = False

    response = TrackerTeleopNode._apply_enabled_callback(
        state, request, SetBool.Response()
    )

    assert response.success and "标定已取消" in response.message
    assert state._workspace_calibration_samples == []
    assert state._mapping_basis == pytest.approx(old_basis)


def test_reference_eef_rejects_workspace_calibration():
    """验证参考末端模式不采集工作空间标定点。"""
    state = WorkspaceCalibrationState()
    state._mapping_mode = "reference_eef"
    state._mapping_basis = np.eye(3)

    response = call_workspace_calibration(state)

    assert not response.success
    assert "reference_eef" in response.message
    assert state._workspace_calibration_samples == []
    assert state._mapping_basis == pytest.approx(np.eye(3))


def test_reference_capture_initializes_then_preserves_mapping_basis():
    """验证首次初始化固定轴向，重新启用只更新运动零点。"""
    state = ReferenceState()
    assert TrackerTeleopNode._capture_reference(state, True) is None
    initialized_basis = state._mapping_basis.copy()

    state._latest_tracker_pose = make_pose(
        (0.2, -0.1, 0.6), (70.0, 5.0, 10.0)
    )
    state._latest_joint_positions = np.full(7, 0.25)
    state._kinematics.pose = make_pose(
        (0.5, 0.2, 0.4), (-30.0, 15.0, 80.0)
    )
    assert TrackerTeleopNode._capture_reference(state, False) is None

    assert state._mapping_basis == pytest.approx(initialized_basis)
    assert state._reference_tracker_pose == pytest.approx(
        state._latest_tracker_pose
    )
    assert state._reference_eef_pose == pytest.approx(state._kinematics.pose)
    assert state._command_positions == pytest.approx(np.full(7, 0.25))
    assert state._command_velocity == pytest.approx(np.zeros(7))


def test_workspace_zero_reset_preserves_calibrated_axes():
    """验证工作空间模式重置运动零点只更新参考位姿，不改变标定 XYZ。"""
    state = ReferenceState()
    state._mapping_mode = "workspace"
    calibrated_basis = Rotation.from_euler(
        "xyz", [10.0, 20.0, 30.0], degrees=True
    ).as_matrix()
    state._mapping_basis = calibrated_basis.copy()

    error = TrackerTeleopNode._capture_reference(state, False)

    assert error is None
    assert state._mapping_basis == pytest.approx(calibrated_basis)
    assert state._reference_tracker_pose == pytest.approx(
        state._latest_tracker_pose
    )


def test_failed_initialization_preserves_existing_reference():
    """验证参考采集失败时不会部分覆盖已有控制状态。"""
    state = ReferenceState()
    assert TrackerTeleopNode._capture_reference(state, True) is None
    old_basis = state._mapping_basis.copy()
    old_tracker = state._reference_tracker_pose.copy()
    old_eef = state._reference_eef_pose.copy()
    state._kinematics.error = RuntimeError("正运动学失败")
    state._latest_tracker_pose = make_pose((1.0, 2.0, 3.0))

    error = TrackerTeleopNode._capture_reference(state, True)

    assert "正运动学失败" in error
    assert state._mapping_basis == pytest.approx(old_basis)
    assert state._reference_tracker_pose == pytest.approx(old_tracker)
    assert state._reference_eef_pose == pytest.approx(old_eef)


def test_home_requires_panel_heartbeat():
    """回位前必须存在面板心跳，失联时不能发出 MoveJ。"""
    state = HomeState()
    state._latest_heartbeat_monotonic = 0.0
    response = call_return_home(state)
    assert not response.success
    assert state._home_publisher.messages == []


def test_panel_loss_stops_active_home():
    """静默阶段面板失联立即取消待发回位并发布停止命令。"""
    state = HomeState()
    call_return_home(state)
    state._latest_heartbeat_monotonic = 0.0
    TrackerTeleopNode._control_tick(state)
    assert not state._homing
    assert state._pending_home_command is None
    assert state._home_publisher.messages == []
    assert len(state._move_stop_publisher.messages) == 1
    assert "面板心跳" in state._status_publisher.messages[-1].data
