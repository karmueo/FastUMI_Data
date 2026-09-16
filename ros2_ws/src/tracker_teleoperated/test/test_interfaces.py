"""验证遥操 ROS 接口、跳变恢复状态和键盘提示契约。"""

from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from fastumi_interfaces.msg import TrackerStatus
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger

from tracker_teleoperated.keyboard import (
    TrackerTeleopKeyboard,
    is_calibrate_key,
    is_home_key,
    keyboard_instructions,
    requested_state_for_key,
)
from tracker_teleoperated.core import (
    PoseStreamValidator,
    parse_home_joint_positions,
)
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

    def _disable(self, _reason, publish_hold):
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
        self._homing = False
        self._home_started_monotonic = 0.0
        self._home_timeout_s = 30.0
        self._home_speed_percent = 20
        self._feedback_timeout_s = 0.25
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

    def get_logger(self):
        """返回空实现日志器。"""
        return self._logger

    def _disable(self, _reason, publish_hold):
        """模拟暂停遥操。"""
        self._enabled = False
        self.publish_hold = publish_hold

    def _publish_joint_target(self, positions):
        """记录调试关节目标。"""
        self.joint_targets.append(np.asarray(positions).copy())


class TrackerInputState(WorkspaceCalibrationState):
    """使用真实输入校验和暂停逻辑验证跳变后的控制与标定状态。"""

    _accept_tracker_pose = TrackerTeleopNode._accept_tracker_pose
    _disable = TrackerTeleopNode._disable
    _clear_control_reference = TrackerTeleopNode._clear_control_reference
    _publish_enabled = TrackerTeleopNode._publish_enabled
    _mapping_recovery_instruction = (
        TrackerTeleopNode._mapping_recovery_instruction
    )

    def __init__(self):
        """创建已标定、已启用且关节反馈新鲜的遥操状态。"""
        super().__init__()
        self._mapping_basis = np.eye(3)
        self._enabled = True
        self._pose_validator = PoseStreamValidator(recovery_samples=3)
        self._latest_joint_positions = np.zeros(7)
        self._latest_joint_monotonic = time.monotonic()
        self._last_tick_monotonic = time.monotonic()
        self._feedback_timeout_s = 0.25
        self._heartbeat_timeout_s = 0.50
        self._latest_heartbeat_monotonic = time.monotonic()
        self._latest_status_stamp_ns = None
        self._pending_tracker_sample = None
        self._command_positions = np.zeros(7)
        self._command_velocity = np.zeros(7)
        self._enabled_publisher = FakePublisher()
        # 记录保持目标，验证恢复输入不会自动继续驱动机械臂。
        self.commands = []

    def _publish_command(self, positions):
        """记录指令，避免向实机发送数据。"""
        self.commands.append(np.asarray(positions).copy())


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


@pytest.mark.parametrize(
    ("key", "enabled", "expected"),
    [
        (" ", False, True),
        (" ", True, False),
        ("s", True, False),
        ("i", True, None),
        ("I", True, None),
        ("x", True, None),
    ],
)
def test_keyboard_key_mapping(key, enabled, expected):
    """验证键盘切换和显式暂停按键。"""
    assert requested_state_for_key(key, enabled) is expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [("c", True), ("C", True), ("i", False), (" ", False), (None, False)],
)
def test_keyboard_calibrate_key_mapping(key, expected):
    """验证大小写 c 都会触发工作空间标定采样。"""
    assert is_calibrate_key(key) is expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [("h", True), ("H", True), ("i", False), (" ", False), (None, False)],
)
def test_keyboard_home_key_mapping(key, expected):
    """验证大小写 h 都会触发回到启动初始位姿。"""
    assert is_home_key(key) is expected


def test_keyboard_instructions_cover_both_mapping_modes():
    """验证键盘提示工作空间标定和参考末端模式的启用步骤。"""
    instructions = keyboard_instructions()

    assert "[c]" in instructions
    assert "[i]" not in instructions
    assert "mapping_mode=workspace" in instructions
    assert "mapping_mode=reference_eef" in instructions
    assert "起点" in instructions
    assert "大致向上" in instructions
    assert "大致向前" in instructions
    assert "[空格]" in instructions


def test_keyboard_displays_async_pause_and_mapping_preservation(capsys):
    """验证未操作键盘时也会显示暂停原因和标定方向保留提示。"""
    keyboard = SimpleNamespace(enabled=True, _last_status="")
    paused = String(data="遥操已暂停: Tracker 位姿发生跳变")
    preserved = String(data="标定方向已保留，待输入稳定后按空格重新启用")

    TrackerTeleopKeyboard._enabled_callback(keyboard, Bool(data=False))
    TrackerTeleopKeyboard._status_callback(keyboard, paused)
    TrackerTeleopKeyboard._status_callback(keyboard, paused)
    TrackerTeleopKeyboard._status_callback(keyboard, preserved)
    output = capsys.readouterr().out

    assert not keyboard.enabled
    assert output.count("Tracker 位姿发生跳变") == 1
    assert "标定方向已保留" in output
    assert requested_state_for_key(" ", keyboard.enabled) is True


def test_shutdown_request_marks_control_node_for_exit():
    """验证键盘退出请求会让控制节点结束主循环。"""
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


def call_return_home(state):
    """调用回位回调并返回 Trigger 响应。"""
    return TrackerTeleopNode._return_home_callback(
        state, Trigger.Request(), Trigger.Response()
    )


def test_return_home_pauses_teleop_and_publishes_movej():
    """验证启用状态下回位会先暂停并发送一次 MoveJ。"""
    state = HomeState()
    state._enabled = True
    state._workspace_calibration_samples.append(make_pose())
    original_mapping = state._mapping_basis.copy()

    response = call_return_home(state)

    assert response.success and "开始回到" in response.message
    assert not state._enabled
    assert state._homing
    assert state.publish_hold is True
    assert state._workspace_calibration_samples == []
    assert state.joint_targets[0] == pytest.approx(state._home_joint_positions)
    assert len(state._home_publisher.messages) == 1
    command = state._home_publisher.messages[0]
    assert command.joint == pytest.approx(state._home_joint_positions)
    assert command.speed == 20 and command.block
    assert state._reference_tracker_pose is None
    assert state._reference_eef_pose is None
    assert state._mapping_basis == pytest.approx(original_mapping)


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
    """验证回位期间禁止启用，暂停会发布规划轨迹停止命令。"""
    state = HomeState()
    state._homing = True
    enable = SetBool.Request()
    enable.data = True
    reject = TrackerTeleopNode._set_enabled_callback(
        state, enable, SetBool.Response()
    )
    pause = SetBool.Request()
    pause.data = False
    stopped = TrackerTeleopNode._set_enabled_callback(
        state, pause, SetBool.Response()
    )

    assert not reject.success and "正在回位" in reject.message
    assert stopped.success and "回位已停止" in stopped.message
    assert not state._homing
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
    TrackerTeleopNode._home_result_callback(completed, Bool(data=True))

    failed = HomeState()
    failed._homing = True
    TrackerTeleopNode._home_result_callback(failed, Bool(data=False))

    timed_out = HomeState()
    timed_out._homing = True
    timed_out._home_started_monotonic = time.monotonic() - 31.0
    TrackerTeleopNode._control_tick(timed_out)

    assert not completed._homing
    assert not failed._homing
    assert not timed_out._homing
    assert len(timed_out._move_stop_publisher.messages) == 1


def test_joint_feedback_loss_stops_homing():
    """验证回位期间关节反馈中断会请求停止规划轨迹。"""
    state = HomeState()
    state._homing = True
    state._home_started_monotonic = time.monotonic()
    state._latest_joint_monotonic = time.monotonic() - 1.0

    TrackerTeleopNode._control_tick(state)

    assert not state._homing
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
    missing = TrackerTeleopNode._set_enabled_callback(
        state, request, SetBool.Response()
    )
    state._workspace_calibration_samples.append(make_pose())
    in_progress = TrackerTeleopNode._set_enabled_callback(
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

    response = TrackerTeleopNode._set_enabled_callback(
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
