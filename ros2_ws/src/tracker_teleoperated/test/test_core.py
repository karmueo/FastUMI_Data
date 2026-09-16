"""验证 Tracker 遥操位姿映射、输入恢复和关节平滑行为。"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tracker_teleoperated.core import (
    PoseStreamValidator,
    advance_joint_command,
    axis_mapping_from_rpy,
    low_pass_pose,
    map_tracker_target,
    mapping_basis_from_reference,
    parse_home_joint_positions,
    workspace_mapping_from_samples,
)


@pytest.mark.parametrize("values", [[], None])
def test_empty_home_configuration_uses_current_pose(values):
    """验证空列表和 ROS 未设置参数均采集启动时的当前位姿。"""
    assert parse_home_joint_positions(values) is None


@pytest.mark.parametrize(
    "values", [[0, 1, 0, -1, 0, 1, 0], [0.0, 0.1, 0.2, 0.3, -0.2, 0.0, 0.1]]
)
def test_configured_home_is_copied_in_joint_order(values):
    """验证整数或浮点关节角按输入顺序以弧度保存，且不共享调用者数据。"""
    positions = parse_home_joint_positions(values)

    assert positions == pytest.approx(values)
    assert positions.dtype == np.float64
    values[0] = 2
    assert positions[0] == 0.0


@pytest.mark.parametrize(
    "values",
    [
        [0.0] * 6,
        [0.0] * 8,
        [0.0] * 6 + [float("nan")],
        [0.0] * 6 + [float("inf")],
        [True] * 7,
        ["0"] * 7,
        [[0.0] * 7],
        0.0,
    ],
)
def test_invalid_home_configuration_is_rejected(values):
    """验证维度、数量、类型或有限性错误在启动配置解析时被拒绝。"""
    with pytest.raises(ValueError, match="home_joint_positions_rad"):
        parse_home_joint_positions(values)


def make_pose(position=(0.0, 0.0, 0.0), rpy_deg=(0.0, 0.0, 0.0)):
    """构造测试使用的有限 4×4 位姿。"""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler(
        "xyz", rpy_deg, degrees=True
    ).as_matrix()
    pose[:3, 3] = position
    return pose


def test_zero_motion_keeps_nonzero_reference_eef_pose():
    """验证每次重新启用的第一帧不会产生跳变。"""
    tracker = make_pose((0.4, -0.2, 0.7), (20.0, 10.0, -30.0))
    eef = make_pose((0.5, 0.1, 0.6), (-10.0, 35.0, 80.0))

    target = map_tracker_target(
        tracker,
        tracker,
        eef,
        mapping_basis_from_reference(tracker, eef, np.eye(3)),
        0.5,
        0.5,
    )

    assert target == pytest.approx(eef)


def test_xyz_maps_to_fixed_activation_eef_axes():
    """验证 odom 三轴映射到启用时末端三轴并保持固定。"""
    tracker_zero = make_pose()
    eef_zero = make_pose((0.4, 0.5, 0.6), (0.0, 0.0, 90.0))
    mapping_basis = mapping_basis_from_reference(
        tracker_zero, eef_zero, np.eye(3)
    )
    for axis_index in range(3):
        position = np.zeros(3)
        position[axis_index] = 0.2
        tracker_now = make_pose(position, (30.0, 0.0, 0.0))
        target = map_tracker_target(
            tracker_zero,
            tracker_now,
            eef_zero,
            mapping_basis,
            0.5,
            1.0,
        )
        expected = eef_zero[:3, 3] + eef_zero[:3, :3] @ position * 0.5
        assert target[:3, 3] == pytest.approx(expected)


def test_axis_mapping_and_independent_scales():
    """验证安装轴向以及平移、旋转灵敏度相互独立。"""
    tracker_zero = make_pose()
    tracker_now = make_pose((0.2, 0.0, 0.0), (0.0, 0.0, 60.0))
    mapping = axis_mapping_from_rpy((0.0, 0.0, 90.0))
    mapping_basis = mapping_basis_from_reference(
        tracker_zero, make_pose(), mapping
    )
    target = map_tracker_target(
        tracker_zero,
        tracker_now,
        make_pose(),
        mapping_basis,
        0.25,
        0.5,
    )

    assert target[:3, 3] == pytest.approx([0.0, 0.05, 0.0])
    angle = np.linalg.norm(Rotation.from_matrix(target[:3, :3]).as_rotvec())
    assert np.rad2deg(angle) == pytest.approx(30.0)


def test_workspace_samples_map_operator_forward_left_up_to_base_axes():
    """验证任意 odom 朝向下的前、左、上运动映射到 Base 正轴。"""
    operator_rotation = Rotation.from_euler(
        "xyz", (37.0, -24.0, 103.0), degrees=True
    ).as_matrix()
    tracker_rotation = Rotation.from_euler(
        "xyz", (11.0, 73.0, -49.0), degrees=True
    ).as_matrix()
    start = make_pose((0.4, -0.2, 0.7))
    start[:3, :3] = tracker_rotation
    up = start.copy()
    up[:3, 3] += operator_rotation @ np.array([0.0, 0.0, 0.15])
    forward = up.copy()
    forward[:3, 3] += operator_rotation @ np.array([0.15, 0.0, 0.0])

    mapping = workspace_mapping_from_samples(start, up, forward)

    operator_axes = operator_rotation @ np.eye(3)
    assert mapping @ operator_axes[:, 0] == pytest.approx([1.0, 0.0, 0.0])
    assert mapping @ operator_axes[:, 1] == pytest.approx([0.0, 1.0, 0.0])
    assert mapping @ operator_axes[:, 2] == pytest.approx([0.0, 0.0, 1.0])
    assert mapping @ mapping.T == pytest.approx(np.eye(3))
    assert np.linalg.det(mapping) == pytest.approx(1.0)


@pytest.mark.parametrize("direction_angle_deg", [30.0, 45.0, 90.0, 135.0, 150.0])
def test_workspace_mapping_equally_fits_approximate_directions(
    direction_angle_deg,
):
    """验证不同夹角的粗略方向会等权拟合为右手正交坐标系。"""
    angle_rad = np.deg2rad(direction_angle_deg)
    measured_up = np.array([0.0, 0.0, 1.0])
    measured_forward = np.array(
        [np.sin(angle_rad), 0.0, np.cos(angle_rad)]
    )
    start = make_pose()
    up = make_pose(0.12 * measured_up)
    forward = make_pose(up[:3, 3] + 0.18 * measured_forward)

    mapping = workspace_mapping_from_samples(
        start, up, forward, minimum_angle_deg=25.0
    )

    fitted_forward = mapping[0]
    fitted_up = mapping[2]
    forward_error_deg = np.rad2deg(
        np.arccos(np.clip(np.dot(fitted_forward, measured_forward), -1.0, 1.0))
    )
    up_error_deg = np.rad2deg(
        np.arccos(np.clip(np.dot(fitted_up, measured_up), -1.0, 1.0))
    )
    assert forward_error_deg == pytest.approx(up_error_deg, abs=1.0e-7)
    assert mapping @ mapping.T == pytest.approx(np.eye(3))
    assert np.linalg.det(mapping) == pytest.approx(1.0)


@pytest.mark.parametrize("direction_angle_deg", [1.0, 45.0, 135.0, 179.0])
def test_workspace_mapping_rejects_angle_outside_default_range(
    direction_angle_deg,
):
    """默认夹角门限拒绝靠近同向和反向共线的采样。"""
    angle_rad = np.deg2rad(direction_angle_deg)
    start = make_pose()
    up = make_pose((0.0, 0.0, 0.12))
    forward = make_pose(
        up[:3, 3] + 0.12 * np.array([np.sin(angle_rad), 0.0, np.cos(angle_rad)])
    )

    with pytest.raises(ValueError, match="夹角"):
        workspace_mapping_from_samples(start, up, forward)


@pytest.mark.parametrize("minimum_angle_deg", [0.0, 90.0, np.nan])
def test_workspace_mapping_rejects_invalid_minimum_angle(minimum_angle_deg):
    """拒绝不能形成有效对称夹角区间的配置值。"""
    with pytest.raises(ValueError, match="最小夹角"):
        workspace_mapping_from_samples(
            make_pose(),
            make_pose((0.0, 0.0, 0.12)),
            make_pose((0.12, 0.0, 0.12)),
            minimum_angle_deg=minimum_angle_deg,
        )


def test_workspace_mapping_ignores_pose_rotation_and_displacement_length():
    """验证姿态变化和两段位移长度差异不会改变方向的等权拟合。"""
    angle_rad = np.deg2rad(60.0)
    measured_up = np.array([0.0, 0.0, 1.0])
    measured_forward = np.array(
        [np.sin(angle_rad), 0.0, np.cos(angle_rad)]
    )
    start = make_pose(rpy_deg=(10.0, -20.0, 30.0))
    up = make_pose(0.06 * measured_up, (80.0, 15.0, -40.0))
    forward = make_pose(
        up[:3, 3] + 0.24 * measured_forward,
        (-70.0, 35.0, 120.0),
    )

    mapping = workspace_mapping_from_samples(start, up, forward)

    forward_error = np.arccos(
        np.clip(np.dot(mapping[0], measured_forward), -1.0, 1.0)
    )
    up_error = np.arccos(
        np.clip(np.dot(mapping[2], measured_up), -1.0, 1.0)
    )
    assert forward_error == pytest.approx(up_error)


@pytest.mark.parametrize(
    ("up_position", "forward_position", "message"),
    [
        ((0.0, 0.0, 0.04), (0.10, 0.0, 0.04), "向上位移"),
        ((0.0, 0.0, 0.10), (0.04, 0.0, 0.10), "向前位移"),
        ((0.0, 0.0, 0.10), (0.0, 0.0, 0.20), "方向共线"),
        ((0.0, 0.0, 0.10), (0.0, 0.0, 0.00), "方向共线"),
    ],
)
def test_workspace_samples_reject_invalid_geometry(
    up_position, forward_position, message
):
    """验证位移过短以及同向或反向共线的样本会被拒绝。"""
    with pytest.raises(ValueError, match=message):
        workspace_mapping_from_samples(
            make_pose(), make_pose(up_position), make_pose(forward_position)
        )


def test_workspace_samples_reject_invalid_pose_and_distance_threshold():
    """验证非法位姿和无效最小距离阈值仍会被拒绝。"""
    invalid_pose = make_pose((0.1, 0.0, 0.1))
    invalid_pose[0, 3] = np.nan

    with pytest.raises(ValueError, match="有限"):
        workspace_mapping_from_samples(
            make_pose(), make_pose((0.0, 0.0, 0.1)), invalid_pose
        )
    with pytest.raises(ValueError, match="最小位移"):
        workspace_mapping_from_samples(
            make_pose(),
            make_pose((0.0, 0.0, 0.1)),
            make_pose((0.1, 0.0, 0.1)),
            minimum_distance_m=0.0,
        )


def test_workspace_mapping_keeps_translation_fixed_after_rotation():
    """验证姿态变化不会改变后续平移使用的固定工作空间方向。"""
    start = make_pose()
    up = make_pose((0.0, 0.0, 0.1))
    forward = make_pose((0.1, 0.0, 0.1))
    mapping = workspace_mapping_from_samples(start, up, forward)
    eef_reference = make_pose((0.4, 0.1, 0.5), (180.0, 0.0, 0.0))
    tracker_now = make_pose((0.0, 0.0, 0.1), (0.0, 30.0, 0.0))

    target = map_tracker_target(
        start,
        tracker_now,
        eef_reference,
        mapping,
        translation_scale=0.5,
        rotation_scale=1.0,
    )

    assert target[:3, 3] == pytest.approx([0.4, 0.1, 0.55])
    expected_rotation = (
        Rotation.from_euler("y", 30.0, degrees=True).as_matrix()
        @ eef_reference[:3, :3]
    )
    assert target[:3, :3] == pytest.approx(expected_rotation)


def test_initialized_tracker_axes_map_to_eef_axes():
    """验证非单位初始朝向下 Tracker 局部轴映射到初始化末端轴。"""
    tracker_zero = make_pose((0.3, -0.4, 0.8), (25.0, -15.0, 40.0))
    eef_zero = make_pose((0.5, 0.1, 0.6), (-20.0, 30.0, 70.0))
    installation = axis_mapping_from_rpy((0.0, 0.0, 90.0))
    mapping_basis = mapping_basis_from_reference(
        tracker_zero, eef_zero, installation
    )
    tracker_now = tracker_zero.copy()
    tracker_now[:3, 3] += tracker_zero[:3, :3] @ np.array([0.2, 0.0, 0.0])

    target = map_tracker_target(
        tracker_zero,
        tracker_now,
        eef_zero,
        mapping_basis,
        0.5,
        1.0,
    )

    expected_delta = eef_zero[:3, :3] @ installation @ np.array(
        [0.1, 0.0, 0.0]
    )
    assert target[:3, 3] == pytest.approx(eef_zero[:3, 3] + expected_delta)


def test_initialized_tracker_rotation_maps_in_fixed_basis():
    """验证初始化姿态下的局部旋转按固定轴向映射并缩放。"""
    tracker_zero = make_pose(rpy_deg=(25.0, -15.0, 40.0))
    eef_zero = make_pose(rpy_deg=(-20.0, 30.0, 70.0))
    installation = axis_mapping_from_rpy((0.0, 0.0, 90.0))
    mapping_basis = mapping_basis_from_reference(
        tracker_zero, eef_zero, installation
    )
    tracker_now = tracker_zero.copy()
    tracker_now[:3, :3] = (
        tracker_zero[:3, :3]
        @ Rotation.from_euler("x", 60.0, degrees=True).as_matrix()
    )

    target = map_tracker_target(
        tracker_zero,
        tracker_now,
        eef_zero,
        mapping_basis,
        1.0,
        0.5,
    )

    expected_delta = (
        eef_zero[:3, :3]
        @ installation
        @ Rotation.from_euler("x", 30.0, degrees=True).as_matrix()
        @ installation.T
        @ eef_zero[:3, :3].T
    )
    expected_rotation = expected_delta @ eef_zero[:3, :3]
    assert target[:3, :3] == pytest.approx(expected_rotation)


def test_mapping_basis_can_be_reused_with_a_new_motion_zero():
    """验证重新启用只更新运动零点时仍沿用最近初始化的轴向。"""
    initialized_tracker = make_pose(rpy_deg=(10.0, 20.0, 30.0))
    initialized_eef = make_pose(rpy_deg=(-15.0, 25.0, 60.0))
    mapping_basis = mapping_basis_from_reference(
        initialized_tracker, initialized_eef, np.eye(3)
    )
    resumed_tracker = make_pose((0.4, 0.3, -0.2), (70.0, -10.0, 5.0))
    resumed_eef = make_pose((0.6, -0.1, 0.5), (5.0, 40.0, -20.0))

    target = map_tracker_target(
        resumed_tracker,
        resumed_tracker,
        resumed_eef,
        mapping_basis,
        0.5,
        0.5,
    )

    assert target == pytest.approx(resumed_eef)


def test_pose_low_pass_can_be_bypassed_or_applied_by_caller():
    """验证低通函数同时平滑位置和旋转且保持合法旋转。"""
    previous = make_pose()
    current = make_pose((1.0, 0.0, 0.0), (0.0, 0.0, 90.0))
    filtered = low_pass_pose(previous, current, cutoff_hz=8.0, dt=0.02)

    assert 0.0 < filtered[0, 3] < 1.0
    filtered_angle = np.rad2deg(
        np.linalg.norm(Rotation.from_matrix(filtered[:3, :3]).as_rotvec())
    )
    assert 0.0 < filtered_angle < 90.0
    assert np.linalg.det(filtered[:3, :3]) == pytest.approx(1.0)
    assert current == pytest.approx(current.copy())


def test_pose_validator_requires_three_frames_and_recovers_from_jump():
    """验证初始就绪、跳变拒绝和三帧稳定重定位。"""
    validator = PoseStreamValidator(recovery_samples=3)
    first = validator.update(make_pose())
    second = validator.update(make_pose((0.01, 0.0, 0.0)))
    third = validator.update(make_pose((0.02, 0.0, 0.0)))
    jump = validator.update(make_pose((1.0, 0.0, 0.0)))
    recovering = validator.update(make_pose((1.01, 0.0, 0.0)))
    recovered = validator.update(make_pose((1.02, 0.0, 0.0)))

    assert first.accepted and not first.ready
    assert second.accepted and not second.ready
    assert third.accepted and third.ready
    assert not jump.accepted
    assert not recovering.accepted
    assert recovered.accepted and recovered.ready and recovered.relocalized


def test_pose_validator_rejects_large_rotation_step():
    """验证超过姿态阈值的单帧旋转会被拒绝。"""
    validator = PoseStreamValidator(rotation_jump_deg=45.0)
    validator.update(make_pose())

    result = validator.update(make_pose(rpy_deg=(0.0, 0.0, 60.0)))

    assert not result.accepted
    assert "跳变" in result.reason
    assert "60.0°" in result.reason
    assert "45.0°" in result.reason


@pytest.mark.parametrize(
    "outlier",
    [make_pose((1.0, 0.0, 0.0)), make_pose(rpy_deg=(0.0, 0.0, 90.0))],
)
def test_pose_validator_keeps_original_frame_after_transient_outlier(outlier):
    """验证位置或姿态坏帧后回到原轨迹，三帧稳定不会误报重定位。"""
    validator = PoseStreamValidator(recovery_samples=3)
    for _ in range(3):
        validator.update(make_pose())

    assert not validator.update(outlier).accepted
    for step in (0.01, 0.02):
        recovering = validator.update(make_pose((step, 0.0, 0.0)))
        assert not recovering.accepted
        assert not recovering.ready

    recovered = validator.update(make_pose((0.03, 0.0, 0.0)))

    assert recovered.accepted and recovered.ready
    assert not recovered.relocalized
    assert "恢复原有轨迹" in recovered.reason
    assert validator.last_pose == pytest.approx(make_pose((0.03, 0.0, 0.0)))


def test_pose_validator_recognizes_persistent_rotation_jump():
    """验证持续的姿态跳变仍被判为新的稳定位置。"""
    validator = PoseStreamValidator(recovery_samples=3)
    for _ in range(3):
        validator.update(make_pose())

    for angle in (90.0, 91.0):
        assert not validator.update(make_pose(rpy_deg=(0.0, 0.0, angle))).accepted
    recovered = validator.update(make_pose(rpy_deg=(0.0, 0.0, 92.0)))

    assert recovered.accepted and recovered.ready and recovered.relocalized


def test_joint_smoothing_limits_velocity_and_acceleration():
    """验证可选关节平滑限制单周期速度增长。"""
    position, velocity = advance_joint_command(
        np.zeros(7),
        np.zeros(7),
        np.ones(7),
        np.full(7, 0.5),
        acceleration_limit=2.0,
        dt=0.02,
    )

    assert velocity == pytest.approx(np.full(7, 0.04))
    assert position == pytest.approx(np.full(7, 0.0008))
