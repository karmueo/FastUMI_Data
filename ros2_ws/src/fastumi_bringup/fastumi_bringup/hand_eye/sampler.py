"""“看向标定板”的随机位姿生成与可行性过滤。

相机位置在以板中心为球心的球冠上采样（距离、相对板法线倾角、方位角），视线指向板中心
附近，再绕该视线随机滚转。视线默认为光轴；夹爪遮住画面中部时改为 aim_direction，
使板成像在未遮挡区域。^base T_flange = ^base T_cam · (^flange T_cam)⁻¹ 经 IK 求关节。
候选需满足：IK 收敛且离限位有余量、法兰在工作空间盒内、单次关节变化不超限、预测视图中
整块板在画面内且不与夹爪掩膜相交。在可行候选中选与已采样本姿态差最大的一个。
"""

from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from fastumi_bringup.hand_eye.occlusion import check_board_view
from fastumi_bringup.hand_eye.transforms import (
    axis_angle_matrix, invert, make_transform, rotation_angle,
)


@dataclass(frozen=True)
class SamplerSettings:
    """采样范围与安全过滤参数；长度米，角度度，关节弧度。"""

    distance_m: tuple = (0.25, 0.45)
    tilt_deg: tuple = (10.0, 40.0)
    roll_deg: float = 45.0
    target_jitter_m: float = 0.03
    workspace_min: tuple = (-0.6, -0.5, 0.16)
    workspace_max: tuple = (0.6, 0.5, 0.60)
    max_joint_step_rad: float = 1.2
    joint_limit_margin_rad: float = 0.1
    candidate_count: int = 12
    max_tries: int = 200
    view_margin_px: int = 30
    min_board_area_fraction: float = 0.03
    joint_distance_weight: float = 0.05
    aim_direction: tuple = (0.0, 0.0, 1.0)  # 相机系单位向量：板中心应出现的视线方向


@dataclass
class PlanContext:
    """规划所需的当前估计；transforms 均为 4x4，角度弧度。"""

    base_from_board: np.ndarray
    flange_from_camera: np.ndarray
    reference_base_from_camera: np.ndarray
    sampled_base_from_flange: list = field(default_factory=list)


@dataclass(frozen=True)
class Candidate:
    """可执行的目标：关节弧度与对应的法兰、相机位姿。"""

    joints: np.ndarray
    base_from_flange: np.ndarray
    base_from_camera: np.ndarray
    score: float


def viewing_side(base_from_board, base_from_camera):
    """相机位于板法线的哪一侧（+1 / -1），采样保持在同一侧。"""
    camera_in_board = invert(base_from_board) @ base_from_camera[:, 3]
    return 1.0 if camera_in_board[2] >= 0.0 else -1.0


def sample_camera_pose(context, spec, rng, settings):
    """随机生成一个 ^base T_cam（尚未检查可达性）。"""
    board = context.base_from_board
    side = viewing_side(board, context.reference_base_from_camera)
    normal = board[:3, 2] * side
    board_x, board_y = board[:3, 0], board[:3, 1]
    center = board[:3, :3] @ spec.center_point() + board[:3, 3]

    tilt = math.radians(rng.uniform(*settings.tilt_deg))
    azimuth = rng.uniform(0.0, 2.0 * math.pi)
    direction = (math.cos(tilt) * normal + math.sin(tilt) * (
        math.cos(azimuth) * board_x + math.sin(azimuth) * board_y))
    position = center + rng.uniform(*settings.distance_m) * direction
    jitter = rng.uniform(-settings.target_jitter_m, settings.target_jitter_m, 2)
    target = center + jitter[0] * board_x + jitter[1] * board_y

    z_axis = target - position
    z_axis /= np.linalg.norm(z_axis)
    reference_x = context.reference_base_from_camera[:3, 0]
    x_axis = reference_x - (reference_x @ z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(board_y, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    roll = math.radians(rng.uniform(-settings.roll_deg, settings.roll_deg))
    x_axis = axis_angle_matrix(z_axis, roll) @ x_axis
    y_axis = np.cross(z_axis, x_axis)
    # 上面构造的姿态让光轴指向目标；再右乘 align（把 aim_direction 转到光轴），
    # 使目标沿 aim_direction 方向成像。
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1) @ _align_to_optical_axis(
        settings.aim_direction)
    return make_transform(rotation, position)


def _align_to_optical_axis(direction):
    """返回旋转 A，使 A · direction = (0, 0, 1)。"""
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    axis = np.cross(direction, (0.0, 0.0, 1.0))
    sine = np.linalg.norm(axis)
    if sine < 1e-9:
        return np.eye(3)
    return axis_angle_matrix(axis / sine, math.atan2(sine, direction[2]))


def aim_for_mask(camera, mask, center_weight=0.5):
    """在夹爪掩膜之外选择板中心的瞄准像素，返回 ((u, v) 像素, 相机系单位视线)。

    取未遮挡区域内切圆半径（到掩膜和图像边界的距离）最大、且靠近画面中心的点；
    mask 为 None 时瞄准画面中心。
    """
    center = np.array([camera.width / 2.0, camera.height / 2.0])
    if mask is None or not np.any(mask):
        pixel = center
    else:
        free = (mask == 0).astype(np.uint8)
        free[[0, -1], :] = 0
        free[:, [0, -1]] = 0
        distance = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        rows, cols = np.indices(distance.shape)
        score = distance - center_weight * np.hypot(cols - center[0], rows - center[1])
        row, col = np.unravel_index(np.argmax(score), score.shape)
        pixel = np.array([col, row], dtype=float)
    normalized = cv2.fisheye.undistortPoints(
        pixel.reshape(1, 1, 2), camera.K, camera.D).reshape(2)
    ray = np.array([normalized[0], normalized[1], 1.0])
    return (float(pixel[0]), float(pixel[1])), ray / np.linalg.norm(ray)


def check_reachable(kinematics, base_from_flange, current_joints, settings):
    """IK 及安全过滤；返回 (关节, 原因)，不可行时关节为 None。"""
    position = base_from_flange[:3, 3]
    if (np.any(position < np.asarray(settings.workspace_min)) or
            np.any(position > np.asarray(settings.workspace_max))):
        return None, "法兰超出工作空间"
    joints = kinematics.solve_ik(base_from_flange, current_joints)
    if joints is None:
        return None, "IK 不收敛"
    if not kinematics.within_limits(joints, settings.joint_limit_margin_rad):
        return None, "接近关节限位"
    step = float(np.max(np.abs(joints - np.asarray(current_joints))))
    if step > settings.max_joint_step_rad:
        return None, f"单次关节变化 {math.degrees(step):.0f}° 过大"
    return joints, ""


def plan_next(kinematics, spec, camera, mask, context, current_joints, rng,
              settings=SamplerSettings()):
    """生成下一个采样目标；无可行候选时返回 (None, 拒绝原因统计)。"""
    camera_from_flange = invert(context.flange_from_camera)
    candidates = []
    reasons = {}
    for _ in range(settings.max_tries):
        if len(candidates) >= settings.candidate_count:
            break
        base_from_camera = sample_camera_pose(context, spec, rng, settings)
        view = check_board_view(
            spec, camera, invert(base_from_camera) @ context.base_from_board, mask,
            margin_px=settings.view_margin_px,
            min_area_fraction=settings.min_board_area_fraction)
        if not view.ok:
            key = view.reason.split(" ")[0]
            reasons[key] = reasons.get(key, 0) + 1
            continue
        base_from_flange = base_from_camera @ camera_from_flange
        joints, reason = check_reachable(kinematics, base_from_flange, current_joints, settings)
        if joints is None:
            key = reason.split(" ")[0]
            reasons[key] = reasons.get(key, 0) + 1
            continue
        novelty = min((rotation_angle(base_from_flange[:3, :3], sample[:3, :3])
                       for sample in context.sampled_base_from_flange), default=math.pi)
        joint_distance = float(np.linalg.norm(joints - np.asarray(current_joints)))
        candidates.append(Candidate(
            joints=joints, base_from_flange=base_from_flange,
            base_from_camera=base_from_camera,
            score=novelty - settings.joint_distance_weight * joint_distance))
    if not candidates:
        return None, reasons
    return max(candidates, key=lambda candidate: candidate.score), reasons


# 自举阶段相对参考法兰位姿的小姿态扰动（法兰系轴，角度度）。
BOOTSTRAP_ROTATIONS = (
    ((1, 0, 0), 8.0), ((1, 0, 0), -8.0), ((0, 1, 0), 8.0), ((0, 1, 0), -8.0),
    ((0, 0, 1), 15.0), ((0, 0, 1), -15.0), ((1, 1, 0), 8.0), ((1, -1, 0), -8.0),
)


def bootstrap_targets(reference_base_from_flange, pivot_in_flange=(0.0, 0.0, 0.0)):
    """围绕法兰系中 pivot 点的小旋转目标列表，用于没有可靠外参时的粗标定。

    pivot 取相机前方板附近的点时，旋转后板仍大致留在视野中心。
    """
    pivot = make_transform(np.eye(3), pivot_in_flange)
    targets = []
    for axis, angle in BOOTSTRAP_ROTATIONS:
        rotation = make_transform(axis_angle_matrix(axis, math.radians(angle)), np.zeros(3))
        targets.append(reference_base_from_flange @ pivot @ rotation @ invert(pivot))
    return targets
