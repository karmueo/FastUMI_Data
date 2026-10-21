"""齐次变换、四元数与旋转差工具；四元数统一为 (x, y, z, w)。"""

import math

import numpy as np


def make_transform(rotation, translation):
    """由 3x3 旋转和 3 维平移（米）组成 4x4 齐次矩阵。"""
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def invert(transform):
    """刚体变换求逆，避免通用矩阵求逆的数值误差。"""
    rotation = transform[:3, :3]
    return make_transform(rotation.T, -rotation.T @ transform[:3, 3])


def axis_angle_matrix(axis, angle):
    """绕单位轴旋转 angle 弧度的 3x3 矩阵（Rodrigues）。"""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew


def rpy_matrix(roll, pitch, yaw):
    """URDF 约定的 rpy：R = Rz(yaw) · Ry(pitch) · Rx(roll)。"""
    return (axis_angle_matrix((0, 0, 1), yaw) @ axis_angle_matrix((0, 1, 0), pitch)
            @ axis_angle_matrix((1, 0, 0), roll))


def rotation_log(rotation):
    """3x3 旋转矩阵的轴角向量（弧度），用于姿态误差。"""
    cos_angle = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3)
    vector = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    if math.pi - angle < 1e-6:
        # 接近 180° 时反对称部分退化，改用对称部分取轴。
        diagonal = np.sqrt(np.clip((np.diag(rotation) + 1.0) / 2.0, 0.0, None))
        index = int(np.argmax(diagonal))
        axis = rotation[:, index] + np.eye(3)[:, index]
        return axis / np.linalg.norm(axis) * angle
    return vector / (2.0 * math.sin(angle)) * angle


def rotation_angle(rotation_a, rotation_b):
    """两个旋转之间的夹角（弧度）。"""
    return float(np.linalg.norm(rotation_log(rotation_a.T @ rotation_b)))


def quaternion_from_matrix(rotation):
    """3x3 旋转矩阵转 (x, y, z, w) 单位四元数，w ≥ 0。"""
    m = np.asarray(rotation, dtype=float)
    trace = np.trace(m)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s,
             (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s,
             (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return q if q[3] >= 0.0 else -q


def matrix_from_quaternion(quaternion):
    """(x, y, z, w) 四元数转 3x3 旋转矩阵。"""
    x, y, z, w = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def average_transforms(transforms):
    """多个相近刚体变换的平均：平移取均值，旋转取弦距离均值后 SVD 投影。"""
    transforms = list(transforms)
    rotation_sum = sum(transform[:3, :3] for transform in transforms)
    u, _, vt = np.linalg.svd(rotation_sum)
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    translation = np.mean([transform[:3, 3] for transform in transforms], axis=0)
    return make_transform(rotation, translation)
