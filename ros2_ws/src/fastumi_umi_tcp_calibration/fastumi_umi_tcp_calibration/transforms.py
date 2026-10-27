"""刚体变换工具，提取自 jetson_dev；平移单位米，角度单位弧度。"""

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


def rotation_log(rotation):
    """3x3 旋转矩阵的轴角向量（弧度），用于姿态误差。"""
    cos_angle = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3)
    vector = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
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


def average_transforms(transforms):
    """多个相近刚体变换的平均：平移取均值，旋转取弦距离均值后 SVD 投影。"""
    transforms = list(transforms)
    rotation_sum = sum(transform[:3, :3] for transform in transforms)
    u, _, vt = np.linalg.svd(rotation_sum)
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    translation = np.mean([transform[:3, 3] for transform in transforms], axis=0)
    return make_transform(rotation, translation)
