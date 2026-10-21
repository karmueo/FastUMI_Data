"""RM75 串联链 FK / IK：解析 URDF 的 joint1..7，输出 ^base T_flange（Link7）。

只依赖 numpy，不使用控制器侧 IK（movej_p 受控制器当前工具/工作坐标系影响），
从而能在下发前检查关节限位和单次关节跳变。关节角单位弧度，长度单位米。
"""

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ElementTree

import numpy as np

from fastumi_bringup.transforms import (
    axis_angle_matrix, make_transform, rotation_log, rpy_matrix,
)


JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))


@dataclass(frozen=True)
class RevoluteJoint:
    """单个转动关节：父系到关节系的固定 origin、关节轴和限位（弧度）。"""

    name: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class ArmKinematics:
    """按 URDF 关节顺序构造的 7 自由度串联链。"""

    def __init__(self, joints):
        self.joints = tuple(joints)
        if len(self.joints) != 7:
            raise ValueError(f"RM75 需要 7 个关节，URDF 提供 {len(self.joints)} 个")
        self.lower = np.array([joint.lower for joint in self.joints])
        self.upper = np.array([joint.upper for joint in self.joints])

    @classmethod
    def from_urdf(cls, path, joint_names=JOINT_NAMES):
        """按 joint_names 顺序读取转动关节；缺失关节或非转动关节直接报错。"""
        root = ElementTree.parse(Path(path)).getroot()
        by_name = {element.get("name"): element for element in root.findall("joint")}
        joints = []
        for name in joint_names:
            element = by_name.get(name)
            if element is None or element.get("type") != "revolute":
                raise ValueError(f"URDF 缺少转动关节 {name}")
            origin = element.find("origin")
            xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None
                                      else "0 0 0").split()]
            rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None
                                      else "0 0 0").split()]
            axis = np.array([float(v) for v in element.find("axis").get("xyz").split()])
            limit = element.find("limit")
            joints.append(RevoluteJoint(
                name=name,
                origin=make_transform(rpy_matrix(*rpy), xyz),
                axis=axis / np.linalg.norm(axis),
                lower=float(limit.get("lower")),
                upper=float(limit.get("upper")),
            ))
        return cls(joints)

    def _frames(self, q):
        """返回每个关节在 base 下的位姿（含关节转角后）及末端位姿。"""
        transform = np.eye(4)
        frames = []
        for joint, angle in zip(self.joints, q):
            transform = transform @ joint.origin
            transform[:3, :3] = transform[:3, :3] @ axis_angle_matrix(joint.axis, float(angle))
            frames.append(transform)
        return frames, transform

    def forward(self, q):
        """^base T_flange，q 为 7 个关节弧度。"""
        return self._frames(np.asarray(q, dtype=float))[1]

    def jacobian(self, q):
        """base 系下 6x7 几何雅可比，行顺序 [v; ω]。"""
        return self._jacobian_from_frames(*self._frames(np.asarray(q, dtype=float)))

    def _jacobian_from_frames(self, frames, end):
        """由已计算的各关节位姿构造雅可比，避免 IK 迭代中重复 FK。"""
        jacobian = np.zeros((6, 7))
        for index, (joint, frame) in enumerate(zip(self.joints, frames)):
            axis = frame[:3, :3] @ joint.axis
            jacobian[:3, index] = np.cross(axis, end[:3, 3] - frame[:3, 3])
            jacobian[3:, index] = axis
        return jacobian

    def within_limits(self, q, margin=0.0):
        """所有关节都在 [lower+margin, upper-margin] 内。"""
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= self.lower + margin) and np.all(q <= self.upper - margin))

    def solve_ik(self, target, seed, *, max_iterations=300, damping=0.02,
                 position_tolerance=5e-4, rotation_tolerance=math.radians(0.1),
                 null_space_gain=0.05):
        """阻尼最小二乘 IK，从 seed 出发求 ^base T_flange = target。

        冗余自由度用零空间项拉向 seed，避免 7 轴构型在样本间大幅翻转。
        返回关节弧度数组；不收敛或越过 URDF 限位时返回 None。
        """
        seed = np.asarray(seed, dtype=float)
        q = seed.copy()
        best_error = math.inf
        stalled = 0
        for _ in range(max_iterations):
            frames, current = self._frames(q)
            position_error = target[:3, 3] - current[:3, 3]
            rotation_error = current[:3, :3] @ rotation_log(
                current[:3, :3].T @ target[:3, :3])
            if (np.linalg.norm(position_error) < position_tolerance and
                    np.linalg.norm(rotation_error) < rotation_tolerance):
                return q if self.within_limits(q) else None
            # 误差 0.1 m ≈ 0.1 rad 等权；连续 20 次无明显下降视为不可达（奇异或限位）。
            total_error = np.linalg.norm(position_error) + np.linalg.norm(rotation_error)
            if total_error < best_error * 0.99:
                best_error = total_error
                stalled = 0
            else:
                stalled += 1
                if stalled >= 20:
                    return None
            error = np.concatenate([position_error, rotation_error])
            jacobian = self._jacobian_from_frames(frames, current)
            jjt = jacobian @ jacobian.T + damping ** 2 * np.eye(6)
            pseudo_inverse = jacobian.T @ np.linalg.solve(jjt, np.eye(6))
            null_space = np.eye(7) - pseudo_inverse @ jacobian
            step = pseudo_inverse @ error + null_space @ (null_space_gain * (seed - q))
            # 单步限幅，保证线性化有效。
            largest = np.max(np.abs(step))
            if largest > 0.2:
                step *= 0.2 / largest
            q = np.clip(q + step, self.lower, self.upper)
        return None
