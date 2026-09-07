"""使用 URDF 运动链计算批量正运动学，仅依赖 NumPy/SciPy，不加载网格。"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def _vector(element, attribute, default):
    """读取有限三维向量；缺少属性时使用 URDF 默认值。"""
    value = default if element is None else element.get(attribute, default)
    result = np.fromstring(value, sep=" ", dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"Invalid URDF {attribute}: {value}")
    return result


class UrdfKinematics:
    """从指定基座到末端解析单链，输入弧度关节角并返回米制齐次变换。"""

    def __init__(self, urdf_path, joint_names, base_link="base_link", end_link="Link7"):
        """验证关节顺序、类型和运动链，忽略视觉、碰撞与惯性定义。"""
        self.joint_names = tuple(joint_names)
        self.base_link = base_link
        self.end_link = end_link
        root = ET.parse(Path(urdf_path)).getroot()
        links = {link.get("name") for link in root.findall("link")}
        if base_link not in links or end_link not in links:
            raise ValueError("Base or end link missing from URDF")
        by_child = {}
        for joint in root.findall("joint"):
            child = joint.find("child").get("link")
            if child in by_child:
                raise ValueError(f"Multiple parents for {child}")
            by_child[child] = joint
        chain = []
        current = end_link
        visited = set()
        while current != base_link:
            if current in visited or current not in by_child:
                raise ValueError("URDF chain is disconnected or cyclic")
            visited.add(current)
            joint = by_child[current]
            chain.append(joint)
            current = joint.find("parent").get("link")
        chain.reverse()
        movable_names = tuple(joint.get("name") for joint in chain if joint.get("type") != "fixed")
        if movable_names != self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError(f"Joint order mismatch: URDF={movable_names}, input={self.joint_names}")
        self.chain = []
        for joint in chain:
            kind = joint.get("type")
            if kind not in ("fixed", "revolute", "continuous") or joint.find("mimic") is not None:
                raise ValueError(f"Unsupported joint: {joint.get('name')} ({kind})")
            origin = joint.find("origin")
            matrix = np.eye(4)
            matrix[:3, 3] = _vector(origin, "xyz", "0 0 0")
            # URDF 固定轴 RPY：Rz(yaw) @ Ry(pitch) @ Rx(roll)。
            matrix[:3, :3] = Rotation.from_euler("xyz", _vector(origin, "rpy", "0 0 0")).as_matrix()
            axis = _vector(joint.find("axis"), "xyz", "1 0 0")
            if np.linalg.norm(axis) < 1e-12:
                raise ValueError("Joint axis must be nonzero")
            self.chain.append((matrix, axis / np.linalg.norm(axis), kind != "fixed"))

    def forward(self, joint_angles):
        """对形状 [...,关节数] 的弧度数组执行 FK，返回 [...,4,4]；拒绝非有限输入。"""
        angles = np.asarray(joint_angles, dtype=np.float64)
        if angles.ndim < 1 or angles.shape[-1] != len(self.joint_names) or not np.isfinite(angles).all():
            raise ValueError("Invalid joint angles or dimension")
        shape = angles.shape[:-1]
        flat = angles.reshape(-1, len(self.joint_names))
        result = np.broadcast_to(np.eye(4), (len(flat), 4, 4)).copy()
        index = 0
        for origin, axis, movable in self.chain:
            result = result @ origin
            if movable:
                motion = np.broadcast_to(np.eye(4), result.shape).copy()
                motion[:, :3, :3] = Rotation.from_rotvec(flat[:, index, None] * axis).as_matrix()
                result = result @ motion
                index += 1
        return result.reshape(shape + (4, 4))
