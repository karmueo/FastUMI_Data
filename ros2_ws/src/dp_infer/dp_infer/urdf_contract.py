"""检查训练 URDF 与 Placo 控制器 URDF 的七轴运动学一致性。"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))


def _joint_signature(path):
    """提取 base_link→Link7 链的关节坐标、轴和位置/速度限制。"""
    root = ET.parse(Path(path)).getroot()
    links = {link.get("name") for link in root.findall("link")}
    if not {"base_link", "Link7"}.issubset(links):
        raise ValueError(f"URDF lacks base_link or Link7: {path}")
    by_name = {joint.get("name"): joint for joint in root.findall("joint")}
    if len(by_name) != len(root.findall("joint")):
        raise ValueError(f"URDF has duplicate joint names: {path}")
    signature = []
    for index, name in enumerate(JOINT_NAMES, 1):
        joint = by_name.get(name)
        if joint is None or joint.get("type") != "revolute":
            raise ValueError(f"URDF lacks revolute {name}: {path}")
        parent = joint.find("parent")
        child = joint.find("child")
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        expected_parent = "base_link" if index == 1 else f"Link{index - 1}"
        if (parent is None or child is None or origin is None or axis is None
                or limit is None or parent.get("link") != expected_parent
                or child.get("link") != f"Link{index}"):
            raise ValueError(f"Unexpected {name} chain: {path}")
        values = []
        for element, attribute, size in (
            (origin, "xyz", 3), (origin, "rpy", 3), (axis, "xyz", 3),
            (limit, "lower", 1), (limit, "upper", 1), (limit, "velocity", 1),
        ):
            parsed = np.fromstring(element.get(attribute, ""), sep=" ")
            if parsed.shape != (size,) or not np.isfinite(parsed).all():
                raise ValueError(f"Invalid {name} {attribute}: {path}")
            values.extend(parsed.tolist())
        signature.append(values)
    return np.asarray(signature, dtype=np.float64)


def validate_kinematic_equivalence(training_urdf, control_urdf):
    """拒绝与训练七轴坐标、轴或运动限制不一致的控制 URDF。"""
    training = _joint_signature(training_urdf)
    control = _joint_signature(control_urdf)
    if not np.array_equal(training, control):
        raise ValueError("RM75 controller URDF differs from training kinematic chain")
