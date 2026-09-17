"""封装 RM75 URDF 加载、正运动学和 Placo 逆运动学求解。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import numpy as np


# RM75 七个关节的固定驱动顺序。
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))


def resolve_rm75_urdf(override: str = "") -> Path:
    """解析参数覆盖或 rm_description 包内的 RM75 URDF 路径。"""
    if override:
        urdf_path = Path(override).expanduser().resolve()
    else:
        description_share = Path(get_package_share_directory("rm_description"))
        urdf_path = description_share / "urdf" / "rm_75.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"RM75 URDF 不存在: {urdf_path}")
    return urdf_path


def load_joint_velocity_limits(urdf_path: Path) -> np.ndarray:
    """按驱动顺序读取 RM75 URDF 中的七轴速度上限。"""
    root = ET.parse(urdf_path).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    limits = []
    for joint_name in JOINT_NAMES:
        joint = joints.get(joint_name)
        limit = None if joint is None else joint.find("limit")
        if limit is None or "velocity" not in limit.attrib:
            raise ValueError(f"URDF 缺少 {joint_name} 的速度上限")
        velocity = float(limit.attrib["velocity"])
        if not np.isfinite(velocity) or velocity <= 0.0:
            raise ValueError(f"URDF 中 {joint_name} 的速度上限无效")
        limits.append(velocity)
    return np.asarray(limits, dtype=np.float64)


class PlacoRm75Kinematics:
    """为遥操节点提供可替换的 RM75 运动学接口。"""

    def __init__(
        self,
        urdf_path: Path,
        base_frame: str = "base_link",
        eef_frame: str = "Link7",
        control_rate_hz: float = 50.0,
    ) -> None:
        """加载机器人模型并创建带关节约束的末端位姿任务。"""
        del base_frame  # RM75 URDF 的浮动基座由求解器固定，保留参数供接口说明。
        try:
            import placo
        except ImportError as error:
            raise RuntimeError(
                "缺少 placo 0.9.23，请按 ros2_ws/README.md 配置共享的 "
                "NumPy 2 虚拟环境"
            ) from error

        self._placo = placo
        self._eef_frame = str(eef_frame)
        self._robot = placo.RobotWrapper(
            str(urdf_path), placo.Flags.ignore_collisions
        )
        if self._robot.state.q.shape[0] < 14:
            raise ValueError("RM75 Placo 模型未提供七个可控关节")
        self._robot.state.q[7:] = np.zeros(7, dtype=np.float64)
        self._robot.update_kinematics()
        self._solver = placo.KinematicsSolver(self._robot)
        self._solver.dt = 1.0 / float(control_rate_hz)
        self._solver.mask_fbase(True)
        self._solver.enable_joint_limits(True)
        self._solver.enable_velocity_limits(True)
        initial_pose = self._robot.get_T_world_frame(self._eef_frame)
        self._eef_task = self._solver.add_frame_task(
            self._eef_frame, initial_pose
        )
        self._eef_task.configure("tracker_teleop_target", "soft", 1.0)

    def set_joint_positions(self, positions: np.ndarray) -> None:
        """把七轴关节角写入模型并更新正运动学。"""
        joints = np.asarray(positions, dtype=np.float64)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("RM75 关节位置必须包含七个有限数值")
        self._robot.state.q[7:] = joints
        self._robot.update_kinematics()

    def end_effector_pose(self, positions: np.ndarray) -> np.ndarray:
        """返回指定七轴关节角对应的基座到 Link7 齐次变换。"""
        self.set_joint_positions(positions)
        return np.asarray(
            self._robot.get_T_world_frame(self._eef_frame),
            dtype=np.float64,
        ).copy()

    def solve(
        self, target_pose: np.ndarray, seed_positions: np.ndarray
    ) -> np.ndarray:
        """以给定关节角为种子求解 Link7 目标位姿。"""
        target = np.asarray(target_pose, dtype=np.float64)
        if target.shape != (4, 4) or not np.all(np.isfinite(target)):
            raise ValueError("IK 目标必须是有限的 4x4 位姿矩阵")
        self.set_joint_positions(seed_positions)
        self._eef_task.T_world_frame = target
        self._solver.solve(True)
        self._robot.update_kinematics()
        solution = np.asarray(
            self._robot.state.q[7:], dtype=np.float64
        ).copy()
        if solution.shape != (7,) or not np.all(np.isfinite(solution)):
            raise RuntimeError("Placo 返回了无效的七轴关节目标")
        return solution
