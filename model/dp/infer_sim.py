"""提供 DP policy 在仿真中的推理、动作后处理和 ROS 2 命令联动入口。"""

import argparse
import csv
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import hydra
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from umi.common.pose_util import (
    mat_to_pose,
    mat_to_pose10d,
    pose10d_to_mat,
    pose_to_mat,
)

STACK_CUBE_GRASP_QUAT_WXYZ = np.array([0.0, 0.99992577, 0.01218451, 0.0], dtype=np.float64)
"""Stack Cube 成功轨迹统计得到的固定顶抓姿态四元数。"""


def _format_debug_array(values, precision=6):
    """将调试数组格式化为紧凑的 Python list 字符串。"""
    return np.asarray(values, dtype=np.float64).round(precision).tolist()


def _object_pose_error_fields(abs_action, object_name, object_pose):
    """生成目标动作相对单个物体中心的 XY/Z 误差字段。"""
    pose = np.asarray(object_pose, dtype=np.float64).reshape(-1)
    if pose.shape[0] < 3:
        return []

    target_pos = np.asarray(abs_action[:3], dtype=np.float64)
    object_pos = pose[:3]
    xy_error = float(np.linalg.norm(target_pos[:2] - object_pos[:2]))
    z_error = float(target_pos[2] - object_pos[2])
    return [
        f"{object_name}_pos={_format_debug_array(object_pos)}",
        f"{object_name}_xy_error={xy_error:.6f}",
        f"{object_name}_z_error={z_error:.6f}",
    ]


def _object_pose_error_values(abs_action, object_pose):
    """计算目标动作相对单个物体中心的 XY/Z 误差。

    Args:
        abs_action: 8 维动作 `[x,y,z,w,qx,qy,qz,gripper]`。
        object_pose: 物体世界位姿 `[x,y,z,w,qx,qy,qz]`。

    Returns:
        tuple[float, float]: 水平距离误差和 Z 方向误差。
    """
    pose = np.asarray(object_pose, dtype=np.float64).reshape(-1)
    if pose.shape[0] < 3:
        return None, None

    target_pos = np.asarray(abs_action[:3], dtype=np.float64)
    object_pos = pose[:3]
    xy_error = float(np.linalg.norm(target_pos[:2] - object_pos[:2]))
    z_error = float(target_pos[2] - object_pos[2])
    return xy_error, z_error


def _csv_float(value):
    """把 CSV 中的浮点值格式化为紧凑字符串。"""
    if value is None:
        return ""
    return f"{float(value):.12g}"


ROLLOUT_LOG_FIELDS = [
    "timestamp",
    "action_index",
    "action_pose_mode",
    "debug_cup_xy_locked",
    "cur_ee_x",
    "cur_ee_y",
    "cur_ee_z",
    "rel_x",
    "rel_y",
    "rel_z",
    "raw_x",
    "raw_y",
    "raw_z",
    "raw_w",
    "raw_qx",
    "raw_qy",
    "raw_qz",
    "published_x",
    "published_y",
    "published_z",
    "published_w",
    "published_qx",
    "published_qy",
    "published_qz",
    "gripper_width",
    "min_tcp_z",
    "z_clamped",
    "cup_xy_error",
    "cup_z_error",
    "plate_xy_error",
    "plate_z_error",
]
"""DP rollout CSV 诊断字段。"""


def append_rollout_log_row(
    csv_path,
    action_index,
    cur_ee_pose,
    rel_action,
    raw_action,
    published_action,
    min_tcp_z=None,
    action_pose_mode="relative",
    debug_cup_xy_locked=False,
    debug_object_poses=None,
):
    """追加一行 DP rollout 动作诊断 CSV。

    Args:
        csv_path: CSV 输出路径；为 None 时不写入。
        action_index: 当前动作在模型输出序列中的索引。
        cur_ee_pose: 当前末端位姿 `[x,y,z,w,qx,qy,qz]`。
        rel_action: 模型后处理后的相对动作。
        raw_action: z clamp 前的 8 维绝对动作。
        published_action: 实际发布的 8 维绝对动作。
        min_tcp_z: 当前配置的 TCP z 下限。
        action_pose_mode: 模型动作解释模式。
        debug_cup_xy_locked: 当前动作是否被调试逻辑锁定到 cup XY。
        debug_object_poses: 可选 cup/plate 位姿字典。
    """
    if csv_path is None:
        return

    log_path = Path(csv_path)  # rollout CSV 输出路径
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cur_ee = np.asarray(cur_ee_pose, dtype=np.float64)
    rel = np.asarray(rel_action, dtype=np.float64)
    raw = np.asarray(raw_action, dtype=np.float64)
    published = np.asarray(published_action, dtype=np.float64)
    z_clamped = int(not np.isclose(float(raw[2]), float(published[2])))
    cup_xy_error, cup_z_error = None, None
    plate_xy_error, plate_z_error = None, None
    if debug_object_poses:
        if "cup_pose" in debug_object_poses:
            cup_xy_error, cup_z_error = _object_pose_error_values(
                published, debug_object_poses["cup_pose"]
            )
        if "plate_pose" in debug_object_poses:
            plate_xy_error, plate_z_error = _object_pose_error_values(
                published, debug_object_poses["plate_pose"]
            )

    row = {
        "timestamp": _csv_float(time.time()),
        "action_index": str(int(action_index)),
        "action_pose_mode": str(action_pose_mode),
        "debug_cup_xy_locked": str(int(bool(debug_cup_xy_locked))),
        "cur_ee_x": _csv_float(cur_ee[0]),
        "cur_ee_y": _csv_float(cur_ee[1]),
        "cur_ee_z": _csv_float(cur_ee[2]),
        "rel_x": _csv_float(rel[0]),
        "rel_y": _csv_float(rel[1]),
        "rel_z": _csv_float(rel[2]),
        "raw_x": _csv_float(raw[0]),
        "raw_y": _csv_float(raw[1]),
        "raw_z": _csv_float(raw[2]),
        "raw_w": _csv_float(raw[3]),
        "raw_qx": _csv_float(raw[4]),
        "raw_qy": _csv_float(raw[5]),
        "raw_qz": _csv_float(raw[6]),
        "published_x": _csv_float(published[0]),
        "published_y": _csv_float(published[1]),
        "published_z": _csv_float(published[2]),
        "published_w": _csv_float(published[3]),
        "published_qx": _csv_float(published[4]),
        "published_qy": _csv_float(published[5]),
        "published_qz": _csv_float(published[6]),
        "gripper_width": _csv_float(published[-1]),
        "min_tcp_z": _csv_float(min_tcp_z),
        "z_clamped": str(z_clamped),
        "cup_xy_error": _csv_float(cup_xy_error),
        "cup_z_error": _csv_float(cup_z_error),
        "plate_xy_error": _csv_float(plate_xy_error),
        "plate_z_error": _csv_float(plate_z_error),
    }

    should_write_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=ROLLOUT_LOG_FIELDS)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


def build_action_diagnostic_message(
    action_index,
    cur_ee_pose,
    rel_action,
    abs_action,
    action_pose_mode="relative",
    debug_cup_xy_locked=False,
    debug_object_poses=None,
):
    """构造 DP 单步动作诊断日志。

    Args:
        action_index: 当前动作在本次模型输出序列中的索引。
        cur_ee_pose: 当前末端世界位姿 `[x,y,z,w,qx,qy,qz]`。
        rel_action: 模型后处理后的相对动作 `[x,y,z,rotvec,gripper]`。
        abs_action: 发布给仿真的世界动作 `[x,y,z,w,qx,qy,qz,gripper]`。
        action_pose_mode: 模型动作解释模式。
        debug_cup_xy_locked: 当前动作是否被调试逻辑锁定到 cup XY。
        debug_object_poses: 可选 cup/plate 位姿字典，值为 `[x,y,z,w,qx,qy,qz]`。

    Returns:
        str: 单行诊断日志。
    """
    message_fields = [
        "[DP DEBUG]",
        f"action_index={action_index}",
        f"action_pose_mode={action_pose_mode}",
        f"debug_cup_xy_locked={int(bool(debug_cup_xy_locked))}",
        f"cur_ee={_format_debug_array(cur_ee_pose)}",
        f"rel_action={_format_debug_array(rel_action)}",
        f"abs_action={_format_debug_array(abs_action)}",
        f"gripper_width={float(abs_action[-1]):.6g}",
    ]

    for object_key in ("cup_pose", "plate_pose"):
        if not debug_object_poses or object_key not in debug_object_poses:
            continue
        object_name = object_key.removesuffix("_pose")
        message_fields.extend(
            _object_pose_error_fields(abs_action, object_name, debug_object_poses[object_key])
        )

    return " ".join(message_fields)


def should_publish_inference(last_publish_time, now, min_interval_sec):
    """判断当前同步观测是否允许触发一次推理发布。

    Args:
        last_publish_time: 上一次发布动作的单调时钟秒数；首次发布时为 None。
        now: 当前单调时钟秒数。
        min_interval_sec: 最小发布间隔，非正数表示不节流。

    Returns:
        bool: 达到发布间隔时返回 True。
    """
    if min_interval_sec <= 0.0 or last_publish_time is None:
        return True
    return (float(now) - float(last_publish_time)) >= float(min_interval_sec)


def resolve_n_action_steps(cli_n_action_steps, ckpt_n_action_steps):
    """解析部署时实际执行的动作步数。

    Args:
        cli_n_action_steps: 命令行指定的动作步数。
        ckpt_n_action_steps: checkpoint 中保存的训练配置动作步数。

    Returns:
        int: 实际部署执行的动作步数；命令行值优先。
    """
    selected_steps = cli_n_action_steps if cli_n_action_steps is not None else ckpt_n_action_steps
    return int(selected_steps)


def parse_debug_object_pose(raw_pose):
    """解析逗号分隔或括号列表形式的调试物体位姿。"""
    if raw_pose is None:
        return None
    cleaned_pose = raw_pose.strip().strip("[]")
    if not cleaned_pose:
        return None
    pose = np.array([float(value) for value in cleaned_pose.split(",")], dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Debug object pose must contain 7 values, got {pose.shape[0]}.")
    return pose


def load_debug_object_poses(cup_pose_arg=None, plate_pose_arg=None):
    """从命令行或环境变量加载可选 cup/plate 调试位姿。"""
    object_poses = {}
    cup_pose = parse_debug_object_pose(cup_pose_arg or os.environ.get("UMI_DEBUG_CUP_POSE"))
    plate_pose = parse_debug_object_pose(plate_pose_arg or os.environ.get("UMI_DEBUG_PLATE_POSE"))
    if cup_pose is not None:
        object_poses["cup_pose"] = cup_pose
    if plate_pose is not None:
        object_poses["plate_pose"] = plate_pose
    return object_poses


def pose_to_matrix(pose, rot_type="rotvec"):
    """
    将 6D 位姿 (平移 + 旋转向量 或 四元数) 转换为 4×4 齐次变换矩阵。

    参数:
        pose (array-like):
            - 如果 rot_type='rotvec'，则为长度为 6 的数组，前三个是平移
              (x, y, z)，后三个是旋转向量。
            - 如果 rot_type='quat'，则为长度为 7 的数组，前三个是平移
              (x, y, z)，后四个是四元数 (w, x, y, z)。
        rot_type (str): 旋转的表示类型，'rotvec' 或 'quat'。

    返回:
        np.ndarray: 4×4 齐次变换矩阵。
    """
    translation = np.array(pose[:3])

    if rot_type == "rotvec":
        rot_vec = np.array(pose[3:])
        rot_mat = Rotation.from_rotvec(rot_vec).as_matrix()
    elif rot_type == "quat":
        quat_wxyz = np.array(pose[3:])
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rot_mat = Rotation.from_quat(quat_xyzw).as_matrix()
    else:
        raise ValueError(f"Unsupported rot_type: {rot_type}")

    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = translation

    return T


def matrix_to_pose(T):
    """
    将 4x4 齐次变换矩阵转换为 3D 位置 (x, y, z) 和四元数 (w, x, y, z).

    参数:
        T (np.ndarray): 4x4 齐次变换矩阵.

    返回:
        tuple: (pos, quat)
            - pos (np.ndarray): 长度为 3 的平移向量 (x, y, z).
            - quat (np.ndarray): 长度为 4 的四元数 (w, x, y, z).
    """
    # 提取平移部分
    pos = T[:3, 3]

    # 提取旋转矩阵部分
    rot_mat = T[:3, :3]

    # 旋转矩阵转换为四元数 (默认返回 [x, y, z, w])
    rot = Rotation.from_matrix(rot_mat)
    quat = rot.as_quat()

    # 转换为 (w, x, y, z) 格式
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])

    return np.concatenate([pos, quat_wxyz])


def relative_to_absolute_pose(cur_pose, target_rel_pos):
    """
    cur_pose: 当前时刻世界坐标系下的位姿, 默认的格式是 [x, y, z, w,x,y,z, gripper_width]
    target_pos: 相对目标位姿, 默认的格式是 [x, y, z, rot_x,rot_y,rot_z, gripper_width]

    """
    gripper_width = target_rel_pos[-1]
    target_rot_mat = pose_to_matrix(target_rel_pos[:6], rot_type="rotvec")
    cur_rot_mat = pose_to_matrix(cur_pose[:7], rot_type="quat")

    world_rot_mat = cur_rot_mat @ target_rot_mat  # 左乘
    ee_pose = matrix_to_pose(world_rot_mat)  # 转为 [x,y,z,w,x,y,z] 位置 + 四元数

    target_action = np.concatenate([ee_pose, [gripper_width]])
    return target_action


def absolute_rotvec_to_absolute_pose(model_action):
    """把模型输出的世界系 rotvec 动作转换为仿真发布的 8 维绝对动作。

    Args:
        model_action: 模型后处理动作 `[x, y, z, rx, ry, rz, gripper_width]`。

    Returns:
        np.ndarray: 世界系动作 `[x, y, z, w, qx, qy, qz, gripper_width]`。
    """
    action = np.asarray(model_action, dtype=np.float64).reshape(-1)
    if action.shape[0] != 7:
        raise ValueError(f"Absolute rotvec action must contain 7 values, got {action.shape[0]}.")

    quat_xyzw = Rotation.from_rotvec(action[3:6]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return np.concatenate([action[:3], quat_wxyz, [action[-1]]])


def model_action_to_absolute_pose(cur_pose, model_action, action_pose_mode="relative"):
    """按配置把模型动作解释为发布给仿真的世界系 8 维动作。

    Args:
        cur_pose: 当前末端世界位姿 `[x, y, z, w, qx, qy, qz]`。
        model_action: 模型后处理动作 `[x, y, z, rx, ry, rz, gripper_width]`。
        action_pose_mode: 动作解释模式，`relative` 或 `absolute_rotvec`。

    Returns:
        np.ndarray: 发布动作 `[x, y, z, w, qx, qy, qz, gripper_width]`。
    """
    if action_pose_mode == "relative":
        return relative_to_absolute_pose(cur_pose=cur_pose, target_rel_pos=model_action)
    if action_pose_mode == "absolute_rotvec":
        return absolute_rotvec_to_absolute_pose(model_action)
    raise ValueError(f"Unsupported action_pose_mode: {action_pose_mode}")


def apply_debug_cup_xy_lock(
    action,
    debug_object_poses=None,
    enabled=False,
    released=False,
    z_below=0.22,
    close_width=0.035,
):
    """在 Cup/Plate 调试中把抓取前段动作 XY 锁定到已知 cup 中心。

    Args:
        action: 8 维世界系发布动作 `[x, y, z, w, qx, qy, qz, gripper_width]`。
        debug_object_poses: 包含 `cup_pose` 的调试物体位姿字典。
        enabled: 是否启用锁定。
        released: 是否已经因首次闭合动作释放锁定。
        z_below: 仅当动作 z 不高于该阈值时锁定 XY。
        close_width: 夹爪宽度不大于该值时视为闭合动作，并在本步后释放锁定。

    Returns:
        tuple[np.ndarray, bool, bool]: 修正后的动作、新的释放状态、当前步是否锁定。
    """
    target_action = np.array(action, dtype=np.float64, copy=True)
    if (
        not enabled
        or released
        or not debug_object_poses
        or "cup_pose" not in debug_object_poses
        or float(target_action[2]) > float(z_below)
    ):
        return target_action, bool(released), False

    cup_pose = np.asarray(debug_object_poses["cup_pose"], dtype=np.float64).reshape(-1)
    if cup_pose.shape[0] < 2:
        return target_action, bool(released), False

    target_action[:2] = cup_pose[:2]
    should_release = float(target_action[-1]) <= float(close_width)
    return target_action, bool(released or should_release), True


def apply_fixed_grasp_orientation(action, fixed_quat_wxyz):
    """将动作姿态替换为固定抓取姿态。

    Args:
        action: 8 维动作 `[x, y, z, w, qx, qy, qz, gripper_width]`。
        fixed_quat_wxyz: 固定姿态四元数 `[w, qx, qy, qz]`；为 None 时保持原动作。

    Returns:
        np.ndarray: 姿态被替换后的新动作数组。

    Raises:
        ValueError: 固定四元数不是 4 维或范数为 0 时抛出。
    """
    target_action = np.array(action, dtype=np.float64, copy=True)
    if fixed_quat_wxyz is None:
        return target_action

    fixed_quat = np.asarray(fixed_quat_wxyz, dtype=np.float64)
    if fixed_quat.shape != (4,):
        raise ValueError(f"Fixed grasp quaternion must have shape (4,), got {fixed_quat.shape}.")

    # 四元数归一化，避免 IK 接口收到非单位姿态。
    quat_norm = np.linalg.norm(fixed_quat)
    if quat_norm <= 0:
        raise ValueError("Fixed grasp quaternion norm must be greater than 0.")
    target_action[3:7] = fixed_quat / quat_norm
    return target_action


def apply_min_tcp_z(action, min_tcp_z, action_index=None):
    """应用可选 TCP z 安全下限。

    Args:
        action: 8 维动作 `[x, y, z, w, qx, qy, qz, gripper_width]`。
        min_tcp_z: TCP z 方向最小世界坐标；为 None 时保持动作不变。
        action_index: 当前动作索引，仅用于安全日志。

    Returns:
        np.ndarray: 应用 z 下限后的新动作数组。
    """
    target_action = np.array(action, dtype=np.float64, copy=True)
    if min_tcp_z is None:
        return target_action

    min_z = float(min_tcp_z)  # TCP z 安全下限
    raw_z = float(target_action[2])  # 原始 TCP z 目标
    if raw_z >= min_z:
        return target_action

    target_action[2] = min_z
    print(
        "[DP SAFETY] "
        f"action_index={action_index} raw_z={raw_z:.12g} "
        f"clamped_z={float(target_action[2]):.12g} min_tcp_z={min_z:.12g}"
    )
    return target_action


def apply_fixed_grasp_roll_pitch_keep_yaw(action, fixed_quat_wxyz):
    """固定抓取姿态的 roll/pitch，同时保留模型输出的 yaw。

    Args:
        action: 8 维动作 `[x, y, z, w, qx, qy, qz, gripper_width]`。
        fixed_quat_wxyz: 提供固定 roll/pitch 的顶抓姿态四元数 `[w, qx, qy, qz]`。

    Returns:
        np.ndarray: roll/pitch 被固定、yaw 保留模型输出的新动作数组。
    """
    target_action = np.array(action, dtype=np.float64, copy=True)
    fixed_quat = np.asarray(fixed_quat_wxyz, dtype=np.float64)
    if fixed_quat.shape != (4,):
        raise ValueError(f"Fixed grasp quaternion must have shape (4,), got {fixed_quat.shape}.")

    fixed_norm = np.linalg.norm(fixed_quat)
    if fixed_norm <= 0:
        raise ValueError("Fixed grasp quaternion norm must be greater than 0.")

    model_quat = target_action[3:7]
    model_norm = np.linalg.norm(model_quat)
    if model_norm <= 0:
        raise ValueError("Model quaternion norm must be greater than 0.")

    # SciPy 使用 xyzw 顺序；这里转换后提取世界系 xyz 欧拉角中的 yaw。
    model_euler = Rotation.from_quat(np.roll(model_quat / model_norm, -1)).as_euler("xyz")
    fixed_euler = Rotation.from_quat(np.roll(fixed_quat / fixed_norm, -1)).as_euler("xyz")
    locked_quat_xyzw = Rotation.from_euler(
        "xyz", [fixed_euler[0], fixed_euler[1], model_euler[2]]
    ).as_quat()
    target_action[3:7] = np.roll(locked_quat_xyzw, 1)
    return target_action


def quaternion_to_rotvec(quat):
    """
    将四元数 (w, x, y, z) 转换为旋转向量 (Rodrigues 旋转公式).

    参数:
        quat (array-like): 长度为 4 的数组，表示四元数 (w, x, y, z).

    返回:
        np.ndarray: 旋转向量 (3D).
    """
    # scipy 需要 [x, y, z, w] 格式，因此调整顺序
    rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return rot.as_rotvec()


def resize_and_center_crop(image: Image.Image, target_size=224):
    # 获取原始尺寸
    width, height = image.size

    # 计算缩放比例，使短边变为 target_size
    scale = target_size / min(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 先等比例缩放
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 计算中心裁剪区域
    left = (new_width - target_size) // 2
    top = (new_height - target_size) // 2
    right = left + target_size
    bottom = top + target_size

    # 进行中心裁剪
    image = image.crop((left, top, right, bottom))

    return image


def process_data(raw_data_dict, num_robot=1, obs_pose_repr="relative"):
    """
    raw_data_dict: 包含图像、位姿等数据的字典
    num_robot: 机器人的数量, 单臂为1
    obs_pose_repr: 位姿的表示，默认使用相对于初始位置的位姿
    """
    data = raw_data_dict
    obs_dict = dict()

    rgb_key = "camera0_rgb"
    # Step1: 处理 RGB 图像
    # move channel last to channel first
    # T,H,W,C
    # convert uint8 image to float32
    obs_dict[rgb_key] = np.moveaxis(data[rgb_key], -1, -3).astype(np.float32) / 255.0

    # Step2: 处理其他数据
    for key in raw_data_dict:
        if key == rgb_key:  # RGB 图像原来就是 Uint 类型
            continue
        obs_dict[key] = data[key].astype(np.float32)

    # Step3: 产生两个机器手之间的相对位姿，没有就不作用
    # generate relative pose between two ees
    for robot_id in range(num_robot):
        # convert pose to mat
        pose_mat = pose_to_mat(
            np.concatenate(
                [
                    obs_dict[f"robot{robot_id}_eef_pos"],
                    obs_dict[f"robot{robot_id}_eef_rot_axis_angle"],
                ],
                axis=-1,
            )
        )
        for other_robot_id in range(num_robot):
            if robot_id == other_robot_id:
                continue
            if not f"robot{robot_id}_eef_pos_wrt{other_robot_id}" in self.lowdim_keys:
                continue
            other_pose_mat = pose_to_mat(
                np.concatenate(
                    [
                        obs_dict[f"robot{other_robot_id}_eef_pos"],
                        obs_dict[f"robot{other_robot_id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat, base_pose_mat=other_pose_mat[-1], pose_rep="relative", backward=False
            )
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            obs_dict[f"robot{robot_id}_eef_pos_wrt{other_robot_id}"] = rel_obs_pose[:, :3]
            obs_dict[f"robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}"] = rel_obs_pose[
                :, 3:
            ]

    # Step4: 产生相对于初始状态的末端位姿
    # generate relative pose with respect to episode start
    for robot_id in range(num_robot):
        # convert pose to mat
        pose_mat = pose_to_mat(
            np.concatenate(
                [
                    obs_dict[f"robot{robot_id}_eef_pos"],
                    obs_dict[f"robot{robot_id}_eef_rot_axis_angle"],
                ],
                axis=-1,
            )
        )

        # get start pose
        start_pose = obs_dict[f"robot{robot_id}_demo_start_pose"][0]
        start_pose_mat = pose_to_mat(start_pose)
        rel_obs_pose_mat = convert_pose_mat_rep(
            pose_mat, base_pose_mat=start_pose_mat, pose_rep="relative", backward=False
        )

        rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
        # obs_dict[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
        obs_dict[f"robot{robot_id}_eef_rot_axis_angle_wrt_start"] = rel_obs_pose[:, 3:]

    del_keys = list()
    for key in obs_dict:
        if key.endswith("_demo_start_pose") or key.endswith("_demo_end_pose"):
            del_keys.append(key)
    for key in del_keys:
        del obs_dict[key]

    # Step5: 产生观测数据以及 action
    for robot_id in range(num_robot):
        # convert pose to mat
        pose_mat = pose_to_mat(
            np.concatenate(
                [
                    obs_dict[f"robot{robot_id}_eef_pos"],
                    obs_dict[f"robot{robot_id}_eef_rot_axis_angle"],
                ],
                axis=-1,
            )
        )

        # solve relative obs
        obs_pose_mat = convert_pose_mat_rep(
            pose_mat, base_pose_mat=pose_mat[-1], pose_rep=obs_pose_repr, backward=False
        )

        # convert pose to pos + rot6d representation
        obs_pose = mat_to_pose10d(obs_pose_mat)

        # generate data
        obs_dict[f"robot{robot_id}_eef_pos"] = obs_pose[:, :3]
        obs_dict[f"robot{robot_id}_eef_rot_axis_angle"] = obs_pose[:, 3:]

    # torch_data = {
    #     'obs': dict_apply(obs_dict, torch.from_numpy),
    # }
    return dict_apply(obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))


class FrameBuffer:
    def __init__(self, buffer_size=5):
        """
        初始化帧缓存
        :param buffer_size: 缓存的帧数
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.initialized = False  # 记录是否已初始化

    def update(self, new_frame):
        """更新缓冲区，如果是第一帧，则填充整个 buffer"""
        if not self.initialized:
            self.buffer.extend([new_frame] * self.buffer_size)  # 用第一帧填充整个 buffer
            self.initialized = True
        else:
            self.buffer.append(new_frame)  # 追加新帧，自动丢弃最旧帧

    def get_previous_frame(self):
        """获取上一帧数据（如果存在）"""
        return self.buffer[-1] if self.buffer else None

    def get_all_frames(self):
        """获取整个 buffer 的所有数据"""
        return list(self.buffer)

    def clear(self):
        """清空历史观测帧缓存。"""
        self.buffer.clear()
        self.initialized = False


def clear_runtime_buffers(command=None):
    """根据任务命令清空推理运行时缓存。

    Args:
        command: 任务命令；为 `s` 或 None 时清空观测、动作和初始末端位姿缓存。
    """
    global debug_cup_xy_lock_released, init_ee_pose

    if command not in {None, "s"}:
        return

    obs_buffer_value = globals().get("obs_buffer")
    if obs_buffer_value is not None and hasattr(obs_buffer_value, "clear"):
        obs_buffer_value.clear()

    action_buffer_value = globals().get("action_buffer")
    if action_buffer_value is not None:
        action_buffer_value.clear()

    init_ee_pose = None
    debug_cup_xy_lock_released = False


def debug():
    import debugpy

    # 启动调试器，指定调试端口
    debugpy.listen(5678)

    print("Waiting for debugger to attach...")

    # 在这里设置断点
    debugpy.wait_for_client()


def load_model(ckpt_path):
    """
    加载训练好的模型
    :param ckpt_path: 模型检查点文件路径
    :return: 加载好的模型
    """
    import dill

    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    n_action_steps = cfg["n_action_steps"]
    num_inference_steps = cfg["policy"]["num_inference_steps"]

    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    print("workspace:", workspace)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy.eval().to(device)
    # import ipdb; ipdb.set_trace()
    # policy.num_inference_steps = 2
    # policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    return policy, device, policy.num_inference_steps, n_action_steps


def inference(policy, obs_dict, device):
    """
    模型推理
    :param policy: 加载好的模型
    :param obs_dict: 预处理后的观测数据
    :param device:
    :return: 推理得到的动作
    """
    with torch.no_grad():
        # import ipdb; ipdb.set_trace()
        result = policy.predict_action(obs_dict)
        action = result["action"][0]
        action = action.detach().to("cpu").numpy()
    return action

    # return torch.cat([
    #     obs_dict["robot0_eef_pos"][0][-1],
    #     obs_dict["robot0_eef_rot_axis_angle"][0][-1],
    #     obs_dict["robot0_gripper_width"][0][-1],
    # ]).unsqueeze(0).detach().to("cpu").numpy()


def postprocess_action(action):
    """
    后处理动作数据
    :param action: 推理得到的动作
    :return: 后处理后的动作
    : 返回
    """
    action_pose10d = action[:, :9]
    action_grip = action[:, 9:]
    action_pose = mat_to_pose(pose10d_to_mat(action_pose10d))
    action = np.concatenate([action_pose, action_grip], axis=-1)
    return action


def main(rgb_msg, joint_msg, ee_pose_msg):
    """
    主函数，完成模型加载、推理和结果处理
    :param ckpt_path: 模型检查点文件路径
    """
    global debug_cup_xy_lock_released, init_ee_pose, last_publish_time
    now = time.monotonic()
    if not should_publish_inference(last_publish_time, now, min_publish_interval_sec):
        return None
    last_publish_time = now

    ee_pose = (ee_pose_msg.pose.position,)
    rgb_data = rgb_msg.data
    joint_pos = joint_msg.position

    if len(action_buffer) != 0:
        target_action = action_buffer.popleft()
    else:
        # 输入是世界坐标系下的绝对 pose + 第 0 时刻的 pose +  RGB 图像
        position = ee_pose_msg.pose.position
        orientation = ee_pose_msg.pose.orientation
        ee_pose = np.array(
            [
                position.x,
                position.y,
                position.z,
                orientation.w,
                orientation.x,
                orientation.y,
                orientation.z,
            ]
        )
        if init_ee_pose is None:
            init_ee_pose = ee_pose
            rot_vec = quaternion_to_rotvec(init_ee_pose[3:7])
            init_ee_pose = np.concatenate([init_ee_pose[:3], rot_vec])  # shape: (6,)

        cur_ee_pose = np.array(ee_pose)  # wxyz 格式的四元数表示旋转
        gripper_width = np.array(joint_pos)[-1]

        # 2. 解码图像
        height = rgb_msg.height
        width = rgb_msg.width
        img_np = np.frombuffer(rgb_data, dtype=np.uint8)
        img_np = img_np.reshape((height, width, 3))

        resized_img_array = np.array(
            resize_and_center_crop(Image.fromarray(img_np), target_size=img_size)
        )

        obs_raw_data = {
            "robot0_eef_pos": cur_ee_pose[:3].astype(np.float32),
            "robot0_eef_rot_axis_angle": quaternion_to_rotvec(cur_ee_pose[3:]).astype(
                np.float32
            ),  # 四元数转旋转向量
            "robot0_gripper_width": np.array([gripper_width]),
            "robot0_demo_start_pose": init_ee_pose,  # 四元数转旋转向量
            "camera0_rgb": resized_img_array.astype(np.float32),
        }

        # 更新 buffer
        obs_buffer.update(obs_raw_data)
        # 合并之前的观测
        history_obs = obs_buffer.get_all_frames()
        all_obs = defaultdict(list)
        for d in history_obs:
            for key, value in d.items():
                all_obs[key].append(value)

        # 转换为 numpy 数组
        all_obs = {key: np.array(value) for key, value in all_obs.items()}

        # 预处理数据
        obs_dict = process_data(all_obs)

        # # 两张图片只使用一张
        # obs_dict['camera0_rgb'] = obs_dict['camera0_rgb'][0][-1].unsqueeze(0).unsqueeze(0)

        # 模型推理
        actions = inference(policy, obs_dict, device)

        # 后处理动作数据
        actions = postprocess_action(actions)  # 旋转向量动作，具体语义由 action_pose_mode 决定

        action_steps_to_execute = min(n_action_steps, len(actions))
        for i in range(action_steps_to_execute):
            # 将模型动作转为世界坐标下的绝对发布动作
            abs_pose = model_action_to_absolute_pose(
                cur_pose=cur_ee_pose,
                model_action=actions[i],
                action_pose_mode=action_pose_mode,
            )
            abs_pose = apply_fixed_grasp_orientation(abs_pose, fixed_grasp_quat_wxyz)
            if lock_grasp_roll_pitch:
                abs_pose = apply_fixed_grasp_roll_pitch_keep_yaw(
                    abs_pose, STACK_CUBE_GRASP_QUAT_WXYZ
                )
            abs_pose, debug_cup_xy_lock_released, debug_cup_xy_locked = (
                apply_debug_cup_xy_lock(
                    abs_pose,
                    debug_object_poses=debug_object_poses,
                    enabled=lock_debug_cup_xy_until_close,
                    released=debug_cup_xy_lock_released,
                    z_below=debug_cup_xy_lock_z_below,
                    close_width=debug_cup_xy_lock_close_width,
                )
            )
            raw_abs_pose = np.array(abs_pose, dtype=np.float64, copy=True)
            abs_pose = apply_min_tcp_z(abs_pose, min_tcp_z, action_index=i)
            append_rollout_log_row(
                rollout_log_csv,
                action_index=i,
                cur_ee_pose=cur_ee_pose,
                rel_action=actions[i],
                raw_action=raw_abs_pose,
                published_action=abs_pose,
                min_tcp_z=min_tcp_z,
                action_pose_mode=action_pose_mode,
                debug_cup_xy_locked=debug_cup_xy_locked,
                debug_object_poses=debug_object_poses,
            )

            if i == 0:
                target_action = abs_pose
            else:
                action_buffer.append(abs_pose)
            print(
                build_action_diagnostic_message(
                    action_index=i,
                    cur_ee_pose=cur_ee_pose,
                    rel_action=actions[i],
                    abs_action=abs_pose,
                    action_pose_mode=action_pose_mode,
                    debug_cup_xy_locked=debug_cup_xy_locked,
                    debug_object_poses=debug_object_poses,
                )
            )

    # 将结果返回给仿真环境
    # 这里需要根据实际的仿真环境接口进行修改
    print("actions:", target_action)
    print("actions shape:", target_action.shape)

    # 暂时只返回一帧的
    return target_action


def load_data_from_file(file_path):
    """
    从文件中加载数据
    :param file_path: 数据文件路径
    :return: numpy 数组
    """
    return np.loadtxt(file_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--ckpt_path", type=str, required=True, help="The path to your saved ckpt file"
    )
    args.add_argument("--n_obs_steps", type=int, default=2)
    args.add_argument(
        "--n_action_steps",
        type=int,
        default=4,
        help="How many of predicted actions will be executed before next inference",
    )
    args.add_argument("--img_size", type=int, default=224, help="the image size during inference")
    args.add_argument(
        "--lock_stack_cube_grasp_orientation",
        action="store_true",
        help="Lock published end-effector orientation to the Stack Cube top-grasp quaternion.",
    )
    args.add_argument(
        "--lock_stack_cube_grasp_roll_pitch",
        action="store_true",
        help="Lock Stack Cube top-grasp roll/pitch while preserving model-predicted yaw.",
    )
    args.add_argument(
        "--publish_hz",
        type=float,
        default=20.0,
        help=(
            "Maximum DP action publish frequency. "
            "Use 0 or a negative value to disable throttling."
        ),
    )
    args.add_argument(
        "--debug_cup_pose",
        type=str,
        default=None,
        help="Optional cup pose for DP diagnostics: x,y,z,w,qx,qy,qz.",
    )
    args.add_argument(
        "--debug_plate_pose",
        type=str,
        default=None,
        help="Optional plate pose for DP diagnostics: x,y,z,w,qx,qy,qz.",
    )
    args.add_argument(
        "--min_tcp_z",
        type=float,
        default=None,
        help="Optional minimum TCP world z before publishing DP actions.",
    )
    args.add_argument(
        "--rollout_log_csv",
        type=Path,
        default=None,
        help="Optional CSV path for logging raw/clamped rollout actions and object errors.",
    )
    args.add_argument(
        "--action_pose_mode",
        choices=("relative", "absolute_rotvec"),
        default="relative",
        help=(
            "How to interpret the policy pose action after postprocess_action. "
            "Use absolute_rotvec for Cup/Plate checkpoints trained on world-frame actions."
        ),
    )
    args.add_argument(
        "--lock_debug_cup_xy_until_close",
        action="store_true",
        help=(
            "Debug-only Cup/Plate assist: lock published action XY to --debug_cup_pose "
            "while descending, then release after the first close command."
        ),
    )
    args.add_argument(
        "--debug_cup_xy_lock_z_below",
        type=float,
        default=0.22,
        help="Maximum action z for --lock_debug_cup_xy_until_close.",
    )
    args.add_argument(
        "--debug_cup_xy_lock_close_width",
        type=float,
        default=0.035,
        help="Gripper width that releases --lock_debug_cup_xy_until_close.",
    )
    parsed_args = args.parse_args()
    if (
        parsed_args.lock_stack_cube_grasp_orientation
        and parsed_args.lock_stack_cube_grasp_roll_pitch
    ):
        raise ValueError(
            "`--lock_stack_cube_grasp_orientation` and "
            "`--lock_stack_cube_grasp_roll_pitch` cannot be used together."
        )
    # debug()
    ckpt_path = parsed_args.ckpt_path
    n_obs_steps = parsed_args.n_obs_steps
    img_size = parsed_args.img_size
    min_publish_interval_sec = 0.0 if parsed_args.publish_hz <= 0 else 1.0 / parsed_args.publish_hz
    debug_object_poses = load_debug_object_poses(
        cup_pose_arg=parsed_args.debug_cup_pose,
        plate_pose_arg=parsed_args.debug_plate_pose,
    )
    fixed_grasp_quat_wxyz = (
        STACK_CUBE_GRASP_QUAT_WXYZ if parsed_args.lock_stack_cube_grasp_orientation else None
    )
    lock_grasp_roll_pitch = parsed_args.lock_stack_cube_grasp_roll_pitch
    min_tcp_z = parsed_args.min_tcp_z
    rollout_log_csv = parsed_args.rollout_log_csv
    action_pose_mode = parsed_args.action_pose_mode
    lock_debug_cup_xy_until_close = parsed_args.lock_debug_cup_xy_until_close
    debug_cup_xy_lock_z_below = parsed_args.debug_cup_xy_lock_z_below
    debug_cup_xy_lock_close_width = parsed_args.debug_cup_xy_lock_close_width
    if fixed_grasp_quat_wxyz is not None:
        print("fixed_grasp_quat_wxyz:", fixed_grasp_quat_wxyz)
    if lock_grasp_roll_pitch:
        print("lock_stack_cube_grasp_roll_pitch: True")
    if debug_object_poses:
        print("debug_object_poses:", debug_object_poses)
    if min_tcp_z is not None:
        print("min_tcp_z:", min_tcp_z)
    if rollout_log_csv is not None:
        print("rollout_log_csv:", rollout_log_csv)
    print("action_pose_mode:", action_pose_mode)
    if lock_debug_cup_xy_until_close:
        print(
            "lock_debug_cup_xy_until_close: "
            f"z_below={debug_cup_xy_lock_z_below}, "
            f"close_width={debug_cup_xy_lock_close_width}"
        )
    print("publish_hz:", parsed_args.publish_hz)

    policy, device, num_inference_steps, ckpt_n_action_steps = load_model(ckpt_path)
    n_action_steps = resolve_n_action_steps(parsed_args.n_action_steps, ckpt_n_action_steps)
    if n_action_steps != ckpt_n_action_steps:
        print(
            "n_action_steps override: "
            f"cli={parsed_args.n_action_steps}, checkpoint={ckpt_n_action_steps}, "
            f"effective={n_action_steps}"
        )

    obs_buffer = FrameBuffer(buffer_size=n_obs_steps)  # 存储历史观测数据
    action_buffer = deque(maxlen=n_action_steps)  # 存储输出的动作，因为一次可能输出多个动作
    init_ee_pose = None
    last_publish_time = None
    debug_cup_xy_lock_released = False

    from ros_receiver import RgbJointEePoseActionNode

    node = RgbJointEePoseActionNode(process_fn=main, on_command=clear_runtime_buffers)
