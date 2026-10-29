"""构造训练一致的 TCP 观测，并把模型输出的绝对 TCP 动作整理为带观测时间的八点序列。

坐标约定：
- 观测位姿 ``Observation.pose`` 为 ``base_link → TCP``。共享运行层用 ``FK(q) @ inv(C)`` 生成观测，
  本后端令 ``C = inverse(T_Link7_TCP)``，因此观测即 ``FK(q) @ T_Link7_TCP``，米制 4×4。
- 模型 state/action 原始布局 ``[x, y, z, rx, ry, rz, gripper]``：位置米、轴角弧度、归一化夹爪。
  模型内部只使用相对当前帧的位姿，因此 base 坐标系的选取不影响结果。
- :func:`decode_actions` 输出 ``base_link → TCP`` 序列；发布前由 :func:`correct_sequence` 右乘
  ``C`` 得到 ``base_link → Link7``，四元数 xyzw，第 k 个点对应观测时刻 + k/30 秒。
"""

import numpy as np
from scipy.spatial.transform import Rotation

from dp_infer.common import ActionSequence

from .config import IMAGE_KEY, RAW_DIM, STATE_KEY

FREQUENCY_HZ = 30  # 训练数据与输出动作的固定频率。
ACTION_STEPS = 8  # 每次推理输出的动作数（n_action_steps）。
IMAGE_SHAPE = (3, 224, 224)  # letterbox 后的 CHW 图像形状。
LAST_POINT_NS = round((ACTION_STEPS - 1) * 1e9 / FREQUENCY_HZ)  # 末点相对观测时刻的纳秒偏移。


def rigid_matrix(matrix):
    """校验米制刚体变换；拒绝反射、退化或非有限矩阵。"""
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6, rtol=0)):
        raise ValueError("Expected a finite rigid transform")
    return matrix


def pose_to_raw(pose):
    """``[4,4]`` 位姿 → ``[x,y,z,rx,ry,rz]``（米、轴角弧度）。"""
    pose = np.asarray(pose, dtype=np.float64)
    return np.concatenate([pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_rotvec()])


def raw_to_pose(raw):
    """``[..., 6+]`` 原始布局 → ``[..., 4, 4]`` 位姿，仅使用前 6 维。"""
    raw = np.asarray(raw, dtype=np.float64)
    poses = np.tile(np.eye(4), (*raw.shape[:-1], 1, 1))
    poses[..., :3, :3] = Rotation.from_rotvec(raw[..., 3:6].reshape(-1, 3)).as_matrix().reshape(
        *raw.shape[:-1], 3, 3)
    poses[..., :3, 3] = raw[..., :3]
    return poses


def build_observations(history, start_pose=None):
    """把两帧同步观测转换为模型原始输入。

    Args:
        history: 两个 ``dp_infer.common.Observation``，时间递增；pose 为 base→TCP。
        start_pose: 兼容共享运行层调用约定，本模型不使用。

    Returns:
        ``{IMAGE_KEY: [1,2,3,224,224], STATE_KEY: [1,2,7]}`` float32。

    Raises:
        ValueError: 帧数、形状或数值不合法。
    """
    if len(history) != 2:
        raise ValueError("Exactly two synchronized observations are required")
    images = np.stack([np.asarray(item.image, dtype=np.float32) for item in history])[None]
    state = np.stack([np.r_[pose_to_raw(item.pose), item.gripper] for item in history])[None]
    if images.shape != (1, 2, *IMAGE_SHAPE) or not np.isfinite(images).all() or np.any(
            (images < 0) | (images > 1)):
        raise ValueError(f"Invalid observation {IMAGE_KEY}")
    if state.shape != (1, 2, RAW_DIM) or not np.isfinite(state).all() or np.any(
            (state[..., 6] < 0) | (state[..., 6] > 1)):
        raise ValueError(f"Invalid observation {STATE_KEY}")
    return {IMAGE_KEY: images, STATE_KEY: state.astype(np.float32)}


def validate_sequence(sequence, count=ACTION_STEPS):
    """校验并规范化 ``count`` 点序列：四元数归一且相邻同号，时间为 k/30，夹爪截断到 [0,1]。

    Raises:
        ValueError: 形状、数值、四元数或时间不合法。
    """
    for name, shape in (("positions", (count, 3)), ("quaternions", (count, 4)),
                        ("gripper_openness", (count,)), ("time_from_start", (count,))):
        values = np.asarray(getattr(sequence, name), dtype=np.float64).copy()
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid sequence field {name}: expected finite {shape}")
        setattr(sequence, name, values)
    norms = np.linalg.norm(sequence.quaternions, axis=-1)
    if np.any(norms < 1e-8):
        raise ValueError("Degenerate quaternion")
    sequence.quaternions /= norms[:, None]
    for index in range(1, count):
        if np.dot(sequence.quaternions[index - 1], sequence.quaternions[index]) < 0:
            sequence.quaternions[index] *= -1
    if not np.allclose(sequence.time_from_start, np.arange(count) / FREQUENCY_HZ, atol=1e-9, rtol=0):
        raise ValueError("Action times must retain the trained 30 Hz horizon")
    sequence.gripper_openness = np.clip(sequence.gripper_openness, 0, 1)
    return sequence


def decode_actions(actions, context=None):
    """把 ``[8,7]`` 绝对 TCP 动作转换为 base→TCP 目标序列。

    Args:
        actions: 后处理后的原始布局动作，第 0 个对应观测时刻。
        context: 兼容引擎调用约定；TCP→Link7 换算由 :func:`correct_sequence` 完成。
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (ACTION_STEPS, RAW_DIM) or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite [{ACTION_STEPS},{RAW_DIM}] actions")
    poses = raw_to_pose(actions)
    return validate_sequence(ActionSequence(
        poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat(),
        actions[:, 6], np.arange(ACTION_STEPS) / FREQUENCY_HZ))


def correct_sequence(sequence, correction):
    """全部八点右乘 ``C = inverse(T_Link7_TCP)``，把 base→TCP 目标换算为 base→Link7。"""
    sequence = validate_sequence(sequence)
    poses = np.tile(np.eye(4), (ACTION_STEPS, 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(sequence.quaternions).as_matrix()
    poses[:, :3, 3] = sequence.positions
    poses = poses @ rigid_matrix(correction)
    return validate_sequence(ActionSequence(
        poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat(),
        sequence.gripper_openness, sequence.time_from_start))


def result_is_usable(sequence, context, now_ns):
    """采集时间不后移；末点（观测 + 7/30 秒）恰好到期也拒绝发布。"""
    validate_sequence(sequence)
    return context.stamp_ns <= now_ns < context.stamp_ns + LAST_POINT_NS
