"""构造训练一致的 TCP 观测，并把不同坐标契约的模型动作还原为绝对 TCP 目标并整理为带观测时间的 8～15 点序列。

坐标约定：
- 观测位姿 ``Observation.pose`` 为 ``base_link → TCP``。共享运行层用 ``FK(q) @ inv(C)`` 生成观测，
  本后端令 ``C = inverse(T_Link7_TCP)``，因此观测即 ``FK(q) @ T_Link7_TCP``，米制 4×4。
- 模型 state/action 原始布局 ``[x, y, z, rx, ry, rz, gripper]``：位置米、轴角弧度、归一化夹爪。
  旧模型内部使用相对当前帧的位姿；新模型输入和输出均相对固定回合起点 TCP。
  新模型输出左乘冻结的 T_base_start 后恢复为 base_link → TCP。
- :func:`decode_actions` 输出 ``base_link → TCP`` 序列；发布前由 :func:`correct_sequence` 右乘
  ``C`` 得到 ``base_link → Link7``，四元数 xyzw，第 k 个点对应观测时刻 + k/30 秒。
- 训练执行 8 个动作；部署可多发布 horizon 中后续的预测点（最多 15 个，约 467 ms），供控制器前瞻
  抵消 RM75 低跟随滞后。过期判定始终以第 8 点（观测 + 7/30 秒）为准，不因多发布而放宽。
"""

import numpy as np
from scipy.spatial.transform import Rotation

from dp_infer.common import ActionSequence

from .config import IMAGE_KEY, RAW_DIM, STATE_KEY

FREQUENCY_HZ = 30  # 训练数据与输出动作的固定频率。
ACTION_STEPS = 8  # 训练每次执行的动作数（n_action_steps），也是过期判定的基准。
MAX_ACTION_STEPS = 15  # horizon 16 - (n_obs_steps 2 - 1)：可发布的最多预测点数。
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


def build_observations(history, start_pose=None, pose_frame="current_tcp", gripper_threshold=0.75):
    """把两帧同步观测转换为模型原始输入。

    Args:
        history: 两个 ``dp_infer.common.Observation``，时间递增；pose 为 base→TCP。
        start_pose: 固定回合起点 base→TCP，米制 4×4；旧模型忽略。
        pose_frame: current_tcp 或 episode_start_tcp。
        gripper_threshold: 新模型观测夹爪二值化阈值。

    Returns:
        ``{IMAGE_KEY: [1,2,3,224,224], STATE_KEY: [1,2,7]}`` float32。

    Raises:
        ValueError: 帧数、形状或数值不合法。
    """
    if len(history) != 2:
        raise ValueError("Exactly two synchronized observations are required")
    images = np.stack([np.asarray(item.image, dtype=np.float32) for item in history])[None]
    if pose_frame not in ("current_tcp", "episode_start_tcp"):
        raise ValueError("Unsupported pose frame")
    if not np.isfinite(gripper_threshold) or not 0 <= gripper_threshold <= 1:
        raise ValueError("Invalid gripper threshold")
    grippers = np.asarray([item.gripper for item in history], dtype=np.float64)
    if not np.isfinite(grippers).all() or np.any((grippers < 0) | (grippers > 1)):
        raise ValueError("Invalid gripper state")
    transform = np.eye(4)
    if pose_frame == "episode_start_tcp":
        transform = np.linalg.inv(rigid_matrix(start_pose))
        grippers = (grippers >= gripper_threshold).astype(np.float64)
    state = np.stack([np.r_[pose_to_raw(transform @ item.pose), gripper]
                      for item, gripper in zip(history, grippers)])[None]
    if images.shape != (1, 2, *IMAGE_SHAPE) or not np.isfinite(images).all() or np.any(
            (images < 0) | (images > 1)):
        raise ValueError(f"Invalid observation {IMAGE_KEY}")
    if state.shape != (1, 2, RAW_DIM) or not np.isfinite(state).all() or np.any(
            (state[..., 6] < 0) | (state[..., 6] > 1)):
        raise ValueError(f"Invalid observation {STATE_KEY}")
    return {IMAGE_KEY: images, STATE_KEY: state.astype(np.float32)}


def validate_sequence(sequence, count=None):
    """校验并规范化 ``count`` 点序列：四元数归一且相邻同号，时间为 k/30，夹爪截断到 [0,1]。

    Args:
        count: 期望点数；None 时取序列自身长度，须在 [ACTION_STEPS, MAX_ACTION_STEPS] 内。

    Raises:
        ValueError: 点数、形状、数值、四元数或时间不合法。
    """
    if count is None:
        count = len(np.asarray(sequence.positions))
    if not ACTION_STEPS <= count <= MAX_ACTION_STEPS:
        raise ValueError(f"Sequence must contain {ACTION_STEPS}..{MAX_ACTION_STEPS} points")
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


def decode_actions(actions, context=None, pose_frame="current_tcp"):
    """把 ``[N,7]``（N 为 8～15）绝对 TCP 动作转换为 base→TCP 目标序列。

    Args:
        actions: 后处理后的原始布局动作，第 0 个对应观测时刻。
        context: 冻结的回合起点；TCP→Link7 换算由共享运行层完成。
        pose_frame: 动作坐标系契约，固定起点模型需先恢复 base→TCP。
    """
    actions = np.asarray(actions, dtype=np.float64)
    if (actions.ndim != 2 or not ACTION_STEPS <= actions.shape[0] <= MAX_ACTION_STEPS
            or actions.shape[1] != RAW_DIM or not np.isfinite(actions).all()):
        raise ValueError(f"Expected finite [{ACTION_STEPS}..{MAX_ACTION_STEPS},{RAW_DIM}] actions")
    if pose_frame not in ("current_tcp", "episode_start_tcp"):
        raise ValueError("Unsupported pose frame")
    poses = raw_to_pose(actions)
    if pose_frame == "episode_start_tcp":
        poses = rigid_matrix(context.reference_pose) @ poses
    return validate_sequence(ActionSequence(
        poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat(),
        actions[:, 6], np.arange(len(actions)) / FREQUENCY_HZ))


def correct_sequence(sequence, correction):
    """全部点右乘 ``C = inverse(T_Link7_TCP)``，把 base→TCP 目标换算为 base→Link7。"""
    sequence = validate_sequence(sequence)
    poses = np.tile(np.eye(4), (len(sequence.positions), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(sequence.quaternions).as_matrix()
    poses[:, :3, 3] = sequence.positions
    poses = poses @ rigid_matrix(correction)
    return validate_sequence(ActionSequence(
        poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat(),
        sequence.gripper_openness, sequence.time_from_start))


def result_is_usable(sequence, context, now_ns):
    """采集时间不后移；第 8 点（观测 + 7/30 秒）恰好到期也拒绝发布，与发布点数无关。"""
    validate_sequence(sequence)
    return context.stamp_ns <= now_ns < context.stamp_ns + LAST_POINT_NS
