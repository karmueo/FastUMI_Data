"""SE(3) 相对位姿与 rot6d 转换（torch FP32）。

逐字摘自 LeRobot fork 6168a52 ``lerobot/processor/relative_pose_processor.py``（Apache-2.0），
原始布局位姿组为 ``[x, y, z, rx, ry, rz]``（米、轴角弧度），模型布局为 ``[x, y, z] + rot6d``
（旋转矩阵前两行）。位姿组以外的维度（夹爪）原样保留。
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

RAW_POSE_DIM = 6  # 原始布局位姿组宽度（位置 3 + 轴角 3）。
MODEL_POSE_DIM = 9  # 模型布局位姿组宽度（位置 3 + rot6d 6）。


def _skew(vector: Tensor) -> Tensor:
    """向量 (..., 3) 的反对称矩阵 (..., 3, 3)。"""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """轴角 (..., 3) 转旋转矩阵 (..., 3, 3)，采用 Rodrigues 公式。

    直接使用未归一化的反对称矩阵 ``K = [r]x``：``R = I + A K + B K^2``，其中
    ``A = sin(θ)/θ``、``B = (1 - cos θ)/θ²``。θ 接近 0 时用泰勒展开，避免除零。
    """
    angle_sq = (axis_angle * axis_angle).sum(-1, keepdim=True)
    angle = angle_sq.sqrt()
    small = angle < 1e-4
    safe_angle = torch.where(small, torch.ones_like(angle), angle)
    coef_a = torch.where(small, 1.0 - angle_sq / 6.0, torch.sin(safe_angle) / safe_angle)
    coef_b = torch.where(small, 0.5 - angle_sq / 24.0, (1.0 - torch.cos(safe_angle)) / safe_angle**2)
    skew = _skew(axis_angle)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand_as(skew)
    return eye + coef_a[..., None] * skew + coef_b[..., None] * (skew @ skew)


def _sqrt_positive_part(value: Tensor) -> Tensor:
    """sqrt(max(0, x))，且在 0 处梯度为 0。"""
    result = torch.zeros_like(value)
    positive = value > 0
    result[positive] = torch.sqrt(value[positive])
    return result


def matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """旋转矩阵 (..., 3, 3) 转四元数 (..., 4)，顺序为 wxyz，且保证 w >= 0。

    从四个候选解中选取分母最大的一个，在旋转角接近 π 时依旧数值稳定。
    """
    batch_shape = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix.reshape(*batch_shape, 9).unbind(-1)
    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(floor))
    best = F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5
    quaternion = candidates[best].reshape(*batch_shape, 4)
    # 四元数 q 与 -q 表示同一旋转，统一取 w >= 0，使转出的轴角满足 ‖r‖ <= π
    return torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)


def quaternion_to_axis_angle(quaternion: Tensor) -> Tensor:
    """四元数 (..., 4，wxyz) 转轴角 (..., 3)。"""
    vector = quaternion[..., 1:]
    sin_half = vector.norm(dim=-1, keepdim=True)
    half_angle = torch.atan2(sin_half, quaternion[..., :1])
    angle = 2.0 * half_angle
    small = angle.abs() < 1e-6
    safe_angle = torch.where(small, torch.ones_like(angle), angle)
    # rotvec = vector * angle / sin(angle / 2)；小角度下 sin(θ/2)/θ ≈ 1/2 - θ²/48
    sin_half_over_angle = torch.where(small, 0.5 - angle**2 / 48.0, torch.sin(half_angle) / safe_angle)
    return vector / sin_half_over_angle


def matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    """旋转矩阵 (..., 3, 3) 转规范轴角 (..., 3)，‖r‖ <= π。"""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """旋转矩阵 (..., 3, 3) 转 rot6d (..., 6)：取前两行，与 UMI 的 ``mat_to_rot6d`` 一致。"""
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    """rot6d (..., 6) 转旋转矩阵 (..., 3, 3)，用 Gram-Schmidt 正交化，结果按行堆叠。

    模型输出的 6 维不一定正交，正交化保证解码结果是合法旋转。
    """
    a1, a2 = rotation_6d[..., :3], rotation_6d[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def validate_pose_starts(starts: Sequence[int], raw_dim: int, name: str = "pose_starts") -> None:
    """检查位姿组起始下标：升序、不重叠，且每组 6 维都落在 ``raw_dim`` 之内。"""
    previous_end = 0
    for start in starts:
        if start < previous_end:
            raise ValueError(
                f"`{name}` must be ascending and non-overlapping 6-wide groups, got {list(starts)}"
            )
        if start + RAW_POSE_DIM > raw_dim:
            raise ValueError(
                f"`{name}` group starting at {start} needs {RAW_POSE_DIM} dims but the feature has {raw_dim}"
            )
        previous_end = start + RAW_POSE_DIM


def relative_pose_model_dim(raw_dim: int, num_groups: int) -> int:
    """原始维度对应的模型维度：每个位姿组从 6 维扩展为 9 维。"""
    return raw_dim + (MODEL_POSE_DIM - RAW_POSE_DIM) * num_groups


def _pose_from_raw(raw: Tensor, start: int) -> tuple[Tensor, Tensor]:
    """从原始布局取出某个位姿组，返回 (位置 (..., 3), 旋转矩阵 (..., 3, 3))。"""
    position = raw[..., start : start + 3]
    rotation = axis_angle_to_matrix(raw[..., start + 3 : start + RAW_POSE_DIM])
    return position, rotation


def _broadcast_reference(reference: Tensor, values: Tensor) -> Tensor:
    """参考位姿为 (B, D)，待转换值可能带时间维 (B, T, D)，此时在时间维上广播。"""
    if values.ndim == reference.ndim + 1:
        return reference.unsqueeze(-2)
    return reference


def to_relative_pose(
    values: Tensor,
    reference: Tensor,
    value_starts: Sequence[int],
    reference_starts: Sequence[int],
) -> Tensor:
    """把原始布局的位姿转换为相对参考位姿的模型布局。

    Args:
        values: (B, D_raw) 或 (B, T, D_raw)，原始布局（位置 + 轴角）。
        reference: (B, D_ref)，参考（当前帧）状态。
        value_starts: ``values`` 中各位姿组的起始下标。
        reference_starts: ``reference`` 中与之一一对应的位姿组起始下标。

    Returns:
        (B, D_model) 或 (B, T, D_model)。位姿组变为 ``[p_rel, rot6d(R_rel)]``，其中
        ``R_rel = R_ref^T R``、``p_rel = R_ref^T (p - p_ref)``，即 ``inv(T_ref) @ T``；其余维度原样保留。
    """
    if len(value_starts) != len(reference_starts):
        raise ValueError("value_starts and reference_starts must contain the same number of pose groups")
    reference = _broadcast_reference(reference.to(device=values.device, dtype=values.dtype), values)
    segments: list[Tensor] = []
    cursor = 0
    for value_start, reference_start in zip(value_starts, reference_starts, strict=True):
        segments.append(values[..., cursor:value_start])
        position, rotation = _pose_from_raw(values, value_start)
        ref_position, ref_rotation = _pose_from_raw(reference, reference_start)
        ref_rotation_t = ref_rotation.transpose(-1, -2)
        rel_rotation = ref_rotation_t @ rotation
        rel_position = (ref_rotation_t @ (position - ref_position).unsqueeze(-1)).squeeze(-1)
        segments.append(rel_position)
        segments.append(matrix_to_rotation_6d(rel_rotation))
        cursor = value_start + RAW_POSE_DIM
    segments.append(values[..., cursor:])
    return torch.cat(segments, dim=-1)


def to_absolute_pose(
    values: Tensor,
    reference: Tensor,
    value_starts: Sequence[int],
    reference_starts: Sequence[int],
) -> Tensor:
    """``to_relative_pose`` 的逆变换：把模型布局的相对位姿还原成原始布局的绝对位姿。

    Args:
        values: (B, D_model) 或 (B, T, D_model)。
        reference: (B, D_ref)，生成这些相对量时所用的参考状态（原始布局）。
        value_starts: 位姿组在 **原始** 布局中的起始下标。
        reference_starts: ``reference`` 中对应的位姿组起始下标。

    Returns:
        (B, D_raw) 或 (B, T, D_raw)，位姿组为 ``T_ref @ T_rel`` 的位置 + 规范轴角。
    """
    if len(value_starts) != len(reference_starts):
        raise ValueError("value_starts and reference_starts must contain the same number of pose groups")
    reference = _broadcast_reference(reference.to(device=values.device, dtype=values.dtype), values)
    segments: list[Tensor] = []
    model_cursor = 0
    for group_index, (value_start, reference_start) in enumerate(
        zip(value_starts, reference_starts, strict=True)
    ):
        # 原始布局中前面每个位姿组比模型布局少 3 维，据此换算模型布局中的起始位置
        model_start = value_start + (MODEL_POSE_DIM - RAW_POSE_DIM) * group_index
        segments.append(values[..., model_cursor:model_start])
        rel_position = values[..., model_start : model_start + 3]
        rel_rotation = rotation_6d_to_matrix(values[..., model_start + 3 : model_start + MODEL_POSE_DIM])
        ref_position, ref_rotation = _pose_from_raw(reference, reference_start)
        rotation = ref_rotation @ rel_rotation
        position = (ref_rotation @ rel_position.unsqueeze(-1)).squeeze(-1) + ref_position
        segments.append(position)
        segments.append(matrix_to_axis_angle(rotation))
        model_cursor = model_start + MODEL_POSE_DIM
    segments.append(values[..., model_cursor:])
    return torch.cat(segments, dim=-1)


def set_frame_to_identity(relative: Tensor, frame_index: int, value_starts: Sequence[int]) -> Tensor:
    """把模型布局历史 (..., T, D_model) 中第 ``frame_index`` 帧的位姿组写成精确的单位位姿。

    当前帧相对自身理论上恒为单位变换，但 ``R^T R`` 的浮点误差会留下 1e-7（float32）量级的
    抖动。若统计量中该维度近乎常量（如 ``n_obs_steps=1``），归一化会把这点误差放大到 1e8 以上。
    统一写成精确值后，训练、推理与统计量在这些维度上完全一致。
    """
    out = relative.clone()
    frame = out[..., frame_index, :]
    for group_index, value_start in enumerate(value_starts):
        model_start = value_start + (MODEL_POSE_DIM - RAW_POSE_DIM) * group_index
        frame[..., model_start : model_start + 3] = 0.0
        frame[..., model_start + 3 : model_start + MODEL_POSE_DIM] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=out.dtype, device=out.device
        )
    return out
