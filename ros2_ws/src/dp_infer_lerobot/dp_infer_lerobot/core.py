"""训练一致的图像处理、逐步 Link7 积分和带采集时间的九点轨迹。"""

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.spatial.transform import Rotation

from dp_infer.common import ActionSequence


def compressed_image_to_rgb(data):
    """与数据转换器一样通过 Pillow 解码 JPEG/PNG 为 HWC uint8 RGB。"""
    try:
        with Image.open(BytesIO(bytes(data))) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ValueError("Expected JPEG/PNG CompressedImage")
            return np.asarray(image.convert("RGB")).copy()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Invalid compressed image: {error}") from error


def letterbox_rgb(rgb):
    """BOX 等比缩放及居中黑边；返回 [3,224,224]、RGB、float32 [0,1]。"""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 1:
        raise ValueError("Expected nonempty HWC uint8 RGB")
    image = Image.fromarray(rgb)
    scale = 224 / max(image.size)
    size = tuple(max(1, round(s * scale)) for s in image.size)
    image = image.resize(size, Image.Resampling.BOX)
    output = Image.new("RGB", (224, 224), (0, 0, 0))
    output.paste(image, ((224 - size[0]) // 2, (224 - size[1]) // 2))
    return np.asarray(output).transpose(2, 0, 1).astype(np.float32) / 255.0


def build_observations(history, start_pose=None):
    """构造完整两帧窗口；位姿仅用于输出锚点，不送入该模型。"""
    if len(history) != 2:
        raise ValueError("Exactly two synchronized observations are required")
    observations = {
        "observation.images.cam0": np.stack([item.image for item in history])[None],
        "observation.state": np.asarray([[[item.gripper] for item in history]], dtype=np.float32),
    }
    for key, shape in (("observation.images.cam0", (1, 2, 3, 224, 224)),
                       ("observation.state", (1, 2, 1))):
        value = observations[key]
        if value.shape != shape or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError(f"Invalid observation {key}")
        observations[key] = value.astype(np.float32, copy=False)
    return observations


def rigid_matrix(matrix):
    """校验米制刚体变换；拒绝反射、退化或非有限矩阵。"""
    matrix = np.asarray(matrix, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6, rtol=0)):
        raise ValueError("Expected a finite rigid transform")
    return matrix


def validate_sequence(sequence):
    """新契约固定九点：八个增量加当前锚点，不改变夹爪预测范围。"""
    for name, shape in (("positions", (9, 3)), ("quaternions", (9, 4)),
                        ("gripper_openness", (9,)), ("time_from_start", (9,))):
        value = np.asarray(getattr(sequence, name), dtype=np.float64).copy()
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid sequence {name}")
        setattr(sequence, name, value)
    norms = np.linalg.norm(sequence.quaternions, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("Degenerate quaternion")
    sequence.quaternions /= norms[:, None]
    for k in range(1, 9):
        if np.dot(sequence.quaternions[k - 1], sequence.quaternions[k]) < 0:
            sequence.quaternions[k] *= -1
    if not np.allclose(sequence.time_from_start, np.arange(9) / 30, atol=1e-9, rtol=0):
        raise ValueError("Expected nine consecutive 30 Hz trajectory points")
    if np.any((sequence.gripper_openness < 0) | (sequence.gripper_openness > 1)):
        raise ValueError("Gripper prediction must be within [0, 1]")
    return sequence


def decode_actions(actions, context):
    """将 [8,10] 的局部 k→k+1 增量右乘累计，夹爪标签对应 k 时刻。"""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (8, 10) or not np.isfinite(actions).all():
        raise ValueError("Expected finite [8,10] actions")
    poses = [rigid_matrix(context.reference_pose).copy()]
    for action in actions:
        first, second = action[3:6], action[6:9]
        if np.linalg.norm(first) < 1e-8:
            raise ValueError("Degenerate first rotation row")
        first = first / np.linalg.norm(first)
        second = second - np.dot(first, second) * first
        if np.linalg.norm(second) < 1e-8:
            raise ValueError("Degenerate second rotation row")
        second = second / np.linalg.norm(second)
        delta = np.eye(4)
        delta[:3, :3] = np.stack((first, second, np.cross(first, second)))
        delta[:3, 3] = action[:3]
        poses.append(poses[-1] @ delta)
    poses = np.stack(poses)
    return validate_sequence(ActionSequence(
        poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat(),
        np.r_[actions[:, 9], actions[-1, 9]], np.arange(9) / 30.0))


def correct_sequence(sequence, correction):
    """全部九点右乘 C，从虚拟 Link7 转回物理 Link7。"""
    sequence = validate_sequence(sequence)
    poses = np.tile(np.eye(4), (9, 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(sequence.quaternions).as_matrix()
    poses[:, :3, 3] = sequence.positions
    poses = poses @ rigid_matrix(correction)
    return validate_sequence(ActionSequence(
        poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat(),
        sequence.gripper_openness, sequence.time_from_start))


def result_is_usable(sequence, context, now_ns):
    """采集时间不后移；末点恰好到期也拒绝发布。"""
    validate_sequence(sequence)
    return context.stamp_ns <= now_ns < context.stamp_ns + round(8e9 / 30)
