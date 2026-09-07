"""构造训练一致的 UMI 观测，执行推理并将相对动作还原为绝对目标序列。"""

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from umi.common.pose_util import mat_to_pose10d, pose10d_to_mat

# 模型训练使用的七轴顺序，ROS 消息可采用任意排列。
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
# 五键观测的单帧形状。
OBS_SHAPES = {
    "camera0_rgb": (3, 224, 224),
    "robot0_eef_pos": (3,),
    "robot0_eef_rot_axis_angle": (6,),
    "robot0_gripper_width": (1,),
    "robot0_eef_rot_axis_angle_wrt_start": (6,),
}


def ordered_joints(names, positions):
    """按训练关节顺序提取弧度位置；缺失、重名或非有限值时抛出 ValueError。"""
    if len(names) != len(positions) or len(set(names)) != len(names):
        raise ValueError("JointState names/positions are incomplete or duplicated")
    if not set(JOINT_NAMES).issubset(names):
        raise ValueError("JointState must contain joint1 through joint7")
    values = np.asarray([positions[names.index(name)] for name in JOINT_NAMES], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Joint positions must be finite radians")
    return values


def image_to_rgb(data, height, width, step, encoding):
    """解码带行填充的 rgb8/bgr8 数据，返回拥有独立存储的 HWC uint8 RGB。"""
    if encoding not in ("rgb8", "bgr8") or min(height, width) <= 0 or step < width * 3:
        raise ValueError("Expected rgb8/bgr8 image with a valid row stride")
    values = np.frombuffer(data, dtype=np.uint8)
    if values.size != height * step:
        raise ValueError("Image data size does not match height * step")
    rgb = values.reshape(height, step)[:, :width * 3].reshape(height, width, 3)
    return (rgb[..., ::-1] if encoding == "bgr8" else rgb).copy()


def letterbox_rgb(rgb, size=224):
    """保持训练预处理的缩放、居中黑边和 RGB 语义，返回 CHW [0,1] 浮点图像。"""
    height, width = rgb.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA)
    output = np.zeros((size, size, 3), dtype=np.uint8)  # 与训练转换一致的黑色背景。
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return np.moveaxis(output, -1, 0).astype(np.float32) / 255.0


@dataclass(frozen=True)
class Observation:
    """一个已同步观测：ROS 纳秒时间、CHW RGB、base→Link7 变换与归一化夹爪。"""

    stamp_ns: int  # 图像采集时间。
    image: np.ndarray  # [3,224,224] 浮点 RGB。
    pose: np.ndarray  # [4,4] 米制绝对位姿。
    gripper: float  # 训练使用的归一化开度。


@dataclass(frozen=True)
class InferenceContext:
    """一次推理固定的参考状态；传入后处理器，不随实时订阅更新。"""

    reference_pose: np.ndarray  # 本次最新观测的 base→Link7 变换。
    stamp_ns: int  # 本次最新图像采集时间。
    episode_id: int  # 重置时递增的 episode 标识。
    sequence_id: int  # 每次提交推理时递增的编号。


@dataclass
class ActionSequence:
    """完整目标序列，位置单位为米，四元数顺序为 xyzw，时间单位为秒。"""

    positions: np.ndarray  # [16,3]，base_link 下的 Link7 目标位置。
    quaternions: np.ndarray  # [16,4]，base_link 下的 Link7 目标旋转。
    gripper_openness: np.ndarray  # [16]，归一化夹爪目标。
    time_from_start: np.ndarray  # [16]，相对观测时间的预测偏移。


def build_observations(history, start_pose):
    """将两帧同步状态转换成与 UmiDataset 相同的五键、带 batch 维度的观测。"""
    if len(history) != 2:
        raise ValueError("Exactly two observations are required")
    poses = np.stack([item.pose for item in history])
    relative = mat_to_pose10d(convert_pose_mat_rep(poses, poses[-1], "relative"))
    start_relative = mat_to_pose10d(convert_pose_mat_rep(poses, start_pose, "relative"))
    observations = {
        "camera0_rgb": np.stack([item.image for item in history]),
        "robot0_eef_pos": relative[:, :3],
        "robot0_eef_rot_axis_angle": relative[:, 3:],
        "robot0_gripper_width": np.array([[item.gripper] for item in history]),
        "robot0_eef_rot_axis_angle_wrt_start": start_relative[:, 3:],
    }
    return {key: value[None].astype(np.float32) for key, value in observations.items()}


def validate_sequence(sequence):
    """校验处理器输出并规范化四元数和夹爪；退化旋转、非法时间或形状抛出异常。"""
    for name, shape in (("positions", (16, 3)), ("quaternions", (16, 4)),
                        ("gripper_openness", (16,)), ("time_from_start", (16,))):
        values = np.asarray(getattr(sequence, name), dtype=np.float64).copy()
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid sequence field {name}: expected finite {shape}")
        setattr(sequence, name, values)
    norms = np.linalg.norm(sequence.quaternions, axis=-1)
    if np.any(norms < 1e-8):
        raise ValueError("Degenerate quaternion")
    sequence.quaternions /= norms[:, None]
    for index in range(1, 16):
        if np.dot(sequence.quaternions[index - 1], sequence.quaternions[index]) < 0:
            sequence.quaternions[index] *= -1
    if not np.allclose(sequence.time_from_start, np.arange(16) / 30, atol=1e-9, rtol=0):
        raise ValueError("Action times must retain the trained 30 Hz horizon")
    sequence.gripper_openness = np.clip(sequence.gripper_openness, 0, 1)
    return sequence


def decode_actions(actions, context):
    """把 [16,10] 相对动作左乘冻结观测位姿，输出完整绝对目标；拒绝退化 6D 旋转。"""
    values = np.asarray(actions, dtype=np.float64)
    if values.shape != (16, 10) or not np.isfinite(values).all():
        raise ValueError("Expected finite [16,10] actions")
    first, second = values[:, 3:6], values[:, 6:9]
    first_norm = np.linalg.norm(first, axis=-1)
    if np.any(first_norm < 1e-8):
        raise ValueError("Degenerate first rotation axis")
    unit = first / first_norm[:, None]
    orthogonal = second - np.sum(second * unit, axis=-1, keepdims=True) * unit
    if np.any(np.linalg.norm(orthogonal, axis=-1) < 1e-8):
        raise ValueError("Degenerate second rotation axis")
    matrices = context.reference_pose @ pose10d_to_mat(values[:, :9])
    return validate_sequence(ActionSequence(
        matrices[:, :3, 3], Rotation.from_matrix(matrices[:, :3, :3]).as_quat(),
        values[:, 9], np.arange(16) / 30.0,
    ))


def load_processors(specifications):
    """加载 module:factory 列表；每个无参工厂返回具有 process(sequence, context) 的对象。"""
    processors = []  # 按配置顺序执行的后处理器。
    for specification in specifications:
        module, attribute = specification.split(":", 1)
        processor = getattr(importlib.import_module(module), attribute)()
        if not callable(getattr(processor, "process", None)):
            raise ValueError(f"Processor {specification} must implement process")
        processors.append(processor)
    return processors


def validate_contract(cfg):
    """限制为当前 RM75 五键/10D/30Hz 训练契约，防止误用其他 checkpoint。"""
    contract = cfg.task.get("contract", {})
    expected = {"base_frame": "base_link", "end_frame": "Link7", "tool_offset": "identity",
                "gripper_representation": "normalized_0_1",
                "action_reference": "current_observed_end_frame", "frequency_hz": 30}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint does not match the RM75 Link7 deployment contract")
    shape_meta = cfg.shape_meta
    urdf_sha256 = contract.get("urdf_sha256")
    if (not isinstance(urdf_sha256, str) or len(urdf_sha256) != 64
            or any(character not in "0123456789abcdef" for character in urdf_sha256.lower())):
        raise ValueError("Checkpoint does not contain a valid training URDF SHA-256")
    if set(shape_meta.obs) != set(OBS_SHAPES):
        raise ValueError("Checkpoint must contain exactly the five UMI observations")
    for key, shape in OBS_SHAPES.items():
        meta = shape_meta.obs[key]
        if tuple(meta.shape) != shape or meta.horizon != 2 or meta.down_sample_steps != 1:
            raise ValueError(f"Unsupported observation contract: {key}")
        if meta.get("latency_steps", 0) != 0:
            raise ValueError("Observation latency must match the aligned training dataset")
    if (list(shape_meta.action.shape) != [10] or shape_meta.action.horizon != 16
            or shape_meta.action.down_sample_steps != 1 or shape_meta.action.latency_steps != 0):
        raise ValueError("Checkpoint must predict 16 consecutive 10D actions")
    if any(cfg.task.pose_repr[key] != "relative" for key in ("obs_pose_repr", "action_pose_repr")):
        raise ValueError("Both pose representations must be relative")


def validate_urdf(urdf_path, cfg):
    """校验部署 URDF 的原始文件摘要与 checkpoint 中的训练摘要完全一致。"""
    path = Path(urdf_path)
    if not path.is_file():
        raise ValueError(f"URDF must name an existing file: {path}")
    expected = cfg.task.contract.urdf_sha256.lower()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"Deployment URDF SHA-256 does not match checkpoint: expected {expected}, got {actual}"
        )


class PolicyEngine:
    """单线程使用的策略实例；ROS 层负责调度，模型依赖仅在创建时导入。"""

    def __init__(self, checkpoint, device="cuda:0", processors=()):
        """加载现有评估入口使用的 EMA/model 权重并验证契约；异常阻止节点启动。"""
        from evaluate_vr_umi import load_policy

        self.policy, self.cfg = load_policy(checkpoint, device)
        validate_contract(self.cfg)
        self.device = device  # 推理张量所在设备。
        self.processors = list(processors)  # 已构造的后处理扩展。

    def predict(self, observations, context):
        """执行一次推理与后处理，返回绝对动作序列；异常由 ROS 层记录并丢弃。"""
        import torch

        inputs = {key: torch.from_numpy(value).to(self.device) for key, value in observations.items()}
        with torch.inference_mode():
            prediction = self.policy.predict_action(inputs)["action_pred"].cpu().numpy()
        if prediction.shape != (1, 16, 10):
            raise ValueError(f"Unexpected policy shape: {prediction.shape}")
        sequence = decode_actions(prediction[0], context)
        for processor in self.processors:
            sequence = validate_sequence(processor.process(sequence, context))
        return sequence
