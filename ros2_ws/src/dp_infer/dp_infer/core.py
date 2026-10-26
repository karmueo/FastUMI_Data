"""构造训练一致的 UMI 观测，执行推理并将相对动作还原为绝对目标序列。"""

import hashlib
import importlib
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from umi.common.pose_util import mat_to_pose10d, pose10d_to_mat

from dp_infer.common import (
    ActionSequence, InferenceContext, JOINT_NAMES, Observation,
    compressed_image_to_rgb, image_to_rgb, load_link7_correction, ordered_joints,
)

# 五键观测的单帧形状。
OBS_SHAPES = {
    "camera0_rgb": (3, 224, 224),
    "robot0_eef_pos": (3,),
    "robot0_eef_rot_axis_angle": (6,),
    "robot0_gripper_width": (1,),
    "robot0_eef_rot_axis_angle_wrt_start": (6,),
}


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


def correct_sequence(sequence, correction):
    """把策略虚拟 Link7 目标右乘修正 C，得到真实 Link7 目标；单位阵时原样返回。"""
    if np.array_equal(correction, np.eye(4)):
        return sequence
    matrices = np.tile(np.eye(4), (len(sequence.positions), 1, 1))
    matrices[:, :3, :3] = Rotation.from_quat(sequence.quaternions).as_matrix()
    matrices[:, :3, 3] = sequence.positions
    matrices = matrices @ correction
    return validate_sequence(ActionSequence(
        matrices[:, :3, 3], Rotation.from_matrix(matrices[:, :3, :3]).as_quat(),
        sequence.gripper_openness, sequence.time_from_start,
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


def validate_contract(cfg, expected_urdf_sha256=None):
    """限制为当前 RM75 五键/10D/30Hz 契约，并返回可信的训练 URDF 摘要。"""
    contract = cfg.task.get("contract", {})
    expected = {"base_frame": "base_link", "end_frame": "Link7", "tool_offset": "identity",
                "gripper_representation": "normalized_0_1",
                "action_reference": "current_observed_end_frame", "frequency_hz": 30}
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint does not match the RM75 Link7 deployment contract")
    shape_meta = cfg.shape_meta
    urdf_sha256 = contract.get("urdf_sha256") or expected_urdf_sha256
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
    return urdf_sha256.lower()


def validate_urdf(urdf_path, cfg, expected_urdf_sha256=None):
    """校验部署 URDF 与 checkpoint 内嵌或显式训练摘要完全一致。"""
    path = Path(urdf_path)
    if not path.is_file():
        raise ValueError(f"URDF must name an existing file: {path}")
    configured = getattr(cfg.task.contract, "urdf_sha256", None)  # 新 checkpoint 内嵌的训练摘要。
    expected = (configured or expected_urdf_sha256 or "").lower()
    if (len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)):
        raise ValueError("A valid training URDF SHA-256 is required")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"Deployment URDF SHA-256 does not match checkpoint: expected {expected}, got {actual}"
        )


class PolicyEngine:
    """单线程使用的策略实例；ROS 层负责调度，模型依赖仅在创建时导入。"""

    def __init__(self, checkpoint, device="cuda:0", processors=(),
                 expected_urdf_sha256=None, num_inference_steps=8):
        """加载现有评估入口使用的 EMA/model 权重并验证契约；异常阻止节点启动。"""
        from dp_infer.policy_loader import load_policy

        self.policy, self.cfg = load_policy(checkpoint, device, num_inference_steps)
        self.urdf_sha256 = validate_contract(self.cfg, expected_urdf_sha256)
        self.device = device  # 推理张量所在设备。
        self.processors = list(processors)  # 已构造的后处理扩展。

    def warmup(self):
        """仅在 GPU 上预热模型；合成输入的输出不进入 ROS 控制链路。"""
        import torch

        observations = {
            key: np.zeros((1, int(meta.horizon), *meta.shape), dtype=np.float32)
            for key, meta in self.cfg.shape_meta.obs.items()
        }
        for _ in range(6):
            self.predict_raw(observations)
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize(torch.device(self.device))

    def predict_raw(self, observations):
        """执行一次模型前向推理，返回 CPU 上的 ``[1,16,10]`` 原始动作。"""
        import torch

        inputs = {key: torch.from_numpy(value).to(self.device) for key, value in observations.items()}
        with torch.inference_mode():
            prediction = self.policy.predict_action(inputs)["action_pred"].cpu().numpy()
        if prediction.shape != (1, 16, 10):
            raise ValueError(f"Unexpected policy shape: {prediction.shape}")
        if not np.isfinite(prediction).all():
            raise ValueError("Policy returned non-finite actions")
        return prediction

    def predict(self, observations, context, num_inference_steps=None):
        """在工作线程设置本次去噪步数，再推理并返回绝对动作序列。"""
        if num_inference_steps is not None:
            self.policy.num_inference_steps = num_inference_steps
        prediction = self.predict_raw(observations)
        sequence = decode_actions(prediction[0], context)
        for processor in self.processors:
            sequence = validate_sequence(processor.process(sequence, context))
        return sequence
