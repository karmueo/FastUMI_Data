"""构造训练一致的 UMI 观测，执行推理并将相对动作还原为绝对目标序列。

checkpoint 的末端契约有两种：

- ``Link7``：训练位姿即 Link7，工具偏移为单位阵（旧 Link7 数据）。
- ``tcp``：训练位姿为夹爪抓取中心 TCP（``fastumi-tcp-v1``）。部署端用
  ``T_Link7_TCP = inverse(T_cam_end) × T_d_cam_tcp`` 把 FK 得到的 Link7 换算为 TCP 作为观测，
  再把预测的 TCP 目标右乘 ``inverse(T_Link7_TCP)`` 还原为 Link7 目标发布。
"""

from dataclasses import dataclass, field
import hashlib
import importlib
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

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
# 两种可部署 checkpoint 契约的共同部分。
COMMON_CONTRACT = {"base_frame": "base_link", "gripper_representation": "normalized_0_1",
                   "action_reference": "current_observed_end_frame", "frequency_hz": 30}
# 末端坐标系 → 该契约要求的工具偏移声明。
END_FRAME_TOOL_OFFSETS = {"Link7": "identity", "tcp": "deployment_calibration"}
# 示教 TCP 与部署 TCP 坐标轴允许的最大夹角（度）；超过说明两端 TCP 轴向约定不一致。
DEFAULT_MAX_TCP_AXIS_DEG = 10.0
# 齐次刚体变换合法性检查的数值容差。
_ROTATION_TOLERANCE = 1e-5


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
    """一个已同步观测：ROS 纳秒时间、CHW RGB、base→策略末端变换与归一化夹爪。"""

    stamp_ns: int  # 图像采集时间。
    image: np.ndarray  # [3,224,224] 浮点 RGB。
    pose: np.ndarray  # [4,4] 米制绝对位姿，末端为 checkpoint 契约的 Link7 或 TCP。
    gripper: float  # 训练使用的归一化开度。


@dataclass(frozen=True)
class InferenceContext:
    """一次推理固定的参考状态；传入后处理器，不随实时订阅更新。"""

    reference_pose: np.ndarray  # 本次最新观测的 base→策略末端（Link7 或 TCP）变换。
    stamp_ns: int  # 本次最新图像采集时间。
    episode_id: int  # 重置时递增的 episode 标识。
    sequence_id: int  # 每次提交推理时递增的编号。
    # T_Link7_TCP；Link7 契约为单位阵。解码时用它把策略末端目标还原为 Link7 目标。
    link7_from_tool: np.ndarray = field(default_factory=lambda: np.eye(4))


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
    """把 [16,10] 相对动作左乘冻结观测位姿并换算为 Link7，输出完整绝对目标；拒绝退化 6D 旋转。

    ``T_base_Link7 = T_base_tool(ref) × relative × inverse(T_Link7_tool)``，Link7 契约下
    ``T_Link7_tool`` 为单位阵。
    """
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
    matrices = (context.reference_pose @ pose10d_to_mat(values[:, :9])
                @ np.linalg.inv(context.link7_from_tool))
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


def contract_end_frame(cfg):
    """返回 checkpoint 契约的末端坐标系 ``Link7`` 或 ``tcp``；其他取值抛出 ValueError。"""
    contract = getattr(cfg.task, "contract", None)  # 兼容 OmegaConf 与测试注入的简单对象。
    end_frame = getattr(contract, "end_frame", None) if contract is not None else None
    if end_frame not in END_FRAME_TOOL_OFFSETS:
        raise ValueError(f"Unsupported checkpoint end_frame: {end_frame!r}")
    return end_frame


def validate_contract(cfg, expected_urdf_sha256=None):
    """限制为当前 RM75 五键/10D/30Hz 契约（Link7 或 TCP 末端），并返回可信的训练 URDF 摘要。"""
    contract = cfg.task.get("contract", {})
    end_frame = contract.get("end_frame")
    expected = dict(COMMON_CONTRACT, end_frame=end_frame,
                    tool_offset=END_FRAME_TOOL_OFFSETS.get(end_frame))
    if end_frame not in END_FRAME_TOOL_OFFSETS or any(
            contract.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint does not match the RM75 Link7/TCP deployment contract")
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


def _rigid_transform(value, name):
    """要求 4×4 有限齐次刚体变换，返回 float64 副本；否则抛出 ValueError。"""
    try:
        matrix = np.array(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not a numeric matrix") from error
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=_ROTATION_TOLERANCE)
            or abs(np.linalg.det(rotation) - 1) > _ROTATION_TOLERANCE):
        raise ValueError(f"{name} must be a rigid transform")
    return matrix


def _read_yaml(path, name):
    """读取 YAML 映射；文件缺失或格式错误时抛出 ValueError。"""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{name} must name an existing file: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{name} {path} must be a YAML mapping")
    return document


def load_deploy_tcp(end_camera_path, deploy_tcp_path):
    """读取部署外参，返回 ``(T_Link7_TCP, T_Cd_TCP)``，平移单位米。

    Args:
        end_camera_path: ``T_cam_end.yaml``，``cam0.T_cam_hand`` 为部署相机←Link7。
        deploy_tcp_path: ``T_d_cam_tcp.yaml``，顶层 ``T_d_cam_tcp`` 为部署相机←TCP。

    Raises:
        ValueError: 文件缺失、缺少矩阵或矩阵不是刚体变换。
    """
    end = _read_yaml(end_camera_path, "end_camera_extrinsic").get("cam0") or {}
    if "T_cam_hand" not in end:
        raise ValueError(f"{end_camera_path} is missing cam0.T_cam_hand")
    camera_from_link7 = _rigid_transform(end["T_cam_hand"], "T_cam_end")
    document = _read_yaml(deploy_tcp_path, "deploy_tcp_extrinsic")
    if "T_d_cam_tcp" not in document:
        raise ValueError(f"{deploy_tcp_path} is missing T_d_cam_tcp")
    camera_from_tcp = _rigid_transform(document["T_d_cam_tcp"], "T_d_cam_tcp")
    return np.linalg.inv(camera_from_link7) @ camera_from_tcp, camera_from_tcp


def compare_tcp_frames(umi_camera_from_tcp, deploy_camera_from_tcp,
                       max_axis_deg=DEFAULT_MAX_TCP_AXIS_DEG):
    """比较示教与部署两端的相机←TCP 外参，返回 ``(平移差 mm, 旋转差 度)``。

    两端 TCP 视为同一物理点，坐标轴必须采用同一约定。旋转差超过 ``max_axis_deg``
    说明轴向约定不一致（常见为 90°/180°），继续部署会让相对动作整体错乱。
    平移差和较小的旋转差来自两套夹具/相机安装差异，只影响图像视角，返回供日志记录。

    Raises:
        ValueError: 矩阵非法或旋转差超过阈值。
    """
    umi = _rigid_transform(umi_camera_from_tcp, "umi_camera_from_tcp")
    deploy = _rigid_transform(deploy_camera_from_tcp, "T_d_cam_tcp")
    difference = np.linalg.inv(umi) @ deploy
    angle = float(np.degrees(np.linalg.norm(Rotation.from_matrix(difference[:3, :3]).as_rotvec())))
    offset_mm = float(np.linalg.norm(difference[:3, 3]) * 1000)
    if angle > max_axis_deg:
        raise ValueError(
            f"UMI and deployment TCP axes differ by {angle:.1f} deg (> {max_axis_deg} deg); "
            "both TCP frames must use the same axis convention")
    return offset_mm, angle


class PolicyEngine:
    """单线程使用的策略实例；ROS 层负责调度，模型依赖仅在创建时导入。"""

    def __init__(self, checkpoint, device="cuda:0", processors=(), expected_urdf_sha256=None):
        """加载现有评估入口使用的 EMA/model 权重并验证契约；异常阻止节点启动。"""
        from dp_infer.policy_loader import load_policy

        self.policy, self.cfg = load_policy(checkpoint, device)
        self.urdf_sha256 = validate_contract(self.cfg, expected_urdf_sha256)
        self.end_frame = contract_end_frame(self.cfg)  # 策略末端：Link7 或 tcp。
        self.device = device  # 推理张量所在设备。
        self.processors = list(processors)  # 已构造的后处理扩展。

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

    def predict(self, observations, context):
        """执行一次推理与后处理，返回绝对动作序列；异常由 ROS 层记录并丢弃。"""
        prediction = self.predict_raw(observations)
        sequence = decode_actions(prediction[0], context)
        for processor in self.processors:
            sequence = validate_sequence(processor.process(sequence, context))
        return sequence
