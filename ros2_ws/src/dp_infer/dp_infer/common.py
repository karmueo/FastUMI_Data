"""模型无关的同步观测、动作消息数据与 RGB/关节输入校验。"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))


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



def compressed_image_to_rgb(data):
    """将相机 JPEG/PNG 解码为独立的 HWC RGB uint8 数组。"""
    encoded = np.frombuffer(data, dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError("Compressed image payload is empty")
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Compressed image cannot be decoded as three-channel color")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)



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

    positions: np.ndarray  # [N,3]，base_link 下的 Link7 目标位置。
    quaternions: np.ndarray  # [N,4]，base_link 下的 Link7 目标旋转。
    gripper_openness: np.ndarray  # [N]，归一化夹爪目标。
    time_from_start: np.ndarray  # [N]，相对观测时间的预测偏移。



def load_link7_correction(path):
    """读取 Link7 部署修正 C（真实 Link7 在策略虚拟 Link7 系中的 [4,4] 位姿，米）。

    训练标签按 UMI 相机≡RM75 相机生成；两台相机相对手指的安装不同时，发布目标取
    ``策略 Link7 · C``、策略位姿观测取 ``FK Link7 · C⁻¹``，使 RM75 手指落到训练手指位置。
    空路径返回单位阵；文件由仓库根 ``compute_link7_correction.py`` 生成。
    """
    if not path:
        return np.eye(4)
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    matrix = np.asarray(document["link7_correction"]["T_virtual_link7"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("link7 correction must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1]) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            or np.linalg.det(rotation) <= 0):
        raise ValueError("link7 correction must be a rigid transform")
    # 文件按 9 位小数保存，重新正交化以免累积缩放。
    corrected = np.eye(4)
    corrected[:3, :3] = Rotation.from_matrix(rotation).as_matrix()
    corrected[:3, 3] = matrix[:3, 3]
    return corrected

