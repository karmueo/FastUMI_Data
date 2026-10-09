"""RM75 数值约定和观测同步；不导入 ROS 或模型运行环境。"""

from collections import deque
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))


def normalization_bounds(urdf):
    """返回训练七轴弧度范围和夹爪 [0,1]，共八维。"""
    joints = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}
    lower, upper = [], []
    for name in JOINT_NAMES:
        limit = joints[name].find("limit")
        lower.append(float(limit.get("lower")))
        upper.append(float(limit.get("upper")))
    lower, upper = np.array(lower + [0.0]), np.array(upper + [1.0])
    if not np.isfinite([lower, upper]).all() or np.any(lower >= upper):
        raise ValueError("invalid training joint normalization bounds")
    return lower, upper


def normalize_state(state, lower, upper):
    """物理状态仅做 URDF 线性缩放，不做 action 的统计归一化。"""
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("state must contain eight finite values")
    if np.any(state < lower) or np.any(state > upper):
        raise ValueError("state is outside training limits")
    return (2 * (state - lower) / (upper - lower) - 1).astype(np.float32)


def denormalize_actions(actions, lower, upper):
    """撤销 URDF 缩放；policy 已撤销 action 统计 normalizer。"""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (64, 8) or not np.isfinite(actions).all():
        raise ValueError("policy must return finite actions with shape (64,8)")
    result = (actions + 1) / 2 * (upper - lower) + lower
    if np.any(result < lower) or np.any(result > upper):
        raise ValueError("predicted actions are outside physical limits")
    return result


@dataclass(frozen=True)
class Observation:
    """一个同帧图像与 mask，附带同步后的关节和夹爪物理状态。"""

    stamp_ns: int
    tracking_id: int
    bgr: np.ndarray
    mask: np.ndarray
    state: np.ndarray


class ObservationBuffer:
    """缓存两秒源帧，精确匹配 mask，反馈采用 50 ms 最近邻。"""

    def __init__(self, slop_s=0.05, capacity=256):
        self.slop_ns = int(slop_s * 1e9)
        self.capacity = capacity
        self.clear()

    def clear(self):
        """清除 episode 内尚未消费的观测。"""
        self.images = {}
        self.masks = {}
        self.joints = deque(maxlen=self.capacity)
        self.grippers = deque(maxlen=self.capacity)

    def image(self, stamp, frame, bgr):
        self.images[int(stamp)] = (frame, bgr)
        self._prune(int(stamp))

    def mask(self, stamp, frame, tracking_id, mask):
        self.masks[int(stamp)] = (frame, int(tracking_id), mask)
        self._prune(int(stamp))

    def _prune(self, stamp):
        for mapping in (self.images, self.masks):
            for key in list(mapping):
                if key < stamp - 2_000_000_000:
                    del mapping[key]
            while len(mapping) > self.capacity:
                del mapping[min(mapping)]

    def _nearest(self, values, stamp):
        if not values:
            return None
        time_ns, value = min(values, key=lambda item: abs(item[0] - stamp))
        return value if abs(time_ns - stamp) <= self.slop_ns else None

    def latest(self, now_ns, max_age_s=1.0, tracking_id=None, copy_images=True):
        """不允许用 mask 到达时间匹配另一张最新图像。"""
        for stamp in sorted(self.images.keys() & self.masks.keys(), reverse=True):
            if not 0 <= now_ns - stamp <= int(max_age_s * 1e9):
                continue
            frame, bgr = self.images[stamp]
            mask_frame, identity, mask = self.masks[stamp]
            if frame != mask_frame or (tracking_id is not None and identity != tracking_id):
                continue
            if mask.dtype != np.uint8 or mask.shape != bgr.shape[:2]:
                continue
            if not np.isin(mask, [0, 255]).all() or not np.any(mask):
                continue
            joints = self._nearest(self.joints, stamp)
            gripper = self._nearest(self.grippers, stamp)
            if joints is not None and gripper is not None:
                return Observation(stamp, identity, bgr.copy() if copy_images else bgr,
                                   mask.copy() if copy_images else mask,
                                   np.r_[joints, gripper])
        return None
