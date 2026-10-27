"""Zarr 与 LeRobot 两种导出器共用的 Link7 数值与图像转换。

只依赖 NumPy、SciPy 和 OpenCV，不依赖 Zarr、JPEG XL 或 LeRobot。位姿均位于
episode 起始 Link7 坐标系：位置单位米，旋转向量单位弧度，夹爪为归一化开度 [0,1]。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# 导出图像的边长（像素），与部署端 letterbox 一致。
IMAGE_SIZE = 224


def letterbox_rgb_uint8(rgb: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """与部署 ``letterbox_rgb`` 相同的缩放、居中黑边，返回 ``(size, size, 3)`` uint8。"""
    height, width = rgb.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    output = np.zeros((size, size, 3), dtype=np.uint8)
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return output


def quaternion_pose_to_rotvec(qpos: np.ndarray) -> np.ndarray:
    """把 ``(N, 8)`` 的 ``[xyz, 四元数 xyzw, 夹爪]`` 转为 ``(N, 7)`` float64 ``[xyz, 旋转向量, 夹爪]``。

    旋转向量的转角位于 ``[0, π]``；接近 π 时相邻帧的轴可能整体反号，但对应的旋转矩阵连续。
    """
    values = np.asarray(qpos, dtype=np.float64)
    rotvec = Rotation.from_quat(values[:, 3:7]).as_rotvec()
    return np.concatenate((values[:, :3], rotvec, values[:, 7:8]), axis=1)


def numeric_episode(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """把 HDF5 的 qpos/action（xyz + 四元数 xyzw + 夹爪）转换为 Zarr 数值数组。"""
    qpos = np.asarray(arrays["qpos"], dtype=np.float64)
    state = quaternion_pose_to_rotvec(qpos)
    action = quaternion_pose_to_rotvec(arrays["action"])
    count = len(qpos)
    return {
        "robot0_eef_pos": state[:, :3].astype(np.float32),
        "robot0_eef_rot_axis_angle": state[:, 3:6].astype(np.float32),
        "robot0_gripper_width": state[:, 6:7].astype(np.float32),
        "robot0_demo_start_pose": np.repeat(state[:1, :6], count, axis=0).astype(np.float32),
        "action": action.astype(np.float32),
        "timestamp": np.asarray(arrays["timestamp"], dtype=np.float64),
        "nominal_timestamp_ns": np.asarray(arrays["nominal_timestamp_ns"], dtype=np.int64),
        "image_timestamp_ns": np.asarray(arrays["image_timestamp_ns"], dtype=np.int64),
        "source_image_index": np.asarray(arrays["source_image_index"], dtype=np.int64),
        "dropped_frames_before": np.asarray(arrays["dropped_frames_before"], dtype=np.int32),
    }


def to_jsonable(value: Any) -> Any:
    """把含 NumPy 标量/数组的嵌套结构转换为纯 JSON 类型。"""

    def default(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"无法序列化 {type(item).__name__}")

    return json.loads(json.dumps(value, default=default))
