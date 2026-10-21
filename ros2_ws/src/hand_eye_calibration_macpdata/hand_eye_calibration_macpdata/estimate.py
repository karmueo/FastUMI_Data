"""在线粗手眼估计、板在基座下的位姿与样本多样性统计。

在线估计只用于规划“看向标定板”的位姿，最终标定由 export_hand_eye_dataset 命令（本包 solve.py）完成。
输入：base_from_flange = ^base T_flange（FK），camera_from_board = ^cam T_board（PnP）。
输出：flange_from_camera = ^flange T_cam（眼在手上外参）。
"""

from dataclasses import dataclass
import itertools
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from fastumi_bringup.transforms import (
    average_transforms, invert, rotation_angle, rotation_log,
)


def load_prior(path, convention="cam_from_flange"):
    """读取已有外参 YAML 作为 ^flange T_cam 先验。

    支持 ``cam0.T_cam_hand``（仓库 config/calibration/T_cam_end.yaml，按 ^A T_B 约定
    即 ^cam T_flange，convention=cam_from_flange）以及 hand_eye_calibration 输出的
    ``T``（眼在手上时 parent=effector，即 ^flange T_cam，convention=flange_from_cam）。
    """
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        # hand_eye_calibration 用 cv::FileStorage 写出 %YAML:1.0 和 !!opencv-matrix。
        data = {}
    if "cam0" in data and "T_cam_hand" in data["cam0"]:
        matrix = np.asarray(data["cam0"]["T_cam_hand"], dtype=float)
    else:
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        matrix = storage.getNode("T").mat() if storage.isOpened() else None
        storage.release()
        if matrix is None:
            raise ValueError(f"{path} 中没有 cam0.T_cam_hand 或 T")
        matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("外参矩阵必须为 4x4")
    if convention == "cam_from_flange":
        return invert(matrix)
    if convention == "flange_from_cam":
        return matrix
    raise ValueError(f"未知外参方向 {convention}")


def estimate_flange_from_camera(base_from_flange, camera_from_board,
                                method=cv2.CALIB_HAND_EYE_PARK):
    """OpenCV calibrateHandEye（眼在手上）；样本不足 3 个或数值异常时返回 None。"""
    if len(base_from_flange) < 3:
        return None
    rotation, translation = cv2.calibrateHandEye(
        [transform[:3, :3] for transform in base_from_flange],
        [transform[:3, 3] for transform in base_from_flange],
        [transform[:3, :3] for transform in camera_from_board],
        [transform[:3, 3] for transform in camera_from_board],
        method=method,
    )
    if rotation is None or not np.all(np.isfinite(rotation)) or not np.all(
            np.isfinite(translation)):
        return None
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation).reshape(3)
    return result


@dataclass(frozen=True)
class BoardEstimate:
    """板在基座下的平均位姿及各样本相对平均值的离散度（毫米、度）。"""

    base_from_board: np.ndarray
    spread_mm: float
    spread_deg: float


def estimate_board_in_base(base_from_flange, camera_from_board, flange_from_camera):
    """^base T_board = ^base T_flange · ^flange T_cam · ^cam T_board，对样本取平均。"""
    boards = [flange @ flange_from_camera @ board
              for flange, board in zip(base_from_flange, camera_from_board)]
    mean = average_transforms(boards)
    spread_mm = max(np.linalg.norm(board[:3, 3] - mean[:3, 3]) for board in boards) * 1000.0
    spread_deg = max(math.degrees(rotation_angle(board[:3, :3], mean[:3, :3]))
                     for board in boards)
    return BoardEstimate(mean, float(spread_mm), float(spread_deg))


@dataclass(frozen=True)
class DiversityStats:
    """样本激励统计：最大两两姿态差、相对运动旋转轴最大夹角（度）。"""

    sample_count: int
    rotation_span_deg: float
    axis_separation_deg: float


def diversity_stats(base_from_flange, min_motion_deg=5.0):
    """估计手眼标定质量门关注的旋转跨度与轴分离度。"""
    rotations = [transform[:3, :3] for transform in base_from_flange]
    span = 0.0
    axes = []
    for first, second in itertools.combinations(rotations, 2):
        relative = rotation_log(first.T @ second)
        angle = float(np.linalg.norm(relative))
        span = max(span, angle)
        if angle > math.radians(min_motion_deg):
            axes.append(relative / angle)
    separation = 0.0
    for first, second in itertools.combinations(axes, 2):
        # 轴方向正负等价，取 [0°, 90°]。
        separation = max(separation, math.acos(min(1.0, abs(float(first @ second)))))
    return DiversityStats(len(rotations), math.degrees(span), math.degrees(separation))
