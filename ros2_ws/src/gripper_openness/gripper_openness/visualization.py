"""绘制夹爪 ArUco 检测结果和标定状态的调试图像。"""

from __future__ import annotations

from typing import Optional

import cv2
from gripper_openness.calibration import CameraCalibration
from gripper_openness.vision import VisionResult
import numpy as np


def draw_vision_result(
    image: np.ndarray,
    result: VisionResult,
    label: str = "",
    distance_range: Optional[tuple[float, float]] = None,
    camera_calibration: Optional[CameraCalibration] = None,
    marker_size_mm: float = 16.0,
) -> np.ndarray:
    """在图像副本上绘制标记框、连接线和距离状态。"""
    output = image.copy()
    for corners, color, name in (
        (result.left_corners, (0, 0, 255), "L"),
        (result.right_corners, (0, 255, 0), "R"),
    ):
        if corners is None:
            continue
        points = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [points], True, color, 2)
        center = tuple(np.round(corners.mean(axis=0)).astype(np.int32))
        cv2.circle(output, center, 4, color, -1)
        cv2.putText(
            output,
            name,
            (center[0] + 6, center[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    if result.left_corners is not None and result.right_corners is not None:
        left_center = tuple(np.round(result.left_corners.mean(axis=0)).astype(np.int32))
        right_center = tuple(np.round(result.right_corners.mean(axis=0)).astype(np.int32))
        cv2.line(output, left_center, right_center, (255, 255, 0), 2)
    if camera_calibration is not None:
        for pose in (result.left_pose, result.right_pose):
            if pose is None:
                continue
            _draw_pose_axes(output, pose, camera_calibration, marker_size_mm)
    status = label or ("valid" if result.valid else "invalid")
    if result.distance_mm is not None:
        status += f"  distance={result.distance_mm:.2f} mm"
    else:
        status += f"  markers={result.detected_marker_count}/2"
    if distance_range is not None:
        status += f"  range={distance_range[0]:.2f}..{distance_range[1]:.2f} mm"
    cv2.putText(
        output,
        status,
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _draw_pose_axes(
    image: np.ndarray,
    pose,
    camera_calibration: CameraCalibration,
    marker_size_mm: float,
) -> None:
    """将标记坐标系的 XYZ 轴投影回原始图像。"""
    axis = np.asarray(
        [[0.0, 0.0, 0.0], [marker_size_mm, 0.0, 0.0],
         [0.0, marker_size_mm, 0.0], [0.0, 0.0, marker_size_mm]],
        dtype=np.float64,
    ).reshape(-1, 1, 3)
    rotation = np.asarray(pose.rotation_vector, dtype=np.float64).reshape(3, 1)
    translation = np.asarray(pose.translation_mm, dtype=np.float64).reshape(3, 1)
    try:
        if camera_calibration.is_fisheye:
            points, _ = cv2.fisheye.projectPoints(
                axis,
                rotation,
                translation,
                camera_calibration.camera_matrix,
                camera_calibration.distortion_coefficients.reshape(4, 1),
            )
        else:
            points, _ = cv2.projectPoints(
                axis,
                rotation,
                translation,
                camera_calibration.camera_matrix,
                camera_calibration.distortion_coefficients,
            )
    except cv2.error:
        return
    projected = np.round(points.reshape(-1, 2)).astype(np.int32)
    origin = tuple(projected[0])
    cv2.line(image, origin, tuple(projected[1]), (0, 0, 255), 2)
    cv2.line(image, origin, tuple(projected[2]), (0, 255, 0), 2)
    cv2.line(image, origin, tuple(projected[3]), (255, 0, 0), 2)
