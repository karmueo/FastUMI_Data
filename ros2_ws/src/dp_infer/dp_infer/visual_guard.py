"""从腕部图像定位当前抓球目标，并判断球是否进入夹爪中心区域。"""

from __future__ import annotations

import cv2
import numpy as np


def _marked_ball_center(white: np.ndarray, blue: np.ndarray,
                        orange: np.ndarray, scale: float) -> tuple[float, float] | None:
    """白球与亮背景连通时，利用尺寸受限的蓝橙标记配对定位球心。"""
    height, width = white.shape
    markers = []
    for mask, min_area in ((blue, 40), (orange, 80)):
        count, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for index in range(1, count):
            _, _, box_width, box_height, area = stats[index]
            center_x, center_y = centers[index]
            if (area >= max(2, min_area * scale**2)
                    and 10 * scale <= box_width <= 140 * scale
                    and 10 * scale <= box_height <= 140 * scale
                    and width * 0.1 < center_x < width * 0.9
                    and center_y < height * 0.78):
                candidates.append((int(area), np.array([center_x, center_y])))
        markers.append(candidates)
    pairs = []
    radius = max(3, round(90 * scale))
    for blue_area, blue_center in markers[0]:
        for orange_area, orange_center in markers[1]:
            if np.linalg.norm(blue_center - orange_center) > 110 * scale:
                continue
            center = (blue_center + orange_center) / 2
            x, y = center.astype(int)
            region = white[max(0, y - radius):min(height, y + radius),
                           max(0, x - radius):min(width, x + radius)]
            if cv2.countNonZero(region) >= max(20, 1000 * scale**2):
                pairs.append((min(blue_area, orange_area), center))
    if not pairs:
        return None
    center = max(pairs, key=lambda pair: pair[0])[1]
    return float(center[0]), float(center[1])


def ball_center_bgr(pixels: np.ndarray) -> tuple[float, float] | None:
    """返回 BGR 图像中白色球身的像素中心；蓝橙标记用于排除托盘和背景。"""
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError("expected HWC uint8 BGR image")
    height, width = pixels.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    # 每帧只需定位球心，检测图像缩到 640 像素可减少大幅 HSV/连通域运算。
    if width > 640:
        detection_height = max(1, round(height * 640 / width))
        detection_image = cv2.resize(pixels, (640, detection_height),
                                     interpolation=cv2.INTER_AREA)
        center = ball_center_bgr(detection_image)
        if center is None:
            return None
        return center[0] * width / 640, center[1] * height / detection_height
    scale = width / 1280.0
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)
    # 自动曝光变暗时，白球的 V 值也可能降到 120 左右；仅在整帧明显欠曝时
    # 放宽白色阈值，仍要求球身面积、尺寸及蓝橙标记同时成立。
    dark_frame = np.percentile(hsv[::16, ::16, 2], 95) < 100
    white_floor = 120 if dark_frame else 155
    white = cv2.inRange(hsv, (0, 0, white_floor), (179, 40, 255))
    white[int(height * 0.82):] = 0
    orange = cv2.inRange(hsv, (4, 90, 60), (30, 255, 255))
    blue = cv2.inRange(hsv, (90, 80, 25), (135, 255, 255))
    count, _, stats, centers = cv2.connectedComponentsWithStats(white)
    choices = []
    for index in range(1, count):
        x, y, box_width, box_height, area = stats[index]
        center_x, center_y = centers[index]
        # 夹爪附近的反光、电缆与标签也可能同时带有蓝橙色；排除小块白色碎片。
        if (area < max(8, 1000 * scale**2)
                or not 30 * scale <= box_width <= 180 * scale
                or not 30 * scale <= box_height <= 180 * scale
                or not width * 0.1 < center_x < width * 0.9
                or not center_y < height * 0.8):
            continue
        padding = max(2, round(15 * scale))
        bounds = (
            slice(max(0, y - padding), min(height, y + box_height + padding)),
            slice(max(0, x - padding), min(width, x + box_width + padding)),
        )
        if (cv2.countNonZero(blue[bounds]) >= max(2, 40 * scale**2)
                and cv2.countNonZero(orange[bounds]) >= max(4, 100 * scale**2)):
            choices.append(index)
    if not choices:
        return _marked_ball_center(white, blue, orange, scale)
    jaw_region = np.array([width * 0.5, height * 0.67])
    near_jaw = [index for index in choices
                if np.linalg.norm(centers[index] - jaw_region) < width * 0.2]
    # 靠近夹爪时，画面上方的白色衣物可能比球更大。
    selected = max(near_jaw or choices, key=lambda index: stats[index, cv2.CC_STAT_AREA])
    return float(centers[selected, 0]), float(centers[selected, 1])


def ball_near_jaws(
    center: tuple[float, float] | None,
    width: int,
    height: int,
    max_error_fraction: float = 0.0625,
) -> bool:
    """球心距图像中夹爪闭合区域不超过宽度比例；默认 1280 图像为 80 像素。"""
    if center is None:
        return False
    if width <= 0 or height <= 0 or not 0 < max_error_fraction < 0.5:
        raise ValueError("invalid image dimensions or jaw proximity threshold")
    values = np.asarray(center, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all():
        return False
    goal = np.array([width * 0.5, height * 0.67])
    return bool(np.linalg.norm(values - goal) <= width * max_error_fraction)
