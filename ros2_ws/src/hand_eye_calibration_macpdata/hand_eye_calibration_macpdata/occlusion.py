"""夹爪遮挡掩膜估计与样本遮挡判定。

夹爪开合度固定且相对相机刚性安装，因此在相机运动中夹爪像素几乎不变，而场景像素随视角
变化。掩膜由多视角帧的逐像素时间标准差得到：低方差、与图像边界连通的区域即夹爪。
掩膜为 mono8，非零表示夹爪（不可用于观测标定板）。
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MaskSettings:
    """掩膜估计参数：在 1/downsample 分辨率上计算，std 单位为灰度级。"""

    downsample: int = 4
    std_threshold: float = 6.0
    min_area_fraction: float = 0.002
    dilate_fraction: float = 0.01  # 膨胀半径占图像宽度的比例（1920 宽约 19 px）
    # 被掩膜包围、面积不超过该比例的空洞视为夹爪本体（如贴在夹爪上的 ArUco 标签：
    # 黑白边缘在抖动和 JPEG 下方差偏大而漏判）；更大的封闭区域可能是指间真实观察窗口，保留。
    max_hole_area_fraction: float = 0.01
    min_frames: int = 6


def estimate_gripper_mask(frames_bgr, board_hulls=(), settings=MaskSettings()):
    """由多个不同位姿下的图像估计夹爪掩膜（原分辨率 mono8）。

    board_hulls 为各帧中检测到的标定板像素凸包（N,2），这些像素属于运动场景，
    会从掩膜中剔除，防止大面积均匀背景被误判。
    """
    frames_bgr = list(frames_bgr)
    if len(frames_bgr) < settings.min_frames:
        raise ValueError(f"估计夹爪掩膜至少需要 {settings.min_frames} 帧，实际 {len(frames_bgr)}")
    height, width = frames_bgr[0].shape[:2]
    small_size = (width // settings.downsample, height // settings.downsample)
    stack = np.stack([
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), small_size,
                   interpolation=cv2.INTER_AREA).astype(np.float32)
        for frame in frames_bgr
    ])
    std = stack.std(axis=0)
    mean = stack.mean(axis=0)
    # 过曝/全黑区域（镜头暗角外）没有纹理信息，视作不可用区域。
    candidate = ((std < settings.std_threshold) | (mean < 8) | (mean > 250)).astype(np.uint8)

    scene = np.zeros_like(candidate)
    for hull in board_hulls:
        points = (np.asarray(hull, dtype=np.float64) / settings.downsample).astype(np.int32)
        cv2.fillConvexPoly(scene, cv2.convexHull(points), 1)
    candidate[scene > 0] = 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    small_mask = np.zeros_like(candidate)
    min_area = settings.min_area_fraction * candidate.size
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        touches_border = (x == 0 or y == 0 or x + w >= small_size[0] or
                          y + h >= small_size[1])
        if touches_border and area >= min_area:
            small_mask[labels == label] = 255
    fill_enclosed_holes(small_mask, settings.max_hole_area_fraction)
    mask = cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    dilate_px = int(round(settings.dilate_fraction * width))
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, kernel)
    return mask


def fill_enclosed_holes(mask, max_area_fraction):
    """原地填充 mono8 掩膜中不与图像边界连通、面积不超过 max_area_fraction 的空洞。"""
    holes = (mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    height, width = mask.shape
    max_area = max_area_fraction * mask.size
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        touches_border = x == 0 or y == 0 or x + w >= width or y + h >= height
        if not touches_border and area <= max_area:
            mask[labels == label] = 255
    return mask


def load_mask(path, width, height):
    """读取人工提供的掩膜 PNG；尺寸不符时报错。"""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取夹爪掩膜 {path}")
    if mask.shape != (height, width):
        raise ValueError(f"夹爪掩膜尺寸 {mask.shape[::-1]} 与相机 {width}x{height} 不符")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


@dataclass(frozen=True)
class OcclusionCheck:
    """遮挡判定结果：ok 为 False 时 reason 为可读原因。"""

    ok: bool
    reason: str = ""
    overlap_px: int = 0


def board_polygon(spec, camera, camera_from_board, samples_per_edge=12):
    """板外框投影到鱼眼图像的多边形（沿边采样以近似鱼眼弯曲）；不可投影时返回 None。"""
    outline = spec.outline_points()
    points = []
    for start, end in zip(outline, np.roll(outline, -1, axis=0)):
        for fraction in np.linspace(0.0, 1.0, samples_per_edge, endpoint=False):
            points.append(start + (end - start) * fraction)
    points = np.asarray(points)
    camera_points = points @ camera_from_board[:3, :3].T + camera_from_board[:3, 3]
    pixels = camera.project(camera_points)
    if not np.all(np.isfinite(pixels)):
        return None
    return pixels


def min_cell_px(spec, camera, camera_from_board):
    """板上最小特征（标签黑框边 / 棋盘方格边）投影后每格的像素数；不可投影时返回 0。"""
    segments = spec.feature_segments()
    points = segments.reshape(-1, 3) @ camera_from_board[:3, :3].T + camera_from_board[:3, 3]
    pixels = camera.project(points).reshape(-1, 2, 2)
    lengths = np.linalg.norm(pixels[:, 1] - pixels[:, 0], axis=1)
    if not np.all(np.isfinite(lengths)):
        return 0.0
    return float(lengths.min()) / spec.cells_per_feature


def check_board_view(spec, camera, camera_from_board, mask, *, margin_px=20,
                     min_area_fraction=0.03, min_cell_px_required=0.0):
    """预测或复核某位姿下的板视图：全板在画面内、面积与特征像素足够、不与夹爪掩膜相交。

    min_cell_px_required > 0 时，要求最小特征每格至少这么多像素（小标签板远距离斜视时
    无法解码）；为 0 时不检查。
    """
    polygon = board_polygon(spec, camera, camera_from_board)
    if polygon is None:
        return OcclusionCheck(False, "板在相机后方")
    if (np.any(polygon[:, 0] < margin_px) or np.any(polygon[:, 1] < margin_px) or
            np.any(polygon[:, 0] > camera.width - 1 - margin_px) or
            np.any(polygon[:, 1] > camera.height - 1 - margin_px)):
        return OcclusionCheck(False, "板超出画面边距")
    area = cv2.contourArea(polygon.astype(np.float32))
    if area < min_area_fraction * camera.width * camera.height:
        return OcclusionCheck(False, f"板像面积过小 {area / (camera.width * camera.height):.1%}")
    if min_cell_px_required > 0.0:
        cell = min_cell_px(spec, camera, camera_from_board)
        if cell < min_cell_px_required:
            return OcclusionCheck(False, f"特征过小 {cell:.1f} px/格")
    if mask is None:
        return OcclusionCheck(True)
    board_region = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(board_region, [np.round(polygon).astype(np.int32)], 255)
    overlap = int(np.count_nonzero((board_region > 0) & (mask > 0)))
    if overlap > 0:
        return OcclusionCheck(False, f"板与夹爪掩膜重叠 {overlap} px", overlap)
    return OcclusionCheck(True)
