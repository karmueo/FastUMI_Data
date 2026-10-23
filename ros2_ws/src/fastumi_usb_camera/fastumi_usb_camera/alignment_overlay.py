"""末端相机安装对齐：把 UMI 夹爪参考 mask 叠加到实时图像（纯函数，不依赖 ROS）。

参考图由仓库根目录 ``extract_umi_gripper_mask.py`` 生成，已重投影到末端相机像素系：
- PNG 为 BGRA（BGR 为 UMI 夹爪中位图，alpha 为 mask），或单通道灰度 mask（按固定颜色着色）；
- 同名 ``.yaml`` 旁注（可选）给出手指码参考角点 ``tag_corners_px``：id → 4×2 像素，左上起顺时针。
图像均为 OpenCV BGR uint8，形状 [H,W,3]。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

# 手指码字典与 id，与 extract_umi_gripper_mask.py / compute_link7_correction.py 一致。
TAG_DICTIONARY = cv2.aruco.DICT_4X4_50
TAG_IDS = (0, 1)
# 单通道 mask 的着色（BGR，品红，易与夹爪黑色和橙色区分）。
GRAY_MASK_COLOR = (255, 0, 255)
CONTOUR_COLOR = (0, 255, 0)
REFERENCE_TAG_COLOR = (0, 0, 255)
LIVE_TAG_COLOR = (0, 255, 255)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)


@dataclass
class AlignmentReference:
    """叠加参考：BGR 参考图、[0,1] float32 mask、mask 外轮廓与参考码角点。"""

    image: np.ndarray
    mask: np.ndarray
    contours: list = field(default_factory=list)
    tag_corners: dict[int, np.ndarray] = field(default_factory=dict)


def load_reference(path: Path) -> AlignmentReference:
    """读取 BGRA 或灰度 mask PNG 及可选同名 YAML 旁注。"""
    path = Path(path)
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"无法读取对齐 mask: {path}")
    if raw.dtype != np.uint8:
        raise ValueError(f"对齐 mask 必须是 8 位图像: {path}")
    if raw.ndim == 3 and raw.shape[2] == 4:
        image, mask = np.ascontiguousarray(raw[..., :3]), raw[..., 3]
    elif raw.ndim == 2:
        image = np.empty(raw.shape + (3,), np.uint8)
        image[:] = GRAY_MASK_COLOR
        mask = raw
    else:
        raise ValueError(f"对齐 mask 必须是 BGRA 或单通道灰度 PNG: {path}")
    binary = (mask >= 128).astype(np.uint8)
    if not binary.any():
        raise ValueError(f"对齐 mask 为空: {path}")
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tags = {}
    sidecar = path.with_suffix(".yaml")
    if sidecar.exists():
        document = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
        corners = document.get("umi_gripper_mask", {}).get("tag_corners_px", {})
        tags = {int(key): np.asarray(value, np.float64).reshape(4, 2)
                for key, value in corners.items()}
    return AlignmentReference(image, mask.astype(np.float32) / 255.0, list(contours), tags)


def blend(frame: np.ndarray, reference: AlignmentReference, alpha: float) -> np.ndarray:
    """按 ``out = frame·(1−α·m) + ref·(α·m)`` 叠加；mask 外像素保持不变。"""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 必须在 [0, 1] 内")
    if frame.shape != reference.image.shape:
        raise ValueError(
            f"图像尺寸 {frame.shape[1]}x{frame.shape[0]} 与对齐 mask "
            f"{reference.image.shape[1]}x{reference.image.shape[0]} 不一致")
    # 只在 mask 外接框内混合，框外直接复制，降低全分辨率逐帧开销。
    rows, columns = np.nonzero(reference.mask.any(axis=1))[0], np.nonzero(reference.mask.any(axis=0))[0]
    box = (slice(rows[0], rows[-1] + 1), slice(columns[0], columns[-1] + 1))
    weight = (alpha * reference.mask[box])[..., None]
    mixed = (frame[box].astype(np.float32) * (1.0 - weight)
             + reference.image[box].astype(np.float32) * weight)
    output = frame.copy()
    output[box] = np.clip(mixed + 0.5, 0, 255).astype(np.uint8)
    return output


def tag_errors(live: dict[int, np.ndarray],
               reference: dict[int, np.ndarray]) -> dict[int, float]:
    """实拍码角点与参考角点的平均像素距离；只统计两侧都有的 id。"""
    return {
        marker_id: float(np.linalg.norm(live[marker_id] - reference[marker_id], axis=1).mean())
        for marker_id in sorted(set(live) & set(reference))
    }


class TagDetector:
    """检测实拍图像中的手指码角点（像素，左上起顺时针）。"""

    def __init__(self) -> None:
        self._detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(TAG_DICTIONARY), cv2.aruco.DetectorParameters())

    def detect(self, frame: np.ndarray) -> dict[int, np.ndarray]:
        # 全分辨率检测在 Jetson 上过慢：半分辨率检测，再在原图上亚像素精修角点。
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(
            cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        if ids is None:
            return {}
        result = {}
        for corner, marker_id in zip(corners, ids.ravel()):
            if int(marker_id) not in TAG_IDS:
                continue
            points = ((corner.reshape(4, 1, 2) + 0.5) * 2.0 - 0.5).astype(np.float32)
            cv2.cornerSubPix(gray, points, (5, 5), (-1, -1), SUBPIX_CRITERIA)
            result[int(marker_id)] = points.reshape(4, 2).astype(np.float64)
        return result


def render(frame: np.ndarray, reference: AlignmentReference, alpha: float,
           live_tags: dict[int, np.ndarray] | None = None) -> tuple[np.ndarray, dict[int, float]]:
    """返回叠加后的图像及码角点误差：绿色为 mask 轮廓，红色为参考码，黄色为实拍码。"""
    output = blend(frame, reference, alpha)
    cv2.drawContours(output, reference.contours, -1, CONTOUR_COLOR, 2, cv2.LINE_AA)
    for quad in reference.tag_corners.values():
        cv2.polylines(output, [np.round(quad).astype(np.int32)], True,
                      REFERENCE_TAG_COLOR, 2, cv2.LINE_AA)
    live_tags = live_tags or {}
    for quad in live_tags.values():
        cv2.polylines(output, [np.round(quad).astype(np.int32)], True,
                      LIVE_TAG_COLOR, 2, cv2.LINE_AA)
    errors = tag_errors(live_tags, reference.tag_corners)
    if reference.tag_corners:
        text = "  ".join(f"tag{key}: {value:.1f}px" for key, value in errors.items()) \
            or "tags not detected"
        cv2.putText(output, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6,
                    cv2.LINE_AA)
        cv2.putText(output, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, LIVE_TAG_COLOR, 2,
                    cv2.LINE_AA)
    return output, errors
