#!/usr/bin/env python3
"""Reusable single-class Jingbao YOLO detector and offline benchmark CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np


DEFAULT_MODEL = (
    Path(".local/dexgraspvla/weights/detection/yolo26s_jingbao.pt")
)


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    latency_seconds: float = 0.0


class JingbaoDetector:
    """Select the highest-confidence detection for one configured class."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        *,
        target_class: str = "jingbao",
        confidence_threshold: float = 0.25,
        device: str | int | None = None,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be in [0, 1]")
        model_path = Path(model_path)
        if model is None:
            if not model_path.is_file():
                raise FileNotFoundError(f"YOLO checkpoint does not exist: {model_path}")
            import torch
            from ultralytics import YOLO

            model = YOLO(str(model_path))
            if device is None:
                device = 0 if torch.cuda.is_available() else "cpu"
        self.model = model
        self.device = device
        self.target_class = target_class
        self.confidence_threshold = float(confidence_threshold)

    def detect(self, image_bgr: np.ndarray) -> Detection | None:
        started = time.perf_counter()
        image_bgr = np.asarray(image_bgr)
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"expected BGR image (H,W,3), got {image_bgr.shape}")
        kwargs = {"conf": self.confidence_threshold, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model(image_bgr, **kwargs)
        if not results:
            return None
        candidates: list[Detection] = []
        for box in results[0].boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = str(self.model.names[class_id])
            if class_name != self.target_class:
                continue
            coordinates = tuple(float(value) for value in box.xyxy[0].tolist())
            if len(coordinates) != 4 or not np.isfinite(coordinates).all():
                continue
            candidates.append(
                Detection(
                    coordinates,
                    confidence,
                    class_id,
                    class_name,
                    time.perf_counter() - started,
                )
            )
        return max(candidates, key=lambda item: item.confidence, default=None)



# Adapted from DexGraspSJ-deploy c5bbb47be2d09936f68fe05e31d690c5a0c3cf96.
