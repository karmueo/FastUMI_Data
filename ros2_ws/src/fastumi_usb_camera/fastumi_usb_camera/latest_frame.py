"""提供线程安全的单帧缓存，防止相机发布链路累积延迟。"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any


@dataclass(frozen=True)
class CapturedFrame:
    """保存同一完整 MJPEG 帧的主机接收时间戳与原始字节。"""

    stamp: Any
    jpeg: bytes


class LatestFrame:
    """只保留最新待处理帧，并统计被覆盖的旧帧。"""

    def __init__(self) -> None:
        """创建空缓存及采集、覆盖计数器。"""
        self._lock = threading.Lock()
        self._pending: CapturedFrame | None = None
        self.captured_count = 0
        self.overwritten_count = 0

    def put(self, frame: CapturedFrame) -> None:
        """替换尚未处理的帧；处理中的帧不受影响。"""
        with self._lock:
            if self._pending is not None:
                self.overwritten_count += 1
            self._pending = frame
            self.captured_count += 1

    def take(self) -> CapturedFrame | None:
        """取出并清空待处理帧，避免后续唤醒重复发布。"""
        with self._lock:
            frame = self._pending
            self._pending = None
            return frame
