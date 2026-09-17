"""按 USB 标识选择 UVC 相机，并在独立线程读取完整 MJPEG 帧。"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable


def _usb_id(value: Any) -> int:
    """把设备枚举返回的整数或十六进制字符串转换为 USB 标识。"""
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return int(value, 16)
    return int(value)


def select_device(devices: list[dict[str, Any]], vendor_id: int,
                  product_id: int) -> dict[str, Any]:
    """选择唯一匹配的设备，否则提供可诊断的候选设备信息。"""
    matches = []
    candidates = []
    for device in devices:
        try:
            vendor = _usb_id(device["idVendor"])
            product = _usb_id(device["idProduct"])
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append(f"{vendor:04x}:{product:04x} ({device.get('uid', '?')})")
        if (vendor, product) == (vendor_id, product_id):
            matches.append(device)
    if len(matches) != 1:
        reason = "未找到" if not matches else "找到多个"
        raise RuntimeError(
            f"{reason} USB 相机 {vendor_id:04x}:{product_id:04x}；"
            f"候选设备: {', '.join(candidates) or '无'}"
        )
    return matches[0]


class UvcCamera:
    """持有 UVC 设备并在线程中递送 MJPEG 字节，关闭时停止 USB 采集。"""

    def __init__(self, *, vendor_id: int, product_id: int, width: int,
                 height: int, fps: int, uvc_module: Any = None) -> None:
        """打开唯一相机并精确协商分辨率与帧率，失败时释放设备。"""
        if uvc_module is None:
            import uvc as uvc_module
        device = select_device(uvc_module.device_list(), vendor_id, product_id)
        capture = uvc_module.Capture(device["uid"])
        try:
            requested = (width, height, fps)
            modes = list(capture.available_modes)
            matching = [mode for mode in modes if tuple(mode[:3]) == requested]
            if not matching:
                raise RuntimeError(
                    f"UVC 模式 {requested} 不可用；可用模式: "
                    f"{', '.join(map(str, modes)) or '无'}"
                )
            capture.frame_mode = matching[0]
        except Exception:
            capture.close()
            raise
        self._capture = capture
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: Exception | None = None
        self.incomplete_count = 0

    def start(self, callback: Callable[[bytes], None]) -> None:
        """启动唯一采集线程；回调在该线程中运行。"""
        if self._thread is not None:
            raise RuntimeError("相机采集线程已经启动")
        self._thread = threading.Thread(
            target=self._run, args=(callback,), name="fastumi-uvc", daemon=True
        )
        self._thread.start()

    def _run(self, callback: Callable[[bytes], None]) -> None:
        """循环读取 MJPEG 帧并跳过不完整数据。"""
        last_warning = 0.0
        try:
            while not self._stop.is_set():
                try:
                    frame = self._capture.get_frame(timeout=0.1)
                except TimeoutError:
                    continue
                jpeg = getattr(frame, "jpeg_buffer", None)
                if not getattr(frame, "data_fully_received", True) or jpeg is None:
                    self.incomplete_count += 1
                    now = time.monotonic()
                    if now - last_warning >= 5.0:
                        logging.warning("UVC 收到不完整或非 MJPEG 帧，已跳过")
                        last_warning = now
                    continue
                callback(bytes(jpeg))
        except Exception as error:
            self.last_error = error
            logging.exception("UVC 采集线程异常退出")
        finally:
            self._stop.set()

    def stopped(self) -> bool:
        """判断采集线程是否已停止。"""
        return self._stop.is_set()

    def close(self) -> None:
        """等待限时读帧结束，再安全关闭 UVC 句柄。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("UVC 采集线程未在 2 秒内停止")
        self._capture.close()
