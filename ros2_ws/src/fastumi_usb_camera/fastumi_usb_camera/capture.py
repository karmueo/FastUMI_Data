"""按视频设备路径或 USB 标识选择相机，并在独立线程读取 MJPEG 帧。"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable


BY_PATH_ROOT = Path("/dev/v4l/by-path")
VIDEO_DEVICE_PATTERN = re.compile(r"[^/]+-video-index0")


def is_physical_video_device_path(
    video_device: str, by_path_root: Path = BY_PATH_ROOT,
) -> bool:
    """判断路径是否为主视频节点的稳定 udev 物理端口链接。"""
    if not isinstance(video_device, str) or not video_device:
        return False
    path = Path(video_device)
    return (
        path.is_absolute()
        and path.parent == by_path_root
        and VIDEO_DEVICE_PATTERN.fullmatch(path.name) is not None
    )


def physical_port_label(video_device: str) -> str:
    """从 by-path 文件名提取便于界面展示的 USB Hub 端口链。"""
    match = re.search(
        r"-usb(?:v\d+)?-\d+:([^:]+):\d+\.\d+-video-index0$",
        Path(video_device).name,
    )
    return match.group(1) if match else Path(video_device).name


def _usb_id(value: Any) -> int:
    """把设备枚举返回的整数或十六进制字符串转换为 USB 标识。"""
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return int(value, 16)
    return int(value)


def _usb_location_from_video_device(
    video_device: str, sysfs_root: Path = Path("/sys/class/video4linux"),
    device_root: Path = Path("/dev"), by_path_root: Path = BY_PATH_ROOT,
) -> tuple[int, int]:
    """校验稳定物理路径，并沿 sysfs 查找 USB 总线号和当前设备地址。"""
    if not is_physical_video_device_path(video_device, by_path_root):
        raise ValueError(
            "视频设备必须使用 /dev/v4l/by-path/*-video-index0 物理端口路径："
            f"{video_device}"
        )
    try:
        resolved = Path(video_device).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"视频设备 {video_device} 不存在或无法解析") from error
    if resolved.parent != device_root.resolve() or not (
        resolved.name.startswith("video") and resolved.name[5:].isdigit()
    ):
        raise ValueError(f"视频设备必须指向 /dev/video*：{video_device}")
    try:
        usb_interface = (sysfs_root / resolved.name / "device").resolve(
            strict=True
        )
    except (OSError, RuntimeError) as error:
        raise ValueError(f"无法读取 {video_device} 的 sysfs 设备信息") from error
    for parent in (usb_interface, *usb_interface.parents):
        try:
            return (int((parent / "busnum").read_text().strip()),
                    int((parent / "devnum").read_text().strip()))
        except (OSError, ValueError):
            continue
    raise ValueError(f"{video_device} 未关联 USB 设备")


def select_device(devices: list[dict[str, Any]], vendor_id: int | None = None,
                  product_id: int | None = None, device_uid: str = "",
                  usb_location: tuple[int, int] | None = None) -> dict[str, Any]:
    """按视频设备对应的 USB 地址或 VID/PID 选择唯一设备。"""
    if usb_location is None and (vendor_id is None or product_id is None):
        raise ValueError("未指定视频设备时必须提供 vendor_id 和 product_id")
    matches = []
    candidates = []
    for device in devices:
        try:
            vendor = _usb_id(device["idVendor"])
            product = _usb_id(device["idProduct"])
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append(f"{vendor:04x}:{product:04x} ({device.get('uid', '?')})")
        if (usb_location is not None or
                (vendor, product) == (vendor_id, product_id)) and (
            not device_uid or device.get("uid") == device_uid
        ) and (
            usb_location is None or (
                device.get("bus_number"), device.get("device_address")
            ) == usb_location
        ):
            matches.append(device)
    if len(matches) != 1:
        reason = "未找到" if not matches else "找到多个"
        target = (
            f"USB 地址 {usb_location}" if usb_location is not None
            else f"USB 相机 {vendor_id:04x}:{product_id:04x}"
        )
        raise RuntimeError(
            f"{reason} {target}"
            f"{f' (device_uid={device_uid})' if device_uid else ''}；"
            f"候选设备: {', '.join(candidates) or '无'}；"
            "同型号设备可指定 video_device 或 device_uid"
        )
    return matches[0]


class UvcCamera:
    """持有 UVC 设备并在线程中递送 MJPEG 字节，关闭时停止 USB 采集。"""

    def __init__(self, *, width: int, height: int, fps: int,
                 vendor_id: int | None = None, product_id: int | None = None,
                 device_uid: str = "", video_device: str = "",
                 uvc_module: Any = None) -> None:
        """打开唯一相机并精确协商分辨率与帧率，失败时释放设备。"""
        if device_uid and video_device:
            raise ValueError("device_uid 和 video_device 只能指定其中一个")
        if uvc_module is None:
            import uvc as uvc_module
        usb_location = (
            _usb_location_from_video_device(video_device) if video_device else None
        )
        device = select_device(
            uvc_module.device_list(), vendor_id, product_id, device_uid,
            usb_location,
        )
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
