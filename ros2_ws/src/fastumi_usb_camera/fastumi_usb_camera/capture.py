"""提供稳定 V4L2 物理路径校验和 USB 端口展示工具。"""

from pathlib import Path
import re


# udev 创建稳定视频设备链接的标准目录。
BY_PATH_ROOT = Path("/dev/v4l/by-path")
# FastUMI 只接受每个 UVC 接口的主视频节点。
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


def _usb_location_from_video_device(
    video_device: str, sysfs_root: Path = Path("/sys/class/video4linux"),
    device_root: Path = Path("/dev"), by_path_root: Path = BY_PATH_ROOT,
) -> tuple[int, int]:
    """校验稳定物理路径，并沿 sysfs 查找 USB 总线号和设备地址。"""
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
            return (
                int((parent / "busnum").read_text().strip()),
                int((parent / "devnum").read_text().strip()),
            )
        except (OSError, ValueError):
            continue
    raise ValueError(f"{video_device} 未关联 USB 设备")
