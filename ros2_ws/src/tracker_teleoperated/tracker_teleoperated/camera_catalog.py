"""后台枚举稳定的 USB 物理端口视频入口。"""

import fcntl
import os
from pathlib import Path
import re
import struct

from fastumi_usb_camera.capture import (
    _usb_location_from_video_device,
    physical_port_label,
)


def scan_devices(by_path_root=Path('/dev/v4l/by-path'),
                 sysfs_root=Path('/sys/class/video4linux'),
                 device_root=Path('/dev')):
    """读取 by-path 主视频入口的 QUERYCAP，并合并同一节点的重复别名。"""
    devices = []
    aliases = {}
    for path in by_path_root.glob('*-video-index0'):
        if not re.fullmatch(r'.+-video-index0', path.name):
            continue
        try:
            target = path.resolve(strict=True)
            aliases.setdefault(target, []).append(path)
        except (OSError, RuntimeError) as error:
            devices.append({
                'device': str(path), 'name': '', 'physical_port': physical_port_label(str(path)),
                'kernel_device': '', 'error': str(error),
            })
    for target, paths in sorted(aliases.items(), key=lambda item: str(item[0])):
        # udev 可能同时生成 usb 与 usbv2 链接；优先使用更通用的无版本名称。
        path = min(paths, key=lambda value: ('-usbv' in value.name, len(value.name), value.name))
        try:
            _usb_location_from_video_device(
                str(path), sysfs_root, device_root, by_path_root)
            # v4l2_capability 固定 104 字节，QUERYCAP 不申请采集缓冲区。
            capability = bytearray(104)
            descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            try:
                fcntl.ioctl(descriptor, 0x80685600, capability, True)
            finally:
                os.close(descriptor)
            caps, device_caps = struct.unpack_from('II', capability, 84)
            caps = device_caps if caps & 0x80000000 else caps
            if not caps & (0x1 | 0x1000):
                continue
            name = capability[16:48].split(b'\0', 1)[0].decode(errors='replace')
            devices.append({
                'device': str(path), 'name': name,
                'physical_port': physical_port_label(str(path)),
                'kernel_device': str(target), 'error': '',
            })
        except (OSError, RuntimeError, ValueError) as error:
            devices.append({
                'device': str(path), 'name': '',
                'physical_port': physical_port_label(str(path)),
                'kernel_device': str(target), 'error': str(error),
            })
    def port_key(item):
        """把点分隔端口链转换为自然数字顺序。"""
        return tuple(int(part) for part in item['physical_port'].split('.')
                     if part.isdigit()), item['device']

    return sorted(devices, key=port_key)
