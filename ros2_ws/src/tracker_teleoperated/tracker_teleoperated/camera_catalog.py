"""后台枚举 USB 视频采集入口，复用相机驱动的物理设备映射。"""

import fcntl
import os
from pathlib import Path
import re
import struct

from fastumi_usb_camera.capture import _usb_location_from_video_device


def scan_devices(device_root=Path('/dev'), sysfs_root=Path('/sys/class/video4linux')):
    """读取 QUERYCAP（不启动视频流），过滤元数据并按 USB 身份去重。"""
    devices = []
    identities = set()
    for path in sorted(device_root.glob('video*'), key=lambda p: int(p.name[5:])
                       if re.fullmatch(r'video\d+', p.name) else -1):
        if not re.fullmatch(r'video\d+', path.name):
            continue
        try:
            identity = _usb_location_from_video_device(str(path), sysfs_root, device_root)
            # v4l2_capability 固定 104 字节，QUERYCAP 不申请采集缓冲区。
            capability = bytearray(104)
            descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            try:
                fcntl.ioctl(descriptor, 0x80685600, capability, True)
            finally:
                os.close(descriptor)
            caps, device_caps = struct.unpack_from('II', capability, 84)
            caps = device_caps if caps & 0x80000000 else caps
            if not caps & (0x1 | 0x1000) or identity in identities:
                continue
            identities.add(identity)
            name = capability[16:48].split(b'\0', 1)[0].decode(errors='replace')
            devices.append({'device': str(path), 'name': name, 'error': ''})
        except (OSError, ValueError) as error:
            devices.append({'device': str(path), 'name': '', 'error': str(error)})
    return devices
