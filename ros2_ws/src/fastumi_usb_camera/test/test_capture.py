"""验证供 launch 和遥操面板使用的 V4L2 物理路径工具。"""

import pytest

from fastumi_usb_camera import capture


# 测试使用的稳定主视频设备路径。
PHYSICAL_DEVICE = "/dev/v4l/by-path/pci-test-usb-0:2.4:1.0-video-index0"


def test_video_device_resolves_usb_bus_and_address(tmp_path):
    """通过视频节点的 sysfs 祖先目录定位 USB 总线号和设备地址。"""
    device_root = tmp_path / "dev"
    device_root.mkdir()
    video = device_root / "video0"
    video.touch()
    by_path_root = device_root / "v4l" / "by-path"
    by_path_root.mkdir(parents=True)
    alias = by_path_root / "pci-test-usb-0:2.4:1.0-video-index0"
    alias.symlink_to(video)
    usb_device = tmp_path / "sys" / "devices" / "1-1.4"
    interface = usb_device / "1-1.4:1.0"
    interface.mkdir(parents=True)
    (usb_device / "busnum").write_text("1\n")
    (usb_device / "devnum").write_text("17\n")
    sysfs_root = tmp_path / "sys" / "class" / "video4linux"
    video_class = sysfs_root / "video0"
    video_class.mkdir(parents=True)
    (video_class / "device").symlink_to(interface)

    assert capture._usb_location_from_video_device(
        str(alias), sysfs_root, device_root, by_path_root
    ) == (1, 17)
    with pytest.raises(ValueError, match="by-path"):
        capture._usb_location_from_video_device(
            str(device_root / "video9"), sysfs_root, device_root, by_path_root
        )
    broken = by_path_root / "pci-test-usb-0:2.5:1.0-video-index0"
    broken.symlink_to(device_root / "video9")
    with pytest.raises(ValueError, match="不存在"):
        capture._usb_location_from_video_device(
            str(broken), sysfs_root, device_root, by_path_root
        )


def test_physical_path_validation_and_port_label():
    """只接受主视频 by-path，并提取 Hub 端口链。"""
    assert capture.is_physical_video_device_path(PHYSICAL_DEVICE)
    assert capture.physical_port_label(PHYSICAL_DEVICE) == "2.4"
    assert not capture.is_physical_video_device_path("/dev/video0")
    assert not capture.is_physical_video_device_path(
        "/dev/v4l/by-path/pci-test-usb-0:2.4:1.0-video-index1"
    )
