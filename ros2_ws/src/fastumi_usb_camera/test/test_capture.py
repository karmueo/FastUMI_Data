"""验证 UVC 设备选择、模式协商、坏帧跳过和异常释放。"""

import threading

import pytest

from fastumi_usb_camera import capture
from fastumi_usb_camera.capture import UvcCamera, select_device


# 测试使用的 USB 设备标识和 MJPEG 模式。
DEVICE = {"uid": "camera-1", "idVendor": "0x1bcf", "idProduct": "0x28c4"}
MODE = (1280, 960, 30)


class FakeCapture:
    """模拟 pyuvc Capture 的模式协商、读取和设备释放。"""

    def __init__(self, uid):
        """记录打开的设备并准备帧队列。"""
        self.uid = uid
        self.available_modes = [MODE]
        self.frame_mode = None
        self.frames = []
        self.closed = False

    def get_frame(self, timeout=0.1):
        """返回队列中的一帧，空队列时模拟读取超时。"""
        if self.frames:
            frame = self.frames.pop(0)
            if isinstance(frame, Exception):
                raise frame
            return frame
        raise TimeoutError

    def close(self):
        """记录设备句柄释放。"""
        self.closed = True


class FakeUvc:
    """模拟 pyuvc 模块提供的设备枚举及 Capture 构造。"""

    def __init__(self, devices=None):
        """设置可枚举设备和捕获实例。"""
        self.devices = [DEVICE] if devices is None else devices
        self.capture = None

    def device_list(self):
        """返回当前测试的 USB 设备。"""
        return self.devices

    def Capture(self, uid):
        """创建并记录模拟采集对象。"""
        self.capture = FakeCapture(uid)
        return self.capture


def make_camera(uvc):
    """使用参考相机默认参数创建待测采集对象。"""
    return UvcCamera(
        vendor_id=0x1BCF, product_id=0x28C4,
        width=1280, height=960, fps=30, uvc_module=uvc,
    )


def test_select_device_reports_missing_and_duplicate_candidates():
    """缺失和重复设备都给出精确的候选设备信息。"""
    assert select_device([DEVICE], 0x1BCF, 0x28C4) == DEVICE
    with pytest.raises(RuntimeError, match="候选设备.*1bcf:28c4"):
        select_device([DEVICE], 0x1234, 0x5678)
    with pytest.raises(RuntimeError, match="找到多个.*camera-2"):
        select_device([DEVICE, {**DEVICE, "uid": "camera-2"}], 0x1BCF, 0x28C4)


def test_select_device_uses_uid_for_identical_cameras():
    """同 VID/PID 的相机按 UID 精确选择，不匹配时保留候选提示。"""
    devices = [DEVICE, {**DEVICE, "uid": "camera-2"}]
    assert select_device(devices, 0x1BCF, 0x28C4, "camera-2") == devices[1]
    with pytest.raises(RuntimeError, match="device_uid=missing.*camera-2"):
        select_device(devices, 0x1BCF, 0x28C4, "missing")
    uvc = FakeUvc(devices)
    camera = UvcCamera(
        vendor_id=0x1BCF, product_id=0x28C4, device_uid="camera-2",
        width=1280, height=960, fps=30, uvc_module=uvc,
    )
    assert uvc.capture.uid == "camera-2"
    camera.close()


def test_video_device_resolves_usb_bus_and_address(tmp_path):
    """通过视频节点的 sysfs 父设备定位当前 USB 地址。"""
    device_root = tmp_path / "dev"
    device_root.mkdir()
    video = device_root / "video0"
    video.touch()
    alias = device_root / "camera-front"
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
        str(alias), sysfs_root, device_root
    ) == (1, 17)
    with pytest.raises(ValueError, match="不存在"):
        capture._usb_location_from_video_device(
            str(device_root / "video9"), sysfs_root, device_root
        )


def test_video_device_selects_matching_pyuvc_device(monkeypatch):
    """相同 VID/PID 的相机按视频节点对应的 USB 地址选择。"""
    devices = [
        {**DEVICE, "uid": "1:17", "bus_number": 1, "device_address": 17},
        {**DEVICE, "uid": "1:14", "bus_number": 1, "device_address": 14},
    ]
    monkeypatch.setattr(
        capture, "_usb_location_from_video_device", lambda _: (1, 17)
    )
    uvc = FakeUvc(devices)
    camera = UvcCamera(
        video_device="/dev/video0",
        width=1280, height=960, fps=30, uvc_module=uvc,
    )
    assert uvc.capture.uid == "1:17"
    camera.close()
    assert select_device(devices, usb_location=(1, 14)) == devices[1]
    with pytest.raises(ValueError, match="只能指定其中一个"):
        UvcCamera(
            vendor_id=0x1BCF, product_id=0x28C4, device_uid="1:17",
            video_device="/dev/video0", width=1280, height=960, fps=30,
            uvc_module=uvc,
        )


def test_unsupported_mode_releases_device():
    """请求模式不可用时列出可用模式并关闭已打开设备。"""
    uvc = FakeUvc()
    with pytest.raises(RuntimeError, match="可用模式.*1280"):
        UvcCamera(
            vendor_id=0x1BCF, product_id=0x28C4,
            width=640, height=480, fps=60, uvc_module=uvc,
        )
    assert uvc.capture.closed


def test_capture_skips_incomplete_and_reports_failure():
    """只递送完整帧，设备读取失败后可安全停止和释放。"""
    uvc = FakeUvc()
    camera = make_camera(uvc)
    assert uvc.capture.frame_mode == MODE
    incomplete = type("Frame", (), {
        "data_fully_received": False, "jpeg_buffer": b"bad"
    })()
    complete = type("Frame", (), {
        "data_fully_received": True, "jpeg_buffer": b"\xff\xd8jpeg\xff\xd9"
    })()
    uvc.capture.frames = [incomplete, complete, OSError("disconnected")]
    received = []
    ready = threading.Event()

    def on_frame(jpeg):
        """保存回调收到的完整 JPEG 帧。"""
        received.append(jpeg)
        ready.set()

    camera.start(on_frame)
    assert ready.wait(1.0)
    assert camera._stop.wait(1.0)
    assert camera.incomplete_count == 1
    assert received == [complete.jpeg_buffer]
    assert isinstance(camera.last_error, OSError)
    camera.close()
    assert uvc.capture.closed


def test_close_before_streaming_releases_device():
    """初始化成功但尚未开始读帧时也可安全释放设备。"""
    uvc = FakeUvc()
    camera = make_camera(uvc)
    camera.close()
    assert uvc.capture.closed
