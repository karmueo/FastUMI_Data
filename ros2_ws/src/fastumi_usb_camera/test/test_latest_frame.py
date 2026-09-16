"""验证单帧缓存覆盖旧帧且取出后不重复返回。"""

from fastumi_usb_camera.latest_frame import CapturedFrame, LatestFrame


def test_latest_frame_overwrites_pending_without_replay():
    """积压时仅保留最新图像并正确统计覆盖次数。"""
    frames = LatestFrame()
    first = CapturedFrame(stamp=1, jpeg=b"first")
    second = CapturedFrame(stamp=2, jpeg=b"second")
    frames.put(first)
    frames.put(second)
    assert frames.take() == second
    assert frames.take() is None
    assert frames.captured_count == 2
    assert frames.overwritten_count == 1
