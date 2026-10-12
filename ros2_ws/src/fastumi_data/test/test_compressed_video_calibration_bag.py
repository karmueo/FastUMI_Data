"""验证标定读取 MCAP 时对 JPEG/PNG/H.264 的 BGR 解码、抽帧和排空。"""

from pathlib import Path

import numpy as np
import pytest

from fastumi_data.tracker_camera_bag import iter_image_frames
from fastumi_data.video_decoding import DecodeStats, VideoDecodeError
from video_bag_builder import IMAGE_TOPIC, write_session_bag
from video_fixtures import band_column, frame_pixels_bgr, ffmpeg_message, stamp_of


def _bag(tmp_path: Path, mode: str, **options) -> str:
    """写出合成会话并返回 bag 路径。"""
    path = tmp_path / "bag"
    options.setdefault("episodes", [(1, 8)])
    options.setdefault("frames", 25)
    write_session_bag(path, mode, **options)
    return str(path)


@pytest.mark.parametrize("mode", ["compressed", "h264"])
def test_frames_are_bgr_with_source_time_and_bag_time(
    tmp_path: Path, mode: str
) -> None:
    """解码帧是 BGR，Header 时间和 bag 时间分别保留。"""
    frames = list(iter_image_frames(_bag(tmp_path, mode), IMAGE_TOPIC, image_mode=mode))

    assert [f.timestamp_ns for f in frames] == [stamp_of(i) for i in range(25)]
    assert [f.bag_timestamp_ns for f in frames] == [stamp_of(i) for i in range(25)]
    corner = frames[0].image[0, -1].astype(int)
    assert corner[0] > 150 and corner[2] < 100  # 蓝色底在 BGR 的第 0 通道。
    assert abs(band_column(frames[7].image) - band_column(frame_pixels_bgr(7))) <= 2


@pytest.mark.parametrize("mode", ["compressed", "h264"])
def test_stride_applies_after_decoding(tmp_path: Path, mode: str) -> None:
    """抽帧按解码输出计数，因此参考帧不会因抽帧丢失。"""
    frames = list(
        iter_image_frames(_bag(tmp_path, mode), IMAGE_TOPIC, frame_stride=3, image_mode=mode)
    )

    assert [f.timestamp_ns for f in frames] == [stamp_of(i) for i in range(0, 25, 3)]
    for frame, index in zip(frames, range(0, 25, 3)):
        assert abs(band_column(frame.image) - band_column(frame_pixels_bgr(index))) <= 2


def test_h264_prefix_is_skipped_and_reported(tmp_path: Path) -> None:
    """文件从 GOP 中段开始时跳过前缀，并通过统计报告。"""
    stats = DecodeStats()
    path = _bag(tmp_path, "h264", drop_images=range(0, 4))

    frames = list(iter_image_frames(path, IMAGE_TOPIC, image_mode="h264", stats=stats))

    assert stats.skipped_before_keyframe == 6
    assert [f.timestamp_ns for f in frames] == [stamp_of(i) for i in range(10, 25)]
    assert stats.frames == 15


def test_mode_must_match_message_type(tmp_path: Path) -> None:
    """默认 raw 读取压缩话题，或 h264 读取 JPEG 话题，都说明类型不一致。"""
    path = _bag(tmp_path, "compressed")

    with pytest.raises(ValueError, match="CompressedImage.*image_mode=raw"):
        list(iter_image_frames(path, IMAGE_TOPIC))
    with pytest.raises(ValueError, match="image_mode=h264"):
        list(iter_image_frames(path, IMAGE_TOPIC, image_mode="h264"))


def test_non_increasing_header_time_fails(tmp_path: Path) -> None:
    """压缩图像的 Header 时间必须严格递增。"""
    def reorder(index, message):
        if index == 6:
            message.header.stamp.sec -= 5
        return message

    path = _bag(tmp_path, "compressed", image_override=reorder)

    with pytest.raises(ValueError, match="严格递增"):
        list(iter_image_frames(path, IMAGE_TOPIC, image_mode="compressed"))


def test_corrupt_h264_packet_fails_with_bag_time(tmp_path: Path) -> None:
    """损坏的 H.264 包报告话题和 bag 时间。"""
    def corrupt(index, message):
        if index == 12:
            data = bytes(message.data)
            message = ffmpeg_message(index, data[:12] + bytes(len(data) - 12))
        return message

    path = _bag(tmp_path, "h264", image_override=corrupt)

    with pytest.raises(VideoDecodeError, match=r"/camera/image.*bag 时间"):
        list(iter_image_frames(path, IMAGE_TOPIC, image_mode="h264"))


def test_raw_mode_still_works_with_default_arguments(tmp_path: Path) -> None:
    """未传 image_mode 时仍按原始 Image 读取。"""
    frames = list(iter_image_frames(_bag(tmp_path, "raw"), IMAGE_TOPIC, frame_stride=5))

    assert [f.timestamp_ns for f in frames] == [stamp_of(i) for i in range(0, 25, 5)]
    assert np.array_equal(frames[1].image, frame_pixels_bgr(5))
