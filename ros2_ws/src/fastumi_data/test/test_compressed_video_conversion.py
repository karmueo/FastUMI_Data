"""用合成 MCAP 验证 JPEG/PNG/H.264 会话到 RGB HDF5 的转换。"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from fastumi_data.mcap_converter import (
    McapEpisodeConverter,
    _load_processing_document,
)
from fastumi_data.models import ProcessingConfig
from fastumi_data.video_decoding import VideoDecodeError
from video_bag_builder import (
    IMAGE_TOPIC,
    TOPICS,
    write_extrinsic,
    write_session_bag,
)
from video_fixtures import (
    FRAME_PERIOD_NS,
    band_column,
    ffmpeg_message,
    frame_pixels_bgr,
    stamp_of,
)


def _convert(tmp_path: Path, mode: str, **bag_options):
    """写出合成会话并转换，返回 (摘要, 输出目录)。"""
    bag = tmp_path / "raw" / "bag"
    write_session_bag(bag, mode, **bag_options)
    converter = McapEpisodeConverter(
        str(bag),
        str(tmp_path),
        ProcessingConfig(image_mode=mode, minimum_samples=5),
        dict(TOPICS),
        str(write_extrinsic(tmp_path / "tracker_to_tcp.yaml")),
        force=False,
    )
    return converter.convert(), tmp_path


@pytest.mark.parametrize("mode", ["compressed", "h264"])
def test_encoded_session_converts_to_rgb_hdf5(tmp_path: Path, mode: str) -> None:
    """压缩视频转换为 RGB HDF5，帧数和色带位置与源帧一致。"""
    summary, output = _convert(
        tmp_path, mode, frames=40, episodes=[(2, 32)], gop_size=10
    )

    assert summary["converted"] == 1 and summary["rejected"] == 0
    assert summary["image_mode"] == mode
    assert summary["image_frames_decoded"] == 40
    assert summary["image_packets_skipped_before_keyframe"] == 0
    with h5py.File(output / "episodes" / "episode_0000.hdf5", "r") as root:
        images = root["observations/images/front"][:]
        timestamps = root["observations/timestamp_ns"][:]
    assert images.shape[1:] == (48, 64, 3)
    # 采样网格 20 Hz、时长 1 s，约 20 个样本。
    assert 18 <= len(images) <= 21
    # 底色 (B,G,R)=(200,40,40) 在 RGB 输出中应为 (R,G,B)=(40,40,200)。
    corner = images[0][0, -1].astype(int)
    assert corner[2] > 150 and corner[0] < 100
    for sample, stamp in zip(images, timestamps):
        source = int(round((int(stamp) - stamp_of(0)) / FRAME_PERIOD_NS))
        expected = band_column(frame_pixels_bgr(source))
        assert abs(band_column(sample[:, :, ::-1]) - expected) <= 5


def test_h264_state_spans_episodes_with_mid_gop_second_start(tmp_path: Path) -> None:
    """第二个 episode 从 GOP 中段开始仍可解码，因为参考包持续输入。"""
    summary, output = _convert(
        tmp_path, "h264", frames=70, episodes=[(1, 25), (33, 63)], gop_size=10
    )

    assert summary["converted"] == 2
    for name in ("episode_0000", "episode_0001"):
        with h5py.File(output / "episodes" / f"{name}.hdf5", "r") as root:
            assert root["observations/images/front"].shape[0] >= 10


def test_h264_prefix_before_first_keyframe_is_skipped_and_reported(
    tmp_path: Path,
) -> None:
    """文件从 GOP 中段开始、episode 在首个关键帧之后：前缀被跳过并报告。"""
    summary, _ = _convert(
        tmp_path, "h264", frames=50, episodes=[(14, 44)], gop_size=10,
        drop_images=range(0, 4),
    )

    assert summary["converted"] == 1
    assert summary["image_packets_skipped_before_keyframe"] == 6
    assert summary["image_frames_decoded"] == 40


def test_h264_episode_started_before_first_keyframe_fails(tmp_path: Path) -> None:
    """episode 内出现关键帧之前的包时明确失败，而不是悄悄缺帧。"""
    with pytest.raises(VideoDecodeError, match="首个 H.264 关键帧之前"):
        _convert(
            tmp_path, "h264", frames=50, episodes=[(5, 40)], gop_size=10,
            drop_images=range(0, 4),
        )


def test_corrupt_h264_packet_inside_episode_fails(tmp_path: Path) -> None:
    """episode 内的损坏包使转换失败并说明 bag 时间。"""
    def corrupt(index, message):
        if index == 12:
            data = bytes(message.data)
            message = ffmpeg_message(index, data[:12] + bytes(len(data) - 12))
        return message

    with pytest.raises(VideoDecodeError, match="解码失败.*bag 时间"):
        _convert(
            tmp_path, "h264", frames=40, episodes=[(2, 32)], gop_size=10,
            image_override=corrupt,
        )


def test_h264_stream_with_b_frames_fails(tmp_path: Path) -> None:
    """含 B 帧的码流报告原因并失败。"""
    with pytest.raises(VideoDecodeError, match="B 帧"):
        _convert(
            tmp_path, "h264", frames=30, episodes=[(2, 25)], b_frames=2,
        )


def test_corrupt_jpeg_fails(tmp_path: Path) -> None:
    """损坏的 JPEG 使转换失败。"""
    def corrupt(index, message):
        if index == 5:
            message.data = bytes(message.data)[:30]
        return message

    with pytest.raises(VideoDecodeError, match="解码失败"):
        _convert(
            tmp_path, "compressed", frames=30, episodes=[(1, 25)],
            image_override=corrupt,
        )


@pytest.mark.parametrize(
    "configured, registered",
    [
        ("h264", "sensor_msgs/msg/CompressedImage"),
        ("raw", "sensor_msgs/msg/CompressedImage"),
        ("compressed", "ffmpeg_image_transport_msgs/msg/FFMPEGPacket"),
    ],
)
def test_configuration_must_match_bag_message_type(
    tmp_path: Path, configured: str, registered: str
) -> None:
    """processing.yaml 的图像模式与 MCAP 话题类型不一致时明确失败。"""
    source_mode = "compressed" if "Compressed" in registered else "h264"
    bag = tmp_path / "raw" / "bag"
    write_session_bag(bag, source_mode, frames=20, episodes=[(1, 15)])
    converter = McapEpisodeConverter(
        str(bag), str(tmp_path),
        ProcessingConfig(image_mode=configured, minimum_samples=5),
        dict(TOPICS), str(write_extrinsic(tmp_path / "x.yaml")), force=False,
    )

    with pytest.raises(ValueError, match="image_mode"):
        converter.convert()


def test_processing_document_reads_image_mode(tmp_path: Path) -> None:
    """processing.yaml 的 image_mode 缺省为 raw，非法取值被拒绝。"""
    base = {
        "topics": {
            "image": IMAGE_TOPIC, "tracker_pose": "/p", "gripper_state": "/g",
            "episode_event": "/e", "tracker_status": "/s",
        },
    }
    path = tmp_path / "processing.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    assert _load_processing_document(str(path))[0].image_mode == "raw"
    path.write_text(yaml.safe_dump({**base, "image_mode": "h264"}), encoding="utf-8")
    assert _load_processing_document(str(path))[0].image_mode == "h264"
    path.write_text(yaml.safe_dump({**base, "image_mode": "mjpeg"}), encoding="utf-8")
    with pytest.raises(ValueError, match="image_mode"):
        _load_processing_document(str(path))
