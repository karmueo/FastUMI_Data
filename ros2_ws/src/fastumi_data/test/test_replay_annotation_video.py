"""验证回放补标对 JPEG/PNG 与 H.264 的图像话题检测、重映射和解码命令。"""

from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from fastumi_data.replay_annotation import (
    ImageSource,
    _build_parser,
    _wait_for_playback,
    build_decoder_command,
    build_player_command,
    detect_image_source,
    detect_image_topic,
    replay_encoded_topic,
)

# 三种图像消息类型。
RAW = "sensor_msgs/msg/Image"
JPEG = "sensor_msgs/msg/CompressedImage"
H264 = "ffmpeg_image_transport_msgs/msg/FFMPEGPacket"


def _metadata(monkeypatch, topics: dict) -> None:
    """让源包元数据返回指定的话题与类型。"""
    monkeypatch.setattr(
        "fastumi_data.replay_annotation.source_topics",
        lambda uri: {n: SimpleNamespace(type=t) for n, t in topics.items()},
    )


def test_single_encoded_topic_is_detected_with_its_mode(monkeypatch) -> None:
    """只有一个图像话题时按其消息类型得到模式。"""
    for topic, type_name, mode in (
        ("/cam/image_raw/compressed", JPEG, "compressed"),
        ("/cam/image_raw/ffmpeg", H264, "h264"),
        ("/cam/image_raw", RAW, "raw"),
    ):
        _metadata(monkeypatch, {topic: type_name, "/gripper/state": "x/msg/Y"})
        assert detect_image_source(Path("bag")) == ImageSource(topic, mode)
        assert detect_image_topic(Path("bag")) == topic


def test_requested_mode_filters_candidates(monkeypatch) -> None:
    """多个图像话题时按 --image-mode 限定后可唯一确定，否则要求显式选择。"""
    _metadata(monkeypatch, {"/a": RAW, "/b/compressed": JPEG})
    with pytest.raises(ValueError, match="唯一图像话题"):
        detect_image_source(Path("bag"))
    assert detect_image_source(Path("bag"), requested_mode="compressed") == ImageSource(
        "/b/compressed", "compressed"
    )
    assert detect_image_source(Path("bag"), "/a") == ImageSource("/a", "raw")
    with pytest.raises(ValueError, match="--image-topic"):
        detect_image_source(Path("bag"), "/a", "h264")
    with pytest.raises(ValueError, match="--image-mode"):
        detect_image_source(Path("bag"), requested_mode="mjpeg")


def test_xv_raw_topic_is_still_preferred(monkeypatch) -> None:
    """存在唯一 XV RGB 原始图像时仍优先，不受压缩调试话题影响。"""
    _metadata(monkeypatch, {"/xv_sdk/SN/rgb/image": RAW, "/dbg/compressed": JPEG})

    assert detect_image_source(Path("bag")) == ImageSource("/xv_sdk/SN/rgb/image", "raw")


def test_player_remaps_by_mode() -> None:
    """raw 直达 RViz 话题；编码图像重映射到回放编码话题等待解码。"""
    raw = build_player_command(Path("bag"), "/cam", 1.0)
    jpeg = build_player_command(Path("bag"), "/cam/compressed", 1.0, "compressed")
    h264 = build_player_command(Path("bag"), "/cam/ffmpeg", 2.0, "h264")

    assert "/cam:=/fastumi/replay/image" in raw
    assert "/cam/compressed:=/fastumi/replay/encoded/compressed" in jpeg
    assert "/cam/ffmpeg:=/fastumi/replay/encoded/ffmpeg" in h264
    assert replay_encoded_topic("h264") == "/fastumi/replay/encoded/ffmpeg"


def test_decoder_commands_publish_the_rviz_topic() -> None:
    """JPEG 用自带 QoS 的解码节点，H.264 用接收器；两者输出到 RViz 订阅的话题。"""
    assert build_decoder_command("raw") is None
    jpeg = build_decoder_command("compressed")
    assert jpeg[:4] == ["ros2", "run", "fastumi_data", "image_decoder"]
    assert "input_topic:=/fastumi/replay/encoded/compressed" in jpeg
    assert "output_topic:=/fastumi/replay/image" in jpeg

    h264 = build_decoder_command("h264", "auto")
    assert h264[2:4] == ["fastumi_usb_camera", "usb_camera_receiver"]
    assert "input_topic:=/fastumi/replay/encoded" in h264
    assert "output_topic:=/fastumi/replay/image" in h264
    assert "decoder_backend:=auto" in h264
    assert "fastumi.replay.encoded.ffmpeg.decoders.h264:=h264" in h264
    with pytest.raises(ValueError, match="decoder-backend"):
        build_decoder_command("h264", "tpu")


def test_parser_defaults_and_choices() -> None:
    """默认自动检测模式、auto 后端；非法取值被拒绝。"""
    arguments = _build_parser().parse_args(["--bag", "a", "--output", "b"])

    assert (arguments.image_mode, arguments.decoder_backend) == ("auto", "auto")
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--bag", "a", "--output", "b", "--image-mode", "x"])


def test_decoder_exit_during_playback_fails_explicitly() -> None:
    """回放期间解码节点退出会明确失败，而不是留下黑屏继续标注。"""
    class Process:
        """可设置退出码的进程替身。"""

        def __init__(self, code=None):
            """保存退出码。"""
            self.returncode = code

        def poll(self):
            """返回退出码。"""
            return self.returncode

    with pytest.raises(RuntimeError, match="解码节点.*退出码为 3"):
        _wait_for_playback(Process(), Process(), threading.Event(), Process(3))
