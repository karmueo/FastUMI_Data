"""验证旧连续录制命令的图像模式、话题、QoS 配置和首个 episode 门控。"""

from __future__ import annotations

import os
from pathlib import Path
import random
import sys
from unittest.mock import Mock

import pytest
import yaml

# 在导入 rclpy 前隔离 DDS，避免与真实设备或并行测试串话。
os.environ["ROS_DOMAIN_ID"] = str(random.randint(150, 232))
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

import rclpy  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

from fastumi_data.episode_manager import EpisodeManager  # noqa: E402
from fastumi_data.session_recorder import (  # noqa: E402
    DEFAULT_TOPICS,
    NON_IMAGE_TOPICS,
    _build_bag_command,
    _build_parser,
    default_topics,
    main,
    resolve_record_topics,
)


def test_default_topics_replace_the_image_topic_per_mode() -> None:
    """默认录制列表以所选图像话题开头，其余话题不变。"""
    assert DEFAULT_TOPICS[0] == "/umi_camera/image_raw"
    assert default_topics("/jetson/ffmpeg") == ["/jetson/ffmpeg", *NON_IMAGE_TOPICS]
    assert resolve_record_topics(None, "/cam/compressed")[0] == "/cam/compressed"


def test_explicit_topic_list_must_contain_the_image_topic() -> None:
    """显式话题列表缺少所选图像话题时明确失败。"""
    assert resolve_record_topics(["/cam/ffmpeg", "/gripper/state"], "/cam/ffmpeg") == [
        "/cam/ffmpeg", "/gripper/state",
    ]
    with pytest.raises(ValueError, match="必须包含所选图像话题 /cam/ffmpeg"):
        resolve_record_topics(["/umi_camera/image_raw"], "/cam/ffmpeg")


def test_parser_image_defaults_keep_raw_best_effort() -> None:
    """未指定时保持 raw、best_effort、深度 30，与既有行为一致。"""
    arguments = _build_parser().parse_args(
        ["--task", "t", "--extrinsic", "e.yaml"]
    )

    assert (arguments.image_mode, arguments.image_topic) == ("raw", "")
    assert (arguments.image_reliability, arguments.image_qos_depth) == ("best_effort", 30)


def test_bag_command_passes_qos_overrides_before_topics() -> None:
    """QoS 覆盖文件作为 rosbag 选项出现在话题列表之前。"""
    command = _build_bag_command(
        Path("/tmp/s/raw"), ["/cam"], Path("/tmp/s/qos.yaml")
    )

    assert command[command.index("--qos-profile-overrides-path") + 1] == "/tmp/s/qos.yaml"
    assert command.index("--qos-profile-overrides-path") < command.index("--topics")


def test_main_records_selected_image_topic_and_snapshots_mode(
    tmp_path: Path, monkeypatch
) -> None:
    """h264 会话共用录制与服务进程，快照与清单记录实际模式和 QoS。"""
    extrinsic = tmp_path / "tracker_to_tcp.yaml"
    processing = tmp_path / "processing.yaml"
    extrinsic.write_text("tracker_to_tcp: {}\n", encoding="utf-8")
    processing.write_text(
        yaml.safe_dump({"sample_rate_hz": 20.0, "topics": {"image": "/old"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "fastumi_data.session_recorder.load_tracker_tcp_extrinsic",
        Mock(return_value=Mock(metadata={
            "translation_rmse_mm": 1.0, "rotation_rmse_deg": 0.5,
        })),
    )
    popen = Mock(return_value=Mock())
    monkeypatch.setattr("fastumi_data.session_recorder.subprocess.Popen", popen)
    monkeypatch.setattr(
        "fastumi_data.session_recorder._wait_for_recording_processes", Mock()
    )

    main([
        "--task", "pick", "--session-id", "s", "--dataset-root", str(tmp_path),
        "--extrinsic", str(extrinsic), "--processing-config", str(processing),
        "--image-mode", "h264", "--image-topic", "/jetson/camera/ffmpeg",
        "--image-reliability", "reliable", "--image-qos-depth", "50",
    ])

    session = tmp_path / "pick" / "s"
    popen.assert_called_once()
    bag_command = popen.call_args.args[0]
    assert bag_command[:3] == [sys.executable, "-m", "fastumi_data.continuous_recorder"]
    assert bag_command[bag_command.index("--bag-uri") + 1] == str(session / "raw" / "bag")
    for fragment in (
        "image_mode:=h264", "image_topic:=/jetson/camera/ffmpeg",
        "image_reliability:=reliable", "image_qos_depth:=50",
    ):
        assert fragment in bag_command
    topics = [bag_command[i + 1] for i, arg in enumerate(bag_command) if arg == "--topic"]
    assert topics[0] == "/jetson/camera/ffmpeg" and "/umi_camera/image_raw" not in topics
    overrides = yaml.safe_load(
        (session / "image_qos_overrides.yaml")
        .read_text(encoding="utf-8")
    )
    assert overrides == {"/jetson/camera/ffmpeg": {
        "history": "keep_last", "depth": 50,
        "reliability": "reliable", "durability": "volatile",
    }}
    snapshot = yaml.safe_load(
        (session / "calibration_snapshot" / "processing.yaml").read_text("utf-8")
    )
    assert snapshot["image_mode"] == "h264"
    assert snapshot["topics"]["image"] == "/jetson/camera/ffmpeg"
    assert snapshot["sample_rate_hz"] == 20.0
    manifest = yaml.safe_load((session / "session.yaml").read_text("utf-8"))
    assert manifest["image"] == {
        "mode": "h264", "topic": "/jetson/camera/ffmpeg",
        "message_type": "ffmpeg_image_transport_msgs/msg/FFMPEGPacket",
        "reliability": "reliable", "qos_depth": 50,
    }
    assert manifest["processing_config"]["sha256"]


def test_main_rejects_explicit_topics_without_image_before_creating_session(
    tmp_path: Path,
) -> None:
    """显式话题缺少图像话题时在创建会话目录前失败。"""
    with pytest.raises(ValueError, match="必须包含所选图像话题"):
        main([
            "--task", "pick", "--session-id", "s", "--dataset-root", str(tmp_path),
            "--extrinsic", str(tmp_path / "e.yaml"),
            "--image-mode", "compressed", "--topic", "/gripper/state",
        ])

    assert not (tmp_path / "pick").exists()


# ---------------------------------------------------------------------------
# EpisodeManager 禁止独立订阅绕过录包端门控。
# ---------------------------------------------------------------------------
def _start(manager: EpisodeManager) -> Trigger.Response:
    """直接调用开始服务处理函数。"""
    return manager._start(Trigger.Request(), Trigger.Response())


@pytest.mark.parametrize("mode", ["compressed", "h264"])
def test_encoded_manager_requires_recording_readiness(mode: str) -> None:
    """编码模式缺少实际录包端状态时拒绝启动，避免独立订阅提前确认关键帧。"""
    rclpy.init(args=["--ros-args", "-p", f"image_mode:={mode}"])
    try:
        with pytest.raises(ValueError, match="录包端.*record_session"):
            EpisodeManager()
    finally:
        rclpy.shutdown()


def test_raw_mode_is_not_gated() -> None:
    """默认 raw 不订阅图像，行为与原来一致。"""
    rclpy.init()
    manager = EpisodeManager()
    try:
        assert _start(manager).success
        assert not hasattr(manager, "_image_subscription")
    finally:
        manager.destroy_node()
        rclpy.shutdown()
