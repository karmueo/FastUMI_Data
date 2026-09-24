"""采集质量检查及交互处理的 MCAP 集成测试。"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
from nav_msgs.msg import Odometry
from rclpy.serialization import serialize_message
import rosbag2_py
from rm_ros_interfaces.msg import Jointpos
from sensor_msgs.msg import CompressedImage, JointState

from audit_hardware_mcap import (
    Thresholds, audit_episode, discover_episodes, main, process_abnormal,
    scan_episodes, select_abnormal,
)


BASE_NS = 1_800_000_000_000_000_000
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


def _stamp(message, timestamp_ns: int) -> None:
    message.header.stamp.sec = timestamp_ns // 1_000_000_000
    message.header.stamp.nanosec = timestamp_ns % 1_000_000_000


def _h264_packets(tmp_path: Path) -> list[tuple[bytes, bool]]:
    """生成 12 个含两个 IDR 的真实 H.264 access unit。"""
    frames = b"".join(np.full((32, 32, 3), index * 20, dtype=np.uint8).tobytes()
                      for index in range(12))
    bitstream = tmp_path / "frames.h264"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
         "-video_size", "32x32", "-framerate", "30", "-i", "pipe:0",
         "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
         "-g", "10", "-bf", "0",
         "-x264-params", "keyint=10:min-keyint=10:scenecut=0:repeat-headers=1",
         "-f", "h264", str(bitstream)],
        input=frames, capture_output=True, check=True,
    )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-f", "h264", "-show_packets",
         "-show_entries", "packet=pos,size,flags", "-of", "csv=p=0",
         str(bitstream)], capture_output=True, text=True, check=True,
    )
    encoded = bitstream.read_bytes()
    packets = []
    for line in probe.stdout.splitlines():
        match = re.fullmatch(r"(\d+),(\d+),([A-Z_]+)", line)
        if match:
            size, offset, flags = match.groups()
            packets.append((encoded[int(offset):int(offset) + int(size)], "K" in flags))
    assert len(packets) == 12
    return packets


def _episode(root: Path, number: int = 0, *, mode: str = "jpeg",
             camera_offsets: tuple[int, ...] = (0, 50, 100, 150, 200),
             joint_offsets: tuple[int, ...] = (0, 50, 100, 150, 200),
             action_offsets: tuple[int, ...] | None = None,
             camera_header_offsets: tuple[int, ...] | None = None,
             joint_header_offsets: tuple[int, ...] | None = None,
             bad_joint: bool = False, bad_action: bool = False,
             bad_tracker: bool = False, corrupt_jpeg: bool = False,
             corrupt_h264: bool = False) -> Path:
    path = root / f"episode_{number}"
    path.mkdir(parents=True)
    camera_topic = ("/wrist_camera/image_raw/ffmpeg" if mode == "h264"
                    else "/wrist_camera/image_raw/compressed")
    topics = {
        "/joint_states": "sensor_msgs/msg/JointState",
        "/rm_driver/movej_canfd_cmd": "rm_ros_interfaces/msg/Jointpos",
        camera_topic: ("ffmpeg_image_transport_msgs/msg/FFMPEGPacket" if mode == "h264"
                       else "sensor_msgs/msg/CompressedImage"),
        "/vive_tracker/odom": "nav_msgs/msg/Odometry",
    }
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path / "bag"), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    for topic, kind in topics.items():
        writer.create_topic(rosbag2_py.TopicMetadata(
            name=topic, type=kind, serialization_format="cdr"))
    records = []
    joint_headers = joint_header_offsets or joint_offsets
    for index, (offset, header_offset) in enumerate(zip(joint_offsets, joint_headers)):
        positions = [0.0] * 7
        if bad_joint and index == 2:
            positions[3] = float("nan")
        state = JointState(name=JOINT_NAMES, position=positions)
        _stamp(state, BASE_NS + header_offset * 1_000_000)
        records.append((offset, "/joint_states", state))
    for index, offset in enumerate(action_offsets if action_offsets is not None
                                   else joint_offsets):
        records.append((offset, "/rm_driver/movej_canfd_cmd",
                        Jointpos(dof=7, joint=([float("nan")] + [0.0] * 6
                                               if bad_action and index == 2
                                               else [0.0] * 7))))
    camera_headers = camera_header_offsets or camera_offsets
    packets = _h264_packets(root) if mode == "h264" else None
    for index, (offset, header_offset) in enumerate(zip(camera_offsets, camera_headers)):
        if mode == "h264":
            assert packets is not None
            payload, keyframe = packets[index]
            if corrupt_h264 and index == 1:
                payload = b"\x00\x00\x00\x01\x09\xf0"
            image = FFMPEGPacket(
                encoding="h264;yuv420p;bgr8;bgr8", width=32, height=32,
                pts=BASE_NS + header_offset * 1_000_000,
                flags=1 if keyframe else 0, data=payload)
        else:
            image = CompressedImage(format="jpeg", data=b"\xff\xd8\xff\xd9")
            # 替换为真实 JPEG，确保 imdecode 检查通过。
            success, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
            assert success
            image.data = (b"\xff\xd8invalid\xff\xd9" if corrupt_jpeg and index == 2
                          else encoded.tobytes())
        _stamp(image, BASE_NS + header_offset * 1_000_000)
        records.append((offset, camera_topic, image))
    tracker = Odometry()
    tracker.pose.pose.position.x = float("nan") if bad_tracker else 0.0
    records.append((0, "/vive_tracker/odom", tracker))
    for offset, topic, message in sorted(records, key=lambda item: item[0]):
        writer.write(topic, serialize_message(message), BASE_NS + offset * 1_000_000)
    del writer
    (path / "recording.json").write_text(json.dumps({"recording_id": "synthetic"}))
    return path


def _codes(path: Path, thresholds: Thresholds = Thresholds()) -> set[str]:
    return {finding.code for finding in audit_episode(path, thresholds).findings}


def test_normal_episode_and_tracker_values_are_ignored(tmp_path: Path) -> None:
    path = _episode(tmp_path, bad_tracker=True)
    assert _codes(path) == set()


def test_timestamps_and_joint_format_are_reported(tmp_path: Path) -> None:
    path = _episode(
        tmp_path, camera_offsets=(0, 50, 100, 350, 400),
        camera_header_offsets=(0, 50, 100, 50, 400),
        joint_header_offsets=(0, 50, 100, 25, 200), bad_joint=True)
    codes = _codes(path)
    assert {"camera_bag_gap", "camera_source_backward", "joint_state_source_backward",
            "joint_state_invalid"} <= codes
    assert "camera_bag_gap" not in _codes(path, Thresholds(camera_ms=300))


def test_joint_action_gap_is_ignored_but_invalid_data_is_reported(tmp_path: Path) -> None:
    path = _episode(tmp_path, joint_offsets=(0, 50, 100, 350, 400),
                    bad_action=True, corrupt_jpeg=True)
    assert {"joint_state_bag_gap", "joint_action_invalid",
            "camera_invalid"} <= _codes(path)
    assert "joint_action_bag_gap" not in _codes(path)


def test_joint_action_gap_alone_does_not_mark_episode_abnormal(
        tmp_path: Path, monkeypatch) -> None:
    path = _episode(tmp_path, action_offsets=(0, 50, 100, 350, 400))
    assert _codes(path) == set()
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    assert main(["--input", str(tmp_path), "--joint-action-gap-ms", "200"]) == 0


def test_missing_files_and_failed_episode_do_not_stop_discovery(tmp_path: Path) -> None:
    good = _episode(tmp_path, 0)
    missing = tmp_path / "episode_1"
    missing.mkdir()
    quarantined = tmp_path / ".quarantine" / "episode_2"
    quarantined.mkdir(parents=True)
    assert discover_episodes(tmp_path) == [good, missing]
    assert "missing_bag" in _codes(missing)
    assert _codes(good) == set()


def test_h264_decoder_detects_missing_frame(tmp_path: Path) -> None:
    offsets = tuple(index * 33 for index in range(12))
    good = _episode(tmp_path, 0, mode="h264", camera_offsets=offsets,
                    joint_offsets=offsets)
    bad = _episode(tmp_path, 1, mode="h264", camera_offsets=offsets,
                   joint_offsets=offsets, corrupt_h264=True)
    assert _codes(good) == set()
    assert "camera_frame_mismatch" in _codes(bad)


def test_parallel_scan_matches_sequential_results(tmp_path: Path) -> None:
    episodes = [
        _episode(tmp_path, 0, bad_tracker=True),
        _episode(tmp_path, 1, camera_offsets=(0, 50, 100, 350, 400)),
        tmp_path / "episode_2",
    ]
    episodes[-1].mkdir()
    thresholds = Thresholds()
    serial = dict(scan_episodes(episodes, thresholds, 1))
    parallel = dict(scan_episodes(episodes, thresholds, 2))
    assert {index: report.findings for index, report in parallel.items()} == {
        index: report.findings for index, report in serial.items()}
    assert {index: report.counts for index, report in parallel.items()} == {
        index: report.counts for index, report in serial.items()}
    assert {index: report.snapshot for index, report in parallel.items()} == {
        index: report.snapshot for index, report in serial.items()}


@pytest.mark.parametrize("choice,confirmation,expected", [
    ("n", None, "unchanged"),
    ("m", None, "moved"),
    ("d", "wrong", "unchanged"),
    ("d", "DELETE 1", "deleted"),
])
def test_cli_handles_all_abnormal_together(tmp_path: Path, monkeypatch, choice,
                                          confirmation, expected) -> None:
    root = tmp_path / "collection"
    task = root / "rm75" / "task"
    normal = _episode(task, 0)
    abnormal = _episode(task, 1, camera_offsets=(0, 50, 100, 350, 400))
    answers = iter(["a", choice] + ([confirmation] if confirmation is not None else []))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    assert main(["--input", str(root), "--workers", "1"]) == (
        0 if expected != "unchanged" else 1)
    assert normal.is_dir()
    assert abnormal.exists() == (expected == "unchanged")
    quarantine = root / ".quarantine" / "rm75" / "task" / "episode_1"
    assert quarantine.exists() == (expected == "moved")


def test_cli_moves_only_camera_abnormal_episodes(tmp_path: Path, monkeypatch) -> None:
    camera = _episode(tmp_path, 0, camera_offsets=(0, 50, 100, 350, 400))
    joint = _episode(tmp_path, 1, joint_offsets=(0, 50, 100, 350, 400))
    both = _episode(tmp_path, 2, camera_offsets=(0, 50, 100, 350, 400),
                    joint_offsets=(0, 50, 100, 350, 400))
    answers = iter(["c", "m"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    assert main(["--input", str(tmp_path), "--workers", "2"]) == 0
    assert joint.is_dir()
    for episode in (camera, both):
        assert not episode.exists()
        assert (tmp_path / ".quarantine" / episode.name).is_dir()


def test_cli_deletes_only_specified_abnormal_episode(tmp_path: Path, monkeypatch) -> None:
    first = _episode(tmp_path, 0, camera_offsets=(0, 50, 100, 350, 400))
    second = _episode(tmp_path, 1, joint_offsets=(0, 50, 100, 350, 400))
    answers = iter(["s", "episode_1", "d", "DELETE 1"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    assert main(["--input", str(tmp_path), "--workers", "1"]) == 0
    assert first.is_dir()
    assert not second.exists()


def test_select_abnormal_rejects_normal_and_ambiguous_names(tmp_path: Path) -> None:
    first = _episode(tmp_path / "task_a", 1, camera_offsets=(0, 50, 100, 350, 400))
    second = _episode(tmp_path / "task_b", 1, joint_offsets=(0, 50, 100, 350, 400))
    normal = _episode(tmp_path / "task_a", 2)
    abnormal = [audit_episode(path, Thresholds()) for path in (first, second)]
    with pytest.raises(ValueError, match="多个目录"):
        select_abnormal(tmp_path, abnormal, "s", "episode_1")
    with pytest.raises(ValueError, match="不在异常"):
        select_abnormal(tmp_path, abnormal, "s", normal.name)
    assert select_abnormal(tmp_path, abnormal, "s", "task_b/episode_1") == [abnormal[1]]
    assert select_abnormal(tmp_path, abnormal, "s",
                           "task_b/episode_1, task_a/episode_1") == abnormal


def test_noninteractive_mode_and_changed_snapshot_preserve_data(tmp_path: Path,
                                                                 monkeypatch) -> None:
    path = _episode(tmp_path, camera_offsets=(0, 50, 100, 350, 400))
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    assert main(["--input", str(tmp_path)]) == 1
    assert path.is_dir()
    report = audit_episode(path, Thresholds())
    (path / "new.txt").write_text("changed")
    with pytest.raises(RuntimeError, match="重新扫描"):
        process_abnormal(tmp_path, [report], "delete")
    assert path.is_dir()


def test_recorder_lock_and_quarantine_collision_block_all_changes(tmp_path: Path) -> None:
    first = _episode(tmp_path, 0, camera_offsets=(0, 50, 100, 350, 400))
    second = _episode(tmp_path, 1, camera_offsets=(0, 50, 100, 350, 400))
    reports = [audit_episode(path, Thresholds()) for path in (first, second)]
    lock_path = tmp_path / ".recorder.lock"
    with lock_path.open("wb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="录制器正在占用"):
            process_abnormal(tmp_path, reports, "move")
        fcntl.flock(lock, fcntl.LOCK_UN)
    collision = tmp_path / ".quarantine" / "episode_1"
    collision.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="隔离目标已存在"):
        process_abnormal(tmp_path, reports, "move")
    assert first.is_dir() and second.is_dir()
