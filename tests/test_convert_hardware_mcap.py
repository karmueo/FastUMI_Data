"""Integration checks for hardware MCAP conversion and timestamp alignment."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import cv2
import h5py
import numpy as np
import pytest
import rosbag2_py
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
from nav_msgs.msg import Odometry
from rclpy.serialization import serialize_message
from rm_ros_interfaces.msg import Jointpos
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float32

from convert_hardware_mcap import (
    CAMERA_TOPICS,
    TOPICS,
    ConversionError,
    convert_episode,
    discover_episodes,
)


BASE_NS = 1_800_000_000_000_000_000


def _stamp(message, timestamp_ns: int) -> None:
    message.header.stamp.sec = timestamp_ns // 1_000_000_000
    message.header.stamp.nanosec = timestamp_ns % 1_000_000_000


def _h264_packets() -> list[tuple[bytes, bool]]:
    """Encode five synthetic frames and split them at FFmpeg packet boundaries."""
    frames = b"".join(
        np.full((16, 16, 3), index * 35, dtype=np.uint8).tobytes()
        for index in range(5)
    )
    encoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "bgr24",
         "-video_size", "16x16", "-framerate", "10", "-i", "pipe:0",
         "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
         "-bf", "0", "-g", "2", "-x264-params", "keyint=2:min-keyint=2:scenecut=0:repeat-headers=1",
         "-f", "h264", "pipe:1"],
        input=frames, capture_output=True, check=True,
    ).stdout
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-f", "h264", "-show_packets",
         "-show_entries", "packet=pos,size,flags", "-of", "json", "pipe:0"],
        input=encoded, capture_output=True, check=True,
    )
    packets = json.loads(probe.stdout)["packets"]
    assert len(packets) == 5
    return [
        (encoded[int(packet["pos"]):int(packet["pos"]) + int(packet["size"])],
         "K" in packet["flags"])
        for packet in packets
    ]


def _image(mode: str, index: int, timestamp_ns: int,
           packets: list[tuple[bytes, bool]] | None):
    image = np.full((16, 16, 3), index * 35, dtype=np.uint8)
    if mode == "raw":
        message = Image(width=16, height=16, encoding="bgr8", step=48,
                        data=image.tobytes())
    elif mode == "jpeg":
        success, encoded = cv2.imencode(".jpg", image)
        assert success
        message = CompressedImage(format="jpeg", data=encoded.tobytes())
    else:
        assert packets is not None
        payload, keyframe = packets[index]
        message = FFMPEGPacket(width=16, height=16,
                               encoding="h264;yuv420p;bgr8;bgr8",
                               pts=timestamp_ns, flags=1 if keyframe else 0,
                               data=payload)
    _stamp(message, timestamp_ns)
    return message


def _make_bag(root: Path, mode: str, *, include_tracker: bool = True,
              state_offsets: tuple[int, int] = (150, 350),
              skip_first_h264_packet: bool = False,
              corrupt_jpeg_index: int | None = None) -> Path:
    episode = root / "episode_12"
    episode.mkdir(parents=True)
    bag = episode / "bag"
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {topic: ros_type for topic, ros_type, _ in TOPICS.values()}
    camera_topic = next(topic for topic, (camera_mode, _, _) in CAMERA_TOPICS.items()
                        if camera_mode == mode)
    topic_types[camera_topic] = CAMERA_TOPICS[camera_topic][1]
    if not include_tracker:
        del topic_types[TOPICS["tracker"][0]]
    for topic, ros_type in topic_types.items():
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=topic, type=ros_type, serialization_format="cdr"
            )
        )
    records = []

    def add(offset_ms: int, topic: str, message) -> None:
        if topic in topic_types:
            records.append((BASE_NS + offset_ms * 1_000_000, topic, message))

    for offset, positions in ((100, [1, 2, 3, 4, 5, 6, 7]),
                              (500, [7, 6, 5, 4, 3, 2, 1])):
        add(offset, TOPICS["joint_action"][0],
            Jointpos(joint=[float(value) for value in positions], dof=7))
    for offset in state_offsets:
        message = JointState(name=list(reversed([f"joint{i}" for i in range(1, 8)])),
                             position=[float(i) for i in range(7, 0, -1)])
        _stamp(message, BASE_NS + offset * 1_000_000)
        add(offset, TOPICS["joint_state"][0], message)
        tracker = Odometry()
        _stamp(tracker, BASE_NS + offset * 1_000_000 + 25_000_000)
        tracker.header.frame_id = "vive_tracker_odom"
        tracker.child_frame_id = "vive_tracker"
        tracker.pose.pose.position.x = offset / 1000.0
        tracker.pose.pose.orientation.w = 1.0
        add(offset, TOPICS["tracker"][0], tracker)
        add(offset, TOPICS["gripper_state"][0], Float32(data=offset / 1000.0))
    # The first command predates the joint-action window but remains active.
    add(0, TOPICS["gripper_action"][0], Float32(data=0.2))
    add(300, TOPICS["gripper_action"][0], Float32(data=0.8))
    packets = _h264_packets() if mode == "h264" else None
    for index in range(1 if skip_first_h264_packet else 0, 5):
        offset = index * 100
        timestamp_ns = BASE_NS + offset * 1_000_000
        image = _image(mode, index, timestamp_ns, packets)
        if mode == "jpeg" and index == corrupt_jpeg_index:
            image.data = b"\xff\xd8garbage\xff\xd9"
        add(offset, camera_topic, image)
    for timestamp, topic, message in sorted(records, key=lambda item: item[0]):
        writer.write(topic, serialize_message(message), timestamp)
    del writer
    (episode / "recording.json").write_text(
        json.dumps({"recording_id": "synthetic"}), encoding="utf-8"
    )
    return episode


@pytest.mark.parametrize("mode", ("raw", "jpeg", "h264"))
def test_converts_all_camera_modes_with_common_bag_clock(tmp_path: Path,
                                                         mode: str) -> None:
    source = _make_bag(tmp_path / "source", mode)
    output = tmp_path / "output"
    report = convert_episode(source, output)
    episode = output / source.name
    assert list(path.name for path in discover_episodes(source.parent)) == ["episode_12"]
    assert report["recording_id"] == "synthetic"
    assert report["bag_minus_header"]["tracker"]["median_ms"] == pytest.approx(-25)
    assert report["bag_minus_camera_pts"]["count"] == (5 if mode == "h264" else 0)
    assert report["video_fps"] == 10.0
    expected_frames = 3 if mode == "h264" else 4
    assert report["output_counts"]["video_frames"] == expected_frames
    with h5py.File(episode / "proprio.hdf5") as root:
        assert root.attrs["format_version"] == "rm75-single-arm-v1"
        assert root.attrs["timestamp_domain"] == "rosbag_receive"
        np.testing.assert_allclose(root["observations/joint_state/qpos"][0],
                                   [1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_allclose(root["action/gripper_action/position"][:, 0],
                                   [0.2, 0.8])
        np.testing.assert_allclose(root["observations/gripper_state/position"][:, 0],
                                   [0.15, 0.35])
        np.testing.assert_allclose(root["observations/vr_pos/position"][:, 0],
                                   [0.15, 0.35])
        assert root["observations/vr_flag_B/value"].shape == (0,)
        camera_times = root["observations/images/cam_gripper_timestamp"][:]
        assert len(camera_times) == expected_frames
        for name in ("observations/joint_state/timestamp",
                     "action/joint_action/timestamp",
                     "observations/gripper_state/timestamp",
                     "action/gripper_action/timestamp",
                     "observations/vr_pos/timestamp",
                     "observations/images/cam_gripper_timestamp"):
            timestamps = root[name][:]
            assert np.all(np.diff(timestamps) >= 0)
            assert timestamps[0] >= root.attrs["alignment_start_timestamp"]
            assert timestamps[-1] <= root.attrs["alignment_end_timestamp"]
    capture = cv2.VideoCapture(str(episode / "gripper.mp4"))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_frames
    capture.release()
    assert json.loads((episode / "conversion.json").read_text())["camera_mode"] == mode
    assert not (episode / "gripper.json").exists()
    with pytest.raises(FileExistsError):
        convert_episode(source, output)


def test_missing_required_topic_does_not_publish_episode(tmp_path: Path) -> None:
    source = _make_bag(tmp_path / "source", "jpeg", include_tracker=False)
    output = tmp_path / "output"
    with pytest.raises(ConversionError, match="tracker"):
        convert_episode(source, output)
    assert not (output / source.name).exists()
    assert not list(output.glob(".episode_12.tmp-*"))


def test_disjoint_camera_and_state_times_do_not_publish_episode(tmp_path: Path) -> None:
    source = _make_bag(tmp_path / "source", "jpeg", state_offsets=(450, 480))
    output = tmp_path / "output"
    with pytest.raises(ConversionError, match="没有共同时间区间"):
        convert_episode(source, output)
    assert not (output / source.name).exists()
    assert not list(output.glob(".episode_12.tmp-*"))
