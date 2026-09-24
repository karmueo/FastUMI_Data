"""Integration checks for Rerun export of the hardware MCAP topics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pytest

pytest.importorskip("rerun")
import pyarrow as pa
from rerun.chunk import RrdReader

from convert_hardware_mcap import ConversionError
from convert_hardware_mcap_to_rerun import _log_camera, _read_episode, convert_episode, main
from test_convert_hardware_mcap import BASE_NS, _make_bag


@pytest.mark.parametrize("mode", ("raw", "jpeg", "h264"))
def test_exports_all_six_topics_without_cropping(tmp_path: Path, mode: str) -> None:
    episode = _make_bag(tmp_path / "source", mode)
    with tempfile.TemporaryFile(mode="w+b") as spool:
        data = _read_episode(episode / "bag", spool)
    assert data["frame_id"] == "vive_tracker_odom"
    assert [timestamp for timestamp, _ in data["series"]["joint_action"]] == [
        BASE_NS + 100_000_000, BASE_NS + 500_000_000
    ]
    command_ns, command = data["series"]["gripper_action"][0]
    assert command_ns == BASE_NS
    assert command == pytest.approx(0.2)
    np.testing.assert_array_equal(data["series"]["joint_state"][0][1],
                                  [1, 2, 3, 4, 5, 6, 7])
    position, quaternion = data["series"]["tracker"][0][1]
    np.testing.assert_array_equal(position, [0.15, 0, 0])
    np.testing.assert_array_equal(quaternion, [0, 0, 0, 1])

    output = tmp_path / "output"
    report = convert_episode(episode, output)
    assert (output / "episode_12.rrd").stat().st_size > 0
    assert report["counts"] == {
        "joint_state": 2, "joint_action": 2, "gripper_state": 2,
        "gripper_action": 2, "tracker": 2, "camera": 5,
    }
    assert report["camera_skipped_until_keyframe"] == 0
    rows = {}
    camera_timestamps = []
    camera_columns = set()
    first_encoded_frame = None
    reader = RrdReader(output / "episode_12.rrd")
    assert len(reader.blueprints()) == 1
    for chunk in reader.stream():
        assert chunk.entity_path != "/__warnings"
        batch = chunk.to_record_batch()
        if "bag_receive_time" not in batch.schema.names:
            continue
        rows[chunk.entity_path] = rows.get(chunk.entity_path, 0) + len(batch)
        if chunk.entity_path == "/camera/wrist":
            camera_columns.update(batch.schema.names)
            camera_timestamps.extend(
                batch.column("bag_receive_time").cast(pa.int64()).to_pylist()
            )
            if "EncodedImage:blob" in batch.schema.names and first_encoded_frame is None:
                first_encoded_frame = bytes(batch.column("EncodedImage:blob")[0].as_py()[0])
    assert rows["/camera/wrist"] == 5
    assert rows["/world/vive_tracker_odom/tracker"] == 2
    assert rows["/joint/feedback/joint1"] == 2
    assert rows["/gripper/command"] == 2
    assert sorted(camera_timestamps) == [
        BASE_NS + index * 100_000_000 for index in range(5)
    ]
    assert "VideoFrameReference:timestamp" not in camera_columns
    if mode == "raw":
        assert "Image:buffer" in camera_columns
    else:
        assert "EncodedImage:blob" in camera_columns
        image = cv2.imdecode(np.frombuffer(first_encoded_frame, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (16, 16, 3)
    with pytest.raises(FileExistsError):
        convert_episode(episode, output)


def test_missing_tracker_does_not_publish_rrd(tmp_path: Path) -> None:
    episode = _make_bag(tmp_path / "source", "jpeg", include_tracker=False)
    with pytest.raises(ConversionError, match="tracker"):
        convert_episode(episode, tmp_path / "output")
    assert not list((tmp_path / "output").glob("*.rrd"))


def test_corrupt_jpeg_is_skipped_and_reported(tmp_path: Path) -> None:
    episode = _make_bag(tmp_path / "source", "jpeg", corrupt_jpeg_index=2)
    output = tmp_path / "output"
    report = convert_episode(episode, output)
    assert report["counts"]["camera"] == 4
    assert report["invalid"]["camera"] == 1

    camera_times = []
    for chunk in RrdReader(output / "episode_12.rrd").stream():
        if chunk.entity_path != "/camera/wrist":
            continue
        batch = chunk.to_record_batch()
        if "bag_receive_time" in batch.schema.names:
            camera_times.extend(batch.column("bag_receive_time").cast(pa.int64()).to_pylist())
    assert sorted(camera_times) == [
        BASE_NS + index * 100_000_000 for index in (0, 1, 3, 4)
    ]


def test_h264_without_keyframe_is_reported(tmp_path: Path) -> None:
    episode = _make_bag(tmp_path / "source", "h264")
    with tempfile.TemporaryFile(mode="w+b") as spool:
        data = _read_episode(episode / "bag", spool)
        data["camera"] = [replace(frame, keyframe=False) for frame in data["camera"]]
        with pytest.raises(ConversionError, match="关键帧"):
            _log_camera(None, data, spool, tmp_path)


def test_h264_starts_at_first_available_keyframe(tmp_path: Path) -> None:
    episode = _make_bag(tmp_path / "source", "h264", skip_first_h264_packet=True)
    report = convert_episode(episode, tmp_path / "output")
    assert report["camera_skipped_until_keyframe"] == 1
    assert report["counts"]["camera"] == 3
    camera_times = []
    for chunk in RrdReader(tmp_path / "output" / "episode_12.rrd").stream():
        if chunk.entity_path != "/camera/wrist":
            continue
        batch = chunk.to_record_batch()
        if "bag_receive_time" in batch.schema.names:
            camera_times.extend(batch.column("bag_receive_time").cast(pa.int64()).to_pylist())
    assert sorted(camera_times) == [
        BASE_NS + index * 100_000_000 for index in (2, 3, 4)
    ]


def test_parent_directory_continues_after_existing_output(tmp_path: Path) -> None:
    first = _make_bag(tmp_path / "source", "raw")
    second = tmp_path / "source" / "episode_13"
    second.symlink_to(first, target_is_directory=True)
    output = tmp_path / "output"
    convert_episode(first, output)
    assert main(["--input", str(first.parent), "--output", str(output)]) == 1
    assert (output / "episode_13.rrd").is_file()
