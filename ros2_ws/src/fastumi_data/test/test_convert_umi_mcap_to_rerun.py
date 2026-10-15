"""验证根目录 convert_umi_mcap_to_rerun.py：在微型真实 MCAP 上导出 .rrd 并读回核对。"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pytest
import yaml

pytest.importorskip("rerun")
rosbag2_py = pytest.importorskip("rosbag2_py")
cv2 = pytest.importorskip("cv2")
import pyarrow as pa  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402
from fastumi_interfaces.msg import EpisodeEvent, GripperState, TrackerStatus  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from rclpy.serialization import serialize_message  # noqa: E402
from rerun.chunk import RrdReader  # noqa: E402
from sensor_msgs.msg import CompressedImage, Image  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
import convert_umi_mcap_to_rerun as cr  # noqa: E402

BASE_NS = 1_700_000_000_000_000_000
FRAME_NS = 100_000_000
# 接收时间相对消息头的固定延迟，用于区分两条时间轴。
RECEIVE_DELAY_NS = 7_000_000
FRAMES = 5
IMAGE_TOPICS = {"compressed": "/umi_camera/image_raw/compressed", "raw": "/umi_camera/image_raw"}


def time_of(ns: int) -> Time:
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def make_record(
    root: Path,
    name: str,
    mode: str = "compressed",
    include_image: bool = True,
    corrupt_image_index: int = -1,
    bad_pose_index: int = -1,
) -> Path:
    """写出一条含图像、位姿、Tracker 状态、夹爪和事件的微型录制。

    夹爪第 1 帧 raw_openness 为 NaN；filtered_openness 为 1,1,0,0,1。
    """
    record = root / name
    record.mkdir(parents=True)
    image_topic = IMAGE_TOPICS[mode]
    topics = [
        ("/vive_tracker/pose", "geometry_msgs/msg/PoseStamped"),
        ("/vive_tracker/status", "fastumi_interfaces/msg/TrackerStatus"),
        ("/gripper/state", "fastumi_interfaces/msg/GripperState"),
        ("/fastumi/episode/events", "fastumi_interfaces/msg/EpisodeEvent"),
    ]
    if include_image:
        image_type = ("sensor_msgs/msg/CompressedImage" if mode == "compressed"
                      else "sensor_msgs/msg/Image")
        topics.append((image_topic, image_type))
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(record / "raw" / "bag"), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    for index, (topic, type_name) in enumerate(topics):
        writer.create_topic(rosbag2_py.TopicMetadata(
            id=index, name=topic, type=type_name, serialization_format="cdr"))
    pixels = np.full((16, 16, 3), (10, 120, 230), dtype=np.uint8)
    jpeg = cv2.imencode(".jpg", pixels)[1].tobytes()
    filtered = [1.0, 1.0, 0.0, 0.0, 1.0]
    for i in range(FRAMES):
        stamp = BASE_NS + i * FRAME_NS
        received = stamp + RECEIVE_DELAY_NS
        if i == 0:
            event = EpisodeEvent(event_type=EpisodeEvent.START, episode_index=3, task_name="t")
            event.header.stamp = time_of(stamp)
            writer.write("/fastumi/episode/events", serialize_message(event), received)
        if include_image:
            if mode == "compressed":
                image = CompressedImage(format="jpeg")
                image.data = b"broken" if i == corrupt_image_index else jpeg
            else:
                image = Image(height=16, width=16, encoding="rgb8", step=48)
                image.data = pixels.tobytes()
            image.header.stamp = time_of(stamp)
            writer.write(image_topic, serialize_message(image), received)
        pose = PoseStamped()
        pose.header.stamp = time_of(stamp)
        pose.header.frame_id = "steamvr_tracking"
        pose.pose.position.x = 0.1 * i
        pose.pose.orientation.w = 0.5 if i == bad_pose_index else 1.0
        writer.write("/vive_tracker/pose", serialize_message(pose), received)
        status = TrackerStatus(device_connected=True, pose_valid=True,
                               tracking_state=TrackerStatus.TRACKING_RUNNING_OK)
        status.header.stamp = time_of(stamp)
        writer.write("/vive_tracker/status", serialize_message(status), received)
        gripper = GripperState(
            raw_openness=float("nan") if i == 1 else 0.5,
            filtered_openness=filtered[i], marker_distance_mm=20.0,
            detected_marker_count=2, valid=True,
        )
        gripper.header.stamp = time_of(stamp)
        writer.write("/gripper/state", serialize_message(gripper), received)
    stop = EpisodeEvent(event_type=EpisodeEvent.STOP, episode_index=3, task_name="t")
    stop.header.stamp = time_of(BASE_NS + FRAMES * FRAME_NS)
    writer.write("/fastumi/episode/events", serialize_message(stop),
                 BASE_NS + FRAMES * FRAME_NS + RECEIVE_DELAY_NS)
    del writer
    session = {
        "schema_version": 3, "bag_uri": "raw/bag", "state": "saved",
        "topic_map": {"image": image_topic, "tracker_pose": "/vive_tracker/pose",
                      "tracker_status": "/vive_tracker/status",
                      "gripper_state": "/gripper/state",
                      "episode_event": "/fastumi/episode/events"},
    }
    (record / "session.yaml").write_text(yaml.safe_dump(session), encoding="utf-8")
    return record


def read_rrd(path: Path) -> Tuple[int, Dict[str, List[Tuple[int, int, object]]]]:
    """返回 ``(blueprint 数, {实体: [(bag 时间, header 时间, 首列值)]})``，只含时序数据。"""
    reader = RrdReader(path)
    rows: Dict[str, List[Tuple[int, int, object]]] = {}
    for chunk in reader.stream():
        assert chunk.entity_path != "/__warnings"
        batch = chunk.to_record_batch()
        if cr.TIMELINE_BAG not in batch.schema.names:
            continue
        bag = batch.column(cr.TIMELINE_BAG).cast(pa.int64()).to_pylist()
        header = batch.column(cr.TIMELINE_HEADER).cast(pa.int64()).to_pylist()
        data = [name for name in batch.schema.names
                if ":" in name and not name.startswith("rerun.")]
        values = batch.column(data[0]).to_pylist() if data else [None] * len(bag)
        rows.setdefault(str(chunk.entity_path), []).extend(zip(bag, header, values))
    for items in rows.values():
        items.sort(key=lambda item: item[0])
    return len(reader.blueprints()), rows


def run(source: Path, output: Path) -> int:
    return cr.main(["--input", str(source), "--output", str(output)])


@pytest.mark.parametrize("mode", ("compressed", "raw"))
def test_exports_all_streams_with_both_timelines(tmp_path: Path, mode: str) -> None:
    record = make_record(tmp_path / "task", "rec_a", mode=mode)
    output = tmp_path / "out"
    assert run(record, output) == cr.EXIT_OK
    destination = output / "rec_a.rrd"
    blueprints, rows = read_rrd(destination)
    assert blueprints == 1

    stamps = [BASE_NS + i * FRAME_NS for i in range(FRAMES)]
    expected_times = [(s + RECEIVE_DELAY_NS, s) for s in stamps]
    camera = rows["/camera/umi"]
    assert [(bag, header) for bag, header, _ in camera] == expected_times
    blob = camera[0][2]
    blob = blob[0] if isinstance(blob, list) and len(blob) == 1 else blob
    decoded = cv2.imdecode(np.frombuffer(bytes(blob), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (16, 16, 3)

    assert len(rows["/world/tracker"]) == FRAMES
    assert len(rows["/tracker/status/pose_valid"]) == FRAMES
    # NaN 的 raw_openness 不写入，filtered 保留完整 0/1 序列。
    assert [header for _, header, _ in rows["/gripper/openness/raw"]] == \
        [s for i, s in enumerate(stamps) if i != 1]
    filtered = [value[0] if isinstance(value, list) else value
                for _, _, value in rows["/gripper/openness/filtered"]]
    assert filtered == [1.0, 1.0, 0.0, 0.0, 1.0]
    assert len(rows["/events"]) == 2


def test_invalid_image_and_pose_are_skipped_and_reported(tmp_path: Path) -> None:
    record = make_record(tmp_path / "task", "rec_a", corrupt_image_index=2, bad_pose_index=3)
    rt = cr.load_runtime()
    report = cr.convert_record(rt, record, tmp_path / "out" / "rec_a.rrd")
    assert report["counts"]["image"] == FRAMES - 1
    assert report["counts"]["tracker_pose"] == FRAMES - 1
    assert report["invalid"] == {"image": 1, "tracker_pose": 1}
    assert report["tracker_frame"] == "steamvr_tracking"


def test_existing_output_is_not_overwritten(tmp_path: Path) -> None:
    record = make_record(tmp_path / "task", "rec_a")
    output = tmp_path / "out"
    output.mkdir()
    (output / "rec_a.rrd").write_bytes(b"keep")
    assert run(record, output) == cr.EXIT_FAILED
    assert (output / "rec_a.rrd").read_bytes() == b"keep"
    assert sorted(path.name for path in output.iterdir()) == ["rec_a.rrd"]


def test_missing_image_fails_only_that_record(tmp_path: Path) -> None:
    source = tmp_path / "task"
    make_record(source, "rec_a")
    make_record(source, "rec_b", include_image=False)
    output = tmp_path / "out"
    assert run(source, output) == cr.EXIT_FAILED
    assert (output / "rec_a.rrd").is_file()
    assert sorted(path.name for path in output.iterdir()) == ["rec_a.rrd"]


def test_nested_records_keep_relative_paths_and_skip_analysis(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    make_record(source / "task_a", "rec_a")
    make_record(source / "task_b" / "session_1", "rec_b")
    make_record(source / "_analysis", "rec_c")
    output = tmp_path / "out"
    assert run(source, output) == cr.EXIT_OK
    assert sorted(path.relative_to(output).as_posix() for path in output.rglob("*.rrd")) == [
        "task_a/rec_a.rrd", "task_b/session_1/rec_b.rrd",
    ]


def iter_views(node):
    """深度优先遍历 blueprint 容器，产出全部视图。"""
    for child in getattr(node, "contents", None) or []:
        if hasattr(child, "origin"):
            yield child
        else:
            yield from iter_views(child)


def test_filtered_and_raw_openness_have_separate_views() -> None:
    """二值 filtered 与连续 raw 各占一个视图，且没有视图同时包含两者。"""
    rt = cr.load_runtime()
    views = list(iter_views(cr.blueprint(rt).root_container))
    origins = [str(view.origin) for view in views]
    assert origins.count("/gripper/openness/filtered") == 1
    assert origins.count("/gripper/openness/raw") == 1
    assert "/gripper/openness" not in origins
    assert "/gripper" not in origins


def test_input_without_records_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert run(tmp_path / "empty", tmp_path / "out") == cr.EXIT_ERROR
