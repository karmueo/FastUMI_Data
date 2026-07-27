"""使用合成 MCAP 验证 episode 事件到 FastUMI HDF5 的端到端链路。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from fastumi_data.models import ProcessingConfig


try:
    # 正式采集机具备以下依赖；精简算法环境允许跳过 I/O 集成项。
    from builtin_interfaces.msg import Time
    from fastumi_interfaces.msg import (
        EpisodeEvent,
        GripperState,
        TrackerStatus,
    )
    from geometry_msgs.msg import PoseStamped
    import h5py
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Image

    from fastumi_data.mcap_converter import McapEpisodeConverter
except ModuleNotFoundError:
    INTEGRATION_AVAILABLE = False
    h5py = None
    McapEpisodeConverter = None
else:
    INTEGRATION_AVAILABLE = True


def _time(timestamp_ns: int) -> Time:
    """把纳秒整数转换为 ROS Time 消息。"""
    message = Time()
    message.sec = timestamp_ns // 1_000_000_000
    message.nanosec = timestamp_ns % 1_000_000_000
    return message


def _write_topic(
    writer: rosbag2_py.SequentialWriter,
    topic_id: int,
    name: str,
    message_type: str,
) -> None:
    """在合成 rosbag2 中注册一个 CDR 话题。"""
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            id=topic_id,
            name=name,
            type=message_type,
            serialization_format="cdr",
        )
    )


@pytest.mark.skipif(
    not INTEGRATION_AVAILABLE,
    reason="当前 Python 环境缺少 ROS2 MCAP 或 h5py 依赖",
)
def test_synthetic_mcap_to_hdf5(tmp_path: Path) -> None:
    """验证 MCAP 切分、同步、状态质量和 HDF5 数组长度。"""
    bag_uri = tmp_path / "raw" / "bag"
    bag_uri.parent.mkdir(parents=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag_uri), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topics = {
        "image": "/camera/image",
        "tracker_pose": "/vive_tracker/pose",
        "tracker_status": "/vive_tracker/status",
        "gripper_state": "/gripper/state",
        "episode_event": "/fastumi/episode/events",
    }
    topic_types = {
        topics["image"]: "sensor_msgs/msg/Image",
        topics["tracker_pose"]: "geometry_msgs/msg/PoseStamped",
        topics["tracker_status"]: "fastumi_interfaces/msg/TrackerStatus",
        topics["gripper_state"]: "fastumi_interfaces/msg/GripperState",
        topics["episode_event"]: "fastumi_interfaces/msg/EpisodeEvent",
    }
    for topic_id, (name, message_type) in enumerate(topic_types.items()):
        _write_topic(writer, topic_id, name, message_type)

    start_ns = 1_000_000_000
    stop_ns = 2_000_000_000
    records = []
    start_event = EpisodeEvent()
    start_event.header.stamp = _time(start_ns)
    start_event.task_name = "task"
    start_event.session_id = "session"
    start_event.episode_index = 0
    start_event.event_type = EpisodeEvent.START
    records.append((start_ns, topics["episode_event"], start_event))

    for index in range(31):
        timestamp_ns = start_ns + int(round(index / 30.0 * 1.0e9))
        image = Image()
        image.header.stamp = _time(timestamp_ns)
        image.height = 8
        image.width = 8
        image.encoding = "rgb8"
        image.step = 24
        image.data = np.zeros((8, 8, 3), dtype=np.uint8).tobytes()
        records.append((timestamp_ns, topics["image"], image))

        pose = PoseStamped()
        pose.header.stamp = _time(timestamp_ns)
        pose.header.frame_id = "steamvr_tracking_ros"
        pose.pose.position.x = index / 300.0
        pose.pose.orientation.w = 1.0
        records.append((timestamp_ns, topics["tracker_pose"], pose))

        status = TrackerStatus()
        status.header.stamp = _time(timestamp_ns)
        status.serial_number = "LHR-TEST"
        status.device_connected = True
        status.pose_valid = True
        status.tracking_state = TrackerStatus.TRACKING_RUNNING_OK
        records.append((timestamp_ns, topics["tracker_status"], status))

        gripper = GripperState()
        gripper.header.stamp = _time(timestamp_ns)
        gripper.raw_openness = index / 30.0
        gripper.filtered_openness = index / 30.0
        gripper.marker_distance_mm = 50.0 + index
        gripper.detected_marker_count = 2
        gripper.valid = True
        records.append((timestamp_ns, topics["gripper_state"], gripper))

    stop_event = EpisodeEvent()
    stop_event.header.stamp = _time(stop_ns)
    stop_event.task_name = "task"
    stop_event.session_id = "session"
    stop_event.episode_index = 0
    stop_event.event_type = EpisodeEvent.STOP
    records.append((stop_ns, topics["episode_event"], stop_event))
    for timestamp_ns, topic, message in sorted(
        records, key=lambda record: (record[0], record[1])
    ):
        writer.write(topic, serialize_message(message), timestamp_ns)
    del writer

    extrinsic_path = tmp_path / "tracker_to_tcp.yaml"
    extrinsic_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "tracker_serial": "LHR-TEST",
                "translation_rmse_mm": 0.0,
                "rotation_rmse_deg": 0.0,
                "tracker_to_tcp": {
                    "translation_m": [0.0, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ),
        encoding="utf-8",
    )
    converter = McapEpisodeConverter(
        str(bag_uri),
        str(tmp_path),
        ProcessingConfig(),
        topics,
        str(extrinsic_path),
        force=False,
    )
    summary = converter.convert()

    assert summary["converted"] == 1
    output = tmp_path / "episodes" / "episode_0000.hdf5"
    with h5py.File(output, "r") as root:
        lengths = {
            root["action"].shape[0],
            root["observations/qpos"].shape[0],
            root["observations/images/front"].shape[0],
            root["observations/timestamp_ns"].shape[0],
        }
        assert lengths == {20}
        assert np.all(
            root["observations/quality/tracker_tracking_ok"][:]
        )
