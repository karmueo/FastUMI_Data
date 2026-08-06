"""验证双 ArUco 标定使用的 MCAP 时间线和开度插值。"""

from pathlib import Path

import numpy as np
import pytest


try:
    from builtin_interfaces.msg import Time
    from fastumi_interfaces.msg import GripperState, TrackerStatus
    from geometry_msgs.msg import PoseStamped
    import rosbag2_py
    from rclpy.serialization import serialize_message
except ModuleNotFoundError:
    ROS_AVAILABLE = False
else:
    ROS_AVAILABLE = True

if ROS_AVAILABLE:
    from fastumi_data.aruco_tcp_bag import read_aruco_tcp_timeline


def _time(timestamp_ns: int) -> Time:
    """把纳秒时间戳转换为 ROS Time 消息。"""
    message = Time()
    message.sec = timestamp_ns // 1_000_000_000
    message.nanosec = timestamp_ns % 1_000_000_000
    return message


def _write_topic(writer, topic_id: int, name: str, message_type: str) -> None:
    """注册一个合成 MCAP 话题。"""
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            id=topic_id,
            name=name,
            type=message_type,
            serialization_format="cdr",
        )
    )


def _make_pose(timestamp_ns: int) -> PoseStamped:
    """构造带 header 时间的 Tracker pose。"""
    message = PoseStamped()
    message.header.stamp = _time(timestamp_ns)
    message.pose.orientation.w = 1.0
    return message


def _make_status(timestamp_ns: int, valid: bool = True) -> TrackerStatus:
    """构造带 header 时间的 Tracker status。"""
    message = TrackerStatus()
    message.header.stamp = _time(timestamp_ns)
    message.serial_number = "LHR-TEST"
    message.device_connected = valid
    message.pose_valid = valid
    message.tracking_state = TrackerStatus.TRACKING_RUNNING_OK if valid else 0
    return message


def _make_gripper(timestamp_ns: int, openness: float, valid: bool = True):
    """构造带 raw_openness 的 GripperState。"""
    message = GripperState()
    message.header.stamp = _time(timestamp_ns)
    message.raw_openness = openness
    message.filtered_openness = openness
    message.detected_marker_count = 2 if valid else 0
    message.valid = valid
    return message


def _write_timeline_bag(
    path: Path,
    pose_times=(1_000_000_000, 2_000_000_000, 3_000_000_000),
    gripper_values=(0.0, 0.5, 1.0),
    gripper_valid=(True, True, True),
    status_valid=(True, True, True),
) -> tuple[str, str, str]:
    """写入最小 pose/status/gripper MCAP 并故意错开 bag 写入时间。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topics = {
        "pose": "/vive_tracker/pose",
        "status": "/vive_tracker/status",
        "gripper": "/gripper/state",
    }
    message_types = {
        topics["pose"]: "geometry_msgs/msg/PoseStamped",
        topics["status"]: "fastumi_interfaces/msg/TrackerStatus",
        topics["gripper"]: "fastumi_interfaces/msg/GripperState",
    }
    for topic_id, (name, message_type) in enumerate(message_types.items()):
        _write_topic(writer, topic_id, name, message_type)
    records = []
    for index, timestamp_ns in enumerate(pose_times):
        bag_base_ns = 1_000_000_000 + index * 1_000_000_000
        records.append(
            (
                bag_base_ns + 100_000_000,
                topics["pose"],
                _make_pose(timestamp_ns),
            )
        )
        records.append(
            (
                bag_base_ns + 200_000_000,
                topics["status"],
                _make_status(timestamp_ns, status_valid[index]),
            )
        )
        records.append(
            (
                bag_base_ns + 300_000_000,
                topics["gripper"],
                _make_gripper(
                    timestamp_ns,
                    gripper_values[index],
                    gripper_valid[index],
                ),
            )
        )
    for bag_timestamp_ns, topic, message in sorted(
        records, key=lambda record: record[0]
    ):
        writer.write(topic, serialize_message(message), bag_timestamp_ns)
    del writer
    return topics["pose"], topics["status"], topics["gripper"]


@pytest.mark.skipif(not ROS_AVAILABLE, reason="需要 ROS 2 MCAP 依赖")
def test_timeline_uses_header_time_and_interpolates_raw_openness(tmp_path):
    """时间线应忽略 bag 写入时间并按 raw_openness 线性插值。"""
    bag = tmp_path / "bag"
    pose_topic, status_topic, gripper_topic = _write_timeline_bag(bag)

    timeline = read_aruco_tcp_timeline(
        str(bag), pose_topic, status_topic, gripper_topic
    )

    assert tuple(timeline.pose_timestamps_ns) == (
        1_000_000_000,
        2_000_000_000,
        3_000_000_000,
    )
    assert timeline.interpolate_openness(1_500_000_000, 1_200.0) == pytest.approx(
        0.25
    )
    assert timeline.pose_timestamps_ns.flags.writeable is False
    assert timeline.status_timestamps_ns.flags.writeable is False
    assert timeline.gripper_timestamps_ns.flags.writeable is False


@pytest.mark.skipif(not ROS_AVAILABLE, reason="需要 ROS 2 MCAP 依赖")
def test_timeline_rejects_duplicate_or_descending_header_time(tmp_path):
    """单话题 header 时间重复或倒序时必须拒绝整条时间线。"""
    for times in (
        (1_000_000_000, 2_000_000_000, 2_000_000_000),
        (2_000_000_000, 1_000_000_000, 3_000_000_000),
    ):
        bag = tmp_path / str(times[1]) / "bag"
        pose_topic, status_topic, gripper_topic = _write_timeline_bag(
            bag, pose_times=times
        )
        with pytest.raises(ValueError, match="严格递增"):
            read_aruco_tcp_timeline(
                str(bag), pose_topic, status_topic, gripper_topic
            )


@pytest.mark.skipif(not ROS_AVAILABLE, reason="需要 ROS 2 MCAP 依赖")
def test_interpolation_returns_none_for_invalid_samples_and_large_gaps(tmp_path):
    """无效开度、非有限值和超过 gap 的查询应返回 None。"""
    bag = tmp_path / "bag"
    topics = _write_timeline_bag(
        bag,
        gripper_values=(0.0, float("nan"), 1.0),
        gripper_valid=(True, False, True),
        status_valid=(True, False, True),
    )
    timeline = read_aruco_tcp_timeline(str(bag), *topics)

    assert timeline.interpolate_openness(1_500_000_000, 100.0) is None
    assert timeline.interpolate_openness(1_500_000_000, 600.0) is None
    assert timeline.interpolate_openness(500_000_000, 600.0) is None
