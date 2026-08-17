"""验证独立内参包的 MCAP 图像抽取与 header 时间语义。"""

from pathlib import Path

from builtin_interfaces.msg import Time
import pytest
from rclpy.serialization import deserialize_message, serialize_message
import rosbag2_py
from sensor_msgs.msg import Image
from std_msgs.msg import String

from fastumi_camera_calibration.calibration import (
    extract_image_topic_to_sqlite3,
)


def make_stamp(timestamp_ns: int) -> Time:
    """把纳秒时间转换成 ROS Time 消息。"""
    # ROS Time 分开保存秒和纳秒。
    stamp = Time()
    stamp.sec = timestamp_ns // 1_000_000_000
    stamp.nanosec = timestamp_ns % 1_000_000_000
    return stamp


def write_topic(
    writer: rosbag2_py.SequentialWriter,
    topic_id: int,
    name: str,
    message_type: str,
) -> None:
    """在测试 MCAP 中注册一个 CDR 话题。"""
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            id=topic_id,
            name=name,
            type=message_type,
            serialization_format="cdr",
        )
    )


def write_source_bag(
    path: Path,
    timestamps: tuple[int, ...],
    message_type: str = "sensor_msgs/msg/Image",
) -> list[tuple[bytes, int]]:
    """写入内参抽取所需的图像和无关话题 MCAP。"""
    # MCAP writer 构造独立测试输入。
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    write_topic(writer, 0, "/camera/image", message_type)
    write_topic(writer, 1, "/unrelated", "std_msgs/msg/String")
    # 返回原始 CDR 和 bag 时间用于逐字节验证。
    records = []
    for index, timestamp_ns in enumerate(timestamps):
        # 图像内容按帧序号区分。
        image = Image()
        image.header.stamp = make_stamp(timestamp_ns)
        image.width = 4
        image.height = 3
        image.encoding = "bgr8"
        image.step = 12
        image.data = bytes([index]) * 36
        # 原始序列化字节应在抽取后保持一致。
        serialized = serialize_message(image)
        # bag 写入时间故意与 header 时间不同。
        bag_timestamp_ns = 9_000_000_000 + index
        writer.write("/camera/image", serialized, bag_timestamp_ns)
        writer.write(
            "/unrelated",
            serialize_message(String(data=str(index))),
            bag_timestamp_ns + 100,
        )
        records.append((serialized, bag_timestamp_ns))
    del writer
    return records


def read_sqlite_image_records(path: Path) -> list[tuple[bytes, int]]:
    """读取抽取的 SQLite3 bag。"""
    # Kalibr ROS 2 移植使用 SQLite3 输入。
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    assert [
        item.name for item in reader.get_all_topics_and_types()
    ] == ["/camera/image"]
    # 返回写入顺序中的原始记录。
    records = []
    while reader.has_next():
        # 单话题输出不应包含输入 bag 的无关记录。
        topic, serialized, timestamp_ns = reader.read_next()
        assert topic == "/camera/image"
        records.append((serialized, timestamp_ns))
    return records


def test_extraction_preserves_cdr_and_header_sampling(
    tmp_path: Path,
) -> None:
    """4 Hz 抽取应保留原始 CDR 和 bag 时间。"""
    # header 时间覆盖边界前、边界和第二个边界。
    source = tmp_path / "source"
    headers = (0, 249_000_000, 250_000_000, 500_000_000)
    records = write_source_bag(source, headers)
    extracted = extract_image_topic_to_sqlite3(
        source, "/camera/image", tmp_path / "sqlite", 4.0
    )
    selected = read_sqlite_image_records(extracted.bag_uri)
    assert extracted.frame_count == 3
    assert extracted.resolution == (4, 3)
    assert selected == [records[index] for index in (0, 2, 3)]
    # 反序列化仅用于再次确认 header 时间。
    message_type = __import__(
        "rosidl_runtime_py.utilities", fromlist=["get_message"]
    ).get_message("sensor_msgs/msg/Image")
    assert [
        deserialize_message(raw, message_type).header.stamp.nanosec
        for raw, _ in selected
    ] == [0, 250_000_000, 500_000_000]


def test_extraction_honors_header_end_boundary(tmp_path: Path) -> None:
    """采样时长应包含边界且排除其后帧。"""
    # 0.25 秒上界包含第二帧。
    source = tmp_path / "source"
    records = write_source_bag(
        source, (0, 250_000_000, 500_000_000)
    )
    extracted = extract_image_topic_to_sqlite3(
        source,
        "/camera/image",
        tmp_path / "sqlite",
        4.0,
        0.25,
    )
    assert read_sqlite_image_records(extracted.bag_uri) == records[:2]


def test_extraction_rejects_type_order_resolution_and_empty(
    tmp_path: Path,
) -> None:
    """错误消息类型、乱序、变分辨率和空话题必须失败。"""
    # 错误消息类型在反序列化前拒绝。
    wrong = tmp_path / "wrong"
    write_source_bag(
        wrong, (0,), "geometry_msgs/msg/PoseStamped"
    )
    with pytest.raises(ValueError, match="必须使用"):
        extract_image_topic_to_sqlite3(
            wrong, "/camera/image", tmp_path / "wrong_sql", 4.0
        )

    # header 时间必须严格递增。
    unordered = tmp_path / "unordered"
    write_source_bag(unordered, (100, 99))
    with pytest.raises(ValueError, match="严格递增"):
        extract_image_topic_to_sqlite3(
            unordered,
            "/camera/image",
            tmp_path / "unordered_sql",
            4.0,
        )

    # 空图像话题不能提交临时 bag。
    empty = tmp_path / "empty"
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(empty), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    write_topic(writer, 0, "/camera/image", "sensor_msgs/msg/Image")
    del writer
    with pytest.raises(ValueError, match="没有可用于"):
        extract_image_topic_to_sqlite3(
            empty, "/camera/image", tmp_path / "empty_sql", 4.0
        )
