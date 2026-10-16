"""验证普通与连续录制路径的真实 MCAP 字节、顺序、时间和元数据完整性。"""

from collections import Counter
from pathlib import Path
import random

from fastumi_interfaces.msg import EpisodeEvent
import pytest
import rclpy
from rclpy.serialization import deserialize_message, serialize_message
import rosbag2_py
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from fastumi_data.collection_writer import QueuedBagWriter, RosbagMcapBackend, TopicSpec
from fastumi_data.continuous_recorder import EVENT_TOPIC, ContinuousRecorder
from fastumi_data.episode_manager import EpisodeManager
from fastumi_data.image_modes import (
    IMAGE_MESSAGE_TYPES,
    MODE_COMPRESSED,
    MODE_H264,
    MODE_RAW,
)
from video_bag_builder import TOPICS
from video_fixtures import compressed_message, ffmpeg_message, h264_packets, raw_message


# 三路写入内容都使用固定递增的 MCAP 日志时间。
_STAMP_START_NS = 1_700_000_000_000_000_000
_STAMP_STEP_NS = 1_000_000
# 普通路径使用固定的三话题集合，覆盖图像与标准消息类型。
_IMAGE_TOPIC = "/integrity/image"
_STATUS_TOPIC = "/integrity/status"
_FLAG_TOPIC = "/integrity/flag"
# 连续录制路径额外注册该标准消息话题。
_CONTINUOUS_STATE_TOPIC = "/integrity/continuous_state"


def _large_raw_message() -> object:
    """创建大于默认 MCAP chunk 的合法伪随机 RGB 图像消息。"""
    width = 1024
    height = 2048
    step = width * 3
    # 固定伪随机像素避免高压缩率掩盖大帧写入问题。
    pixel_data = random.Random(0xFA57_A11).randbytes(height * step)
    message = raw_message(0)
    message.width = width
    message.height = height
    message.encoding = "rgb8"
    message.step = step
    message.data = pixel_data
    return message


def _messages_for_mode(mode: str) -> list[object]:
    """按图像模式构造两条合法 ROS 图像消息。"""
    if mode == MODE_RAW:
        return [_large_raw_message(), raw_message(1)]
    if mode == MODE_COMPRESSED:
        return [compressed_message(index, "jpeg") for index in range(2)]
    packets = h264_packets(2)
    return [ffmpeg_message(index, packets[index]) for index in range(2)]


def _read_mcap(uri: Path):
    """读取真实 MCAP 的注册话题、逐条记录和 rosbag2 元数据。"""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(uri), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    try:
        types = {
            item.name: item.type for item in reader.get_all_topics_and_types()
        }
        records = []
        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()
            records.append((topic, bytes(data), timestamp_ns))
    finally:
        reader.close()
    metadata = rosbag2_py.Info().read_metadata(str(uri), "mcap")
    return types, records, metadata


def _assert_metadata(metadata, expected_types: dict[str, str], records) -> None:
    """核对 MCAP 元数据中的总数、话题类型和各话题消息数量。"""
    expected_counts = dict(Counter(topic for topic, _, _ in records))
    metadata_topics = metadata.topics_with_message_count
    actual_types = {
        item.topic_metadata.name: item.topic_metadata.type
        for item in metadata_topics
    }
    actual_counts = {
        item.topic_metadata.name: item.message_count
        for item in metadata_topics
    }
    assert metadata.message_count == len(records)
    assert actual_types == expected_types
    assert actual_counts == expected_counts


def _assert_records(actual_records, expected_records) -> None:
    """逐条核对话题、完整 CDR 字节和 MCAP 日志时间。"""
    assert len(actual_records) == len(expected_records)
    for actual, expected in zip(actual_records, expected_records):
        assert actual == expected


@pytest.mark.parametrize("mode", [MODE_RAW, MODE_COMPRESSED, MODE_H264])
def test_queued_writer_preserves_interleaved_mcap_records(tmp_path: Path, mode: str) -> None:
    """验证普通队列写入器在三种图像模式下完整往返多话题记录。"""
    image_type = IMAGE_MESSAGE_TYPES[mode]
    expected_types = {
        _IMAGE_TOPIC: image_type,
        _STATUS_TOPIC: "std_msgs/msg/String",
        _FLAG_TOPIC: "std_msgs/msg/Bool",
    }
    specs = [TopicSpec(topic, type_name) for topic, type_name in expected_types.items()]
    writer = QueuedBagWriter(RosbagMcapBackend(), 16, 16 * 1024 * 1024)
    uri = tmp_path / f"queued_{mode}"
    result = None
    expected_records = []
    try:
        writer.start(str(uri), specs)
        image_data = [serialize_message(message) for message in _messages_for_mode(mode)]
        if mode == MODE_RAW:
            assert len(image_data[0]) > 5 * 1024 * 1024

        # 交错写入三路消息，每个话题至少出现两次。
        payloads = [
            (_IMAGE_TOPIC, image_data[0]),
            (_STATUS_TOPIC, serialize_message(String(data="ready"))),
            (_FLAG_TOPIC, serialize_message(Bool(data=True))),
            (_IMAGE_TOPIC, image_data[1]),
            (_STATUS_TOPIC, serialize_message(String(data="capturing"))),
            (_FLAG_TOPIC, serialize_message(Bool(data=False))),
        ]
        for index, (topic, data) in enumerate(payloads):
            timestamp_ns = _STAMP_START_NS + index * _STAMP_STEP_NS
            assert writer.offer(topic, data, timestamp_ns)
            expected_records.append((topic, data, timestamp_ns))
        result = writer.finish(20.0)
    finally:
        if result is None or not result.thread_stopped:
            writer.abort(20.0)

    assert result.ok
    assert result.message_counts == dict(
        Counter(topic for topic, _, _ in expected_records)
    )
    actual_types, actual_records, metadata = _read_mcap(uri)
    assert actual_types == expected_types
    _assert_records(actual_records, expected_records)
    _assert_metadata(metadata, expected_types, actual_records)


@pytest.mark.parametrize("mode", [MODE_COMPRESSED, MODE_H264])
def test_continuous_recorder_preserves_image_events_and_sensor_fifo(
    tmp_path: Path, mode: str,
) -> None:
    """验证连续编码路径保存 readiness、episode 事件和额外话题的完整 FIFO 流。"""
    context_initialized = False
    recorder = None
    manager = None
    recorder_closed = False
    uri = tmp_path / f"continuous_{mode}"
    expected_types = {
        TOPICS["image"]: IMAGE_MESSAGE_TYPES[mode],
        EVENT_TOPIC: "fastumi_interfaces/msg/EpisodeEvent",
        _CONTINUOUS_STATE_TOPIC: "std_msgs/msg/String",
    }
    expected_payloads = []
    event_payloads = []
    try:
        rclpy.init(args=[
            "--ros-args", "-p", f"image_mode:={mode}",
            "-p", f"image_topic:={TOPICS['image']}",
        ])
        context_initialized = True
        recorder = ContinuousRecorder(str(uri), [TOPICS["image"], EVENT_TOPIC])
        recorder._register_topic(_CONTINUOUS_STATE_TOPIC, "std_msgs/msg/String")

        def observe_event(event: EpisodeEvent) -> None:
            """按生产顺序写入 episode 事件并保存其真实序列化字节。"""
            serialized = serialize_message(event)
            recorder.record_event(event)
            event_payloads.append(serialized)

        manager = EpisodeManager(
            event_observer=observe_event,
            image_readiness=recorder.readiness,
        )
        assert not manager._start(Trigger.Request(), Trigger.Response()).success

        if mode == MODE_COMPRESSED:
            images = [compressed_message(index, "jpeg") for index in range(2)]
        else:
            packets = h264_packets(2)
            images = [ffmpeg_message(index, packets[index]) for index in range(2)]

        for index, image in enumerate(images):
            serialized_image = serialize_message(image)
            recorder._on_image(serialized_image)
            expected_payloads.append((TOPICS["image"], serialized_image))
            if index == 0:
                assert recorder.readiness.ready
                assert manager._start(Trigger.Request(), Trigger.Response()).success
                expected_payloads.append((EVENT_TOPIC, event_payloads[-1]))

            sensor_data = serialize_message(String(data=f"sensor-{index}"))
            recorder._write(_CONTINUOUS_STATE_TOPIC, sensor_data)
            expected_payloads.append((_CONTINUOUS_STATE_TOPIC, sensor_data))

        assert manager._stop(Trigger.Request(), Trigger.Response()).success
        expected_payloads.append((EVENT_TOPIC, event_payloads[-1]))
        recorder_closed = True
        recorder.close()

        actual_types, actual_records, metadata = _read_mcap(uri)
        assert actual_types == expected_types
        assert len(actual_records) == len(expected_payloads)
        for actual, (expected_topic, expected_data) in zip(
            actual_records, expected_payloads
        ):
            assert actual[0] == expected_topic
            assert actual[1] == expected_data
        timestamps = [timestamp_ns for _, _, timestamp_ns in actual_records]
        assert timestamps == sorted(timestamps)
        _assert_metadata(metadata, expected_types, actual_records)
        recorded_events = [
            deserialize_message(data, EpisodeEvent).event_type
            for topic, data, _ in actual_records
            if topic == EVENT_TOPIC
        ]
        assert recorded_events == [EpisodeEvent.START, EpisodeEvent.STOP]
    finally:
        try:
            if manager is not None:
                manager.destroy_node()
        finally:
            try:
                if recorder is not None:
                    try:
                        if not recorder_closed:
                            recorder_closed = True
                            recorder.close()
                    finally:
                        recorder.destroy_node()
            finally:
                if context_initialized and rclpy.ok():
                    rclpy.shutdown()
