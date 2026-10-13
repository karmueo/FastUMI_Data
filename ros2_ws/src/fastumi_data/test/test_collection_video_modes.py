"""验证采集控制器对 JPEG/PNG 与 H.264 图像的记录、起点门控和健康报告。"""

import threading
import time

from rclpy.serialization import serialize_message
import yaml

from collection_harness import make_harness
from fastumi_data.collection_state import CollectionConfig, TopicMap
from video_fixtures import (
    compressed_message,
    ffmpeg_message,
    h264_packets,
    stamp_of,
)

# 测试使用的话题。
JPEG_TOPIC = "/umi_camera/image_raw/compressed"
H264_TOPIC = "/umi_camera/image_raw/ffmpeg"
EVENT_TOPIC = "/fastumi/episode/events"


def _config(mode: str, topic: str, **options) -> CollectionConfig:
    """构造指定图像模式的控制器配置。"""
    return CollectionConfig(
        topics=TopicMap(image=topic, image_mode=mode),
        min_free_disk_bytes=0,
        **options,
    )


def _h264_frames(count: int = 25):
    """返回 ``count`` 个序列化的 FFMPEGPacket，帧号即索引。"""
    return [
        serialize_message(ffmpeg_message(i, data))
        for i, data in enumerate(h264_packets(count, gop_size=10))
    ]


def _wait_for_writes(harness, count: int) -> None:
    """等待写入线程把至少 ``count`` 条消息落到后端。"""
    deadline = time.monotonic() + 5
    while len(harness.backends[0].writes) < count:
        assert time.monotonic() < deadline, "写入线程没有及时写出"
        time.sleep(0.005)


def _start_in_thread(harness, request_id="r-start"):
    """在线程中开始采集（h264 会阻塞等待关键帧），返回线程与结果容器。"""
    holder = {}

    def run():
        """调用开始并保存结果。"""
        holder["result"] = harness.controller.start(request_id, "pick", "")

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 5
    # 关键帧门在暂存目录、快照和写入线程就绪后才布防，此前到达的包会被丢弃。
    while harness.controller._start_gate is None:  # noqa: SLF001
        assert time.monotonic() < deadline, "控制器没有布防关键帧门"
        time.sleep(0.005)
    assert harness.controller.status_snapshot().state == "starting"
    return thread, holder


def test_compressed_session_declares_type_and_records_raw_bytes(tmp_path) -> None:
    """JPEG 模式声明 CompressedImage 类型，原样记录消息并写入快照模式。"""
    harness = make_harness(tmp_path, _config("compressed", JPEG_TOPIC))
    started = harness.controller.start("r-start", "pick", "")
    assert started.completed, started
    payloads = [serialize_message(compressed_message(i)) for i in range(3)]
    for payload in payloads:
        harness.controller.on_image(payload)
        harness.clock.advance(33)
    harness.controller.stop_and_save("r-stop", started.collection_uuid)

    backend = harness.backends[0]
    assert {t.name: t.type_name for t in backend.topics}[JPEG_TOPIC] == (
        "sensor_msgs/msg/CompressedImage"
    )
    assert [d for t, d, _ in backend.writes if t == JPEG_TOPIC] == payloads
    destination = harness.storage.find_saved(started.collection_uuid)
    processing = yaml.safe_load(
        (destination / "calibration_snapshot" / "processing.yaml").read_text("utf-8")
    )
    assert processing["image_mode"] == "compressed"
    assert processing["topics"]["image"] == JPEG_TOPIC
    manifest = yaml.safe_load((destination / "session.yaml").read_text("utf-8"))
    assert manifest["image"]["mode"] == "compressed"
    assert manifest["image"]["message_type"] == "sensor_msgs/msg/CompressedImage"
    assert manifest["image"]["reliability"] == "best_effort"
    assert manifest["message_counts"]["image"] == 3


def test_invalid_compressed_image_is_reported_but_still_recorded(tmp_path) -> None:
    """损坏的压缩图像触发无效报警，但按“记录收到的原始消息”仍写入。"""
    harness = make_harness(tmp_path, _config("compressed", JPEG_TOPIC))
    started = harness.controller.start("r-start", "pick", "")
    bad = compressed_message(0)
    bad.data = b"garbage"
    payload = serialize_message(bad)

    harness.controller.on_image(payload)
    snapshot = harness.controller.status_snapshot()

    assert not snapshot.health.image.valid
    assert "不是 JPEG 或 PNG" in snapshot.health.image.detail
    assert "IMAGE_INVALID" in snapshot.alarms
    harness.controller.stop_and_cancel("r-cancel", started.collection_uuid)


def test_h264_start_waits_for_keyframe_then_writes_start_and_first_packet(
    tmp_path,
) -> None:
    """starting 期间丢弃非关键帧；关键帧到达后 START 与首包原子写入再 recording。"""
    harness = make_harness(tmp_path, _config("h264", H264_TOPIC))
    frames = _h264_frames()
    thread, holder = _start_in_thread(harness)

    # GOP 中段的包不可解码，不得写入，也不得结束等待。
    for payload in frames[3:6]:
        harness.controller.on_image(payload)
    assert harness.controller.status_snapshot().state == "starting"
    assert harness.backends[0].writes == []
    assert harness.controller.status_snapshot().health.image.awaiting_keyframe

    harness.clock.advance(100)
    harness.controller.on_image(frames[10])  # 关键帧（第 10 帧）。
    thread.join(timeout=5)
    harness.controller.on_image(frames[11])
    result = holder["result"]

    assert result.completed and result.state == "recording"
    _wait_for_writes(harness, 3)
    writes = harness.backends[0].writes
    assert [t for t, _, _ in writes][:3] == [EVENT_TOPIC, H264_TOPIC, H264_TOPIC]
    assert writes[0][1].decode().endswith(f"|{stamp_of(10)}")  # START 使用关键帧源时间。
    assert writes[1][1] == frames[10] and writes[2][1] == frames[11]
    assert [s for _, _, s in writes] == sorted(s for _, _, s in writes)
    health = harness.controller.status_snapshot().health.image
    assert health.format == "h264 64x48" and health.keyframes == 1
    assert not health.awaiting_keyframe
    harness.controller.stop_and_cancel("r-cancel", result.collection_uuid)


def test_h264_start_times_out_and_cleans_up(tmp_path) -> None:
    """超过 h264_start_timeout_s 没有关键帧：明确失败、清理暂存并回到 idle。"""
    harness = make_harness(
        tmp_path, _config("h264", H264_TOPIC, h264_start_timeout_s=0.2)
    )
    frames = _h264_frames()
    thread, holder = _start_in_thread(harness)
    harness.controller.on_image(frames[4])
    thread.join(timeout=5)
    result = holder["result"]

    assert not result.completed and result.code == "KEYFRAME_TIMEOUT"
    assert "SPS/PPS" in result.message
    assert harness.controller.status_snapshot().state == "idle"
    assert list(harness.storage.staging_root.iterdir()) == []
    assert harness.backends[0].writes == []

    # 迟到的关键帧不得写入已放弃的采集；之后可以正常重新开始。
    harness.controller.on_image(frames[10])
    assert harness.backends[0].writes == []
    thread, holder = _start_in_thread(harness, "r-again")
    harness.controller.on_image(frames[20])
    thread.join(timeout=5)
    assert holder["result"].completed
    harness.controller.stop_and_cancel("r-cancel", holder["result"].collection_uuid)


def test_h264_stop_keeps_existing_drain_semantics(tmp_path) -> None:
    """H.264 采集的停止仍然排空本地队列：写入计数一致，STOP 位于最后。"""
    harness = make_harness(tmp_path, _config("h264", H264_TOPIC))
    frames = _h264_frames()
    thread, holder = _start_in_thread(harness)
    for payload in frames[10:20]:
        harness.controller.on_image(payload)
        harness.clock.advance(33)
    thread.join(timeout=5)
    identifier = holder["result"].collection_uuid

    stopped = harness.controller.stop_and_save("r-stop", identifier)

    assert stopped.completed and stopped.state == "idle"
    topics = [t for t, _, _ in harness.backends[0].writes]
    assert topics[0] == topics[-1] == EVENT_TOPIC
    assert topics.count(H264_TOPIC) == 10
    destination = harness.storage.find_saved(identifier)
    quality = yaml.safe_load((destination / "quality.yaml").read_text("utf-8"))
    assert quality["image_stream"]["mode"] == "h264"
    assert quality["image_stream"]["keyframes_received"] == 1
    assert quality["message_counts"]["image"] == 10


def test_h264_b_frame_packets_are_invalid(tmp_path) -> None:
    """含 B 帧的包使图像流无效并给出原因。"""
    harness = make_harness(tmp_path, _config("h264", H264_TOPIC))
    # 该码流的第 2、3 个包是 B slice，健康状态反映最近一个包。
    for index, data in enumerate(h264_packets(12, b_frames=2)[:4]):
        harness.controller.on_image(serialize_message(ffmpeg_message(index, data)))
    health = harness.controller.status_snapshot().health.image

    assert not health.valid and "B 帧" in health.detail


def test_shutdown_cancels_pending_keyframe_wait(tmp_path) -> None:
    """退出时等待关键帧的开始请求立即失败，并释放暂存数据。"""
    harness = make_harness(
        tmp_path, _config("h264", H264_TOPIC, h264_start_timeout_s=30.0)
    )
    thread, holder = _start_in_thread(harness)

    harness.controller.shutdown()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert holder["result"].code == "START_CANCELLED"
    assert list(harness.storage.staging_root.iterdir()) == []


def test_image_check_blocks_start_with_reason(tmp_path) -> None:
    """发布端类型或 QoS 不兼容时，开始前检查给出该原因。"""
    issue = ("IMAGE_TYPE_MISMATCH", "图像话题的发布类型与 image_mode 不一致")
    harness = make_harness(
        tmp_path, _config("h264", H264_TOPIC), image_check=lambda: issue
    )

    snapshot = harness.controller.status_snapshot()
    result = harness.controller.start("r-start", "pick", "")

    assert not snapshot.can_start and issue[1] in snapshot.start_blockers
    assert not result.accepted and result.code == "PREFLIGHT_FAILED"
    assert issue[1] in result.message


def test_status_message_reports_the_published_video_type(tmp_path) -> None:
    """状态消息携带视频类型与最近一帧格式，面板据此显示。"""
    from fastumi_data.collection_node import status_to_msg

    expectations = (
        ("raw", "/umi_camera/image_raw", None, "raw", ""),
        ("compressed", JPEG_TOPIC, serialize_message(compressed_message(0)), "compressed", "jpeg"),
        ("h264", H264_TOPIC, _h264_frames(1)[0], "h264", "h264 64x48"),
    )
    for mode, topic, payload, expected_mode, expected_format in expectations:
        harness = make_harness(tmp_path / mode, _config(mode, topic))
        if payload is not None:
            harness.controller.on_image(payload)
        message = status_to_msg(harness.controller.status_snapshot())

        assert message.image_mode == expected_mode
        assert message.image_format.startswith(expected_format)


def test_session_records_nominal_source_fps_and_timestamp_source(tmp_path) -> None:
    """采集入口确认的标称源帧率、来源和时间戳来源写入 session.yaml。"""
    harness = make_harness(
        tmp_path,
        _config(
            "compressed", JPEG_TOPIC, nominal_source_fps=30.0,
            nominal_source_fps_source="camera_launch",
            image_timestamp_source="v4l2_buffer_monotonic_mapped_to_ros",
        ),
    )
    started = harness.controller.start("r-start", "pick", "")
    assert started.completed, started
    harness.controller.stop_and_save("r-stop", started.collection_uuid)
    destination = harness.storage.find_saved(started.collection_uuid)
    image = yaml.safe_load((destination / "session.yaml").read_text("utf-8"))["image"]
    assert image["nominal_source_fps"] == 30.0
    assert image["nominal_source_fps_source"] == "camera_launch"
    assert image["timestamp_source"] == "v4l2_buffer_monotonic_mapped_to_ros"


def test_unknown_external_fps_is_recorded_as_unknown(tmp_path) -> None:
    """外部相机未声明帧率时快照标记为未知，而不是采集入口默认值。"""
    harness = make_harness(tmp_path, _config("compressed", JPEG_TOPIC))
    started = harness.controller.start("r-start", "pick", "")
    harness.controller.stop_and_save("r-stop", started.collection_uuid)
    destination = harness.storage.find_saved(started.collection_uuid)
    image = yaml.safe_load((destination / "session.yaml").read_text("utf-8"))["image"]
    assert image["nominal_source_fps"] is None
    assert image["nominal_source_fps_source"] == "unknown"


def _timing_message(generation: int, index: int = 0) -> bytes:
    """构造带映射代号的相机时间诊断。"""
    from sensor_msgs.msg import TimeReference

    message = TimeReference()
    message.header.stamp.sec = 1
    message.time_ref.sec = 2
    message.source = f"mode=v4l2_buffer;clock=monotonic;tsrc=soe;seq={index};gen={generation}"
    return serialize_message(message)


TIMING_TOPIC = "/umi_camera/image_raw/time_reference"


def test_image_timing_topic_is_recorded(tmp_path) -> None:
    """配置时间诊断话题时声明 TimeReference 类型并原样录制。"""
    harness = make_harness(
        tmp_path,
        CollectionConfig(
            topics=TopicMap(image=JPEG_TOPIC, image_mode="compressed", image_timing=TIMING_TOPIC),
            min_free_disk_bytes=0,
        ),
    )
    harness.controller.on_image_timing(_timing_message(1), 1)
    started = harness.controller.start("r-start", "pick", "")
    assert started.completed, started
    payload = _timing_message(1, 5)
    harness.controller.on_image_timing(payload, 1)
    harness.controller.stop_and_save("r-stop", started.collection_uuid)
    backend = harness.backends[0]
    assert {t.name: t.type_name for t in backend.topics}[TIMING_TOPIC] == (
        "sensor_msgs/msg/TimeReference"
    )
    assert [d for t, d, _ in backend.writes if t == TIMING_TOPIC] == [payload]


def test_clock_mapping_reset_during_recording_fails_collection(tmp_path) -> None:
    """采集中映射代号变化（系统时间跳变）使当前采集进入错误状态。"""
    harness = make_harness(
        tmp_path,
        CollectionConfig(
            topics=TopicMap(image=JPEG_TOPIC, image_mode="compressed", image_timing=TIMING_TOPIC),
            min_free_disk_bytes=0,
        ),
    )
    harness.controller.on_image_timing(_timing_message(1), 1)
    started = harness.controller.start("r-start", "pick", "")
    assert started.completed
    harness.controller.on_image_timing(_timing_message(1), 1)
    assert harness.controller.tick().state == "recording"
    harness.controller.on_image_timing(_timing_message(2), 2)
    status = harness.controller.tick()
    assert status.state == "error"
    assert status.last_result_code == "CLOCK_MAPPING_RESET"
    assert "时钟映射" in status.last_error


def test_timing_generation_parser() -> None:
    """从 TimeReference.source 读取 gen 字段。"""
    from fastumi_data.collection_node import timing_generation

    assert timing_generation("mode=v4l2_buffer;gen=3;offset_ns=1") == 3
    assert timing_generation("mode=callback_now") == 0
    assert timing_generation("gen=x") == 0
