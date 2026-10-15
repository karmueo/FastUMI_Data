"""验证连续编码录制只使用已写入的图像门控，并保留可被 Link7 转换读取的 MCAP 数据流。"""

from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import Mock

from fastumi_interfaces.msg import GripperState, TrackerStatus
from geometry_msgs.msg import PoseStamped
import pytest
import rclpy
from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
from rclpy.serialization import serialize_message
import rosbag2_py
from std_srvs.srv import Trigger

from fastumi_data.bag_images import BagImageDecoder
from fastumi_data.continuous_recorder import ContinuousRecorder
from fastumi_data.episode_manager import EpisodeManager
from fastumi_data.image_messages import packet_from_ffmpeg_message
from fastumi_data.link7_config import SOURCE_SNAPSHOT, resolve_processing
from fastumi_data.mcap_converter import default_processing_path
from fastumi_data.video_decoding import COLOR_RGB, DecodeReadiness
from video_bag_builder import TOPICS, _event, _time
from video_fixtures import compressed_message, ffmpeg_message, h264_packets, stamp_of


@pytest.fixture
def recording(tmp_path: Path, request):
    """创建共享录包状态的管理器，测试结束后关闭 MCAP 和 ROS 上下文。"""
    mode = getattr(request, "param", "h264")
    rclpy.init(args=[
        "--ros-args", "-p", f"image_mode:={mode}",
        "-p", f"image_topic:={TOPICS['image']}",
    ])
    recorder = ContinuousRecorder(str(tmp_path / "bag"), list(TOPICS.values()))
    manager = EpisodeManager(
        event_observer=recorder.record_event, image_readiness=recorder.readiness
    )
    try:
        yield recorder, manager, tmp_path
    finally:
        manager.destroy_node()
        recorder.close()
        recorder.destroy_node()
        rclpy.shutdown()


def _start(manager: EpisodeManager) -> Trigger.Response:
    """调用首个 episode 的开始服务。"""
    return manager._start(Trigger.Request(), Trigger.Response())


def test_independent_observer_keyframe_cannot_open_recording_gate(recording) -> None:
    """其他订阅者收到关键帧时，录包输入未收到该包仍拒绝开始。"""
    recorder, manager, _ = recording
    packets = h264_packets(25)
    independent = DecodeReadiness("h264")
    independent.feed(packet_from_ffmpeg_message(ffmpeg_message(0, packets[0])))
    assert independent.ready
    assert not hasattr(manager, "_image_subscription")
    assert manager._readiness is recorder.readiness
    assert not _start(manager).success

    for index in range(4, 10):
        recorder._on_image(serialize_message(ffmpeg_message(index, packets[index])))
    assert not _start(manager).success
    recorder._on_image(serialize_message(ffmpeg_message(10, packets[10])))
    assert _start(manager).success


@pytest.mark.parametrize("recording", ["compressed", "h264"], indirect=True)
def test_failed_image_write_never_opens_gate(recording, monkeypatch) -> None:
    """即使图像可解码，实际写入失败也不允许开始并明确终止后续写入。"""
    recorder, manager, _ = recording
    real_writer = recorder._writer
    failing_writer = Mock(wraps=real_writer)
    failing_writer.write.side_effect = OSError("磁盘已满")
    monkeypatch.setattr(recorder, "_writer", failing_writer)
    message = (
        compressed_message(0) if recorder._mode == "compressed"
        else ffmpeg_message(0, h264_packets(1)[0])
    )
    with pytest.raises(OSError, match="磁盘已满"):
        recorder._on_image(serialize_message(message))
    assert not recorder.readiness.ready
    assert not _start(manager).success
    with pytest.raises(RuntimeError, match="连续录制写入失败"):
        recorder._discover_topics()


def test_failed_event_write_rejects_start_without_committing_state(
    recording, monkeypatch,
) -> None:
    """关键帧已写入但 START 写入失败时，管理器不提交活动状态。"""
    recorder, manager, _ = recording
    recorder._on_image(serialize_message(ffmpeg_message(0, h264_packets(1)[0])))
    real_writer = recorder._writer
    failing_writer = Mock(wraps=real_writer)
    failing_writer.write.side_effect = OSError("事件写入失败")
    monkeypatch.setattr(recorder, "_writer", failing_writer)

    response = _start(manager)

    assert not response.success and "事件写入失败" in response.message
    assert not manager.state_snapshot().active
    assert not manager._first_episode_started


@pytest.mark.parametrize("recording", ["compressed", "h264"], indirect=True)
def test_recorded_video_decodes_after_mid_gop_join(recording, monkeypatch) -> None:
    """晚接入的连续录包先保存解码前缀，关键帧写入后开始的 episode 可解码。"""
    recorder, manager, root = recording
    mode = recorder._mode
    packets = h264_packets(50) if mode == "h264" else None
    for topic, type_name in (
        (TOPICS["tracker_pose"], "geometry_msgs/msg/PoseStamped"),
        (TOPICS["tracker_status"], "fastumi_interfaces/msg/TrackerStatus"),
        (TOPICS["gripper_state"], "fastumi_interfaces/msg/GripperState"),
    ):
        recorder._register_topic(topic, type_name)
    for index in range(4, 50):
        monkeypatch.setattr(
            manager, "_make_event",
            lambda event_type, i=index: _event(0, event_type, stamp_of(i)),
        )
        if index == 14:
            assert _start(manager).success
        if index == 44:
            assert manager._stop(Trigger.Request(), Trigger.Response()).success
        image = (
            compressed_message(index) if mode == "compressed"
            else ffmpeg_message(index, packets[index])
        )
        recorder._on_image(serialize_message(image))
        pose = PoseStamped()
        pose.header.stamp = _time(stamp_of(index))
        pose.pose.position.x = index / 300.0
        pose.pose.orientation.w = 1.0
        status = TrackerStatus()
        status.header.stamp = pose.header.stamp
        status.serial_number = "LHR-TEST"
        status.device_connected = status.pose_valid = True
        status.tracking_state = TrackerStatus.TRACKING_RUNNING_OK
        gripper = GripperState()
        gripper.header.stamp = pose.header.stamp
        gripper.raw_openness = gripper.filtered_openness = index / 50.0
        gripper.detected_marker_count = 2
        gripper.valid = True
        for topic, message in (
            (TOPICS["tracker_pose"], pose),
            (TOPICS["tracker_status"], status),
            (TOPICS["gripper_state"], gripper),
        ):
            recorder._write(topic, serialize_message(message))
    recorder.close()
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(root / "bag"), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    records = []
    while reader.has_next():
        records.append(reader.read_next())
    reader.close()
    assert sum(t == TOPICS["image"] for t, _, _ in records) == 46
    assert sum(t == TOPICS["episode_event"] for t, _, _ in records) == 2
    assert [t for _, _, t in records] == sorted(t for _, _, t in records)
    decoder = BagImageDecoder(mode, COLOR_RGB)
    frames = []
    for topic, data, stamp in records:
        if topic == TOPICS["image"]:
            frames.extend(decoder.feed(bytes(data), stamp))
    frames.extend(decoder.drain())
    assert decoder.stats.skipped_before_keyframe == (6 if packets else 0)
    assert len(frames) == (40 if packets else 46)


@pytest.mark.parametrize("mode", ["compressed", "h264"])
def test_record_session_ros_topics_services_and_shutdown(tmp_path: Path, mode: str) -> None:
    """真实独立进程验证话题发现、边界服务、退出写出索引及快照可被 Link7 转换读取。"""
    from test_collection_node import _Publishers, _write_extrinsic

    extrinsic = tmp_path / "extrinsic.yaml"
    _write_extrinsic(extrinsic)
    rclpy.init()
    helper = rclpy.create_node("continuous_recording_smoke")
    publishers = _Publishers(mode)
    process = subprocess.Popen(
        [
            sys.executable, "-m", "fastumi_data.session_recorder",
            "--task", mode, "--session-id", "smoke",
            "--dataset-root", str(tmp_path), "--extrinsic", str(extrinsic),
            "--image-mode", mode,
        ],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True,
    )
    clients = {}  # 本测试的开始和停止服务客户端。
    try:
        for name in ("start", "stop"):
            clients[name] = helper.create_client(Trigger, f"/fastumi/episode/{name}")
            assert clients[name].wait_for_service(timeout_sec=15)

        def call(name: str) -> Trigger.Response:
            """调用真实服务，使用有界等待避免测试挂起。"""
            future = clients[name].call_async(Trigger.Request())
            rclpy.spin_until_future_complete(helper, future, timeout_sec=10)
            assert future.result() is not None, f"{mode}: {name} 超时"
            return future.result()

        deadline = time.monotonic() + 15
        while True:
            response = call("start")
            if response.success:
                break
            assert time.monotonic() < deadline, response.message
            time.sleep(0.1)
        image_topic = (
            "/umi_camera/image_raw/compressed" if mode == "compressed"
            else "/umi_camera/image_raw/ffmpeg"
        )
        infos = helper.get_subscriptions_info_by_topic(image_topic)
        # DDS 可能先发现端点再发现节点名；唯一图像订阅是本测试需要的约束。
        assert len(infos) == 1
        time.sleep(1.4)  # 30 Hz 合成输入，提供足够的 20 Hz 同步样本。
        assert call("stop").success
        process.send_signal(signal.SIGINT)
        output, _ = process.communicate(timeout=20)
        assert process.returncode == 0, output
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        publishers.stop()
        for client in clients.values():
            helper.destroy_client(client)
        helper.destroy_node()
        rclpy.shutdown()
    session = tmp_path / mode / "smoke"
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(session / "raw" / "bag"), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    metadata = {topic.name: topic for topic in reader.get_all_topics_and_types()}
    tf_qos = rosbag2_py.convert_rclcpp_qos_to_rclpy_qos(
        metadata["/tf_static"].offered_qos_profiles[0]
    )
    assert tf_qos.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert tf_qos.reliability == ReliabilityPolicy.RELIABLE
    reader.close()
    processing = resolve_processing(
        None, session / "calibration_snapshot" / "processing.yaml", default_processing_path()
    )
    assert processing.source == SOURCE_SNAPSHOT and processing.image_mode == mode
    assert (session / "raw" / "bag" / "metadata.yaml").is_file()
    decoder = BagImageDecoder(mode, COLOR_RGB)
    frames = []
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(session / "raw" / "bag"), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic == image_topic:
            frames.extend(decoder.feed(bytes(data), stamp))
    reader.close()
    frames.extend(decoder.drain())
    assert len(frames) >= 30
