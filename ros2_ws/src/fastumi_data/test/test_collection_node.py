"""用合成三路消息验证采集节点：真实服务、真实 MCAP 和 Link7 转换入口。"""

import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

import pytest
import yaml


# 在导入 rclpy 前隔离 DDS：随机域且仅本机，避免与真实设备或并行测试串话。
os.environ["ROS_DOMAIN_ID"] = str(random.randint(150, 232))
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"

import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from fastumi_interfaces.msg import CollectionStatus, EpisodeEvent  # noqa: E402
from fastumi_interfaces.srv import (  # noqa: E402
    CollectionCancel,
    CollectionDelete,
    CollectionGetStatus,
    CollectionList,
    CollectionStart,
    CollectionStop,
    CollectionStopAndSave,
)
from rclpy.serialization import deserialize_message  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
import rosbag2_py  # noqa: E402

from fastumi_data.bag_images import BagImageDecoder  # noqa: E402
from fastumi_data.link7_config import SOURCE_SNAPSHOT, resolve_processing  # noqa: E402
from fastumi_data.link7_record import precheck_tracker_identity  # noqa: E402
from fastumi_data.mcap_converter import (  # noqa: E402
    _snapshot_processing,
    default_processing_path,
    discover_sources,
)
from fastumi_data.video_decoding import COLOR_RGB  # noqa: E402


from synthetic_publisher import HEIGHT, TRACKER_SERIAL, WIDTH  # noqa: E402
# 服务调用等待上限。
CALL_TIMEOUT_S = 20.0


def _write_extrinsic(path: Path) -> None:
    """写入通过 2 mm、1° 验收的最小外参。"""
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "accepted": True,
                "tracker_serial": TRACKER_SERIAL,
                "tracker_to_tcp": {
                    "translation_m": [0.0, 0.0, 0.1],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "translation_rmse_mm": 1.0,
                "rotation_rmse_deg": 0.5,
                "time_offset_ms": 0.0,
            }
        ),
        encoding="utf-8",
    )


class _Publishers:
    """在独立进程中运行合成设备，通过标准输入切换夹爪有效性和图像开关。"""

    def __init__(self, image_mode: str = "raw") -> None:
        """启动发布进程并等待其就绪。"""
        self._process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("synthetic_publisher.py")),
                image_mode,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self._process.stdout.readline().strip() == "ready"

    def _send(self, command: str) -> None:
        """向发布进程发送一行控制命令。"""
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()

    def set_gripper_valid(self, value: bool) -> None:
        """切换夹爪 valid 字段，模拟估计失败。"""
        self._send(f"gripper_valid {int(value)}")

    def set_image_enabled(self, value: bool) -> None:
        """开关图像发布，模拟相机断流。"""
        self._send(f"image {int(value)}")

    def stop(self) -> None:
        """通知发布进程退出并等待。"""
        try:
            self._send("quit")
            self._process.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self._process.kill()


class _Rig:
    """启动采集节点和合成设备两个独立进程，并在本进程提供服务客户端。

    三个角色分属不同进程，与实机拓扑一致；同进程内的可靠发布/订阅会因 GIL
    与 DDS 内部锁交织而阻塞，不能代表真实部署。
    """

    def __init__(
        self,
        root: Path,
        extrinsic: Optional[Path],
        image_mode: str = "raw",
        node_arguments: tuple = (),
    ) -> None:
        """启动节点与合成设备，并等待服务可用。

        Args:
            image_mode: 合成设备发布的图像模式，也是采集节点的默认模式。
            node_arguments: 追加给采集节点的 ``--ros-args`` 参数，可覆盖模式。
        """
        self.root = root
        self.extrinsic = extrinsic
        self.image_mode = image_mode
        self.node_arguments = node_arguments
        self.helper = rclpy.create_node("collection_test_helper")
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.helper)
        self._spinner = threading.Thread(target=self.executor.spin, daemon=True)
        self._spinner.start()
        self.publishers = _Publishers(image_mode)
        self.clients = {}
        self.node_process = self.launch_node()

    def launch_node(self) -> subprocess.Popen:
        """启动采集节点进程，重建服务客户端并等待服务可用。

        重启后必须重建客户端：旧客户端仍匹配已退出的服务端，请求会无人应答。
        """
        arguments = [
            sys.executable,
            "-m",
            "fastumi_data.collection_node",
            "--ros-args",
            "-p", f"dataset_root:={self.root}",
            "-p", "min_free_disk_gib:=0.0",
            "-p", "status_rate_hz:=10.0",
        ]
        if self.extrinsic is not None:
            arguments += ["-p", f"extrinsic_path:={self.extrinsic}"]
        if self.image_mode != "raw":
            arguments += ["-p", f"image_mode:={self.image_mode}"]
        arguments += list(self.node_arguments)
        process = subprocess.Popen(arguments)
        for client in self.clients.values():
            self.helper.destroy_client(client)
        self.clients = {
            name: self.helper.create_client(srv, f"/fastumi/collection/{name}")
            for name, srv in (
                ("start", CollectionStart),
                ("stop", CollectionStop),
                ("cancel", CollectionCancel),
                ("stop_and_save", CollectionStopAndSave),
                ("delete", CollectionDelete),
                ("list", CollectionList),
                ("get_status", CollectionGetStatus),
            )
        }
        for client in self.clients.values():
            assert client.wait_for_service(timeout_sec=CALL_TIMEOUT_S)
        return process

    def call(self, name: str, request, timeout_s: float = CALL_TIMEOUT_S):
        """同步调用服务；超时返回 ``None`` 而不是断言，供重启场景重试。"""
        future = self.clients[name].call_async(request)
        deadline = time.monotonic() + timeout_s
        while not future.done():
            if time.monotonic() >= deadline:
                self.clients[name].remove_pending_request(future)
                return None
            time.sleep(0.01)
        return future.result()

    def checked(self, name: str, request):
        """同步调用服务，超时即测试失败。"""
        response = self.call(name, request)
        assert response is not None, f"服务 {name} 超时"
        return response

    def wait_for(self, predicate: Callable[[], bool], timeout_s: float = 15.0) -> None:
        """轮询等待条件成立。"""
        deadline = time.monotonic() + timeout_s
        while not predicate():
            assert time.monotonic() < deadline, "等待条件超时"
            time.sleep(0.05)

    def status(self) -> CollectionStatus:
        """查询权威状态；超时视为测试失败。"""
        response = self.call("get_status", CollectionGetStatus.Request())
        assert response is not None, "服务 get_status 超时"
        return response.status

    def status_or_none(self, timeout_s: float = 2.0) -> Optional[CollectionStatus]:
        """查询权威状态，超时返回 ``None``。"""
        response = self.call("get_status", CollectionGetStatus.Request(), timeout_s)
        return None if response is None else response.status

    def start(self, request_id: str, task: str = "demo", name: str = ""):
        """发送开始请求并返回结果。"""
        request = CollectionStart.Request()
        request.request_id, request.task_name, request.name = request_id, task, name
        return self.checked("start", request).result

    def modify(self, service: str, request_id: str, uuid: str):
        """发送携带 UUID 的修改请求并返回结果。"""
        request = {"stop": CollectionStop, "cancel": CollectionCancel,
                   "stop_and_save": CollectionStopAndSave,
                   "delete": CollectionDelete}[service].Request()
        request.request_id, request.collection_uuid = request_id, uuid
        return self.checked(service, request).result

    def terminate_node(self, sig: int = signal.SIGINT) -> int:
        """向节点进程发信号并等待退出，返回退出码。"""
        self.node_process.send_signal(sig)
        try:
            return self.node_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.node_process.kill()
            raise

    def close(self) -> None:
        """有序退出：停止合成设备，SIGINT 节点，销毁客户端。"""
        self.publishers.stop()
        if self.node_process.poll() is None:
            self.terminate_node()
        self.executor.shutdown()
        self.helper.destroy_node()


@pytest.fixture()
def rig(tmp_path):
    """初始化 rclpy，创建带合格外参的完整测试台架。"""
    rclpy.init()
    extrinsic = tmp_path / "tracker_to_tcp.yaml"
    _write_extrinsic(extrinsic)
    bench = _Rig(tmp_path / "dataset", extrinsic)
    try:
        bench.wait_for(lambda: bench.status().can_start, timeout_s=30.0)
        yield bench
    finally:
        bench.close()
        rclpy.shutdown()


def _read_bag(bag_uri: Path):
    """读取 MCAP，返回话题类型映射和 (话题, 反序列化前字节, 日志时间) 列表。"""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_uri), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    messages = []
    while reader.has_next():
        messages.append(reader.read_next())
    return types, messages


def test_full_flow_writes_closed_mcap_that_link7_converter_can_read(
    rig, tmp_path
) -> None:
    """验证真实服务流程产生完整 MCAP，其来源、快照配置和 Tracker 身份可被 Link7 转换读取。"""
    status = rig.status()
    assert status.state == "idle" and status.can_start
    assert status.image.fresh and status.tracker.fresh and status.gripper.fresh
    assert status.calibration_status == "calibrated"

    started = rig.start("req-start", task="demo", name="合成数据")
    assert started.accepted and started.completed and started.state == "recording"
    uuid = started.collection_uuid
    time.sleep(3.0)
    recording = rig.status()
    assert recording.state == "recording" and recording.collection_uuid == uuid
    assert recording.image_messages > 30 and recording.tracker_messages > 100
    assert recording.gripper_valid and 49.0 < recording.gripper_percent < 51.0
    assert recording.gripper.matched and recording.tracker.matched
    assert recording.gripper.source_delta_ms == 0.0

    saved = rig.modify("stop_and_save", "req-save", uuid)
    assert saved.accepted and saved.completed and saved.state == "idle", saved
    assert saved.state_version > started.state_version

    final = rig.status()
    assert final.state == "idle" and final.last_request_id == "req-save"
    assert final.last_result_code == "OK"
    listing = CollectionList.Request()
    listing.task_name = "demo"
    records = rig.checked("list", listing)
    assert records.success and records.total == 1
    record = records.records[0]
    assert record.collection_uuid == uuid and record.name == "合成数据"
    assert record.calibrated and record.image_messages >= recording.image_messages

    record_dir = tmp_path / "dataset" / record.relative_path
    manifest = yaml.safe_load((record_dir / "session.yaml").read_text("utf-8"))
    assert manifest["collection_uuid"] == uuid and manifest["state"] == "saved"
    assert manifest["calibration_status"] == "calibrated"
    assert (record_dir / "calibration_snapshot" / "tracker_to_tcp.yaml").is_file()

    types, messages = _read_bag(record_dir / "raw" / "bag")
    assert types == {
        "/umi_camera/image_raw": "sensor_msgs/msg/Image",
        "/vive_tracker/pose": "geometry_msgs/msg/PoseStamped",
        "/vive_tracker/status": "fastumi_interfaces/msg/TrackerStatus",
        "/gripper/state": "fastumi_interfaces/msg/GripperState",
        "/fastumi/episode/events": "fastumi_interfaces/msg/EpisodeEvent",
        "/tf_static": "tf2_msgs/msg/TFMessage",
    }
    events = [
        (stamp, deserialize_message(data, EpisodeEvent))
        for topic, data, stamp in messages
        if topic == "/fastumi/episode/events"
    ]
    assert [e.event_type for _, e in events] == [EpisodeEvent.START, EpisodeEvent.STOP]
    assert all(e.session_id == uuid and e.episode_index == 0 for _, e in events)
    assert all(e.task_name == "demo" for _, e in events)
    start_ns, stop_ns = events[0][0], events[1][0]
    assert [m[2] for m in messages] == sorted(m[2] for m in messages)
    assert messages[0][0] == "/fastumi/episode/events" and messages[0][2] == start_ns
    assert messages[-1][0] == "/fastumi/episode/events" and messages[-1][2] == stop_ns
    counts = {topic: 0 for topic in types}
    for topic, _, stamp in messages:
        counts[topic] += 1
        assert start_ns <= stamp <= stop_ns
    assert counts["/umi_camera/image_raw"] == manifest["message_counts"]["image"]
    assert counts["/vive_tracker/pose"] == manifest["message_counts"]["tracker_pose"]
    assert counts["/gripper/state"] == manifest["message_counts"]["gripper_state"]
    assert counts["/tf_static"] >= 1
    assert counts["/umi_camera/image_raw"] > 30
    info = rosbag2_py.Info().read_metadata(str(record_dir / "raw" / "bag"), "mcap")
    assert info.message_count == len(messages)
    first_image = next(
        deserialize_message(d, Image) for t, d, _ in messages if t == "/umi_camera/image_raw"
    )
    assert (first_image.width, first_image.height) == (WIDTH, HEIGHT)

    # Link7 转换直接发现该记录，消费快照中的处理配置，并预检 Tracker 设备身份。
    [source] = discover_sources(record_dir)
    assert source.record_dir == record_dir.resolve()
    assert source.bag_uri == (record_dir / "raw" / "bag").resolve()
    processing = resolve_processing(
        None, _snapshot_processing(source), default_processing_path()
    )
    assert processing.source == SOURCE_SNAPSHOT and processing.image_mode == "raw"
    identity = precheck_tracker_identity(
        source.bag_uri, processing.topics["tracker_status"], TRACKER_SERIAL
    )
    assert identity["observed_tracker_serial"] == TRACKER_SERIAL
    assert identity["status_messages"] > 100


def test_cancel_and_delete_flow_over_real_services(rig, tmp_path) -> None:
    """验证待保存取消、保存后删除和重复请求经真实服务保持幂等。"""
    first = rig.start("a-start")
    assert first.completed
    duplicate = rig.start("a-start")
    assert duplicate.collection_uuid == first.collection_uuid
    stopped = rig.modify("stop", "a-stop", first.collection_uuid)
    assert stopped.state == "pending"
    blocked = rig.start("a-start2")
    assert not blocked.accepted and blocked.code == "PENDING_RECORD"
    status = rig.status()
    assert status.can_save and status.can_cancel and not status.can_start

    cancelled = rig.modify("cancel", "a-cancel", first.collection_uuid)
    assert cancelled.completed and cancelled.state == "idle"
    assert list((tmp_path / "dataset" / ".staging").iterdir()) == []

    rig.wait_for(lambda: rig.status().can_start)
    second = rig.start("b-start", task="other")
    time.sleep(1.0)
    saved = rig.modify("stop_and_save", "b-save", second.collection_uuid)
    assert saved.completed
    assert not rig.modify("delete", "b-del-bad", first.collection_uuid).accepted
    deleted = rig.modify("delete", "b-del", second.collection_uuid)
    assert deleted.completed and deleted.code == "OK"
    records = rig.checked("list", CollectionList.Request())
    assert records.total == 0
    assert any((tmp_path / "dataset" / ".trash").iterdir())


def test_invalid_gripper_blocks_start_and_raises_alarm_while_recording(rig) -> None:
    """验证夹爪 NaN/无效阻止开始；采集中出现则继续记录并报警留档。"""
    rig.publishers.set_gripper_valid(False)
    rig.wait_for(lambda: not rig.status().can_start)
    status = rig.status()
    assert any("夹爪" in blocker for blocker in status.start_blockers)
    refused = rig.start("g-start")
    assert not refused.accepted and refused.code == "PREFLIGHT_FAILED"

    rig.publishers.set_gripper_valid(True)
    rig.wait_for(lambda: rig.status().can_start)
    started = rig.start("g-start2")
    assert started.completed
    rig.publishers.set_gripper_valid(False)
    rig.wait_for(lambda: "GRIPPER_INVALID" in rig.status().alarms)
    assert rig.status().state == "recording"
    time.sleep(0.5)  # 让周期统计至少记录一次报警。
    saved = rig.modify("stop_and_save", "g-save", started.collection_uuid)
    assert saved.completed
    records = rig.checked("list", CollectionList.Request())
    assert records.records[0].has_alarms


def test_image_stream_loss_blocks_start_after_timeout(rig) -> None:
    """验证相机断流超过 1 s 后不可开始，恢复后可开始。"""
    rig.publishers.set_image_enabled(False)
    rig.wait_for(lambda: not rig.status().image.fresh, timeout_s=10.0)
    assert not rig.status().can_start
    rig.publishers.set_image_enabled(True)
    rig.wait_for(lambda: rig.status().can_start)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGKILL])
def test_restart_cleans_unsaved_data_and_keeps_saved(tmp_path, sig) -> None:
    """验证有序退出和崩溃后重启都会清理未保存采集，已保存记录不受影响。"""
    rclpy.init()
    extrinsic = tmp_path / "tracker_to_tcp.yaml"
    _write_extrinsic(extrinsic)
    root = tmp_path / "dataset"
    bench = _Rig(root, extrinsic)
    try:
        bench.wait_for(lambda: bench.status().can_start, timeout_s=30.0)
        saved = bench.start("k-start")
        time.sleep(0.8)
        bench.modify("stop_and_save", "k-save", saved.collection_uuid)
        bench.wait_for(lambda: bench.status().can_start)
        unsaved = bench.start("k-start2")
        assert unsaved.completed
        assert len(list((root / ".staging").iterdir())) == 1
        bench.terminate_node(sig)
        if sig == signal.SIGINT:
            assert list((root / ".staging").iterdir()) == []

        bench.node_process = bench.launch_node()
        # SIGKILL 后旧服务端在 DDS 租约到期前仍被匹配，请求可能无人应答，需重试。
        bench.wait_for(
            lambda: (bench.status_or_none() or CollectionStatus()).state == "idle",
            timeout_s=60.0,
        )
        assert list((root / ".staging").iterdir()) == []
        records = bench.checked("list", CollectionList.Request())
        assert [r.collection_uuid for r in records.records] == [saved.collection_uuid]
    finally:
        bench.close()
        rclpy.shutdown()


def test_second_node_on_same_root_exits_with_lock_error(rig) -> None:
    """验证同一数据集根目录只允许一个采集服务，第二个以明确错误退出。"""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fastumi_data.collection_node",
            "--ros-args",
            "-p", f"dataset_root:={rig.root}",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode != 0
    assert "ROOT_LOCKED" in result.stderr
    assert rig.status().state in ("idle", "recording")


@pytest.mark.parametrize("image_mode", ["compressed", "h264"])
def test_encoded_image_flow_records_messages_and_decodes_to_rgb(
    tmp_path, image_mode
) -> None:
    """JPEG 与 H.264 经真实节点采集：原样记录，H.264 从关键帧开始，可解码为 RGB 帧。"""
    from fastumi_data.cdr_header import read_ffmpeg_packet_summary
    from fastumi_data.image_modes import IMAGE_MESSAGE_TYPES
    from synthetic_publisher import IMAGE_TOPICS, VIDEO_HEIGHT, VIDEO_WIDTH

    rclpy.init()
    extrinsic = tmp_path / "tracker_to_tcp.yaml"
    _write_extrinsic(extrinsic)
    bench = _Rig(tmp_path / "dataset", extrinsic, image_mode)
    topic = IMAGE_TOPICS[image_mode]
    try:
        bench.wait_for(lambda: bench.status().can_start, timeout_s=30.0)
        started = bench.start("v-start", task="video")
        assert started.completed and started.state == "recording", started
        time.sleep(3.0)
        recording = bench.status()
        assert recording.image_messages > 30
        assert recording.image.valid and recording.image.detail.startswith(
            "h264 64x48" if image_mode == "h264" else "jpeg"
        )
        saved = bench.modify("stop_and_save", "v-save", started.collection_uuid)
        assert saved.completed, saved
        records = bench.checked("list", CollectionList.Request())
        record_dir = tmp_path / "dataset" / records.records[0].relative_path
    finally:
        bench.close()
        rclpy.shutdown()

    types, messages = _read_bag(record_dir / "raw" / "bag")
    assert types[topic] == IMAGE_MESSAGE_TYPES[image_mode]
    assert messages[0][0] == "/fastumi/episode/events"
    images = [(data, stamp) for name, data, stamp in messages if name == topic]
    if image_mode == "h264":
        first = read_ffmpeg_packet_summary(bytes(images[0][0]))
        start_event = deserialize_message(bytes(messages[0][1]), EpisodeEvent)
        start_ns = (
            start_event.header.stamp.sec * 1_000_000_000
            + start_event.header.stamp.nanosec
        )
        assert first.keyframe and first.stamp_ns == start_ns
    [source] = discover_sources(record_dir)
    processing = resolve_processing(
        None, _snapshot_processing(source), default_processing_path()
    )
    assert processing.source == SOURCE_SNAPSHOT
    assert processing.image_mode == image_mode and processing.topics["image"] == topic
    decoder = BagImageDecoder(image_mode, COLOR_RGB)
    frames = []
    for data, stamp in images:
        frames.extend(decoder.feed(bytes(data), stamp))
    frames.extend(decoder.drain())
    assert len(frames) > 30
    assert frames[0].image.shape == (VIDEO_HEIGHT, VIDEO_WIDTH, 3)
    if image_mode == "h264":
        assert decoder.stats.skipped_before_keyframe == 0


@pytest.mark.parametrize(
    "node_arguments, expected",
    [
        (
            ("-p", "image_mode:=compressed", "-p", "image_topic:=/umi_camera/image_raw"),
            ("sensor_msgs/msg/Image", "image_mode=compressed"),
        ),
        (("-p", "image_reliability:=reliable"), ("RELIABLE", "best_effort")),
    ],
)
def test_image_publisher_incompatibility_blocks_start(
    tmp_path, node_arguments, expected
) -> None:
    """发布端消息类型或可靠性与配置不兼容时，开始前检查给出明确原因。"""
    rclpy.init()
    extrinsic = tmp_path / "tracker_to_tcp.yaml"
    _write_extrinsic(extrinsic)
    bench = _Rig(tmp_path / "dataset", extrinsic, "raw", node_arguments)
    try:
        bench.wait_for(
            lambda: any(
                all(fragment in blocker for fragment in expected)
                for blocker in bench.status_or_none().start_blockers
            ),
            timeout_s=30.0,
        )
        refused = bench.start("m-start")
        assert not refused.accepted and refused.code == "PREFLIGHT_FAILED"
        assert not bench.status().can_start
    finally:
        bench.close()
        rclpy.shutdown()


@pytest.mark.parametrize("calibration_mode", [False, True])
@pytest.mark.parametrize("image_mode", ["raw", "compressed"])
def test_missing_gripper_is_optional_only_in_calibration_mode(
    tmp_path, calibration_mode, image_mode
):
    """使用无发布者的夹爪话题验证真实服务预检，并回读标定模式保存的 MCAP。"""
    rclpy.init()
    bench = _Rig(
        tmp_path / "dataset", None, image_mode=image_mode,
        node_arguments=(
            "-p", f"calibration_mode:={str(calibration_mode).lower()}",
            "-p", "gripper_state_topic:=/calibration_test/no_gripper",
        ),
    )
    try:
        bench.wait_for(
            lambda: bench.status().image.received >= 70
            and bench.status().tracker_tracking_ok,
            timeout_s=30.0,
        )
        if not calibration_mode:
            result = bench.start("missing-gripper")
            assert not result.accepted and result.code == "PREFLIGHT_FAILED"
            assert "夹爪话题没有发布者" in result.message
            assert not bench.status().can_start
            return
        bench.wait_for(lambda: bench.status().can_start, timeout_s=30.0)
        started = bench.start("calibration-start", task="gripper")
        assert started.accepted and started.completed
        bench.wait_for(lambda: bench.status().image_messages >= 10)
        saved = bench.modify("stop_and_save", "calibration-save", started.collection_uuid)
        assert saved.accepted and saved.completed
        session_path = next(bench.root.glob("gripper/*/session.yaml"))
        manifest = yaml.safe_load(session_path.read_text())
        assert manifest["calibration_mode"] is True
        assert manifest["message_counts"]["image"] > 0
        assert manifest["message_counts"]["tracker_pose"] > 0
        assert manifest["message_counts"]["gripper_state"] == 0
        _, messages = _read_bag(session_path.parent / manifest["bag_uri"])
        assert messages
        quality = yaml.safe_load((session_path.parent / "quality.yaml").read_text())
        assert quality["required_streams"] == ["image", "tracker"]
        assert not any(code.startswith("GRIPPER_") for code in quality["alarms"])
    finally:
        bench.close()
        rclpy.shutdown()
