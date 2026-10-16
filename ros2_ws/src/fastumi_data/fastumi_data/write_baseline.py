"""在 ROS 2 系统 Python 中测量真实 MCAP 写入和入口回调时延。"""

from __future__ import annotations

import argparse
import ctypes
from collections import Counter
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import shutil
import threading
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


# 基准使用的固定双话题。
IMAGE_TOPIC = "/fastumi_baseline/image"
STRING_TOPIC = "/fastumi_baseline/status"
# 系统 Python 中 ROS 消息的类型名。
MESSAGE_TYPES = {
    "raw": "sensor_msgs/msg/Image",
    "compressed": "sensor_msgs/msg/CompressedImage",
    "h264": "ffmpeg_image_transport_msgs/msg/FFMPEGPacket",
}
# 源端 IPC 队列容量。
SOURCE_QUEUE_DEPTH = 2
# 与生产 QueuedBagWriter 一致的默认容量。
QUEUED_MAX_MESSAGES = 600
QUEUED_MAX_BYTES = 768 * 1024 * 1024
# 每条 MCAP 记录及摘要文件的保守磁盘预留。
DISK_BYTES_PER_MESSAGE_RESERVE = 32 * 1024
DISK_METADATA_RESERVE = 1 * 1024 * 1024
# 为避免单条 IPC 消息过大，限制图像像素数。
MAX_PIXELS = 4_194_304


def percentile_summary(values: Iterable[int]) -> Dict[str, Any]:
    """汇总纳秒序列的数量、均值、分位数和最大值。

    Args:
        values: 非负时长或相邻时间戳间隔，单位纳秒。
    Returns:
        可直接序列化为 JSON 的统计字典；空序列的统计值为 ``None``。
    """
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"count": 0, "mean_ns": None, "p50_ns": None,
                "p95_ns": None, "p99_ns": None, "max_ns": None}

    def quantile(fraction: float) -> float:
        """按线性插值取分位数。"""
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (
            position - lower
        )

    return {
        "count": len(ordered),
        "mean_ns": sum(ordered) / len(ordered),
        "p50_ns": quantile(0.50),
        "p95_ns": quantile(0.95),
        "p99_ns": quantile(0.99),
        "max_ns": ordered[-1],
    }


def validate_settings(
    path: str,
    mode: str,
    duration_s: float,
    fps: float,
    width: int,
    height: int,
    max_input_mib: float,
    max_disk_mib: float,
) -> Dict[str, Any]:
    """校验一次基准的路径、模式、分辨率和硬资源上限。

    Args:
        path: ``queued`` 或 ``continuous``。
        mode: ``raw``、``compressed`` 或 ``h264``。
        duration_s: 最长录制时长，最多 600 秒。
        fps: 源端计划频率，最多 240 Hz。
        width: 图像宽度。
        height: 图像高度。
        max_input_mib: 本轮最多交给入口的序列化字节数。
        max_disk_mib: 输出目录允许占用的最大字节数。
    Returns:
        标准化配置、帧数上限及输入/磁盘有效预算。
    Raises:
        ValueError: 参数无效、模式不受支持或磁盘预留超过上限。
    """
    if path not in ("queued", "continuous"):
        raise ValueError("--path 必须是 queued 或 continuous")
    if mode not in MESSAGE_TYPES:
        raise ValueError("--mode 必须是 raw、compressed 或 h264")
    if path == "continuous" and mode == "raw":
        raise ValueError("continuous 只支持 compressed 或 h264；raw 不适用")
    if not math.isfinite(duration_s) or not 0 < duration_s <= 600:
        raise ValueError("--duration-s 必须大于 0 且不超过 600")
    if not math.isfinite(fps) or not 0 < fps <= 240:
        raise ValueError("--fps 必须大于 0 且不超过 240")
    if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
        raise ValueError(f"图像尺寸必须为正且总像素不超过 {MAX_PIXELS}")
    if mode == "h264" and (width % 2 or height % 2):
        raise ValueError("H.264 要求宽高均为偶数")
    if not math.isfinite(max_input_mib) or not 0 < max_input_mib <= 8192:
        raise ValueError("--max-input-mib 必须大于 0 且不超过 8192")
    if not math.isfinite(max_disk_mib) or not 0 < max_disk_mib <= 16384:
        raise ValueError("--max-disk-mib 必须大于 0 且不超过 16384")

    frame_limit = math.ceil(duration_s * fps)
    message_limit = frame_limit * 2
    disk_limit = int(max_disk_mib * 1024 * 1024)
    safe_payload_limit = (
        disk_limit
        - message_limit * DISK_BYTES_PER_MESSAGE_RESERVE
        - DISK_METADATA_RESERVE
    )
    if safe_payload_limit <= 0:
        raise ValueError("磁盘上限不足以容纳时长对应的记录预留和 MCAP 元数据")
    input_limit = min(int(max_input_mib * 1024 * 1024), safe_payload_limit)
    return {
        "path": path,
        "mode": mode,
        "duration_s": float(duration_s),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "frame_limit": frame_limit,
        "message_limit": message_limit,
        "max_input_bytes": input_limit,
        "max_disk_bytes": disk_limit,
        "disk_payload_reserve_bytes": disk_limit - safe_payload_limit,
    }


def _directory_size(directory: Path) -> int:
    """返回诊断输出目录中当前文件大小总和。"""
    return sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())


class _TimedBackend:
    """计时并检查磁盘上限的真实 MCAP 后端代理。"""

    def __init__(self, backend: Any, bag_path: Path, disk_limit: int) -> None:
        """包装生产后端并初始化写入轨迹。"""
        self._backend = backend
        self._bag_path = bag_path
        self._disk_limit = disk_limit
        self.trace: List[Dict[str, Any]] = []
        self.close_timing: Optional[Dict[str, Any]] = None
        self._trace_lock = threading.Lock()
        self._written_payload_bytes = 0

    def open(self, uri: str, topics: Sequence[Any]) -> None:
        """原样转发生产后端的打开调用。"""
        self._backend.open(uri, topics)

    def write(self, topic: str, data: bytes, timestamp_ns: int) -> None:
        """调用真实 MCAP 写入，并记录单调时钟起止与结果。"""
        payload = bytes(data)
        if self._written_payload_bytes + len(payload) > self._disk_limit:
            raise OSError("达到 --max-disk-mib 的序列化输入预算")
        if _directory_size(self._bag_path.parent) + len(payload) > self._disk_limit:
            raise OSError("写入前检查发现 --max-disk-mib 余量不足")
        started_ns = time.monotonic_ns()
        error = ""
        write_succeeded = False
        ended_ns = started_ns
        try:
            self._backend.write(topic, payload, timestamp_ns)
            write_succeeded = True
            self._written_payload_bytes += len(payload)
        except Exception as caught:
            error = str(caught)
            raise
        finally:
            ended_ns = time.monotonic_ns()
            record = {
                "topic": topic,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "timestamp_ns": int(timestamp_ns),
                "start_ns": started_ns,
                "end_ns": ended_ns,
                "duration_ns": ended_ns - started_ns,
                "ok": write_succeeded,
                "error": error,
            }
            with self._trace_lock:
                self.trace.append(record)
        if _directory_size(self._bag_path.parent) > self._disk_limit:
            raise OSError("MCAP 输出目录已达到 --max-disk-mib")

    def close(self) -> None:
        """关闭真实后端并记录索引落盘的耗时。"""
        started_ns = time.monotonic_ns()
        error = ""
        try:
            self._backend.close()
        except Exception as caught:
            error = str(caught)
            raise
        finally:
            ended_ns = time.monotonic_ns()
            self.close_timing = {
                "start_ns": started_ns,
                "end_ns": ended_ns,
                "duration_ns": ended_ns - started_ns,
                "ok": not error,
                "error": error,
            }


def _source_process(
    config: Mapping[str, Any],
    events: Any,
    ready: Any,
    start_event: Any,
    stop_event: Any,
    start_ns_shared: Any,
    stamps: Any,
    counters: Any,
    base_stamp_ns: int,
) -> None:
    """在 spawn 子进程中定时生成并投送合法 ROS 序列化消息。"""
    source_error = ""
    try:
        import av
        import cv2
        import numpy as np
        cv2.setNumThreads(1)
        from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
        from rclpy.serialization import serialize_message
        from sensor_msgs.msg import CompressedImage, Image
        from std_msgs.msg import String
        from fractions import Fraction

        encoder = None
        if config["mode"] == "h264":
            encoder = av.CodecContext.create("libx264", "w")
            encoder.width = config["width"]
            encoder.height = config["height"]
            encoder.pix_fmt = "yuv420p"
            encoder.time_base = Fraction(1, max(1, int(round(config["fps"]))))
            encoder.thread_count = 1
            encoder.gop_size = max(1, int(round(config["fps"])))
            encoder.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "profile": "baseline",
                "x264-params": "scenecut=0:bframes=0:repeat-headers=1",
            }
            encoder.open()
        rng = np.random.default_rng(20261016)
        ready.set()
        if not start_event.wait(30):
            raise TimeoutError("主进程未启动源端调度")

        start_ns = int(start_ns_shared.value)
        period_ns = int(1_000_000_000 / config["fps"])
        run_end_ns = start_ns + int(config["duration_s"] * 1_000_000_000)
        max_bytes = int(config["max_input_bytes"])
        sent_bytes = 0
        for frame_index in range(config["frame_limit"]):
            if stop_event.is_set() or time.monotonic_ns() >= run_end_ns:
                break
            planned_ns = start_ns + frame_index * period_ns
            if planned_ns >= run_end_ns:
                break
            if stop_event.wait(max(0, planned_ns - time.monotonic_ns()) / 1e9):
                break
            if time.monotonic_ns() >= run_end_ns:
                break
            counters[0] += 1  # 已到计划采样时刻的帧数。
            stamp_ns = base_stamp_ns + frame_index * period_ns
            pixels = rng.integers(
                0, 256, (config["height"], config["width"], 3), dtype=np.uint8
            )
            if config["mode"] == "raw":
                image = Image()
                image.header.stamp.sec, image.header.stamp.nanosec = divmod(
                    stamp_ns, 1_000_000_000
                )
                image.header.frame_id = f"baseline_{frame_index}"
                image.height = config["height"]
                image.width = config["width"]
                image.encoding = "bgr8"
                image.step = config["width"] * 3
                image.data = pixels.tobytes()
            elif config["mode"] == "compressed":
                ok, encoded = cv2.imencode(
                    ".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                if not ok:
                    raise RuntimeError("OpenCV 无法编码 JPEG")
                image = CompressedImage()
                image.header.stamp.sec, image.header.stamp.nanosec = divmod(
                    stamp_ns, 1_000_000_000
                )
                image.header.frame_id = f"baseline_{frame_index}"
                image.format = "jpeg"
                image.data = encoded.tobytes()
            else:
                frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
                frame.pts = frame_index
                packets = encoder.encode(frame)
                if len(packets) != 1:
                    raise RuntimeError(
                        f"H.264 编码器每帧需输出一个访问单元，实际 {len(packets)} 个"
                    )
                image = FFMPEGPacket()
                image.header.stamp.sec, image.header.stamp.nanosec = divmod(
                    stamp_ns, 1_000_000_000
                )
                image.header.frame_id = f"baseline_{frame_index}"
                image.width = config["width"]
                image.height = config["height"]
                image.encoding = "h264;bgr8;bgr8;bgr8"
                image.pts = frame_index
                image.flags = 0
                image.is_bigendian = False
                image.data = bytes(packets[0])
            image_data = bytes(serialize_message(image))
            counters[1] += 1  # 已生成的图像消息数。
            text = String(data=f"frame={frame_index};stamp_ns={stamp_ns}")
            string_data = bytes(serialize_message(text))
            counters[2] += 1  # 已生成的辅助 String 数。
            generated_ns = time.monotonic_ns()
            if time.monotonic_ns() >= run_end_ns:
                break
            if sent_bytes + len(image_data) + len(string_data) > max_bytes:
                counters[5] = 1  # 序列化输入预算达到上限。
                break

            for slot, topic, payload in (
                (0, IMAGE_TOPIC, image_data),
                (1, STRING_TOPIC, string_data),
            ):
                record_index = frame_index * 2 + slot
                offset = record_index * 4
                stamps[offset] = planned_ns
                stamps[offset + 1] = generated_ns
                stamps[offset + 2] = time.monotonic_ns()
                event = {
                    "topic": topic,
                    "data": payload,
                    "timestamp_ns": stamp_ns,
                    "frame_index": frame_index,
                    "planned_ns": planned_ns,
                    "generated_ns": generated_ns,
                    "send_started_ns": stamps[offset + 2],
                }
                events.put(event, timeout=30)
                stamps[offset + 3] = time.monotonic_ns()
                counters[3] += 1  # 完成 Queue.put 的消息数。
                sent_bytes += len(payload)
                counters[4] = sent_bytes
        if encoder is not None:
            encoder.encode(None)
    except BaseException as error:
        source_error = f"{type(error).__name__}: {error}"
        counters[6] = 1
    finally:
        try:
            events.put({"kind": "source_status", "error": source_error}, timeout=30)
            events.put(None, timeout=30)
        except Exception:
            counters[6] = 1


def _heartbeat_loop(stop_event: threading.Event, samples: List[int]) -> None:
    """在基准主进程内独立采样 Python 心跳线程的单调时间。"""
    while not stop_event.is_set():
        samples.append(time.monotonic_ns())
        stop_event.wait(0.01)
    samples.append(time.monotonic_ns())


def _depth_loop(
    stop_event: threading.Event, writer: Any, samples: List[Dict[str, int]]
) -> None:
    """以 1 ms 周期采样普通队列深度。"""
    while not stop_event.is_set():
        samples.append({"time_ns": time.monotonic_ns(), "depth": writer.depth})
        stop_event.wait(0.001)


def _readback(
    bag_path: Path,
    mode: str,
    accepted: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """用真实 rosbag2 SequentialReader 核对话题、字节、顺序和时间戳。"""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image, CompressedImage
    from ffmpeg_image_transport_msgs.msg import FFMPEGPacket

    message_class = {
        "raw": Image,
        "compressed": CompressedImage,
        "h264": FFMPEGPacket,
    }[mode]
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topic_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }
    expected_types = {
        IMAGE_TOPIC: MESSAGE_TYPES[mode],
        STRING_TOPIC: "std_msgs/msg/String",
    }
    topic_type_errors = {
        topic: {"expected": expected_type, "actual": topic_types.get(topic)}
        for topic, expected_type in expected_types.items()
        if topic_types.get(topic) != expected_type
    }
    records: List[Dict[str, Any]] = []
    try:
        while reader.has_next():
            topic, raw_data, log_stamp_ns = reader.read_next()
            data = bytes(raw_data)
            header_stamp_ns = None
            if topic == IMAGE_TOPIC:
                image = deserialize_message(data, message_class)
                header_stamp_ns = (
                    int(image.header.stamp.sec) * 1_000_000_000
                    + int(image.header.stamp.nanosec)
                )
            records.append({
                "topic": topic,
                "sha256": hashlib.sha256(data).hexdigest(),
                "log_stamp_ns": int(log_stamp_ns),
                "header_stamp_ns": header_stamp_ns,
            })
    finally:
        reader.close()

    accepted_keys = [(item["topic"], item["sha256"]) for item in accepted]
    actual_keys = [(item["topic"], item["sha256"]) for item in records]
    missing = Counter(accepted_keys) - Counter(actual_keys)
    extra = Counter(actual_keys) - Counter(accepted_keys)
    queued_stamp_errors = 0
    header_errors = 0
    for record, expected in zip(records, accepted):
        if record["topic"] == IMAGE_TOPIC and (
            record["header_stamp_ns"] != expected["timestamp_ns"]
        ):
            header_errors += 1
        if expected.get("path") == "queued" and (
            record["log_stamp_ns"] != expected["timestamp_ns"]
        ):
            queued_stamp_errors += 1
    log_stamps = [item["log_stamp_ns"] for item in records]
    return {
        "counts_by_topic": dict(Counter(item["topic"] for item in records)),
        "message_count": len(records),
        "topic_types": topic_types,
        "expected_topic_types": expected_types,
        "topic_type_errors": topic_type_errors,
        "topic_types_match": not topic_type_errors,
        "exact_bytes_topics_and_fifo": actual_keys == accepted_keys,
        "missing_count": sum(missing.values()),
        "extra_count": sum(extra.values()),
        "missing_examples": [list(key) for key in list(missing)[:20]],
        "queued_timestamp_errors": queued_stamp_errors,
        "image_header_stamp_errors": header_errors,
        "continuous_log_timestamps_nondecreasing": all(
            left <= right for left, right in zip(log_stamps, log_stamps[1:])
        ),
        "records": records,
    }


def _overlap_report(
    writes: Sequence[Mapping[str, Any]],
    callbacks: Sequence[Mapping[str, Any]],
    heartbeat_times: Sequence[int],
    write_threshold_ns: int,
    callback_threshold_ns: int,
    heartbeat_threshold_ns: int,
) -> Dict[str, Any]:
    """统计长写入与长回调、长心跳间隔的时间重叠。"""
    heartbeat_gaps = [
        (left, right, right - left)
        for left, right in zip(heartbeat_times, heartbeat_times[1:])
    ]
    image_gaps = [
        (int(item["previous_image_start_ns"]), int(item["start_ns"]),
         int(item["image_start_interval_ns"]))
        for item in callbacks
        if item.get("previous_image_start_ns") is not None
    ]
    long_image_gaps = [
        item for item in image_gaps if item[2] >= callback_threshold_ns
    ]
    long_writes = [item for item in writes if item["duration_ns"] >= write_threshold_ns]
    long_callbacks = [
        item for item in callbacks
        if item["duration_ns"] >= callback_threshold_ns
    ]
    long_heartbeats = [
        item for item in heartbeat_gaps if item[2] >= heartbeat_threshold_ns
    ]
    overlaps: List[Dict[str, int]] = []
    for write_index, write in enumerate(long_writes):
        callback_count = sum(
            write["start_ns"] < item["end_ns"]
            and write["end_ns"] > item["start_ns"]
            for item in long_callbacks
        )
        image_gap_count = sum(
            write["start_ns"] < gap_end_ns and write["end_ns"] > gap_start_ns
            for gap_start_ns, gap_end_ns, _ in long_image_gaps
        )
        heartbeat_count = sum(
            write["start_ns"] < end_ns and write["end_ns"] > start_ns
            for start_ns, end_ns, _ in long_heartbeats
        )
        overlaps.append({
            "write_index": write_index,
            "overlapping_long_callbacks": callback_count,
            "overlapping_long_image_callback_gaps": image_gap_count,
            "overlapping_long_heartbeats": heartbeat_count,
        })
    return {
        "write_long_threshold_ns": write_threshold_ns,
        "callback_long_threshold_ns": callback_threshold_ns,
        "heartbeat_long_threshold_ns": heartbeat_threshold_ns,
        "long_write_count": len(long_writes),
        "long_callback_count": len(long_callbacks),
        "long_image_callback_gap_count": len(long_image_gaps),
        "long_heartbeat_gap_count": len(long_heartbeats),
        "long_writes": overlaps,
    }


def _argument_parser() -> argparse.ArgumentParser:
    """创建 ``python -m fastumi_data.write_baseline`` 的命令行解析器。"""
    parser = argparse.ArgumentParser(
        description="在目标文件系统上运行有时长和容量上限的 MCAP 写入基准"
    )
    parser.add_argument("--output-dir", required=True, help="必须是尚不存在的输出目录")
    parser.add_argument("--path", choices=("queued", "continuous"), default="queued")
    parser.add_argument("--mode", choices=tuple(MESSAGE_TYPES), default="raw")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-input-mib", type=float, default=512.0)
    parser.add_argument("--max-disk-mib", type=float, default=1024.0)
    parser.add_argument("--report", help="JSON 报告路径；默认写入 output-dir")
    parser.add_argument("--keep-bag", action="store_true", help="完成后保留临时 MCAP 包")
    return parser


def _prepare_paths(output_dir: str, report_arg: Optional[str]) -> tuple[Path, Path, Path]:
    """创建全新输出目录，并校验报告不会被临时包清理。"""
    output_path = Path(output_dir).expanduser().absolute()
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("--output-dir 必须指向一个尚不存在的目录")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_arg:
        report_path = Path(report_arg).expanduser().absolute()
    else:
        report_path = output_path / "baseline-report.json"
    bag_path = output_path / "bag"
    if report_path == bag_path or bag_path in report_path.parents:
        raise ValueError("--report 不能放在将被清理的 bag 子目录中")
    if report_path.exists():
        raise ValueError("--report 已存在；为避免覆盖，请指定新路径")
    output_path.mkdir()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path, bag_path, report_path


def run_baseline(arguments: argparse.Namespace) -> Dict[str, Any]:
    """运行单轮基准，关闭后读回 MCAP 并按策略清理临时包。"""
    settings = validate_settings(
        arguments.path, arguments.mode, arguments.duration_s, arguments.fps,
        arguments.width, arguments.height, arguments.max_input_mib,
        arguments.max_disk_mib,
    )
    output_path, bag_path, report_path = _prepare_paths(
        arguments.output_dir, arguments.report
    )
    if shutil.disk_usage(output_path).free < settings["max_disk_bytes"]:
        shutil.rmtree(output_path)
        raise ValueError("目标文件系统的可用空间小于 --max-disk-mib")

    report: Dict[str, Any] = {
        "scope": "离线序列化入口回调基准；不覆盖 DDS 与相机硬件链路",
        "limitations": [
            "源端按计划时刻生成合成图像，编码和 IPC 延迟会单独记录",
            "连续录制的生产写入与关闭为同步调用，底层 I/O 卡住时无法由本工具中断",
            "关联结果只描述时间重叠，不据此自动判定磁盘故障或触发修复",
        ],
        "settings": settings,
        "topics": {
            IMAGE_TOPIC: settings["mode"],
            STRING_TOPIC: "std_msgs/msg/String",
        },
        "timing": {},
        "source": {},
        "accepted": {"count": 0, "counts_by_topic": {}},
        "actual_written": {"count": 0, "counts_by_topic": {}},
        "queue_depth_peak": None,
        "queue_capacity": None,
        "readback": None,
        "error": "",
        "bag_kept": False,
    }
    # 记录各资源状态；队列和源进程等待有超时，连续同步 I/O 受底层调用影响。
    queued_writer = None
    recorder = None
    ros_initialized = False
    writer_closed = False
    source = None
    event_queue = None
    stop_source = None
    heartbeat_stop = threading.Event()
    depth_stop = threading.Event()
    heartbeat_times: List[int] = []
    depth_samples: List[Dict[str, int]] = []
    heartbeat_thread: Optional[threading.Thread] = None
    depth_thread: Optional[threading.Thread] = None
    callbacks: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    finish_result = None
    source_arrays = None
    try:
        import rclpy
        from fastumi_data.collection_writer import (
            QueuedBagWriter, RosbagMcapBackend, TopicSpec,
        )
        from fastumi_data.continuous_recorder import ContinuousRecorder

        ros_arguments = []
        if settings["path"] == "continuous":
            ros_arguments = [
                "--ros-args", "-p", f"image_mode:={settings['mode']}",
                "-p", f"image_topic:={IMAGE_TOPIC}",
            ]
        rclpy.init(args=ros_arguments)
        ros_initialized = True
        timed_backend = _TimedBackend(RosbagMcapBackend(), bag_path, settings["max_disk_bytes"])
        if settings["path"] == "queued":
            queue_bytes = min(QUEUED_MAX_BYTES, settings["max_input_bytes"])
            report["queue_capacity"] = {
                "max_messages": QUEUED_MAX_MESSAGES,
                "max_bytes": queue_bytes,
            }
            queued_writer = QueuedBagWriter(
                timed_backend, max_messages=QUEUED_MAX_MESSAGES,
                max_bytes=queue_bytes,
            )
            queued_writer.start(str(bag_path), [
                TopicSpec(IMAGE_TOPIC, MESSAGE_TYPES[settings["mode"]]),
                TopicSpec(STRING_TOPIC, "std_msgs/msg/String"),
            ])
        else:
            recorder = ContinuousRecorder(
                str(bag_path), [IMAGE_TOPIC, STRING_TOPIC]
            )
            recorder._register_topic(STRING_TOPIC, "std_msgs/msg/String")
            recorder._writer = _TimedBackend(
                recorder._writer, bag_path, settings["max_disk_bytes"]
            )
            timed_backend = recorder._writer

        context = mp.get_context("spawn")
        event_queue = context.Queue(maxsize=SOURCE_QUEUE_DEPTH)
        ready = context.Event()
        start_event = context.Event()
        stop_source = context.Event()
        start_ns_shared = context.Value(ctypes.c_longlong, 0, lock=False)
        source_arrays = context.Array(
            ctypes.c_longlong, settings["message_limit"] * 4, lock=False
        )
        source_counters = context.Array(ctypes.c_longlong, 8, lock=False)
        base_stamp_ns = time.time_ns() + 1_000_000_000
        source = context.Process(
            target=_source_process,
            args=(settings, event_queue, ready, start_event, stop_source,
                  start_ns_shared, source_arrays, source_counters, base_stamp_ns),
            name="fastumi-baseline-source",
        )
        source.start()
        if not ready.wait(30):
            raise TimeoutError("源进程未能准备图像编码器")
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop, args=(heartbeat_stop, heartbeat_times),
            name="fastumi-baseline-heartbeat", daemon=True,
        )
        heartbeat_thread.start()
        if queued_writer is not None:
            depth_thread = threading.Thread(
                target=_depth_loop,
                args=(depth_stop, queued_writer, depth_samples),
                name="fastumi-baseline-queue-depth", daemon=True,
            )
            depth_thread.start()
        start_ns = time.monotonic_ns() + 100_000_000
        start_ns_shared.value = start_ns
        start_event.set()
        receive_deadline = time.monotonic() + settings["duration_s"] * 2 + 60
        last_callback_start = None
        last_image_callback_start = None
        source_error = ""
        source_status_seen = False
        while True:
            if time.monotonic() > receive_deadline:
                stop_source.set()
                raise TimeoutError("源进程或入口回调超过有界运行时限")
            try:
                item = event_queue.get(timeout=0.1)
            except queue.Empty:
                if not source.is_alive():
                    break
                continue
            if item is None:
                break
            if item.get("kind") == "source_status":
                source_error = item.get("error", "")
                source_status_seen = True
                continue
            callback_start = time.monotonic_ns()
            callback_record = {
                "topic": item["topic"],
                "frame_index": item["frame_index"],
                "start_ns": callback_start,
                "previous_start_interval_ns": (
                    None if last_callback_start is None
                    else callback_start - last_callback_start
                ),
                "previous_image_start_ns": (
                    last_image_callback_start if item["topic"] == IMAGE_TOPIC else None
                ),
                "image_start_interval_ns": (
                    None if item["topic"] != IMAGE_TOPIC
                    or last_image_callback_start is None
                    else callback_start - last_image_callback_start
                ),
            }
            last_callback_start = callback_start
            if item["topic"] == IMAGE_TOPIC:
                last_image_callback_start = callback_start
            payload = bytes(item["data"])
            payload_hash = hashlib.sha256(payload).hexdigest()
            try:
                if queued_writer is not None:
                    is_accepted = queued_writer.offer(
                        item["topic"], payload, item["timestamp_ns"]
                    )
                    if queued_writer.depth > (report.get("queue_depth_peak") or 0):
                        report["queue_depth_peak"] = queued_writer.depth
                elif item["topic"] == IMAGE_TOPIC:
                    recorder._on_image(payload)
                    is_accepted = True
                else:
                    recorder._write(item["topic"], payload)
                    is_accepted = True
                if is_accepted:
                    accepted.append({
                        "topic": item["topic"],
                        "sha256": payload_hash,
                        "timestamp_ns": item["timestamp_ns"],
                        "frame_index": item["frame_index"],
                        "path": settings["path"],
                    })
                else:
                    rejection_reason = "writer rejected input"
                    rejected.append({
                        "topic": item["topic"],
                        "frame_index": item["frame_index"],
                        "reason": rejection_reason,
                    })
                    report["error"] = report["error"] or rejection_reason
                    stop_source.set()
            except Exception as error:
                rejection_reason = f"{type(error).__name__}: {error}"
                rejected.append({
                    "topic": item["topic"],
                    "frame_index": item["frame_index"],
                    "reason": rejection_reason,
                })
                report["error"] = report["error"] or rejection_reason
                stop_source.set()
            finally:
                callback_end = time.monotonic_ns()
                callback_record["end_ns"] = callback_end
                callback_record["duration_ns"] = callback_end - callback_start
                callbacks.append(callback_record)
        if not source_status_seen:
            raise RuntimeError("源进程未发送完成状态")
        if source_error:
            report["error"] = f"源进程失败：{source_error}"
        source.join(timeout=5)
        if source.is_alive():
            source.terminate()
            source.join(timeout=5)
            raise TimeoutError("源进程未在限定时间内退出，已终止")
        if queued_writer is not None:
            finish_started_ns = time.monotonic_ns()
            finish_result = queued_writer.finish(timeout_s=60)
            finish_ended_ns = time.monotonic_ns()
            writer_finish_timing = {
                "start_ns": finish_started_ns,
                "end_ns": finish_ended_ns,
                "duration_ns": finish_ended_ns - finish_started_ns,
            }
            writer_closed = finish_result.thread_stopped
            if not finish_result.ok:
                report["error"] = finish_result.error_message or finish_result.error_code
        else:
            recorder.close()
            writer_closed = True
        heartbeat_stop.set()
        depth_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2)
        if depth_thread is not None:
            depth_thread.join(timeout=2)
        if writer_closed:
            report["readback"] = _readback(bag_path, settings["mode"], accepted)
            readback = report["readback"]
            if (
                not readback["exact_bytes_topics_and_fifo"]
                or not readback["topic_types_match"]
                or readback["missing_count"]
                or readback["extra_count"]
                or readback["queued_timestamp_errors"]
                or readback["image_header_stamp_errors"]
                or not readback["continuous_log_timestamps_nondecreasing"]
            ):
                report["error"] = report["error"] or "MCAP 读回与入口消息核对失败"
        source_report_rows = []
        for record_index in range(settings["message_limit"]):
            offset = record_index * 4
            planned_ns, generated_ns, send_started_ns, send_done_ns = (
                int(source_arrays[offset + field]) for field in range(4)
            )
            if planned_ns:
                source_report_rows.append({
                    "record_index": record_index,
                    "planned_ns": planned_ns,
                    "generated_ns": generated_ns,
                    "send_started_ns": send_started_ns,
                    "send_done_ns": send_done_ns,
                })
        source_message_count = int(source_counters[3])
        report["source"] = {
            "frames_scheduled": int(source_counters[0]),
            "image_messages_generated": int(source_counters[1]),
            "string_messages_generated": int(source_counters[2]),
            "messages_sent": source_message_count,
            "serialized_bytes_sent": int(source_counters[4]),
            "input_limit_reached": bool(source_counters[5]),
            "source_error_flag": bool(source_counters[6]),
            "schedule_records": source_report_rows,
            "generation_lateness_ns": percentile_summary(
                row["generated_ns"] - row["planned_ns"]
                for row in source_report_rows if row["generated_ns"]
            ),
            "source_lateness_ns": percentile_summary(
                row["send_started_ns"] - row["planned_ns"]
                for row in source_report_rows if row["send_started_ns"]
            ),
        }
        actual_writes = list(timed_backend.trace)
        report["accepted"] = {
            "count": len(accepted),
            "counts_by_topic": dict(Counter(item["topic"] for item in accepted)),
            "rejected_count": len(rejected),
            "rejected_examples": rejected[:50],
        }
        if not report["accepted"]["counts_by_topic"].get(IMAGE_TOPIC, 0):
            report["error"] = report["error"] or "未接收任何图像消息，无法形成有效基线"
        report["actual_written"] = {
            "count": sum(bool(item["ok"]) for item in actual_writes),
            "counts_by_topic": dict(Counter(
                item["topic"] for item in actual_writes if item["ok"]
            )),
            "trace": actual_writes,
            "drain_and_close": writer_finish_timing,
            "close": timed_backend.close_timing,
        }
        heartbeats = [int(value) for value in heartbeat_times]
        heartbeat_gaps = [right - left for left, right in zip(heartbeats, heartbeats[1:])]
        callback_gaps = [
            int(item["previous_start_interval_ns"])
            for item in callbacks if item["previous_start_interval_ns"] is not None
        ]
        write_threshold = max(20_000_000, int(2_000_000_000 / settings["fps"]))
        callback_threshold = int(2_000_000_000 / settings["fps"])
        report["timing"] = {
            "clock": "time.monotonic_ns",
            "heartbeat_period_ms": 10,
            "heartbeat_scope": "input through drain/close; MCAP readback is excluded",
            "heartbeat_times_ns": heartbeats,
            "heartbeat_intervals": percentile_summary(heartbeat_gaps),
            "callback_times": callbacks,
            "callback_durations": percentile_summary(
                item["duration_ns"] for item in callbacks
            ),
            "callback_start_intervals": percentile_summary(callback_gaps),
            "write_durations": percentile_summary(
                item["duration_ns"] for item in actual_writes
            ),
            "overlap": _overlap_report(
                actual_writes, callbacks, heartbeats,
                write_threshold, callback_threshold, 20_000_000,
            ),
        }
        if queued_writer is not None:
            report["queue_depth_peak"] = max(
                [int(report.get("queue_depth_peak") or 0)]
                + [int(item["depth"]) for item in depth_samples]
            )
            report["queue_depth_samples"] = depth_samples
            report["queue_depth_sampling"] = "offer return + 1 ms sampling"
        else:
            report["queue_depth_peak"] = 0
            report["queue_capacity"] = None
        report["accepted_vs_source_missing"] = max(
            0, source_message_count - len(accepted)
        )
        if rejected or report["accepted_vs_source_missing"]:
            report["error"] = report["error"] or "存在入口拒绝或未接收的源消息"
        if report["readback"] is not None:
            report["readback_counts"] = report["readback"]["counts_by_topic"]
        report["source"]["source_timestamps"] = {
            "base_header_stamp_ns": base_stamp_ns,
            "frame_period_ns": int(1_000_000_000 / settings["fps"]),
        }
    except Exception as error:
        if not report["error"]:
            report["error"] = f"{type(error).__name__}: {error}"
    finally:
        if stop_source is not None:
            stop_source.set()
        if source is not None and source.is_alive():
            source.join(timeout=5)
            if source.is_alive():
                source.terminate()
                source.join(timeout=5)
        heartbeat_stop.set()
        depth_stop.set()
        if queued_writer is not None and not writer_closed:
            try:
                finish_result = queued_writer.finish(timeout_s=60)
                writer_closed = finish_result.thread_stopped
            except Exception as error:
                report["error"] = report["error"] or str(error)
        if recorder is not None:
            try:
                if not writer_closed:
                    recorder.close()
                    writer_closed = True
            except Exception as error:
                report["error"] = report["error"] or str(error)
            try:
                recorder.destroy_node()
            except Exception:
                pass
        if ros_initialized:
            try:
                import rclpy
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=2)
            if heartbeat_thread.is_alive():
                report["error"] = report["error"] or "心跳线程未在限定时间内退出"
        if depth_thread is not None:
            depth_thread.join(timeout=2)
            if depth_thread.is_alive():
                report["error"] = report["error"] or "队列采样线程未在限定时间内退出"
        if source is not None and source.is_alive():
            report["error"] = report["error"] or "源进程仍未退出"
        if bag_path.exists() and writer_closed:
            try:
                final_size = _directory_size(bag_path)
                report["actual_disk_bytes"] = final_size
                if final_size > settings["max_disk_bytes"]:
                    report["error"] = report["error"] or (
                        "MCAP 最终目录超过 --max-disk-mib"
                    )
            except Exception as error:
                report["error"] = report["error"] or (
                    f"读取最终磁盘占用失败：{error}"
                )
        if bag_path.exists() and writer_closed and not arguments.keep_bag and not report["error"]:
            shutil.rmtree(bag_path)
        report["bag_kept"] = bag_path.exists()
        if event_queue is not None:
            event_queue.cancel_join_thread()
            event_queue.close()
        try:
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as error:
            report["error"] = report["error"] or f"写入 JSON 报告失败：{error}"
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """解析参数、运行诊断并以结果状态作为进程退出码。"""
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        report = run_baseline(arguments)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({
        "report": str(Path(arguments.report).expanduser().absolute())
        if arguments.report else str(Path(arguments.output_dir).expanduser().absolute()
                                     / "baseline-report.json"),
        "error": report["error"],
        "readback": report["readback"] is not None,
        "bag_kept": report["bag_kept"],
    }, ensure_ascii=False))
    return 0 if not report["error"] and report["readback"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
