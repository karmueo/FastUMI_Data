#!/usr/bin/env python3
"""将 hardware.launch.py 录制的 MCAP 转换为 RM75 episode 或 UMI Zarr。

依赖 ROS 2 消息类型和 rosbag2_py 读取录制，使用 OpenCV、ffmpeg/ffprobe 处理图像；
HDF5 输出使用 h5py，UMI 输出由 convert_hardware_mcap_umi 处理。输出时间戳统一采用
rosbag 接收时钟；消息头时间戳仅用于时间差诊断和视频帧率估计，不用于跨流对齐，
因为录制器与 Tracker 可能位于未同步的主机。

启动：python convert_hardware_mcap.py --input PATH --output PATH
      [--workers N] [--format {hdf5,umi}] [--urdf PATH]
启动参数：
    --input PATH：必填，episode_N 目录或包含多个 episode_N 的目录。
    --output PATH：必填，输出 episode 的父目录；UMI 格式须为新的 .zarr 目录。
    --workers N：可选，正整数进程数，默认 min(4, CPU 数量)，每轮最多使用 episode 数量。
    --format {hdf5,umi}：可选，默认 hdf5。
    --urdf PATH：选择 umi 时必填，训练使用的 RM75 URDF；hdf5 路径不使用。
输入：各 episode_N/bag 中的 MCAP、可选的 recording.json；UMI 格式还读取 URDF。
输出：hdf5 格式写入每轮的 gripper.mp4、proprio.hdf5、conversion.json；
      umi 格式写入合并的 Zarr；进度和错误摘要写入标准输出。
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any
import uuid

import cv2
import numpy as np
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
import rosbag2_py
from rm_ros_interfaces.msg import Jointpos
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float32


# RM75 的七个关节名称，状态与指令均按此顺序排列。
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
# 必需数据流的 ROS 话题、消息类型和反序列化类；UMI 路径可不要求 Tracker。
TOPICS = {
    "joint_state": ("/joint_states", "sensor_msgs/msg/JointState", JointState),
    "joint_action": ("/rm_driver/movej_canfd_cmd", "rm_ros_interfaces/msg/Jointpos", Jointpos),
    "gripper_state": ("/motion_control/gripper_state", "std_msgs/msg/Float32", Float32),
    "gripper_action": ("/motion_control/gripper_command", "std_msgs/msg/Float32", Float32),
    "tracker": ("/vive_tracker/odom", "nav_msgs/msg/Odometry", Odometry),
}
# 支持的相机话题及编码模式；每个输入 bag 必须恰好包含其中一个。
CAMERA_TOPICS = {
    "/wrist_camera/image_raw": ("raw", "sensor_msgs/msg/Image", Image),
    "/wrist_camera/image_raw/compressed": (
        "jpeg", "sensor_msgs/msg/CompressedImage", CompressedImage
    ),
    "/wrist_camera/image_raw/ffmpeg": (
        "h264", "ffmpeg_image_transport_msgs/msg/FFMPEGPacket", FFMPEGPacket
    ),
}
# 写入 HDF5 根属性的数据格式版本。
FORMAT_VERSION = "rm75-single-arm-v1"


class ConversionError(ValueError):
    """输入 episode 无法生成完整且时间对齐的输出时抛出。"""


@dataclass(frozen=True)
class CameraRecord:
    """相机帧在临时缓存中的位置及原始时间信息。

    bag_ns 和 header_ns 分别是接收与消息头时间戳，单位 ns；offset、size 是
    spool 内的字节位置和长度。width、height、step 是像素尺寸和行跨度；
    encoding 保留源编码，keyframe 和 pts 仅用于 H.264 帧。
    """

    bag_ns: int
    header_ns: int
    offset: int
    size: int
    width: int
    height: int
    encoding: str
    step: int = 0
    keyframe: bool = False
    pts: int = 0


def _stamp_ns(stamp: Any) -> int:
    """将 ROS 时间戳转换为纳秒。

    Args:
        stamp: 含 sec 和 nanosec 字段的 ROS 时间戳。

    Returns:
        整数纳秒时间戳。
    """
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _seconds(timestamps_ns: list[int]) -> np.ndarray:
    """将纳秒时间戳序列转换为 float64 秒数组。

    Args:
        timestamps_ns: 纳秒时间戳列表。

    Returns:
        与输入等长的一维秒数组。
    """
    return np.asarray(timestamps_ns, dtype=np.float64) / 1_000_000_000.0


def _diagnostic(values_ns: list[int]) -> dict[str, float | int]:
    """汇总接收时间与源时间之差，不将其视为时钟标定结果。

    Args:
        values_ns: 接收时间减源时间的纳秒差列表。

    Returns:
        样本数及毫秒单位的最小值、中位数、95 百分位和最大值；空列表仅含 count。
    """
    if not values_ns:
        return {"count": 0}
    milliseconds = np.asarray(values_ns, dtype=np.float64) / 1_000_000.0
    return {
        "count": len(values_ns),
        "min_ms": float(np.min(milliseconds)),
        "median_ms": float(np.median(milliseconds)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "max_ms": float(np.max(milliseconds)),
    }


def _read_episode(bag: Path, spool: Any, *,
                  require_tracker: bool = True) -> dict[str, Any]:
    """读取 MCAP 中的有效数据流，并将相机载荷写入可定位临时缓存。

    Args:
        bag: episode 下的 bag 目录路径。
        spool: 支持 tell、write 的二进制临时文件，写入相机载荷后保留供后续读取。
        require_tracker: 是否要求 Tracker 话题存在，默认要求。

    Returns:
        按接收时间排序的非图像序列、相机帧索引、计数及时间差诊断信息。

    Raises:
        ConversionError: 必需话题缺失、类型不符或相机话题数量不为一。
    """
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    available = {item.name: item.type for item in reader.get_all_topics_and_types()}
    missing = [name for name, (topic, _, _) in TOPICS.items()
               if topic not in available and (name != "tracker" or require_tracker)]
    if missing:
        raise ConversionError(f"bag 缺少必需话题: {', '.join(missing)}")
    for name, (topic, expected, _) in TOPICS.items():
        if topic not in available:
            continue
        if available[topic] != expected:
            raise ConversionError(f"{name} 类型应为 {expected}，实际为 {available[topic]}")
    cameras = [topic for topic in CAMERA_TOPICS if topic in available]
    if len(cameras) != 1:
        raise ConversionError(f"预期恰好一个相机话题，实际为 {cameras}")
    camera_topic = cameras[0]
    camera_mode, camera_type, camera_class = CAMERA_TOPICS[camera_topic]
    if available[camera_topic] != camera_type:
        raise ConversionError(f"相机类型应为 {camera_type}，实际为 {available[camera_topic]}")

    by_topic = {spec[0]: (name, spec[2]) for name, spec in TOPICS.items()
                if spec[0] in available}
    by_topic[camera_topic] = ("camera", camera_class)
    # series 元素为（rosbag 接收时间 ns，解码值）；相机载荷单独写入 spool。
    series: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    camera_records: list[CameraRecord] = []
    counts: Counter[str] = Counter()
    invalid: Counter[str] = Counter()
    # 仅用于诊断的接收时间减消息头时间；跨主机时钟未标定。
    deltas: dict[str, list[int]] = defaultdict(list)
    pts_deltas: list[int] = []
    pts_mismatch = 0

    while reader.has_next():
        topic, serialized, bag_ns = reader.read_next()
        if topic not in by_topic:
            continue
        name, message_class = by_topic[topic]
        counts[name] += 1
        bag_ns = int(bag_ns)
        if bag_ns <= 0:
            invalid[name] += 1
            continue
        message = deserialize_message(serialized, message_class)
        if name == "joint_state":
            names = list(message.name)
            positions = list(message.position)
            if len(names) != len(positions) or len(set(names)) != len(names):
                invalid[name] += 1
                continue
            ordered = dict(zip(names, positions))
            if any(joint not in ordered for joint in JOINT_NAMES):
                invalid[name] += 1
                continue
            values = np.asarray([ordered[joint] for joint in JOINT_NAMES], dtype=np.float32)
            if not np.isfinite(values).all():
                invalid[name] += 1
                continue
            series[name].append((bag_ns, values))
            header_ns = _stamp_ns(message.header.stamp)
        elif name == "joint_action":
            values = np.asarray(message.joint, dtype=np.float32)
            if int(message.dof) != 7 or values.shape != (7,) or not np.isfinite(values).all():
                invalid[name] += 1
                continue
            series[name].append((bag_ns, values))
            header_ns = 0
        elif name in ("gripper_state", "gripper_action"):
            value = float(message.data)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                invalid[name] += 1
                continue
            series[name].append((bag_ns, value))
            header_ns = 0
        elif name == "tracker":
            position = message.pose.pose.position
            values = np.asarray([position.x, position.y, position.z], dtype=np.float32)
            if not np.isfinite(values).all():
                invalid[name] += 1
                continue
            series[name].append((bag_ns, values))
            header_ns = _stamp_ns(message.header.stamp)
        else:
            header_ns = _stamp_ns(message.header.stamp)
            if camera_mode == "raw":
                encoding = message.encoding.lower()
                channels = {"bgr8": 3, "rgb8": 3, "mono8": 1}.get(encoding)
                payload = bytes(message.data)
                step = int(message.step)
                valid = bool(
                    channels and message.width > 0 and message.height > 0
                    and step >= message.width * channels
                    and len(payload) >= step * message.height
                )
                keyframe = False
                pts = 0
            elif camera_mode == "jpeg":
                encoding = message.format.lower()
                payload = bytes(message.data)
                valid = bool(
                    ("jpeg" in encoding or "jpg" in encoding)
                    and payload.startswith(b"\xff\xd8")
                    and payload.endswith(b"\xff\xd9")
                )
                step = 0
                keyframe = False
                pts = 0
            else:
                encoding = message.encoding.lower()
                payload = bytes(message.data)
                valid = bool(
                    encoding.split(";", 1)[0] == "h264"
                    and message.width > 0 and message.height > 0 and payload
                )
                step = 0
                keyframe = bool(message.flags & 1)
                pts = int(message.pts)
                if pts != header_ns:
                    pts_mismatch += 1
            if not valid:
                invalid[name] += 1
                continue
            offset = spool.tell()
            spool.write(payload)
            camera_records.append(
                CameraRecord(
                    bag_ns, header_ns, offset, len(payload),
                    int(getattr(message, "width", 0)),
                    int(getattr(message, "height", 0)),
                    encoding, step, keyframe, pts,
                )
            )
            if camera_mode == "h264" and pts > 0:
                pts_deltas.append(bag_ns - pts)
        if header_ns > 0:
            deltas[name].append(bag_ns - header_ns)

    for records in series.values():
        records.sort(key=lambda item: item[0])
    camera_records.sort(key=lambda item: item.bag_ns)
    return {
        "series": series,
        "camera": camera_records,
        "camera_mode": camera_mode,
        "camera_topic": camera_topic,
        "counts": counts,
        "invalid": invalid,
        "deltas": deltas,
        "pts_deltas": pts_deltas,
        "pts_header_mismatch_count": pts_mismatch,
    }


def _select(data: dict[str, Any]) -> dict[str, Any]:
    """按关节指令时间窗选取 HDF5 数据并检查各数据流的共同覆盖区间。

    Args:
        data: _read_episode 返回的已排序录制数据。

    Returns:
        裁剪后的序列、夹爪状态与最近先前指令的配对、相机帧及时间窗，
        时间戳单位为 ns。

    Raises:
        ConversionError: 样本不足、缺少 H.264 关键帧、相机尺寸变化或无共同时间区间。
    """
    series = data["series"]
    for name in TOPICS:
        if not series[name]:
            raise ConversionError(f"缺少有效的 {name} 样本")
    actions = series["joint_action"]
    start_ns, end_ns = actions[0][0], actions[-1][0]
    if start_ns >= end_ns:
        raise ConversionError("有效关节指令的时间区间为空")
    selected = {
        name: [(timestamp, value) for timestamp, value in records
               if start_ns <= timestamp <= end_ns]
        for name, records in series.items()
    }
    for name in ("joint_state", "tracker"):
        if not selected[name]:
            raise ConversionError(f"指令时间区间内缺少 {name} 样本")

    # 时间窗前的最后一条夹爪指令仍可作用于窗内第一个夹爪状态，配对时保留它。
    commands = series["gripper_action"]
    command_times = [timestamp for timestamp, _ in commands]
    gripper_pairs = []
    for timestamp, state in selected["gripper_state"]:
        index = bisect_right(command_times, timestamp) - 1
        if index >= 0:
            gripper_pairs.append((timestamp, state, commands[index][1]))
    if not gripper_pairs:
        raise ConversionError("指令时间区间内没有可配对的夹爪状态与指令")

    camera_in_window = [record for record in data["camera"]
                        if start_ns <= record.bag_ns <= end_ns]
    if data["camera_mode"] == "h264":
        first_key = next((index for index, record in enumerate(camera_in_window)
                          if record.keyframe), None)
        if first_key is None:
            raise ConversionError("指令时间区间内没有 H.264 关键帧")
        camera = camera_in_window[first_key:]
        skipped_until_keyframe = first_key
    else:
        camera = camera_in_window
        skipped_until_keyframe = 0
    if not camera:
        raise ConversionError("指令时间区间内没有可写入的视频帧")
    dimensions = {(record.width, record.height) for record in camera
                  if record.width and record.height}
    if len(dimensions) > 1:
        raise ConversionError("同一 episode 内相机分辨率发生变化")
    coverage = [(records[0][0], records[-1][0]) for records in (
        selected["joint_action"], selected["joint_state"],
        selected["tracker"], gripper_pairs,
    )]
    coverage.append((camera[0].bag_ns, camera[-1].bag_ns))
    if max(start for start, _ in coverage) >= min(end for _, end in coverage):
        raise ConversionError("相机、关节、夹爪和 Tracker 没有共同时间区间")
    return {
        "series": selected,
        "gripper_pairs": gripper_pairs,
        "camera": camera,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "skipped_until_keyframe": skipped_until_keyframe,
    }


def _fps(records: list[CameraRecord]) -> float:
    """根据相机首末帧时间跨度估计视频恒定帧率。

    Args:
        records: 按接收时间排序的相机帧；消息头时间全为正时优先使用消息头时间。

    Returns:
        估计帧率，单位帧/秒；样本不足时返回 30.0。

    Raises:
        ConversionError: 可估计的帧率超出 1 至 120 帧/秒。
    """
    source = [record.header_ns for record in records]
    if not all(timestamp > 0 for timestamp in source):
        source = [record.bag_ns for record in records]
    if len(source) < 2 or source[-1] <= source[0]:
        return 30.0
    # 采集抖动可能使相邻帧间隔的中位数偏离一整档帧率；用完整跨度贴近源时长。
    estimated = (len(source) - 1) * 1_000_000_000.0 / (source[-1] - source[0])
    if not 1.0 <= estimated <= 120.0:
        raise ConversionError(f"相机帧率无法估计: {estimated}")
    nearest = round(estimated)
    return float(nearest if abs(estimated - nearest) / estimated < 0.02
                 else round(estimated, 3))


def _payload(spool: Any, record: CameraRecord) -> bytes:
    """按相机帧索引读取临时缓存中的完整载荷。

    Args:
        spool: 支持 seek、read 的二进制临时文件；读取会改变文件位置。
        record: 包含载荷偏移量和长度的相机帧记录。

    Returns:
        该帧的原始字节。

    Raises:
        ConversionError: 临时缓存中的载荷长度不足。
    """
    spool.seek(record.offset)
    payload = spool.read(record.size)
    if len(payload) != record.size:
        raise ConversionError("相机临时缓存截断")
    return payload


def _probe_frames(path: Path, *, threads: int | None = None) -> list[dict[str, Any]]:
    """调用 ffprobe 获取首个视频流的逐帧包位置。

    Args:
        path: 待检查的视频或 H.264 码流路径。
        threads: 可选的 ffprobe 线程数；None 表示不传线程参数。

    Returns:
        ffprobe JSON 中的 frames 列表。

    Raises:
        ConversionError: ffprobe 失败或标准输出中没有 JSON 对象。
    """
    command = ["ffprobe", "-v", "error"]
    if threads is not None:
        command.extend(["-threads", str(threads)])
    command.extend([
        "-show_frames", "-select_streams", "v:0",
        "-show_entries", "frame=pkt_pos", "-of", "json", str(path),
    ])
    result = subprocess.run(
        command,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ConversionError(f"ffprobe 解码失败: {result.stderr.strip()}")
    # 部分 Jetson FFmpeg 构建会在标准输出的 JSON 前打印 EGL 诊断信息。
    json_start = result.stdout.find("{")
    if json_start < 0:
        raise ConversionError("ffprobe 未返回 JSON 帧信息")
    return json.loads(result.stdout[json_start:]).get("frames", [])


def _run_ffmpeg(command: list[str], payloads: Any = None) -> None:
    """运行 ffmpeg，并可向其标准输入逐块写入视频帧。

    Args:
        command: 完整的 ffmpeg 命令参数列表，包含输出路径。
        payloads: 可选的字节块可迭代对象；提供时经标准输入输送。

    Raises:
        ConversionError: ffmpeg 退出码非零。

    子进程标准输出被丢弃；失败时从标准错误读取详细信息。
    """
    with tempfile.TemporaryFile() as error_file:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE if payloads is not None else subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=error_file,
        )
        try:
            if payloads is not None:
                assert process.stdin is not None
                for payload in payloads:
                    process.stdin.write(payload)
                process.stdin.close()
            return_code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        error_file.seek(0)
        detail = error_file.read().decode(errors="replace").strip()
    if return_code:
        raise ConversionError(f"ffmpeg 视频转换失败: {detail}")


def _decode_frame(spool: Any, record: CameraRecord, mode: str) -> np.ndarray:
    """从临时缓存解码 JPEG 或原始相机帧为 BGR 图像。

    Args:
        spool: 存放相机载荷的可定位二进制临时文件。
        record: 帧偏移、尺寸、行跨度和编码信息。
        mode: 相机模式，调用方在此传入 jpeg 或 raw。

    Returns:
        高 × 宽 × 3 的 uint8 BGR 数组。

    Raises:
        ConversionError: 载荷截断或 JPEG 无法解码。
    """
    payload = _payload(spool, record)
    if mode == "jpeg":
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ConversionError("JPEG 帧无法解码")
        return image
    channels = 1 if record.encoding == "mono8" else 3
    raw = np.frombuffer(payload, dtype=np.uint8, count=record.step * record.height)
    rows = raw.reshape(record.height, record.step)
    image = rows[:, :record.width * channels].reshape(
        record.height, record.width, channels
    )
    if record.encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if record.encoding == "mono8":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _write_video(path: Path, records: list[CameraRecord], mode: str,
                 spool: Any, fps: float, *, threads: int | None = None) -> None:
    """把相机帧写为 MP4，并校验输出帧数与输入记录数一致。

    Args:
        path: 目标 MP4 路径；H.264 模式还会暂时创建同名 .h264 文件。
        records: 按时间排序的相机帧，H.264 模式需从关键帧开始。
        mode: h264、jpeg 或 raw 相机模式。
        spool: 存放相机载荷的可定位二进制临时文件。
        fps: 写入视频的恒定帧率，单位帧/秒。
        threads: 可选的 ffmpeg/ffprobe 线程数。

    Raises:
        ConversionError: 帧无法解码、H.264 包映射不一致、工具失败或输出帧数不符。
    """
    ffmpeg_command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if threads is not None:
        ffmpeg_command.extend(["-filter_threads", "1"])
    if mode == "h264":
        bitstream_path = path.with_suffix(".h264")
        offsets = []
        with bitstream_path.open("wb") as bitstream:
            for record in records:
                offsets.append(bitstream.tell())
                bitstream.write(_payload(spool, record))
        frames = _probe_frames(bitstream_path, threads=threads)
        frame_offsets = [int(frame["pkt_pos"]) for frame in frames
                         if "pkt_pos" in frame]
        if frame_offsets != offsets:
            raise ConversionError(
                "H.264 解码帧与 MCAP 包不是一一对应，拒绝写入错误时间戳"
            )
        _run_ffmpeg(
            ffmpeg_command + [
                "-framerate", str(fps), "-f", "h264", "-i", str(bitstream_path),
                "-an", "-c:v", "copy", "-movflags", "+faststart", str(path),
            ]
        )
        bitstream_path.unlink()
    else:
        first = _decode_frame(spool, records[0], mode)
        height, width = first.shape[:2]

        def frames():
            """依次产出尺寸一致的 BGR 帧字节，供 ffmpeg 标准输入读取。

            Yields:
                每帧按行排列的 BGR 原始字节。

            Raises:
                ConversionError: 后续帧尺寸变化或无法解码。
            """
            yield first.tobytes()
            for record in records[1:]:
                image = _decode_frame(spool, record, mode)
                if image.shape != first.shape:
                    raise ConversionError("视频帧尺寸不一致")
                yield image.tobytes()

        command = ffmpeg_command + [
            "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size",
            f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264",
        ]
        if threads is not None:
            command.extend(["-threads", str(threads)])
        command.extend([
            "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ])
        _run_ffmpeg(command, frames())
    if len(_probe_frames(path, threads=threads)) != len(records):
        raise ConversionError("MP4 帧数与 HDF5 相机时间戳数量不一致")


def _write_hdf5(path: Path, selected: dict[str, Any]) -> None:
    """将裁剪后的本体感知数据和相机时间戳写入 HDF5。

    Args:
        path: 目标 proprio.hdf5 路径，以写入模式创建。
        selected: _select 返回的序列、夹爪配对和相机记录。

    关节数组每行为七个关节的值，Tracker 位置每行为三维 xyz；夹爪值范围为
    0 至 1。时间戳均为 rosbag 接收时间，存储单位为秒；图像内容在独立 MP4 中。
    """
    import h5py

    series = selected["series"]
    with h5py.File(path, "w") as root:
        root.attrs["sim"] = False
        root.attrs["format_version"] = FORMAT_VERSION
        root.attrs["camera_names"] = np.asarray([b"gripper"])
        root.attrs["alignment_reference"] = "joint_action"
        root.attrs["alignment_start_timestamp"] = selected["start_ns"] / 1e9
        root.attrs["alignment_end_timestamp"] = selected["end_ns"] / 1e9
        root.attrs["timestamp_domain"] = "rosbag_receive"
        observation = root.create_group("observations")
        action = root.create_group("action")

        def timed(parent: Any, name: str, values: np.ndarray,
                  timestamps: list[int], value_name: str) -> None:
            """在父组下写入值数组和对应的秒级时间戳。

            Args:
                parent: 目标 HDF5 父组。
                name: 新建子组的名称。
                values: 待写入的数据数组。
                timestamps: 与 values 行对应的接收时间戳，单位 ns。
                value_name: 值数据集的名称。
            """
            group = parent.create_group(name)
            group.create_dataset(value_name, data=values)
            group.create_dataset("timestamp", data=_seconds(timestamps))

        for source, parent, group, value_name in (
            ("joint_state", observation, "joint_state", "qpos"),
            ("joint_action", action, "joint_action", "position"),
            ("tracker", observation, "vr_pos", "position"),
        ):
            records = series[source]
            timed(parent, group,
                  np.stack([value for _, value in records]).astype(np.float32),
                  [timestamp for timestamp, _ in records], value_name)
        paired = selected["gripper_pairs"]
        paired_timestamps = [timestamp for timestamp, _, _ in paired]
        timed(observation, "gripper_state",
              np.asarray([[state] for _, state, _ in paired], dtype=np.float32),
              paired_timestamps, "position")
        timed(action, "gripper_action",
              np.asarray([[command] for _, _, command in paired], dtype=np.float32),
              paired_timestamps, "position")
        timed(observation, "vr_flag_B", np.empty(0, dtype=np.bool_), [], "value")
        images = observation.create_group("images")
        images.create_dataset(
            "cam_gripper_timestamp",
            data=_seconds([record.bag_ns for record in selected["camera"]]),
        )


def discover_episodes(source: Path) -> list[Path]:
    """查找单个 episode 或目录下带有 bag 的编号 episode。

    Args:
        source: episode_N 目录或其父目录。

    Returns:
        按 episode 编号升序排列的路径列表。

    Raises:
        ConversionError: 没有找到符合 episode_N/bag/metadata.yaml 结构的目录。
    """
    if (source / "bag" / "metadata.yaml").is_file():
        episodes = [source]
    elif source.is_dir():
        episodes = [path for path in source.iterdir()
                    if path.is_dir() and path.name.startswith("episode_")
                    and path.name[8:].isdigit()
                    and (path / "bag" / "metadata.yaml").is_file()]
    else:
        episodes = []
    if not episodes:
        raise ConversionError(f"没有找到 episode_N/bag: {source}")
    return sorted(episodes, key=lambda path: int(path.name[8:]))


def convert_episode(source: Path, output_dir: Path, *,
                    video_threads: int | None = None) -> dict[str, Any]:
    """在临时目录中转换单轮录制，完成后发布到输出目录。

    Args:
        source: 含 bag 及可选 recording.json 的 episode_N 目录。
        output_dir: 输出 episode 的父目录。
        video_threads: 可选的 ffmpeg/ffprobe 线程数。

    Returns:
        conversion.json 中写入的转换统计和时间诊断字典。

    Raises:
        FileExistsError: 同名目标 episode 已存在。
        ConversionError: 输入数据不完整、无法对齐或视频转换失败。

    成功时输出 gripper.mp4、proprio.hdf5 和 conversion.json；失败时清理临时目录。
    """
    destination = output_dir / source.name
    if destination.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖: {destination}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = output_dir / f".{source.name}.tmp-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        with tempfile.TemporaryFile(mode="w+b") as spool:
            data = _read_episode(source / "bag", spool)
            selected = _select(data)
            fps = _fps(selected["camera"])
            _write_video(stage / "gripper.mp4", selected["camera"],
                         data["camera_mode"], spool, fps, threads=video_threads)
            _write_hdf5(stage / "proprio.hdf5", selected)

        # 可选元数据只用于提取 recording_id，不参与时间对齐。
        recording_path = source / "recording.json"
        recording = (json.loads(recording_path.read_text(encoding="utf-8"))
                     if recording_path.is_file() else {})
        report = {
            "source_episode": str(source.resolve()),
            "recording_id": recording.get("recording_id"),
            "time_base": "rosbag_receive_timestamp_ns",
            "alignment_reference": "joint_action",
            "alignment_start_ns": selected["start_ns"],
            "alignment_end_ns": selected["end_ns"],
            "camera_topic": data["camera_topic"],
            "camera_mode": data["camera_mode"],
            "video_fps": fps,
            "input_counts": dict(data["counts"]),
            "invalid_counts": dict(data["invalid"]),
            "output_counts": {
                name: len(records) for name, records in selected["series"].items()
            } | {
                "gripper_paired": len(selected["gripper_pairs"]),
                "video_frames": len(selected["camera"]),
                "vr_flag_B": 0,
            },
            "camera_skipped_until_keyframe": selected["skipped_until_keyframe"],
            "camera_outside_action_window": (
                len(data["camera"])
                - sum(selected["start_ns"] <= record.bag_ns <= selected["end_ns"]
                      for record in data["camera"])
            ),
            "bag_minus_header": {
                name: _diagnostic(values) for name, values in data["deltas"].items()
            },
            "bag_minus_camera_pts": _diagnostic(data["pts_deltas"]),
            "camera_pts_header_mismatch_count": data["pts_header_mismatch_count"],
            "notes": ["bag 接收时间为统一时间轴；未标定跨机器采集延迟",
                      "源 bag 不包含 B 键事件，vr_flag_B 为空"],
        }
        (stage / "conversion.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if destination.exists():
            raise FileExistsError(f"目标已存在，拒绝覆盖: {destination}")
        stage.rename(destination)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _positive_int(value: str) -> int:
    """解析命令行中的正整数进程数。

    Args:
        value: --workers 的字符串值。

    Returns:
        大于零的整数。

    Raises:
        argparse.ArgumentTypeError: 无法解析为正整数。
    """
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("并行进程数必须是正整数") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("并行进程数必须是正整数")
    return number


def _init_worker() -> None:
    """将每个转换进程的 OpenCV 内部线程数限制为一。"""
    cv2.setNumThreads(1)


def convert_episodes(episodes: list[Path], output_dir: Path,
                     workers: int):
    """按完成顺序逐轮转换 episode，并分别报告成功或失败。

    Args:
        episodes: 待转换的 episode 目录列表。
        output_dir: 输出 episode 的父目录。
        workers: 并行进程数；为 1 时在当前进程中顺序转换。

    Yields:
        (episode 路径、成功时的报告或 None、失败时的异常或 None) 三元组。

    多进程模式使用 spawn 上下文，并按进程数分配视频处理线程。
    """
    if workers == 1:
        for episode in episodes:
            try:
                yield episode, convert_episode(episode, output_dir), None
            except Exception as error:
                yield episode, None, error
        return

    video_threads = max(1, (os.cpu_count() or 1) // workers)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_init_worker) as executor:
        futures = {
            executor.submit(convert_episode, episode, output_dir,
                            video_threads=video_threads): episode
            for episode in episodes
        }
        for future in as_completed(futures):
            episode = futures[future]
            try:
                yield episode, future.result(), None
            except Exception as error:
                yield episode, None, error


def main(argv: list[str] | None = None) -> int:
    """解析命令行并执行 HDF5 或 UMI 格式转换。

    Args:
        argv: 可选的命令行参数列表；None 时读取进程的命令行参数。

    Returns:
        全部成功时为 0，有 episode 转换失败时为 1；UMI 路径返回其转换器状态。

    Raises:
        ConversionError: 未找到输入 episode。

    参数格式或输入输出路径冲突时由 argparse 终止；HDF5 路径逐轮输出进度摘要。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="episode_N 目录或包含多个 episode_N 的目录")
    parser.add_argument("--output", required=True, type=Path,
                        help="输出 episode_N 的父目录")
    parser.add_argument("--workers", type=_positive_int,
                        default=min(4, os.cpu_count() or 1),
                        help="并行转换进程数，默认最多 4；设为 1 顺序转换")
    parser.add_argument("--format", choices=("hdf5", "umi"), default="hdf5",
                        help="hdf5 为每轮 HDF5/视频；umi 为合并的 Link7 训练 Zarr")
    parser.add_argument("--urdf", type=Path,
                        help="umi 格式必需：训练使用的 RM75 URDF")
    arguments = parser.parse_args(argv)
    source = arguments.input.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    episodes = discover_episodes(source)
    if output == source or output in episodes:
        parser.error("输出目录不能与输入 episode 或父目录相同")
    if arguments.format == "umi":
        if arguments.urdf is None:
            parser.error("--format umi 必须指定 --urdf")
        if output.suffix != ".zarr":
            parser.error("--format umi 的 --output 必须是新的 .zarr 目录")
        from convert_hardware_mcap_umi import convert_umi_episodes

        return convert_umi_episodes(
            episodes, output, arguments.urdf.expanduser().resolve(),
            min(arguments.workers, len(episodes)),
        )
    workers = min(arguments.workers, len(episodes))
    print(f"并发数: {workers}", flush=True)
    failed = 0
    for completed, (episode, report, error) in enumerate(
            convert_episodes(episodes, output, workers), 1):
        if error is not None:
            failed += 1
            print(f"[{completed}/{len(episodes)}] {episode.name}: 失败: {error}",
                  flush=True)
        else:
            print(f"[{completed}/{len(episodes)}] {episode.name}: "
                  f"完成，视频 {report['output_counts']['video_frames']} 帧",
                  flush=True)
    print(f"总计 {len(episodes)} 轮，成功 {len(episodes)-failed}，失败 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
