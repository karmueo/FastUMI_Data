#!/usr/bin/env python3
"""Convert hardware.launch.py MCAP recordings to RM75 HDF5/video episodes.

All output timestamps use the recorder's rosbag receive clock.  Header stamps
are retained as diagnostics only because the recorder and Tracker may run on
different, unsynchronised hosts.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
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


JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
TOPICS = {
    "joint_state": ("/joint_states", "sensor_msgs/msg/JointState", JointState),
    "joint_action": ("/rm_driver/movej_canfd_cmd", "rm_ros_interfaces/msg/Jointpos", Jointpos),
    "gripper_state": ("/motion_control/gripper_state", "std_msgs/msg/Float32", Float32),
    "gripper_action": ("/motion_control/gripper_command", "std_msgs/msg/Float32", Float32),
    "tracker": ("/vive_tracker/odom", "nav_msgs/msg/Odometry", Odometry),
}
CAMERA_TOPICS = {
    "/wrist_camera/image_raw": ("raw", "sensor_msgs/msg/Image", Image),
    "/wrist_camera/image_raw/compressed": (
        "jpeg", "sensor_msgs/msg/CompressedImage", CompressedImage
    ),
    "/wrist_camera/image_raw/ffmpeg": (
        "h264", "ffmpeg_image_transport_msgs/msg/FFMPEGPacket", FFMPEGPacket
    ),
}
FORMAT_VERSION = "rm75-single-arm-v1"


class ConversionError(ValueError):
    """The source episode cannot produce a complete, aligned output."""


@dataclass(frozen=True)
class CameraRecord:
    """One image payload stored in a seekable spool and its original timing."""

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
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _seconds(timestamps_ns: list[int]) -> np.ndarray:
    return np.asarray(timestamps_ns, dtype=np.float64) / 1_000_000_000.0


def _diagnostic(values_ns: list[int]) -> dict[str, float | int]:
    """Summarise bag minus source time, without claiming clock calibration."""
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


def _read_episode(bag: Path, spool: Any) -> dict[str, Any]:
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    available = {item.name: item.type for item in reader.get_all_topics_and_types()}
    missing = [name for name, (topic, _, _) in TOPICS.items() if topic not in available]
    if missing:
        raise ConversionError(f"bag 缺少必需话题: {', '.join(missing)}")
    for name, (topic, expected, _) in TOPICS.items():
        if available[topic] != expected:
            raise ConversionError(f"{name} 类型应为 {expected}，实际为 {available[topic]}")
    cameras = [topic for topic in CAMERA_TOPICS if topic in available]
    if len(cameras) != 1:
        raise ConversionError(f"预期恰好一个相机话题，实际为 {cameras}")
    camera_topic = cameras[0]
    camera_mode, camera_type, camera_class = CAMERA_TOPICS[camera_topic]
    if available[camera_topic] != camera_type:
        raise ConversionError(f"相机类型应为 {camera_type}，实际为 {available[camera_topic]}")

    by_topic = {spec[0]: (name, spec[2]) for name, spec in TOPICS.items()}
    by_topic[camera_topic] = ("camera", camera_class)
    series: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    camera_records: list[CameraRecord] = []
    counts: Counter[str] = Counter()
    invalid: Counter[str] = Counter()
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

    # A command just before the action window remains the active command at
    # the first in-window state sample.  Do not discard it while cropping.
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
    source = [record.header_ns for record in records]
    if not all(timestamp > 0 for timestamp in source):
        source = [record.bag_ns for record in records]
    if len(source) < 2 or source[-1] <= source[0]:
        return 30.0
    # Capture jitter can bias the median interval by a whole FPS.  The full
    # span gives an MP4 duration close to the timestamped source duration.
    estimated = (len(source) - 1) * 1_000_000_000.0 / (source[-1] - source[0])
    if not 1.0 <= estimated <= 120.0:
        raise ConversionError(f"相机帧率无法估计: {estimated}")
    nearest = round(estimated)
    return float(nearest if abs(estimated - nearest) / estimated < 0.02
                 else round(estimated, 3))


def _payload(spool: Any, record: CameraRecord) -> bytes:
    spool.seek(record.offset)
    payload = spool.read(record.size)
    if len(payload) != record.size:
        raise ConversionError("相机临时缓存截断")
    return payload


def _probe_frames(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_frames", "-select_streams", "v:0",
         "-show_entries", "frame=pkt_pos", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ConversionError(f"ffprobe 解码失败: {result.stderr.strip()}")
    return json.loads(result.stdout).get("frames", [])


def _run_ffmpeg(command: list[str], payloads: Any = None) -> None:
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
                 spool: Any, fps: float) -> None:
    if mode == "h264":
        bitstream_path = path.with_suffix(".h264")
        offsets = []
        with bitstream_path.open("wb") as bitstream:
            for record in records:
                offsets.append(bitstream.tell())
                bitstream.write(_payload(spool, record))
        frames = _probe_frames(bitstream_path)
        frame_offsets = [int(frame["pkt_pos"]) for frame in frames
                         if "pkt_pos" in frame]
        if frame_offsets != offsets:
            raise ConversionError(
                "H.264 解码帧与 MCAP 包不是一一对应，拒绝写入错误时间戳"
            )
        _run_ffmpeg(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-framerate", str(fps), "-f", "h264", "-i", str(bitstream_path),
             "-an", "-c:v", "copy", "-movflags", "+faststart", str(path)]
        )
        bitstream_path.unlink()
    else:
        first = _decode_frame(spool, records[0], mode)
        height, width = first.shape[:2]

        def frames():
            yield first.tobytes()
            for record in records[1:]:
                image = _decode_frame(spool, record, mode)
                if image.shape != first.shape:
                    raise ConversionError("视频帧尺寸不一致")
                yield image.tobytes()

        _run_ffmpeg(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size",
             f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
             "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            frames(),
        )
    if len(_probe_frames(path)) != len(records):
        raise ConversionError("MP4 帧数与 HDF5 相机时间戳数量不一致")


def _write_hdf5(path: Path, selected: dict[str, Any]) -> None:
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
    """Accept one episode or a directory containing numbered episodes."""
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


def convert_episode(source: Path, output_dir: Path) -> dict[str, Any]:
    """Convert one episode into a staging directory, then publish it."""
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
                         data["camera_mode"], spool, fps)
            _write_hdf5(stage / "proprio.hdf5", selected)

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="episode_N 目录或包含多个 episode_N 的目录")
    parser.add_argument("--output", required=True, type=Path,
                        help="输出 episode_N 的父目录")
    arguments = parser.parse_args(argv)
    source = arguments.input.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    episodes = discover_episodes(source)
    if output == source or output in episodes:
        parser.error("输出目录不能与输入 episode 或父目录相同")
    failed = 0
    for episode in episodes:
        try:
            report = convert_episode(episode, output)
            print(f"{episode.name}: 完成，视频 {report['output_counts']['video_frames']} 帧")
        except Exception as error:
            failed += 1
            print(f"{episode.name}: 失败: {error}")
    print(f"总计 {len(episodes)} 轮，成功 {len(episodes)-failed}，失败 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
