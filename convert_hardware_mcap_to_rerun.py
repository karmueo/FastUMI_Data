#!/usr/bin/env python3
"""Export hardware MCAP episodes to timestamped Rerun recordings.

The timeline uses rosbag receive timestamps, not ROS header stamps. Tracker
positions are metres in the source odometry frame; joint positions are radians.
No robot or camera extrinsic calibration is applied.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import tempfile
from typing import Any
import uuid

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
import rerun as rr
import rerun.blueprint as rrb
import rosbag2_py

from convert_hardware_mcap import (
    CAMERA_TOPICS,
    JOINT_NAMES,
    TOPICS,
    CameraRecord,
    ConversionError,
    _decode_frame,
    _fps,
    _write_video,
)


def _read_episode(bag: Path, spool: Any) -> dict[str, Any]:
    """Read the six known topics, retaining camera packets in a disk spool."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    available = {item.name: item.type for item in reader.get_all_topics_and_types()}
    for name, (topic, expected, _) in TOPICS.items():
        if topic not in available:
            raise ConversionError(f"bag 缺少必需话题: {topic}")
        if available[topic] != expected:
            raise ConversionError(f"{name} 类型应为 {expected}，实际为 {available[topic]}")
    camera_topics = [topic for topic in CAMERA_TOPICS if topic in available]
    if len(camera_topics) != 1:
        raise ConversionError(f"预期恰好一个相机话题，实际为 {camera_topics}")
    camera_topic = camera_topics[0]
    camera_mode, camera_type, camera_class = CAMERA_TOPICS[camera_topic]
    if available[camera_topic] != camera_type:
        raise ConversionError(f"相机类型应为 {camera_type}，实际为 {available[camera_topic]}")

    by_topic = {topic: (name, cls) for name, (topic, _, cls) in TOPICS.items()}
    by_topic[camera_topic] = ("camera", camera_class)
    series: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    cameras: list[CameraRecord] = []
    invalid: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    frame_id: str | None = None

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
            if len(names) != len(positions) or len(names) != len(set(names)):
                invalid[name] += 1
                continue
            ordered = dict(zip(names, positions))
            if any(joint not in ordered for joint in JOINT_NAMES):
                invalid[name] += 1
                continue
            values = np.asarray([ordered[joint] for joint in JOINT_NAMES], dtype=np.float64)
            if not np.isfinite(values).all():
                invalid[name] += 1
                continue
            series[name].append((bag_ns, values))
        elif name == "joint_action":
            values = np.asarray(message.joint, dtype=np.float64)
            if int(message.dof) != 7 or values.shape != (7,) or not np.isfinite(values).all():
                invalid[name] += 1
                continue
            series[name].append((bag_ns, values))
        elif name in ("gripper_state", "gripper_action"):
            value = float(message.data)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                invalid[name] += 1
                continue
            series[name].append((bag_ns, value))
        elif name == "tracker":
            pose = message.pose.pose
            position = np.asarray(
                [pose.position.x, pose.position.y, pose.position.z], dtype=np.float64
            )
            orientation = np.asarray(
                [pose.orientation.x, pose.orientation.y,
                 pose.orientation.z, pose.orientation.w], dtype=np.float64
            )
            if (not np.isfinite(position).all() or not np.isfinite(orientation).all()
                    or np.linalg.norm(orientation) < 1e-12):
                invalid[name] += 1
                continue
            current_frame = message.header.frame_id
            if not current_frame:
                raise ConversionError("Tracker odom 缺少 header.frame_id")
            if frame_id is not None and current_frame != frame_id:
                raise ConversionError("Tracker odom 的坐标系在同一 episode 内发生变化")
            frame_id = current_frame
            series[name].append((bag_ns, (position, orientation)))
        else:
            payload = bytes(message.data)
            if camera_mode == "raw":
                encoding = message.encoding.lower()
                channels = {"bgr8": 3, "rgb8": 3, "mono8": 1}.get(encoding)
                width, height, step = int(message.width), int(message.height), int(message.step)
                valid = bool(channels and width > 0 and height > 0
                             and step >= width * channels and len(payload) >= step * height)
                keyframe = False
            elif camera_mode == "jpeg":
                encoding = message.format.lower()
                width = height = step = 0
                valid = bool(("jpeg" in encoding or "jpg" in encoding)
                             and payload.startswith(b"\xff\xd8")
                             and payload.endswith(b"\xff\xd9"))
                if valid:
                    try:
                        decoded = cv2.imdecode(
                            np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                    except cv2.error:
                        decoded = None
                    valid = decoded is not None
                keyframe = False
            else:
                encoding = message.encoding.lower()
                width, height, step = int(message.width), int(message.height), 0
                valid = bool(encoding.split(";", 1)[0] == "h264"
                             and width > 0 and height > 0 and payload)
                keyframe = bool(message.flags & 1)
            if not valid:
                invalid[name] += 1
                continue
            offset = spool.tell()
            spool.write(payload)
            cameras.append(CameraRecord(
                bag_ns=bag_ns, header_ns=0, offset=offset, size=len(payload),
                width=width, height=height, encoding=encoding, step=step,
                keyframe=keyframe,
            ))

    for name in TOPICS:
        if not series[name]:
            raise ConversionError(f"缺少有效的 {name} 样本")
    if not cameras:
        raise ConversionError("缺少有效的相机帧")
    for records in series.values():
        records.sort(key=lambda item: item[0])
    cameras.sort(key=lambda item: item.bag_ns)
    return {
        "series": series, "camera": cameras, "camera_mode": camera_mode,
        "camera_topic": camera_topic, "frame_id": frame_id,
        "counts": counts, "invalid": invalid,
    }


def _time(recording: rr.RecordingStream, bag_ns: int) -> None:
    """Preserve the full nanosecond receive timestamp on one Rerun timeline."""
    recording.set_time("bag_receive_time", timestamp=np.datetime64(bag_ns, "ns"))


def _log_numeric(recording: rr.RecordingStream, data: dict[str, Any]) -> None:
    """Log joint and gripper values without aligning their sample rates."""
    series = data["series"]
    for source, root in (("joint_state", "joint/feedback"),
                         ("joint_action", "joint/command")):
        for index, joint in enumerate(JOINT_NAMES):
            recording.log(f"{root}/{joint}", rr.SeriesLines(names=f"{joint} (rad)"),
                          static=True)
        for bag_ns, values in series[source]:
            _time(recording, bag_ns)
            for index, joint in enumerate(JOINT_NAMES):
                recording.log(f"{root}/{joint}", rr.Scalars(float(values[index])))
    for source, root in (("gripper_state", "gripper/feedback"),
                         ("gripper_action", "gripper/command")):
        recording.log(root, rr.SeriesLines(names="openness [0, 1]"), static=True)
        for bag_ns, value in series[source]:
            _time(recording, bag_ns)
            recording.log(root, rr.Scalars(value))


def _log_tracker(recording: rr.RecordingStream, data: dict[str, Any]) -> None:
    """Show poses and the complete path in the source odometry frame."""
    root = f"world/{data['frame_id']}"
    poses = data["series"]["tracker"]
    positions = np.stack([pose[0] for _, pose in poses])
    recording.log(root, rr.ViewCoordinates.FLU, static=True)
    recording.log(f"{root}/path", rr.LineStrips3D([positions]), static=True)
    recording.log(f"{root}/tracker", rr.TransformAxes3D(0.10), static=True)
    for bag_ns, (position, orientation) in poses:
        _time(recording, bag_ns)
        recording.log(
            f"{root}/tracker",
            rr.Transform3D(translation=position, quaternion=orientation),
        )


def _log_camera(recording: rr.RecordingStream, data: dict[str, Any],
                spool: Any, temporary_dir: Path) -> int:
    """Log JPEG images; decode H.264 during export for Viewer compatibility."""
    cameras = data["camera"]
    mode = data["camera_mode"]
    path = "camera/wrist"
    if mode == "h264":
        first_key = next((index for index, frame in enumerate(cameras)
                          if frame.keyframe), None)
        if first_key is None:
            raise ConversionError("相机没有 H.264 关键帧，无法解码")
        decodable = cameras[first_key:]
        video_path = temporary_dir / "wrist.mp4"
        _write_video(video_path, decodable, mode, spool, _fps(decodable))
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ConversionError("H.264 视频无法打开以提取图像帧")
        try:
            for frame in decodable:
                success, bgr = capture.read()
                if not success:
                    raise ConversionError("H.264 解码帧数少于 MCAP 相机包数")
                encoded, jpeg = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                if not encoded:
                    raise ConversionError("H.264 解码图像无法编码为 JPEG")
                _time(recording, frame.bag_ns)
                recording.log(
                    path, rr.EncodedImage(contents=jpeg.tobytes(), media_type="image/jpeg")
                )
            if capture.read()[0]:
                raise ConversionError("H.264 解码帧数多于 MCAP 相机包数")
        finally:
            capture.release()
        return first_key

    for frame in cameras:
        _time(recording, frame.bag_ns)
        if mode == "jpeg":
            spool.seek(frame.offset)
            payload = spool.read(frame.size)
            if len(payload) != frame.size:
                raise ConversionError("JPEG 相机临时缓存截断")
            recording.log(path, rr.EncodedImage(contents=payload, media_type="image/jpeg"))
        else:
            bgr = _decode_frame(spool, frame, mode)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            recording.log(path, rr.Image(rgb))
    return 0


def _blueprint(data: dict[str, Any]) -> rrb.Blueprint:
    """Open the episode with its image, odometry, and numeric plots visible."""
    return rrb.Blueprint(
        rrb.Horizontal(contents=[
            rrb.Spatial2DView(name="Wrist camera", origin="/camera/wrist"),
            rrb.Vertical(contents=[
                rrb.Spatial3DView(
                    name="Tracker (m)", origin=f"/world/{data['frame_id']}"
                ),
                rrb.TimeSeriesView(name="Joints (rad)", origin="/joint"),
                rrb.TimeSeriesView(name="Gripper openness", origin="/gripper"),
            ]),
        ]),
        rrb.TimePanel(state="expanded"),
    )


def convert_episode(source: Path, output_dir: Path) -> dict[str, Any]:
    """Write one RRD via a temporary path and publish it only on success."""
    destination = output_dir / f"{source.name}.rrd"
    if destination.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖: {destination}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = output_dir / f".{source.name}.{uuid.uuid4().hex}.rrd"
    try:
        with tempfile.TemporaryFile(mode="w+b") as spool:
            data = _read_episode(source / "bag", spool)
            with tempfile.TemporaryDirectory() as video_dir:
                with rr.RecordingStream("fastumi_hardware_episode") as recording:
                    recording.save(stage, default_blueprint=_blueprint(data))
                    _log_numeric(recording, data)
                    _log_tracker(recording, data)
                    skipped = _log_camera(recording, data, spool, Path(video_dir))
        if destination.exists():
            raise FileExistsError(f"目标已存在，拒绝覆盖: {destination}")
        stage.rename(destination)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise
    counts = {name: len(data["series"][name]) for name in TOPICS}
    counts["camera"] = len(data["camera"]) - skipped
    return {
        "output": str(destination), "counts": counts,
        "invalid": dict(data["invalid"]), "camera_skipped_until_keyframe": skipped,
        "camera_topic": data["camera_topic"], "tracker_frame": data["frame_id"],
    }


def discover_rerun_episodes(source: Path) -> list[Path]:
    """Find numbered MCAP episodes at any depth below a source directory."""
    if (source / "bag" / "metadata.yaml").is_file():
        return [source]
    if not source.is_dir():
        raise ConversionError(f"没有找到 episode_N/bag: {source}")
    episodes = [
        path for path in source.rglob("episode_*")
        if path.is_dir() and path.name[8:].isdigit()
        and (path / "bag" / "metadata.yaml").is_file()
    ]
    if not episodes:
        raise ConversionError(f"没有找到 episode_N/bag: {source}")
    return sorted(episodes, key=lambda path: (path.parent.relative_to(source).parts,
                                               int(path.name[8:])))


def main(argv: list[str] | None = None) -> int:
    """Export one episode or all numbered episodes below a directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="episode_N 目录或包含 episode_N 的根目录（递归查找）")
    parser.add_argument("--output", required=True, type=Path,
                        help=".rrd 输出根目录（保留输入下的子目录结构）")
    args = parser.parse_args(argv)
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    episodes = discover_rerun_episodes(source)
    single_episode = episodes == [source]
    failed = 0
    for episode in episodes:
        destination_dir = output if single_episode else output / episode.parent.relative_to(source)
        try:
            report = convert_episode(episode, destination_dir)
            print(f"{episode}: 完成 -> {report['output']}，"
                  f"相机 {report['counts']['camera']} 帧，"
                  f"关键帧前跳过 {report['camera_skipped_until_keyframe']} 帧，"
                  f"其他采样 {report['counts']}")
            if report["invalid"]:
                print(f"{episode}: 无效消息 {report['invalid']}")
        except Exception as error:
            failed += 1
            print(f"{episode}: 失败: {error}")
    print(f"总计 {len(episodes)} 轮，成功 {len(episodes) - failed}，失败 {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
