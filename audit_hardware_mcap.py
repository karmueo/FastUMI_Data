#!/usr/bin/env python3
"""检查 ROS 2 MCAP episode 的相机和关节数据，并交互处理异常目录。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from ffmpeg_image_transport_msgs.msg import FFMPEGPacket
from rclpy.serialization import deserialize_message
import rosbag2_py
from rm_ros_interfaces.msg import Jointpos
from sensor_msgs.msg import CompressedImage, Image, JointState


JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
JOINT_TOPICS = {
    "/joint_states": ("joint_state", "sensor_msgs/msg/JointState", JointState),
    "/rm_driver/movej_canfd_cmd": (
        "joint_action", "rm_ros_interfaces/msg/Jointpos", Jointpos),
}
CAMERA_TOPICS = {
    "/wrist_camera/image_raw": ("raw", "sensor_msgs/msg/Image", Image),
    "/wrist_camera/image_raw/compressed": (
        "jpeg", "sensor_msgs/msg/CompressedImage", CompressedImage),
    "/wrist_camera/image_raw/ffmpeg": (
        "h264", "ffmpeg_image_transport_msgs/msg/FFMPEGPacket", FFMPEGPacket),
}


@dataclass(frozen=True)
class Thresholds:
    """相机和关节状态相邻消息的最大允许间隔，单位为毫秒。"""

    camera_ms: float = 200.0
    joint_state_ms: float = 200.0


@dataclass(frozen=True)
class Finding:
    """单轮采集的一项可定位异常。"""

    code: str
    detail: str


@dataclass
class EpisodeReport:
    """一轮采集的检查结果和处理前目录快照。"""

    path: Path
    findings: list[Finding] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    snapshot: tuple[tuple[str, int, int], ...] = ()

    @property
    def abnormal(self) -> bool:
        """只要发现一项问题，即需要人工决定如何处理该轮。"""
        return bool(self.findings)


def _finding(report: EpisodeReport, code: str, detail: str) -> None:
    report.findings.append(Finding(code, detail))


def _snapshot(path: Path) -> tuple[tuple[str, int, int], ...]:
    """记录普通文件的大小和修改时间，防止扫描后文件被替换。"""
    return tuple(sorted(
        (item.relative_to(path).as_posix(), item.stat().st_size, item.stat().st_mtime_ns)
        for item in path.rglob("*") if item.is_file() and not item.is_symlink()
    ))


def discover_episodes(root: Path) -> list[Path]:
    """递归寻找编号目录，不进入隐藏目录或 episode 内部。"""
    episodes = []
    for parent, directories, _ in os.walk(root, followlinks=False):
        visible = []
        for name in directories:
            if name.startswith("."):
                continue
            if name.startswith("episode_") and name[8:].isdigit():
                episodes.append(Path(parent) / name)
            else:
                visible.append(name)
        directories[:] = visible
    return sorted(episodes, key=lambda path: (str(path.parent), int(path.name[8:])))


def _stamp_ns(stamp: object) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _check_timing(report: EpisodeReport, label: str, timestamps: list[int],
                  gap_ms: float) -> None:
    """检查非正时间、倒退、重复和超过阈值的间隔。"""
    if not timestamps:
        _finding(report, f"{label}_empty", f"{label} 没有样本")
        return
    invalid = sum(value <= 0 for value in timestamps)
    if invalid:
        _finding(report, f"{label}_timestamp_invalid", f"{label} 非正时间戳 {invalid} 个")
    deltas = [right - left for left, right in zip(timestamps, timestamps[1:])]
    backward = [delta for delta in deltas if delta < 0]
    if backward:
        _finding(report, f"{label}_backward",
                 f"{label} 时间倒退 {len(backward)} 次，最大 {-min(backward) / 1e6:.1f} ms")
    repeated = sum(delta == 0 for delta in deltas)
    if repeated:
        _finding(report, f"{label}_repeated", f"{label} 时间戳重复 {repeated} 次")
    gaps = [delta for delta in deltas if delta > gap_ms * 1e6]
    if gaps:
        _finding(report, f"{label}_gap",
                 f"{label} 间断 {len(gaps)} 次，最长 {max(gaps) / 1e6:.1f} ms"
                 f"（阈值 {gap_ms:g} ms）")


def _valid_joint_state(message: JointState) -> bool:
    names = list(message.name)
    positions = list(message.position)
    return (len(names) == len(positions) and len(set(names)) == len(names)
            and all(name in names for name in JOINT_NAMES)
            and all(math.isfinite(positions[names.index(name)]) for name in JOINT_NAMES))


def _valid_joint_action(message: Jointpos) -> bool:
    return (int(message.dof) == 7 and len(message.joint) == 7
            and all(math.isfinite(value) for value in message.joint))


def _valid_camera(message: object, mode: str) -> bool:
    if mode == "raw":
        channels = {"bgr8": 3, "rgb8": 3, "mono8": 1}.get(message.encoding.lower())
        return bool(channels and message.width > 0 and message.height > 0
                    and message.step >= message.width * channels
                    and len(message.data) >= message.step * message.height)
    if mode == "jpeg":
        payload = bytes(message.data)
        if not (("jpeg" in message.format.lower() or "jpg" in message.format.lower())
                and payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9")):
            return False
        return cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR) is not None
    return bool(message.encoding.split(";", 1)[0].lower() == "h264"
                and message.width > 0 and message.height > 0 and message.data)


def _check_h264(report: EpisodeReport, packets: list[tuple[bytes, bool]]) -> None:
    """从首个关键帧起，核对 FFmpeg 解码帧与 ROS 包的原始偏移。"""
    first_key = next((i for i, (_, keyframe) in enumerate(packets) if keyframe), None)
    if first_key is None:
        _finding(report, "camera_no_keyframe", "H.264 中没有关键帧")
        return
    with tempfile.TemporaryDirectory(prefix="fastumi-audit-") as directory:
        bitstream = Path(directory) / "camera.h264"
        offsets = []
        with bitstream.open("wb") as stream:
            for payload, _ in packets[first_key:]:
                offsets.append(stream.tell())
                stream.write(payload)
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-f", "h264", "-show_frames",
                 "-select_streams", "v:0", "-show_entries", "frame=pkt_pos",
                 "-of", "default=nw=1:nk=1", str(bitstream)],
                capture_output=True, text=True, check=False,
            )
        except OSError as error:
            _finding(report, "camera_probe_error", f"无法运行 ffprobe: {error}")
            return
    if result.returncode:
        _finding(report, "camera_probe_error",
                 f"ffprobe 解码失败: {result.stderr.strip()[-300:]}")
        return
    # Jetson 的 NVIDIA 库可能在 stdout 写提示；只接受独占一行的十进制偏移。
    decoded = [int(line) for line in result.stdout.splitlines() if line.strip().isdecimal()]
    if decoded != offsets:
        _finding(report, "camera_frame_mismatch",
                 f"H.264 包 {len(offsets)} 个，解码帧 {len(decoded)} 个，偏移不一致")


def audit_episode(path: Path, thresholds: Thresholds) -> EpisodeReport:
    """只读取一轮的相机和关节消息，其他话题不反序列化。"""
    report = EpisodeReport(path)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        _finding(report, "symlink", "episode 路径包含符号链接")
        return report
    if not path.is_dir():
        _finding(report, "missing_episode", "episode 目录不存在")
        return report
    try:
        report.snapshot = _snapshot(path)
        recording = path / "recording.json"
        if not recording.is_file():
            _finding(report, "missing_recording", "缺少 recording.json")
        else:
            try:
                json.loads(recording.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                _finding(report, "invalid_recording", f"recording.json 无法读取: {error}")
        bag = path / "bag"
        if not (bag / "metadata.yaml").is_file() or not any(
                item.is_file() and item.stat().st_size > 0 for item in bag.glob("*.mcap")):
            _finding(report, "missing_bag", "bag 缺少 metadata.yaml 或非空 MCAP 文件")
            return report
        if any(item.is_symlink() for item in path.rglob("*")):
            _finding(report, "symlink", "episode 内容包含符号链接")
            return report
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                    rosbag2_py.ConverterOptions("", ""))
        metadata = rosbag2_py.Info().read_metadata(str(bag), "mcap")
        expected_counts = {
            item.topic_metadata.name: item.message_count
            for item in metadata.topics_with_message_count
        }
        available = {item.name: item.type for item in reader.get_all_topics_and_types()}
        for topic, (label, expected, _) in JOINT_TOPICS.items():
            if topic not in available:
                _finding(report, f"{label}_missing", f"缺少话题 {topic}")
            elif available[topic] != expected:
                _finding(report, f"{label}_type", f"{topic} 类型为 {available[topic]}，应为 {expected}")
        cameras = [topic for topic in CAMERA_TOPICS if topic in available]
        if len(cameras) != 1:
            _finding(report, "camera_topic", f"预期一个相机话题，实际 {cameras}")
        camera_topic = cameras[0] if len(cameras) == 1 else None
        if camera_topic:
            mode, expected, _ = CAMERA_TOPICS[camera_topic]
            if available[camera_topic] != expected:
                _finding(report, "camera_type",
                         f"{camera_topic} 类型为 {available[camera_topic]}，应为 {expected}")
                camera_topic = None
        selected = {topic: spec for topic, spec in JOINT_TOPICS.items()
                    if available.get(topic) == spec[1]}
        if camera_topic:
            selected[camera_topic] = CAMERA_TOPICS[camera_topic]
        if selected:
            reader.set_filter(rosbag2_py.StorageFilter(topics=list(selected)))
        bag_times = {"camera": [], "joint_state": [], "joint_action": []}
        source_times = {"camera": [], "joint_state": []}
        invalid = {"camera": 0, "joint_state": 0, "joint_action": 0}
        pts_header_mismatch = 0
        packets: list[tuple[bytes, bool]] = []
        while reader.has_next():
            topic, raw, bag_ns = reader.read_next()
            if topic not in selected:
                continue
            label, _, message_type = selected[topic]
            name = "camera" if topic == camera_topic else label
            bag_times[name].append(int(bag_ns))
            try:
                message = deserialize_message(raw, message_type)
                if name == "joint_state":
                    source_times[name].append(_stamp_ns(message.header.stamp))
                    valid = _valid_joint_state(message)
                elif name == "joint_action":
                    valid = _valid_joint_action(message)
                else:
                    if mode == "h264" and int(message.pts) != _stamp_ns(message.header.stamp):
                        pts_header_mismatch += 1
                    source_times[name].append(
                        int(message.pts) if mode == "h264" else _stamp_ns(message.header.stamp))
                    valid = _valid_camera(message, mode)
                    if valid and mode == "h264":
                        packets.append((bytes(message.data), bool(message.flags & 1)))
                if not valid:
                    invalid[name] += 1
            except Exception:
                invalid[name] += 1
        for name, count in invalid.items():
            report.counts[name] = len(bag_times[name])
            if count:
                _finding(report, f"{name}_invalid", f"{name} 无效消息 {count} / {len(bag_times[name])}")
        if pts_header_mismatch:
            _finding(report, "camera_pts_header_mismatch",
                     f"相机 PTS 与 header 时间戳不一致 {pts_header_mismatch} 次")
        for topic in selected:
            name = "camera" if topic == camera_topic else selected[topic][0]
            expected = expected_counts.get(topic)
            if expected is not None and expected != len(bag_times[name]):
                _finding(report, f"{name}_metadata_count",
                         f"{name} 元数据计数 {expected}，实际读出 {len(bag_times[name])}")
        for name in ("camera", "joint_state"):
            times = bag_times[name]
            if name == "camera" and camera_topic is None:
                continue
            if name == "joint_state" and "/joint_states" not in selected:
                continue
            _check_timing(report, f"{name}_bag", times, getattr(thresholds, f"{name}_ms"))
        if "/rm_driver/movej_canfd_cmd" in selected and not bag_times["joint_action"]:
            _finding(report, "joint_action_empty", "joint_action 没有样本")
        if source_times["camera"]:
            _check_timing(report, "camera_source", source_times["camera"], thresholds.camera_ms)
        if source_times["joint_state"]:
            _check_timing(report, "joint_state_source", source_times["joint_state"],
                          thresholds.joint_state_ms)
        if camera_topic and mode == "h264" and packets:
            _check_h264(report, packets)
    except Exception as error:
        _finding(report, "read_error", f"读取失败: {type(error).__name__}: {error}")
    return report


def _lock_recorder(root: Path) -> list[object]:
    """检查任一上级录制目录的锁；处理期间保留文件描述符。"""
    locked = []
    try:
        for parent in (root, *root.parents):
            path = parent / ".recorder.lock"
            if path.exists():
                stream = path.open("rb")
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    stream.close()
                    raise
                locked.append(stream)
    except (OSError, BlockingIOError):
        for stream in locked:
            stream.close()
        raise RuntimeError("录制器正在占用数据根目录，拒绝处理") from None
    return locked


def process_abnormal(root: Path, reports: list[EpisodeReport], action: str) -> int:
    """先校验所有异常目录，再执行统一删除或隔离。"""
    if action not in ("delete", "move"):
        raise ValueError("action 只能是 delete 或 move")
    abnormal = [report for report in reports if report.abnormal]
    if not abnormal:
        return 0
    locks = _lock_recorder(root)
    try:
        destinations = []
        for report in abnormal:
            path = report.path
            if (path.is_symlink() or any(parent.is_symlink() for parent in path.parents
                                         if parent == root or parent.is_relative_to(root))
                    or any(item.is_symlink() for item in path.rglob("*"))):
                raise RuntimeError(f"符号链接不允许处理: {path}")
            if not path.is_dir() or not path.is_relative_to(root):
                raise RuntimeError(f"episode 已移动或超出根目录: {path}")
            if _snapshot(path) != report.snapshot:
                raise RuntimeError(f"扫描后文件已变化，请重新扫描: {path}")
            destination = root / ".quarantine" / path.relative_to(root)
            destination_parents = [parent for parent in destination.parents
                                   if parent == root or parent.is_relative_to(root)]
            if action == "move" and (
                    destination.exists() or destination.is_symlink()
                    or any(parent.is_symlink() or (parent.exists() and not parent.is_dir())
                           for parent in destination_parents)):
                raise RuntimeError(f"隔离目标已存在或包含符号链接: {destination}")
            destinations.append((path, destination))
        for path, destination in destinations:
            if action == "delete":
                shutil.rmtree(path)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.rename(destination)
            print(f"已{'删除' if action == 'delete' else '移动'}: {path}")
        return len(destinations)
    finally:
        for stream in locks:
            stream.close()


def _positive_ms(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("阈值必须是有限正数（毫秒）")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("并行进程数必须是正整数")
    return number


def scan_episodes(episodes: list[Path], thresholds: Thresholds,
                  workers: int):
    """并行检查相互独立的 episode，按完成顺序返回扫描进度。"""
    if workers == 1:
        for index, episode in enumerate(episodes):
            yield index, audit_episode(episode, thresholds)
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(episodes)),
                             mp_context=context) as executor:
        futures = {executor.submit(audit_episode, episode, thresholds): index
                   for index, episode in enumerate(episodes)}
        for future in as_completed(futures):
            yield futures[future], future.result()


def select_abnormal(root: Path, abnormal: list[EpisodeReport], scope: str,
                    names: str = "") -> list[EpisodeReport]:
    """从异常轮次中选择全部、含相机异常或明确指定的目录。"""
    if scope == "a":
        return abnormal
    if scope == "c":
        return [report for report in abnormal
                if any(finding.code.startswith("camera_") for finding in report.findings)]
    if scope != "s":
        raise ValueError("未知处理范围")
    tokens = re.split(r"[,，\s]+", names.strip()) if names.strip() else []
    if not tokens:
        raise ValueError("未指定 episode")
    by_path = {report.path.relative_to(root).as_posix(): report for report in abnormal}
    by_name: dict[str, list[EpisodeReport]] = {}
    for report in abnormal:
        by_name.setdefault(report.path.name, []).append(report)
    chosen = set()
    for token in tokens:
        if token.isdecimal():
            token = f"episode_{int(token)}"
        if token in by_path:
            chosen.add(by_path[token].path)
        elif token in by_name:
            matches = by_name[token]
            if len(matches) != 1:
                raise ValueError(f"{token} 对应多个目录，请输入相对路径")
            chosen.add(matches[0].path)
        else:
            raise ValueError(f"{token} 不在异常 episode 列表中")
    return [report for report in abnormal if report.path in chosen]


def main(argv: list[str] | None = None) -> int:
    """扫描、报告并在交互终端选择处理范围和操作。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="采集数据根目录")
    parser.add_argument("--camera-gap-ms", type=_positive_ms, default=200.0)
    parser.add_argument("--joint-state-gap-ms", type=_positive_ms, default=200.0)
    parser.add_argument("--joint-action-gap-ms", type=_positive_ms,
                        help="已弃用；兼容旧命令，不检查关节指令时间间隔")
    parser.add_argument("--workers", type=_positive_int,
                        default=min(4, os.cpu_count() or 1),
                        help="并行检查进程数，默认最多 4；设为 1 顺序检查")
    args = parser.parse_args(argv)
    if args.input.is_symlink() or not args.input.is_dir():
        parser.error("输入必须是非符号链接的现有目录")
    root = args.input.expanduser().resolve(strict=True)
    if root.name.startswith("episode_") and root.name[8:].isdigit():
        parser.error("请输入包含 episode_N 的根目录，而不是单个 episode")
    episodes = discover_episodes(root)
    if not episodes:
        parser.error("没有找到 episode_N 目录")
    thresholds = Thresholds(args.camera_gap_ms, args.joint_state_gap_ms)
    reports: list[EpisodeReport | None] = [None] * len(episodes)
    for completed, (index, report) in enumerate(
            scan_episodes(episodes, thresholds, args.workers), 1):
        reports[index] = report
        print(f"[{completed}/{len(episodes)}] {report.path.relative_to(root)}: "
              f"{'异常' if report.abnormal else '正常'}")
        for finding in report.findings:
            print(f"  - {finding.detail}")
    reports = [report for report in reports if report is not None]
    abnormal = [report for report in reports if report.abnormal]
    print(f"总计 {len(reports)} 轮，正常 {len(reports)-len(abnormal)}，异常 {len(abnormal)}")
    if not abnormal or not sys.stdin.isatty():
        return 1 if abnormal else 0
    camera_count = sum(any(finding.code.startswith("camera_")
                           for finding in report.findings) for report in abnormal)
    scope = input(f"处理范围：[s] 指定 episode / [c] 含相机异常（{camera_count}） / "
                  "[a] 全部异常 / [n] 不修改（默认）：").strip().lower()
    if scope not in ("s", "c", "a"):
        print("未修改任何数据")
        return 1
    names = input("输入异常 episode 编号、名称或相对路径，用逗号或空格分隔：").strip() \
        if scope == "s" else ""
    try:
        selected = select_abnormal(root, abnormal, scope, names)
    except ValueError as error:
        print(f"选择失败: {error}；未修改任何数据")
        return 1
    if not selected:
        print("所选范围没有异常 episode，未修改任何数据")
        return 1
    print(f"已选 {len(selected)} 轮：")
    for report in selected:
        print(f"  {report.path.relative_to(root)}")
    choice = input("处理所选 episode：[d] 删除 / [m] 移动到 .quarantine / "
                   "[n] 不修改（默认）：").strip().lower()
    if choice not in ("d", "m"):
        print("未修改任何数据")
        return 1
    if choice == "d":
        confirmation = input(f"删除 {len(selected)} 个目录不可恢复。请输入 DELETE "
                             f"{len(selected)} 确认：").strip()
        if confirmation != f"DELETE {len(selected)}":
            print("确认不匹配，未修改任何数据")
            return 1
    try:
        count = process_abnormal(root, selected, "delete" if choice == "d" else "move")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"处理失败: {error}", file=sys.stderr)
        return 2
    print(f"处理完成：{count} 个异常 episode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
