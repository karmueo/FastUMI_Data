#!/usr/bin/env python3
"""把 FastUMI MCAP 或 LeRobot v3 数据逐条导出为 Rerun ``.rrd``，用于查看相机、Tracker 与夹爪。

LeRobot 输入是含 meta/info.json 的数据集或其上层目录；每个 episode 输出一个 RRD，
仅支持 FastUMI xyz_rotvec_gripper 契约，无需 ROS，详见 docs/Rerun采集数据可视化.md。

MCAP 输入是含 ``session.yaml`` 的记录目录（例如 ``2026-10-14_16-28-50_0``），或包含多条记录的
任意上层目录（递归查找，跳过 ``_analysis``、``.staging``、``.trash``）。话题按记录自身
``session.yaml`` 的 ``topic_map`` 读取，图像模式按 bag 中的消息类型判定。

导出规则：

* 两条时间轴：``bag_receive_time`` 为 rosbag 接收时间；``header_time`` 为消息头时间
  （UMI 图像头是 V4L2 映射时间，夹爪状态头与对应图像相同）。不裁剪、不插值。
* ``compressed`` 图像原样写入；``raw`` 与 ``h264`` 在导出时解码并编码为 JPEG，
  Viewer 显示不依赖外部 FFmpeg。H.264 首个关键帧之前的包无法解码，会计入跳过数。
* Tracker 位置单位为米，保留源 ``header.frame_id`` 坐标系，不施加任何外参；
  非有限值或四元数范数偏离 1 超过容差的样本被跳过并计数。
* 夹爪开度为 ``[0, 1]``；NaN 帧不写入。二值化副本中 ``filtered_openness`` 为 0/1。
* 源 MCAP 不会被修改；已有同名 ``.rrd`` 时拒绝覆盖。

必须在加载 ROS2 Jazzy、工作空间和 Rerun 环境（NumPy 2）的终端运行::

    source /opt/ros/jazzy/setup.bash
    source ros2_ws/install/setup.bash
    source ros2_ws/.venv-rerun/bin/activate
    python convert_umi_mcap_to_rerun.py --input ros2_ws/dataset/umi_jingbao_binary \\
        --output ros2_ws/dataset/rerun/umi_jingbao_binary

退出码：0 全部成功；1 存在转换失败的记录；2 依赖缺失或输入无效。
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from analyze_collection_quality import (  # noqa: E402
    PACKAGE_SOURCE,
    QUATERNION_TOLERANCE,
    ROLE_EVENT,
    ROLE_GRIPPER,
    ROLE_IMAGE,
    ROLE_POSE,
    ROLE_STATUS,
    ROLE_TYPES,
    AnalysisError,
    discover_records,
    load_yaml_mapping,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ERROR = 2

# session.yaml 缺少 topic_map 时使用的默认话题，与采集节点默认配置一致。
DEFAULT_TOPICS = {
    ROLE_IMAGE: "/umi_camera/image_raw/compressed",
    ROLE_POSE: "/vive_tracker/pose",
    ROLE_STATUS: "/vive_tracker/status",
    ROLE_GRIPPER: "/gripper/state",
    ROLE_EVENT: "/fastumi/episode/events",
}
# 缺失时整条记录失败的角色；其余角色缺失只在报告中列出。
REQUIRED_ROLES = (ROLE_IMAGE, ROLE_POSE)
DEFAULT_BAG_URI = "raw/bag"
# raw/h264 解码后重新编码的 JPEG 质量。
JPEG_QUALITY = 90
# Tracker 坐标轴的显示长度，单位米。
AXIS_LENGTH_M = 0.10
TIMELINE_BAG = "bag_receive_time"
TIMELINE_HEADER = "header_time"
EVENT_NAMES = {1: "START", 2: "STOP", 3: "ABORT"}

ENVIRONMENT_HINT = (
    "请在已加载 ROS2 Jazzy、工作空间和 Rerun 环境的终端运行：\n"
    "  source /opt/ros/jazzy/setup.bash\n"
    "  source ros2_ws/install/setup.bash\n"
    "  source ros2_ws/.venv-rerun/bin/activate\n"
    "环境准备见 docs/Rerun采集数据可视化.md；只想查看参数时 --help 不需要 ROS。"
)


class ConversionError(Exception):
    """单条记录无法导出；只影响该记录。"""


class DependencyError(Exception):
    """缺少运行依赖；整批无法导出。"""


def load_runtime() -> SimpleNamespace:
    """延迟加载 Rerun、ROS 与仓库源码中的图像解码逻辑。

    不导入 ``cv_bridge``：它在 NumPy 2 环境中不可用，raw 图像直接按行跨度解析。

    Raises:
        DependencyError: 缺少任一依赖；附环境准备说明。
    """
    source = str(PACKAGE_SOURCE)
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)
    try:
        import cv2
        import numpy as np
        import rerun as rr
        import rerun.blueprint as rrb
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message

        importlib.import_module("fastumi_interfaces.msg")
        from fastumi_data.image_messages import packet_from_message
        from fastumi_data.image_modes import (
            MODE_COMPRESSED,
            MODE_H264,
            MODE_RAW,
            mode_for_message_type,
        )
        from fastumi_data.video_decoding import COLOR_BGR, create_decoder
    except ImportError as error:
        raise DependencyError(f"无法加载运行依赖：{error}\n" + ENVIRONMENT_HINT) from error
    return SimpleNamespace(
        COLOR_BGR=COLOR_BGR,
        MODE_COMPRESSED=MODE_COMPRESSED,
        MODE_H264=MODE_H264,
        MODE_RAW=MODE_RAW,
        create_decoder=create_decoder,
        cv2=cv2,
        deserialize_message=deserialize_message,
        get_message=get_message,
        mode_for_message_type=mode_for_message_type,
        np=np,
        packet_from_message=packet_from_message,
        rosbag2_py=rosbag2_py,
        rr=rr,
        rrb=rrb,
    )


def stamp_ns(stamp: Any) -> int:
    """``builtin_interfaces/Time`` 转 Unix 纳秒。"""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def resolve_record(record_dir: Path) -> Tuple[Path, Dict[str, str]]:
    """从 ``session.yaml`` 解析 bag 路径与各角色话题。

    Raises:
        ConversionError: session.yaml 不可读，或 bag 的 ``metadata.yaml`` 不存在。
    """
    try:
        session = load_yaml_mapping(record_dir / "session.yaml")
    except AnalysisError as error:
        raise ConversionError(str(error)) from error
    bag = record_dir / str(session.get("bag_uri") or DEFAULT_BAG_URI)
    if not (bag / "metadata.yaml").is_file():
        raise ConversionError(f"没有找到 bag: {bag}/metadata.yaml")
    topic_map = session.get("topic_map")
    topic_map = topic_map if isinstance(topic_map, dict) else {}
    topics = {
        role: str(topic_map.get(role) or default).strip()
        for role, default in DEFAULT_TOPICS.items()
    }
    return bag, topics


def blueprint(rt: SimpleNamespace) -> Any:
    """默认布局：左侧相机，右侧依次为轨迹、二值（filtered）与原始（raw）开度、标签距离和状态。"""
    rrb = rt.rrb
    return rrb.Blueprint(
        rrb.Horizontal(
            contents=[
                rrb.Vertical(
                    contents=[
                        rrb.Spatial2DView(name="UMI camera", origin="/camera/umi"),
                        rrb.TextLogView(name="Episode events", origin="/events"),
                    ],
                    row_shares=[4, 1],
                ),
                rrb.Vertical(
                    contents=[
                        rrb.Spatial3DView(name="Tracker (m)", origin="/world"),
                        # 二值化副本的 filtered 为 0/1，raw 保留连续原值；分开显示避免混淆。
                        rrb.TimeSeriesView(
                            name="Gripper openness (filtered)",
                            origin="/gripper/openness/filtered",
                        ),
                        rrb.TimeSeriesView(
                            name="Gripper openness (raw)", origin="/gripper/openness/raw"
                        ),
                        rrb.TimeSeriesView(
                            name="Marker distance (mm)", origin="/gripper/marker_distance_mm"
                        ),
                        rrb.TimeSeriesView(
                            name="Status",
                            origin="/",
                            contents=["/tracker/status/**", "/gripper/valid",
                                      "/gripper/detected_marker_count"],
                        ),
                    ],
                    row_shares=[3, 2, 2, 1, 2],
                ),
            ],
            column_shares=[3, 2],
        ),
        rrb.TimePanel(state="expanded"),
        collapse_panels=False,
    )


class EpisodeWriter:
    """把一条记录的消息按 bag 顺序写入 RecordingStream，并累计计数。"""

    def __init__(self, rt: SimpleNamespace, recording: Any, image_mode: str) -> None:
        self.rt = rt
        self.recording = recording
        self.image_mode = image_mode
        self.counts: Counter = Counter()
        self.invalid: Counter = Counter()
        self.positions: List[Any] = []
        self.frame_id: Optional[str] = None
        self.decoder = (
            rt.create_decoder(image_mode, rt.COLOR_BGR)
            if image_mode == rt.MODE_H264 else None
        )
        self._log_static()

    def _log_static(self) -> None:
        """一次性写入的样式与坐标约定。"""
        rr, log = self.rt.rr, self.recording.log
        log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        log("world/tracker", rr.TransformAxes3D(AXIS_LENGTH_M), static=True)
        for path, name in (
            ("gripper/openness/raw", "raw_openness"),
            ("gripper/openness/filtered", "filtered_openness"),
            ("gripper/marker_distance_mm", "marker distance (mm)"),
            ("gripper/valid", "gripper valid"),
            ("gripper/detected_marker_count", "detected markers"),
            ("tracker/status/pose_valid", "pose_valid"),
            ("tracker/status/device_connected", "device_connected"),
            ("tracker/status/tracking_state", "tracking_state (3=OK)"),
        ):
            log(path, rr.SeriesLines(names=name), static=True)

    def _time(self, bag_ns: int, header_ns: int) -> None:
        np, recording = self.rt.np, self.recording
        recording.set_time(TIMELINE_BAG, timestamp=np.datetime64(int(bag_ns), "ns"))
        recording.set_time(TIMELINE_HEADER, timestamp=np.datetime64(int(header_ns), "ns"))

    def _scalar(self, path: str, value: float) -> None:
        if math.isfinite(value):
            self.recording.log(path, self.rt.rr.Scalars(float(value)))

    def _log_jpeg(self, payload: bytes, media_type: str = "image/jpeg") -> None:
        self.recording.log(
            "camera/umi", self.rt.rr.EncodedImage(contents=payload, media_type=media_type)
        )

    def _log_bgr(self, bgr: Any) -> bool:
        cv2 = self.rt.cv2
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            self._log_jpeg(encoded.tobytes())
        return bool(ok)

    def image(self, message: Any, bag_ns: int) -> None:
        """按图像模式写入一帧；无法解析的帧计入 invalid。"""
        rt = self.rt
        if self.image_mode == rt.MODE_COMPRESSED:
            payload = bytes(message.data)
            fmt = str(message.format).lower()
            if "png" in fmt and payload.startswith(b"\x89PNG"):
                media_type = "image/png"
            elif payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9"):
                media_type = "image/jpeg"
            else:
                self.invalid[ROLE_IMAGE] += 1
                return
            self._time(bag_ns, stamp_ns(message.header.stamp))
            self._log_jpeg(payload, media_type)
            self.counts[ROLE_IMAGE] += 1
        elif self.image_mode == rt.MODE_RAW:
            bgr = self._raw_to_bgr(message)
            if bgr is None:
                self.invalid[ROLE_IMAGE] += 1
                return
            self._time(bag_ns, stamp_ns(message.header.stamp))
            if self._log_bgr(bgr):
                self.counts[ROLE_IMAGE] += 1
            else:
                self.invalid[ROLE_IMAGE] += 1
        else:
            packet = rt.packet_from_message(message, self.image_mode, bag_ns)
            self._log_decoded(self.decoder.feed(packet))

    def finish_images(self) -> None:
        """冲刷 H.264 解码器中尚未输出的帧。"""
        if self.decoder is not None:
            self._log_decoded(self.decoder.drain())

    def _log_decoded(self, frames: Sequence[Any]) -> None:
        for frame in frames:
            self._time(frame.bag_timestamp_ns, frame.stamp_ns)
            if self._log_bgr(frame.image):
                self.counts[ROLE_IMAGE] += 1
            else:
                self.invalid[ROLE_IMAGE] += 1

    def _raw_to_bgr(self, message: Any) -> Optional[Any]:
        """按 encoding 和行跨度把 ``sensor_msgs/Image`` 转为 BGR；不支持的编码返回 None。"""
        np, cv2 = self.rt.np, self.rt.cv2
        encoding = str(message.encoding).lower()
        channels = {"bgr8": 3, "rgb8": 3, "mono8": 1}.get(encoding)
        width, height, step = int(message.width), int(message.height), int(message.step)
        data = bytes(message.data)
        if not channels or width <= 0 or height <= 0 or step < width * channels \
                or len(data) < step * height:
            return None
        rows = np.frombuffer(data, dtype=np.uint8, count=step * height).reshape(height, step)
        pixels = rows[:, : width * channels].reshape(height, width, channels)
        if encoding == "rgb8":
            return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        if encoding == "mono8":
            return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(pixels)

    def pose(self, message: Any, bag_ns: int) -> None:
        """Tracker 位姿：有效样本写 Transform3D，并累计整条轨迹。"""
        np = self.rt.np
        p, q = message.pose.position, message.pose.orientation
        position = np.asarray([p.x, p.y, p.z], dtype=np.float64)
        quaternion = np.asarray([q.x, q.y, q.z, q.w], dtype=np.float64)
        if not (np.isfinite(position).all() and np.isfinite(quaternion).all()) \
                or abs(float(np.linalg.norm(quaternion)) - 1.0) > QUATERNION_TOLERANCE:
            self.invalid[ROLE_POSE] += 1
            return
        frame_id = str(message.header.frame_id)
        if self.frame_id is None:
            self.frame_id = frame_id
        elif frame_id != self.frame_id:
            raise ConversionError(
                f"Tracker 位姿坐标系在同一记录内变化：{self.frame_id!r} -> {frame_id!r}"
            )
        self._time(bag_ns, stamp_ns(message.header.stamp))
        self.recording.log(
            "world/tracker",
            self.rt.rr.Transform3D(translation=position, quaternion=quaternion),
        )
        self.positions.append(position)
        self.counts[ROLE_POSE] += 1

    def status(self, message: Any, bag_ns: int) -> None:
        """Tracker 连接与跟踪状态曲线。"""
        self._time(bag_ns, stamp_ns(message.header.stamp))
        self._scalar("tracker/status/pose_valid", float(bool(message.pose_valid)))
        self._scalar("tracker/status/device_connected", float(bool(message.device_connected)))
        self._scalar("tracker/status/tracking_state", float(message.tracking_state))
        self.counts[ROLE_STATUS] += 1

    def gripper(self, message: Any, bag_ns: int) -> None:
        """夹爪开度、标签距离与有效标志；NaN 不写入。"""
        self._time(bag_ns, stamp_ns(message.header.stamp))
        self._scalar("gripper/openness/raw", float(message.raw_openness))
        self._scalar("gripper/openness/filtered", float(message.filtered_openness))
        self._scalar("gripper/marker_distance_mm", float(message.marker_distance_mm))
        self._scalar("gripper/valid", float(bool(message.valid)))
        self._scalar("gripper/detected_marker_count", float(message.detected_marker_count))
        self.counts[ROLE_GRIPPER] += 1

    def event(self, message: Any, bag_ns: int) -> None:
        """episode 边界事件写为文本日志，不用于裁剪。"""
        rr = self.rt.rr
        kind = EVENT_NAMES.get(int(message.event_type), f"UNKNOWN({int(message.event_type)})")
        level = rr.TextLogLevel.INFO if kind in ("START", "STOP") else rr.TextLogLevel.WARN
        self._time(bag_ns, stamp_ns(message.header.stamp))
        self.recording.log(
            "events",
            rr.TextLog(f"{kind} episode {int(message.episode_index)} "
                       f"({message.task_name})", level=level),
        )
        self.counts[ROLE_EVENT] += 1

    def finish(self) -> None:
        """写入完整轨迹（静态）。"""
        self.finish_images()
        if self.positions:
            path = self.rt.np.stack(self.positions)
            self.recording.log("world/path", self.rt.rr.LineStrips3D([path]), static=True)


def convert_record(rt: SimpleNamespace, record_dir: Path, destination: Path) -> Dict[str, Any]:
    """导出一条记录到 ``destination``；先写同目录临时文件，成功后再改名。

    Raises:
        FileExistsError: 目标已存在。
        ConversionError: 缺少必需话题、类型不符或必需数据为空。
    """
    if destination.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖: {destination}")
    bag, topics = resolve_record(record_dir)
    reader = rt.rosbag2_py.SequentialReader()
    reader.open(
        rt.rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rt.rosbag2_py.ConverterOptions("", ""),
    )
    available = {item.name: item.type for item in reader.get_all_topics_and_types()}
    for role in REQUIRED_ROLES:
        if topics[role] not in available:
            raise ConversionError(f"bag 缺少必需话题 {topics[role]}（{role}）")
    image_mode = rt.mode_for_message_type(available[topics[ROLE_IMAGE]])
    if image_mode is None:
        raise ConversionError(
            f"不支持的图像类型 {available[topics[ROLE_IMAGE]]}（{topics[ROLE_IMAGE]}）"
        )
    missing: List[str] = []
    by_topic: Dict[str, Tuple[str, Any]] = {}
    for role, topic in topics.items():
        if topic not in available:
            missing.append(role)
            continue
        expected = ROLE_TYPES.get(role)
        if expected is not None and available[topic] != expected:
            raise ConversionError(f"{topic} 类型应为 {expected}，实际为 {available[topic]}")
        by_topic[topic] = (role, rt.get_message(available[topic]))
    reader.set_filter(rt.rosbag2_py.StorageFilter(topics=sorted(by_topic)))

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.stem}.{uuid.uuid4().hex}.rrd"
    try:
        with rt.rr.RecordingStream("fastumi_umi_episode") as recording:
            recording.save(stage, default_blueprint=blueprint(rt))
            writer = EpisodeWriter(rt, recording, image_mode)
            handlers = {
                ROLE_IMAGE: writer.image, ROLE_POSE: writer.pose, ROLE_STATUS: writer.status,
                ROLE_GRIPPER: writer.gripper, ROLE_EVENT: writer.event,
            }
            while reader.has_next():
                topic, serialized, bag_ns = reader.read_next()
                role, message_class = by_topic[topic]
                handlers[role](rt.deserialize_message(serialized, message_class), int(bag_ns))
            writer.finish()
        for role in REQUIRED_ROLES:
            if not writer.counts[role]:
                raise ConversionError(f"缺少有效的 {role} 样本")
        if destination.exists():
            raise FileExistsError(f"目标已存在，拒绝覆盖: {destination}")
        stage.rename(destination)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise
    skipped = writer.decoder.stats.skipped_before_keyframe if writer.decoder is not None else 0
    return {
        "output": str(destination),
        "image_mode": image_mode,
        "counts": {role: writer.counts[role] for role in topics},
        "invalid": dict(writer.invalid),
        "missing_roles": missing,
        "tracker_frame": writer.frame_id,
        "image_skipped_before_keyframe": skipped,
    }


def destination_for(source: Path, output: Path, record_id: str, single: bool) -> Path:
    """单条记录输出到 ``output/<名称>.rrd``；多条时保留相对 ``source`` 的子目录。"""
    relative = Path(record_id)
    if single:
        return output / f"{relative.name}.rrd"
    return output / relative.parent / f"{relative.name}.rrd"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """导出一条记录或目录下的全部记录，逐条报告并汇总。"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="MCAP 记录、FastUMI LeRobot 数据集或其上层目录（递归查找）")
    parser.add_argument("--output", required=True, type=Path,
                        help=".rrd 输出根目录（多条记录时保留输入下的子目录结构）")
    parser.add_argument("--input-format", choices=("auto", "mcap", "lerobot"), default="auto",
                        help="输入格式；auto 自动识别，混合输入需显式指定")
    parser.add_argument("--episodes", nargs="+", type=int,
                        help="仅导出指定 LeRobot episode 编号，例如 --episodes 0 2 5")
    args = parser.parse_args(argv)
    if args.episodes is not None and any(index < 0 for index in args.episodes):
        parser.error("--episodes 编号必须非负")
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    # 在加载 ROS 之前分派格式，使 LeRobot 转换可在纯离线环境运行。
    from lerobot_to_rerun import discover_datasets, run as run_lerobot

    try:
        datasets = discover_datasets(source, output) if args.input_format != "mcap" else []
        records = discover_records(source, skip=[output]) if args.input_format == "auto" else []
    except ValueError as error:
        print(error, file=sys.stderr)
        return EXIT_ERROR
    if args.input_format == "auto" and datasets and records:
        print("输入同时包含 MCAP 与 LeRobot，请指定 --input-format", file=sys.stderr)
        return EXIT_ERROR
    if args.input_format == "lerobot" or datasets:
        if not datasets:
            print(f"{source} 下没有找到 LeRobot 数据集", file=sys.stderr)
            return EXIT_ERROR
        return run_lerobot(source, output, datasets, args.episodes)
    if args.episodes is not None:
        print("--episodes 仅适用于 LeRobot 输入", file=sys.stderr)
        return EXIT_ERROR
    try:
        rt = load_runtime()
    except DependencyError as error:
        print(error, file=sys.stderr)
        return EXIT_ERROR
    try:
        entries = discover_records(source, skip=[output])
    except ValueError as error:
        print(error, file=sys.stderr)
        return EXIT_ERROR
    if not entries:
        print(f"{source} 下没有找到含 session.yaml 的记录", file=sys.stderr)
        return EXIT_ERROR
    single = len(entries) == 1 and entries[0].path == source
    failed = 0
    for entry in entries:
        destination = destination_for(source, output, entry.record_id, single)
        try:
            report = convert_record(rt, entry.path, destination)
        except Exception as error:  # noqa: BLE001 - 单条失败不影响其余记录。
            failed += 1
            print(f"{entry.record_id}: 失败: {error}")
            continue
        print(f"{entry.record_id}: 完成 -> {report['output']}，图像模式 {report['image_mode']}，"
              f"样本 {report['counts']}，Tracker 坐标系 {report['tracker_frame']!r}")
        if report["image_skipped_before_keyframe"]:
            print(f"{entry.record_id}: 关键帧前跳过 {report['image_skipped_before_keyframe']} 个包")
        if report["invalid"]:
            print(f"{entry.record_id}: 无效消息 {report['invalid']}")
        if report["missing_roles"]:
            print(f"{entry.record_id}: 缺少可选话题 {report['missing_roles']}")
    print(f"总计 {len(entries)} 条，成功 {len(entries) - failed}，失败 {failed}")
    return EXIT_FAILED if failed else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
