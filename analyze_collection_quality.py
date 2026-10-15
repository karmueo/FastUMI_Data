#!/usr/bin/env python3
"""只读分析 FastUMI 已保存的采集记录，输出质量报告（Markdown / JSON / CSV / 图表）。

输入可以是单条记录（含 ``session.yaml``）、任务目录或数据集根目录；递归发现记录时
跳过 ``.staging``、``.trash``、``_analysis`` 和目录符号链接。支持 raw、compressed、H.264
三种图像模式，不启动 ROS 节点，不连接硬件，不修改原始数据。

必须在已加载 ROS2 Jazzy、工作空间和 NumPy 1 环境的终端运行::

    python analyze_collection_quality.py --input ros2_ws/dataset/umi_jingbao

同步预检复用 ``ros2_ws/src/fastumi_data`` 源码中的解码器、配置解析、标定核对和
Link7 同步函数，因此结果与 ``convert_mcap`` 的选帧、插值和门限一致。ROS、解码、
绘图依赖在分析开始时才加载，``--help`` 不需要 ROS。

多条记录默认按记录并行分析（``--jobs``）：每条记录仍在单个进程内顺序处理，报告内容、
顺序和退出码与串行一致。

退出码：0 无质量拒绝、分析错误或未完成预检（允许警告）；1 存在质量拒绝；
2 参数、依赖、读取或报告生成错误，或同步预检因环境、配置、标定问题无法完成。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import multiprocessing
import os
import re
import shutil
import sys
import tempfile
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    import yaml
except ImportError:  # 仅影响 --help 之外的分析；load_runtime 会给出环境说明。
    np = None
    yaml = None


REPOSITORY_ROOT = Path(__file__).resolve().parent
PACKAGE_SOURCE = REPOSITORY_ROOT / "ros2_ws" / "src" / "fastumi_data"
DEFAULT_TRACKER_EXTRINSIC = REPOSITORY_ROOT / "config" / "calibration" / "T_cam_tracker.yaml"
DEFAULT_END_EXTRINSIC = REPOSITORY_ROOT / "config" / "calibration" / "T_cam_end.yaml"
DEFAULT_PROCESSING = PACKAGE_SOURCE / "config" / "processing.yaml"

EXIT_OK = 0
EXIT_REJECTED = 1
EXIT_ERROR = 2

# 递归发现记录时跳过的目录名。
SKIP_DIR_NAMES = (".staging", ".trash", "_analysis")
# Tracker 四元数范数相对 1 的容差，与采集节点的位姿有效性判断一致。
QUATERNION_TOLERANCE = 1.0e-3
# 单条记录最多保留的代表帧数、抽样模式下每个异常片段追加的帧数上限。
MAX_PREVIEWS = 12
SAMPLE_PERIOD_S = 1.0
SAMPLE_SEGMENT_FRAMES = 3
SAMPLE_EXTRA_LIMIT = 60
# 报告中列出的明细条数上限，完整数量另有计数。
DETAIL_LIMIT = 20
# --jobs 0 自动选择时的进程数上限，避免磁盘随机读和内存被占满。
MAX_AUTO_JOBS = 8

ROLE_IMAGE = "image"
ROLE_POSE = "tracker_pose"
ROLE_STATUS = "tracker_status"
ROLE_GRIPPER = "gripper_state"
ROLE_EVENT = "episode_event"
ROLES = (ROLE_IMAGE, ROLE_POSE, ROLE_STATUS, ROLE_GRIPPER, ROLE_EVENT)
ROLE_TYPES = {
    ROLE_POSE: "geometry_msgs/msg/PoseStamped",
    ROLE_STATUS: "fastumi_interfaces/msg/TrackerStatus",
    ROLE_GRIPPER: "fastumi_interfaces/msg/GripperState",
    ROLE_EVENT: "fastumi_interfaces/msg/EpisodeEvent",
}
TIMING_TYPE = "sensor_msgs/msg/TimeReference"
EVENT_START, EVENT_STOP, EVENT_ABORT = 1, 2, 3
# TrackerStatus.TRACKING_RUNNING_OK。
TRACKING_OK = 3

# 逐帧解码状态。
DECODE_NOT_CHECKED, DECODE_OK, DECODE_FAILED, DECODE_SKIPPED = 0, 1, 2, 3

STATUS_PASSED = "passed"
STATUS_REJECTED = "rejected"
STATUS_INCOMPLETE = "incomplete"
STATUS_FAILED = "failed"
STATUS_LABELS = {
    STATUS_PASSED: "通过",
    STATUS_REJECTED: "质量拒绝",
    STATUS_INCOMPLETE: "预检未完成",
    STATUS_FAILED: "分析失败",
}
SYNC_ACCEPTED = "accepted"
SYNC_REJECTED = "rejected"
SYNC_INCOMPLETE = "incomplete"
SYNC_BLOCKED = "blocked_by_data_defect"

ENVIRONMENT_HINT = (
    "请在已加载 ROS2 Jazzy、工作空间和 NumPy 1 环境的终端运行：\n"
    "  source /opt/ros/jazzy/setup.bash\n"
    "  source ros2_ws/.venv-numpy1/bin/activate\n"
    "  source ros2_ws/install/setup.bash\n"
    "详见 README.md 的“采集质量分析”一节；只想查看参数时 --help 不需要 ROS。"
)


class AnalysisError(Exception):
    """单条记录无法分析（读取或结构错误）；只影响该记录。"""


class DependencyError(Exception):
    """缺少运行依赖；整批无法分析。"""


class WorkerCrashedError(AnalysisError):
    """分析进程异常退出（例如解码库崩溃或内存不足）；只影响正在处理的记录。"""


def issue(code: str, message: str, **details: Any) -> Dict[str, Any]:
    """构造报告中的结构化问题条目。"""
    item: Dict[str, Any] = {"code": code, "message": message}
    item.update(details)
    return item


# ---------------------------------------------------------------- 严格 JSON
def to_jsonable(value: Any, path: str = "$", replaced: Optional[List[str]] = None) -> Any:
    """递归转换为严格 JSON 可序列化对象，非有限浮点写为 None 并记录路径。"""
    if replaced is None:
        replaced = []
    if isinstance(value, dict):
        return {str(key): to_jsonable(item, f"{path}.{key}", replaced) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item, f"{path}[{index}]", replaced) for index, item in enumerate(value)]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist(), path, replaced)
    if isinstance(value, np.generic):
        return to_jsonable(value.item(), path, replaced)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        replaced.append(path)
        return None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def dumps_strict(document: Any) -> Tuple[str, List[str]]:
    """返回严格 JSON 文本（不含 NaN/Infinity）和被替换为 null 的路径列表。"""
    replaced: List[str] = []
    clean = to_jsonable(document, "$", replaced)
    return json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False) + "\n", replaced


# ---------------------------------------------------------------- 统计函数
def describe(values: Any, reason: str = "没有可用样本") -> Dict[str, Any]:
    """返回 count/min/mean/p95/max；无有限样本时各统计量为 None 并附原因。"""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    report: Dict[str, Any] = {"count": int(finite.size)}
    if array.size != finite.size:
        report["nonfinite_count"] = int(array.size - finite.size)
    if finite.size == 0:
        report.update({"min": None, "mean": None, "p95": None, "max": None, "reason": reason})
        return report
    report.update(
        {
            "min": float(finite.min()),
            "mean": float(finite.mean()),
            "p95": float(np.percentile(finite, 95)),
            "max": float(finite.max()),
        }
    )
    return report


def stream_timing(stamps_ns: Any, bag_ns: Any = None) -> Dict[str, Any]:
    """统计一路消息的源时间：数量、重复/倒序、有效频率、间隔与接收延迟。

    重复与倒序按原始消息顺序检查；频率和间隔按排序去重后的时间计算。样本不足时
    相应统计量为 None 并在 ``reasons`` 说明。
    """
    stamps = np.asarray(stamps_ns, dtype=np.int64).reshape(-1)
    count = int(stamps.size)
    reasons: Dict[str, str] = {}
    deltas = np.diff(stamps) if count > 1 else np.zeros(0, dtype=np.int64)
    ordered = np.unique(stamps)
    report: Dict[str, Any] = {
        "count": count,
        "unique_count": int(ordered.size),
        "duplicate_count": int(np.count_nonzero(deltas == 0)),
        "reversed_count": int(np.count_nonzero(deltas < 0)),
    }
    if ordered.size >= 2:
        span_ns = int(ordered[-1] - ordered[0])
        report["span_s"] = span_ns / 1.0e9
        report["rate_hz"] = (ordered.size - 1) * 1.0e9 / span_ns
        gaps_ms = np.diff(ordered) / 1.0e6
        report["interval_ms"] = describe(gaps_ms)
        report["max_gap_ms"] = float(gaps_ms.max())
    else:
        report["span_s"] = 0.0 if count else None
        report["rate_hz"] = None
        report["interval_ms"] = describe([], "不同时间戳少于 2 个，无法计算间隔")
        report["max_gap_ms"] = None
        reasons["rate_hz"] = reasons["max_gap_ms"] = (
            "没有消息" if count == 0 else "不同时间戳少于 2 个，无法计算频率和间隔"
        )
    if bag_ns is not None:
        latency_ms = (np.asarray(bag_ns, dtype=np.int64).reshape(-1) - stamps) / 1.0e6
        report["latency_ms"] = describe(latency_ms)
        report["negative_latency_count"] = int(np.count_nonzero(latency_ms < 0))
    if reasons:
        report["reasons"] = reasons
    return report


def invalid_runs(
    times_ns: Any, valid: Any, origin_ns: Optional[int] = None
) -> List[Dict[str, Any]]:
    """按时间顺序返回连续无效样本区间。

    ``gap_s`` 为区间前后最近有效样本之间的时间跨度（边界处为 None），即插值需要
    跨过的空洞；``span_s`` 为无效样本自身的首末时间差。
    """
    times = np.asarray(times_ns, dtype=np.int64).reshape(-1)
    flags = np.asarray(valid, dtype=bool).reshape(-1)
    if times.size == 0:
        return []
    order = np.argsort(times, kind="stable")
    times, flags = times[order], flags[order]
    origin = int(times[0]) if origin_ns is None else int(origin_ns)
    padded = np.concatenate(([0], (~flags).astype(np.int8), [0]))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    runs = []
    for first, last in zip(starts.tolist(), ends.tolist()):
        gap = None
        if first > 0 and last + 1 < times.size:
            gap = float(times[last + 1] - times[first - 1]) / 1.0e9
        runs.append(
            {
                "start_index": first,
                "end_index": last,
                "samples": last - first + 1,
                "start_ns": int(times[first]),
                "end_ns": int(times[last]),
                "start_s": float(times[first] - origin) / 1.0e9,
                "end_s": float(times[last] - origin) / 1.0e9,
                "span_s": float(times[last] - times[first]) / 1.0e9,
                "gap_s": gap,
            }
        )
    return runs


def longest_valid_gap_s(times_ns: Any, valid: Any) -> Optional[float]:
    """有效样本之间的最大时间间隔（秒）；有效样本少于 2 个时为 None。"""
    times = np.asarray(times_ns, dtype=np.int64).reshape(-1)
    flags = np.asarray(valid, dtype=bool).reshape(-1)
    good = np.sort(times[flags])
    if good.size < 2:
        return None
    return float(np.diff(good).max()) / 1.0e9


def pose_invalid_mask(positions: Any, quaternions: Any) -> Any:
    """位姿任一分量非有限，或四元数范数偏离 1 超过容差时为 True。"""
    xyz = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    quat = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(quat).all(axis=1)
    with np.errstate(invalid="ignore"):
        norm_ok = np.abs(np.linalg.norm(quat, axis=1) - 1.0) <= QUATERNION_TOLERANCE
    return ~(finite & norm_ok)


def window_mask(stamps_ns: Any, low_ns: int, high_ns: int) -> Any:
    """时间位于闭区间 ``[low_ns, high_ns]`` 的样本掩码。"""
    stamps = np.asarray(stamps_ns, dtype=np.int64)
    return (stamps >= low_ns) & (stamps <= high_ns)


def episode_image_mask(
    stamps_ns: Any, bag_ns: Any, start_ns: int, stop_ns: int, post_roll_ns: int
) -> Any:
    """与转换一致的图像归属：Header 时间在 START/STOP 内，且 STOP 后滚动结束前到达。"""
    stamps = np.asarray(stamps_ns, dtype=np.int64)
    bag = np.asarray(bag_ns, dtype=np.int64)
    return (stamps >= start_ns) & (stamps <= stop_ns) & (bag <= stop_ns + post_roll_ns)


# ---------------------------------------------------------------- 事件状态机
@dataclass(frozen=True)
class EventRecord:
    """从 MCAP 读取的 episode 事件。"""

    bag_ns: int
    stamp_ns: int
    event_type: int
    session_id: str
    task_name: str
    episode_index: int


@dataclass
class EpisodeSpec:
    """由事件序列还原的一条 episode 及其事件缺陷。"""

    index: int
    start: Optional[EventRecord] = None
    stop: Optional[EventRecord] = None
    abort: Optional[EventRecord] = None
    defects: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def closed(self) -> bool:
        """START 与 STOP 配对且没有事件缺陷，可进入图像和同步分析。"""
        return self.start is not None and self.stop is not None and not self.defects


def parse_events(
    events: Sequence[EventRecord], session_id: Optional[str]
) -> Tuple[List[EpisodeSpec], List[Dict[str, Any]]]:
    """按 bag 顺序还原 episode；返回 episode 列表和记录级事件缺陷。

    规则与转换一致：活动期间再次 START、无 START 的 STOP/ABORT、结束仍未 STOP、
    START/STOP 的 session 或编号不一致都是缺陷；另检查 session 与记录身份一致、
    编号重复和 STOP 早于 START。
    """
    specs: List[EpisodeSpec] = []
    record_defects: List[Dict[str, Any]] = []
    active: Optional[EpisodeSpec] = None
    seen_indices: Counter = Counter()

    def check_session(spec: EpisodeSpec, event: EventRecord) -> None:
        if session_id and event.session_id != session_id:
            spec.defects.append(
                issue(
                    "EVENT_SESSION_MISMATCH",
                    f"事件 session_id={event.session_id!r} 与记录 session.yaml 的 {session_id!r} 不一致",
                )
            )

    for event in events:
        if event.event_type not in (EVENT_START, EVENT_STOP, EVENT_ABORT):
            record_defects.append(
                issue("EVENT_TYPE_UNKNOWN", f"未知的 episode 事件类型 {event.event_type}")
            )
            continue
        if event.event_type == EVENT_START:
            if active is not None:
                active.defects.append(
                    issue("EVENT_SEQUENCE", "在活动 episode 内再次收到 START")
                )
            active = EpisodeSpec(index=event.episode_index, start=event)
            seen_indices[event.episode_index] += 1
            if seen_indices[event.episode_index] > 1:
                active.defects.append(
                    issue("EVENT_DUPLICATE_INDEX", f"episode 编号 {event.episode_index} 重复")
                )
            check_session(active, event)
            specs.append(active)
        elif active is None:
            orphan = EpisodeSpec(index=event.episode_index)
            if event.event_type == EVENT_STOP:
                orphan.stop = event
                orphan.defects.append(issue("EVENT_SEQUENCE", "收到 STOP，但此前没有 START"))
            else:
                orphan.abort = event
                orphan.defects.append(issue("EVENT_SEQUENCE", "收到 ABORT，但此前没有 START"))
            check_session(orphan, event)
            specs.append(orphan)
        elif event.event_type == EVENT_STOP:
            active.stop = event
            start = active.start
            if event.episode_index != start.episode_index or event.session_id != start.session_id:
                active.defects.append(
                    issue("EVENT_MISMATCH", "START 与 STOP 的 session 或 episode 编号不一致")
                )
            if event.stamp_ns <= start.stamp_ns:
                active.defects.append(issue("EVENT_ORDER", "STOP 的时间不晚于 START"))
            check_session(active, event)
            active = None
        else:
            active.abort = event
            active.defects.append(issue("ABORTED", "操作员主动放弃 episode"))
            active = None
    if active is not None:
        active.defects.append(issue("EVENT_SEQUENCE", "MCAP 结束时 episode 尚未 STOP"))
    return specs, record_defects


# ---------------------------------------------------------------- 抽样与代表帧
def sample_ordinals(stamps_ns: Any, period_s: float = SAMPLE_PERIOD_S) -> List[int]:
    """抽样模式的基础检查帧：按消息顺序，每间隔 ``period_s`` 秒（Header 时间）取一帧。"""
    stamps = np.asarray(stamps_ns, dtype=np.int64).reshape(-1)
    chosen: List[int] = []
    last: Optional[int] = None
    step = int(round(period_s * 1.0e9))
    for ordinal, stamp in enumerate(stamps.tolist()):
        if last is None or stamp >= last + step:
            chosen.append(ordinal)
            last = stamp
    return chosen


def segment_neighbors(first: int, last: int, total: int, count: int = SAMPLE_SEGMENT_FRAMES) -> List[int]:
    """异常片段 ``[first, last]`` 的前一帧、中间帧和后一帧（越界裁剪、去重）。"""
    candidates = [first - 1, (first + last) // 2, last + 1][:count]
    return sorted({min(max(item, 0), total - 1) for item in candidates}) if total else []


def choose_preview_ordinals(
    total: int, segments: Sequence[Tuple[int, int]], limit: int = MAX_PREVIEWS
) -> List[int]:
    """选择代表帧：先按严重程度覆盖失效段及其前后，再用均匀采样补足，最多 ``limit`` 张。

    Args:
        total: 图像总数。
        segments: 按严重程度从高到低排列的 ``(首帧序号, 末帧序号)``。
    """
    if total <= 0 or limit <= 0:
        return []
    chosen: List[int] = []
    for first, last in segments:
        for ordinal in (first - 1, (first + last) // 2, last + 1):
            ordinal = min(max(ordinal, 0), total - 1)
            if ordinal not in chosen and len(chosen) < limit:
                chosen.append(ordinal)
    remaining = limit - len(chosen)
    if remaining > 0:
        for ordinal in np.linspace(0, total - 1, num=min(remaining, total)).round().astype(int).tolist():
            if ordinal not in chosen and len(chosen) < limit:
                chosen.append(ordinal)
        # 均匀点被占用时，从头补足剩余名额。
        for ordinal in range(total):
            if len(chosen) >= limit:
                break
            if ordinal not in chosen:
                chosen.append(ordinal)
    return sorted(chosen)


def ordinals_between(stamps_ns: Any, ok: Any, start_ns: int, end_ns: int) -> Tuple[int, int]:
    """时间闭区间内图像序号的最小和最大值；没有图像时返回 ``(-1, -1)``。"""
    hits = np.flatnonzero(window_mask(stamps_ns, start_ns, end_ns) & np.asarray(ok, dtype=bool))
    if hits.size == 0:
        return -1, -1
    return int(hits[0]), int(hits[-1])


# ---------------------------------------------------------------- 退出码
def exit_code_for(records: Sequence[Dict[str, Any]]) -> int:
    """按最高严重程度返回退出码：分析失败/未完成预检为 2，质量拒绝为 1，其余为 0。"""
    if any(r.get("failed") or r.get("has_incomplete") for r in records):
        return EXIT_ERROR
    if any(r.get("has_rejection") for r in records):
        return EXIT_REJECTED
    return EXIT_OK


def record_status(failed: bool, has_rejection: bool, has_incomplete: bool) -> str:
    """记录状态：分析失败 > 质量拒绝 > 预检未完成 > 通过。"""
    if failed:
        return STATUS_FAILED
    if has_rejection:
        return STATUS_REJECTED
    if has_incomplete:
        return STATUS_INCOMPLETE
    return STATUS_PASSED


# ---------------------------------------------------------------- 记录发现
@dataclass(frozen=True)
class RecordEntry:
    """一条待分析的记录目录及其稳定标识。"""

    record_id: str
    path: Path


def discover_records(input_path: Path, skip: Sequence[Path] = ()) -> List[RecordEntry]:
    """发现输入下的全部已保存记录（含 ``session.yaml`` 的目录），按路径稳定排序。

    跳过 ``.staging``、``.trash``、``_analysis``、``skip`` 中的目录和目录符号链接。
    """
    root = Path(input_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"输入 {root} 不是目录")
    if (root / "session.yaml").is_file():
        return [RecordEntry(root.name, root)]
    skipped = {Path(item).resolve() for item in skip}
    found: List[Path] = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        current = Path(directory)
        dirs[:] = sorted(
            name for name in dirs
            if name not in SKIP_DIR_NAMES
            and not (current / name).is_symlink()
            and (current / name).resolve() not in skipped
        )
        if "session.yaml" in files:
            found.append(current)
            dirs.clear()
    found.sort()
    return [RecordEntry(item.relative_to(root).as_posix(), item) for item in found]


def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    """读取 YAML 映射；不可读或不是映射时抛出 AnalysisError。"""
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise AnalysisError(f"无法读取 {path}: {error}") from error
    if not isinstance(document, dict):
        raise AnalysisError(f"{path} 必须是 YAML 映射")
    return document


def sha256_file(path: Path) -> str:
    """文件原始字节的 SHA-256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record_sort_key(path: Path) -> float:
    """用于判定“最新记录”的时间：保存时间，缺失时用创建时间或文件修改时间。"""
    try:
        session = yaml.safe_load((path / "session.yaml").read_text(encoding="utf-8")) or {}
        for key in ("saved_at_unix", "created_at_unix"):
            if isinstance(session.get(key), (int, float)):
                return float(session[key])
    except (OSError, yaml.YAMLError, AttributeError):
        pass
    try:
        return (path / "session.yaml").stat().st_mtime
    except OSError:
        return 0.0


# ---------------------------------------------------------------- 运行时依赖
def load_runtime() -> SimpleNamespace:
    """延迟加载 ROS、rosbag2、解码器和仓库源码中的共享逻辑。

    Raises:
        DependencyError: 缺少 NumPy/PyYAML、ROS 或仓库源码；附环境准备说明。
    """
    if np is None or yaml is None:
        raise DependencyError("缺少 NumPy 或 PyYAML。\n" + ENVIRONMENT_HINT)
    if not (PACKAGE_SOURCE / "fastumi_data").is_dir():
        raise DependencyError(f"找不到共享逻辑源码 {PACKAGE_SOURCE}。")
    source = str(PACKAGE_SOURCE)
    if source in sys.path:
        sys.path.remove(source)
    sys.path.insert(0, source)
    try:
        for module in ("cv2", "fastumi_interfaces.msg", "rosbag2_py"):
            importlib.import_module(module)
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        import fastumi_data
        from fastumi_data import frame_selection
        from fastumi_data.bag_images import BagImageDecoder, check_bag_image_topic
        from fastumi_data.cdr_header import read_header_stamp_ns
        from fastumi_data.image_modes import MODE_H264, mode_for_message_type
        from fastumi_data.link7_calibration import (
            CalibrationError,
            check_timestamp_source,
            load_link7_calibration,
        )
        from fastumi_data.link7_config import (
            ConfigError,
            resolve_processing,
            resolve_source_fps,
            validate_source_fps,
        )
        from fastumi_data.link7_record import POST_ROLL_NS, PRE_ROLL_NS, open_reader
        from fastumi_data.link7_sync import synchronize_link7_episode
        from fastumi_data.models import GripperSample, PoseSample, TrackerStatusSample
        from fastumi_data.video_decoding import COLOR_BGR
    except ImportError as error:
        raise DependencyError(f"无法加载运行依赖：{error}\n" + ENVIRONMENT_HINT) from error
    return SimpleNamespace(
        BagImageDecoder=BagImageDecoder,
        CalibrationError=CalibrationError,
        COLOR_BGR=COLOR_BGR,
        ConfigError=ConfigError,
        GripperSample=GripperSample,
        MODE_H264=MODE_H264,
        POST_ROLL_NS=POST_ROLL_NS,
        PRE_ROLL_NS=PRE_ROLL_NS,
        PoseSample=PoseSample,
        TrackerStatusSample=TrackerStatusSample,
        check_bag_image_topic=check_bag_image_topic,
        check_timestamp_source=check_timestamp_source,
        deserialize_message=deserialize_message,
        frame_selection=frame_selection,
        get_message=get_message,
        load_link7_calibration=load_link7_calibration,
        mode_for_message_type=mode_for_message_type,
        open_reader=open_reader,
        read_header_stamp_ns=read_header_stamp_ns,
        resolve_processing=resolve_processing,
        resolve_source_fps=resolve_source_fps,
        shared_source=str(Path(fastumi_data.__file__).resolve().parent),
        synchronize_link7_episode=synchronize_link7_episode,
        validate_source_fps=validate_source_fps,
    )


# ---------------------------------------------------------------- 读取 bag
class ScanResult:
    """一遍扫描得到的轻量样本；图像只保留时间和序号，不保留像素。"""

    def __init__(self) -> None:
        self.read_counts: Counter = Counter()
        self.corrupt: Counter = Counter()
        self.corrupt_examples: List[str] = []
        self.image_stamps: List[int] = []
        self.image_bag: List[int] = []
        self.image_ok: List[bool] = []
        self.pose_rows: List[Tuple[int, int, float, float, float, float, float, float, float]] = []
        self.status_rows: List[Tuple[int, int, bool, bool, int]] = []
        self.gripper_rows: List[Tuple[int, int, float, float, int, bool]] = []
        self.serials: Counter = Counter()
        self.events: List[EventRecord] = []
        self.timing_rows: List[Tuple[int, int, int, int]] = []

    def finalize(self) -> None:
        """把列表转换为 NumPy 数组，便于向量化统计。"""
        self.image_stamps = np.asarray(self.image_stamps, dtype=np.int64)
        self.image_bag = np.asarray(self.image_bag, dtype=np.int64)
        self.image_ok = np.asarray(self.image_ok, dtype=bool)
        pose = np.asarray(self.pose_rows, dtype=np.float64).reshape(-1, 9)
        self.pose_stamps = pose[:, 0].astype(np.int64)
        self.pose_bag = pose[:, 1].astype(np.int64)
        self.pose_xyz = pose[:, 2:5]
        self.pose_quat = pose[:, 5:9]
        status = self.status_rows
        self.status_stamps = np.asarray([row[0] for row in status], dtype=np.int64)
        self.status_bag = np.asarray([row[1] for row in status], dtype=np.int64)
        self.status_connected = np.asarray([row[2] for row in status], dtype=bool)
        self.status_pose_valid = np.asarray([row[3] for row in status], dtype=bool)
        self.status_state = np.asarray([row[4] for row in status], dtype=np.int64)
        gripper = self.gripper_rows
        self.gripper_stamps = np.asarray([row[0] for row in gripper], dtype=np.int64)
        self.gripper_bag = np.asarray([row[1] for row in gripper], dtype=np.int64)
        self.gripper_raw = np.asarray([row[2] for row in gripper], dtype=np.float64)
        self.gripper_filtered = np.asarray([row[3] for row in gripper], dtype=np.float64)
        self.gripper_markers = np.asarray([row[4] for row in gripper], dtype=np.int64)
        self.gripper_valid = np.asarray([row[5] for row in gripper], dtype=bool)

    @property
    def status_ok(self) -> Any:
        """Tracker 状态可用于 6DoF 训练：已连接、位姿有效且为 RUNNING_OK。"""
        return self.status_connected & self.status_pose_valid & (self.status_state == TRACKING_OK)


def stamp_of(message: Any) -> int:
    """Header 时间转 Unix 纳秒。"""
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def parse_time_reference(source: str) -> Tuple[int, int, int]:
    """从 ``TimeReference.source`` 读取 ``(gen, offset_ns, seq)``，缺失为 0。"""
    values = {"gen": 0, "offset_ns": 0, "seq": 0}
    for item in str(source).split(";"):
        key, _, value = item.partition("=")
        key = key.strip()
        if key in values:
            try:
                values[key] = int(value)
            except ValueError:
                pass
    return values["gen"], values["offset_ns"], values["seq"]


def scan_bag(
    rt: SimpleNamespace,
    bag_uri: Path,
    topic_of: Dict[str, str],
    types: Dict[str, str],
    timing_topic: Optional[str],
) -> ScanResult:
    """单遍读取相关话题；只解析所需字段，无法解析的消息计入 ``corrupt``。"""
    scan = ScanResult()
    usable = readable_roles(topic_of, types)
    roles = {topic: role for role, topic in topic_of.items() if role in usable}
    wanted = set(roles)
    if timing_topic:
        wanted.add(timing_topic)
    reader = rt.open_reader(bag_uri, sorted(wanted))
    classes = {
        topic: rt.get_message(types[topic])
        for topic in wanted
        if topic != topic_of.get(ROLE_IMAGE)
    }
    image_topic = topic_of.get(ROLE_IMAGE)
    while reader.has_next():
        topic, data, bag_ns = reader.read_next()
        scan.read_counts[topic] += 1
        if topic == image_topic and topic in roles:
            stamp = rt.read_header_stamp_ns(bytes(data))
            scan.image_stamps.append(0 if stamp is None else stamp)
            scan.image_bag.append(int(bag_ns))
            scan.image_ok.append(stamp is not None)
            if stamp is None:
                scan.corrupt[topic] += 1
            continue
        try:
            message = rt.deserialize_message(bytes(data), classes[topic])
            if topic == timing_topic:
                gen, offset, seq = parse_time_reference(message.source)
                scan.timing_rows.append((stamp_of(message), gen, offset, seq))
                continue
            role = roles[topic]
            stamp = stamp_of(message)
            if role == ROLE_POSE:
                pose = message.pose
                scan.pose_rows.append(
                    (stamp, int(bag_ns), pose.position.x, pose.position.y, pose.position.z,
                     pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
                )
            elif role == ROLE_STATUS:
                scan.serials[str(message.serial_number).strip()] += 1
                scan.status_rows.append(
                    (stamp, int(bag_ns), bool(message.device_connected),
                     bool(message.pose_valid), int(message.tracking_state))
                )
            elif role == ROLE_GRIPPER:
                scan.gripper_rows.append(
                    (stamp, int(bag_ns), float(message.raw_openness),
                     float(message.filtered_openness), int(message.detected_marker_count),
                     bool(message.valid))
                )
            elif role == ROLE_EVENT:
                scan.events.append(
                    EventRecord(int(bag_ns), stamp, int(message.event_type),
                                str(message.session_id), str(message.task_name),
                                int(message.episode_index))
                )
        except Exception as error:  # 单条消息损坏不应中断整条记录。
            scan.corrupt[topic] += 1
            if len(scan.corrupt_examples) < DETAIL_LIMIT:
                scan.corrupt_examples.append(f"{topic}: {type(error).__name__}: {error}")
    scan.finalize()
    return scan


def readable_roles(topic_of: Dict[str, str], types: Dict[str, str]) -> List[str]:
    """已确认存在且类型匹配、可以解析的角色。"""
    usable = []
    for role, topic in topic_of.items():
        if topic not in types:
            continue
        if role != ROLE_IMAGE and types[topic] != ROLE_TYPES[role]:
            continue
        usable.append(role)
    return usable


@dataclass
class DecodeOutcome:
    """图像解码结果：逐消息状态、尺寸集合、失败明细和代表帧。"""

    status: Any
    shapes: Counter
    errors: Dict[int, str]
    previews: Dict[int, Any]
    stats: Dict[str, int]
    drain_error: str = ""


def decode_images(
    rt: SimpleNamespace,
    bag_uri: Path,
    image_topic: str,
    mode: str,
    total: int,
    wanted: Optional[set] = None,
    keep: Optional[set] = None,
) -> DecodeOutcome:
    """按 bag 顺序解码图像并逐帧释放，只保留状态、尺寸和需要的代表帧。

    ``wanted`` 为 None 表示全部解码。H.264 必须顺序喂入所有包才能维持参考帧，
    因此始终解码全部包；解码失败后新建解码器，等待下一个关键帧。
    """
    status = np.full(total, DECODE_NOT_CHECKED, dtype=np.int8)
    shapes: Counter = Counter()
    errors: Dict[int, str] = {}
    previews: Dict[int, Any] = {}
    totals = Counter()
    is_video = mode == rt.MODE_H264
    reader = rt.open_reader(bag_uri, [image_topic])
    decoder = rt.BagImageDecoder(mode, rt.COLOR_BGR)

    def absorb(old) -> None:
        for name in ("packets", "frames", "keyframes", "skipped_before_keyframe"):
            totals[name] += int(getattr(old.stats, name, 0))

    ordinal = -1
    while reader.has_next():
        topic, data, bag_ns = reader.read_next()
        if topic != image_topic:
            continue
        ordinal += 1
        if ordinal >= total:
            break
        if not is_video and wanted is not None and ordinal not in wanted:
            continue
        skipped_before = decoder.stats.skipped_before_keyframe
        try:
            frames = decoder.feed(bytes(data), int(bag_ns))
        except Exception as error:  # 解码库异常类型不固定，统一记为该帧失败。
            status[ordinal] = DECODE_FAILED
            errors[ordinal] = f"{type(error).__name__}: {error}"
            if is_video:
                absorb(decoder)
                decoder = rt.BagImageDecoder(mode, rt.COLOR_BGR)
            continue
        if not frames:
            if is_video and decoder.stats.skipped_before_keyframe != skipped_before:
                status[ordinal] = DECODE_SKIPPED
            else:
                status[ordinal] = DECODE_FAILED
                errors[ordinal] = "解码器没有输出图像帧"
            continue
        status[ordinal] = DECODE_OK
        frame = frames[0]
        shapes[tuple(int(v) for v in frame.image.shape)] += 1
        if keep and ordinal in keep:
            previews[ordinal] = frame.image.copy()
    drain_error = ""
    try:
        decoder.drain()
    except Exception as error:
        drain_error = f"{type(error).__name__}: {error}"
    absorb(decoder)
    return DecodeOutcome(status, shapes, errors, previews, dict(totals), drain_error)


# ---------------------------------------------------------------- 单条记录分析
@dataclass
class Context:
    """整批分析共享的参数和已加载资源。"""

    rt: SimpleNamespace
    args: argparse.Namespace
    explicit_config: Optional[Path]
    calibration: Any
    calibration_problem: Optional[Dict[str, Any]]
    latest_id: str


@dataclass
class RecordResult:
    """一条记录的分析结果；``scan`` 和图像时间仅供绘图，序列化时丢弃。"""

    report: Dict[str, Any]
    scan: Optional[ScanResult] = None
    abnormal_segments: List[Tuple[int, int]] = field(default_factory=list)


def resolve_topics(processing: Any, session: Dict[str, Any]) -> Dict[str, str]:
    """优先使用生效配置的话题映射，配置无效时退回 session.yaml 的 topic_map。"""
    source = processing.topics if processing is not None else session.get("topic_map")
    if not isinstance(source, dict) or any(not str(source.get(role) or "").strip() for role in ROLES):
        raise AnalysisError("无法确定话题映射：生效配置和 session.yaml 的 topic_map 都不完整")
    return {role: str(source[role]).strip() for role in ROLES}


def find_timing_topic(types: Dict[str, str], image_topic: str) -> Optional[str]:
    """相机时间诊断话题：TimeReference 类型，优先与图像话题同一顶层命名空间。"""
    candidates = sorted(name for name, kind in types.items() if kind == TIMING_TYPE)
    namespace = image_topic.split("/")[1] if image_topic.count("/") > 1 else ""
    for name in candidates:
        if namespace and name.split("/")[1:2] == [namespace]:
            return name
    return candidates[0] if candidates else None


def check_snapshots(record_dir: Path, session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """核对 session.yaml 声明的快照存在且摘要一致；返回警告。"""
    warnings = []
    entries = []
    processing = session.get("processing_config")
    if isinstance(processing, dict):
        entries.append(processing)
    additional = session.get("additional_snapshots")
    if isinstance(additional, list):
        entries.extend(item for item in additional if isinstance(item, dict))
    for entry in entries:
        relative = str(entry.get("path") or "")
        expected = str(entry.get("sha256") or "").lower()
        if not relative:
            continue
        target = record_dir / relative
        if not target.is_file():
            warnings.append(issue("SNAPSHOT_MISSING", f"快照 {relative} 不存在", path=relative))
        elif expected and sha256_file(target) != expected:
            warnings.append(
                issue("SNAPSHOT_DIGEST_MISMATCH", f"快照 {relative} 的 SHA-256 与 session.yaml 记录不一致",
                      path=relative)
            )
    missing = session.get("missing_snapshots")
    if missing:
        warnings.append(
            issue("SNAPSHOTS_DECLARED_MISSING", f"采集时已声明缺少快照: {missing}", missing=missing)
        )
    return warnings


def read_bag_metadata(bag_uri: Path) -> Optional[Dict[str, Any]]:
    """读取 rosbag2 ``metadata.yaml`` 的顶层总数、逐话题数量和各分包文件数量。"""
    try:
        document = yaml.safe_load((Path(bag_uri) / "metadata.yaml").read_text(encoding="utf-8"))
        info = document["rosbag2_bagfile_information"]
        return {
            "message_count": int(info.get("message_count", 0)),
            "topics": {
                item["topic_metadata"]["name"]: int(item["message_count"])
                for item in info.get("topics_with_message_count", [])
            },
            "files": [int(item.get("message_count", 0)) for item in info.get("files", [])],
        }
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValueError, AttributeError):
        return None


def check_counts(
    session: Dict[str, Any], topic_of: Dict[str, str], scan: ScanResult,
    bag_meta: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """比较实际读取数量、bag 元数据（总数、逐话题、分包文件合计）和 session.yaml 附带计数。"""
    bag_counts = bag_meta["topics"] if bag_meta else {}
    declared = session.get("message_counts") if isinstance(session.get("message_counts"), dict) else {}
    table: Dict[str, Any] = {}
    warnings = []
    for role in ROLES:
        topic = topic_of[role]
        actual = int(scan.read_counts.get(topic, 0))
        row = {"topic": topic, "read": actual, "bag_metadata": bag_counts.get(topic),
               "session": declared.get(role)}
        table[role] = row
        if row["bag_metadata"] is not None and row["bag_metadata"] != actual:
            warnings.append(
                issue("BAG_COUNT_MISMATCH",
                      f"{role}: bag 元数据记录 {row['bag_metadata']} 条，实际读到 {actual} 条", role=role)
            )
        if isinstance(row["session"], (int, float)) and not isinstance(row["session"], bool):
            if int(row["session"]) != actual:
                ratio = f"（约 {row['session'] / actual:.2f} 倍）" if actual else ""
                warnings.append(
                    issue("SESSION_COUNT_MISMATCH",
                          f"{role}: session.yaml 附带计数 {row['session']} 与实际 {actual} 不一致{ratio}",
                          role=role, declared=row["session"], actual=actual)
                )
    if bag_meta:
        totals = {
            "顶层 message_count": bag_meta["message_count"],
            "逐话题合计": sum(bag_meta["topics"].values()),
            "分包文件合计": sum(bag_meta["files"]),
        }
        if len(set(totals.values())) > 1:
            base = bag_meta["message_count"]
            ratio = f"，分包文件合计为顶层的 {totals['分包文件合计'] / base:.2f} 倍" if base else ""
            warnings.append(
                issue("BAG_METADATA_INCONSISTENT",
                      "metadata.yaml 内部计数不一致（" + "，".join(f"{k} {v}" for k, v in totals.items()) + "）" + ratio,
                      totals=totals)
            )
        table["bag_metadata_totals"] = {
            "message_count": bag_meta["message_count"],
            "topic_sum": sum(bag_meta["topics"].values()),
            "file_sum": sum(bag_meta["files"]),
            "files": bag_meta["files"],
        }
    return table, warnings


def gripper_report(scan: ScanResult, mask: Any, origin_ns: int, support_mask: Any) -> Dict[str, Any]:
    """夹爪状态：有效率、失效区间、可插值的最大有效间隔和数值异常。"""
    stamps = scan.gripper_stamps[mask]
    filtered = scan.gripper_filtered[mask]
    valid = scan.gripper_valid[mask]
    usable = valid & np.isfinite(filtered)
    runs = invalid_runs(stamps, usable, origin_ns)
    total = int(stamps.size)
    report = {
        "samples": total,
        "valid_count": int(np.count_nonzero(usable)),
        "invalid_count": int(total - np.count_nonzero(usable)),
        "valid_ratio": (float(np.count_nonzero(usable)) / total) if total else None,
        "valid_flag_but_nonfinite": int(np.count_nonzero(valid & ~np.isfinite(filtered))),
        "outside_unit_range": int(np.count_nonzero(usable & ((filtered < -1e-6) | (filtered > 1 + 1e-6)))),
        "marker_count_histogram": {
            str(key): int(value) for key, value in sorted(Counter(scan.gripper_markers[mask].tolist()).items())
        },
        "invalid_run_count": len(runs),
        "invalid_runs": runs[:DETAIL_LIMIT],
        "max_invalid_gap_s": max((run["gap_s"] for run in runs if run["gap_s"] is not None), default=None),
        "max_valid_gap_s": longest_valid_gap_s(
            scan.gripper_stamps[support_mask],
            scan.gripper_valid[support_mask] & np.isfinite(scan.gripper_filtered[support_mask]),
        ),
    }
    if total == 0:
        report["reasons"] = {"valid_ratio": "窗口内没有夹爪样本"}
    return report


def status_report(scan: ScanResult, mask: Any, origin_ns: int) -> Dict[str, Any]:
    """Tracker 状态：状态直方图和跟踪异常区间。"""
    stamps = scan.status_stamps[mask]
    ok = scan.status_ok[mask]
    runs = invalid_runs(stamps, ok, origin_ns)
    return {
        "samples": int(stamps.size),
        "ok_count": int(np.count_nonzero(ok)),
        "bad_count": int(stamps.size - np.count_nonzero(ok)),
        "tracking_state_histogram": {
            str(key): int(value) for key, value in sorted(Counter(scan.status_state[mask].tolist()).items())
        },
        "disconnected_count": int(np.count_nonzero(~scan.status_connected[mask])),
        "pose_invalid_flag_count": int(np.count_nonzero(~scan.status_pose_valid[mask])),
        "bad_run_count": len(runs),
        "bad_runs": runs[:DETAIL_LIMIT],
    }


def pose_report(scan: ScanResult, mask: Any) -> Dict[str, Any]:
    """Tracker 位姿：时间统计、数值有效性和四元数范数误差。"""
    report = stream_timing(scan.pose_stamps[mask], scan.pose_bag[mask])
    xyz, quat = scan.pose_xyz[mask], scan.pose_quat[mask]
    bad = pose_invalid_mask(xyz, quat)
    with np.errstate(invalid="ignore"):
        norms = np.linalg.norm(quat, axis=1)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(quat).all(axis=1)
    report.update(
        {
            "invalid_count": int(np.count_nonzero(bad)),
            "nonfinite_count": int(np.count_nonzero(~finite)),
            "max_quaternion_norm_error": (
                float(np.abs(norms[finite] - 1.0).max()) if finite.any() else None
            ),
        }
    )
    if not finite.any():
        report.setdefault("reasons", {})["max_quaternion_norm_error"] = "没有有限位姿样本"
    return report


def integrity_report(
    decode: Optional[DecodeOutcome], mask: Any, decode_mode: str, label: str
) -> Dict[str, Any]:
    """窗口内的图像解码完整性与覆盖率；抽样且未全覆盖时标注“仅抽样验证”。"""
    total = int(np.count_nonzero(mask))
    if decode is None:
        return {"scope": label, "messages": total, "checked": 0, "coverage": 0.0,
                "verdict": "未解码", "failed": 0, "skipped_before_keyframe": 0}
    status = decode.status[: mask.size][mask]
    checked = int(np.count_nonzero(status != DECODE_NOT_CHECKED))
    coverage = (checked / total) if total else None
    full = total > 0 and checked == total
    return {
        "scope": label,
        "messages": total,
        "checked": checked,
        "decoded_ok": int(np.count_nonzero(status == DECODE_OK)),
        "failed": int(np.count_nonzero(status == DECODE_FAILED)),
        "skipped_before_keyframe": int(np.count_nonzero(status == DECODE_SKIPPED)),
        "coverage": coverage,
        "verdict": ("全量验证" if full or total == 0 else "仅抽样验证") if decode_mode == "sample"
        else "全量验证",
    }


def analyze_record(ctx: Context, entry: RecordEntry) -> RecordResult:
    """分析一条记录：基础检查、逐 episode 指标和同步预检。

    Raises:
        AnalysisError: session/bag 无法读取或话题映射不可确定。
    """
    rt, args = ctx.rt, ctx.args
    record_dir = entry.path
    session = load_yaml_mapping(record_dir / "session.yaml")
    rejections: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    problems: List[Dict[str, Any]] = []
    notes: List[str] = []

    # ---- 配置、标称帧率、时间戳来源与标定（缺失只影响同步预检）
    processing = None
    try:
        snapshot_entry = session.get("processing_config")
        relative = snapshot_entry.get("path") if isinstance(snapshot_entry, dict) else None
        snapshot = record_dir / (relative or "calibration_snapshot/processing.yaml")
        processing = rt.resolve_processing(
            ctx.explicit_config, snapshot if snapshot.is_file() else None, DEFAULT_PROCESSING
        )
    except rt.ConfigError as error:
        problems.append(issue("CONFIG_INVALID", str(error)))
    nominal_fps, fps_source, selection_path = None, None, None
    try:
        nominal_fps, fps_source = rt.resolve_source_fps(args.source_camera_fps, session, entry.record_id)
        selection_path = rt.frame_selection.selection_path_for(nominal_fps)
    except rt.ConfigError as error:
        problems.append(issue("NOMINAL_FPS_UNAVAILABLE", str(error)))
    image_meta = session.get("image") if isinstance(session.get("image"), dict) else {}
    if ctx.calibration is None:
        problems.append(ctx.calibration_problem)
    else:
        try:
            note = rt.check_timestamp_source(ctx.calibration, image_meta.get("timestamp_source"))
            if note:
                notes.append(note)
        except rt.CalibrationError as error:
            problems.append(issue("TIMESTAMP_SOURCE_MISMATCH", str(error)))

    topic_of = resolve_topics(processing, session)
    bag_uri = record_dir / str(session.get("bag_uri") or "raw/bag")
    if not bag_uri.is_dir():
        raise AnalysisError(f"bag 目录 {bag_uri} 不存在")
    reader = rt.open_reader(bag_uri)
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    del reader
    bag_meta = read_bag_metadata(bag_uri)
    if bag_meta is None:
        notes.append("无法读取 bag 的 metadata.yaml，跳过元数据计数核对")

    # ---- 话题、类型与图像模式
    for role in ROLES:
        topic = topic_of[role]
        if topic not in types:
            rejections.append(issue("MISSING_STREAM", f"缺少{role}话题 {topic}", role=role, topic=topic))
        elif role != ROLE_IMAGE and types[topic] != ROLE_TYPES[role]:
            rejections.append(
                issue("TOPIC_TYPE_MISMATCH",
                      f"{topic} 类型为 {types[topic]}，应为 {ROLE_TYPES[role]}", role=role, topic=topic)
            )
    image_topic = topic_of[ROLE_IMAGE]
    mode = rt.mode_for_message_type(types.get(image_topic, "")) if image_topic in types else None
    if image_topic in types and mode is None:
        rejections.append(
            issue("TOPIC_TYPE_MISMATCH", f"{image_topic} 类型 {types[image_topic]} 不是支持的图像消息",
                  role=ROLE_IMAGE, topic=image_topic)
        )
    configured_mode = processing.image_mode if processing is not None else None
    if mode and configured_mode and configured_mode != mode:
        problems.append(
            issue("IMAGE_MODE_MISMATCH", f"配置的 image_mode={configured_mode} 与话题实际类型 {mode} 不一致")
        )
    elif mode:
        try:
            rt.check_bag_image_topic(types, image_topic, mode)
        except ValueError as error:
            problems.append(issue("IMAGE_MODE_MISMATCH", str(error)))

    timing_topic = find_timing_topic(types, image_topic)
    scan = scan_bag(rt, bag_uri, topic_of, types, timing_topic)
    counts, count_warnings = check_counts(session, topic_of, scan, bag_meta)
    warnings.extend(count_warnings)
    for topic, count in scan.corrupt.items():
        rejections.append(issue("MESSAGE_DECODE_FAILED", f"{topic} 有 {count} 条消息无法解析", topic=topic,
                                count=int(count), examples=scan.corrupt_examples[:5]))
    for role in (ROLE_IMAGE, ROLE_POSE, ROLE_STATUS, ROLE_GRIPPER):
        topic = topic_of[role]
        if topic in types and scan.read_counts.get(topic, 0) == 0 and (role == ROLE_IMAGE or types[topic] == ROLE_TYPES[role]):
            rejections.append(issue("EMPTY_STREAM", f"话题 {topic} 存在但没有任何消息", role=role, topic=topic))

    # ---- 附带文件与身份
    if not (record_dir / "quality.yaml").is_file():
        warnings.append(issue("QUALITY_YAML_MISSING", "缺少 quality.yaml"))
        quality = None
    else:
        try:
            quality = load_yaml_mapping(record_dir / "quality.yaml")
        except AnalysisError as error:
            warnings.append(issue("QUALITY_YAML_INVALID", str(error)))
            quality = None
    warnings.extend(check_snapshots(record_dir, session))

    serials = {key: int(value) for key, value in scan.serials.items()}
    if "" in serials:
        rejections.append(issue("TRACKER_IDENTITY_INVALID", f"Tracker 状态有 {serials['']} 条空序列号"))
    named = sorted(key for key in serials if key)
    if len(named) > 1:
        rejections.append(issue("TRACKER_IDENTITY_INVALID", f"Tracker 状态含多个序列号 {named}"))
    elif named and ctx.calibration is not None and named[0] != ctx.calibration.tracker_serial:
        problems.append(
            issue("TRACKER_SERIAL_MISMATCH",
                  f"Tracker 序列号 {named[0]} 与标定绑定 {ctx.calibration.tracker_serial} 不一致")
        )

    # ---- 时钟映射
    timing = {"topic": timing_topic, "messages": len(scan.timing_rows)}
    if scan.timing_rows:
        generations = sorted({row[1] for row in scan.timing_rows})
        offsets = np.asarray([row[2] for row in scan.timing_rows], dtype=np.float64)
        sequences = np.asarray([row[3] for row in scan.timing_rows], dtype=np.int64)
        timing.update(
            {
                "generations": generations,
                "offset_ns_range": [float(offsets.min()), float(offsets.max())],
                "v4l2_sequence_gaps": int(np.count_nonzero(np.diff(sequences) > 1)) if sequences.size > 1 else 0,
            }
        )
        if len(generations) > 1:
            rejections.append(
                issue("CLOCK_MAPPING_CHANGED", f"采集期间相机时钟映射代号变化 {generations}，图像时间不可比")
            )
    else:
        timing["reasons"] = {"generations": "记录中没有相机时间诊断话题或消息"}

    # ---- 记录级流统计
    image_ok = scan.image_ok if scan.image_ok.size else np.zeros(0, dtype=bool)
    all_image = np.ones(image_ok.size, dtype=bool)
    streams = {
        ROLE_IMAGE: stream_timing(scan.image_stamps[image_ok], scan.image_bag[image_ok]),
        ROLE_POSE: pose_report(scan, np.ones(scan.pose_stamps.size, dtype=bool)),
        ROLE_STATUS: stream_timing(scan.status_stamps, scan.status_bag),
        ROLE_GRIPPER: stream_timing(scan.gripper_stamps, scan.gripper_bag),
    }
    firsts = [int(a.min()) for a in (scan.image_stamps[image_ok], scan.pose_stamps, scan.gripper_stamps) if a.size]
    origin_ns = min(firsts) if firsts else 0
    all_status = np.ones(scan.status_stamps.size, dtype=bool)
    all_gripper = np.ones(scan.gripper_stamps.size, dtype=bool)
    streams[ROLE_STATUS].update(status_report(scan, all_status, origin_ns))
    streams[ROLE_GRIPPER].update(gripper_report(scan, all_gripper, origin_ns, all_gripper))
    for role in (ROLE_IMAGE, ROLE_POSE, ROLE_STATUS, ROLE_GRIPPER):
        stream = streams[role]
        if stream.get("duplicate_count") or stream.get("reversed_count"):
            warnings.append(
                issue("TIMESTAMP_ORDER", f"{role} 有 {stream['duplicate_count']} 个重复、"
                      f"{stream['reversed_count']} 个倒序时间戳", role=role)
            )
        if stream.get("negative_latency_count"):
            warnings.append(
                issue("NEGATIVE_LATENCY", f"{role} 有 {stream['negative_latency_count']} 条消息的接收时间早于源时间",
                      role=role)
            )
    if streams[ROLE_GRIPPER].get("outside_unit_range"):
        warnings.append(issue("GRIPPER_OUT_OF_RANGE", "夹爪开度存在超出 [0, 1] 的有效样本"))
    if streams[ROLE_GRIPPER].get("valid_flag_but_nonfinite"):
        warnings.append(issue("GRIPPER_VALID_NONFINITE", "夹爪标记有效但开度非有限，同步时按无效处理"))
    if nominal_fps and streams[ROLE_IMAGE].get("rate_hz"):
        streams[ROLE_IMAGE]["nominal_fps"] = nominal_fps
        streams[ROLE_IMAGE]["rate_ratio_to_nominal"] = streams[ROLE_IMAGE]["rate_hz"] / nominal_fps

    # ---- 事件与 episode
    session_id = str(session.get("session_id") or "").strip() or None
    specs, event_defects = parse_events(scan.events, session_id)
    for defect in event_defects:
        rejections.append(defect)
    event_topic = topic_of[ROLE_EVENT]
    if types.get(event_topic) == ROLE_TYPES[ROLE_EVENT] and not specs and not event_defects:
        rejections.append(issue("NO_EPISODE", "记录内没有任何 episode 事件"))
    declared_index = session.get("episode_index")
    if len(specs) == 1 and specs[0].start is not None and isinstance(declared_index, int) \
            and specs[0].index != declared_index:
        warnings.append(
            issue("EPISODE_INDEX_SESSION_MISMATCH",
                  f"事件 episode 编号 {specs[0].index} 与 session.yaml 的 {declared_index} 不一致")
        )

    # ---- 图像解码（全量或抽样）
    total_images = int(scan.image_stamps.size)
    abnormal_segments = collect_abnormal_segments(scan, origin_ns, nominal_fps)
    decode: Optional[DecodeOutcome] = None
    if mode and total_images:
        wanted = None
        if args.decode_mode == "sample":
            valid_ordinals = np.flatnonzero(image_ok)
            wanted = {int(valid_ordinals[i]) for i in sample_ordinals(scan.image_stamps[image_ok])}
            for first, last in abnormal_segments[:SAMPLE_EXTRA_LIMIT]:
                wanted.update(segment_neighbors(first, last, total_images))
        decode = decode_images(rt, bag_uri, image_topic, mode, total_images, wanted)
    record_integrity = integrity_report(decode, all_image, args.decode_mode, "record")
    record_integrity.update(
        {
            "mode": args.decode_mode,
            "image_mode": mode,
            "frame_shapes": {"x".join(map(str, shape)): int(n) for shape, n in decode.shapes.items()}
            if decode else {},
            "failures": [
                {"ordinal": ordinal, "stamp_ns": int(scan.image_stamps[ordinal]), "error": message}
                for ordinal, message in sorted(decode.errors.items())[:DETAIL_LIMIT]
            ] if decode else [],
            "decoder": decode.stats if decode else {},
            "drain_error": decode.drain_error if decode else "",
        }
    )
    if decode and len(decode.shapes) > 1:
        rejections.append(
            issue("IMAGE_SIZE_CHANGED", f"解码图像尺寸不一致: {record_integrity['frame_shapes']}")
        )
    if decode and decode.drain_error:
        rejections.append(issue("IMAGE_DECODE_FAILED", f"排空解码器失败: {decode.drain_error}"))
    if args.decode_mode == "sample" and record_integrity["verdict"] == "仅抽样验证":
        notes.append(
            f"抽样模式：解码覆盖率 {record_integrity['checked']}/{record_integrity['messages']}，"
            "图像完整性仅抽样验证"
        )

    # ---- 逐 episode
    episodes = []
    names: Counter = Counter()
    for spec in specs:
        base_name = f"{entry.record_id}/episode_{spec.index:04d}"
        names[base_name] += 1
        name = base_name if names[base_name] == 1 else f"{base_name}#{names[base_name]}"
        episodes.append(
            analyze_episode(ctx, name, spec, scan, decode, processing, nominal_fps, problems)
        )

    if decode is not None:
        inside = np.zeros(total_images, dtype=bool)
        for spec in specs:
            if spec.start is not None and spec.stop is not None and spec.stop.stamp_ns > spec.start.stamp_ns:
                inside |= episode_image_mask(scan.image_stamps, scan.image_bag, spec.start.stamp_ns,
                                             spec.stop.stamp_ns, rt.POST_ROLL_NS)
        outside = decode.status[~inside]
        failed_outside = int(np.count_nonzero(outside == DECODE_FAILED))
        skipped_outside = int(np.count_nonzero(outside == DECODE_SKIPPED))
        if failed_outside:
            warnings.append(
                issue("IMAGE_DECODE_FAILED_OUTSIDE_EPISODE", f"episode 之外有 {failed_outside} 张图像解码失败",
                      count=failed_outside))
        if skipped_outside:
            warnings.append(
                issue("IMAGE_PREFIX_SKIPPED", f"episode 之外有 {skipped_outside} 个 H.264 包位于关键帧之前或解码失败之后被跳过",
                      count=skipped_outside))
    has_rejection = bool(rejections) or any(e["status"] == STATUS_REJECTED for e in episodes)
    has_incomplete = any(e["status"] == STATUS_INCOMPLETE for e in episodes)
    status = record_status(False, has_rejection, has_incomplete)
    sync_total = len(episodes)
    sync_run = sum(1 for e in episodes if e["sync"]["status"] in (SYNC_ACCEPTED, SYNC_REJECTED))
    session_summary = {
        key: session.get(key)
        for key in ("collection_uuid", "session_id", "task_name", "dir_name", "episode_index",
                    "created_at", "saved_at", "duration_s", "state", "calibration_status")
    }
    report = {
        "record_id": entry.record_id,
        "record_dir": str(record_dir),
        "bag_uri": str(bag_uri),
        "session": session_summary,
        "status": status,
        "failed": False,
        "has_rejection": has_rejection,
        "has_incomplete": has_incomplete,
        "rejections": rejections,
        "warnings": warnings,
        "preflight_problems": problems,
        "notes": notes,
        "is_latest": entry.record_id == ctx.latest_id,
        "image_mode": mode,
        "nominal_source_fps": nominal_fps,
        "nominal_source_fps_source": fps_source,
        "selection_path": selection_path,
        "config": processing.report() if processing is not None else None,
        "topics": topic_of,
        "topic_types": {topic_of[role]: types.get(topic_of[role]) for role in ROLES},
        "counts": counts,
        "streams": streams,
        "image_integrity": record_integrity,
        "clock_mapping": timing,
        "tracker_serials": serials,
        "quality_yaml": (
            {"has_alarms": quality.get("has_alarms"), "alarms": quality.get("alarms")} if quality else None
        ),
        "coverage": {
            "decode_mode": args.decode_mode,
            "image_messages": record_integrity["messages"],
            "image_checked": record_integrity["checked"],
            "image_coverage": record_integrity["coverage"],
            "episodes": sync_total,
            "sync_preflight_run": sync_run,
        },
        "episodes": episodes,
        "origin_ns": origin_ns,
    }
    return RecordResult(report, scan, abnormal_segments)


def collect_abnormal_segments(
    scan: ScanResult, origin_ns: int, nominal_fps: Optional[float]
) -> List[Tuple[int, int]]:
    """把夹爪失效段、Tracker 异常段和图像间隔异常映射为图像序号区间，按严重程度排序。"""
    if scan.image_stamps.size == 0:
        return []
    ok = scan.image_ok
    candidates: List[Tuple[float, int, int]] = []
    gripper_ok = scan.gripper_valid & np.isfinite(scan.gripper_filtered)
    for run in invalid_runs(scan.gripper_stamps, gripper_ok, origin_ns):
        first, last = ordinals_between(scan.image_stamps, ok, run["start_ns"], run["end_ns"])
        if first >= 0:
            candidates.append((run["samples"] + 1000.0, first, last))
    for run in invalid_runs(scan.status_stamps, scan.status_ok, origin_ns):
        first, last = ordinals_between(scan.image_stamps, ok, run["start_ns"], run["end_ns"])
        if first >= 0:
            candidates.append((run["samples"] / 3.0 + 500.0, first, last))
    if nominal_fps:
        stamps = scan.image_stamps
        deltas = np.diff(stamps)
        period = 1.0e9 / nominal_fps
        for index in np.flatnonzero((deltas <= 0) | (deltas > 1.5 * period)).tolist()[:DETAIL_LIMIT]:
            candidates.append((float(abs(deltas[index])) / period, index, index + 1))
    candidates.sort(key=lambda item: -item[0])
    return [(first, last) for _, first, last in candidates]


def analyze_episode(
    ctx: Context, name: str, spec: EpisodeSpec, scan: ScanResult, decode: Optional[DecodeOutcome],
    processing: Any, nominal_fps: Optional[float], problems: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """一条 episode 的基础指标、数据缺陷判定和同步预检。

    ``problems`` 是环境、配置或标定问题：没有数据缺陷时同步无法执行，episode 标记为预检未完成。
    """
    rt = ctx.rt
    rejections: List[Dict[str, Any]] = list(spec.defects)
    warnings: List[Dict[str, Any]] = []
    episode: Dict[str, Any] = {
        "name": name,
        "episode_index": spec.index,
        "start_ns": spec.start.stamp_ns if spec.start else None,
        "stop_ns": spec.stop.stamp_ns if spec.stop else None,
        "duration_s": (
            (spec.stop.stamp_ns - spec.start.stamp_ns) / 1.0e9 if spec.start and spec.stop else None
        ),
        "rejections": rejections,
        "warnings": warnings,
        "sync": {"status": SYNC_BLOCKED},
    }
    if spec.start is None or spec.stop is None:
        episode["status"] = STATUS_REJECTED
        episode["sync"] = {"status": SYNC_BLOCKED, "reason": "事件序列不完整，无法确定时间窗"}
        return episode

    start_ns, stop_ns = spec.start.stamp_ns, spec.stop.stamp_ns
    pre, post = rt.PRE_ROLL_NS, rt.POST_ROLL_NS
    if spec.defects and not (stop_ns > start_ns):
        episode["status"] = STATUS_REJECTED
        episode["sync"] = {"status": SYNC_BLOCKED, "reason": "事件缺陷导致时间窗无效"}
        return episode
    image_mask = episode_image_mask(scan.image_stamps, scan.image_bag, start_ns, stop_ns, post) & scan.image_ok
    image_ordinals = np.flatnonzero(image_mask)
    image_stamps = scan.image_stamps[image_mask]
    support = (lambda stamps: window_mask(stamps, start_ns - pre, stop_ns + post))
    pose_mask = support(scan.pose_stamps)
    status_mask = support(scan.status_stamps)
    gripper_support = support(scan.gripper_stamps)
    gripper_body = window_mask(scan.gripper_stamps, start_ns, stop_ns)
    status_body = window_mask(scan.status_stamps, start_ns, stop_ns)

    episode["images"] = stream_timing(image_stamps, scan.image_bag[image_mask])
    episode["tracker_pose"] = pose_report(scan, pose_mask)
    episode["tracker_status"] = status_report(scan, status_body, start_ns)
    episode["tracker_status"].update(
        {k: v for k, v in stream_timing(scan.status_stamps[status_mask]).items()
         if k in ("count", "rate_hz", "max_gap_ms")}
    )
    episode["gripper"] = gripper_report(scan, gripper_body, start_ns, gripper_support)
    episode["gripper"].update(
        {k: v for k, v in stream_timing(scan.gripper_stamps[gripper_support]).items()
         if k in ("rate_hz", "max_gap_ms")}
    )
    episode["image_integrity"] = integrity_report(decode, image_mask, ctx.args.decode_mode, "episode")

    # ---- 数据缺陷（与外参无关）
    if image_stamps.size < 2:
        rejections.append(issue("INSUFFICIENT_IMAGES", f"episode 内只有 {image_stamps.size} 张图像"))
    if int(np.count_nonzero(pose_mask)) < 2:
        rejections.append(issue("MISSING_STREAM", "episode 内 Tracker 位姿不足两帧"))
    if not gripper_support.any():
        rejections.append(issue("MISSING_STREAM", "episode 内没有夹爪状态"))
    if not status_mask.any():
        rejections.append(issue("MISSING_STREAM", "episode 内没有 Tracker 状态消息"))
    pose_bad = episode["tracker_pose"]["invalid_count"]
    if pose_bad:
        rejections.append(
            issue("INVALID_POSE", f"episode 支撑窗口内有 {pose_bad} 个无效位姿（非有限值或四元数范数异常）")
        )
    if decode is not None:
        local = decode.status[image_ordinals]
        failed = int(np.count_nonzero(local == DECODE_FAILED))
        skipped = int(np.count_nonzero(local == DECODE_SKIPPED))
        if failed:
            first = int(image_ordinals[np.flatnonzero(local == DECODE_FAILED)[0]])
            rejections.append(
                issue("IMAGE_DECODE_FAILED",
                      f"episode 内有 {failed} 张图像解码失败: {decode.errors.get(first, '')}",
                      count=failed, first_ordinal=first)
            )
        if skipped:
            rejections.append(
                issue("IMAGE_NO_KEYFRAME", f"episode 内有 {skipped} 个 H.264 包位于关键帧之前或解码失败之后，无法解码",
                      count=skipped)
            )
        mixed = {tuple(shape) for shape in decode.shapes}
        if len(mixed) > 1:
            rejections.append(issue("IMAGE_SIZE_CHANGED", "episode 内图像尺寸不一致"))
    if rejections:
        episode["status"] = STATUS_REJECTED
        episode["sync"] = {"status": SYNC_BLOCKED, "reason": "存在数据缺陷，未执行同步"}
        return episode

    # ---- 同步预检
    if problems or ctx.calibration is None or nominal_fps is None or processing is None:
        episode["status"] = STATUS_INCOMPLETE
        episode["sync"] = {
            "status": SYNC_INCOMPLETE,
            "reason": "环境、配置或标定问题导致同步无法执行",
            "problems": [problem["code"] for problem in problems],
        }
        return episode
    poses = [
        rt.PoseSample(int(t), x.copy(), q.copy())
        for t, x, q in zip(scan.pose_stamps[pose_mask], scan.pose_xyz[pose_mask], scan.pose_quat[pose_mask])
    ]
    grippers = [
        rt.GripperSample(int(t), float(raw), float(filtered), int(markers), bool(valid))
        for t, raw, filtered, markers, valid in zip(
            scan.gripper_stamps[gripper_support], scan.gripper_raw[gripper_support],
            scan.gripper_filtered[gripper_support], scan.gripper_markers[gripper_support],
            scan.gripper_valid[gripper_support],
        )
    ]
    statuses = [
        rt.TrackerStatusSample(int(t), bool(c), bool(v), int(s))
        for t, c, v, s in zip(
            scan.status_stamps[status_mask], scan.status_connected[status_mask],
            scan.status_pose_valid[status_mask], scan.status_state[status_mask],
        )
    ]
    offset_ns = int(round(ctx.calibration.tracker_time_offset_ms * 1.0e6))
    result = rt.synchronize_link7_episode(
        image_stamps.tolist(), poses, grippers, statuses, ctx.calibration.tracker_to_link7,
        nominal_fps, processing.sync, tracker_time_offset_ns=offset_ns,
    )
    sync: Dict[str, Any] = {"stats": result.stats}
    warning_items = list(result.warnings)
    warnings.extend(warning_items[:DETAIL_LIMIT])
    if len(warning_items) > DETAIL_LIMIT:
        warnings.append(issue("WARNINGS_TRUNCATED", f"另有 {len(warning_items) - DETAIL_LIMIT} 条同步警告未列出"))
    if result.episode is None:
        rejections.extend(result.rejections)
        sync["status"] = SYNC_REJECTED
        episode["status"] = STATUS_REJECTED
    else:
        item = result.episode
        sync.update(
            {
                "status": SYNC_ACCEPTED,
                "output_frames": len(item),
                "trimmed_head": result.stats.get("trimmed_head", 0),
                "trimmed_tail": result.stats.get("trimmed_tail", 0),
                "dropped_frames": result.stats.get("output_dropped_frame_total", 0),
                "pose_gap_ms": describe(item.pose_gap_ms),
                "gripper_gap_ms": describe(item.gripper_gap_ms),
                "gripper_directly_observed_ratio": float(np.mean(item.gripper_observed)),
            }
        )
        episode["status"] = STATUS_PASSED
    episode["sync"] = sync
    return episode


# ---------------------------------------------------------------- 失败记录
def failed_record(entry: RecordEntry, error: BaseException) -> Dict[str, Any]:
    """分析异常时的占位结果：状态为失败，保留错误类型、消息和调用栈末尾。"""
    trace = traceback.format_exception(type(error), error, error.__traceback__)
    return {
        "record_id": entry.record_id,
        "record_dir": str(entry.path),
        "status": STATUS_FAILED,
        "failed": True,
        "has_rejection": False,
        "has_incomplete": False,
        "rejections": [],
        "warnings": [],
        "preflight_problems": [],
        "notes": [],
        "is_latest": False,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback_tail": [line.rstrip() for line in trace[-4:]],
        },
        "episodes": [],
        "coverage": {"decode_mode": None, "image_messages": 0, "image_checked": 0,
                     "image_coverage": None, "episodes": 0, "sync_preflight_run": 0},
    }


# ---------------------------------------------------------------- 代表帧与图表
def safe_name(text: str) -> str:
    """把记录标识转换为可用作文件名的短串。"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "record"


def nearest_index(sorted_values: Any, value: int) -> int:
    """有序数组中与 value 最近的元素下标。"""
    position = int(np.searchsorted(sorted_values, value))
    if position <= 0:
        return 0
    if position >= len(sorted_values):
        return len(sorted_values) - 1
    return position if sorted_values[position] - value < value - sorted_values[position - 1] else position - 1


def annotate_frame(image: Any, lines: Sequence[str]) -> Any:
    """缩放到不超过 640 像素宽，并写入带描边的 ASCII 标注。"""
    import cv2

    height, width = image.shape[:2]
    if width > 640:
        scale = 640.0 / width
        image = cv2.resize(image, (640, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    image = np.ascontiguousarray(image)
    for index, text in enumerate(lines):
        origin = (8, 20 + 20 * index)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def render_previews(
    ctx: Context, result: RecordResult, directory: Path, relative: str
) -> List[Dict[str, Any]]:
    """第二遍解码，保存带时间、有效性和标签数量标注的代表帧（不重新检测标签）。"""
    import cv2

    report, scan = result.report, result.scan
    total = int(scan.image_stamps.size)
    mode = report.get("image_mode")
    if not mode or total == 0:
        return []
    ordinals = choose_preview_ordinals(total, result.abnormal_segments)
    outcome = decode_images(
        ctx.rt, Path(report["bag_uri"]), report["topics"][ROLE_IMAGE], mode, total,
        set(ordinals), set(ordinals),
    )
    gripper_order = np.argsort(scan.gripper_stamps, kind="stable")
    gripper_sorted = scan.gripper_stamps[gripper_order]
    status_order = np.argsort(scan.status_stamps, kind="stable")
    status_sorted = scan.status_stamps[status_order]
    directory.mkdir(parents=True, exist_ok=True)
    saved = []
    for rank, ordinal in enumerate(ordinals):
        image = outcome.previews.get(ordinal)
        stamp = int(scan.image_stamps[ordinal])
        item: Dict[str, Any] = {
            "ordinal": ordinal, "stamp_ns": stamp,
            "t_s": (stamp - report["origin_ns"]) / 1.0e9, "file": None,
        }
        episode_name = next(
            (e["name"] for e in report["episodes"]
             if e.get("start_ns") is not None and e.get("stop_ns") is not None
             and e["start_ns"] <= stamp <= e["stop_ns"]), None)
        item["episode"] = episode_name
        lines = [f"t={item['t_s']:+.3f}s frame#{ordinal}" + (f" ep={episode_name.rsplit('/', 1)[-1]}" if episode_name else "")]
        if gripper_sorted.size:
            index = int(gripper_order[nearest_index(gripper_sorted, stamp)])
            if abs(int(scan.gripper_stamps[index]) - stamp) <= 1_000_000:
                valid = bool(scan.gripper_valid[index]) and bool(np.isfinite(scan.gripper_filtered[index]))
                item.update({"gripper_valid": valid, "marker_count": int(scan.gripper_markers[index]),
                             "openness": float(scan.gripper_filtered[index]) if valid else None})
                lines.append(f"gripper {'VALID' if valid else 'INVALID'} markers={item['marker_count']}"
                             + (f" open={item['openness']:.2f}" if valid else ""))
            else:
                item["gripper_valid"] = None
                lines.append("gripper: no matching message")
        if status_sorted.size:
            index = int(status_order[nearest_index(status_sorted, stamp)])
            ok = bool(scan.status_ok[index])
            item["tracker_ok"] = ok
            lines.append(f"tracker {'OK' if ok else 'BAD'}")
        if image is None:
            lines.append("decode failed or skipped")
            item["decode"] = "unavailable"
        else:
            file_name = f"{rank:02d}_frame{ordinal}.jpg"
            cv2.imwrite(str(directory / file_name), annotate_frame(image, lines),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            item["file"] = f"{relative}/{file_name}"
        saved.append(item)
    return saved


def render_plot(result: RecordResult, path: Path) -> None:
    """记录质量曲线：图像间隔、接收延迟、Tracker 间隔/异常、夹爪有效性。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report, scan = result.report, result.scan
    origin = report["origin_ns"]

    def seconds(stamps: Any) -> Any:
        return (np.asarray(stamps, dtype=np.float64) - origin) / 1.0e9

    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    ok = scan.image_ok
    image_stamps = np.sort(np.unique(scan.image_stamps[ok]))
    if image_stamps.size > 1:
        axes[0].plot(seconds(image_stamps[1:]), np.diff(image_stamps) / 1.0e6, ".", markersize=3)
    fps = report.get("nominal_source_fps")
    if fps:
        axes[0].axhline(1000.0 / fps, color="gray", linestyle="--", linewidth=0.8, label="nominal period")
        axes[0].legend(loc="upper right")
    axes[0].set_ylabel("image interval (ms)")
    for label, stamps, bag in (
        ("image", scan.image_stamps[ok], scan.image_bag[ok]),
        ("tracker pose", scan.pose_stamps, scan.pose_bag),
        ("gripper", scan.gripper_stamps, scan.gripper_bag),
    ):
        if stamps.size:
            axes[1].plot(seconds(stamps), (bag - stamps) / 1.0e6, ".", markersize=2, label=label)
    axes[1].set_ylabel("receive latency (ms)")
    if axes[1].has_data():
        axes[1].legend(loc="upper right", markerscale=4)
    pose_stamps = np.sort(scan.pose_stamps)
    if pose_stamps.size > 1:
        axes[2].plot(seconds(pose_stamps[1:]), np.diff(pose_stamps) / 1.0e6, ".", markersize=2,
                     label="pose interval")
    bad = ~scan.status_ok
    if bad.any():
        axes[2].plot(seconds(scan.status_stamps[bad]), np.zeros(int(bad.sum())), "rx", label="status not OK")
    axes[2].set_ylabel("tracker (ms)")
    if axes[2].has_data():
        axes[2].legend(loc="upper right")
    usable = scan.gripper_valid & np.isfinite(scan.gripper_filtered)
    if usable.any():
        axes[3].plot(seconds(scan.gripper_stamps[usable]), scan.gripper_filtered[usable], ".", markersize=2,
                     label="openness (valid)")
    if (~usable).any():
        axes[3].plot(seconds(scan.gripper_stamps[~usable]), np.full(int((~usable).sum()), -0.05), "r|",
                     label="invalid")
    axes[3].set_ylabel("gripper openness")
    axes[3].set_xlabel("time since first message (s)")
    if axes[3].has_data():
        axes[3].legend(loc="upper right")
    for axis in axes:
        for episode in report["episodes"]:
            if episode.get("start_ns") is not None and episode.get("stop_ns") is not None:
                color = "tab:green" if episode["status"] == STATUS_PASSED else "tab:red"
                axis.axvspan(*seconds([episode["start_ns"], episode["stop_ns"]]), color=color, alpha=0.08)
        axis.grid(alpha=0.3)
    figure.suptitle(f"{report['record_id']}  [{report['status']}]")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def wants_plot(policy: str, report: Dict[str, Any]) -> bool:
    """按 ``--plots`` 策略判断是否为该记录生成图表：异常记录及最新记录，或全部/不生成。"""
    if policy == "all":
        return True
    if policy == "none":
        return False
    return report["status"] != STATUS_PASSED or bool(report.get("is_latest"))


# ---------------------------------------------------------------- 报告输出
def atomic_write(path: Path, text: str) -> None:
    """先写同目录临时文件再替换，避免留下半写文件。"""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


CSV_COLUMNS = (
    "record_id", "task_name", "episode_name", "episode_index", "record_status", "episode_status",
    "sync_status", "rejection_codes", "warning_codes", "duration_s", "image_messages",
    "image_rate_hz", "image_max_gap_ms", "image_checked_ratio", "tracker_rate_hz",
    "tracker_max_gap_ms", "gripper_samples", "gripper_valid_ratio", "gripper_invalid_runs",
    "gripper_max_valid_gap_s", "output_frames", "trimmed_head", "trimmed_tail", "dropped_frames",
)


def csv_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """每条 episode 一行；没有 episode 的记录（失败、缺事件等）单独一行。"""
    rows = []
    for record in records:
        session = record.get("session") or {}
        base = {
            "record_id": record["record_id"], "task_name": session.get("task_name"),
            "record_status": record["status"],
        }
        record_codes = [item["code"] for item in record.get("rejections", [])]
        if record.get("error"):
            record_codes.append(record["error"]["type"])
        episodes = record.get("episodes") or []
        if not episodes:
            rows.append({**base, "episode_status": "", "rejection_codes": ";".join(record_codes),
                         "warning_codes": ";".join(sorted({w["code"] for w in record.get("warnings", [])}))})
            continue
        for episode in episodes:
            sync = episode.get("sync") or {}
            images = episode.get("images") or {}
            tracker = episode.get("tracker_pose") or {}
            gripper = episode.get("gripper") or {}
            integrity = episode.get("image_integrity") or {}
            codes = record_codes + [item["code"] for item in episode.get("rejections", [])]
            warning_codes = sorted({w["code"] for w in record.get("warnings", [])}
                                   | {w["code"] for w in episode.get("warnings", [])})
            rows.append(
                {
                    **base,
                    "episode_name": episode["name"], "episode_index": episode["episode_index"],
                    "episode_status": episode["status"], "sync_status": sync.get("status"),
                    "rejection_codes": ";".join(codes), "warning_codes": ";".join(warning_codes),
                    "duration_s": episode.get("duration_s"),
                    "image_messages": images.get("count"), "image_rate_hz": images.get("rate_hz"),
                    "image_max_gap_ms": images.get("max_gap_ms"),
                    "image_checked_ratio": integrity.get("coverage"),
                    "tracker_rate_hz": tracker.get("rate_hz"), "tracker_max_gap_ms": tracker.get("max_gap_ms"),
                    "gripper_samples": gripper.get("samples"), "gripper_valid_ratio": gripper.get("valid_ratio"),
                    "gripper_invalid_runs": gripper.get("invalid_run_count"),
                    "gripper_max_valid_gap_s": gripper.get("max_valid_gap_s"),
                    "output_frames": sync.get("output_frames"), "trimmed_head": sync.get("trimmed_head"),
                    "trimmed_tail": sync.get("trimmed_tail"), "dropped_frames": sync.get("dropped_frames"),
                }
            )
    return rows


def write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """写 summary.csv（UTF-8 带 BOM，便于表格软件打开）；非有限数写为空。"""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in csv_rows(records):
            clean = {}
            for key in CSV_COLUMNS:
                value = row.get(key)
                if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
                    value = ""
                clean[key] = "" if value is None else value
            writer.writerow(clean)
    os.replace(temporary, path)


def batch_totals(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """批次汇总计数。"""
    episodes = [e for r in records for e in r.get("episodes", [])]
    return {
        "records": len(records),
        "passed": sum(r["status"] == STATUS_PASSED for r in records),
        "rejected": sum(r["status"] == STATUS_REJECTED for r in records),
        "incomplete": sum(r["status"] == STATUS_INCOMPLETE for r in records),
        "failed": sum(r["status"] == STATUS_FAILED for r in records),
        "records_with_incomplete_episodes": sum(bool(r.get("has_incomplete")) for r in records),
        "episodes": len(episodes),
        "episodes_passed": sum(e["status"] == STATUS_PASSED for e in episodes),
        "episodes_rejected": sum(e["status"] == STATUS_REJECTED for e in episodes),
        "episodes_incomplete": sum(e["status"] == STATUS_INCOMPLETE for e in episodes),
        "image_messages": sum(r["coverage"]["image_messages"] for r in records),
        "image_checked": sum(r["coverage"]["image_checked"] for r in records),
    }


def reasons_of(record: Dict[str, Any]) -> List[str]:
    """记录的拒绝/未完成/失败原因（去重，带 episode 名称）。"""
    items: List[str] = []
    if record.get("error"):
        items.append(f"{record['error']['type']}: {record['error']['message']}")
    for item in record.get("rejections", []):
        items.append(f"[{item['code']}] {item['message']}")
    for problem in record.get("preflight_problems", []):
        items.append(f"[{problem['code']}] {problem['message']}")
    for episode in record.get("episodes", []):
        short = episode["name"].rsplit("/", 1)[-1]
        for item in episode.get("rejections", []):
            items.append(f"{short} [{item['code']}] {item['message']}")
    return list(dict.fromkeys(items))


def render_markdown(report: Dict[str, Any]) -> str:
    """中文批次报告：结论、名单、异常原因、警告汇总、覆盖率和图表链接。"""
    totals = report["totals"]
    records = report["records"]
    lines = [
        "# 采集质量分析报告",
        "",
        f"- 输入：`{report['input']}`",
        f"- 生成时间：{report['generated_at']}",
        f"- 解码模式：`{report['decode_mode']}`，图表：`{report['plots']}`",
        f"- 退出码：**{report['exit_code']}**（0 无拒绝/错误，1 存在质量拒绝，2 存在错误或预检未完成）",
        "",
        "## 批次结论",
        "",
    ]
    if report["exit_code"] == EXIT_OK:
        lines.append("**全部记录通过质量检查**（允许存在警告）。")
    elif report["exit_code"] == EXIT_REJECTED:
        lines.append(f"**存在质量拒绝**：{totals['rejected']} 条记录被拒绝，分析均已完成。")
    else:
        lines.append(
            f"**分析未完全完成**：{totals['failed']} 条分析失败，{totals['incomplete']} 条预检未完成，"
            f"{totals['rejected']} 条质量拒绝。"
        )
    lines += [
        "",
        "| 记录 | 通过 | 质量拒绝 | 预检未完成 | 分析失败 | episode 总数 | 通过 | 拒绝 | 未完成 |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| {totals['records']} | {totals['passed']} | {totals['rejected']} | {totals['incomplete']} | "
        f"{totals['failed']} | {totals['episodes']} | {totals['episodes_passed']} | "
        f"{totals['episodes_rejected']} | {totals['episodes_incomplete']} |",
        "",
        f"图像检查覆盖率：{totals['image_checked']}/{totals['image_messages']}"
        + (f"（{totals['image_checked'] / totals['image_messages']:.1%}）" if totals["image_messages"] else ""),
        "",
    ]
    if report["decode_mode"] == "sample":
        lines += ["> 抽样模式：图像完整性结论为**仅抽样验证**，未检查的帧不保证可解码。", ""]
    calibration = report.get("calibration") or {}
    lines += ["## 同步预检依据", ""]
    if calibration.get("available"):
        lines += [
            f"- Tracker 序列号：`{calibration['tracker_serial']}`（{calibration['tracker_serial_source']}）",
            f"- Tracker 查询时间偏移：{calibration['tracker_time_offset_ms']:.3f} ms",
            f"- Tracker–相机外参：`{calibration['tracker_camera_extrinsic_path']}`",
            f"- 末端–相机外参：`{calibration['end_camera_extrinsic_path']}`",
            "- 身份与时间链路匹配只说明外参适用于该记录，**不代表物理标定精度已验收**。",
        ]
    else:
        lines.append(f"- 外参不可用：{calibration.get('error', '未知原因')}（同步预检未完成，基础检查仍已执行）")
    lines.append("")
    for status, title in ((STATUS_REJECTED, "质量拒绝"), (STATUS_INCOMPLETE, "预检未完成"),
                          (STATUS_FAILED, "分析失败")):
        chosen = [r for r in records if r["status"] == status]
        lines += [f"## {title}名单（{len(chosen)}）", ""]
        if not chosen:
            lines += ["无。", ""]
            continue
        for record in chosen:
            lines.append(f"- `{record['record_id']}`")
            for reason in reasons_of(record)[:6]:
                lines.append(f"  - {reason}")
            if record.get("plot"):
                lines.append(f"  - 图表：[{record['plot']}]({record['plot']})")
        lines.append("")
    passed = [r["record_id"] for r in records if r["status"] == STATUS_PASSED]
    lines += [f"## 通过名单（{len(passed)}）", ""]
    lines += [", ".join(f"`{item}`" for item in passed) if passed else "无。", ""]
    warning_counts: Counter = Counter()
    warned: Dict[str, List[str]] = {}
    for record in records:
        codes = {w["code"] for w in record.get("warnings", [])}
        for episode in record.get("episodes", []):
            codes |= {w["code"] for w in episode.get("warnings", [])}
        for code in codes:
            warning_counts[code] += 1
            warned.setdefault(code, []).append(record["record_id"])
    lines += ["## 警告汇总", ""]
    if warning_counts:
        lines += ["| 警告 | 记录数 | 示例 |", "|---|---|---|"]
        for code, count in warning_counts.most_common():
            lines.append(f"| `{code}` | {count} | {', '.join(warned[code][:3])} |")
    else:
        lines.append("无警告。")
    lines += ["", "## 图表与代表帧", ""]
    plotted = [r for r in records if r.get("plot")]
    if plotted:
        for record in plotted:
            lines.append(f"- `{record['record_id']}`：[质量曲线]({record['plot']})"
                         + (f"，代表帧 {len(record.get('previews', []))} 张（`{record['previews_dir']}`）"
                            if record.get("previews") else ""))
    else:
        lines.append("本次未生成图表。")
    notes = report.get("notes") or []
    if notes:
        lines += ["", "## 说明", ""] + [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 命令行
def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="analyze_collection_quality.py",
        description="只读分析已保存的 FastUMI 采集记录并输出质量报告（需已加载 ROS2 Jazzy 与 NumPy 1 环境）。",
    )
    parser.add_argument("--input", required=True, help="单条记录、任务目录或数据集根目录")
    parser.add_argument("--output-dir", default="", help="输出目录，必须尚不存在；缺省 dataset/_analysis/<输入名>_<时间>/")
    parser.add_argument("--decode-mode", choices=("full", "sample"), default="full",
                        help="full 解码全部图像；sample 每秒检查一帧并检查异常片段代表帧（默认 full）")
    parser.add_argument("--plots", choices=("issues", "all", "none"), default="issues",
                        help="issues 仅异常记录及最新记录（默认）；all 全部；none 不生成图表和代表帧")
    parser.add_argument("--config", default="", help="覆盖处理配置，规则同 convert_mcap")
    parser.add_argument("--source-camera-fps", type=float, default=None,
                        help="覆盖标称源帧率；不从实测速率推断")
    parser.add_argument("--tracker-camera-extrinsic", default="",
                        help="T_cam_tracker 外参；缺省为仓库 config/calibration 下的默认外参")
    parser.add_argument("--end-camera-extrinsic", default="",
                        help="T_cam_end 外参；缺省为仓库 config/calibration 下的默认外参")
    parser.add_argument("--tracker-serial", default="",
                        help="外参缺少设备绑定时声明标定所用 Tracker 序列号；不能覆盖既有绑定")
    parser.add_argument("--jobs", type=int, default=0,
                        help=f"并行分析的进程数；0（默认）按 CPU 数和记录数自动选择，上限 {MAX_AUTO_JOBS}；"
                             "1 在当前进程串行。每条记录仍在单个进程内顺序处理")
    return parser


def default_output_dir(input_path: Path, stamp: datetime) -> Path:
    """缺省输出目录：输入路径上最近的 ``dataset`` 目录下的 ``_analysis``，否则仓库 ``dataset/_analysis``。"""
    root = REPOSITORY_ROOT / "dataset"
    for parent in [input_path, *input_path.parents]:
        if parent.name == "dataset":
            root = parent
            break
    return root / "_analysis" / f"{input_path.name}_{stamp.strftime('%Y%m%d_%H%M%S_%f')}"


def load_calibration(rt: SimpleNamespace, args: argparse.Namespace) -> Tuple[Any, Optional[Dict[str, Any]], Dict[str, Any]]:
    """加载外参；失败时返回问题条目，使同步预检标记为未完成而基础检查照常执行。"""
    tracker = Path(args.tracker_camera_extrinsic).expanduser() if args.tracker_camera_extrinsic else DEFAULT_TRACKER_EXTRINSIC
    end = Path(args.end_camera_extrinsic).expanduser() if args.end_camera_extrinsic else DEFAULT_END_EXTRINSIC
    try:
        calibration = rt.load_link7_calibration(str(tracker), str(end), args.tracker_serial)
    except (rt.CalibrationError, OSError) as error:
        return None, issue("CALIBRATION_UNAVAILABLE", str(error)), {"available": False, "error": str(error)}
    summary = {
        "available": True,
        "tracker_serial": calibration.tracker_serial,
        "tracker_serial_source": calibration.tracker_serial_source,
        "tracker_time_offset_ms": calibration.tracker_time_offset_ms,
        "tracker_camera_extrinsic_path": calibration.tracker_camera.path,
        "tracker_camera_extrinsic_sha256": calibration.tracker_camera.sha256,
        "end_camera_extrinsic_path": calibration.end_camera.path,
        "end_camera_extrinsic_sha256": calibration.end_camera.sha256,
        "binding_sha256": calibration.binding.sha256 if calibration.binding else None,
        "time_calibration_timestamp_source": (
            calibration.binding.time_calibration_timestamp_source if calibration.binding else None
        ),
        "notes": list(calibration.notes),
    }
    return calibration, None, summary


# ---------------------------------------------------------------- 逐记录处理与并行调度
def resolve_jobs(requested: int, record_count: int) -> int:
    """0 表示按 CPU 数和记录数自动选择（不超过 ``MAX_AUTO_JOBS``）；显式值只受记录数限制。"""
    if requested == 0:
        return max(1, min(os.cpu_count() or 1, record_count, MAX_AUTO_JOBS))
    return max(1, min(requested, record_count))


def process_record(ctx: Context, entry: RecordEntry, plot_name: str, partial_dir: Path) -> Dict[str, Any]:
    """分析一条记录并按图表策略生成图表和代表帧，返回可序列化的报告；错误只影响本记录。"""
    try:
        result = analyze_record(ctx, entry)
    except Exception as error:  # 逐记录隔离：一个坏记录不阻止其余记录。
        return failed_record(entry, error)
    report = result.report
    if wants_plot(ctx.args.plots, report):
        try:
            render_plot(result, partial_dir / "plots" / f"{plot_name}.png")
            report["plot"] = f"plots/{plot_name}.png"
            report["previews"] = render_previews(
                ctx, result, partial_dir / "previews" / plot_name, f"previews/{plot_name}")
            report["previews_dir"] = f"previews/{plot_name}/"
        except Exception as error:
            report.setdefault("warnings", []).append(
                issue("PLOT_FAILED", f"图表或代表帧生成失败: {type(error).__name__}: {error}"))
    return report


# 工作进程内的分析上下文；ROS 函数集合不可序列化，所以每个进程自行加载。
_WORKER_CONTEXT: Optional[Context] = None
# 工作进程使用单线程数值库，避免多进程叠加线程造成过度订阅。
_SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _init_worker(args: argparse.Namespace, explicit_config: Optional[Path], latest_id: str) -> None:
    """工作进程初始化：加载运行时和外参，构造本进程的分析上下文。"""
    global _WORKER_CONTEXT
    import cv2

    cv2.setNumThreads(1)
    rt = load_runtime()
    calibration, calibration_problem, _ = load_calibration(rt, args)
    _WORKER_CONTEXT = Context(rt, args, explicit_config, calibration, calibration_problem, latest_id)


def _analyze_in_worker(entry: RecordEntry, plot_name: str, partial_dir: str) -> Dict[str, Any]:
    """工作进程内处理一条记录。"""
    return process_record(_WORKER_CONTEXT, entry, plot_name, Path(partial_dir))


def _run_task(task: Any, index: int, entry: RecordEntry, plot_name: str, partial_dir: str, active_dir: str) -> Dict[str, Any]:
    """执行任务并在 ``active_dir`` 留下进行中标记；进程异常退出时标记保留，用于定位受影响的记录。"""
    marker = Path(active_dir) / str(index)
    marker.touch()
    try:
        return task(entry, plot_name, partial_dir)
    finally:
        marker.unlink(missing_ok=True)


def _run_pool(
    workers: int, indices: Sequence[int], entries: Sequence[RecordEntry], plot_names: Sequence[str],
    partial_dir: Path, active_dir: Path, initargs: Tuple[Any, ...], task: Any, finish: Any,
) -> bool:
    """在新进程池中分析 ``indices``，每条完成即交给 ``finish``；进程池损坏时返回 True。"""
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker, initargs=initargs,
    )
    broken = False
    try:
        futures = {
            pool.submit(_run_task, task, index, entries[index], plot_names[index], str(partial_dir), str(active_dir)): index
            for index in indices
        }
        for future in as_completed(futures):
            try:
                report = future.result()
            except BrokenProcessPool:
                broken = True
                break
            finish(futures[future], report)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return broken


def analyze_records(
    ctx: Context, entries: Sequence[RecordEntry], plot_names: Sequence[str], partial_dir: Path, jobs: int,
    on_done: Any, task: Any = _analyze_in_worker,
) -> List[Dict[str, Any]]:
    """分析全部记录，返回与 ``entries`` 同序的报告；``on_done(已完成数, 记录, 报告)`` 按完成顺序回调。

    ``jobs`` 为 1 时在当前进程串行；否则每条记录在独立的 spawn 工作进程内顺序处理（H.264 解码状态
    只在一条记录内保留）。工作进程异常退出时，先单独重跑受影响的记录以确定是哪条导致，再继续其余记录。
    """
    reports: List[Optional[Dict[str, Any]]] = [None] * len(entries)
    completed = 0

    def finish(index: int, report: Dict[str, Any]) -> None:
        nonlocal completed
        reports[index] = report
        completed += 1
        on_done(completed, entries[index], report)

    if jobs <= 1:
        for index, entry in enumerate(entries):
            finish(index, process_record(ctx, entry, plot_names[index], partial_dir))
        return reports  # type: ignore[return-value]

    for name in _SINGLE_THREAD_ENV:
        os.environ.setdefault(name, "1")
    initargs = (ctx.args, ctx.explicit_config, ctx.latest_id)
    active_dir = Path(tempfile.mkdtemp(prefix="analyze_active_"))
    try:
        while True:
            unfinished = [i for i, report in enumerate(reports) if report is None]
            if not unfinished:
                break
            if not _run_pool(min(jobs, len(unfinished)), unfinished, entries, plot_names, partial_dir,
                             active_dir, initargs, task, finish):
                continue
            suspects = sorted(int(marker.name) for marker in active_dir.iterdir())
            for marker in active_dir.iterdir():
                marker.unlink()
            if not suspects:  # 进程在领取记录前就退出（例如初始化失败），无法定位具体记录。
                suspects = unfinished
            for index in suspects:
                if reports[index] is not None:
                    continue
                if _run_pool(1, [index], entries, plot_names, partial_dir, active_dir, initargs, task, finish):
                    for marker in active_dir.iterdir():
                        marker.unlink()
                    finish(index, failed_record(
                        entries[index], WorkerCrashedError("分析进程异常退出（可能是解码库崩溃或内存不足）")))
    finally:
        shutil.rmtree(active_dir, ignore_errors=True)
    return reports  # type: ignore[return-value]


def run(argv: Optional[Sequence[str]] = None) -> int:
    """执行分析并返回退出码。"""
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    started = datetime.now().astimezone()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_dir():
        parser.error(f"--input {input_path} 不是目录")
    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir
        else default_output_dir(input_path, started)
    )
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    for existing in (output_dir, partial_dir):
        if existing.exists():
            parser.error(f"输出目录 {existing} 已存在；请指定新的 --output-dir")
    if args.jobs < 0:
        parser.error("--jobs 不能为负数；0 表示自动选择")
    explicit_config = Path(args.config).expanduser() if args.config else None
    if explicit_config is not None and not explicit_config.is_file():
        parser.error(f"--config {explicit_config} 不存在")
    try:
        rt = load_runtime()
        if args.plots != "none":
            importlib.import_module("matplotlib")
        if args.source_camera_fps is not None:
            rt.validate_source_fps(args.source_camera_fps, "--source-camera-fps")
        entries = discover_records(input_path, skip=[output_dir, partial_dir])
    except DependencyError as error:
        print(f"依赖错误：{error}", file=sys.stderr)
        return EXIT_ERROR
    except ImportError as error:
        print(f"依赖错误：绘图需要 matplotlib（{error}）。可安装后重试，或使用 --plots none。", file=sys.stderr)
        return EXIT_ERROR
    except (rt.ConfigError, ValueError, OSError) as error:
        print(f"参数错误：{error}", file=sys.stderr)
        return EXIT_ERROR
    if not entries:
        print(f"{input_path} 下没有发现含 session.yaml 的采集记录", file=sys.stderr)
        return EXIT_ERROR

    calibration, calibration_problem, calibration_summary = load_calibration(rt, args)
    latest = max(entries, key=lambda item: record_sort_key(item.path))
    ctx = Context(rt, args, explicit_config, calibration, calibration_problem, latest.record_id)
    try:
        partial_dir.mkdir(parents=True)
    except OSError as error:
        print(f"无法创建输出目录 {partial_dir}: {error}", file=sys.stderr)
        return EXIT_ERROR
    jobs = resolve_jobs(args.jobs, len(entries))
    print(f"发现 {len(entries)} 条记录；解码模式 {args.decode_mode}；并行进程 {jobs}；输出 {output_dir}")
    if calibration is None:
        print(f"提示：外参不可用，同步预检将标记为未完成：{calibration_problem['message']}")

    plot_names: List[str] = []
    used_names: Counter = Counter()
    for entry in entries:
        name = safe_name(entry.record_id)
        used_names[name] += 1
        plot_names.append(name if used_names[name] == 1 else f"{name}_{used_names[name]}")

    def report_progress(number: int, entry: RecordEntry, report: Dict[str, Any]) -> None:
        print(f"[{number}/{len(entries)}] {entry.record_id}: {STATUS_LABELS[report['status']]}"
              + (f"（{report['error']['type']}: {report['error']['message']}）" if report.get("error") else ""))

    try:
        records = analyze_records(ctx, entries, plot_names, partial_dir, jobs, report_progress)
        code = exit_code_for(records)
        document = {
            "tool": "analyze_collection_quality.py",
            "generated_at": started.isoformat(),
            "input": str(input_path),
            "output_dir": str(output_dir),
            "arguments": list(sys.argv[1:] if argv is None else argv),
            "decode_mode": args.decode_mode,
            "plots": args.plots,
            "exit_code": code,
            "runtime": {"shared_logic_source": rt.shared_source, "python": sys.version.split()[0]},
            "calibration": calibration_summary,
            "totals": batch_totals(records),
            "lists": {
                status: [r["record_id"] for r in records if r["status"] == status]
                for status in (STATUS_PASSED, STATUS_REJECTED, STATUS_INCOMPLETE, STATUS_FAILED)
            },
            "records": records,
            "notes": [
                "身份与时间链路匹配不代表物理标定精度已验收。",
                "非有限或不可计算的统计量在 JSON 中写为 null，原因见同级 reasons 字段或 null_paths。",
            ],
        }
        text, replaced = dumps_strict(document)
        if replaced:
            document["null_paths"] = replaced
            text, _ = dumps_strict(document)
        write_csv(partial_dir / "summary.csv", records)
        atomic_write(partial_dir / "report.json", text)
        atomic_write(partial_dir / "report.md", render_markdown(to_jsonable(document)))
        os.rename(partial_dir, output_dir)
    except Exception as error:
        print(f"报告生成错误：{type(error).__name__}: {error}\n未完成的输出保留在 {partial_dir}", file=sys.stderr)
        return EXIT_ERROR
    totals = document["totals"]
    print(
        f"完成：通过 {totals['passed']}，质量拒绝 {totals['rejected']}，预检未完成 {totals['incomplete']}，"
        f"分析失败 {totals['failed']}；报告 {output_dir / 'report.md'}"
    )
    return code


def main(argv: Optional[Sequence[str]] = None) -> None:
    """命令行入口。"""
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
