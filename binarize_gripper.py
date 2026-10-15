#!/usr/bin/env python3
"""离线把 FastUMI 录制中的夹爪开合度二值化，写出带二值 ``filtered_openness`` 的副本。

每条录制按一次完整的 **开→闭→开** 处理：用 ``raw_openness`` 经居中中值平滑后搜索三个
连续平台，再以平台中点判定每帧状态，最后输出整条 ``1.0``（开）/ ``0.0``（闭）序列。
时间取 ``header.stamp``，所有时间比较使用整数纳秒。算法细节见各函数 docstring。

只有通过判定的录制才发布完整副本；待核对的录制只生成对照图和报告。原始数据不会被修改，
也不允许覆盖已有输出。副本的约定变化：

* 只改写 ``/gripper/state`` 的 ``filtered_openness``（二值）和 ``valid``；
  ``raw_openness``、标签数量、标签距离、完整 header 和 bag 时间原样保留。
* 副本中 ``valid=True`` 的含义是“该帧二值状态已确定”，不再表示“两枚标签及 PnP 有效”。
  补齐帧的 ``raw_openness`` 仍可能是 NaN；每帧的补值来源写在副本的 ``binarization.yaml``。
* 其余话题按原顺序、原时间、原始序列化字节复制；``quality.yaml`` 保留采集时的诊断。

必须在已加载 ROS2 Jazzy、工作空间和 NumPy 1 环境的终端运行::

    python binarize_gripper.py --input ros2_ws/dataset/umi_jingbao \\
        --output-dir ros2_ws/dataset/umi_jingbao_binary

输出目录的 ``_analysis`` 下有 ``binarization_report.json``、``binarization_report.csv`` 和
逐条录制的对照图 ``plots/*.png``。ROS 依赖在处理开始时才加载，``--help`` 不需要 ROS。

退出码：0 全部通过；1 存在待核对项；2 存在依赖、读取或写入错误。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    import yaml
except ImportError:  # 仅影响 --help 之外的处理；load_runtime 会给出环境说明。
    np = None
    yaml = None

REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from analyze_collection_quality import (  # noqa: E402
    ENVIRONMENT_HINT,
    AnalysisError,
    DependencyError,
    RecordEntry,
    atomic_write,
    discover_records,
    load_yaml_mapping,
    read_bag_metadata,
)

EXIT_OK = 0
EXIT_REVIEW = 1
EXIT_ERROR = 2

GRIPPER_TYPE = "fastumi_interfaces/msg/GripperState"
DEFAULT_GRIPPER_TOPIC = "/gripper/state"
DEFAULT_BAG_URI = "raw/bag"
ANALYSIS_DIR = "_analysis"
STAGING_DIR = ".staging"
BINARIZATION_FILE = "binarization.yaml"

# 算法门槛（纳秒为整数）。
SMOOTH_HALF_WINDOW_NS = 100_000_000   # 居中中值窗口 [t-0.1 s, t+0.1 s]，含两端。
MAX_MISSING_NS = 500_000_000          # 首部、内部、尾部缺失的时间跨度上限。
MIN_PLATEAU_NS = 500_000_000          # 每个平台首尾帧的最短时间跨度。
MIN_STATE_NS = 500_000_000            # 持续状态区间的最短时间。
MIN_PLATEAU_DIFF = 0.045              # 前开/后开平台与闭合平台的最小差值。
# 浮点等值边界的容差：恰好等于门槛或中点的值按“达到”处理。
FLOAT_TOLERANCE = 1.0e-9

STATUS_PASSED = "passed"
STATUS_REVIEW = "review"
STATUS_ERROR = "error"
STATUS_LABELS = {STATUS_PASSED: "通过", STATUS_REVIEW: "待核对", STATUS_ERROR: "错误"}

STATE_OPEN, STATE_CLOSED = 1, 0
FILL_INTERNAL, FILL_LEADING, FILL_TRAILING = "internal", "leading", "trailing"
FILL_METHODS = {
    FILL_INTERNAL: "linear_interpolation",
    FILL_LEADING: "nearest_valid_sample",
    FILL_TRAILING: "nearest_valid_sample",
}


def issue(code: str, message: str, **details: Any) -> Dict[str, Any]:
    """构造报告中的结构化问题条目。"""
    item: Dict[str, Any] = {"code": code, "message": message}
    item.update(details)
    return item


# ---------------------------------------------------------------- 纯算法
@dataclass
class Binarization:
    """一条录制的二值化过程和结果；失败时保留已算出的候选结果供对照图使用。"""

    stamps_ns: Any
    raw: Any
    usable: Any
    reasons: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    filled: Any = None            # 补齐后的开度，仅用于分析。
    fill_kind: List[Optional[str]] = field(default_factory=list)
    smoothed: Any = None
    plateau_bounds: Optional[Tuple[int, int]] = None  # 三段为 [0,i)、[i,j)、[j,n)。
    plateau_values: Optional[Tuple[float, float, float]] = None
    minimum_index: Optional[int] = None    # 中间平台首次最低值所在帧，归入下降侧。
    thresholds: Optional[Tuple[float, float]] = None  # (下降侧中点, 回升侧中点)。
    provisional: Any = None       # 合并前的逐帧临时状态。
    states: Any = None            # 合并后的逐帧状态，1 开 0 闭。
    merged_runs: List[Dict[str, Any]] = field(default_factory=list)
    close_index: Optional[int] = None
    open_index: Optional[int] = None

    @property
    def passed(self) -> bool:
        """没有任何待核对原因。"""
        return not self.reasons

    @property
    def binary(self) -> Any:
        """逐帧二值序列（开 1.0，闭 0.0）；尚无候选结果时为 ``None``。"""
        return None if self.states is None else self.states.astype(np.float64)


def find_missing_gaps(stamps_ns: Any, usable: Any) -> List[Dict[str, Any]]:
    """列出每段连续缺失及其时间跨度（纳秒）。

    内部缺失取前后有效样本的时间差；首部从录制第一帧算到第一个有效样本，尾部从最后一个
    有效样本算到录制最后一帧。要求至少有一个有效样本。
    """
    valid_index = np.flatnonzero(usable)
    first, last = int(valid_index[0]), int(valid_index[-1])
    gaps: List[Dict[str, Any]] = []
    if first > 0:
        gaps.append({"kind": FILL_LEADING, "first": 0, "last": first - 1,
                     "span_ns": int(stamps_ns[first] - stamps_ns[0])})
    for before, after in zip(valid_index[:-1], valid_index[1:]):
        if after - before > 1:
            gaps.append({"kind": FILL_INTERNAL, "first": int(before) + 1, "last": int(after) - 1,
                         "span_ns": int(stamps_ns[after] - stamps_ns[before])})
    if last < len(stamps_ns) - 1:
        gaps.append({"kind": FILL_TRAILING, "first": last + 1, "last": len(stamps_ns) - 1,
                     "span_ns": int(stamps_ns[-1] - stamps_ns[last])})
    return gaps


def fill_missing(stamps_ns: Any, raw: Any, usable: Any, gaps: Sequence[Dict[str, Any]]) -> Tuple[Any, List[Optional[str]]]:
    """内部缺失按时间线性补齐，首尾缺失取最近的有效值；返回补齐值和每帧补值类型。"""
    times = stamps_ns.astype(np.float64)
    filled = np.interp(times, times[usable], raw[usable])
    kinds: List[Optional[str]] = [None] * len(stamps_ns)
    for gap in gaps:
        for index in range(gap["first"], gap["last"] + 1):
            kinds[index] = gap["kind"]
    return filled, kinds


def median_smooth(stamps_ns: Any, values: Any, half_window_ns: int = SMOOTH_HALF_WINDOW_NS) -> Any:
    """居中时间窗中值平滑，窗口 ``[t-half, t+half]`` 含两端；不引入相位滞后。"""
    low = np.searchsorted(stamps_ns, stamps_ns - half_window_ns, side="left")
    high = np.searchsorted(stamps_ns, stamps_ns + half_window_ns, side="right")
    return np.asarray([np.median(values[a:b]) for a, b in zip(low, high)], dtype=np.float64)


def fit_three_plateaus(stamps_ns: Any, values: Any, min_span_ns: int = MIN_PLATEAU_NS) -> Optional[Tuple[int, int]]:
    """把序列分成三段连续平台 ``[0,i)``、``[i,j)``、``[j,n)``，返回 ``(i, j)``。

    每段首尾帧的时间跨度至少 ``min_span_ns``，在此约束下取三段各自均值的平方误差之和最小
    的分法；误差相同（容差 1e-9）时依次取较早的第一、第二分界。无可行分法时返回 ``None``。
    """
    count = len(values)
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(values * values)))

    def sse(start: Any, end: Any) -> Any:
        total = prefix[end] - prefix[start]
        return (prefix_sq[end] - prefix_sq[start]) - total * total / (end - start)

    first_min = int(np.searchsorted(stamps_ns, stamps_ns[0] + min_span_ns, side="left")) + 1
    last_max = int(np.searchsorted(stamps_ns, stamps_ns[-1] - min_span_ns, side="right")) - 1
    best: Optional[Tuple[float, int, int]] = None
    for i in range(first_min, count):
        second_min = int(np.searchsorted(stamps_ns, stamps_ns[i] + min_span_ns, side="left")) + 1
        if second_min > last_max:
            continue
        ends = np.arange(second_min, last_max + 1)
        cost = sse(0, i) + sse(i, ends) + sse(ends, count)
        pick = int(np.flatnonzero(cost <= cost.min() + FLOAT_TOLERANCE)[0])
        if best is None or cost[pick] < best[0] - FLOAT_TOLERANCE:
            best = (float(cost[pick]), i, int(ends[pick]))
    return None if best is None else (best[1], best[2])


def classify_states(smoothed: Any, minimum_index: int, thresholds: Tuple[float, float]) -> Any:
    """按平台中点给每帧判定临时状态。

    ``minimum_index`` 及以前为下降侧，开度小于或等于下降侧中点判为闭；之后为回升侧，开度
    大于或等于回升侧中点判为开。其余帧取相反状态。
    """
    descent, rise = thresholds
    states = np.empty(len(smoothed), dtype=np.int8)
    index = np.arange(len(smoothed))
    states[:] = np.where(
        index <= minimum_index,
        np.where(smoothed <= descent + FLOAT_TOLERANCE, STATE_CLOSED, STATE_OPEN),
        np.where(smoothed >= rise - FLOAT_TOLERANCE, STATE_OPEN, STATE_CLOSED),
    )
    return states


def split_runs(states: Any) -> List[Tuple[int, int, int]]:
    """把状态序列切成连续同值区间，返回 ``(状态, 起始帧, 结束帧不含)`` 列表。"""
    boundaries = np.flatnonzero(np.diff(states)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(states)]))
    return [(int(states[a]), int(a), int(b)) for a, b in zip(starts, ends)]


def run_durations(stamps_ns: Any, runs: Sequence[Tuple[int, int, int]]) -> List[int]:
    """每个区间的持续纳秒：从它的第一帧算到下一区间的第一帧，最后一个区间算到最后一帧。"""
    return [
        int(stamps_ns[runs[k + 1][1]] - stamps_ns[runs[k][1]]) if k + 1 < len(runs)
        else int(stamps_ns[-1] - stamps_ns[runs[k][1]])
        for k in range(len(runs))
    ]


def merge_short_runs(stamps_ns: Any, states: Any, min_state_ns: int = MIN_STATE_NS) -> Tuple[Any, List[Dict[str, Any]]]:
    """把不足 ``min_state_ns`` 的区间并入相邻持续区间的状态，返回新状态序列和并入明细。

    区间持续时间从它的第一帧算到下一区间的第一帧，最后一个区间算到录制最后一帧。短区间并入
    前一个持续区间；位于开头的短区间并入其后第一个持续区间。没有持续区间时返回的状态与
    输入相同，明细为空，由调用方列为待核对。
    """
    runs = split_runs(states)
    durations = run_durations(stamps_ns, runs)
    durable = [k for k, duration in enumerate(durations) if duration >= min_state_ns]
    if not durable:
        return states.copy(), []
    merged = states.copy()
    details: List[Dict[str, Any]] = []
    for k, (state, start, end) in enumerate(runs):
        if durations[k] >= min_state_ns:
            continue
        earlier = [d for d in durable if d < k]
        target = earlier[-1] if earlier else next(d for d in durable if d > k)
        merged[start:end] = runs[target][0]
        details.append({"first_frame": start, "last_frame": end - 1, "duration_ns": durations[k],
                        "from_state": state, "to_state": runs[target][0]})
    return merged, details


def binarize_openness(stamps_ns: Any, raw: Any, valid: Any) -> Binarization:
    """对一条录制的夹爪开度做开→闭→开二值化。

    Args:
        stamps_ns: ``header.stamp`` 的 Unix 纳秒，int64，形状 ``(N,)``，须严格递增。
        raw: ``raw_openness``，无量纲归一化开度，形状 ``(N,)``，无效时可为 NaN。
        valid: 原始 ``valid`` 标志，形状 ``(N,)``。

    Returns:
        ``Binarization``；``reasons`` 为空表示通过。失败时已算出的阶段结果仍保留为候选。
    """
    stamps_ns = np.asarray(stamps_ns, dtype=np.int64)
    raw = np.asarray(raw, dtype=np.float64)
    usable = np.asarray(valid, dtype=bool) & np.isfinite(raw)
    result = Binarization(stamps_ns=stamps_ns, raw=raw, usable=usable)
    if len(stamps_ns) == 0:
        result.reasons.append(issue("no_gripper_samples", "录制中没有夹爪消息"))
        return result
    if np.any(np.diff(stamps_ns) <= 0):
        result.reasons.append(issue("timestamps_not_increasing", "夹爪 header 时间戳不严格递增"))
        return result
    if not usable.any():
        result.reasons.append(issue("all_samples_invalid", "全部夹爪样本无效"))
        return result

    result.gaps = find_missing_gaps(stamps_ns, usable)
    for gap in result.gaps:
        if gap["span_ns"] > MAX_MISSING_NS:
            result.reasons.append(issue(
                "missing_gap_too_long",
                f"{gap['kind']} 缺失跨度 {gap['span_ns'] / 1e9:.3f} s 超过 {MAX_MISSING_NS / 1e9:.1f} s",
                gap_kind=gap["kind"], first_frame=gap["first"], last_frame=gap["last"],
                span_s=gap["span_ns"] / 1e9))
    result.filled, result.fill_kind = fill_missing(stamps_ns, raw, usable, result.gaps)
    result.smoothed = median_smooth(stamps_ns, result.filled)

    bounds = fit_three_plateaus(stamps_ns, result.smoothed)
    if bounds is None:
        result.reasons.append(issue(
            "no_three_plateaus", f"找不到三个时间跨度至少 {MIN_PLATEAU_NS / 1e9:.1f} s 的连续平台"))
        return result
    result.plateau_bounds = bounds
    first, second = bounds
    smoothed = result.smoothed
    before, closed, after = (
        float(np.median(smoothed[:first])), float(np.median(smoothed[first:second])),
        float(np.median(smoothed[second:])))
    result.plateau_values = (before, closed, after)
    drop, rise = before - closed, after - closed
    if drop < MIN_PLATEAU_DIFF - FLOAT_TOLERANCE or rise < MIN_PLATEAU_DIFF - FLOAT_TOLERANCE:
        result.reasons.append(issue(
            "plateau_difference_too_small",
            f"平台差不足 {MIN_PLATEAU_DIFF}：下降 {drop:.4f}，回升 {rise:.4f}",
            drop=drop, rise=rise))

    result.minimum_index = first + int(np.argmin(smoothed[first:second]))
    result.thresholds = ((before + closed) / 2.0, (closed + after) / 2.0)
    result.provisional = classify_states(smoothed, result.minimum_index, result.thresholds)
    result.states, result.merged_runs = merge_short_runs(stamps_ns, result.provisional)
    runs = split_runs(result.states)
    pattern = [run[0] for run in runs]
    provisional_runs = split_runs(result.provisional)
    if not any(d >= MIN_STATE_NS for d in run_durations(stamps_ns, provisional_runs)):
        result.reasons.append(issue("no_durable_state", "没有持续时间足够的状态区间"))
    elif pattern != [STATE_OPEN, STATE_CLOSED, STATE_OPEN]:
        result.reasons.append(issue(
            "unexpected_state_pattern", f"持续状态序列为 {pattern}，不是恰好一次开→闭→开",
            pattern=pattern))
    else:
        result.close_index, result.open_index = runs[1][1], runs[2][1]
    return result


# ---------------------------------------------------------------- 运行时依赖
def load_runtime() -> SimpleNamespace:
    """延迟加载 rosbag2 和消息类型。

    Raises:
        DependencyError: 缺少 NumPy/PyYAML 或 ROS 依赖；附环境准备说明。
    """
    if np is None or yaml is None:
        raise DependencyError("缺少 NumPy 或 PyYAML。\n" + ENVIRONMENT_HINT)
    try:
        import rosbag2_py
        from fastumi_interfaces.msg import GripperState
        from rclpy.serialization import deserialize_message, serialize_message
    except ImportError as error:
        raise DependencyError(f"无法加载运行依赖：{error}\n" + ENVIRONMENT_HINT) from error
    return SimpleNamespace(
        rosbag2_py=rosbag2_py, GripperState=GripperState,
        deserialize_message=deserialize_message, serialize_message=serialize_message)


# ---------------------------------------------------------------- 读取与写入 bag
def open_reader(rt: SimpleNamespace, bag_uri: Path, topics: Optional[Sequence[str]] = None) -> Any:
    """以 MCAP 存储打开 rosbag2，可选只读取指定话题。"""
    reader = rt.rosbag2_py.SequentialReader()
    reader.open(rt.rosbag2_py.StorageOptions(uri=str(bag_uri), storage_id="mcap"),
                rt.rosbag2_py.ConverterOptions("", ""))
    if topics:
        reader.set_filter(rt.rosbag2_py.StorageFilter(topics=list(topics)))
    return reader


def stamp_ns_of(message: Any) -> int:
    """Header 时间转 Unix 纳秒。"""
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def read_gripper_samples(rt: SimpleNamespace, bag_uri: Path, topic: str) -> Tuple[Any, Any, Any]:
    """按 bag 顺序读取夹爪消息，返回 ``(header 纳秒, raw_openness, valid)`` 三个数组。

    Raises:
        AnalysisError: 话题缺失、类型不是 ``GripperState`` 或消息无法解析。
    """
    types = {item.name: item.type for item in open_reader(rt, bag_uri).get_all_topics_and_types()}
    if topic not in types:
        raise AnalysisError(f"{bag_uri} 缺少夹爪话题 {topic}")
    if types[topic] != GRIPPER_TYPE:
        raise AnalysisError(f"{topic} 类型为 {types[topic]}，需要 {GRIPPER_TYPE}")
    reader = open_reader(rt, bag_uri, [topic])
    stamps: List[int] = []
    raw: List[float] = []
    valid: List[bool] = []
    while reader.has_next():
        _, data, _ = reader.read_next()
        try:
            message = rt.deserialize_message(bytes(data), rt.GripperState)
        except Exception as error:  # 单条损坏的消息使整条录制无法判定。
            raise AnalysisError(f"{topic} 消息无法解析：{type(error).__name__}: {error}") from error
        stamps.append(stamp_ns_of(message))
        raw.append(float(message.raw_openness))
        valid.append(bool(message.valid))
    return np.asarray(stamps, dtype=np.int64), np.asarray(raw, dtype=np.float64), np.asarray(valid, dtype=bool)


def write_binary_bag(rt: SimpleNamespace, source_bag: Path, output_bag: Path, topic: str, binary: Any) -> Dict[str, int]:
    """复制源 bag 到不存在的 ``output_bag``，只把夹爪的 ``filtered_openness`` 改为二值、``valid`` 置真。

    其余消息保持话题、原始序列化字节和 bag 时间不变，话题元数据（含 QoS）原样复制。
    返回各话题写入的消息数量。

    Raises:
        AnalysisError: 夹爪消息数量与二值序列长度不一致。
    """
    reader = open_reader(rt, source_bag)
    writer = rt.rosbag2_py.SequentialWriter()
    writer.open(rt.rosbag2_py.StorageOptions(uri=str(output_bag), storage_id="mcap",
                                             storage_preset_profile="zstd_fast"),
                rt.rosbag2_py.ConverterOptions("", ""))
    for metadata in reader.get_all_topics_and_types():
        writer.create_topic(metadata)
    counts: Dict[str, int] = {}
    ordinal = 0
    while reader.has_next():
        name, data, bag_ns = reader.read_next()
        data = bytes(data)
        if name == topic:
            if ordinal >= len(binary):
                raise AnalysisError("夹爪消息数量多于二值序列")
            message = rt.deserialize_message(data, rt.GripperState)
            message.filtered_openness = float(binary[ordinal])
            message.valid = True
            data = bytes(rt.serialize_message(message))
            ordinal += 1
        writer.write(name, data, int(bag_ns))
        counts[name] = counts.get(name, 0) + 1
    if ordinal != len(binary):
        raise AnalysisError("夹爪消息数量少于二值序列")
    del writer  # 析构时写出 MCAP 索引和 metadata.yaml。
    return counts


def nan_equal(a: float, b: float) -> bool:
    """float32 数值相等，NaN 视为相等。"""
    return (math.isnan(a) and math.isnan(b)) or a == b


def verify_copy(rt: SimpleNamespace, source_bag: Path, output_bag: Path, topic: str, binary: Any) -> None:
    """流式核对副本：非夹爪消息逐字节一致，夹爪保留字段一致且为二值，数量与元数据一致。

    Raises:
        AnalysisError: 任一项不一致。
    """
    source, output = open_reader(rt, source_bag), open_reader(rt, output_bag)
    ordinal = 0
    while source.has_next():
        if not output.has_next():
            raise AnalysisError("副本缺少消息")
        s_topic, s_data, s_ns = source.read_next()
        o_topic, o_data, o_ns = output.read_next()
        if (s_topic, int(s_ns)) != (o_topic, int(o_ns)):
            raise AnalysisError(f"副本消息顺序或时间不一致：{s_topic}@{s_ns} / {o_topic}@{o_ns}")
        if s_topic != topic:
            if bytes(s_data) != bytes(o_data):
                raise AnalysisError(f"副本未逐字节保留 {s_topic}")
            continue
        old = rt.deserialize_message(bytes(s_data), rt.GripperState)
        new = rt.deserialize_message(bytes(o_data), rt.GripperState)
        same = (
            old.header == new.header
            and nan_equal(old.raw_openness, new.raw_openness)
            and nan_equal(old.marker_distance_mm, new.marker_distance_mm)
            and old.detected_marker_count == new.detected_marker_count
        )
        if not same:
            raise AnalysisError(f"夹爪第 {ordinal} 帧的保留字段发生变化")
        if new.filtered_openness != binary[ordinal] or new.filtered_openness not in (0.0, 1.0) or not new.valid:
            raise AnalysisError(f"夹爪第 {ordinal} 帧不是确定的二值状态")
        ordinal += 1
    if output.has_next():
        raise AnalysisError("副本多出消息")
    if ordinal != len(binary):
        raise AnalysisError("副本夹爪消息数量不一致")
    before, after = read_bag_metadata(source_bag), read_bag_metadata(output_bag)
    # Writer 可把多个源分包合并；仅核对与文件布局无关的计数。
    if before is None or after is None or (
        before["message_count"] != after["message_count"]
        or before["topics"] != after["topics"]
        or sum(before["files"]) != sum(after["files"])
    ):
        raise AnalysisError(f"副本 metadata.yaml 的消息计数与原 bag 不一致：{before} / {after}")


# ---------------------------------------------------------------- 报告
def to_jsonable(value: Any) -> Any:
    """把 NumPy 标量、数组和非有限浮点转成可严格序列化的 JSON 值。"""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_summary(record_id: str, result: Optional[Binarization], status: str, error: str = "") -> Dict[str, Any]:
    """生成一条录制的汇总字典（JSON/CSV 共用）。"""
    summary: Dict[str, Any] = {"record_id": record_id, "status": status, "reasons": [], "error": error}
    if result is None:
        return summary
    stamps = result.stamps_ns
    origin = int(stamps[0]) if len(stamps) else 0
    filled = [i for i, kind in enumerate(result.fill_kind) if kind]
    summary.update({
        "reasons": result.reasons,
        "frames": int(len(stamps)),
        "valid_frames": int(result.usable.sum()),
        "filled_frames": len(filled),
        "merged_short_runs": len(result.merged_runs),
        "gaps": result.gaps,
    })
    if result.plateau_values is not None:
        before, closed, after = result.plateau_values
        summary.update({"open_before": before, "closed": closed, "open_after": after,
                        "drop": before - closed, "rise": after - closed})
    if result.close_index is not None:
        close_ns, open_ns = int(stamps[result.close_index]), int(stamps[result.open_index])
        summary.update({"close_stamp_ns": close_ns, "open_stamp_ns": open_ns,
                        "close_time_s": (close_ns - origin) / 1e9, "open_time_s": (open_ns - origin) / 1e9,
                        "closed_duration_s": (open_ns - close_ns) / 1e9})
    return summary


def build_binarization_document(record_id: str, topic: str, result: Binarization, original_valid: Any) -> Dict[str, Any]:
    """副本中的 ``binarization.yaml``：参数、平台值、切换时刻、各帧原有效性和补值位置。"""
    stamps = result.stamps_ns
    origin = int(stamps[0])
    before, closed, after = result.plateau_values
    filled = [
        {"frame": i, "stamp_ns": int(stamps[i]), "kind": kind, "method": FILL_METHODS[kind],
         "original_valid": bool(original_valid[i]), "filled_value": float(result.filled[i])}
        for i, kind in enumerate(result.fill_kind) if kind
    ]
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_record": record_id,
        "topic": topic,
        "input_field": "raw_openness",
        "rewritten_fields": ["filtered_openness", "valid"],
        "valid_semantics": "该帧二值状态已确定；不再表示两枚标签及 PnP 有效，补齐帧的 raw_openness 可能为 NaN",
        "parameters": {
            "smooth_half_window_ns": SMOOTH_HALF_WINDOW_NS,
            "max_missing_ns": MAX_MISSING_NS,
            "min_plateau_ns": MIN_PLATEAU_NS,
            "min_state_ns": MIN_STATE_NS,
            "min_plateau_difference": MIN_PLATEAU_DIFF,
        },
        "plateaus": {"open_before": before, "closed": closed, "open_after": after,
                     "drop": before - closed, "rise": after - closed,
                     "boundary_frames": list(result.plateau_bounds),
                     "thresholds": list(result.thresholds)},
        "switch": {
            "close_frame": result.close_index, "close_stamp_ns": int(stamps[result.close_index]),
            "close_time_s": (int(stamps[result.close_index]) - origin) / 1e9,
            "open_frame": result.open_index, "open_stamp_ns": int(stamps[result.open_index]),
            "open_time_s": (int(stamps[result.open_index]) - origin) / 1e9,
        },
        "merged_short_runs": result.merged_runs,
        "frame_count": int(len(stamps)),
        "original_valid": "".join("1" if flag else "0" for flag in original_valid),
        "filled_frame_count": len(filled),
        "filled_frames": filled,
    }


def render_plot(record_id: str, status: str, result: Optional[Binarization], path: Path) -> None:
    """对照图：原始曲线、平滑曲线和二值结果（未通过的画候选结果）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 4.5))
    title = f"{record_id}  [{status}]"
    if result is not None and len(result.stamps_ns):
        seconds = (result.stamps_ns - result.stamps_ns[0]) / 1e9
        axis.plot(seconds[result.usable], result.raw[result.usable], ".", color="0.55", markersize=3, label="raw (valid)")
        if (~result.usable).any():
            axis.plot(seconds[~result.usable], np.full(int((~result.usable).sum()), -0.03), "r|", label="missing (filled)")
        if result.smoothed is not None:
            axis.plot(seconds, result.smoothed, color="tab:blue", linewidth=1.2, label="median-smoothed")
        if result.thresholds is not None:
            for level in result.thresholds:
                axis.axhline(level, color="tab:orange", linestyle=":", linewidth=0.8)
        if result.binary is not None:
            twin = axis.twinx()
            twin.step(seconds, result.binary, where="post", color="tab:green", linewidth=1.6, label="binary (1=open)")
            twin.set_ylim(-0.1, 1.6)
            twin.set_ylabel("binary state")
            twin.legend(loc="upper right")
        for index in (result.close_index, result.open_index):
            if index is not None:
                axis.axvline(seconds[index], color="tab:red", linestyle="--", linewidth=0.8)
        if result.reasons:
            title += "  " + "; ".join(item["code"] for item in result.reasons)
        axis.legend(loc="upper left")
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("time since first gripper message (s)")
    axis.set_ylabel("raw_openness")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


CSV_COLUMNS = (
    "record_id", "status", "reason_codes", "frames", "valid_frames", "filled_frames", "merged_short_runs",
    "open_before", "closed", "open_after", "drop", "rise", "close_time_s", "open_time_s", "closed_duration_s",
    "published", "error",
)


def write_reports(analysis_dir: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    """写 JSON 与 CSV 汇总到 ``analysis_dir``。"""
    counts = {key: sum(1 for item in summaries if item["status"] == key) for key in STATUS_LABELS}
    document = {"generated_at": datetime.now(timezone.utc).isoformat(), "totals": counts,
                "records": to_jsonable(list(summaries))}
    atomic_write(analysis_dir / "binarization_report.json",
                 json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    path = analysis_dir / "binarization_report.csv"
    with open(path.with_name(path.name + ".tmp"), "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for item in summaries:
            row = {key: item.get(key) for key in CSV_COLUMNS}
            row["reason_codes"] = ";".join(reason["code"] for reason in item["reasons"])
            writer.writerow(row)
    os.replace(path.with_name(path.name + ".tmp"), path)


# ---------------------------------------------------------------- 单条录制与批处理
def safe_name(text: str) -> str:
    """记录标识转文件名。"""
    return text.replace("/", "__").replace(" ", "_")


def directory_size_bytes(path: Path) -> int:
    """统计目录内常规文件字节数，忽略符号链接。"""
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file() and not child.is_symlink())


def update_session_size(record_dir: Path) -> None:
    """把副本 ``session.yaml`` 的 ``size_bytes`` 更新为目录实际大小（改写本文件会影响大小，迭代到稳定）。"""
    path = record_dir / "session.yaml"
    session = load_yaml_mapping(path)
    for _ in range(5):
        size = directory_size_bytes(record_dir)
        if session.get("size_bytes") == size:
            return
        session["size_bytes"] = size
        atomic_write(path, yaml.safe_dump(session, allow_unicode=True, sort_keys=False))
    raise AnalysisError("session.yaml 的 size_bytes 未能稳定")


def build_copy(rt: SimpleNamespace, entry: RecordEntry, bag_rel: str, topic: str, result: Binarization,
               original_valid: Any, staging: Path) -> None:
    """在不存在的 ``staging`` 目录构建完整副本并校验；失败时抛出异常由调用方清理。"""
    bag_source = entry.path / bag_rel
    ignored = bag_source.resolve()
    shutil.copytree(entry.path, staging, symlinks=True,
                    ignore=lambda directory, names: [n for n in names if (Path(directory) / n).resolve() == ignored])
    bag_output = staging / bag_rel
    binary = result.binary
    write_binary_bag(rt, bag_source, bag_output, topic, binary)
    verify_copy(rt, bag_source, bag_output, topic, binary)
    document = build_binarization_document(entry.record_id, topic, result, original_valid)
    atomic_write(staging / BINARIZATION_FILE,
                 yaml.safe_dump(to_jsonable(document), allow_unicode=True, sort_keys=False))
    update_session_size(staging)


def process_record(rt: SimpleNamespace, entry: RecordEntry, output: Path, analysis_dir: Path) -> Dict[str, Any]:
    """处理一条录制：判定、画图，通过则构建、校验并原子发布副本。读取或写入错误记为 ``error``。"""
    result: Optional[Binarization] = None
    staging = output / STAGING_DIR / safe_name(entry.record_id)
    try:
        session = load_yaml_mapping(entry.path / "session.yaml")
        topic = str((session.get("topic_map") or {}).get("gripper_state") or DEFAULT_GRIPPER_TOPIC)
        bag_rel = str(session.get("bag_uri") or DEFAULT_BAG_URI)
        stamps, raw, valid = read_gripper_samples(rt, entry.path / bag_rel, topic)
        result = binarize_openness(stamps, raw, valid)
        status = STATUS_PASSED if result.passed else STATUS_REVIEW
        summary = build_summary(entry.record_id, result, status)
        summary["published"] = False
        render_plot(entry.record_id, status, result, analysis_dir / "plots" / f"{safe_name(entry.record_id)}.png")
        if result.passed:
            final = output / entry.record_id
            staging.parent.mkdir(parents=True, exist_ok=True)
            build_copy(rt, entry, bag_rel, topic, result, valid, staging)
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, final)
            summary["published"] = True
        return summary
    except Exception as error:  # 单条录制出错不影响其余录制，也不发布残缺副本。
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        detail = f"{type(error).__name__}: {error}"
        print(traceback.format_exc(), file=sys.stderr)
        summary = build_summary(entry.record_id, result, STATUS_ERROR, detail)
        summary["published"] = False
        return summary


def check_paths(input_path: Path, output_path: Path) -> Tuple[Path, Path]:
    """校验输入存在、输出尚不存在且两者不重叠，返回解析后的路径。

    Raises:
        ValueError: 路径不满足要求。
    """
    source, target = input_path.expanduser().resolve(), output_path.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"输入 {source} 不是目录")
    if target.exists() or target.is_symlink():
        raise ValueError(f"输出目录 {target} 已存在，不允许覆盖")
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("输入与输出路径不能重叠")
    return source, target


def exit_code_for(summaries: Sequence[Dict[str, Any]]) -> int:
    """全部通过 0；存在待核对项 1；存在错误 2（错误优先）。"""
    statuses = {item["status"] for item in summaries}
    if STATUS_ERROR in statuses:
        return EXIT_ERROR
    return EXIT_REVIEW if STATUS_REVIEW in statuses else EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    parser = argparse.ArgumentParser(
        description="把 FastUMI 录制的夹爪开合度二值化（开→闭→开），写出带二值 filtered_openness 的副本。",
        epilog="退出码：0 全部通过；1 存在待核对项；2 存在依赖、读取或写入错误。")
    parser.add_argument("--input", required=True, type=Path, help="任务目录或单条录制目录")
    parser.add_argument("--output-dir", required=True, type=Path, help="尚不存在的输出目录")
    return parser


def run(argv: Optional[Sequence[str]] = None) -> int:
    """执行完整流程并返回退出码。"""
    args = build_parser().parse_args(argv)
    try:
        source, output = check_paths(args.input, args.output_dir)
        rt = load_runtime()
        entries = discover_records(source, skip=[output])
        if not entries:
            raise ValueError(f"{source} 下没有找到含 session.yaml 的录制")
    except (DependencyError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return EXIT_ERROR
    analysis_dir = output / ANALYSIS_DIR
    analysis_dir.mkdir(parents=True)
    summaries: List[Dict[str, Any]] = []
    for number, entry in enumerate(entries, start=1):
        summary = process_record(rt, entry, output, analysis_dir)
        summaries.append(summary)
        detail = "；".join(item["message"] for item in summary["reasons"]) or summary["error"]
        print(f"[{number}/{len(entries)}] {entry.record_id}: {STATUS_LABELS[summary['status']]}"
              + (f"（{detail}）" if detail else ""), flush=True)
    staging_root = output / STAGING_DIR
    if staging_root.is_dir() and not any(staging_root.iterdir()):
        staging_root.rmdir()
    try:
        write_reports(analysis_dir, summaries)
    except OSError as error:
        print(f"错误：无法写出报告：{error}", file=sys.stderr)
        return EXIT_ERROR
    totals = {STATUS_LABELS[key]: sum(1 for item in summaries if item["status"] == key) for key in STATUS_LABELS}
    print(f"完成：{totals}；报告见 {analysis_dir}", flush=True)
    return exit_code_for(summaries)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """命令行入口。"""
    sys.exit(run(argv))


if __name__ == "__main__":
    main()
