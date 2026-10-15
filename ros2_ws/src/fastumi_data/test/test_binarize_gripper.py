"""验证根目录 binarize_gripper.py：判定算法的各个门槛，以及微型真实 MCAP 上的副本写出与校验。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
import binarize_gripper as bg  # noqa: E402

BASE_NS = 1_700_000_000_000_000_000


# ------------------------------------------------------------------ 合成序列
def make_series(
    levels: Sequence[Tuple[float, float]],
    rate_hz: float = 30.0,
    noise: float = 0.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 ``(开度, 持续秒数)`` 逐段拼接，返回 ``(纳秒时间, raw, valid)``；段间为阶跃。"""
    step = round(1e9 / rate_hz)
    rng = np.random.default_rng(seed)
    values: List[float] = []
    for level, seconds in levels:
        values.extend([level] * int(round(seconds * rate_hz)))
    raw = np.asarray(values, dtype=np.float64)
    if noise:
        raw = raw + rng.normal(0.0, noise, raw.shape)
    stamps = BASE_NS + step * np.arange(len(raw), dtype=np.int64)
    return stamps, raw, np.ones(len(raw), dtype=bool)


def knock_out(valid: np.ndarray, raw: np.ndarray, first: int, last: int) -> None:
    """把 ``[first, last]`` 帧标为无效并置 NaN。"""
    valid[first:last + 1] = False
    raw[first:last + 1] = float("nan")


BASIC = [(0.9, 3.0), (0.7, 3.0), (0.9, 3.0)]


def test_basic_open_close_open_passes_with_binary_sequence() -> None:
    """理想阶跃：恰好一次开→闭→开，切换时刻为平台首帧，输出整条二值序列。"""
    stamps, raw, valid = make_series(BASIC)
    result = bg.binarize_openness(stamps, raw, valid)
    assert result.passed and result.reasons == []
    assert result.plateau_values == pytest.approx((0.9, 0.7, 0.9))
    assert result.close_index == 90 and result.open_index == 180
    expected = np.concatenate([np.ones(90), np.zeros(90), np.ones(90)])
    assert np.array_equal(result.binary, expected)
    assert result.merged_runs == []


@pytest.mark.parametrize("closed", [0.3, 0.6, 0.85])
def test_different_closed_plateaus(closed: float) -> None:
    """闭合平台高低不同，只要与前后开平台至少差 0.045 就通过。"""
    result = bg.binarize_openness(*make_series([(0.9, 3.0), (closed, 3.0), (0.9, 3.0)]))
    assert result.passed
    assert result.plateau_values[1] == pytest.approx(closed)


def test_reopen_plateau_lower_than_initial_open_passes() -> None:
    """后开平台低于前开平台（回升只有 0.08）仍通过，回升侧用自己的中点。"""
    result = bg.binarize_openness(*make_series([(0.9, 3.0), (0.7, 3.0), (0.78, 3.0)]))
    assert result.passed
    assert result.thresholds == pytest.approx((0.8, 0.74))
    assert result.open_index == 180


def test_noise_and_spikes_do_not_create_extra_switches() -> None:
    """平台噪声和单帧毛刺被居中中值平滑吸收。"""
    stamps, raw, valid = make_series(BASIC, noise=0.005, seed=3)
    for index in (20, 50, 120, 150, 230):
        raw[index] = 0.1 if raw[index] > 0.8 else 0.95
    result = bg.binarize_openness(stamps, raw, valid)
    assert result.passed
    assert result.close_index == pytest.approx(90, abs=2)
    assert result.open_index == pytest.approx(180, abs=2)


def test_internal_short_gap_is_interpolated_and_recorded() -> None:
    """内部短缺失线性补齐；跨在切换点上的缺失也不会产生非二值。"""
    stamps, raw, valid = make_series(BASIC)
    knock_out(valid, raw, 40, 44)
    knock_out(valid, raw, 86, 95)
    result = bg.binarize_openness(stamps, raw, valid)
    assert result.passed
    assert [i for i, kind in enumerate(result.fill_kind) if kind] == list(range(40, 45)) + list(range(86, 96))
    assert set(result.fill_kind) == {None, bg.FILL_INTERNAL}
    assert set(result.binary.tolist()) == {0.0, 1.0}
    assert result.close_index in range(86, 96)


def test_leading_and_trailing_short_gaps_use_nearest_valid_value() -> None:
    """首尾短缺失取最近有效值，类型分别记为 leading 和 trailing。"""
    stamps, raw, valid = make_series(BASIC)
    knock_out(valid, raw, 0, 7)
    knock_out(valid, raw, 262, 269)
    result = bg.binarize_openness(stamps, raw, valid)
    assert result.passed
    assert result.fill_kind[0] == bg.FILL_LEADING and result.fill_kind[269] == bg.FILL_TRAILING
    assert result.filled[0] == pytest.approx(0.9) and result.filled[269] == pytest.approx(0.9)
    assert np.isnan(result.raw[0])


@pytest.mark.parametrize("first,last,kind", [(100, 130, bg.FILL_INTERNAL), (0, 20, bg.FILL_LEADING),
                                              (250, 269, bg.FILL_TRAILING)])
def test_long_gaps_are_listed_for_review(first: int, last: int, kind: str) -> None:
    """内部、首部、尾部缺失超过 0.5 秒都列为待核对，并报告缺失类型。"""
    stamps, raw, valid = make_series(BASIC)
    knock_out(valid, raw, first, last)
    result = bg.binarize_openness(stamps, raw, valid)
    codes = [item["code"] for item in result.reasons]
    assert "missing_gap_too_long" in codes and not result.passed
    assert [item["gap_kind"] for item in result.reasons if item["code"] == "missing_gap_too_long"] == [kind]


def test_gap_threshold_is_inclusive_at_exactly_half_second() -> None:
    """缺失跨度（前后有效样本的时间差）恰好 0.5 秒通过，多 1 ns 不通过。"""
    stamps, raw, valid = make_series(BASIC, rate_hz=10.0)
    knock_out(valid, raw, 40, 43)  # 有效样本 39 与 44 相隔 5 帧 = 0.5 s。
    assert bg.binarize_openness(stamps, raw, valid).passed
    late = stamps.copy()
    late[44:] += 1
    result = bg.binarize_openness(late, raw, valid)
    assert [item["code"] for item in result.reasons] == ["missing_gap_too_long"]


def test_nonfinite_raw_with_valid_flag_counts_as_missing() -> None:
    """valid=True 但开度非有限的帧同样视为缺失。"""
    stamps, raw, valid = make_series(BASIC)
    raw[30] = float("inf")
    result = bg.binarize_openness(stamps, raw, valid)
    assert result.passed and result.fill_kind[30] == bg.FILL_INTERNAL


def test_constant_openness_is_listed_for_review() -> None:
    """恒定开度没有平台差。"""
    result = bg.binarize_openness(*make_series([(0.8, 9.0)]))
    assert [item["code"] for item in result.reasons][0] == "plateau_difference_too_small"


def test_too_short_closed_plateau_is_listed_for_review() -> None:
    """闭合只有 0.4 秒，找不到满足最短时长的三段平台。"""
    result = bg.binarize_openness(*make_series([(0.9, 3.0), (0.6, 0.4), (0.9, 3.0)]))
    assert not result.passed and result.reasons[0]["code"] != "missing_gap_too_long"


def test_repeated_grasps_are_listed_for_review() -> None:
    """两次抓放不是恰好一次开→闭→开。"""
    levels = [(0.9, 2.0), (0.6, 2.0), (0.9, 2.0), (0.6, 2.0), (0.9, 2.0)]
    result = bg.binarize_openness(*make_series(levels))
    assert not result.passed
    assert any(item["code"] == "unexpected_state_pattern" for item in result.reasons)


@pytest.mark.parametrize("drop,rise,passed", [
    (0.045, 0.045, True), (0.2, 0.0449, False), (0.0449, 0.2, False)])
def test_plateau_difference_threshold_is_inclusive(drop: float, rise: float, passed: bool) -> None:
    """前开−闭合与后开−闭合都至少 0.045，恰好 0.045 通过。"""
    closed = 0.6
    result = bg.binarize_openness(*make_series([(closed + drop, 3.0), (closed, 3.0), (closed + rise, 3.0)]))
    assert result.passed is passed
    if not passed:
        assert result.reasons[0]["code"] == "plateau_difference_too_small"


def test_no_gripper_all_invalid_and_bad_timestamps() -> None:
    """空输入、全部无效和时间戳不严格递增都列为待核对，不抛异常。"""
    empty = bg.binarize_openness([], [], [])
    assert empty.reasons[0]["code"] == "no_gripper_samples"
    stamps, raw, valid = make_series(BASIC)
    assert bg.binarize_openness(stamps, raw, np.zeros_like(valid)).reasons[0]["code"] == "all_samples_invalid"
    stamps[10] = stamps[9]
    assert bg.binarize_openness(stamps, raw, valid).reasons[0]["code"] == "timestamps_not_increasing"


# ------------------------------------------------------------------ 分步函数边界
def test_median_smoothing_window_includes_both_ends() -> None:
    """窗口 [t-0.1 s, t+0.1 s] 含两端：端点上的样本参与中值。"""
    stamps = np.asarray([0, 50, 100, 150, 200], dtype=np.int64) * 1_000_000
    smoothed = bg.median_smooth(stamps, np.asarray([0.0, 9.0, 5.0, 9.0, 0.0]))
    assert smoothed[2] == pytest.approx(5.0)  # 含端点中值为 5；不含端点则为 9。


def test_plateau_minimum_span_is_inclusive() -> None:
    """每段首尾帧跨度恰好 0.5 秒可行，少一帧则无可行分法。"""
    step = 100_000_000
    values = np.asarray([1.0] * 6 + [0.0] * 6 + [1.0] * 6)
    stamps = step * np.arange(18, dtype=np.int64)
    assert bg.fit_three_plateaus(stamps, values) == (6, 12)
    assert bg.fit_three_plateaus(stamps[:17], values[:17]) is None


def test_plateau_ties_choose_earlier_boundaries() -> None:
    """误差相同时依次取较早的两个分界。"""
    stamps = 100_000_000 * np.arange(24, dtype=np.int64)
    assert bg.fit_three_plateaus(stamps, np.full(24, 0.5)) == (6, 12)


def test_midpoint_equality_boundaries() -> None:
    """下降侧小于或等于中点判闭，回升侧大于或等于中点判开；最低帧归入下降侧。"""
    smoothed = np.asarray([0.9, 0.5, 0.1, 0.5])
    states = bg.classify_states(smoothed, 1, (0.5, 0.5))
    assert states.tolist() == [1, 0, 0, 1]
    # 最低帧在下降侧：同值 0.1 之后的 0.4 在回升侧、低于中点，仍为闭。
    assert bg.classify_states(np.asarray([0.9, 0.1, 0.4, 0.9]), 1, (0.5, 0.5)).tolist() == [1, 0, 0, 1]


def test_short_runs_merge_into_previous_and_leading_into_next() -> None:
    """不足 0.5 秒的区间并入前一个持续区间；开头的并入其后第一个持续区间；恰好 0.5 秒保留。"""
    stamps = 100_000_000 * np.arange(40, dtype=np.int64)
    states = np.asarray([0] * 3 + [1] * 12 + [0] * 4 + [1] * 5 + [0] * 12 + [1] * 4, dtype=np.int8)
    merged, details = bg.merge_short_runs(stamps, states)
    # 开头 3 帧(0.3 s)并入其后的开；4 帧闭(0.4 s)并入前一个开；5 帧开(恰好 0.5 s)保留；
    # 末尾 4 帧开(最后一个区间算到最后一帧，0.3 s)并入前一个闭。
    assert bg.split_runs(merged) == [(1, 0, 24), (0, 24, 40)]
    assert [(d["first_frame"], d["to_state"]) for d in details] == [(0, 1), (15, 1), (36, 0)]


def test_short_run_merge_without_durable_state_changes_nothing() -> None:
    """没有任何持续区间时不合并，由判定函数列为待核对。"""
    stamps = 100_000_000 * np.arange(8, dtype=np.int64)
    states = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    merged, details = bg.merge_short_runs(stamps, states)
    assert merged.tolist() == states.tolist() and details == []


# ------------------------------------------------------------------ 微型真实 MCAP
try:
    import rosbag2_py
    from builtin_interfaces.msg import Time
    from fastumi_interfaces.msg import GripperState
    from geometry_msgs.msg import PoseStamped
    from rclpy.serialization import deserialize_message, serialize_message
    HAVE_ROS = True
except ImportError:  # 纯算法测试不需要 ROS。
    HAVE_ROS = False

requires_ros = pytest.mark.skipif(not HAVE_ROS, reason="需要已加载 ROS2 Jazzy 和工作空间的环境")

GRIPPER_TOPIC = "/gripper/state"
POSE_TOPIC = "/vive_tracker/pose"


def time_of(ns: int) -> Time:
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def make_record(
    root: Path, name: str, levels: Sequence[Tuple[float, float]],
    missing: Sequence[Tuple[int, int]] = (), split_at: Optional[int] = None,
) -> Path:
    """写出带夹爪、位姿话题的微型录制（MCAP + session.yaml + quality.yaml + 标定快照）。"""
    record = root / name
    (record / "calibration_snapshot").mkdir(parents=True)
    (record / "calibration_snapshot" / "calib.yaml").write_text("k: 1\n", encoding="utf-8")
    (record / "quality.yaml").write_text("has_alarms: false\n", encoding="utf-8")
    stamps, raw, valid = make_series(levels)
    for first, last in missing:
        knock_out(valid, raw, first, last)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(record / "raw" / "bag"), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    for index, (topic, type_name) in enumerate(
            ((GRIPPER_TOPIC, "fastumi_interfaces/msg/GripperState"), (POSE_TOPIC, "geometry_msgs/msg/PoseStamped"))):
        writer.create_topic(rosbag2_py.TopicMetadata(id=index, name=topic, type=type_name, serialization_format="cdr"))
    for i, stamp in enumerate(stamps.tolist()):
        if i == split_at:
            writer.split_bagfile()
        gripper = GripperState()
        gripper.header.stamp = time_of(stamp)
        gripper.header.frame_id = "umi_camera"
        gripper.raw_openness = float(raw[i])
        gripper.filtered_openness = float(raw[i]) * 0.5
        gripper.marker_distance_mm = 12.5 + i
        gripper.detected_marker_count = 2 if valid[i] else 1
        gripper.valid = bool(valid[i])
        writer.write(GRIPPER_TOPIC, serialize_message(gripper), stamp + 5_000_000)
        pose = PoseStamped()
        pose.header.stamp = time_of(stamp)
        pose.pose.position.x = float(i)
        writer.write(POSE_TOPIC, serialize_message(pose), stamp + 6_000_000)
    del writer
    session = {"schema_version": 3, "bag_uri": "raw/bag", "topic_map": {"gripper_state": GRIPPER_TOPIC},
               "size_bytes": 1, "state": "saved"}
    (record / "session.yaml").write_text(yaml.safe_dump(session), encoding="utf-8")
    return record


def tree_digest(root: Path) -> str:
    """目录内全部文件的相对路径与内容摘要，用于确认原始数据未变。"""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def read_messages(bag: Path):
    reader = bg.open_reader(bg.load_runtime(), bag)
    items = []
    while reader.has_next():
        topic, data, bag_ns = reader.read_next()
        items.append((topic, bytes(data), int(bag_ns)))
    return items


def run_cli(source: Path, output: Path) -> int:
    return bg.run(["--input", str(source), "--output-dir", str(output)])


@requires_ros
@pytest.mark.parametrize("split_at", [None, 100])
def test_published_copy_preserves_other_messages_and_binarizes_gripper(
    tmp_path: Path, split_at: Optional[int],
) -> None:
    """通过的录制：非夹爪消息逐字节一致，夹爪保留字段不变，补值来源有记录，size_bytes 为实际大小。"""
    source = tmp_path / "task"
    record = make_record(source, "rec_a", BASIC, missing=[(40, 44), (86, 95)], split_at=split_at)
    before = tree_digest(source)
    output = tmp_path / "task_binary"
    assert run_cli(source, output) == bg.EXIT_OK
    assert tree_digest(source) == before

    copy = output / "rec_a"
    # 明确覆盖两个源 MCAP 合并成一个副本 MCAP 的发布场景。
    source_metadata = bg.read_bag_metadata(record / "raw" / "bag")
    copy_metadata = bg.read_bag_metadata(copy / "raw" / "bag")
    assert len(source_metadata["files"]) == (2 if split_at is not None else 1)
    assert len(copy_metadata["files"]) == 1
    original = read_messages(record / "raw" / "bag")
    rewritten = read_messages(copy / "raw" / "bag")
    assert [(t, n) for t, _, n in original] == [(t, n) for t, _, n in rewritten]
    for (topic, old, _), (_, new, _) in zip(original, rewritten):
        if topic == POSE_TOPIC:
            assert old == new
    old_g = [deserialize_message(d, GripperState) for t, d, _ in original if t == GRIPPER_TOPIC]
    new_g = [deserialize_message(d, GripperState) for t, d, _ in rewritten if t == GRIPPER_TOPIC]
    assert len(new_g) == len(old_g) == 270
    for before_msg, after_msg in zip(old_g, new_g):
        assert after_msg.header == before_msg.header
        assert after_msg.detected_marker_count == before_msg.detected_marker_count
        assert after_msg.marker_distance_mm == before_msg.marker_distance_mm
        assert (math.isnan(after_msg.raw_openness) and math.isnan(before_msg.raw_openness)) \
            or after_msg.raw_openness == before_msg.raw_openness
        assert after_msg.filtered_openness in (0.0, 1.0) and after_msg.valid is True
    assert math.isnan(new_g[40].raw_openness) and new_g[40].valid

    document = yaml.safe_load((copy / bg.BINARIZATION_FILE).read_text(encoding="utf-8"))
    assert document["input_field"] == "raw_openness" and document["filled_frame_count"] == 15
    assert [item["frame"] for item in document["filled_frames"]] == list(range(40, 45)) + list(range(86, 96))
    assert all(item["original_valid"] is False for item in document["filled_frames"])
    assert document["original_valid"].count("0") == 15 and len(document["original_valid"]) == 270
    assert (copy / "quality.yaml").read_text(encoding="utf-8") == "has_alarms: false\n"
    assert (copy / "calibration_snapshot" / "calib.yaml").is_file()
    session = yaml.safe_load((copy / "session.yaml").read_text(encoding="utf-8"))
    assert session["size_bytes"] == bg.directory_size_bytes(copy)

    report = json.loads((output / "_analysis" / "binarization_report.json").read_text(encoding="utf-8"))
    assert report["totals"] == {"passed": 1, "review": 0, "error": 0}
    assert (output / "_analysis" / "plots" / "rec_a.png").is_file()
    assert (output / "_analysis" / "binarization_report.csv").is_file()
    assert not (output / bg.STAGING_DIR).exists()


@requires_ros
@pytest.mark.parametrize("field", ["message_count", "topics", "files"])
def test_metadata_count_mismatch_is_rejected(tmp_path: Path, field: str) -> None:
    """分包布局可变化，但总数、逐话题计数和文件计数合计不一致时仍拒绝副本。"""
    record = make_record(tmp_path / "src", "rec", BASIC, split_at=100)
    source_bag = record / "raw" / "bag"
    output_bag = tmp_path / "copy_bag"
    rt = bg.load_runtime()
    result = bg.binarize_openness(*bg.read_gripper_samples(rt, source_bag, GRIPPER_TOPIC))
    bg.write_binary_bag(rt, source_bag, output_bag, GRIPPER_TOPIC, result.binary)
    # 只破坏元数据，消息内容保持一致，确认校验仍能发现计数错误。
    metadata_path = output_bag / "metadata.yaml"
    document = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    info = document["rosbag2_bagfile_information"]
    if field == "message_count":
        info["message_count"] += 1
    elif field == "topics":
        info["topics_with_message_count"][0]["message_count"] += 1
    else:
        info["files"][0]["message_count"] += 1
    metadata_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(bg.AnalysisError, match="消息计数与原 bag 不一致"):
        bg.verify_copy(rt, source_bag, output_bag, GRIPPER_TOPIC, result.binary)


@requires_ros
def test_review_record_gets_report_only_and_exit_code_one(tmp_path: Path) -> None:
    """待核对的录制只生成报告和对照图，不发布副本；整体退出码为 1。"""
    source = tmp_path / "task"
    make_record(source, "good", BASIC)
    make_record(source, "flat", [(0.8, 9.0)])
    output = tmp_path / "out"
    assert run_cli(source, output) == bg.EXIT_REVIEW
    assert (output / "good" / "session.yaml").is_file() and not (output / "flat").exists()
    assert (output / "_analysis" / "plots" / "flat.png").is_file()
    report = json.loads((output / "_analysis" / "binarization_report.json").read_text(encoding="utf-8"))
    assert {item["record_id"]: item["status"] for item in report["records"]} == {"good": "passed", "flat": "review"}


@requires_ros
def test_single_record_input_is_published_under_its_name(tmp_path: Path) -> None:
    """输入为单条录制时，副本放在输出目录下的同名子目录。"""
    record = make_record(tmp_path / "src", "only", BASIC)
    output = tmp_path / "out"
    assert run_cli(record, output) == bg.EXIT_OK
    assert (output / "only" / bg.BINARIZATION_FILE).is_file()


@requires_ros
def test_failed_verification_publishes_nothing_and_exits_with_error(tmp_path: Path, monkeypatch) -> None:
    """写入后校验失败：不发布残缺副本、不留临时目录，退出码为 2。"""
    source = tmp_path / "task"
    make_record(source, "rec", BASIC)

    def broken(*args, **kwargs):
        raise bg.AnalysisError("人为制造的校验失败")

    monkeypatch.setattr(bg, "verify_copy", broken)
    output = tmp_path / "out"
    assert run_cli(source, output) == bg.EXIT_ERROR
    assert not (output / "rec").exists() and not (output / bg.STAGING_DIR / "rec").exists()
    report = json.loads((output / "_analysis" / "binarization_report.json").read_text(encoding="utf-8"))
    assert report["records"][0]["status"] == "error" and "人为制造" in report["records"][0]["error"]


@requires_ros
def test_missing_gripper_topic_is_a_read_error(tmp_path: Path) -> None:
    """session.yaml 指向不存在的夹爪话题：该录制为读取错误，不发布。"""
    source = tmp_path / "task"
    record = make_record(source, "rec", BASIC)
    session = yaml.safe_load((record / "session.yaml").read_text(encoding="utf-8"))
    session["topic_map"]["gripper_state"] = "/nope"
    (record / "session.yaml").write_text(yaml.safe_dump(session), encoding="utf-8")
    output = tmp_path / "out"
    assert run_cli(source, output) == bg.EXIT_ERROR
    assert not (output / "rec").exists()


@requires_ros
def test_existing_and_overlapping_output_paths_are_refused(tmp_path: Path) -> None:
    """输出目录已存在、与输入重叠或位于输入内部都拒绝，且不修改任何内容。"""
    source = tmp_path / "task"
    make_record(source, "rec", BASIC)
    digest = tree_digest(source)
    existing = tmp_path / "existing"
    existing.mkdir()
    for output in (existing, source, source / "inside"):
        assert run_cli(source, output) == bg.EXIT_ERROR
    assert tree_digest(source) == digest and list(existing.iterdir()) == []
    assert not (source / "inside").exists()


@requires_ros
def test_analysis_directory_is_skipped_by_discovery(tmp_path: Path) -> None:
    """输出目录作为输入时，``_analysis`` 不会被当作录制。"""
    source = tmp_path / "task"
    make_record(source, "rec", BASIC)
    output = tmp_path / "out"
    assert run_cli(source, output) == bg.EXIT_OK
    entries = bg.discover_records(output)
    assert [entry.record_id for entry in entries] == ["rec"]
