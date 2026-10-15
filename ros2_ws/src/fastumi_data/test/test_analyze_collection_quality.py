"""验证根目录 analyze_collection_quality.py：纯统计函数、事件状态机，以及合成 ROS2 录包上的端到端分析。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
import analyze_collection_quality as aq  # noqa: E402
from test_link7_pipeline import arguments, calibration, make_record  # noqa: E402,F401  (calibration 为 fixture)
from video_bag_builder import TOPICS  # noqa: E402
from video_fixtures import stamp_of  # noqa: E402


# ------------------------------------------------------------------ 纯函数
def test_stream_timing_uses_sorted_unique_times_and_original_order_checks() -> None:
    """频率和间隔按排序去重后的时间计算，重复与倒序按原始顺序计数。"""
    base = 1_000_000_000
    stamps = [base + i * 33_333_333 for i in range(10)]
    stamps[4], stamps[5] = stamps[5], stamps[4]
    stamps.append(stamps[-1])
    report = aq.stream_timing(stamps, np.asarray(stamps) + 5_000_000)
    assert report["count"] == 11 and report["unique_count"] == 10
    assert report["reversed_count"] == 1 and report["duplicate_count"] == 1
    assert report["rate_hz"] == pytest.approx(30.0, rel=1e-6)
    assert report["interval_ms"]["max"] == pytest.approx(33.333333)
    assert report["latency_ms"]["mean"] == pytest.approx(5.0)


@pytest.mark.parametrize("stamps", [[], [5], [7, 7, 7]])
def test_stream_timing_handles_empty_single_and_identical_samples(stamps) -> None:
    """空流、单样本和全部重复时间戳给出 None 与原因，而不是异常或 NaN。"""
    report = aq.stream_timing(stamps)
    assert report["rate_hz"] is None and report["max_gap_ms"] is None
    assert "rate_hz" in report["reasons"]
    assert report["interval_ms"]["reason"]
    text, replaced = aq.dumps_strict(report)
    assert replaced == [] and "NaN" not in text


def test_describe_filters_nonfinite_values() -> None:
    """非有限样本不参与统计，并给出数量。"""
    report = aq.describe([1.0, float("nan"), 3.0, float("inf")])
    assert report["count"] == 2 and report["nonfinite_count"] == 2
    assert report["mean"] == pytest.approx(2.0)
    empty = aq.describe([float("nan")])
    assert empty["max"] is None and empty["reason"]


def test_invalid_runs_report_gap_across_valid_neighbors() -> None:
    """连续失效区间给出样本数、起止时间和需要跨过的有效样本间隔。"""
    times = np.arange(10) * 100_000_000
    valid = np.ones(10, dtype=bool)
    valid[[0, 3, 4, 9]] = False
    runs = aq.invalid_runs(times, valid, origin_ns=0)
    assert [(r["start_index"], r["end_index"], r["samples"]) for r in runs] == [(0, 0, 1), (3, 4, 2), (9, 9, 1)]
    assert runs[1]["gap_s"] == pytest.approx(0.3)
    assert runs[0]["gap_s"] is None and runs[2]["gap_s"] is None
    assert aq.invalid_runs([], []) == []
    assert aq.invalid_runs(times, np.ones(10, dtype=bool)) == []
    assert aq.longest_valid_gap_s(times, valid) == pytest.approx(0.3)
    assert aq.longest_valid_gap_s(times, np.zeros(10, dtype=bool)) is None


def test_pose_invalid_mask_flags_nonfinite_and_bad_norm() -> None:
    """非有限位姿和四元数范数偏离单位长度均判为无效。"""
    xyz = np.zeros((4, 3))
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (4, 1))
    xyz[1, 0] = np.nan
    quat[2] = [0.0, 0.0, 0.0, 1.1]
    quat[3] = [0.0, 0.0, 0.0, 0.0]
    assert aq.pose_invalid_mask(xyz, quat).tolist() == [False, True, True, True]


def test_strict_json_replaces_nonfinite_numbers_and_numpy_types() -> None:
    """严格 JSON 不含 NaN/Infinity，numpy 类型转换为内置类型，路径可追溯。"""
    document = {"a": float("nan"), "b": [np.float64(1.5), np.int64(2), float("inf")],
                "c": np.array([1.0, np.nan]), "d": Path("/x"), "e": np.bool_(True)}
    text, replaced = aq.dumps_strict(document)
    parsed = json.loads(text, parse_constant=lambda name: pytest.fail(name))
    assert parsed == {"a": None, "b": [1.5, 2, None], "c": [1.0, None], "d": "/x", "e": True}
    assert sorted(replaced) == ["$.a", "$.b[2]", "$.c[1]"]


def test_exit_code_uses_highest_severity() -> None:
    """错误或未完成预检为 2，质量拒绝为 1，警告不影响退出码。"""
    passed = {"failed": False, "has_rejection": False, "has_incomplete": False}
    rejected = {**passed, "has_rejection": True}
    incomplete = {**passed, "has_incomplete": True}
    failed = {**passed, "failed": True}
    assert aq.exit_code_for([passed, passed]) == 0
    assert aq.exit_code_for([passed, rejected]) == 1
    assert aq.exit_code_for([rejected, incomplete]) == 2
    assert aq.exit_code_for([passed, failed]) == 2
    assert aq.record_status(False, True, True) == aq.STATUS_REJECTED
    assert aq.record_status(True, True, True) == aq.STATUS_FAILED
    assert aq.record_status(False, False, True) == aq.STATUS_INCOMPLETE


def event(kind, stamp=0, index=0, session="s", bag=0):
    """构造事件。"""
    return aq.EventRecord(bag, stamp, kind, session, "task", index)


def codes(spec):
    """episode 缺陷代码。"""
    return [item["code"] for item in spec.defects]


def test_parse_events_valid_and_defective_sequences() -> None:
    """START/STOP 配对正常；重复 START、孤立 STOP/ABORT、未结束、编号和身份不一致均为缺陷。"""
    specs, extra = aq.parse_events([event(1, 10), event(2, 20)], "s")
    assert extra == [] and len(specs) == 1 and specs[0].closed
    specs, _ = aq.parse_events([event(1, 10), event(1, 20, 1), event(2, 30, 1)], "s")
    assert codes(specs[0]) == ["EVENT_SEQUENCE"] and specs[1].closed
    specs, _ = aq.parse_events([event(2, 5), event(3, 6)], "s")
    assert all(codes(spec) == ["EVENT_SEQUENCE"] for spec in specs) and len(specs) == 2
    specs, _ = aq.parse_events([event(1, 10), event(3, 11)], "s")
    assert codes(specs[0]) == ["ABORTED"]
    specs, _ = aq.parse_events([event(1, 10)], "s")
    assert codes(specs[0]) == ["EVENT_SEQUENCE"]
    specs, _ = aq.parse_events([event(1, 10, 0), event(2, 20, 1)], "s")
    assert "EVENT_MISMATCH" in codes(specs[0])
    specs, _ = aq.parse_events([event(1, 10, 0), event(2, 20, 0), event(1, 30, 0), event(2, 40, 0)], "s")
    assert specs[0].closed and "EVENT_DUPLICATE_INDEX" in codes(specs[1])
    specs, _ = aq.parse_events([event(1, 10), event(2, 20)], "other")
    assert "EVENT_SESSION_MISMATCH" in codes(specs[0])
    specs, _ = aq.parse_events([event(1, 20), event(2, 10)], "s")
    assert "EVENT_ORDER" in codes(specs[0])
    _, extra = aq.parse_events([event(9, 1)], "s")
    assert extra[0]["code"] == "EVENT_TYPE_UNKNOWN"


def test_episode_image_mask_follows_header_time_and_post_roll() -> None:
    """图像按 Header 时间归入 START/STOP，STOP 后滚动结束之后到达的迟到帧不归入。"""
    stamps = np.asarray([5, 10, 15, 20, 25, 30])
    bag = np.asarray([5, 10, 15, 20, 26, 90])
    mask = aq.episode_image_mask(stamps, bag, 10, 20, post_roll_ns=10)
    assert mask.tolist() == [False, True, True, True, False, False]
    mask = aq.episode_image_mask(np.asarray([10, 20]), np.asarray([10, 35]), 10, 20, post_roll_ns=10)
    assert mask.tolist() == [True, False]


def test_sampling_and_preview_selection() -> None:
    """抽样每秒取一帧；代表帧优先覆盖严重失效段及其前后，再均匀补足，最多 12 张。"""
    stamps = np.arange(0, 3_500_000_000, 33_333_333)
    chosen = aq.sample_ordinals(stamps)
    assert chosen[0] == 0 and len(chosen) in (4,)
    assert all(stamps[b] - stamps[a] >= 1_000_000_000 for a, b in zip(chosen, chosen[1:]))
    assert aq.segment_neighbors(10, 14, 100) == [9, 12, 15]
    assert aq.segment_neighbors(0, 0, 3) == [0, 1]
    previews = aq.choose_preview_ordinals(300, [(100, 110), (40, 41)])
    assert len(previews) == 12
    assert {99, 105, 111, 39, 40, 42} <= set(previews)
    assert previews == sorted(set(previews))
    assert aq.choose_preview_ordinals(5, []) == [0, 1, 2, 3, 4]
    assert aq.choose_preview_ordinals(0, [(1, 2)]) == []
    many = [(i * 10, i * 10 + 1) for i in range(30)]
    assert len(aq.choose_preview_ordinals(1000, many)) == 12


def test_discover_records_skips_internal_dirs_symlinks_and_outputs(tmp_path: Path) -> None:
    """递归发现记录时跳过 .staging、.trash、_analysis、指定输出目录和目录符号链接。"""
    for relative in ("task/a", "task/b", "other/c", "task/.staging/x", "task/.trash/y",
                     "_analysis/old/z", "mine/out/w"):
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        (directory / "session.yaml").write_text("{}", encoding="utf-8")
    (tmp_path / "linked").symlink_to(tmp_path / "task" / "a", target_is_directory=True)
    found = aq.discover_records(tmp_path, skip=[tmp_path / "mine" / "out"])
    assert [entry.record_id for entry in found] == ["other/c", "task/a", "task/b"]
    single = aq.discover_records(tmp_path / "task" / "a")
    assert [entry.record_id for entry in single] == ["a"]
    assert aq.discover_records(tmp_path / "task" / ".staging" / "x")[0].record_id == "x"
    with pytest.raises(ValueError):
        aq.discover_records(tmp_path / "missing")


def test_default_output_dir_prefers_enclosing_dataset(tmp_path: Path) -> None:
    """缺省输出放在输入所在 dataset 目录的 _analysis 下，名称含微秒。"""
    from datetime import datetime

    stamp = datetime(2026, 10, 14, 9, 8, 7, 654321)
    inside = aq.default_output_dir(tmp_path / "dataset" / "task", stamp)
    assert inside == tmp_path / "dataset" / "_analysis" / "task_20261014_090807_654321"
    outside = aq.default_output_dir(tmp_path / "task", stamp)
    assert outside == aq.REPOSITORY_ROOT / "dataset" / "_analysis" / "task_20261014_090807_654321"


def test_help_runs_without_ros_environment() -> None:
    """未加载 ROS 的干净环境中 --help 正常，且导入脚本不会加载 rclpy。"""
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("AMENT", "ROS", "COLCON", "PYTHONPATH", "LD_LIBRARY_PATH"))}
    script = REPOSITORY_ROOT / "analyze_collection_quality.py"
    result = subprocess.run([sys.executable, "-I", str(script), "--help"], env=env,
                            capture_output=True, text=True, cwd="/")
    assert result.returncode == 0, result.stderr
    assert "--decode-mode" in result.stdout and "--plots" in result.stdout
    probe = subprocess.run(
        [sys.executable, "-I", "-c",
         f"import sys; sys.path.insert(0, {str(REPOSITORY_ROOT)!r}); "
         "import analyze_collection_quality; "
         "assert 'rclpy' not in sys.modules and 'cv2' not in sys.modules"],
        env=env, capture_output=True, text=True, cwd="/")
    assert probe.returncode == 0, probe.stderr


# ------------------------------------------------------------------ 合成录包
def analyze(source: Path, output: Path, cal: dict, *extra: str, calibrated: bool = True):
    """运行分析并返回 ``(退出码, report.json 或 None)``。"""
    command = ["--input", str(source), "--output-dir", str(output), "--plots", "none", *extra]
    if calibrated:
        command += ["--tracker-camera-extrinsic", str(cal["tracker"]),
                    "--end-camera-extrinsic", str(cal["end"])]
    code = aq.run(command)
    path = output / "report.json"
    return code, (json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)


def record_of(report: dict, name: str = "") -> dict:
    """按记录标识（后缀）取记录报告。"""
    return next(item for item in report["records"] if item["record_id"].endswith(name))


def rejection_codes(record: dict) -> set:
    """记录及其 episode 的全部拒绝代码。"""
    found = {item["code"] for item in record["rejections"]}
    for episode in record["episodes"]:
        found |= {item["code"] for item in episode["rejections"]}
    return found


def tree_digest(root: Path) -> str:
    """目录内全部文件内容与相对路径的摘要。"""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def corrupt_at(target: int):
    """返回把第 ``target`` 帧压缩图像替换为无法解码数据的覆盖函数。"""
    def override(index: int, message):
        if index == target:
            message.data = b"\xff\xd8\xff\xe0 not really a jpeg"
        return message
    return override


def corrupt_compressed(index: int, message):
    """第 30 帧替换为带 JPEG 文件头但无法解码的数据。"""
    return corrupt_at(30)(index, message)


def corrupt_h264(index: int, message):
    """第 30 帧替换为不是 Annex B 的数据。"""
    if index == 30:
        message.data = b"garbage"
    return message


@pytest.mark.parametrize("mode", ["raw", "compressed", "h264"])
def test_clean_record_matches_conversion_frames(tmp_path, calibration, mode) -> None:
    """三种图像模式的正常记录通过，并与 convert_mcap 输出的帧数一致。"""
    from fastumi_data.mcap_converter import run_batch

    record = make_record(tmp_path / "data", "task", "r", mode=mode)
    before = tree_digest(tmp_path / "data")
    code, report = analyze(record, tmp_path / "out", calibration)
    assert code == 0, report
    item = record_of(report)
    assert item["status"] == "passed" and item["image_mode"] == mode
    assert item["image_integrity"]["checked"] == item["image_integrity"]["messages"] == 80
    assert item["image_integrity"]["verdict"] == "全量验证"
    assert item["streams"]["image"]["rate_hz"] == pytest.approx(30.0, rel=1e-3)
    assert "QUALITY_YAML_MISSING" in {w["code"] for w in item["warnings"]}
    assert (tmp_path / "out" / "summary.csv").read_text(encoding="utf-8-sig").count("\n") == 2
    assert tree_digest(tmp_path / "data") == before
    assert run_batch(arguments(record, tmp_path / "converted", calibration)) == 0
    with h5py.File(tmp_path / "converted/records/task/r/episodes/episode_0000.hdf5", "r") as root:
        assert item["episodes"][0]["sync"]["output_frames"] == root["observations/qpos"].shape[0]


def test_multi_episode_h264_state_crosses_gop_and_episodes(tmp_path, calibration) -> None:
    """H.264 参考帧跨 episode 保留；两条 episode 帧数与转换一致。"""
    from fastumi_data.mcap_converter import run_batch

    record = make_record(tmp_path / "data", "task", "r", mode="h264", episodes=((2, 40), (45, 75)),
                         gop_size=7)
    code, report = analyze(record, tmp_path / "out", calibration)
    assert code == 0, report
    item = record_of(report)
    assert [e["status"] for e in item["episodes"]] == ["passed", "passed"]
    assert item["image_integrity"]["decoder"]["keyframes"] >= 80 // 7
    assert run_batch(arguments(record, tmp_path / "converted", calibration)) == 0
    for index, episode in enumerate(item["episodes"]):
        path = tmp_path / f"converted/records/task/r/episodes/episode_{index:04d}.hdf5"
        with h5py.File(path, "r") as root:
            assert episode["sync"]["output_frames"] == root["observations/qpos"].shape[0]


@pytest.mark.parametrize("mode, override", [("compressed", corrupt_compressed), ("h264", corrupt_h264)])
def test_corrupt_image_rejects_only_that_record(tmp_path, calibration, mode, override) -> None:
    """损坏图像使该记录被拒绝；同批其余记录仍完成分析。"""
    data = tmp_path / "data"
    make_record(data, "task", "a_good", mode=mode)
    make_record(data, "task", "b_bad", mode=mode, image_override=override)
    code, report = analyze(data, tmp_path / "out", calibration)
    assert code == 1
    assert record_of(report, "a_good")["status"] == "passed"
    bad = record_of(report, "b_bad")
    assert bad["status"] == "rejected" and "IMAGE_DECODE_FAILED" in rejection_codes(bad)
    assert bad["image_integrity"]["failed"] >= 1
    assert report["lists"]["rejected"] == ["task/b_bad"] or report["lists"]["rejected"] == ["b_bad"]


def test_sample_mode_reports_coverage_and_sampled_only_label(tmp_path, calibration) -> None:
    """抽样模式只解码部分图像，覆盖率和“仅抽样验证”写入记录与 Markdown。"""
    record = make_record(tmp_path / "data", "task", "r", mode="compressed", frames=120,
                         episodes=((2, 110),))
    code, report = analyze(record, tmp_path / "out", calibration, "--decode-mode", "sample")
    assert code == 0
    integrity = record_of(report)["image_integrity"]
    assert 0 < integrity["checked"] < integrity["messages"] == 120
    assert integrity["verdict"] == "仅抽样验证" and integrity["coverage"] < 1.0
    assert "仅抽样验证" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert report["totals"]["image_checked"] < report["totals"]["image_messages"]


def test_sample_mode_checks_abnormal_segment_frames(tmp_path, calibration) -> None:
    """抽样模式额外检查夹爪失效片段附近的帧，因此能发现其中的损坏图像。"""
    def invalid_gripper(topic, index, message):
        if topic == "gripper_state" and 30 <= index <= 36:
            message.valid = False
            message.filtered_openness = float("nan")
        return message

    record = make_record(tmp_path / "data", "task", "r", mode="compressed", frames=120,
                         episodes=((2, 110),), sensor_override=invalid_gripper,
                         image_override=corrupt_at(33))
    code, report = analyze(record, tmp_path / "out", calibration, "--decode-mode", "sample")
    assert code == 1
    item = record_of(report)
    assert "SYNC_GAP" not in rejection_codes(item)  # 图像缺陷先于同步被识别
    assert "IMAGE_DECODE_FAILED" in rejection_codes(item)


def test_gripper_gap_threshold_decides_rejection(tmp_path, calibration) -> None:
    """单帧夹爪漏检由插值门限覆盖；连续缺口超过门限则拒绝，并保留代表帧。"""
    def invalid_range(first, last):
        def override(topic, index, message):
            if topic == "gripper_state" and first <= index <= last:
                message.valid = False
                message.filtered_openness = float("nan")
                message.detected_marker_count = 1
            return message
        return override

    data = tmp_path / "data"
    make_record(data, "task", "a_single", sensor_override=invalid_range(30, 30))
    make_record(data, "task", "b_gap", sensor_override=invalid_range(30, 36))
    code = aq.run(["--input", str(data), "--output-dir", str(tmp_path / "out"), "--plots", "all",
                   "--tracker-camera-extrinsic", str(calibration["tracker"]),
                   "--end-camera-extrinsic", str(calibration["end"])])
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert code == 1
    single, gap = record_of(report, "a_single"), record_of(report, "b_gap")
    assert single["status"] == "passed" and single["episodes"][0]["gripper"]["invalid_run_count"] == 1
    assert gap["status"] == "rejected" and "SYNC_GAP" in rejection_codes(gap)
    assert gap["episodes"][0]["gripper"]["max_invalid_gap_s"] == pytest.approx(8 / 30, abs=1e-3)
    ordinals = {item["ordinal"] for item in gap["previews"]}
    assert {29, 33, 37} <= ordinals and len(ordinals) <= aq.MAX_PREVIEWS
    saved = [item for item in gap["previews"] if item["file"]]
    assert saved and all((tmp_path / "out" / item["file"]).is_file() for item in saved)
    assert any(item["marker_count"] == 1 and item["gripper_valid"] is False
               for item in gap["previews"] if "marker_count" in item)
    assert (tmp_path / "out" / gap["plot"]).is_file()
    assert "b_gap" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def test_invalid_pose_and_tracking_lost_are_rejected(tmp_path, calibration) -> None:
    """非有限位姿和 Tracker 内部跟踪丢失均拒绝且不抛异常。"""
    def bad_pose(topic, index, message):
        if topic == "tracker_pose" and index == 30:
            message.pose.position.x = float("nan")
        return message

    def lost(topic, index, message):
        if topic == "tracker_status" and 30 <= index <= 40:
            message.tracking_state = 0
        return message

    data = tmp_path / "data"
    make_record(data, "task", "a_pose", sensor_override=bad_pose)
    make_record(data, "task", "b_lost", sensor_override=lost)
    code, report = analyze(data, tmp_path / "out", calibration)
    assert code == 1
    assert "INVALID_POSE" in rejection_codes(record_of(report, "a_pose"))
    assert "TRACKING_LOST" in rejection_codes(record_of(report, "b_lost"))
    assert record_of(report, "b_lost")["streams"]["tracker_status"]["bad_count"] == 11
    text = (tmp_path / "out" / "report.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text


def reorder_stamps(index: int, message):
    """交换第 20、21 帧的 Header 时间，制造倒序时间戳。"""
    if index in (20, 21):
        stamp = stamp_of(41 - index)
        message.header.stamp.sec, message.header.stamp.nanosec = stamp // 10**9, stamp % 10**9
    return message


def duplicate_stamp(index: int, message):
    """第 20 帧复用第 19 帧时间，制造重复时间戳。"""
    if index == 20:
        stamp = stamp_of(19)
        message.header.stamp.sec, message.header.stamp.nanosec = stamp // 10**9, stamp % 10**9
    return message


@pytest.mark.parametrize("override, field", [(reorder_stamps, "reversed_count"), (duplicate_stamp, "duplicate_count")])
def test_disordered_image_times_are_reported_and_rejected(tmp_path, calibration, override, field) -> None:
    """倒序或重复的图像时间戳在原始顺序上计数，并由同步预检拒绝。"""
    record = make_record(tmp_path / "data", "task", "r", image_override=override)
    code, report = analyze(record, tmp_path / "out", calibration)
    assert code == 1
    item = record_of(report)
    assert item["streams"]["image"][field] >= 1
    assert "TIMESTAMP_ORDER" in {w["code"] for w in item["warnings"]}
    assert "FRAME_INTERVAL_ANOMALY" in rejection_codes(item)


def write_config(path: Path, **topics) -> Path:
    """写自定义话题映射的处理配置。"""
    mapping = dict(TOPICS)
    mapping.update(topics)
    path.write_text(yaml.safe_dump({"topics": mapping, "image_mode": "compressed"}), encoding="utf-8")
    return path


def test_topic_mapping_and_missing_streams(tmp_path, calibration) -> None:
    """话题映射来自生效配置；映射到不存在的话题记为关键流缺失，其余分析继续。"""
    record = make_record(tmp_path / "data", "task", "r")
    config = write_config(tmp_path / "bad.yaml", gripper_state="/gripper/missing")
    code, report = analyze(record, tmp_path / "out1", calibration, "--config", str(config))
    item = record_of(report)
    assert code == 1 and item["status"] == "rejected"
    assert {"MISSING_STREAM"} <= {r["code"] for r in item["rejections"]}
    assert item["topics"]["gripper_state"] == "/gripper/missing"
    assert item["streams"]["tracker_pose"]["count"] == 80  # 其他流照常分析
    code, report = analyze(record, tmp_path / "out2", calibration,
                           "--config", str(write_config(tmp_path / "ok.yaml")))
    assert code == 0 and record_of(report)["config"]["source"] == "explicit_config"


def test_missing_event_stream_and_unclosed_episode(tmp_path, calibration) -> None:
    """缺少事件话题、没有 episode 事件、末尾未 STOP 均为事件缺陷。"""
    data = tmp_path / "data"
    make_record(data, "task", "a_noevents", episodes=())
    make_record(data, "task", "b_unclosed", unclosed_tail=True)
    make_record(data, "task", "c_nomap")
    config = write_config(tmp_path / "bad.yaml", episode_event="/events/missing")
    code, report = analyze(data, tmp_path / "out", calibration)
    assert code == 1
    assert "NO_EPISODE" in rejection_codes(record_of(report, "a_noevents"))
    assert "EVENT_SEQUENCE" in rejection_codes(record_of(report, "b_unclosed"))
    code, report = analyze(data / "task" / "c_nomap", tmp_path / "out2", calibration, "--config", str(config))
    assert "MISSING_STREAM" in rejection_codes(record_of(report))


def test_empty_streams_single_sample_and_strict_report(tmp_path, calibration) -> None:
    """图像流为空、单帧 episode 都生成明确结果，报告 JSON 严格且无 NaN。"""
    data = tmp_path / "data"
    make_record(data, "task", "a_empty", frames=20, episodes=((2, 15),), drop_images=list(range(20)))
    make_record(data, "task", "b_single", frames=20, episodes=((5, 5),))
    code, report = analyze(data, tmp_path / "out", calibration)
    assert code == 1 and report["totals"]["failed"] == 0
    empty = record_of(report, "a_empty")
    assert "EMPTY_STREAM" in rejection_codes(empty) and empty["streams"]["image"]["rate_hz"] is None
    assert "INSUFFICIENT_IMAGES" in rejection_codes(empty)
    assert rejection_codes(record_of(report, "b_single")) >= {"EVENT_ORDER"}
    text = (tmp_path / "out" / "report.json").read_text(encoding="utf-8")
    json.loads(text, parse_constant=lambda name: pytest.fail(name))


def test_clock_mapping_generation_change_is_rejected(tmp_path, calibration) -> None:
    """采集期间相机时钟映射代号变化使图像时间不可比，记录被拒绝。"""
    stable = make_record(tmp_path / "data", "task", "a_stable", clock_generation_of=lambda i: 1)
    changed = make_record(tmp_path / "data", "task", "b_changed",
                          clock_generation_of=lambda i: 1 if i < 40 else 2)
    code, report = analyze(tmp_path / "data", tmp_path / "out", calibration)
    assert code == 1 and stable and changed
    assert record_of(report, "a_stable")["clock_mapping"]["generations"] == [1]
    assert "CLOCK_MAPPING_CHANGED" in {r["code"] for r in record_of(report, "b_changed")["rejections"]}


def test_split_bag_counts_and_doubled_file_counts_warning(tmp_path, calibration) -> None:
    """分包录包按实际读取计数；rosbag2 写出的分包文件计数翻倍和 session 计数不一致都只是警告。"""
    record = make_record(tmp_path / "data", "task", "r", mode="raw", frames=300, episodes=((2, 290),),
                         max_bagfile_size=300_000)
    bag = record / "raw" / "bag"
    assert len(list(bag.glob("*.mcap"))) > 1
    session = yaml.safe_load((record / "session.yaml").read_text(encoding="utf-8"))
    session["message_counts"] = {"image": 300, "tracker_pose": 300, "tracker_status": 300,
                                 "gripper_state": 300, "episode_event": 2}
    (record / "session.yaml").write_text(yaml.safe_dump(session), encoding="utf-8")
    code, report = analyze(record, tmp_path / "out1", calibration)
    item = record_of(report)
    totals = item["counts"]["bag_metadata_totals"]
    assert code == 0 and item["status"] == "passed" and item["counts"]["image"]["read"] == 300
    assert len(totals["files"]) > 1 and totals["topic_sum"] == totals["message_count"]
    # 当前 rosbag2 的 files[].message_count 是实际消息数的两倍，必须被识别为警告。
    assert totals["file_sum"] == 2 * totals["topic_sum"]
    warning = {w["code"]: w for w in item["warnings"]}
    assert "2.00 倍" in warning["BAG_METADATA_INCONSISTENT"]["message"]
    assert "SESSION_COUNT_MISMATCH" not in warning and "BAG_COUNT_MISMATCH" not in warning

    metadata = yaml.safe_load((bag / "metadata.yaml").read_text(encoding="utf-8"))
    for entry in metadata["rosbag2_bagfile_information"]["files"]:
        entry["message_count"] //= 2
    (bag / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    code, report = analyze(record, tmp_path / "out2", calibration)
    assert {w["code"] for w in record_of(report)["warnings"]} == {"QUALITY_YAML_MISSING"}

    session["message_counts"]["image"] = 600
    (record / "session.yaml").write_text(yaml.safe_dump(session), encoding="utf-8")
    code, report = analyze(record, tmp_path / "out3", calibration)
    item = record_of(report)
    assert code == 0 and item["status"] == "passed"
    mismatch = next(w for w in item["warnings"] if w["code"] == "SESSION_COUNT_MISMATCH")
    assert mismatch["declared"] == 600 and "2.00 倍" in mismatch["message"]


def test_calibration_problems_leave_preflight_incomplete(tmp_path, calibration) -> None:
    """序列号、时间戳来源不匹配或外参缺失时，基础检查完成但同步预检标记未完成（退出码 2）。"""
    record = make_record(tmp_path / "data", "task", "r", timestamp_source="callback_now")
    binding = calibration["tracker"].parent / "T_cam_tracker.binding.yaml"
    document = yaml.safe_load(binding.read_text(encoding="utf-8"))

    document["tracker_serial"] = "LHR-OTHER"
    binding.write_text(yaml.safe_dump(document), encoding="utf-8")
    code, report = analyze(record, tmp_path / "out1", calibration)
    item = record_of(report)
    assert code == 2 and item["status"] == "incomplete" and item["has_incomplete"]
    assert {p["code"] for p in item["preflight_problems"]} == {"TRACKER_SERIAL_MISMATCH"}
    assert item["episodes"][0]["sync"]["status"] == "incomplete"
    assert item["episodes"][0]["images"]["count"] == 69  # 基础检查不依赖外参

    document["tracker_serial"] = "LHR-TEST"
    document["time_calibration_timestamp_source"] = "v4l2_buffer_monotonic_mapped_to_ros"
    binding.write_text(yaml.safe_dump(document), encoding="utf-8")
    code, report = analyze(record, tmp_path / "out2", calibration)
    assert code == 2
    assert {p["code"] for p in record_of(report)["preflight_problems"]} == {"TIMESTAMP_SOURCE_MISMATCH"}

    code, report = analyze(record, tmp_path / "out3", calibration, "--tracker-camera-extrinsic",
                           str(tmp_path / "missing.yaml"), calibrated=False)
    assert code == 2 and report["calibration"]["available"] is False
    assert {p["code"] for p in record_of(report)["preflight_problems"]} == {"CALIBRATION_UNAVAILABLE"}


def test_data_defect_takes_precedence_over_calibration_problem(tmp_path, calibration) -> None:
    """已发现数据缺陷时记录为质量拒绝，即使外参不可用；同步不会执行。"""
    record = make_record(tmp_path / "data", "task", "r", mode="compressed", image_override=corrupt_compressed)
    code, report = analyze(record, tmp_path / "out", calibration, "--tracker-camera-extrinsic",
                           str(tmp_path / "missing.yaml"), calibrated=False)
    item = record_of(report)
    assert code == 1 and item["status"] == "rejected"
    assert item["episodes"][0]["sync"]["status"] == "blocked_by_data_defect"


def test_source_fps_override_and_requirement(tmp_path, calibration) -> None:
    """快照缺少标称帧率时预检未完成，--source-camera-fps 声明后恢复；不支持的值是参数错误。"""
    record = make_record(tmp_path / "data", "task", "r", fps=None)
    code, report = analyze(record, tmp_path / "out1", calibration)
    assert code == 2 and "NOMINAL_FPS_UNAVAILABLE" in {
        p["code"] for p in record_of(report)["preflight_problems"]}
    code, report = analyze(record, tmp_path / "out2", calibration, "--source-camera-fps", "30")
    assert code == 0 and record_of(report)["nominal_source_fps_source"] == "cli_override"
    assert analyze(record, tmp_path / "out3", calibration, "--source-camera-fps", "45")[0] == 2
    assert not (tmp_path / "out3").exists()


def test_batch_isolates_unreadable_records(tmp_path, calibration) -> None:
    """session.yaml 损坏或缺少 bag 的记录标记为分析失败，其余记录继续输出（退出码 2）。"""
    data = tmp_path / "data"
    make_record(data, "task", "a_good")
    broken = make_record(data, "task", "b_broken_session")
    (broken / "session.yaml").write_text("{ unclosed", encoding="utf-8")
    nobag = make_record(data, "task", "c_nobag")
    import shutil
    shutil.rmtree(nobag / "raw")
    make_record(data, "task", "d_good")
    code, report = analyze(data, tmp_path / "out", calibration)
    assert code == 2
    assert [r["status"] for r in report["records"]] == ["passed", "failed", "failed", "passed"]
    assert report["lists"]["failed"] == ["task/b_broken_session", "task/c_nobag"]
    assert record_of(report, "b_broken_session")["error"]["type"] == "AnalysisError"
    assert "分析失败" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    rows = (tmp_path / "out" / "summary.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(rows) == 5


def crash_on_marked_record(entry, plot_name, partial_dir):
    """并行测试用任务：名称含 crash_me 的记录让工作进程直接退出，其余照常分析。"""
    if entry.record_id.endswith("crash_me"):
        os._exit(7)
    return aq._analyze_in_worker(entry, plot_name, partial_dir)


def stable_view(report: dict) -> dict:
    """去掉运行时间、输出位置和参数等与并行方式无关的字段，便于比较串行与并行结果。"""
    return {key: value for key, value in report.items()
            if key not in ("generated_at", "output_dir", "arguments")}


def test_resolve_jobs() -> None:
    """0 按 CPU 数、记录数和上限自动选择；显式值只受记录数限制。"""
    assert aq.resolve_jobs(0, 1) == 1
    assert aq.resolve_jobs(0, 1000) == min(os.cpu_count() or 1, aq.MAX_AUTO_JOBS)
    assert aq.resolve_jobs(1, 10) == 1
    assert aq.resolve_jobs(32, 10) == 10
    assert aq.resolve_jobs(32, 100) == 32


def test_negative_jobs_is_a_usage_error(tmp_path) -> None:
    """--jobs 为负数时按参数错误退出，且不创建输出。"""
    record = make_record(tmp_path / "data", "task", "r")
    with pytest.raises(SystemExit) as raised:
        aq.run(["--input", str(record), "--output-dir", str(tmp_path / "out"), "--jobs", "-1"])
    assert raised.value.code == 2
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.partial").exists()


def test_parallel_results_match_serial(tmp_path, calibration) -> None:
    """多进程并行与串行得到相同的报告、记录顺序、图表和代表帧文件，坏记录仍被隔离。"""
    data = tmp_path / "data"
    make_record(data, "task", "a_good")
    make_record(data, "task", "b_bad", image_override=corrupt_compressed)
    broken = make_record(data, "task", "c_broken_session")
    (broken / "session.yaml").write_text("{ unclosed", encoding="utf-8")
    make_record(data, "task", "d_good")
    serial_code, serial = analyze(data, tmp_path / "serial", calibration, "--plots", "all", "--jobs", "1")
    parallel_code, parallel = analyze(data, tmp_path / "parallel", calibration, "--plots", "all", "--jobs", "3")
    assert serial_code == parallel_code == 2
    assert [r["status"] for r in parallel["records"]] == ["passed", "rejected", "failed", "passed"]
    assert stable_view(parallel) == stable_view(serial)

    def files(root: Path) -> list:
        return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())

    assert files(tmp_path / "parallel") == files(tmp_path / "serial")
    assert (tmp_path / "parallel" / "summary.csv").read_bytes() == (tmp_path / "serial" / "summary.csv").read_bytes()


def test_worker_crash_only_fails_the_crashing_record(tmp_path, calibration) -> None:
    """工作进程异常退出时，先单独重跑受影响的记录定位元凶；其余记录仍得到正常结果。"""
    data = tmp_path / "data"
    for name in ("a_good", "b_crash_me", "c_good", "d_good"):
        make_record(data, "task", name)
    args = aq.build_parser().parse_args([
        "--input", str(data), "--plots", "none",
        "--tracker-camera-extrinsic", str(calibration["tracker"]),
        "--end-camera-extrinsic", str(calibration["end"])])
    entries = aq.discover_records(data)
    ctx = aq.Context(None, args, None, None, None, entries[-1].record_id)
    partial = tmp_path / "partial"
    partial.mkdir()
    done = []
    reports = aq.analyze_records(
        ctx, entries, [f"p{i}" for i in range(len(entries))], partial, 2,
        lambda number, entry, report: done.append(entry.record_id), task=crash_on_marked_record)
    assert [r["status"] for r in reports] == ["passed", "failed", "passed", "passed"]
    assert reports[1]["error"]["type"] == "WorkerCrashedError"
    assert sorted(done) == sorted(e.record_id for e in entries)
    assert not list(Path(tempfile.gettempdir()).glob("analyze_active_*"))


def test_existing_output_directory_is_refused(tmp_path, calibration) -> None:
    """输出目录已存在时拒绝覆盖（退出码 2），已有内容和原始数据保持不变。"""
    record = make_record(tmp_path / "data", "task", "r")
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    before = tree_digest(tmp_path)
    with pytest.raises(SystemExit) as raised:
        aq.run(["--input", str(record), "--output-dir", str(output)])
    assert raised.value.code == 2
    assert tree_digest(tmp_path) == before
    assert (tmp_path / "out.partial").exists() is False


def test_output_is_published_atomically_and_plot_policy(tmp_path, calibration) -> None:
    """成功后才出现最终目录（无 .partial、无 .tmp）；issues 策略只为异常和最新记录绘图。"""
    data = tmp_path / "data"
    make_record(data, "task", "a_first")
    make_record(data, "task", "b_second")
    make_record(data, "task", "c_bad", image_override=corrupt_compressed)
    for name, stamp in (("a_first", 1), ("b_second", 3), ("c_bad", 2)):
        path = data / "task" / name / "session.yaml"
        session = yaml.safe_load(path.read_text(encoding="utf-8"))
        session["saved_at_unix"] = float(stamp)
        path.write_text(yaml.safe_dump(session), encoding="utf-8")
    output = tmp_path / "out"
    code = aq.run(["--input", str(data), "--output-dir", str(output),
                   "--tracker-camera-extrinsic", str(calibration["tracker"]),
                   "--end-camera-extrinsic", str(calibration["end"])])
    assert code == 1
    assert not (tmp_path / "out.partial").exists()
    assert not list(output.rglob("*.tmp"))
    assert sorted(path.name for path in (output / "plots").iterdir()) == ["task_b_second.png", "task_c_bad.png"]
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert {item["record_id"] for item in report["records"] if item.get("plot")} == {"task/b_second", "task/c_bad"}
    assert record_of(report, "b_second")["is_latest"] is True
