"""离线验证抓放切分、标注持久化、阶段归零和发布保护。"""

from copy import deepcopy
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ros2_ws/src/fastumi_data"))

from pick_place.core import (
    SCHEMA_VERSION, binary_values, delete_annotation, detect_events, fingerprint, initialize_annotation,
    load_manifest, midpoint_index, rebase_qpos, reconcile_annotation, save_json,
    restore_annotation, stage_issues, stage_rows, update_annotation,
)
from pick_place.pipeline import prepare_stage, publish_directory, validate_paths


def sample_record(count=120):
    """构造包含稳定开闭开事件的 30 Hz 记录，时间采用真实量级 Unix 纳秒。"""
    stamps = 1_800_000_000_000_000_000 + np.arange(count, dtype=np.int64) * 33_333_333
    record = {
        "id": "test/record", "source_rel": "test/record", "fingerprint": "source",
        "processing_signature": "config", "image_stamps_ns": stamps.tolist(),
        "source_image_index": list(range(count)), "dropped_frames_before": [0] * count,
        "quality_errors": [], "events": {
            "close_ns": int(stamps[30]), "open_ns": int(stamps[90]),
            "passed": True, "reasons": [],
        },
    }
    initialize_annotation(record)
    return record


def write_hdf5(output, record):
    """写含旋转、平移和非二值夹爪的最小同步样例，图像形状为 N×224×224×3。"""
    count = len(record["image_stamps_ns"])
    time = np.arange(count) / 30
    qpos = np.column_stack((time, -time, time / 2,
                           Rotation.from_euler("z", 0.7 + time).as_quat(),
                           np.linspace(0.2, 0.9, count)))
    path = output / "sample.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("observations/qpos", data=qpos)
        handle.create_dataset("action", data=qpos)
        handle.create_dataset("observations/image_timestamp_ns", data=record["image_stamps_ns"])
        handle.create_dataset("observations/source_image_index", data=record["source_image_index"])
        handle.create_dataset("observations/images/front", data=np.broadcast_to(
            np.arange(count, dtype=np.uint8)[:, None, None, None], (count, 224, 224, 3)))
        handle.attrs["metadata_json"] = "{}"
    record["hdf5"] = path.name
    return qpos


def test_midpoint_uses_exact_ns_and_earlier_tie():
    """大时间戳下的半纳秒距离不会被 float 舍入。"""
    origin = 1_800_000_000_000_000_000
    assert midpoint_index([origin, origin + 10], origin + 2, origin + 8) == 0
    assert midpoint_index([origin, origin + 10], origin + 2, origin + 9) == 1
    with pytest.raises(ValueError):
        midpoint_index([10, 10], 1, 20)


@pytest.mark.parametrize("noise", [False, True])
def test_detection_accepts_intermediate_hold_and_short_missing(noise):
    """持物时 0.65 开度不能被固定 0.5 阈值遗漏，短噪声和缺失可以处理。"""
    stamps = np.arange(120, dtype=np.int64) * 33_333_333
    raw = np.r_[np.full(30, 0.95), np.full(60, 0.65), np.full(30, 0.95)]
    valid = np.ones(120, dtype=bool)
    if noise:
        raw[50] = 1.0
        valid[65:67] = False
        raw[65:67] = np.nan
    result = detect_events(stamps, raw, valid)
    assert result["passed"]
    assert result["close_ns"] < result["open_ns"]


@pytest.mark.parametrize("raw", [
    np.r_[np.ones(30), np.zeros(90)],
    np.tile(np.r_[np.ones(30), np.zeros(30)], 2),
    np.full(120, 0.8),
])
def test_incomplete_multiple_or_unclear_cycles_require_review(raw):
    """缺失释放、多次抓放、无明显开合都不得自动通过。"""
    result = detect_events(np.arange(120, dtype=np.int64) * 33_333_333, raw, np.ones(120, bool))
    assert not result["passed"]
    assert result["reasons"]


def test_shared_boundary_independent_stages_and_short_gap_validation():
    """两段共享一帧；单阶段缺口不牵连另一阶段，短片段不可确认。"""
    record = sample_record()
    assert stage_rows(record, "pick") == (0, 61)
    assert stage_rows(record, "place") == (60, 120)
    record["dropped_frames_before"][100] = 3
    assert not stage_issues(record, "pick")
    assert stage_issues(record, "place")
    update_annotation(record, decisions={
        "pick": {"status": "accepted", "reason": "成功"},
        "place": {"status": "excluded", "reason": "放置失败"},
    })
    assert record["annotation"]["stages"]["pick"]["status"] == "accepted"
    record["annotation"]["start"] = 55
    with pytest.raises(ValueError):
        update_annotation(record, decisions={"pick": {"status": "accepted", "reason": ""}})


def test_events_recompute_midpoint_manual_override_and_persistence(tmp_path):
    """事件修正重算中点，显式切点覆盖并保存恢复；更新后撤销旧确认。"""
    record = sample_record()
    stamps = record["image_stamps_ns"]
    update_annotation(record, events={"close_ns": stamps[40], "open_ns": stamps[100]})
    assert record["annotation"]["split"] == 70
    assert record["annotation"]["stages"]["pick"]["status"] == "review"
    update_annotation(record, bounds={"split": 65})
    assert record["annotation"]["split_source"] == "manual_override"
    path = tmp_path / "segments.json"
    save_json(path, {"schema_version": SCHEMA_VERSION, "records": [record]})
    assert load_manifest(path)["records"][0] == record


def test_changed_source_invalidates_review_and_unchanged_preserves():
    """指纹未变保留人工标注，指纹变化后即使自动检测通过也须重新核对。"""
    previous = sample_record()
    update_annotation(previous, bounds={"split": 65})
    assert reconcile_annotation(sample_record(), previous)["annotation"]["split"] == 65
    changed = sample_record()
    changed["fingerprint"] = "different"
    result = reconcile_annotation(changed, previous)
    assert result["annotation"]["stages"]["pick"]["status"] == "review"
    assert result["invalidated_annotation"]["split"] == 65


def test_fingerprint_detects_same_size_content_changes(tmp_path):
    """内容指纹不依赖文件长度或文件时间。"""
    path = tmp_path / "data.mcap"
    path.write_bytes(b"abc")
    original = fingerprint(tmp_path)
    path.write_bytes(b"abd")
    assert fingerprint(tmp_path) != original


@pytest.mark.parametrize("mode", ["binary", "continuous"])
def test_stage_export_preserves_relative_geometry_shared_frame_and_gripper(tmp_path, mode):
    """两个首帧归零，同一交界图像一致，state/action 同时刻且夹爪模式正确。"""
    record = sample_record()
    original = write_hdf5(tmp_path, record)
    pick = prepare_stage(record, "pick", mode, tmp_path, tmp_path / "npz")
    place = prepare_stage(record, "place", mode, tmp_path, tmp_path / "npz")
    with np.load(pick["npz"]) as first, np.load(place["npz"]) as second:
        np.testing.assert_array_equal(first["images"][-1], second["images"][0])
        for arrays in (first, second):
            np.testing.assert_allclose(arrays["state"][0, :6], 0, atol=1e-7)
            np.testing.assert_array_equal(arrays["state"], arrays["action"])
        if mode == "binary":
            assert set(np.unique(first["state"][:, 6])) == {0, 1}
            assert second["state"][0, 6] == 0
            assert second["state"][-1, 6] == 1
        else:
            np.testing.assert_allclose(second["state"][:, 6], original[60:, 7])
    relative = rebase_qpos(original[60:])
    anchor = Rotation.from_quat(original[60, 3:7])
    np.testing.assert_allclose(anchor.apply(relative[:, :3]) + original[60, :3], original[60:, :3])
    np.testing.assert_allclose((anchor * Rotation.from_quat(relative[:, 3:7])).as_matrix(),
                               Rotation.from_quat(original[60:, 3:7]).as_matrix(), atol=1e-12)


def test_binary_event_correction_and_pick_without_release():
    """事件直接决定二值边界，缺失释放时已确认的抓取片段仍可使用。"""
    np.testing.assert_array_equal(binary_values([1, 2, 3, 4, 5], 2, 4), [1, 0, 0, 1, 1])
    np.testing.assert_array_equal(binary_values([1, 2, 3], 2, None), [1, 0, 0])
    record = sample_record()
    record["annotation"]["open_ns"] = None
    assert not stage_issues(record, "pick")
    assert stage_issues(record, "place")


def test_manual_confirmation_cannot_bypass_quality_errors():
    """即使直接要求确认，硬质量问题也必须保持拒绝。"""
    record = sample_record()
    record["quality_errors"] = ["跟踪失效"]
    before = deepcopy(record)
    with pytest.raises(ValueError, match="跟踪失效"):
        update_annotation(record, decisions={"pick": {"status": "accepted", "reason": ""}})
    assert record == before


def test_publish_refuses_overwrite_and_rolls_back_on_failure(tmp_path, monkeypatch):
    """新数据发布失败时旧数据仍完整可用。"""
    import pick_place.pipeline as pipeline

    target, prepared = tmp_path / "published", tmp_path / "staging" / "prepared"
    target.mkdir()
    prepared.mkdir(parents=True)
    (target / "old").write_text("old")
    (prepared / "new").write_text("new")
    with pytest.raises(FileExistsError):
        publish_directory(prepared, target, False)
    rename = pipeline.os.rename

    def fail_new(source, destination):
        """模拟第二次重命名失败，仅拒绝新数据发布。"""
        if source == prepared:
            raise OSError("模拟发布失败")
        return rename(source, destination)

    monkeypatch.setattr(pipeline.os, "rename", fail_new)
    with pytest.raises(OSError):
        publish_directory(prepared, target, True)
    assert (target / "old").read_text() == "old"


def test_output_must_not_contain_input(tmp_path):
    """防止输出写入源目录，或覆盖包含源目录的父目录。"""
    with pytest.raises(ValueError):
        validate_paths(str(tmp_path), str(tmp_path / "out"))
    with pytest.raises(ValueError):
        validate_paths(str(tmp_path / "raw"), str(tmp_path))


def test_review_offscreen_save_and_resume(tmp_path, monkeypatch):
    """无显示器验证窗口加载、逐帧定位、边界保存与恢复。"""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    from pick_place.review import ReviewWindow

    record = sample_record()
    write_hdf5(tmp_path, record)
    record["annotation"]["stages"]["pick"]["status"] = "review"
    save_json(tmp_path / "segments.json", {"schema_version": SCHEMA_VERSION, "input_root": str(tmp_path), "records": [record]})
    app = QApplication.instance() or QApplication([])
    window = ReviewWindow(tmp_path)
    window.records.setCurrentRow(0)
    window.slider.setValue(65)
    window.use_current("split")
    assert window.fields["split"].value() == 65
    assert window.save_current()
    saved = load_manifest(tmp_path / "segments.json")["records"][0]
    assert saved["annotation"]["split"] == 65
    window.close()
    app.processEvents()


def test_deleted_annotation_survives_analysis_and_restores_safely():
    """删除覆盖两个阶段，重分析仍保持删除，源变化后恢复为待核对。"""
    record = sample_record()
    original = deepcopy(record["annotation"])
    delete_annotation(record)
    assert all(state["status"] == "excluded" for state in record["annotation"]["stages"].values())
    unchanged = reconcile_annotation(sample_record(), record)
    assert unchanged["annotation"]["deleted"]
    restore_annotation(unchanged)
    assert unchanged["annotation"] == original
    changed = sample_record()
    changed["fingerprint"] = "changed"
    changed = reconcile_annotation(changed, record)
    assert changed["annotation"]["deleted"]
    restore_annotation(changed)
    assert all(state["status"] == "review" for state in changed["annotation"]["stages"].values())


def test_review_list_status_delete_restore_and_empty_selection(tmp_path, monkeypatch):
    """验证保存标签刷新、删除后切换、重开隐藏、恢复及删除末条清空预览。"""
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    from pick_place.review import ReviewWindow

    first = sample_record()
    write_hdf5(tmp_path, first)
    for state in first["annotation"]["stages"].values():
        state["status"] = "review"
    second = deepcopy(first)
    second["id"] = "test/second"
    save_json(tmp_path / "segments.json", {
        "schema_version": SCHEMA_VERSION, "input_root": str(tmp_path), "records": [first, second],
    })
    app = QApplication.instance() or QApplication([])
    window = ReviewWindow(tmp_path)
    window.records.setCurrentRow(0)
    for combo in window.decisions.values():
        combo.setCurrentIndex(combo.findData("accepted"))
    assert window.save_current()
    assert "确认保留 / 确认保留" in window.records.item(0).text()
    assert "待核对" not in window.records.item(0).text()
    # 保存时下拉框选择待核对，就必须忠实保留该状态。
    combo = window.decisions["place"]
    combo.setCurrentIndex(combo.findData("review"))
    assert window.save_current()
    assert "确认保留 / 待核对" in window.records.item(0).text()
    window.delete_current()
    assert window.records.item(0).isHidden()
    assert window.record["id"] == "test/second"
    saved = load_manifest(tmp_path / "segments.json")["records"][0]
    assert saved["annotation"]["deleted"]
    assert (tmp_path / first["hdf5"]).exists()
    window.close()
    window = ReviewWindow(tmp_path)
    assert window.records.count() == 1
    window.show_all.setChecked(True)
    window.records.setCurrentRow(0)
    assert "已删除" in window.records.item(0).text()
    assert window.restore_button.isEnabled()
    assert not window.save_button.isEnabled()
    window.restore_current()
    assert "确认保留 / 待核对" in window.records.item(0).text()
    assert window.save_button.isEnabled()
    window.show_all.setChecked(False)
    window.records.setCurrentRow(0)
    window.delete_current()
    window.delete_current()
    assert window.record is None
    assert window.video is None
    assert not window.delete_button.isEnabled()
    window.close()
    app.processEvents()
