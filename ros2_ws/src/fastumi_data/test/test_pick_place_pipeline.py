"""用合成 MCAP 验证抓放分割、完整视频解码和真实 LeRobot 回读。"""

import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from pick_place.core import load_manifest, save_json, update_annotation
from pick_place.pipeline import main, prepare_stage
from fastumi_data.lerobot_export import resolve_lerobot_python, writer_environment
from test_link7_pipeline import calibration, make_record  # noqa: F401


def cycle_sensor(topic, index, message):
    """把合成录制夹爪设为稳定单次开闭开，保留其他传感器。"""
    if topic == "gripper_state":
        message.raw_openness = message.filtered_openness = 0.6 if 35 <= index < 85 else 0.95
    return message


def analyze_args(source, output, calibration):
    """构造绑定测试 Tracker 外参的串行分析命令。"""
    return ["analyze", "--input", str(source), "--output", str(output), "--jobs", "1",
            "--tracker-camera-extrinsic", str(calibration["tracker"]),
            "--umi-tcp-extrinsic", str(calibration["umi_tcp"]), "--urdf", str(calibration["urdf"])]


@pytest.mark.parametrize("mode", ["compressed", "h264"])
def test_mcap_complete_decode_split_rebase_and_resume(tmp_path, calibration, mode):
    """START 位于 GOP 内时也能完整解码；共享交界、独立归零且重跑保留人工边界。"""
    if mode == "h264":
        pytest.importorskip("ffmpeg_image_transport_msgs.msg")
    source, output = tmp_path / "source", tmp_path / "output"
    make_record(source, "task", "record", mode=mode, frames=120, episodes=((2, 117),), sensor_override=cycle_sensor)
    args = analyze_args(source, output, calibration)
    assert main(args) == 0
    manifest = load_manifest(output / "segments.json")
    record = manifest["records"][0]
    assert not record["quality_errors"]
    assert record["events"]["passed"]
    assert all(value["status"] == "auto" for value in record["annotation"]["stages"].values())
    pick = prepare_stage(record, "pick", "binary", output, tmp_path / "npz")
    place = prepare_stage(record, "place", "binary", output, tmp_path / "npz")
    assert pick["frames"] + place["frames"] == len(record["image_stamps_ns"]) + 1
    with np.load(pick["npz"]) as first, np.load(place["npz"]) as second:
        np.testing.assert_array_equal(first["images"][-1], second["images"][0])
        np.testing.assert_allclose(second["state"][0, :6], 0, atol=1e-7)
        assert first["state"][-1, 6] == second["state"][0, 6] == 0
    update_annotation(record, bounds={"split": record["annotation"]["split"] + 1})
    save_json(output / "segments.json", manifest)
    assert main(args) == 0
    restored = load_manifest(output / "segments.json")["records"][0]
    assert restored["annotation"] == record["annotation"]
    # 即使文件大小很小，新增采集元信息也必须使人工标注失效。
    (source / "task/record/quality.yaml").write_text("updated: true\n")
    assert main(args) == 0
    invalidated = load_manifest(output / "segments.json")["records"][0]
    assert "invalidated_annotation" in invalidated
    assert all(value["status"] == "review" for value in invalidated["annotation"]["stages"].values())


def test_real_lerobot_export_readback_and_source_change_guard(tmp_path, calibration):
    """真实写入两套数据并按训练窗口读取；覆盖与源变更均有保护。"""
    try:
        python = resolve_lerobot_python()
    except RuntimeError:
        pytest.skip("没有 LeRobot Python")
    source, output = tmp_path / "source", tmp_path / "output"
    make_record(source, "task", "record", frames=120, episodes=((2, 117),), sensor_override=cycle_sensor)
    assert main(analyze_args(source, output, calibration)) == 0
    arguments = ["export", "--input", str(source), "--output", str(output), "--lerobot-python", str(python)]
    assert main(arguments) == 0
    for stage in ("pick", "place"):
        root = output / "lerobot" / stage
        meta = json.loads((root / "meta/fastumi.json").read_text())
        assert meta["stage"] == stage
        assert meta["urdf_sha256"]
        assert np.asarray(meta["umi_camera_from_tcp"]).shape == (4, 4)
        assert meta["episodes"][0]["source_record"] == "task/record"
        assert meta["episodes"][0]["annotation"]["split_source"] == "automatic_midpoint"
        code = '''
import sys, numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset(sys.argv[1], root=sys.argv[2], delta_timestamps={
    "observation.state": [-1/30, 0], "action": [k/30 for k in range(-1, 15)]})
assert dataset.num_episodes == 1
sample = dataset[0]
assert sample["action"].shape == (16, 7)
assert np.allclose(sample["observation.state"][-1, :6], 0, atol=1e-6)
print("readback-ok")
'''
        completed = subprocess.run([str(python), "-I", "-c", code, f"fastumi/{stage}", str(root)],
                                   env=writer_environment(), capture_output=True, text=True, timeout=120)
        assert completed.returncode == 0, completed.stderr
        assert "readback-ok" in completed.stdout
    assert main(arguments) == 2
    before = (output / "lerobot/pick/meta/fastumi.json").read_bytes()
    (source / "task/record/quality.yaml").write_text("changed: true\n")
    assert main(arguments + ["--overwrite"]) == 2
    assert (output / "lerobot/pick/meta/fastumi.json").read_bytes() == before


def test_quality_rejection_has_preview_but_cannot_export(tmp_path, calibration, monkeypatch):
    """坏跟踪记录可以回放审查，但不能通过手动确认绕过同步拒绝。"""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from pick_place.review import preview_path

    def bad_sensor(topic, index, message):
        """注入跟踪失效而不破坏视频。"""
        message = cycle_sensor(topic, index, message)
        if topic == "tracker_status" and index == 60:
            message.pose_valid = False
        return message

    source, output = tmp_path / "source", tmp_path / "output"
    make_record(source, "task", "record", mode="compressed", frames=120, episodes=((2, 117),), sensor_override=bad_sensor)
    assert main(analyze_args(source, output, calibration)) == 0
    record = load_manifest(output / "segments.json")["records"][0]
    assert record["quality_errors"]
    path, name, stamps = preview_path(record, output, source)
    with h5py.File(path, "r") as handle:
        assert len(handle[name]) == len(stamps) == 120
    with pytest.raises(ValueError):
        update_annotation(record, decisions={"pick": {"status": "accepted", "reason": ""}})
