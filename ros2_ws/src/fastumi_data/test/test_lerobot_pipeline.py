"""端到端验证 convert_mcap 的 LeRobot 模式：补帧切分、写入、回读、padding 窗口与失败处理。

需要真实 LeRobot 环境的用例在 LeRobot Python 不可用时跳过（解析规则同 ``--lerobot-python``）。
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

import fastumi_data.mcap_converter as converter
from fastumi_data.lerobot_export import (
    LeRobotExportError,
    resolve_lerobot_python,
    writer_environment,
)
from fastumi_data.link7_arrays import letterbox_rgb_uint8
from fastumi_data.mcap_converter import main, run_batch
from test_link7_pipeline import arguments, calibration, make_record  # noqa: F401  calibration 为夹具


def _lerobot_python():
    """返回可用的 LeRobot Python，不可用时返回 None。"""
    try:
        return resolve_lerobot_python("")
    except LeRobotExportError:
        return None


LEROBOT_PYTHON = _lerobot_python()
requires_lerobot = pytest.mark.skipif(LEROBOT_PYTHON is None, reason="没有可用的 LeRobot Python")

# 在 LeRobot 环境中按训练配置（n_obs_steps=2、horizon=16）读取数据集并输出 JSON 摘要。
READBACK_SCRIPT = r'''
import json, sys
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id, root = sys.argv[1], sys.argv[2]
fps = 30
delta = {
    "observation.state": [-1 / fps, 0.0],
    "observation.images.front": [-1 / fps, 0.0],
    "action": [k / fps for k in range(-1, 15)],
}
dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta)
episodes = dataset.meta.episodes
summary = {"episodes": [], "info": dataset.meta.info["features"]["observation.images.front"]["info"],
           "tasks": sorted(dataset.meta.tasks.index.tolist())}
for index in range(dataset.num_episodes):
    start = int(episodes[index]["dataset_from_index"])
    stop = int(episodes[index]["dataset_to_index"])
    count = stop - start
    first = dataset[start]
    last_start = dataset[start + count - 8]
    rows = dataset.hf_dataset.select(range(start, stop))
    summary["episodes"].append({
        "length": count,
        "task": first["task"],
        "state": np.stack([np.asarray(x, dtype=np.float32) for x in rows["observation.state"]]).tolist(),
        "first_state_is_pad": first["observation.state_is_pad"].tolist(),
        "first_image_is_pad": first["observation.images.front_is_pad"].tolist(),
        "first_action_is_pad": first["action_is_pad"].tolist(),
        "last_start_action_is_pad": last_start["action_is_pad"].tolist(),
        "image_shape": list(first["observation.images.front"].shape),
        "action_shape": list(first["action"].shape),
    })
print(json.dumps(summary))
'''


def readback(root: Path, repo_id: str, tmp_path: Path) -> dict:
    """在 LeRobot 子进程中读取数据集。"""
    script = tmp_path / "readback.py"
    script.write_text(READBACK_SCRIPT, encoding="utf-8")
    completed = subprocess.run(
        [str(LEROBOT_PYTHON), "-I", str(script), repo_id, str(root)],
        capture_output=True, text=True, env=writer_environment(), check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def lerobot_arguments(source: Path, output: Path, calibration: dict, *extra: str) -> list:
    """组装 LeRobot 模式参数。"""
    return arguments(source, output, calibration, "--repo-id", "local/fastumi_test", *extra,
                     output_format="lerobot")


@requires_lerobot
def test_lerobot_conversion_fills_splits_and_reads_back(tmp_path: Path, calibration) -> None:
    """短缺口补齐、长缺口切分；数据集可读，元数据、任务、数值和 padding 窗口符合契约。"""
    data = tmp_path / "data"
    make_record(data, "task_a", "r1", drop_images=[30])
    make_record(data, "task_b", "r2", drop_images=[35, 36, 37])
    output = tmp_path / "out"
    assert run_batch(lerobot_arguments(data, output, calibration)) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "succeeded" and manifest["format"] == "lerobot-v3.0"
    assert manifest["zarr"] is None and manifest["lerobot_export"]["max_fill_gap"] == 2
    assert manifest["lerobot"]["episodes"] == 3
    root = output / "lerobot"
    assert not (output / ".lerobot_staging").exists() and not (output / "dataset.zarr").exists()
    hdf5_path = output / "records/task_a/r1/episodes/episode_0000.hdf5"
    assert hdf5_path.is_file()
    # LeRobot 模式在暂存前 letterbox，HDF5 只保存 224 图像，且与 Zarr 模式源分辨率帧的
    # letterbox 结果逐像素一致。
    zarr_output = tmp_path / "zarr_out"
    assert run_batch(arguments(data, zarr_output, calibration)) == 0
    with h5py.File(hdf5_path, "r") as handle:
        assert handle.attrs["image_preprocessing"] == "letterbox_inter_area_black_border"
        assert handle.attrs["image_size"] == 224
        boxed_frames = handle["observations/images/front"][:]
    with h5py.File(zarr_output / "records/task_a/r1/episodes/episode_0000.hdf5", "r") as handle:
        assert handle.attrs["image_preprocessing"] == "none" and handle.attrs["image_size"] == 0
        source_frames = handle["observations/images/front"][:]
    assert boxed_frames.shape == (len(source_frames), 224, 224, 3)
    assert source_frames.shape[1:] != (224, 224, 3)
    np.testing.assert_array_equal(
        boxed_frames, np.stack([letterbox_rgb_uint8(frame) for frame in source_frames])
    )

    meta = json.loads((root / "meta/fastumi.json").read_text())
    assert meta["action_time_reference"] == "image_timestamp"
    assert meta["pose_frame"] == "episode_start_tcp" and meta["video_codec"] == "h264"
    assert meta["end_frame"] == "tcp" and meta["source_format"] == "fastumi_ros2_tcp_v1"
    assert meta["umi_camera_from_tcp"] == yaml.safe_load(calibration["umi_tcp"].read_text())["T_u_cam_tcp"]
    names = [item["name"] for item in meta["episodes"]]
    assert names == [
        "task_a/r1/episode_0000",
        "task_b/r2/episode_0000/segment_00",
        "task_b/r2/episode_0000/segment_01",
    ]
    first, split_a, split_b = meta["episodes"]
    assert len(first["filled_frames"]) == 1 and first["split"] is False
    assert first["source_hdf5"] == "records/task_a/r1/episodes/episode_0000.hdf5"
    assert split_a["split"] and split_b["source_rows"][0] == split_a["source_rows"][1]
    assert split_b["nominal_frame_start"] == split_a["source_rows"][1] + 3
    for path in (root / "meta").rglob("*"):
        if path.is_file():
            assert str(output / ".lerobot_staging").encode() not in path.read_bytes()
    record_report = json.loads((output / "records/task_b/r2/record_report.json").read_text())
    assert record_report["lerobot"]["status"] == "succeeded"

    summary = readback(root, "local/fastumi_test", tmp_path)
    assert summary["info"]["video.codec"] == "h264"
    # 合成 bag 的 START 事件自带任务名 "task"，优先于记录任务名。
    assert summary["tasks"] == ["task"]
    episodes = summary["episodes"]
    assert [item["length"] for item in episodes] == [item["frames"] for item in meta["episodes"]]
    with h5py.File(hdf5_path, "r") as handle:
        qpos = handle["observations/qpos"][:]
    state = np.asarray(episodes[0]["state"], dtype=np.float64)
    keep = np.delete(np.arange(len(state)), first["filled_frames"])
    np.testing.assert_allclose(state[keep, :3], qpos[:, :3], atol=1e-6)
    np.testing.assert_allclose(
        Rotation.from_rotvec(state[keep, 3:6]).as_matrix(),
        Rotation.from_quat(qpos[:, 3:7]).as_matrix(), atol=1e-5,
    )
    np.testing.assert_allclose(state[keep, 6], qpos[:, 7], atol=1e-6)
    # 第二段从零计时，但位姿仍在原 episode 起始系，首帧不是原点。
    assert np.linalg.norm(np.asarray(episodes[2]["state"][0][:3])) > 0
    for item in episodes:
        assert item["image_shape"] == [2, 3, 224, 224] and item["action_shape"] == [16, 7]
        assert item["first_state_is_pad"] == [True, False]
        assert item["first_image_is_pad"] == [True, False]
        assert item["first_action_is_pad"] == [True] + [False] * 15
        # 最后一个训练起点 t=N-8：窗口 t-1…t+14 延伸到 N+6，末尾 7 步为 padding。
        assert item["last_start_action_is_pad"] == [False] * 9 + [True] * 7


@requires_lerobot
def test_record_with_all_segments_dropped_is_marked_failed(tmp_path: Path, calibration) -> None:
    """--max-fill-gap 0 时短片段全部舍弃的记录标记失败，其余记录仍正常发布。"""
    data = tmp_path / "data"
    make_record(data, "task", "good")
    make_record(data, "task", "short", episodes=((2, 30),), drop_images=[15])
    output = tmp_path / "out"
    assert run_batch(lerobot_arguments(data, output, calibration, "--max-fill-gap", "0")) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    reports = {item["record_id"]: item["lerobot"] for item in manifest["sources"]}
    assert reports["task/good"]["status"] == "succeeded"
    assert reports["task/short"]["status"] == "failed"
    assert reports["task/short"]["error"] == "片段全部被舍弃"
    assert {item["code"] for item in reports["task/short"]["dropped_segments"]} == {"SEGMENT_TOO_SHORT"}
    assert manifest["lerobot"]["episodes"] == 1


@pytest.fixture
def fake_lerobot(monkeypatch, tmp_path: Path):
    """替换 LeRobot 解析与预检，使失败路径不依赖真实 LeRobot 环境。"""
    monkeypatch.setattr(converter, "resolve_lerobot_python", lambda _value: tmp_path / "python")
    monkeypatch.setattr(converter, "preflight_lerobot", lambda _python: {"status": "succeeded"})


def test_zero_valid_episodes_fail_without_dataset(tmp_path: Path, calibration, fake_lerobot) -> None:
    """全部被质量拒绝时整次转换失败，不发布 LeRobot 数据集。"""
    data = tmp_path / "data"
    make_record(data, "task", "r", episodes=((2, 10),))
    output = tmp_path / "out"
    assert run_batch(lerobot_arguments(data, output, calibration)) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["error"] == "没有任何有效 episode"
    assert manifest["sources"][0]["lerobot"]["error"] == "没有有效 episode"
    assert not (output / "lerobot").exists()


def test_export_failure_keeps_hdf5_npz_and_diagnostics(
    tmp_path: Path, calibration, fake_lerobot, monkeypatch,
) -> None:
    """导出失败时清单标记失败，保留 HDF5、临时 .npz，不发布数据集。"""
    def broken_export(*_args, **_kwargs):
        raise LeRobotExportError("模拟写入失败")

    monkeypatch.setattr(converter, "export_lerobot", broken_export)
    data = tmp_path / "data"
    make_record(data, "task", "r")
    output = tmp_path / "out"
    assert run_batch(lerobot_arguments(data, output, calibration)) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "模拟写入失败" in manifest["error"]
    assert (output / "records/task/r/episodes/episode_0000.hdf5").is_file()
    assert list((output / ".lerobot_staging/npz/task/r").glob("*.npz"))
    assert not (output / "lerobot").exists()


def test_unusable_lerobot_python_fails_before_output(tmp_path: Path, calibration, capsys) -> None:
    """指定的解释器没有 LeRobot 时预检失败，不创建输出目录。"""
    data = tmp_path / "data"
    make_record(data, "task", "r")
    output = tmp_path / "out"
    code = run_batch(lerobot_arguments(data, output, calibration, "--lerobot-python", "/usr/bin/python3"))
    assert code == 1 and not output.exists()
    assert "LeRobot 写入器（preflight）失败" in capsys.readouterr().err
    missing = run_batch(
        lerobot_arguments(data, output, calibration, "--lerobot-python", str(tmp_path / "none"))
    )
    assert missing == 1 and not output.exists()
    assert "--lerobot-python" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra",
    [("--max-fill-gap", "-1"), ("--max-fill-gap", "1.5"), ("--repo-id", "bad")],
)
def test_invalid_lerobot_options_are_usage_errors(tmp_path: Path, calibration, extra) -> None:
    """负数或非整数补帧上限、格式错误的 repo-id 都是用法错误。"""
    data = tmp_path / "data"
    make_record(data, "task", "r")
    with pytest.raises(SystemExit):
        main(lerobot_arguments(data, tmp_path / "out", calibration, *extra))
    assert not (tmp_path / "out").exists()


def test_lerobot_mode_requires_repo_id(tmp_path: Path, calibration) -> None:
    """默认 LeRobot 模式缺少 --repo-id 时报用法错误。"""
    data = tmp_path / "data"
    make_record(data, "task", "r")
    with pytest.raises(SystemExit):
        main(arguments(data, tmp_path / "out", calibration, output_format="lerobot"))
