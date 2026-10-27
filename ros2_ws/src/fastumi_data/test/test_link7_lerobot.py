"""验证 LeRobot 导出的纯数值逻辑：旋转向量、丢帧补齐、切分、任务名与 LeRobot Python 解析。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import fastumi_data.lerobot_export as lerobot_export
from fastumi_data.lerobot_export import (
    LEROBOT_PYTHON_ENV,
    LeRobotExportError,
    resolve_lerobot_python,
    run_writer,
    validate_repo_id,
    writer_environment,
)
from fastumi_data.link7_arrays import letterbox_rgb_uint8, quaternion_pose_to_rotvec
from fastumi_data.link7_lerobot import (
    SEGMENT_TOO_SHORT,
    fill_rows,
    plan_filled_segments,
    prepare_lerobot_episode,
    resolve_task,
    split_rows,
)


def make_qpos(count: int, gripper=None) -> np.ndarray:
    """构造平滑运动的 ``(count, 8)`` qpos：位置线性变化、绕 z 轴匀速旋转。"""
    t = np.arange(count, dtype=np.float64)
    quat = Rotation.from_rotvec(np.stack([np.zeros(count), np.zeros(count), 0.05 * t], 1)).as_quat()
    grip = np.linspace(0.0, 1.0, count) if gripper is None else np.asarray(gripper, dtype=np.float64)
    return np.concatenate((np.stack([0.01 * t, -0.02 * t, 0.003 * t], 1), quat, grip[:, None]), 1)


def test_rotvec_identity_general_and_near_pi() -> None:
    """单位旋转为零向量；一般旋转与接近 π 的旋转均与原四元数表示同一旋转矩阵。"""
    rotvecs = np.array([[0, 0, 0], [0.3, -0.2, 0.5], [0, 0, np.pi - 1e-6], [np.pi - 1e-4, 0, 0]])
    quat = Rotation.from_rotvec(rotvecs).as_quat()
    qpos = np.concatenate((np.ones((4, 3)), quat, np.full((4, 1), 0.5)), axis=1)
    state = quaternion_pose_to_rotvec(qpos)
    np.testing.assert_allclose(state[0, 3:6], 0.0, atol=1e-12)
    np.testing.assert_allclose(
        Rotation.from_rotvec(state[:, 3:6]).as_matrix(), Rotation.from_quat(quat).as_matrix(), atol=1e-9
    )
    assert np.all(np.linalg.norm(state[:, 3:6], axis=1) <= np.pi + 1e-9)
    # 四元数取反表示同一旋转，转换结果不变。
    np.testing.assert_allclose(quaternion_pose_to_rotvec(np.concatenate(
        (qpos[:, :3], -quat, qpos[:, 7:]), 1))[1], state[1], atol=1e-12)


@pytest.mark.parametrize(
    "max_gap, expected",
    [(0, [(0, 5), (5, 9), (9, 12)]), (2, [(0, 9), (9, 12)]), (3, [(0, 12)])],
)
def test_split_rows_by_missing_frame_count(max_gap, expected) -> None:
    """按缺失帧数而非有效帧跨度切分：0 全切，恰好等于上限补齐，超过上限切分。"""
    dropped = np.zeros(12, dtype=np.int32)
    dropped[5], dropped[9] = 2, 3
    assert split_rows(dropped, max_gap) == expected


def test_split_rows_rejects_negative_gap() -> None:
    """负的上限是调用错误。"""
    with pytest.raises(ValueError):
        split_rows(np.zeros(3), -1)


def test_single_gap_fill_uses_slerp_lerp_and_earlier_image_when_equidistant() -> None:
    """单帧缺口：位置中点、SLERP 中点、等距取较早帧图像，名义帧号连续。"""
    qpos = make_qpos(6)
    dropped = np.zeros(6, dtype=np.int32)
    dropped[3] = 1
    segment = fill_rows(qpos, dropped, (0, 6), gripper_nearest=False)
    assert len(segment) == 7
    np.testing.assert_array_equal(segment.source_row, [0, 1, 2, 2, 3, 4, 5])
    np.testing.assert_array_equal(segment.filled, [0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(segment.nominal_frame, np.arange(7))
    np.testing.assert_allclose(segment.state[3, :3], (qpos[2, :3] + qpos[3, :3]) / 2)
    np.testing.assert_allclose(segment.state[3, 5], (0.05 * 2 + 0.05 * 3) / 2, atol=1e-12)
    np.testing.assert_allclose(segment.state[3, 6], (qpos[2, 7] + qpos[3, 7]) / 2)
    original = np.delete(segment.state, 3, axis=0)
    np.testing.assert_allclose(original, quaternion_pose_to_rotvec(qpos), atol=1e-12)


def test_consecutive_gap_fill_and_binary_gripper_stays_binary() -> None:
    """连续两帧缺口：比例 1/3、2/3，图像取最近邻；二值夹爪不产生中间值。"""
    gripper = np.array([0, 0, 1, 1, 1], dtype=np.float64)
    qpos = make_qpos(5, gripper)
    dropped = np.zeros(5, dtype=np.int32)
    dropped[2] = 2
    segment = fill_rows(qpos, dropped, (0, 5), gripper_nearest=True)
    np.testing.assert_array_equal(segment.source_row, [0, 1, 1, 2, 2, 3, 4])
    np.testing.assert_array_equal(segment.state[:, 6], [0, 0, 0, 1, 1, 1, 1])
    assert set(np.unique(segment.state[:, 6])) <= {0.0, 1.0}
    np.testing.assert_allclose(segment.state[2, :3], qpos[1, :3] + (qpos[2, :3] - qpos[1, :3]) / 3)
    np.testing.assert_allclose(segment.state[3, 5], 0.05 + 0.05 * 2 / 3, atol=1e-12)
    continuous = fill_rows(qpos, dropped, (0, 5), gripper_nearest=False)
    np.testing.assert_allclose(continuous.state[2:4, 6], [1 / 3, 2 / 3])


def test_length_is_checked_after_fill() -> None:
    """14 个有效帧补齐 2 帧后达到 16 帧保留；超过上限切分后的短片段被舍弃。"""
    qpos = make_qpos(14)
    dropped = np.zeros(14, dtype=np.int32)
    dropped[7] = 2
    kept, removed = plan_filled_segments(qpos, dropped, 2, gripper_nearest=False)
    assert [len(item) for item in kept] == [16] and removed == []
    kept, removed = plan_filled_segments(qpos, dropped, 1, gripper_nearest=False)
    assert kept == []
    assert [(item["code"], item["segment_index"], item["frames_after_fill"]) for item in removed] == [
        (SEGMENT_TOO_SHORT, 0, 7), (SEGMENT_TOO_SHORT, 1, 7),
    ]


def test_split_segment_restarts_index_but_keeps_frame_and_mapping() -> None:
    """切分后的第二段帧号从零写入，但位姿仍在原起始系，并保留源行和原名义帧号。"""
    qpos = make_qpos(40)
    dropped = np.zeros(40, dtype=np.int32)
    dropped[20] = 5
    kept, removed = plan_filled_segments(qpos, dropped, 2, gripper_nearest=False, minimum_frames=16)
    assert removed == [] and [item.rows for item in kept] == [(0, 20), (20, 40)]
    second = kept[1]
    assert second.split_index == 1
    np.testing.assert_array_equal(second.source_row, np.arange(20, 40))
    np.testing.assert_array_equal(second.nominal_frame, np.arange(25, 45))
    np.testing.assert_allclose(second.state, quaternion_pose_to_rotvec(qpos[20:40]), atol=1e-12)
    assert not np.allclose(second.state[0, :3], 0.0)


def write_hdf5(path: Path, qpos: np.ndarray, dropped: np.ndarray) -> np.ndarray:
    """写最小 Link7 HDF5（只含 LeRobot 预处理读取的字段），返回原图。"""
    count = len(qpos)
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(count, 48, 64, 3), dtype=np.uint8)
    with h5py.File(path, "w") as root:
        observations = root.create_group("observations")
        observations.create_group("images").create_dataset("front", data=images)
        observations.create_dataset("qpos", data=qpos)
        frames = np.cumsum(np.r_[0, 1 + dropped[1:]])
        observations.create_dataset("timestamp", data=frames / 30.0)
        observations.create_dataset("nominal_timestamp_ns", data=(frames * 1e9 / 30).astype(np.int64))
        observations.create_dataset("image_timestamp_ns", data=(frames * 1e9 / 30).astype(np.int64))
        observations.create_dataset("source_image_index", data=np.arange(count, dtype=np.int64) + 100)
        observations.create_dataset("dropped_frames_before", data=dropped.astype(np.int32))
        root.create_dataset("action", data=qpos)
    return images


def test_prepare_episode_writes_npz_with_letterbox_and_provenance(tmp_path: Path) -> None:
    """``.npz`` 含 letterbox 图像、state==action、补齐帧取最近邻图像及溯源字段。"""
    qpos = make_qpos(50, np.r_[np.zeros(20), np.ones(30)])
    dropped = np.zeros(50, dtype=np.int32)
    dropped[10], dropped[30] = 1, 4
    path = tmp_path / "records" / "r" / "episode_0000.hdf5"
    path.parent.mkdir(parents=True)
    images = write_hdf5(path, qpos, dropped)
    result = prepare_lerobot_episode(
        path, "task/r/episode_0000", tmp_path / "npz", 2, "nearest", "pick", relative_to=tmp_path,
    )
    assert result["dropped_segments"] == []
    first, second = result["segments"]
    assert first["name"] == "task/r/episode_0000/segment_00" and first["split"] is True
    assert first["source_hdf5"] == "records/r/episode_0000.hdf5"
    assert first["frames"] == 31 and first["filled_frames"] == [10]
    assert second["source_rows"] == [30, 50] and second["nominal_frame_start"] == 35
    with np.load(first["npz"]) as arrays:
        assert arrays["images"].shape == (31, 224, 224, 3) and arrays["images"].dtype == np.uint8
        assert arrays["state"].dtype == np.float32
        np.testing.assert_array_equal(arrays["state"], arrays["action"])
        np.testing.assert_array_equal(arrays["images"][10], letterbox_rgb_uint8(images[9]))
        np.testing.assert_array_equal(arrays["images"][11], letterbox_rgb_uint8(images[10]))
        assert arrays["source_image_index"][10] == 109
        assert set(np.unique(arrays["state"][:, 6])) <= {0.0, 1.0}
    json.dumps(result)



def test_letterbox_is_identity_on_target_size() -> None:
    """224×224 输入经 letterbox 逐像素不变，转换器提前缩放后再次 letterbox 不改变图像。"""
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    np.testing.assert_array_equal(letterbox_rgb_uint8(image), image)
    boxed = letterbox_rgb_uint8(rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8))
    np.testing.assert_array_equal(letterbox_rgb_uint8(boxed), boxed)


def test_resolve_task_fallback_order() -> None:
    """任务名依次取 episode、记录、记录标识，忽略空白。"""
    assert resolve_task("pick", "record", "id") == "pick"
    assert resolve_task(" ", "record", "id") == "record"
    assert resolve_task(None, "", "task/r") == "task/r"


def test_validate_repo_id() -> None:
    """repo-id 必须是两段非空名称。"""
    assert validate_repo_id(" user/data ") == "user/data"
    for value in ("", "data", "a/b/c", "/data", "a b/c"):
        with pytest.raises(ValueError):
            validate_repo_id(value)


def make_executable(path: Path) -> Path:
    """写一个可执行文件。"""
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_lerobot_python_resolution_order(tmp_path: Path, monkeypatch) -> None:
    """命令行优先，其次环境变量，最后缺省路径；都不可用时报错且不回退到当前解释器。"""
    explicit = make_executable(tmp_path / "explicit")
    from_env = make_executable(tmp_path / "env")
    default = make_executable(tmp_path / "default")
    monkeypatch.setattr(lerobot_export, "DEFAULT_LEROBOT_PYTHON", default)
    assert resolve_lerobot_python(str(explicit), {LEROBOT_PYTHON_ENV: str(from_env)}) == explicit
    assert resolve_lerobot_python("", {LEROBOT_PYTHON_ENV: str(from_env)}) == from_env
    assert resolve_lerobot_python("", {}) == default
    with pytest.raises(LeRobotExportError, match=LEROBOT_PYTHON_ENV):
        resolve_lerobot_python("", {LEROBOT_PYTHON_ENV: str(tmp_path / "missing")})
    monkeypatch.setattr(lerobot_export, "DEFAULT_LEROBOT_PYTHON", tmp_path / "missing")
    with pytest.raises(LeRobotExportError, match="--output-format zarr"):
        resolve_lerobot_python("", {})
    (tmp_path / "plain").write_text("", encoding="utf-8")
    with pytest.raises(LeRobotExportError):
        resolve_lerobot_python(str(tmp_path / "plain"), {})


def test_writer_subprocess_does_not_inherit_ros_pythonpath(tmp_path: Path, monkeypatch) -> None:
    """子进程环境移除 PYTHONPATH，并以 -I 隔离模式运行；进度行被解析，其余输出写入日志。"""
    assert "PYTHONPATH" not in writer_environment({"PYTHONPATH": "/opt/ros", "A": "1"})
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, os, sys\n"
        "print('FASTUMI_PROGRESS 1 2', flush=True)\n"
        "print('noise')\n"
        "json.dump({'status': 'succeeded', 'pythonpath': os.environ.get('PYTHONPATH'),\n"
        "           'isolated': sys.flags.isolated, 'offline': os.environ.get('HF_HUB_OFFLINE')},\n"
        "          open(sys.argv[2], 'w'))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(lerobot_export, "WRITER_SCRIPT", probe)
    monkeypatch.setenv("PYTHONPATH", "/opt/ros/jazzy/lib/python3.12/site-packages")
    seen = []
    result = run_writer(Path(sys.executable), {"mode": "probe"}, tmp_path / "work",
                        lambda done, total: seen.append((done, total)))
    assert result["pythonpath"] is None and result["isolated"] == 1 and result["offline"] == "1"
    assert seen == [(1, 2)]
    assert "noise" in (tmp_path / "work" / "writer_probe.log").read_text(encoding="utf-8")


def test_writer_failure_reports_log_path(tmp_path: Path, monkeypatch) -> None:
    """子进程失败时抛出带日志路径的错误，并保留任务文件。"""
    failing = tmp_path / "fail.py"
    failing.write_text("import sys\nsys.exit(3)\n", encoding="utf-8")
    monkeypatch.setattr(lerobot_export, "WRITER_SCRIPT", failing)
    with pytest.raises(LeRobotExportError, match="writer_probe.log"):
        run_writer(Path(sys.executable), {"mode": "probe"}, tmp_path / "work")
    assert (tmp_path / "work" / "job_probe.json").is_file()
    assert os.access(tmp_path / "work" / "writer_probe.log", os.R_OK)
