"""将单轮遥操数据裁剪并原子保存为 HDF5 和 MP4。"""

from __future__ import annotations

from itertools import chain
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import h5py
import imageio_ffmpeg
import numpy as np

from tracker_teleoperated.recorder_core import CAMERA_NAME, EpisodeBuffer


class EpisodeAlignmentError(ValueError):
    """表示录制缺少可用的七轴动作时间范围。"""


def alignment_bounds(episode: EpisodeBuffer) -> tuple[float, float]:
    """返回首条动作至停止按键的闭区间。"""
    if not episode.joint_action:
        raise EpisodeAlignmentError("本轮无机械臂指令，已丢弃")
    timestamps = np.asarray([item[0] for item in episode.joint_action])
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise EpisodeAlignmentError("机械臂指令时间戳无效或逆序，已丢弃")
    return float(timestamps[0]), float(timestamps[-1])


def _window(records, bounds):
    """过滤录制边界外的时序记录。"""
    return [item for item in records if bounds[0] <= item[0] <= bounds[1]]


def _times(records) -> np.ndarray:
    """取得浮点秒时间戳列。"""
    return np.asarray([item[0] for item in records], dtype=np.float64)


def _values(records, width: int) -> np.ndarray:
    """取得固定宽度的 float32 值矩阵。"""
    return np.asarray([item[1] for item in records], dtype=np.float32).reshape(
        len(records), width
    )


def write_hdf5(path: Path, episode: EpisodeBuffer, bounds) -> int:
    """写入版本化 HDF5，并返回窗口内相机帧数。"""
    joints = _window(episode.joint_state, bounds)
    actions = _window(episode.joint_action, bounds)
    grip_states = _window(episode.gripper_state, bounds)
    grip_actions = _window(episode.gripper_action, bounds)
    tracker = _window(episode.tracker_pose, bounds)
    camera_times = np.fromiter(
        (stamp for stamp, _ in episode.images if bounds[0] <= stamp <= bounds[1]),
        dtype=np.float64,
    )
    with h5py.File(path, "w") as root:
        root.attrs["sim"] = False
        root.attrs["format_version"] = "rm75-tracker-single-arm-v1"
        root.attrs["camera_names"] = np.asarray([CAMERA_NAME], dtype="S")
        root.attrs["alignment_reference"] = "joint_action"
        root.attrs["alignment_start_timestamp"] = bounds[0]
        root.attrs["alignment_end_timestamp"] = bounds[1]
        observations = root.create_group("observations")
        action = root.create_group("action")

        joint = observations.create_group("joint_state")
        joint.create_dataset("qpos", data=_values(joints, 7))
        joint.create_dataset("timestamp", data=_times(joints))
        gripper = observations.create_group("gripper_state")
        gripper.create_dataset("position", data=_values(grip_states, 1))
        gripper.create_dataset("timestamp", data=_times(grip_states))
        pose = observations.create_group("tracker_pose")
        pose.attrs["frame_id"] = episode.tracker_frame_id
        pose.attrs["position_unit"] = "m"
        pose.attrs["orientation_order"] = "xyzw"
        pose.create_dataset("position", data=_values(tracker, 3))
        pose.create_dataset(
            "orientation",
            data=np.asarray([item[2] for item in tracker], dtype=np.float32).reshape(
                len(tracker), 4
            ),
        )
        pose.create_dataset("timestamp", data=_times(tracker))
        observations.create_group("images").create_dataset(
            f"cam_{CAMERA_NAME}_timestamp", data=camera_times
        )
        joint_action = action.create_group("joint_action")
        joint_action.create_dataset("position", data=_values(actions, 7))
        joint_action.create_dataset("timestamp", data=_times(actions))
        gripper_action = action.create_group("gripper_action")
        gripper_action.create_dataset("position", data=_values(grip_actions, 1))
        gripper_action.create_dataset("timestamp", data=_times(grip_actions))
    return len(camera_times)


def write_video(path: Path, episode: EpisodeBuffer, bounds, fps: int) -> int:
    """经 FFmpeg 将时间窗口内的 JPEG 流编码为 H.264。"""
    frames = (
        jpeg for stamp, jpeg in episode.images if bounds[0] <= stamp <= bounds[1]
    )
    first = next(frames, None)
    if first is None:
        return 0
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
        "-y", "-f", "mjpeg", "-framerate", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=errors, start_new_session=True,
        )
        count = 0
        try:
            assert process.stdin is not None
            for jpeg in chain((first,), frames):
                process.stdin.write(jpeg)
                count += 1
            process.stdin.close()
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        errors.seek(0)
        detail = errors.read().decode(errors="replace").strip()
    if code:
        raise RuntimeError(f"视频编码失败: {detail}")
    return count


def next_episode_path(output_root: Path) -> Path:
    """按已有最大 episode 编号选取下一个路径。"""
    indices = []
    for path in output_root.glob("episode_*"):
        suffix = path.name.removeprefix("episode_")
        if suffix.isdigit():
            indices.append(int(suffix))
    return output_root / f"episode_{max(indices, default=-1) + 1}"


def write_episode(output_root: Path, episode: EpisodeBuffer, fps: int):
    """写入临时目录后公布完整 episode，失败时清理临时数据。"""
    started = time.monotonic()
    bounds = alignment_bounds(episode)
    output_root.mkdir(parents=True, exist_ok=True)
    target = next_episode_path(output_root)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=output_root))
    try:
        expected = write_hdf5(temporary / "proprio.hdf5", episode, bounds)
        actual = write_video(temporary / f"{CAMERA_NAME}.mp4", episode, bounds, fps)
        if actual != expected:
            raise RuntimeError("视频帧数与 HDF5 相机时间戳数量不一致")
        if target.exists():
            raise FileExistsError(f"Episode 路径已存在: {target}")
        os.rename(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target, len(episode.joint_action), actual, time.monotonic() - started
