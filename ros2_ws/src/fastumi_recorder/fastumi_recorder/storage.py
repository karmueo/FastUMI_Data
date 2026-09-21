"""将单轮遥操数据裁剪并原子保存为 HDF5 和 MP4。"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np

from fastumi_recorder.core import CAMERA_NAME, EpisodeBuffer


class EpisodeAlignmentError(ValueError):
    """表示录制缺少可用的七轴动作时间范围。"""


def alignment_bounds(episode: EpisodeBuffer) -> tuple[float, float]:
    """返回首条动作至停止按键的闭区间。"""
    if not episode.joint_action:
        return episode.started_at, episode.stopped_at
    timestamps = np.asarray([item[0] for item in episode.joint_action])
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
        raise EpisodeAlignmentError("机械臂指令时间戳无效或逆序，临时数据已保留")
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


def select_video_samples(episode: EpisodeBuffer, bounds):
    """按数据窗口裁剪视频，并让 H.264 从第一个关键帧开始。"""
    codec = None
    started = False
    for item in episode.images:
        if not bounds[0] <= item[0] <= bounds[1]:
            continue
        if codec is None:
            codec = item[2]
        elif item[2] != codec:
            raise ValueError(
                f"单轮录制包含多种视频编码: {codec}, {item[2]}"
            )
        if codec == "h264" and not started:
            if not item[3]:
                continue
            started = True
        else:
            started = True
        yield item


def write_hdf5(path: Path, episode: EpisodeBuffer, bounds, camera_times) -> int:
    """写入版本化 HDF5，并返回窗口内相机帧数。"""
    joints = _window(episode.joint_state, bounds)
    actions = _window(episode.joint_action, bounds)
    grip_states = _window(episode.gripper_state, bounds)
    grip_actions = _window(episode.gripper_action, bounds)
    tracker = _window(episode.tracker_pose, bounds)
    camera_times = np.asarray(camera_times, dtype=np.float64)
    with h5py.File(path, "w") as root:
        root.attrs["sim"] = False
        root.attrs["format_version"] = "rm75-tracker-single-arm-v1"
        root.attrs["camera_names"] = np.asarray([CAMERA_NAME], dtype="S")
        root.attrs["alignment_reference"] = "joint_action" if episode.joint_action else "recording_window"
        root.attrs["has_joint_action"] = bool(episode.joint_action)
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


def write_video(path: Path, video_samples, fps: int, encoder) -> int:
    """将 JPEG 编码为 H.264，或把已有 H.264 无重编码封装为 MP4。"""
    frames = iter(video_samples)
    first = next(frames, None)
    if first is None:
        return 0
    codec = first[2]
    if codec == "h264":
        command = [
            encoder.executable, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "h264", "-r", str(fps), "-i", "pipe:0", "-an",
            "-c:v", "copy", "-movflags", "+faststart", str(path),
        ]
    elif codec == "mjpeg":
        command = [
            encoder.executable, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "mjpeg", "-framerate", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
        ]
    else:
        raise ValueError(f"不支持的视频编码: {codec}")
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=errors, start_new_session=True,
        )
        encoder.register(process)
        count = 0
        try:
            assert process.stdin is not None
            for _, data, _, _ in chain((first,), frames):
                process.stdin.write(data)
                count += 1
            process.stdin.close()
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            encoder.unregister(process)
        errors.seek(0)
        detail = errors.read().decode(errors="replace").strip()
    if code:
        raise RuntimeError(f"视频封装或编码失败: {detail}")
    return count


def write_episode(temporary: Path, episode: EpisodeBuffer, fps: int, encoder):
    """写入保留的临时目录；失败由管理器保留现场和错误。"""
    bounds = alignment_bounds(episode)
    camera_times = [
        sample[0] for sample in select_video_samples(episode, bounds)
    ]
    expected = write_hdf5(
        temporary / "proprio.hdf5", episode, bounds, camera_times
    )
    actual = write_video(
        temporary / f"{CAMERA_NAME}.mp4",
        select_video_samples(episode, bounds), fps, encoder
    )
    if actual != expected:
        raise RuntimeError("视频帧数与 HDF5 相机时间戳数量不一致")
    return dict(
        joint_samples=len(_window(episode.joint_state, bounds)),
        action_samples=len(_window(episode.joint_action, bounds)),
        gripper_samples=len(_window(episode.gripper_state, bounds)),
        tracker_samples=len(_window(episode.tracker_pose, bounds)),
        image_frames=actual, has_joint_action=bool(episode.joint_action),
    )
