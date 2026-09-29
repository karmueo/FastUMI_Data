"""Build a trainable RM75 Link7 UMI Zarr from hardware MCAP episodes.

All sensor alignment uses rosbag receive nanoseconds. Stored poses are absolute
base_link-to-Link7 xyz (metres) and rotation vectors (radians); UmiDataset makes
the relative 10D training actions when it samples this 7D on-disk contract.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import cv2
import numpy as np
from numcodecs import Blosc
from scipy.spatial.transform import Rotation
import zarr

from convert_hardware_mcap import (
    JOINT_NAMES, ConversionError, _decode_frame, _diagnostic, _fps, _init_worker,
    _read_episode, _write_video,
)
from model.dp.convert_vr_target import letterbox_rgb
from model.dp.diffusion_policy.common.urdf_kinematics import UrdfKinematics


FREQUENCY_HZ = 30
IMAGE_SIZE = 224
IMAGE_BATCH_FRAMES = 16
STEP_NS = round(1_000_000_000 / FREQUENCY_HZ)
REQUIRED_STREAMS = ("joint_state", "joint_action", "gripper_state", "gripper_action")


def _strict_times(records, name):
    """Reject empty or duplicate receive timestamps before causal sampling."""
    times = np.asarray([item[0] for item in records], dtype=np.int64)
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ConversionError(f"{name} 缺少样本或接收时间戳不严格递增")
    return times


def _select_training(data):
    """Select the action window and the common coverage of training streams."""
    series = data["series"]
    times = {name: _strict_times(series[name], name) for name in REQUIRED_STREAMS}
    action_start, action_end = times["joint_action"][[0, -1]]
    if action_start >= action_end:
        raise ConversionError("有效关节指令的时间区间为空")
    camera = [record for record in data["camera"]
              if action_start <= record.bag_ns <= action_end]
    skipped_keyframes = 0
    if data["camera_mode"] == "h264":
        skipped_keyframes = next((i for i, item in enumerate(camera) if item.keyframe), -1)
        if skipped_keyframes < 0:
            raise ConversionError("指令时间区间内没有 H.264 关键帧")
        camera = camera[skipped_keyframes:]
    camera_times = _strict_times([(item.bag_ns, None) for item in camera], "camera")
    dimensions = {(item.width, item.height) for item in camera
                  if item.width and item.height}
    if len(dimensions) > 1:
        raise ConversionError("同一 episode 内相机分辨率发生变化")
    start = max([camera_times[0], action_start] + [item[0] for item in times.values()])
    end = min([camera_times[-1], action_end] + [item[-1] for item in times.values()])
    if start >= end:
        raise ConversionError("相机、关节和夹爪没有共同时间区间")
    query = start + np.arange((end - start) // STEP_NS + 1, dtype=np.int64) * STEP_NS
    if len(query) < 16:
        raise ConversionError("共同时间区间不足 16 个 30 Hz 对齐样本")
    indices = {name: np.searchsorted(values, query, side="right") - 1
               for name, values in times.items()}
    image_indices = np.searchsorted(camera_times, query, side="right") - 1
    return series, camera, query, indices, image_indices, skipped_keyframes


def _decode_images(camera, image_indices, data, spool, video_threads, output):
    """Decode source frames and write aligned RGB samples in bounded batches."""
    counts = np.bincount(image_indices, minlength=len(camera))
    frames = []
    written = 0

    def flush_frames():
        nonlocal written
        if frames:
            end = written + len(frames)
            output[written:end] = np.stack(frames)
            written = end
            frames.clear()

    def retain_frame(frame, count):
        if count:
            rgb = letterbox_rgb(frame, IMAGE_SIZE)
            for _ in range(int(count)):
                frames.append(rgb)
                if len(frames) == IMAGE_BATCH_FRAMES:
                    flush_frames()

    if data["camera_mode"] == "h264":
        with tempfile.TemporaryDirectory(prefix="fastumi-umi-video-") as temp:
            video = Path(temp) / "camera.mp4"
            _write_video(video, camera, "h264", spool, _fps(camera),
                         threads=video_threads)
            capture = cv2.VideoCapture(str(video))
            if not capture.isOpened():
                raise ConversionError("H.264 临时视频无法打开")
            try:
                for index in range(len(camera)):
                    success, frame = capture.read()
                    if not success:
                        raise ConversionError("H.264 解码帧数少于 MCAP 包数")
                    retain_frame(frame, counts[index])
                if capture.read()[0]:
                    raise ConversionError("H.264 解码帧数多于 MCAP 包数")
            finally:
                capture.release()
    else:
        first_shape = None
        for index, record in enumerate(camera):
            frame = _decode_frame(spool, record, data["camera_mode"])
            if first_shape is None:
                first_shape = frame.shape
            elif frame.shape != first_shape:
                raise ConversionError("相机解码帧尺寸不一致")
            retain_frame(frame, counts[index])
    flush_frames()
    if written != len(image_indices):
        raise ConversionError("图像解码帧与对齐时间戳数量不一致")


def _pose(matrices):
    """Convert base_link-to-Link7 transforms to xyz + rotation vectors."""
    return np.concatenate((matrices[:, :3, 3],
                           Rotation.from_matrix(matrices[:, :3, :3]).as_rotvec()),
                          axis=-1).astype(np.float32)


def read_umi_episode(source, urdf_path, work_path, *, video_threads=None):
    """Write one episode to temporary Zarr without returning image arrays."""
    with tempfile.TemporaryFile(mode="w+b") as spool:
        data = _read_episode(source / "bag", spool, require_tracker=False)
        series, camera, query, indices, image_indices, skipped = _select_training(data)
        work = zarr.open_group(str(work_path), mode="w-")
        codec = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
        images = work.create_dataset(
            "camera0_rgb", shape=(len(query), IMAGE_SIZE, IMAGE_SIZE, 3),
            chunks=(IMAGE_BATCH_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3),
            dtype="u1", compressor=codec,
        )
        _decode_images(camera, image_indices, data, spool, video_threads, images)
    sampled = {}
    for name in REQUIRED_STREAMS:
        values = np.asarray([value for _, value in series[name]], dtype=np.float32)
        sampled[name] = values[indices[name]]
    fk = UrdfKinematics(urdf_path, JOINT_NAMES)
    observed = _pose(fk.forward(sampled["joint_state"]))
    targets = _pose(fk.forward(sampled["joint_action"]))
    arrays = {
        "robot0_eef_pos": observed[:, :3],
        "robot0_eef_rot_axis_angle": observed[:, 3:],
        "robot0_gripper_width": sampled["gripper_state"].reshape(-1, 1),
        "action": np.concatenate((targets, sampled["gripper_action"].reshape(-1, 1)), axis=-1),
        "robot0_demo_start_pose": np.repeat(observed[:1], len(query), axis=0),
        "timestamp": query.astype(np.float64) / 1_000_000_000,
        "source_image_index": image_indices.astype(np.int64),
    }
    if any(len(value) != len(query) for value in arrays.values()):
        raise ConversionError("对齐后的字段帧数不一致")
    if any(not np.isfinite(value).all() for key, value in arrays.items()
           if key != "source_image_index"):
        raise ConversionError("输出含非有限数值")
    for key, value in arrays.items():
        work.create_dataset(key, data=value, compressor=codec)
    report = {
        "episode": source.name,
        "source_episode": str(source.resolve()),
        "source_stream_counts": {name: len(series[name]) for name in REQUIRED_STREAMS},
        "input_counts": dict(data["counts"]),
        "invalid_counts": dict(data["invalid"]),
        "bag_minus_header": {
            name: _diagnostic(values) for name, values in data["deltas"].items()
        },
        "bag_minus_camera_pts": _diagnostic(data["pts_deltas"]),
        "camera_pts_header_mismatch_count": data["pts_header_mismatch_count"],
        "camera_mode": data["camera_mode"],
        "source_frames": len(camera),
        "camera_skipped_until_keyframe": skipped,
        "aligned_frames": len(query),
        "unused_source_frames": len(camera) - len(np.unique(image_indices)),
        "repeated_source_frames": len(query) - len(np.unique(image_indices)),
        "start_timestamp": float(arrays["timestamp"][0]),
        "end_timestamp": float(arrays["timestamp"][-1]),
    }
    return report


def _episode_results(episodes, urdf_path, work_root, workers):
    """Yield small reports in numeric order with at most workers jobs queued."""
    if workers == 1:
        for index, episode in enumerate(episodes):
            work_path = work_root / f"{index}.zarr"
            try:
                report = read_umi_episode(episode, urdf_path, work_path)
                yield episode, work_path, report, None
            except Exception as error:
                yield episode, work_path, None, error
        return
    context = multiprocessing.get_context("spawn")
    video_threads = max(1, (os.cpu_count() or 1) // workers)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=_init_worker) as executor:
        remaining = iter(enumerate(episodes))
        pending = deque()
        for _ in range(workers):
            index, episode = next(remaining)
            work_path = work_root / f"{index}.zarr"
            pending.append((episode, work_path, executor.submit(
                read_umi_episode, episode, urdf_path, work_path,
                video_threads=video_threads)))
        while pending:
            episode, work_path, future = pending.popleft()
            try:
                yield episode, work_path, future.result(), None
            except Exception as error:
                yield episode, work_path, None, error
            next_job = next(remaining, None)
            if next_job is not None:
                index, next_episode = next_job
                next_work_path = work_root / f"{index}.zarr"
                pending.append((next_episode, next_work_path, executor.submit(
                    read_umi_episode, next_episode, urdf_path, next_work_path,
                    video_threads=video_threads)))


def _append(data, work_path, codec):
    """Copy one staged episode to the output and verify bounded batches."""
    arrays = zarr.open_group(str(work_path), mode="r")
    start = len(data["action"]) if "action" in data else 0
    count = len(arrays["action"])
    for key, values in arrays.arrays():
        if len(values) != count:
            raise ConversionError(f"临时 Zarr 字段帧数不一致: {key}")
        if key not in data:
            chunk_length = 1 if key == "camera0_rgb" else 1024
            chunks = (chunk_length,) + values.shape[1:]
            data.create_dataset(key, shape=(start + count,) + values.shape[1:],
                                dtype=values.dtype, chunks=chunks, compressor=codec)
        else:
            data[key].resize((start + count,) + values.shape[1:])
        batch_size = IMAGE_BATCH_FRAMES if key == "camera0_rgb" else 1024
        for offset in range(0, count, batch_size):
            batch = values[offset:offset + batch_size]
            destination = slice(start + offset, start + offset + len(batch))
            data[key][destination] = batch
            if not np.array_equal(data[key][destination], batch):
                raise ConversionError(f"Zarr 写入校验失败: {key}")
    return start + count


def convert_umi_episodes(episodes, output, urdf_path, workers):
    """Publish one complete trainable Zarr, skipping invalid source episodes."""
    if output.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖: {output}")
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF 不存在: {urdf_path}")
    # Freeze one URDF snapshot: every worker, the saved copy and the hash
    # must describe the same kinematic chain even if the source later changes.
    urdf_bytes = urdf_path.read_bytes()
    urdf_hash = hashlib.sha256(urdf_bytes).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    report = {
        "source_episodes": [str(item.resolve()) for item in episodes],
        "output": str(output), "frequency": FREQUENCY_HZ,
        "image_size": IMAGE_SIZE, "urdf_sha256": urdf_hash,
        "episodes": [], "skipped": [],
    }
    try:
        root = zarr.open_group(str(stage), mode="w-")
        staged_urdf = stage / "rm_75.urdf"
        staged_urdf.write_bytes(urdf_bytes)
        UrdfKinematics(staged_urdf, JOINT_NAMES)
        root.attrs.update({
            "format": "rm75-umi-pose-v1", "complete": False,
            "frequency": FREQUENCY_HZ, "image_size": IMAGE_SIZE,
            "action_layout": "pose10", "stored_action_layout": "xyz_rotvec_gripper",
            "base_frame": "base_link", "end_frame": "Link7",
            "tool_offset": np.eye(4).tolist(), "position_unit": "m",
            "rotation_unit": "rad", "gripper_representation": "normalized_0_1",
            "joint_names": list(JOINT_NAMES), "urdf_sha256": urdf_hash,
            "timestamp_domain": "rosbag_receive",
        })
        data = root.create_group("data")
        meta = root.create_group("meta")
        work_root = stage / ".work"
        work_root.mkdir()
        codec = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
        ends, names = [], []
        print(f"并发数: {workers}", flush=True)
        for completed, (episode, work_path, episode_report, error) in enumerate(
                _episode_results(episodes, staged_urdf, work_root, workers), 1):
            try:
                if error is not None:
                    report["skipped"].append({"episode": episode.name, "reason": str(error)})
                    print(f"[{completed}/{len(episodes)}] {episode.name}: 跳过: {error}", flush=True)
                    continue
                end = _append(data, work_path, codec)
                episode_report["offset_start"] = ends[-1] if ends else 0
                episode_report["offset_end"] = end
                report["episodes"].append(episode_report)
                names.append(episode.name)
                ends.append(end)
                print(f"[{completed}/{len(episodes)}] {episode.name}: 完成，"
                      f"{episode_report['aligned_frames']} 帧", flush=True)
            finally:
                if work_path.exists():
                    shutil.rmtree(work_path)
        work_root.rmdir()
        report["episode_count"] = len(names)
        report["aligned_frames"] = ends[-1] if ends else 0
        report["skipped_count"] = len(report["skipped"])
        if not names:
            failure_report = output.parent / f"{output.name}.report.json"
            failure_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
            print(f"全部 {len(episodes)} 轮失败；报告: {failure_report}", flush=True)
            return 1
        meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
        meta.create_dataset("episode_names", data=np.asarray(names, dtype="U64"))
        if any(len(data[key]) != ends[-1] for key in data.array_keys()):
            raise ConversionError("Zarr 字段帧数与 episode 边界不一致")
        (stage / "conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        root.attrs["complete"] = True
        if output.exists():
            raise FileExistsError(f"目标已存在，拒绝覆盖: {output}")
        stage.rename(output)
        print(f"总计 {len(episodes)} 轮，成功 {len(names)}，跳过 {len(episodes)-len(names)}；"
              f"训练集: {output}", flush=True)
        return 0
    finally:
        if stage.exists():
            shutil.rmtree(stage)
