"""将 RM75 遥操 HDF5 与腕部视频按因果时间对齐，导出关节策略 Zarr。"""

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm

# 必需数据流及其数值字段、维度；夹爪编码保持采集值。
STREAMS = {
    "robot0_joint_pos": ("observations/joint_state", "qpos", 7),
    "robot0_gripper_position": ("observations/gripper_state", "position", 1),
    "joint_action": ("action/joint_action", "position", 7),
    "gripper_action": ("action/gripper_action", "position", 1),
}


def validate_timestamps(times, name):
    """检查非空、有限且严格递增的一维秒时间戳，异常时报告字段名。"""
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError(f"{name}: invalid timestamps")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{name}: timestamps must increase strictly")


def causal_indices(times, query):
    """返回查询时刻之前（含相等）的最新索引，禁止查询超出覆盖区间。"""
    if np.any(query < times[0]) or np.any(query > times[-1]):
        raise ValueError("Query outside stream coverage")
    return np.searchsorted(times, query, side="right") - 1


def letterbox_rgb(frame, size=224):
    """将 BGR 图像等比缩放、居中补黑边并转为 uint8 RGB。"""
    height, width = frame.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    output = np.zeros((size, size, 3), dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    output[y:y + resized.shape[0], x:x + resized.shape[1]] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return output


def read_episode(path, frequency=30.0, image_size=224):
    """只读采集文件并按共同覆盖区间重采样；返回数组和可追溯统计。

    不读取 JSON。所有状态、控制目标和图像均用前值保持，禁止未来观测。
    视频解码数量必须与 HDF5 图像时间戳数量一致。
    """
    streams = {}
    with h5py.File(path / "proprio.hdf5", "r") as source:
        for key, (group, field, dimension) in STREAMS.items():
            times = source[f"{group}/timestamp"][:].astype(np.float64)
            values = source[f"{group}/{field}"][:].astype(np.float32)
            validate_timestamps(times, group)
            if values.shape != (len(times), dimension) or not np.isfinite(values).all():
                raise ValueError(f"{path.name}/{group}: invalid values or shape")
            if "gripper" in key and (np.any(values < 0) or np.any(values > 1)):
                raise ValueError(f"{path.name}/{group}: gripper outside [0, 1]")
            streams[key] = (times, values)
        image_times = source["observations/images/cam_gripper_timestamp"][:].astype(np.float64)
    validate_timestamps(image_times, "camera0_rgb")
    start = max([image_times[0]] + [times[0] for times, _ in streams.values()])
    end = min([image_times[-1]] + [times[-1] for times, _ in streams.values()])
    if end <= start:
        raise ValueError(f"{path.name}: no common time coverage")
    query = start + np.arange(int(np.floor((end - start) * frequency)) + 1) / frequency
    query = query[query <= end]
    if len(query) < 16:
        raise ValueError(f"{path.name}: fewer than 16 aligned samples")
    image_indices = causal_indices(image_times, query)
    frames = []
    capture = cv2.VideoCapture(str(path / "gripper.mp4"))
    if not capture.isOpened():
        raise ValueError(f"{path.name}: cannot open video")
    frame_idx = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_idx >= len(image_times):
                raise ValueError(f"{path.name}: video has extra frames")
            # 重采样可能多次引用同一源帧，保持顺序并避免解码多遍。
            count = int(np.count_nonzero(image_indices == frame_idx))
            if count:
                rgb = letterbox_rgb(frame, image_size)
                frames.extend([rgb] * count)
            frame_idx += 1
    finally:
        capture.release()
    if frame_idx != len(image_times):
        raise ValueError(f"{path.name}: decoded {frame_idx}, expected {len(image_times)} frames")
    output = {key: values[causal_indices(times, query)] for key, (times, values) in streams.items()}
    output["action"] = np.concatenate([output.pop("joint_action"), output.pop("gripper_action")], axis=-1)
    output["camera0_rgb"] = np.stack(frames)
    output["timestamp"] = query
    output["source_image_index"] = image_indices.astype(np.int64)
    report = {
        "episode": path.name, "source_frames": frame_idx, "aligned_frames": len(query),
        "unused_source_frames": frame_idx - len(np.unique(image_indices)),
        "repeated_source_frames": len(query) - len(np.unique(image_indices)),
        "start_timestamp": float(query[0]), "end_timestamp": float(query[-1]),
        "source_stream_counts": {key: len(times) for key, (times, _) in streams.items()},
    }
    return output, report


def convert(input_path, output_path, frequency=30.0, image_size=224):
    """转换全部 episode，输出压缩目录 Zarr 与 JSON 报告；拒绝覆盖已有结果。"""
    if not np.isfinite(frequency) or frequency <= 0 or image_size <= 0:
        raise ValueError("Frequency and image size must be positive")
    episodes = sorted(input_path.glob("episode_*"), key=lambda path: int(path.name.split("_")[-1]))
    if not episodes:
        raise ValueError("No episodes found")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output_path), mode="w-")
    root.attrs.update({"format": "rm75-joint-image-v1", "frequency": frequency,
                       "image_size": image_size, "action_layout": "joint8", "complete": False})
    data = root.create_group("data")
    meta = root.create_group("meta")
    reports = []
    ends = []
    total = 0
    codec = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    for path in tqdm(episodes, desc="Convert episodes"):
        try:
            arrays, report = read_episode(path, frequency, image_size)
        except Exception as exc:
            raise ValueError(f"Failed episode {path.name}: {exc}") from exc
        for key, values in arrays.items():
            if key not in data:
                chunks = (1,) + values.shape[1:] if key == "camera0_rgb" else (1024,) + values.shape[1:]
                data.create_dataset(key, data=values, chunks=chunks, compressor=codec)
            else:
                data[key].append(values)
        report["offset_start"] = total
        total += len(arrays["action"])
        report["offset_end"] = total
        reports.append(report)
        ends.append(total)
    meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
    meta.create_dataset("episode_names", data=np.asarray([path.name for path in episodes], dtype="U32"))
    report = {"source": str(input_path.resolve()), "frequency": frequency, "image_size": image_size,
              "episode_count": len(episodes), "aligned_frames": total,
              "source_frames": sum(item["source_frames"] for item in reports),
              "unused_source_frames": sum(item["unused_source_frames"] for item in reports),
              "episodes": reports}
    (output_path.parent / f"{output_path.name}.report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    root.attrs["complete"] = True
    return report


def main():
    """解析显式输入输出路径，运行转换并打印简要统计。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    report = convert(args.input, args.output, args.frequency, args.image_size)
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
