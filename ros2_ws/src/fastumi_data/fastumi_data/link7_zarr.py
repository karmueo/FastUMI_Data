"""把多条 ``fastumi_ros2_link7_v1`` HDF5 合并导出为 ``fastumi-rm75-link7-v1`` Zarr。

输出 ReplayBuffer 结构（N 为总帧数）::

    data/camera0_rgb               (N, 224, 224, 3) uint8，JPEG XL 无损，每帧一个 chunk
    data/robot0_eef_pos            (N, 3) float32，episode 起始 Link7 系，米
    data/robot0_eef_rot_axis_angle (N, 3) float32，旋转向量，弧度
    data/robot0_gripper_width      (N, 1) float32，归一化开度 [0,1]
    data/robot0_demo_start_pose    (N, 6) float32，本段首帧 [xyz, rotvec]
    data/action                    (N, 7) float32 [xyz, rotvec, gripper]
    data/timestamp                 (N,) float64 名义采样时间（秒）
    data/nominal_timestamp_ns      (N,) int64
    data/image_timestamp_ns        (N,) int64 实际图像时间
    data/source_image_index        (N,) int64
    data/dropped_frames_before     (N,) int32
    meta/episode_ends              (E,) int64
    meta/episode_names             (E,) str

图像按部署预处理等比例缩放（INTER_AREA）并居中补黑边至 224×224。先在同目录临时
路径构建，全部写入并逐帧回读验收后才写 ``complete=true`` 并原子改名发布。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from fastumi_data.jxl_codec import lossless_jpegxl
from fastumi_data.link7_hdf5 import iter_link7_images, read_link7_arrays


ZARR_FORMAT = "fastumi-rm75-link7-v1"
IMAGE_SIZE = 224
# 图像逐批写入及回读校验的帧数。
_IMAGE_BATCH = 64
# 根属性固定契约；urdf_sha256 与 complete 运行时写入。
ROOT_CONTRACT: Dict[str, Any] = {
    "format": ZARR_FORMAT,
    "frequency": 30,
    "timestamp_semantics": "nominal_frame_time",
    "state_time_reference": "image_timestamp",
    "action_time_reference": "image_timestamp",
    "image_size": IMAGE_SIZE,
    "action_layout": "pose10",
    "stored_action_layout": "xyz_rotvec_gripper",
    "pose_frame": "episode_start_link7",
    "deployment_base_frame": "base_link",
    "end_frame": "Link7",
    "tool_offset": np.eye(4).tolist(),
    "position_unit": "m",
    "rotation_unit": "rad",
    "gripper_representation": "normalized_0_1",
    "normalization_contract": "fixed-gripper-0-1-v1",
    "gripper_lower": 0.0,
    "gripper_upper": 1.0,
    "action_source": "demonstration_trajectory",
}
# 数值数组：名称 → (尾部形状, dtype)。
NUMERIC_ARRAYS = {
    "robot0_eef_pos": ((3,), "f4"),
    "robot0_eef_rot_axis_angle": ((3,), "f4"),
    "robot0_gripper_width": ((1,), "f4"),
    "robot0_demo_start_pose": ((6,), "f4"),
    "action": ((7,), "f4"),
    "timestamp": ((), "f8"),
    "nominal_timestamp_ns": ((), "i8"),
    "image_timestamp_ns": ((), "i8"),
    "source_image_index": ((), "i8"),
    "dropped_frames_before": ((), "i4"),
}


@dataclass(frozen=True)
class ZarrEpisodeSource:
    """一条待合并 episode：唯一名称、HDF5 路径和逐 episode 元数据。"""

    name: str
    hdf5_path: Path
    metadata: Dict[str, Any]


def letterbox_rgb_uint8(rgb: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """与部署 ``letterbox_rgb`` 相同的缩放、居中黑边，返回 ``(size, size, 3)`` uint8。"""
    height, width = rgb.shape[:2]
    scale = size / max(height, width)
    resized = cv2.resize(
        rgb,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    output = np.zeros((size, size, 3), dtype=np.uint8)
    y, x = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return output


def numeric_episode(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """把 HDF5 的 qpos/action（xyz + 四元数 xyzw + 夹爪）转换为 Zarr 数值数组。"""
    qpos = np.asarray(arrays["qpos"], dtype=np.float64)
    action = np.asarray(arrays["action"], dtype=np.float64)
    position = qpos[:, :3]
    rotvec = Rotation.from_quat(qpos[:, 3:7]).as_rotvec()
    action_rotvec = Rotation.from_quat(action[:, 3:7]).as_rotvec()
    start = np.concatenate((position[:1], rotvec[:1]), axis=1)
    count = len(qpos)
    return {
        "robot0_eef_pos": position.astype(np.float32),
        "robot0_eef_rot_axis_angle": rotvec.astype(np.float32),
        "robot0_gripper_width": qpos[:, 7:8].astype(np.float32),
        "robot0_demo_start_pose": np.repeat(start, count, axis=0).astype(np.float32),
        "action": np.concatenate(
            (action[:, :3], action_rotvec, action[:, 7:8]), axis=1
        ).astype(np.float32),
        "timestamp": np.asarray(arrays["timestamp"], dtype=np.float64),
        "nominal_timestamp_ns": np.asarray(arrays["nominal_timestamp_ns"], dtype=np.int64),
        "image_timestamp_ns": np.asarray(arrays["image_timestamp_ns"], dtype=np.int64),
        "source_image_index": np.asarray(arrays["source_image_index"], dtype=np.int64),
        "dropped_frames_before": np.asarray(arrays["dropped_frames_before"], dtype=np.int32),
    }


def _check_episode_arrays(name: str, values: Dict[str, np.ndarray]) -> None:
    """导出前检查单条 episode 的数值、单调性和夹爪范围。"""
    for key in ("robot0_eef_pos", "robot0_eef_rot_axis_angle", "action", "timestamp"):
        if not np.all(np.isfinite(values[key])):
            raise ValueError(f"{name} 的 {key} 含非有限值")
    gripper = values["robot0_gripper_width"][:, 0]
    if np.any(gripper < 0.0) or np.any(gripper > 1.0):
        raise ValueError(f"{name} 的夹爪开度超出 [0,1]")
    for key in ("source_image_index", "image_timestamp_ns", "nominal_timestamp_ns"):
        if np.any(np.diff(values[key]) <= 0):
            raise ValueError(f"{name} 的 {key} 不是严格递增")
    if np.any(values["source_image_index"] < 0):
        raise ValueError(f"{name} 的 source_image_index 含负值")


def to_jsonable(value: Any) -> Any:
    """把含 NumPy 标量/数组的嵌套结构转换为纯 JSON 类型。"""

    def default(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"无法序列化 {type(item).__name__}")

    return json.loads(json.dumps(value, default=default))


def export_link7_zarr(
    episodes: Sequence[ZarrEpisodeSource],
    output_path: Path,
    urdf_sha256: str,
    extra_attributes: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """合并导出 Zarr，验收后发布到 ``output_path``；失败时保留临时目录供诊断。

    Returns:
        帧数、episode 数及临时目录等摘要。

    Raises:
        FileExistsError: 目标已存在。
        ValueError: 输入为空、名称重复或数值检查失败。
    """
    import zarr
    from numcodecs import Blosc

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"{output} 已存在，拒绝覆盖")
    if not episodes:
        raise ValueError("没有可导出的 episode")
    names = [episode.name for episode in episodes]
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("episode 名称必须非空且唯一")
    if len(urdf_sha256) != 64:
        raise ValueError("urdf_sha256 必须是 64 位十六进制摘要")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    root = zarr.open_group(zarr.DirectoryStore(str(temporary)), mode="w-")
    data = root.create_group("data")
    meta = root.create_group("meta")
    numeric_codec = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
    image_codec = lossless_jpegxl()
    images = data.create_dataset(
        "camera0_rgb",
        shape=(0, IMAGE_SIZE, IMAGE_SIZE, 3),
        chunks=(1, IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype="u1",
        compressor=image_codec,
    )
    numeric = {
        key: data.create_dataset(
            key,
            shape=(0,) + tail,
            chunks=(1024,) + tail,
            dtype=dtype,
            compressor=numeric_codec,
        )
        for key, (tail, dtype) in NUMERIC_ARRAYS.items()
    }
    ends: List[int] = []
    episode_metadata: List[Dict[str, Any]] = []
    total = 0
    for episode in episodes:
        values = numeric_episode(read_link7_arrays(episode.hdf5_path))
        _check_episode_arrays(episode.name, values)
        count = len(values["timestamp"])
        start = total
        for key, array in numeric.items():
            array.append(values[key], axis=0)
        images.resize((start + count, IMAGE_SIZE, IMAGE_SIZE, 3))
        batch: List[np.ndarray] = []
        batch_start = start
        written = 0
        for _, frame in iter_link7_images(episode.hdf5_path):
            batch.append(letterbox_rgb_uint8(frame))
            if len(batch) == _IMAGE_BATCH:
                _write_verified_images(images, batch_start, batch, episode.name)
                batch_start += len(batch)
                written += len(batch)
                batch = []
        if batch:
            _write_verified_images(images, batch_start, batch, episode.name)
            written += len(batch)
        if written != count:
            raise ValueError(f"{episode.name} 图像 {written} 帧与数值 {count} 帧不一致")
        total += count
        ends.append(total)
        record = dict(episode.metadata)
        record.update({"name": episode.name, "start": start, "end": total, "frames": count})
        episode_metadata.append(to_jsonable(record))

    meta.create_dataset("episode_ends", data=np.asarray(ends, dtype=np.int64))
    meta.create_dataset("episode_names", data=np.asarray(names, dtype=str))
    # 回读数值数组，确认与源 HDF5 逐元素一致。
    for episode, record in zip(episodes, episode_metadata):
        expected = numeric_episode(read_link7_arrays(episode.hdf5_path))
        for key in NUMERIC_ARRAYS:
            if not np.array_equal(numeric[key][record["start"]:record["end"]], expected[key]):
                raise ValueError(f"{episode.name} 的 {key} 回读不一致")
    attributes = dict(ROOT_CONTRACT)
    attributes.update(to_jsonable(extra_attributes or {}))
    attributes.update(
        {
            "urdf_sha256": urdf_sha256.lower(),
            "episode_count": len(ends),
            "frame_count": total,
            "episode_metadata": episode_metadata,
            "complete": False,
        }
    )
    root.attrs.update(attributes)
    root.attrs["complete"] = True
    os.rename(temporary, output)
    return {
        "path": str(output),
        "frames": total,
        "episodes": len(ends),
    }


def _write_verified_images(images, start: int, batch: List[np.ndarray], name: str) -> None:
    """写入一批图像并立即回读，确认 JPEG XL 逐像素无损。"""
    stacked = np.stack(batch)
    images[start:start + len(batch)] = stacked
    if not np.array_equal(images[start:start + len(batch)], stacked):
        raise ValueError(f"{name} 的图像 JPEG XL 回读与原像素不一致")
