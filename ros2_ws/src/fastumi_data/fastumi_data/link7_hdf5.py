"""读写 ``fastumi_ros2_link7_v1`` 格式的单条 episode HDF5。

结构（N 为帧数）::

    /observations/images/front        (N, H, W, 3) uint8 RGB，每帧一个 chunk；Zarr 模式为
                                      源分辨率，LeRobot 模式为 224×224 letterbox
                                      （见根属性 image_preprocessing、image_size）
    /observations/qpos                (N, 8) float64 [x, y, z, qx, qy, qz, qw, gripper]
    /observations/timestamp           (N,) float64 名义采样时间（秒，首帧为 0）
    /observations/nominal_timestamp_ns(N,) int64  精确名义时间
    /observations/image_timestamp_ns  (N,) int64  实际图像消息时间（ROS 纳秒）
    /observations/source_image_index  (N,) int64  源 bag 解码图像序号
    /observations/dropped_frames_before (N,) int32
    /observations/quality/...         gripper_observed、tracker_tracking_ok、
                                      pose_gap_ms、gripper_gap_ms；高帧率路径另有
                                      match_error_ns 与 reused
    /action                           (N, 8) 与 qpos 相同：同一条示教轨迹在图像时刻的值

位姿为 episode 起始 UMI TCP 相对位姿，位置单位米，夹爪 [0,1]。根属性记录格式、
时间语义、选帧路径、源帧率、外参与时间偏移来源、Tracker 序列号和 URDF 摘要；
嵌套元数据以 JSON 字符串保存在 ``metadata_json``。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import h5py
import numpy as np

from fastumi_data.frame_selection import HIGH_RATE
from fastumi_data.frame_spool import FrameSpool
from fastumi_data.link7_sync import Link7Episode


# 坐标语义改为 UMI TCP 后升级版本；旧 ``fastumi_ros2_link7_v1`` 文件不再接受。
HDF5_SCHEMA_VERSION = "fastumi_ros2_tcp_v1"
# 根属性中的固定时间与坐标语义。
FIXED_ATTRIBUTES = {
    "schema_version": HDF5_SCHEMA_VERSION,
    "pose_frame": "episode_start_tcp",
    "end_frame": "tcp",
    "tcp_definition": "gripper_grasp_center",
    "sample_rate_hz": 30,
    "timestamp_semantics": "nominal_frame_time",
    "state_time_reference": "image_timestamp",
    "action_time_reference": "image_timestamp",
    "action_source": "demonstration_trajectory",
    "qpos_layout": "xyz_quaternion_xyzw_gripper",
    "position_unit": "m",
    "gripper_representation": "normalized_0_1",
    "image_encoding": "rgb8",
}


def _scalar_attribute(value: Any) -> Any:
    """把可直接存为 HDF5 属性的标量返回原值，其余返回 JSON 字符串。"""
    if isinstance(value, (str, bool, int, float, np.integer, np.floating)):
        return value
    return json.dumps(value, ensure_ascii=False)


def write_link7_episode_hdf5(
    output_path: Path,
    episode: Link7Episode,
    spool: FrameSpool,
    metadata: Dict[str, Any],
) -> None:
    """原子写入一条示教 episode：先写同目录临时文件，关闭后再替换。

    Args:
        output_path: 目标 HDF5 路径，必须尚不存在。
        episode: 同步结果。
        spool: 源图像暂存，按 ``episode.selection.positions`` 逐帧读取。
        metadata: 来源、外参、时间、质量等元数据；标量直接成为根属性。
    """
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"{destination} 已存在，拒绝覆盖")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    selection = episode.selection
    count = len(episode)
    frame_shape = spool.frame_shape
    if frame_shape is None:
        raise ValueError("帧暂存为空，无法写出图像")
    try:
        with h5py.File(temporary, "w") as root:
            for key, value in metadata.items():
                root.attrs[key] = _scalar_attribute(value)
            root.attrs.update(FIXED_ATTRIBUTES)
            root.attrs["selection_path"] = selection.path
            root.attrs["nominal_source_fps"] = float(selection.nominal_source_fps)
            root.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
            root.attrs["frame_count"] = count
            observations = root.create_group("observations")
            images = observations.create_group("images").create_dataset(
                "front",
                shape=(count,) + frame_shape,
                dtype=np.uint8,
                chunks=(1,) + frame_shape,
                # gzip 为通用 HDF5 过滤器；级别 1 写入约快 30%，体积仅增约 3%。
                compression="gzip",
                compression_opts=1,
            )
            for index, position in enumerate(selection.positions.tolist()):
                images[index] = spool.read(position)
            observations.create_dataset("qpos", data=episode.qpos.astype(np.float64))
            nominal_ns = selection.nominal_timestamp_ns
            observations.create_dataset("timestamp", data=nominal_ns.astype(np.float64) / 1.0e9)
            observations.create_dataset("nominal_timestamp_ns", data=nominal_ns)
            observations.create_dataset(
                "image_timestamp_ns", data=selection.image_timestamps_ns.astype(np.int64)
            )
            observations.create_dataset(
                "source_image_index",
                data=spool.source_indices[selection.positions].astype(np.int64),
            )
            observations.create_dataset(
                "dropped_frames_before", data=selection.dropped_frames_before
            )
            quality = observations.create_group("quality")
            quality.create_dataset("gripper_observed", data=episode.gripper_observed)
            quality.create_dataset("tracker_tracking_ok", data=episode.tracker_tracking_ok)
            quality.create_dataset("pose_gap_ms", data=episode.pose_gap_ms)
            quality.create_dataset("gripper_gap_ms", data=episode.gripper_gap_ms)
            if selection.path == HIGH_RATE:
                quality.create_dataset("match_error_ns", data=selection.match_error_ns)
                reused = np.zeros(count, dtype=bool)
                reused[1:] = np.diff(selection.positions) <= 0
                quality.create_dataset("reused", data=reused)
            root.create_dataset("action", data=episode.qpos.astype(np.float64))
            root.create_dataset(
                "world_from_start_tcp", data=episode.world_from_start_tcp
            )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_link7_metadata(path: Path) -> Dict[str, Any]:
    """读取 HDF5 根属性中的完整元数据并校验格式标识。"""
    with h5py.File(path, "r") as root:
        schema = root.attrs.get("schema_version")
        if isinstance(schema, bytes):
            schema = schema.decode()
        if schema != HDF5_SCHEMA_VERSION:
            raise ValueError(f"{path} 不是 {HDF5_SCHEMA_VERSION}（实际 {schema!r}）")
        metadata = json.loads(root.attrs["metadata_json"])
        metadata["selection_path"] = str(root.attrs["selection_path"])
        metadata["nominal_source_fps"] = float(root.attrs["nominal_source_fps"])
        return metadata


def read_link7_arrays(path: Path) -> Dict[str, np.ndarray]:
    """读取全部数值数组（不含图像）。"""
    with h5py.File(path, "r") as root:
        observations = root["observations"]
        arrays = {
            name: observations[name][:]
            for name in (
                "qpos", "timestamp", "nominal_timestamp_ns", "image_timestamp_ns",
                "source_image_index", "dropped_frames_before",
            )
        }
        arrays["action"] = root["action"][:]
        return arrays


def count_link7_images(path: Path) -> int:
    """返回 HDF5 中原分辨率图像的帧数（不读取像素）。"""
    with h5py.File(path, "r") as root:
        return int(root["observations/images/front"].shape[0])


def iter_link7_images(
    path: Path, start: int = 0, stop: Optional[int] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """逐帧读取原分辨率 RGB 图像，避免整段载入内存；``[start, stop)`` 限定帧范围（缺省全部）。"""
    with h5py.File(path, "r") as root:
        images = root["observations/images/front"]
        for index in range(start, images.shape[0] if stop is None else min(stop, images.shape[0])):
            yield index, images[index]
