"""把同步后的 FastUMI episode 写入版本化 HDF5 和质量报告。"""

import json
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np

from fastumi_data.models import SynchronizedEpisode


SCHEMA_VERSION = "fastumi_ros2_v1"


def write_episode_hdf5(
    output_path: str,
    episode: SynchronizedEpisode,
    task_name: str,
    session_id: str,
    episode_index: int,
    sample_rate_hz: float,
    calibration_hash: str,
    tracker_time_offset_ms: float = 0.0,
    source_calibration_sha256: str = "",
    calibration_verified: bool = True,
    calibration_method: str = "",
    aruco_config_sha256: str = "",
) -> None:
    """原子式写入一条 FastUMI HDF5 episode。

    数据先写入同目录临时文件，成功关闭后再替换目标文件，避免中断时留下
    看似完整的损坏数据。
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with h5py.File(temporary, "w", rdcc_nbytes=8 * 1024**2) as root:
        root.attrs["sim"] = False
        root.attrs["schema_version"] = SCHEMA_VERSION
        root.attrs["task_name"] = task_name
        root.attrs["session_id"] = session_id
        root.attrs["episode_index"] = episode_index
        root.attrs["sample_rate_hz"] = sample_rate_hz
        root.attrs["pose_frame"] = "episode_start_tcp"
        root.attrs["image_encoding"] = "rgb8"
        root.attrs["calibration_sha256"] = calibration_hash
        root.attrs["tracker_time_offset_ms"] = tracker_time_offset_ms
        root.attrs["source_calibration_sha256"] = source_calibration_sha256
        root.attrs["calibration_verified"] = calibration_verified
        root.attrs["calibration_method"] = calibration_method
        root.attrs["aruco_config_sha256"] = aruco_config_sha256
        observations = root.create_group("observations")
        images = observations.create_group("images")
        images.create_dataset(
            "front",
            data=episode.images_rgb,
            dtype=np.uint8,
            chunks=(1,) + episode.images_rgb.shape[1:],
            compression="gzip",
            compression_opts=4,
        )
        observations.create_dataset("qpos", data=episode.qpos)
        observations.create_dataset(
            "timestamp_ns", data=episode.timestamp_ns
        )
        quality = observations.create_group("quality")
        quality.create_dataset(
            "gripper_observed", data=episode.gripper_observed
        )
        quality.create_dataset(
            "tracker_tracking_ok", data=episode.tracker_tracking_ok
        )
        quality.create_dataset("pose_gap_ms", data=episode.pose_gap_ms)
        quality.create_dataset(
            "gripper_gap_ms", data=episode.gripper_gap_ms
        )
        root.create_dataset("action", data=episode.qpos)
    temporary.replace(destination)


def build_quality_report(
    episode: SynchronizedEpisode,
    warnings: List[str],
    calibration_hash: str,
    tracker_time_offset_ms: float = 0.0,
    source_calibration_sha256: str = "",
    calibration_verified: bool = True,
    calibration_method: str = "",
    aruco_config_sha256: str = "",
) -> Dict[str, Any]:
    """生成可序列化的 episode 质量摘要。"""
    quaternion_norms = np.linalg.norm(episode.qpos[:, 3:7], axis=1)
    return {
        "accepted": True,
        "sample_count": int(episode.qpos.shape[0]),
        "start_timestamp_ns": int(episode.timestamp_ns[0]),
        "stop_timestamp_ns": int(episode.timestamp_ns[-1]),
        "gripper_observed_ratio": float(
            np.mean(episode.gripper_observed.astype(np.float32))
        ),
        "tracker_tracking_ok_ratio": float(
            np.mean(episode.tracker_tracking_ok.astype(np.float32))
        ),
        "maximum_pose_gap_ms": float(np.max(episode.pose_gap_ms)),
        "maximum_gripper_gap_ms": float(np.max(episode.gripper_gap_ms)),
        "maximum_quaternion_norm_error": float(
            np.max(np.abs(quaternion_norms - 1.0))
        ),
        "calibration_sha256": calibration_hash,
        "tracker_time_offset_ms": tracker_time_offset_ms,
        "source_calibration_sha256": source_calibration_sha256,
        "calibration_verified": calibration_verified,
        "calibration_method": calibration_method,
        "aruco_config_sha256": aruco_config_sha256,
        "warnings": warnings,
    }


def write_json_report(path: str, report: Dict[str, Any]) -> None:
    """使用 UTF-8 和稳定缩进写入质量报告。"""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
