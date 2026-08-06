"""加载、校验并哈希 Tracker 到 UMI TCP 的版本化外参文件。"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from fastumi_data.pose_math import pose_to_matrix


@dataclass(frozen=True)
class TrackerTcpExtrinsic:
    """保存 Tracker 到 UMI TCP 外参及其可追溯元数据。"""

    matrix: np.ndarray
    tracker_serial: str
    calibration_hash: str
    time_offset_ms: float
    source_calibration_sha256: str
    metadata: Dict[str, Any]
    calibration_verified: bool = True
    calibration_method: str = ""
    aruco_config_sha256: str = ""


def load_tracker_tcp_extrinsic(
    path: str, allow_unverified: bool = False
) -> TrackerTcpExtrinsic:
    """加载并严格校验 Tracker 到 TCP 外参 YAML。"""
    calibration_path = Path(path)
    try:
        file_bytes = calibration_path.read_bytes()
        document = yaml.safe_load(file_bytes)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取外参文件 {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("外参文件 schema_version 必须为 1")
    calibration_verified = document.get("calibration_verified", True)
    if not isinstance(calibration_verified, bool):
        raise ValueError("calibration_verified 必须是 YAML bool")
    if not calibration_verified and not allow_unverified:
        raise ValueError(
            "外参 calibration_verified=false，需要显式传入 "
            "--allow-unverified-extrinsic 才能加载"
        )
    transform = document.get("tracker_to_tcp")
    if not isinstance(transform, dict):
        raise ValueError("外参文件缺少 tracker_to_tcp")
    translation = np.asarray(transform.get("translation_m"), dtype=np.float64)
    quaternion = np.asarray(
        transform.get("quaternion_xyzw"), dtype=np.float64
    )
    matrix = pose_to_matrix(translation, quaternion)
    tracker_serial = str(document.get("tracker_serial", "")).strip()
    if not tracker_serial:
        raise ValueError("外参文件必须包含 tracker_serial")
    try:
        time_offset_ms = float(document.get("time_offset_ms", 0.0))
    except (TypeError, ValueError) as error:
        raise ValueError("外参 time_offset_ms 必须为有限数") from error
    if not np.isfinite(time_offset_ms):
        raise ValueError("外参 time_offset_ms 必须为有限数")
    source_calibration = document.get("source_calibration", {})
    if source_calibration is None:
        source_calibration = {}
    if not isinstance(source_calibration, dict):
        raise ValueError("source_calibration 必须是映射")
    source_calibration_sha256 = str(
        source_calibration.get("sha256", "")
    ).strip()
    calibration_method = str(
        document.get("calibration_method", document.get("method", ""))
    ).strip()
    aruco_config_sha256 = str(
        document.get("aruco_config_sha256", "")
    ).strip()
    calibration_hash = hashlib.sha256(file_bytes).hexdigest()
    return TrackerTcpExtrinsic(
        matrix=matrix,
        tracker_serial=tracker_serial,
        calibration_hash=calibration_hash,
        time_offset_ms=time_offset_ms,
        source_calibration_sha256=source_calibration_sha256,
        metadata=document,
        calibration_verified=calibration_verified,
        calibration_method=calibration_method,
        aruco_config_sha256=aruco_config_sha256,
    )
