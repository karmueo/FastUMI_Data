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
    metadata: Dict[str, Any]


def load_tracker_tcp_extrinsic(path: str) -> TrackerTcpExtrinsic:
    """加载并严格校验 Tracker 到 TCP 外参 YAML。"""
    calibration_path = Path(path)
    try:
        file_bytes = calibration_path.read_bytes()
        document = yaml.safe_load(file_bytes)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取外参文件 {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("外参文件 schema_version 必须为 1")
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
    calibration_hash = hashlib.sha256(file_bytes).hexdigest()
    return TrackerTcpExtrinsic(
        matrix=matrix,
        tracker_serial=tracker_serial,
        calibration_hash=calibration_hash,
        metadata=document,
    )
