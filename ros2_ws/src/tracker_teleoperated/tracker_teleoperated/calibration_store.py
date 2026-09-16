"""持久化 Tracker 工作空间标定方向，并校验坐标系与矩阵格式。"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import numpy as np
import yaml


# 当前工作空间标定文件格式版本。
CALIBRATION_VERSION = 1


class CalibrationStoreError(RuntimeError):
    """表示标定文件内容无效或持久化操作失败。"""


def resolve_workspace_calibration_path(configured_path: str) -> Path:
    """解析配置路径，空值使用 ROS_HOME 下的默认标定文件。

    Args:
        configured_path: 参数指定的路径；空字符串表示使用默认位置。

    Returns:
        展开用户目录后的标定文件路径。
    """
    if configured_path.strip():
        return Path(configured_path).expanduser()
    # ROS_HOME 为空时回退到 ROS 2 的常用用户数据目录。
    ros_home = os.environ.get("ROS_HOME") or str(Path.home() / ".ros")
    return (
        Path(ros_home).expanduser()
        / "tracker_teleoperated"
        / "workspace_calibration.yaml"
    )


def validate_mapping_basis(mapping_basis: np.ndarray) -> np.ndarray:
    """校验并复制一个有限、右手正交的 3×3 映射矩阵。"""
    basis = np.asarray(mapping_basis, dtype=np.float64)
    if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
        raise CalibrationStoreError("mapping_basis 必须是有限的 3x3 矩阵")
    if not np.allclose(basis @ basis.T, np.eye(3), atol=1.0e-6):
        raise CalibrationStoreError("mapping_basis 必须是正交矩阵")
    if not np.isclose(np.linalg.det(basis), 1.0, atol=1.0e-6):
        raise CalibrationStoreError("mapping_basis 必须为右手坐标映射")
    return basis.copy()


def load_workspace_calibration(
    path: Path, odom_frame: str, base_frame: str
) -> np.ndarray:
    """读取并校验指定坐标系的工作空间标定方向。

    Args:
        path: YAML 标定文件路径。
        odom_frame: 当前 Tracker 里程计坐标系名称。
        base_frame: 当前机械臂基座坐标系名称。

    Returns:
        校验通过的 3×3 odom 到 Base 映射矩阵。

    Raises:
        FileNotFoundError: 标定文件不存在。
        CalibrationStoreError: 文件无法读取、格式错误或坐标系不匹配。
    """
    calibration_path = Path(path)
    try:
        document = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CalibrationStoreError(f"无法读取标定文件: {error}") from error
    if not isinstance(document, dict):
        raise CalibrationStoreError("标定文件根节点必须是映射")
    if document.get("version") != CALIBRATION_VERSION:
        raise CalibrationStoreError(
            f"标定文件版本必须为 {CALIBRATION_VERSION}"
        )
    if document.get("odom_frame") != odom_frame:
        raise CalibrationStoreError(
            "标定文件 odom_frame 与当前配置不一致: "
            f"{document.get('odom_frame')!r} != {odom_frame!r}"
        )
    if document.get("base_frame") != base_frame:
        raise CalibrationStoreError(
            "标定文件 base_frame 与当前配置不一致: "
            f"{document.get('base_frame')!r} != {base_frame!r}"
        )
    try:
        return validate_mapping_basis(document.get("mapping_basis"))
    except (TypeError, ValueError) as error:
        raise CalibrationStoreError(f"mapping_basis 数值无效: {error}") from error


def save_workspace_calibration(
    path: Path,
    odom_frame: str,
    base_frame: str,
    mapping_basis: np.ndarray,
) -> None:
    """以临时文件和原子替换保存工作空间标定方向。

    Args:
        path: YAML 标定文件路径。
        odom_frame: Tracker 里程计坐标系名称。
        base_frame: 机械臂基座坐标系名称。
        mapping_basis: 3×3 odom 到 Base 映射矩阵。

    Raises:
        CalibrationStoreError: 矩阵无效、目录创建或文件写入失败。
    """
    basis = validate_mapping_basis(mapping_basis)
    calibration_path = Path(path)
    document = {
        "version": CALIBRATION_VERSION,
        "odom_frame": str(odom_frame),
        "base_frame": str(base_frame),
        "mapping_basis": basis.tolist(),
    }
    temporary_path: Path | None = None
    try:
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{calibration_path.name}.",
            suffix=".tmp",
            dir=calibration_path.parent,
            text=True,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(
                document,
                stream,
                allow_unicode=True,
                sort_keys=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, calibration_path)
        temporary_path = None
    except (OSError, yaml.YAMLError) as error:
        raise CalibrationStoreError(f"无法保存标定文件: {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
