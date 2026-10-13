"""加载示教相机/末端相机手眼外参，核对 Tracker 绑定，组合 Tracker→Link7 变换。

约定 ``T_A_B`` 把 B 坐标映射到 A。两份外参文件中的 ``cam0.T_cam_hand`` 分别为
``T_Cu_Tracker``（示教 UMI 相机←Tracker）和 ``T_Cd_Link7``（部署相机←Link7），
两台相机光学坐标轴约定一致，因此::

    T_W_Link7(t) = T_W_Tracker(t − offset) × inverse(T_Cu_Tracker) × T_Cd_Link7

Tracker 外参的时间偏移单位为秒，符号语义必须由绑定清单（或显式参数）声明，
不能只凭 ``time_offset`` 字段名推断。部署外参的时间偏移不参与示教同步。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


# 时间偏移语义 → Tracker 查询时间相对图像时间的符号：t_query = t_image + sign × offset。
TIME_OFFSET_CONVENTIONS = {
    # t_camera = t_tracker + offset（原 handeye_result 的 t_eye = t_hand + offset）。
    "t_camera_eq_t_tracker_plus_offset": -1.0,
    # t_tracker = t_camera + offset（Kalibr timeshift_cam_imu 一类定义）。
    "t_tracker_eq_t_camera_plus_offset": 1.0,
}
# 时间标定状态：只有新时间戳链路重新标定并验收后才能为 verified。
TIME_CALIBRATION_STATUSES = ("provisional", "verified")
# 本次改动前的采集记录未写时间戳来源，它们的图像 Header 均为回调 now()。
LEGACY_TIMESTAMP_SOURCE = "callback_now"
# 绑定清单文件名后缀，与外参同目录：T_cam_tracker.yaml → T_cam_tracker.binding.yaml。
BINDING_SUFFIX = ".binding.yaml"
# 齐次变换合法性检查的数值容差。
_ROTATION_TOLERANCE = 1.0e-5


class CalibrationError(ValueError):
    """外参、绑定或设备身份不一致；属于硬错误，必须中止整批转换。"""


def sha256_bytes(data: bytes) -> str:
    """返回字节串的 SHA-256 十六进制摘要。"""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class HandEyeExtrinsic:
    """一份手眼外参文件：``T_cam_hand``、可选时间偏移及原始字节摘要。"""

    path: str
    sha256: str
    matrix: np.ndarray
    """``cam0.T_cam_hand``，4×4，平移单位米。"""
    time_offset_s: Optional[float]
    inline_tracker_serial: str
    inline_time_offset_convention: str


@dataclass(frozen=True)
class TrackerBinding:
    """Tracker 外参与设备序列号、来源记录和时间偏移语义的绑定清单。"""

    path: str
    sha256: str
    calibration_sha256: str
    tracker_serial: str
    source_collection_uuid: str
    source_status_topic: str
    time_offset_convention: str
    time_calibration_status: str
    time_calibration_timestamp_source: str = ""
    """标定数据的图像 Header 时间戳来源；为空表示未声明。"""
    document: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Link7Calibration:
    """组合后的示教→Link7 外参和全部可追溯元数据。"""

    tracker_to_link7: np.ndarray
    """``inverse(T_Cu_Tracker) × T_Cd_Link7``，把 Link7 坐标映射到 Tracker 坐标。"""
    tracker_camera: HandEyeExtrinsic
    end_camera: HandEyeExtrinsic
    binding: Optional[TrackerBinding]
    tracker_serial: str
    tracker_serial_source: str
    """``binding``、``inline``、``inline+binding`` 或 ``user_declared``。"""
    time_offset_convention: str
    tracker_time_offset_ms: float
    """Tracker 查询时间 = 图像时间 + 本值（毫秒）。"""
    time_calibration_status: str
    notes: tuple = ()

    def metadata(self) -> Dict[str, Any]:
        """返回写入 HDF5、报告和 Zarr 的 JSON 可序列化元数据。"""
        return {
            "tracker_camera_extrinsic_path": self.tracker_camera.path,
            "tracker_camera_extrinsic_sha256": self.tracker_camera.sha256,
            "end_camera_extrinsic_path": self.end_camera.path,
            "end_camera_extrinsic_sha256": self.end_camera.sha256,
            "tracker_binding_path": self.binding.path if self.binding else "",
            "tracker_binding_sha256": self.binding.sha256 if self.binding else "",
            "expected_tracker_serial": self.tracker_serial,
            "tracker_serial_source": self.tracker_serial_source,
            "binding_source_collection_uuid": (
                self.binding.source_collection_uuid if self.binding else ""
            ),
            "T_Cu_Tracker": self.tracker_camera.matrix.tolist(),
            "T_Cd_Link7": self.end_camera.matrix.tolist(),
            "T_Tracker_Link7": self.tracker_to_link7.tolist(),
            "tracker_time_offset_source_s": self.tracker_camera.time_offset_s,
            "time_offset_convention": self.time_offset_convention,
            "tracker_time_offset_ms": self.tracker_time_offset_ms,
            "time_calibration_status": self.time_calibration_status,
            "time_calibration_timestamp_source": (
                self.binding.time_calibration_timestamp_source if self.binding else ""
            ),
            "time_calibration_conditions": (
                str(self.binding.document.get("time_calibration_conditions") or "")
                if self.binding else ""
            ),
            "calibration_notes": list(self.notes),
        }


def _check_transform(matrix: np.ndarray, name: str) -> np.ndarray:
    """要求 4×4 有限齐次刚体变换，返回 float64 副本。"""
    try:
        transform = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CalibrationError(f"{name} 不是数值矩阵") from error
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise CalibrationError(f"{name} 必须是包含有限数值的 4×4 矩阵")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9):
        raise CalibrationError(f"{name} 最后一行必须为 [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=_ROTATION_TOLERANCE
    ) or abs(float(np.linalg.det(rotation)) - 1.0) > _ROTATION_TOLERANCE:
        raise CalibrationError(f"{name} 的旋转部分不是正交且行列式为 1 的矩阵")
    return transform.copy()


def _read_yaml_bytes(path: Path, label: str) -> tuple[bytes, Dict[str, Any]]:
    """读取 YAML 原始字节和映射内容。"""
    try:
        data = path.read_bytes()
        document = yaml.safe_load(data)
    except (OSError, yaml.YAMLError) as error:
        raise CalibrationError(f"无法读取{label} {path}: {error}") from error
    if not isinstance(document, dict):
        raise CalibrationError(f"{label} {path} 必须是 YAML 映射")
    return data, document


def load_hand_eye_extrinsic(path: str, label: str) -> HandEyeExtrinsic:
    """加载 ``cam0.T_cam_hand`` 格式的手眼外参，保留原始字节摘要。

    Args:
        path: 外参 YAML 路径。
        label: 错误信息中的外参名称。

    Raises:
        CalibrationError: 文件不可读、缺少矩阵或矩阵不是刚体变换。
    """
    file_path = Path(path).expanduser().resolve()
    data, document = _read_yaml_bytes(file_path, label)
    camera = document.get("cam0")
    if not isinstance(camera, dict) or "T_cam_hand" not in camera:
        raise CalibrationError(f"{label} {file_path} 缺少 cam0.T_cam_hand")
    matrix = _check_transform(camera["T_cam_hand"], f"{label} T_cam_hand")
    time_offset: Optional[float] = None
    if "time_offset" in document:
        value = document["time_offset"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CalibrationError(f"{label} time_offset 必须是秒为单位的数值")
        time_offset = float(value)
        if not np.isfinite(time_offset):
            raise CalibrationError(f"{label} time_offset 必须为有限数")
    serial = document.get("tracker_serial", "")
    convention = document.get("time_offset_convention", "")
    return HandEyeExtrinsic(
        path=str(file_path),
        sha256=sha256_bytes(data),
        matrix=matrix,
        time_offset_s=time_offset,
        inline_tracker_serial=str(serial or "").strip(),
        inline_time_offset_convention=str(convention or "").strip(),
    )


def binding_path_for(extrinsic_path: str) -> Path:
    """返回外参同目录的配套绑定清单路径。"""
    path = Path(extrinsic_path)
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    return path.with_name(stem + BINDING_SUFFIX)


def load_tracker_binding(path: Path, extrinsic: HandEyeExtrinsic) -> TrackerBinding:
    """加载绑定清单，并要求其摘要与外参原始字节一致。

    Raises:
        CalibrationError: 字段缺失、取值非法或摘要不匹配。
    """
    data, document = _read_yaml_bytes(path, "Tracker 外参绑定清单")
    serial = str(document.get("tracker_serial") or "").strip()
    digest = str(document.get("calibration_sha256") or "").strip().lower()
    if not serial:
        raise CalibrationError(f"绑定清单 {path} 缺少 tracker_serial")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise CalibrationError(f"绑定清单 {path} 的 calibration_sha256 不是 SHA-256")
    if digest != extrinsic.sha256:
        raise CalibrationError(
            f"绑定清单 {path} 的摘要 {digest} 与外参 {extrinsic.path} 的实际摘要 "
            f"{extrinsic.sha256} 不一致；外参文件已改变，须重新确认绑定"
        )
    calibration_file = str(document.get("calibration_file") or "").strip()
    if calibration_file and calibration_file != Path(extrinsic.path).name:
        raise CalibrationError(
            f"绑定清单 {path} 指向 {calibration_file}，与外参文件名不一致"
        )
    convention = str(document.get("time_offset_convention") or "").strip()
    if convention and convention not in TIME_OFFSET_CONVENTIONS:
        raise CalibrationError(
            f"绑定清单 time_offset_convention 必须是 {sorted(TIME_OFFSET_CONVENTIONS)} 之一"
        )
    status = str(document.get("time_calibration_status") or "provisional").strip()
    if status not in TIME_CALIBRATION_STATUSES:
        raise CalibrationError(
            f"绑定清单 time_calibration_status 必须是 {TIME_CALIBRATION_STATUSES} 之一"
        )
    for key in ("source_collection_uuid", "source_status_topic"):
        if not str(document.get(key) or "").strip():
            raise CalibrationError(f"绑定清单 {path} 缺少 {key}")
    return TrackerBinding(
        path=str(path.resolve()),
        sha256=sha256_bytes(data),
        calibration_sha256=digest,
        tracker_serial=serial,
        source_collection_uuid=str(document["source_collection_uuid"]).strip(),
        source_status_topic=str(document["source_status_topic"]).strip(),
        time_offset_convention=convention,
        time_calibration_status=status,
        time_calibration_timestamp_source=str(
            document.get("time_calibration_timestamp_source") or ""
        ).strip(),
        document=document,
    )


def compose_tracker_to_link7(
    camera_from_tracker: np.ndarray, camera_from_link7: np.ndarray
) -> np.ndarray:
    """返回 ``inverse(T_Cu_Tracker) × T_Cd_Link7``，即 ``T_Tracker_Link7``。"""
    tracker = _check_transform(camera_from_tracker, "T_Cu_Tracker")
    end = _check_transform(camera_from_link7, "T_Cd_Link7")
    return np.linalg.inv(tracker) @ end


def tracker_query_offset_ms(time_offset_s: float, convention: str) -> float:
    """把源时间偏移按语义换算为“Tracker 查询时间 = 图像时间 + 返回值”的毫秒数。"""
    if convention not in TIME_OFFSET_CONVENTIONS:
        raise CalibrationError(f"未知时间偏移语义 {convention!r}")
    return TIME_OFFSET_CONVENTIONS[convention] * float(time_offset_s) * 1000.0


def load_link7_calibration(
    tracker_camera_path: str,
    end_camera_path: str,
    declared_tracker_serial: str = "",
    declared_time_offset_convention: str = "",
) -> Link7Calibration:
    """加载两份外参，解析 Tracker 身份和时间偏移语义并组合 Link7 变换。

    身份优先级：外参文件内 ``tracker_serial`` 与同目录绑定清单；两者同时存在时
    必须一致。均缺失时必须由 ``declared_tracker_serial`` 声明（记为用户声明）；
    该声明只能补全缺失身份，与既有身份冲突时报错。

    Args:
        tracker_camera_path: ``T_cam_tracker`` 外参路径。
        end_camera_path: ``T_cam_end`` 外参路径。
        declared_tracker_serial: 命令行 ``--tracker-serial``，可为空。
        declared_time_offset_convention: 绑定与文件均未声明语义时的显式语义。

    Raises:
        CalibrationError: 任一身份、摘要、语义或矩阵检查失败。
    """
    tracker = load_hand_eye_extrinsic(tracker_camera_path, "Tracker 相机外参")
    end = load_hand_eye_extrinsic(end_camera_path, "末端相机外参")
    binding_file = binding_path_for(tracker.path)
    binding = (
        load_tracker_binding(binding_file, tracker) if binding_file.is_file() else None
    )
    notes = []
    serial = ""
    serial_source = ""
    if tracker.inline_tracker_serial and binding is not None:
        if tracker.inline_tracker_serial != binding.tracker_serial:
            raise CalibrationError(
                f"外参内 tracker_serial={tracker.inline_tracker_serial} 与绑定清单 "
                f"{binding.tracker_serial} 冲突"
            )
        serial, serial_source = binding.tracker_serial, "inline+binding"
    elif binding is not None:
        serial, serial_source = binding.tracker_serial, "binding"
    elif tracker.inline_tracker_serial:
        serial, serial_source = tracker.inline_tracker_serial, "inline"
    declared = declared_tracker_serial.strip()
    if serial:
        if declared and declared != serial:
            raise CalibrationError(
                f"--tracker-serial={declared} 与外参既有身份 {serial}（{serial_source}）冲突；"
                "该参数只能补全缺失身份，不能覆盖既有绑定"
            )
        if declared:
            notes.append("--tracker-serial 与既有身份一致，仅作核对")
    elif declared:
        serial, serial_source = declared, "user_declared"
        notes.append("外参缺少设备身份，Tracker 序列号由用户通过 --tracker-serial 声明")
    else:
        raise CalibrationError(
            f"Tracker 外参 {tracker.path} 缺少设备身份：既无文件内 tracker_serial，"
            f"也无绑定清单 {binding_file.name}；请用 --tracker-serial 声明标定所用设备"
        )

    convention = (
        (binding.time_offset_convention if binding else "")
        or tracker.inline_time_offset_convention
    )
    declared_convention = declared_time_offset_convention.strip()
    if convention and declared_convention and declared_convention != convention:
        raise CalibrationError(
            f"显式时间偏移语义 {declared_convention} 与外参声明 {convention} 冲突"
        )
    convention = convention or declared_convention
    time_offset_s = tracker.time_offset_s
    if time_offset_s is None:
        time_offset_s = 0.0
        notes.append("Tracker 外参没有 time_offset，按 0 处理")
        convention = convention or "t_camera_eq_t_tracker_plus_offset"
    elif not convention:
        if time_offset_s != 0.0:
            raise CalibrationError(
                "Tracker 外参含非零 time_offset，但绑定清单和文件均未声明符号语义；"
                "请在绑定清单写入 time_offset_convention 或显式提供语义"
            )
        convention = "t_camera_eq_t_tracker_plus_offset"
    if convention not in TIME_OFFSET_CONVENTIONS:
        raise CalibrationError(
            f"时间偏移语义必须是 {sorted(TIME_OFFSET_CONVENTIONS)} 之一，当前为 {convention!r}"
        )
    if end.time_offset_s is not None:
        notes.append("末端相机外参的 time_offset 属于部署标定，不参与示教同步")
    status = binding.time_calibration_status if binding else "provisional"
    return Link7Calibration(
        tracker_to_link7=compose_tracker_to_link7(tracker.matrix, end.matrix),
        tracker_camera=tracker,
        end_camera=end,
        binding=binding,
        tracker_serial=serial,
        tracker_serial_source=serial_source,
        time_offset_convention=convention,
        tracker_time_offset_ms=tracker_query_offset_ms(time_offset_s, convention),
        time_calibration_status=status,
        notes=tuple(notes),
    )


def normalize_timestamp_source(value: Any) -> str:
    """规范化时间戳来源；旧记录的 ``unrecorded`` 视为回调 ``now()``。"""
    text = str(value or "").strip() or "unrecorded"
    return LEGACY_TIMESTAMP_SOURCE if text == "unrecorded" else text


def check_timestamp_source(calibration: Link7Calibration, record_source: Any) -> str:
    """要求记录的图像时间戳来源与时间偏移标定所用来源一致。

    不同时间戳链路的时间偏移不能互相套用（实测 V4L2 缓冲区时间比回调 ``now()`` 早约
    一个帧周期）。绑定清单未声明来源时无法核对，返回提示文本。

    Returns:
        需要写入报告的提示；一致时为空字符串。

    Raises:
        CalibrationError: 来源不一致。
    """
    expected = calibration.binding.time_calibration_timestamp_source if calibration.binding else ""
    actual = normalize_timestamp_source(record_source)
    if not expected:
        return f"外参未声明标定时间戳来源，无法核对记录的 {actual}"
    if normalize_timestamp_source(expected) != actual:
        raise CalibrationError(
            f"记录的图像时间戳来源 {actual} 与时间偏移标定所用来源 {expected} 不一致；"
            "不同时间戳链路的时间偏移不能直接套用，须在该链路下重新标定"
        )
    return ""
