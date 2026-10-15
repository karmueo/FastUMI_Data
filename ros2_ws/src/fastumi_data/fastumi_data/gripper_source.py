"""选择 Link7 转换使用的夹爪开度数据：二值化结果、原平滑开度或未平滑开度。

``binarize_gripper.py`` 写出的录制副本在记录目录下带 ``binarization.yaml``，并把
``/gripper/state`` 的 ``filtered_openness`` 改写为 0/1（1 开 0 闭）、``valid`` 全置真；
``raw_openness`` 原样保留，原 ``valid`` 以逐帧 ``0/1`` 串保存在 ``original_valid``。

选项（``--gripper-source``）：

* ``auto``：有 ``binarization.yaml`` 用 ``binary``，否则用 ``filtered``。
* ``binary``：二值 ``filtered_openness``，按最近邻取值，不产生中间值；须为二值化记录。
* ``filtered``：采集时的平滑开度 ``filtered_openness``，线性插值；二值化记录的原值已被覆盖，报错。
* ``raw``：未平滑的 ``raw_openness``，线性插值；二值化记录按 ``original_valid`` 恢复原有效性。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from fastumi_data.link7_config import ConfigError


BINARIZATION_FILE = "binarization.yaml"
BINARIZATION_SCHEMA_VERSION = 1

SOURCE_AUTO = "auto"
SOURCE_BINARY = "binary"
SOURCE_FILTERED = "filtered"
SOURCE_RAW = "raw"
GRIPPER_SOURCE_CHOICES = (SOURCE_AUTO, SOURCE_BINARY, SOURCE_FILTERED, SOURCE_RAW)

INTERPOLATION_LINEAR = "linear"
INTERPOLATION_NEAREST = "nearest"

_FIELDS = {
    SOURCE_BINARY: ("filtered_openness", INTERPOLATION_NEAREST),
    SOURCE_FILTERED: ("filtered_openness", INTERPOLATION_LINEAR),
    SOURCE_RAW: ("raw_openness", INTERPOLATION_LINEAR),
}


@dataclass(frozen=True)
class GripperSource:
    """一条记录生效的夹爪数据来源。"""

    requested: str
    effective: str
    field: str
    """``GripperSample`` 中读取开度的字段名。"""
    interpolation: str
    binarization_path: str = ""
    binarization_sha256: str = ""
    frame_count: Optional[int] = None
    """二值化时记录的夹爪消息数；None 表示非二值化记录。"""
    original_valid: Optional[Tuple[bool, ...]] = None
    """``raw`` 模式下二值化记录的逐帧原有效性；其余情况为 None（沿用消息 ``valid``）。"""

    @property
    def binarized(self) -> bool:
        """记录是否为二值化副本。"""
        return self.frame_count is not None

    def report(self) -> Dict[str, Any]:
        """返回可写入报告与元数据的摘要（不含逐帧数组）。"""
        return {
            "requested": self.requested,
            "effective": self.effective,
            "field": self.field,
            "interpolation": self.interpolation,
            "binarized_record": self.binarized,
            "binarization_file": self.binarization_path,
            "binarization_sha256": self.binarization_sha256,
        }


DEFAULT_GRIPPER_SOURCE = GripperSource(
    SOURCE_FILTERED, SOURCE_FILTERED, "filtered_openness", INTERPOLATION_LINEAR
)


def _load_binarization(path: Path, gripper_topic: str) -> Tuple[Dict[str, Any], str, Tuple[bool, ...]]:
    """读取并校验 ``binarization.yaml``，返回文档、SHA-256 和逐帧原有效性。"""
    try:
        data = path.read_bytes()
        document = yaml.safe_load(data)
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"无法读取 {path}: {error}") from error
    if not isinstance(document, dict):
        raise ConfigError(f"{path} 必须是 YAML 映射")
    if document.get("schema_version") != BINARIZATION_SCHEMA_VERSION:
        raise ConfigError(
            f"{path} 的 schema_version={document.get('schema_version')!r}，"
            f"只支持 {BINARIZATION_SCHEMA_VERSION}"
        )
    topic = document.get("topic")
    if topic != gripper_topic:
        raise ConfigError(f"{path} 的夹爪话题 {topic!r} 与处理配置 {gripper_topic!r} 不一致")
    frame_count = document.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ConfigError(f"{path} 的 frame_count={frame_count!r} 必须是正整数")
    flags = document.get("original_valid")
    if not isinstance(flags, str) or set(flags) - {"0", "1"}:
        raise ConfigError(f"{path} 的 original_valid 必须是只含 0/1 的字符串")
    if len(flags) != frame_count:
        raise ConfigError(
            f"{path} 的 original_valid 长度 {len(flags)} 与 frame_count {frame_count} 不一致"
        )
    return document, hashlib.sha256(data).hexdigest(), tuple(flag == "1" for flag in flags)


def resolve_gripper_source(
    requested: str, record_dir: Optional[Path], gripper_topic: str
) -> GripperSource:
    """按命令行选项和记录目录中的 ``binarization.yaml`` 决定夹爪数据来源。

    Args:
        requested: ``GRIPPER_SOURCE_CHOICES`` 之一。
        record_dir: 含 ``session.yaml`` 的记录目录；裸 bag 为 None，视为未二值化。
        gripper_topic: 处理配置中的夹爪话题。

    Raises:
        ConfigError: 选项与记录不兼容，或 ``binarization.yaml`` 无效。
    """
    if requested not in GRIPPER_SOURCE_CHOICES:
        raise ConfigError(f"未知的夹爪数据来源 {requested!r}")
    path = None if record_dir is None else Path(record_dir) / BINARIZATION_FILE
    binarized = path is not None and path.is_file()
    if requested == SOURCE_AUTO:
        effective = SOURCE_BINARY if binarized else SOURCE_FILTERED
    else:
        effective = requested
    if effective == SOURCE_BINARY and not binarized:
        raise ConfigError(f"记录没有 {BINARIZATION_FILE}，不能使用二值化夹爪数据")
    if effective == SOURCE_FILTERED and binarized:
        raise ConfigError(
            f"记录含 {BINARIZATION_FILE}，其 filtered_openness 已被二值结果覆盖；"
            "请使用 binary 或 raw"
        )
    field_name, interpolation = _FIELDS[effective]
    if not binarized:
        return GripperSource(requested, effective, field_name, interpolation)
    _, sha256, original_valid = _load_binarization(path, gripper_topic)
    return GripperSource(
        requested,
        effective,
        field_name,
        interpolation,
        binarization_path=str(path),
        binarization_sha256=sha256,
        frame_count=len(original_valid),
        original_valid=original_valid if effective == SOURCE_RAW else None,
    )
