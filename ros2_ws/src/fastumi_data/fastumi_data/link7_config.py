"""解析 Link7 转换的处理配置与标称源帧率，强制固定 30 Hz 输出契约。

配置来源按“显式 ``--config``、记录 processing 快照、包默认配置”选择首个适用文档。
``sample_rate_hz`` 规则：

* 显式配置给出 30 接受、给出其他值报错；未给出时用契约默认值 30。
* 记录快照或包默认配置给出其他合法值（如旧 20 Hz）时覆盖为 30 并记录原因；
  缺少字段时默认 30。
* 任一来源给出非数值、非有限值或非正值均报错。

``require_tracker_status``：显式 false 报错；快照/包默认的 false 升级为 true 并记录。
所有调整（字段、原值、生效值、原因）写入报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from fastumi_data.frame_selection import (
    FrameSelectionConfig,
    TARGET_FREQUENCY_HZ,
    selection_path_for,
)
from fastumi_data.image_modes import validate_image_mode
from fastumi_data.link7_sync import Link7SyncConfig, MINIMUM_LINK7_FRAMES


SOURCE_EXPLICIT = "explicit_config"
SOURCE_SNAPSHOT = "record_snapshot"
SOURCE_PACKAGE_DEFAULT = "package_default"
# 新路径必需的话题键。
REQUIRED_TOPICS = ("image", "tracker_pose", "tracker_status", "gripper_state", "episode_event")
# 标称源帧率来源。
FPS_FROM_CLI = "cli_override"
FPS_FROM_SNAPSHOT = "record_snapshot"


class ConfigError(ValueError):
    """处理配置或标称源帧率无效；必须在转换开始前报错。"""


@dataclass(frozen=True)
class ResolvedProcessing:
    """一条记录生效的同步配置、话题映射及来源说明。"""

    sync: Link7SyncConfig
    topics: Dict[str, str]
    image_mode: Optional[str]
    """文档显式给出的图像模式；None 表示按 bag 话题类型识别。"""
    source: str
    source_path: str
    adjustments: List[Dict[str, Any]] = field(default_factory=list)

    def report(self) -> Dict[str, Any]:
        """返回可写入报告的配置摘要。"""
        selection = self.sync.selection
        return {
            "source": self.source,
            "path": self.source_path,
            "sample_rate_hz": TARGET_FREQUENCY_HZ,
            "max_pose_gap_s": self.sync.max_pose_gap_s,
            "max_gripper_gap_s": self.sync.max_gripper_gap_s,
            "minimum_samples": self.sync.minimum_samples,
            "require_tracker_status": True,
            "interval_tolerance_s": selection.interval_tolerance_s,
            "max_rate_deviation_ratio": selection.max_rate_deviation_ratio,
            "min_effective_rate_ratio": selection.min_effective_rate_ratio,
            "high_rate_match_margin_s": selection.high_rate_match_margin_s,
            "topics": dict(self.topics),
            "image_mode": self.image_mode,
            "adjustments": list(self.adjustments),
        }


def _number(document: Dict, name: str, default: float, source: str) -> float:
    """读取有限正数字段；显式给出非法值时报错。"""
    if name not in document or document[name] is None:
        return float(default)
    value = document[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{source} 的 {name} 必须为数值，当前为 {value!r}")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ConfigError(f"{source} 的 {name} 必须为有限正数，当前为 {value!r}")
    return number


def load_document(path: Path) -> Dict[str, Any]:
    """读取 YAML 映射。"""
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"无法读取处理配置 {path}: {error}") from error
    if not isinstance(document, dict):
        raise ConfigError(f"处理配置 {path} 必须是 YAML 映射")
    return document


def resolve_processing(
    explicit_path: Optional[Path],
    snapshot_path: Optional[Path],
    default_path: Path,
) -> ResolvedProcessing:
    """选择首个适用的配置文档并按新格式契约校验、调整。

    Raises:
        ConfigError: 配置不可读、字段非法或违反固定 30 Hz/Tracker 状态契约。
    """
    if explicit_path is not None:
        source, path = SOURCE_EXPLICIT, Path(explicit_path)
    elif snapshot_path is not None and Path(snapshot_path).is_file():
        source, path = SOURCE_SNAPSHOT, Path(snapshot_path)
    else:
        source, path = SOURCE_PACKAGE_DEFAULT, Path(default_path)
    document = load_document(path)
    label = f"{source}（{path}）"
    adjustments: List[Dict[str, Any]] = []

    if "sample_rate_hz" in document and document["sample_rate_hz"] is not None:
        rate = _number(document, "sample_rate_hz", TARGET_FREQUENCY_HZ, label)
        if rate != TARGET_FREQUENCY_HZ:
            if source == SOURCE_EXPLICIT:
                raise ConfigError(
                    f"{label} 指定 sample_rate_hz={document['sample_rate_hz']!r}；"
                    f"Link7 新格式固定为 {TARGET_FREQUENCY_HZ} Hz，不得静默覆盖"
                )
            adjustments.append(
                {
                    "field": "sample_rate_hz",
                    "original": document["sample_rate_hz"],
                    "effective": TARGET_FREQUENCY_HZ,
                    "reason": "旧处理配置的采样率；新格式输出固定 30 Hz，"
                    "源图像能力由标称源帧率和质量检查决定",
                }
            )
    else:
        adjustments.append(
            {
                "field": "sample_rate_hz",
                "original": None,
                "effective": TARGET_FREQUENCY_HZ,
                "reason": "字段缺省，采用格式契约默认值",
            }
        )

    if "require_tracker_status" in document:
        value = document["require_tracker_status"]
        if not isinstance(value, bool):
            raise ConfigError(f"{label} 的 require_tracker_status 必须为 bool，当前为 {value!r}")
        if value is False:
            if source == SOURCE_EXPLICIT:
                raise ConfigError(
                    f"{label} 显式设置 require_tracker_status=false；新格式必须检查 Tracker 状态"
                )
            adjustments.append(
                {
                    "field": "require_tracker_status",
                    "original": False,
                    "effective": True,
                    "reason": "新格式强制检查 Tracker 状态，旧快照的 false 升级为 true",
                }
            )

    max_pose_gap_s = _number(document, "max_pose_gap_s", 0.10, label)
    max_gripper_gap_s = _number(document, "max_gripper_gap_s", 0.20, label)
    if max_gripper_gap_s <= 1.0 / TARGET_FREQUENCY_HZ:
        raise ConfigError(
            f"{label} 的 max_gripper_gap_s={max_gripper_gap_s:g} 必须大于 30 Hz 夹爪周期"
        )
    minimum_value = document.get("minimum_samples", MINIMUM_LINK7_FRAMES)
    if isinstance(minimum_value, bool) or not isinstance(minimum_value, int) or minimum_value <= 0:
        raise ConfigError(f"{label} 的 minimum_samples 必须为正整数，当前为 {minimum_value!r}")
    minimum = max(int(minimum_value), MINIMUM_LINK7_FRAMES)
    if minimum != minimum_value:
        adjustments.append(
            {
                "field": "minimum_samples",
                "original": minimum_value,
                "effective": minimum,
                "reason": f"新格式最终至少保留 {MINIMUM_LINK7_FRAMES} 帧",
            }
        )
    defaults = FrameSelectionConfig()
    try:
        selection = FrameSelectionConfig(
            interval_tolerance_s=_number(
                document, "interval_tolerance_s", defaults.interval_tolerance_s, label
            ),
            max_rate_deviation_ratio=_number(
                document, "max_rate_deviation_ratio", defaults.max_rate_deviation_ratio, label
            ),
            min_effective_rate_ratio=_number(
                document, "min_effective_rate_ratio", defaults.min_effective_rate_ratio, label
            ),
            high_rate_match_margin_s=_number(
                document, "high_rate_match_margin_s", defaults.high_rate_match_margin_s, label
            ),
        ).validate()
    except ValueError as error:
        raise ConfigError(f"{label}: {error}") from error

    topics = document.get("topics")
    if not isinstance(topics, dict):
        raise ConfigError(f"{label} 缺少 topics 映射")
    missing = [name for name in REQUIRED_TOPICS if not str(topics.get(name) or "").strip()]
    if missing:
        raise ConfigError(f"{label} 缺少话题: {', '.join(missing)}")
    image_mode = None
    if document.get("image_mode") is not None:
        try:
            image_mode = validate_image_mode(document["image_mode"])
        except ValueError as error:
            raise ConfigError(f"{label}: {error}") from error
    return ResolvedProcessing(
        sync=Link7SyncConfig(
            max_pose_gap_s=max_pose_gap_s,
            max_gripper_gap_s=max_gripper_gap_s,
            minimum_samples=minimum,
            selection=selection,
        ),
        topics={name: str(value) for name, value in topics.items() if value},
        image_mode=image_mode,
        source=source,
        source_path=str(path.resolve()),
        adjustments=adjustments,
    )


def validate_source_fps(value: Any, label: str) -> float:
    """要求标称源帧率为有限正数，并属于同频或高帧率路径。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} 的标称源帧率必须为数值，当前为 {value!r}")
    fps = float(value)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ConfigError(f"{label} 的标称源帧率必须为有限正数，当前为 {value!r}")
    try:
        selection_path_for(fps)
    except ValueError as error:
        raise ConfigError(f"{label}: {error}") from error
    return fps


def resolve_source_fps(
    cli_value: Optional[float], session: Optional[Dict[str, Any]], record_label: str
) -> tuple[float, str]:
    """返回 ``(标称源帧率, 来源)``；缺失时报错，不用实测帧率推断。"""
    if cli_value is not None:
        return validate_source_fps(cli_value, "--source-camera-fps"), FPS_FROM_CLI
    image = (session or {}).get("image") if isinstance(session, dict) else None
    value = image.get("nominal_source_fps") if isinstance(image, dict) else None
    if value is None:
        raise ConfigError(
            f"{record_label} 的采集快照没有标称源帧率（image.nominal_source_fps）；"
            "请用 --source-camera-fps 按其实际采集配置声明"
        )
    return validate_source_fps(value, f"{record_label} 快照"), FPS_FROM_SNAPSHOT
