"""验证 Link7 转换的配置来源优先级、固定 30 Hz 契约和标称源帧率解析。"""

from pathlib import Path

import pytest
import yaml

from fastumi_data.link7_config import (
    FPS_FROM_CLI,
    FPS_FROM_SNAPSHOT,
    SOURCE_EXPLICIT,
    SOURCE_PACKAGE_DEFAULT,
    SOURCE_SNAPSHOT,
    ConfigError,
    resolve_processing,
    resolve_source_fps,
)


TOPICS = {
    "image": "/umi_camera/image_raw/compressed",
    "tracker_pose": "/vive_tracker/pose",
    "tracker_status": "/vive_tracker/status",
    "gripper_state": "/gripper/state",
    "episode_event": "/fastumi/episode/events",
}


def write(path: Path, **fields) -> Path:
    """写入处理配置。"""
    document = {"topics": dict(TOPICS)}
    document.update(fields)
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.fixture
def default(tmp_path: Path) -> Path:
    """包默认配置：旧路径的 20 Hz。"""
    return write(tmp_path / "default.yaml", sample_rate_hz=20.0, require_tracker_status=True)


@pytest.mark.parametrize("rate", [30, 30.0])
def test_explicit_thirty_hz_is_accepted(tmp_path, default, rate) -> None:
    """显式配置 30 或 30.0 接受，无覆盖记录。"""
    resolved = resolve_processing(write(tmp_path / "c.yaml", sample_rate_hz=rate), None, default)
    assert resolved.source == SOURCE_EXPLICIT
    assert not [item for item in resolved.adjustments if item["field"] == "sample_rate_hz"]


def test_explicit_other_rate_fails_before_conversion(tmp_path, default) -> None:
    """显式配置其他采样率直接报错，不得静默覆盖。"""
    with pytest.raises(ConfigError, match="30 Hz"):
        resolve_processing(write(tmp_path / "c.yaml", sample_rate_hz=20), None, default)


def test_explicit_missing_rate_defaults_to_contract(tmp_path, default) -> None:
    """显式配置未给出采样率时采用 30 并记录字段缺省。"""
    resolved = resolve_processing(write(tmp_path / "c.yaml"), None, default)
    item = next(item for item in resolved.adjustments if item["field"] == "sample_rate_hz")
    assert item["original"] is None and item["effective"] == 30


def test_snapshot_used_when_no_explicit_config(tmp_path, default) -> None:
    """无显式配置时采用记录快照；30 Hz 快照直接采用。"""
    resolved = resolve_processing(None, write(tmp_path / "s.yaml", sample_rate_hz=30), default)
    assert resolved.source == SOURCE_SNAPSHOT
    assert not [item for item in resolved.adjustments if item["field"] == "sample_rate_hz"]


def test_legacy_snapshot_rate_is_overridden_with_report(tmp_path, default) -> None:
    """旧快照的 20 Hz 兼容输出为 30 Hz，并记录原值与原因。"""
    resolved = resolve_processing(None, write(tmp_path / "s.yaml", sample_rate_hz=20.0), default)
    item = next(item for item in resolved.adjustments if item["field"] == "sample_rate_hz")
    assert item["original"] == 20.0 and item["effective"] == 30 and item["reason"]
    assert resolved.report()["sample_rate_hz"] == 30


def test_package_default_used_last(tmp_path, default) -> None:
    """快照缺失时使用包默认配置，其 20 Hz 同样被覆盖记录。"""
    resolved = resolve_processing(None, tmp_path / "missing.yaml", default)
    assert resolved.source == SOURCE_PACKAGE_DEFAULT
    assert any(item["field"] == "sample_rate_hz" for item in resolved.adjustments)


@pytest.mark.parametrize("value", ["30", True, float("nan"), float("inf"), 0, -30])
@pytest.mark.parametrize("source", ["explicit", "snapshot"])
def test_invalid_rate_values_are_errors_for_every_source(tmp_path, default, value, source) -> None:
    """非数值、非有限或非正采样率在任何来源都报错。"""
    path = write(tmp_path / "c.yaml", sample_rate_hz=value)
    with pytest.raises(ConfigError):
        if source == "explicit":
            resolve_processing(path, None, default)
        else:
            resolve_processing(None, path, default)


def test_explicit_disabled_tracker_status_is_rejected(tmp_path, default) -> None:
    """显式 require_tracker_status=false 报错。"""
    with pytest.raises(ConfigError, match="Tracker"):
        resolve_processing(write(tmp_path / "c.yaml", require_tracker_status=False), None, default)


def test_legacy_disabled_tracker_status_is_upgraded(tmp_path, default) -> None:
    """旧快照的 false 升级为 true 并记录原因。"""
    resolved = resolve_processing(
        None, write(tmp_path / "s.yaml", require_tracker_status=False), default
    )
    item = next(item for item in resolved.adjustments if item["field"] == "require_tracker_status")
    assert item["original"] is False and item["effective"] is True


def test_tracker_status_topic_is_required(tmp_path, default) -> None:
    """新路径必须配置 Tracker 状态话题。"""
    path = tmp_path / "c.yaml"
    topics = dict(TOPICS)
    del topics["tracker_status"]
    path.write_text(yaml.safe_dump({"topics": topics}), encoding="utf-8")
    with pytest.raises(ConfigError, match="tracker_status"):
        resolve_processing(path, None, default)


def test_quality_thresholds_are_read_and_validated(tmp_path, default) -> None:
    """质量门限从所选文档读取，非法值报错；最少样本提升到 16。"""
    resolved = resolve_processing(
        write(tmp_path / "c.yaml", interval_tolerance_s=0.006, minimum_samples=10), None, default
    )
    assert resolved.sync.selection.interval_tolerance_s == 0.006
    assert resolved.sync.minimum_samples == 16
    with pytest.raises(ConfigError):
        resolve_processing(write(tmp_path / "d.yaml", max_rate_deviation_ratio=-1), None, default)


def test_source_fps_cli_override_and_snapshot() -> None:
    """标称源帧率优先使用命令行，其次读取快照。"""
    session = {"image": {"nominal_source_fps": 90}}
    assert resolve_source_fps(29.97, session, "r") == (29.97, FPS_FROM_CLI)
    assert resolve_source_fps(None, session, "r") == (90.0, FPS_FROM_SNAPSHOT)


def test_missing_source_fps_requires_declaration() -> None:
    """快照缺少标称源帧率时报错，要求显式声明。"""
    with pytest.raises(ConfigError, match="--source-camera-fps"):
        resolve_source_fps(None, {"image": {"mode": "compressed"}}, "tag_jpeg_1")
    with pytest.raises(ConfigError):
        resolve_source_fps(None, None, "bag")


@pytest.mark.parametrize("value", [25, 45, 0, -30, float("nan"), "30", True])
def test_invalid_or_unsupported_source_fps_rejected(value) -> None:
    """非法或不属于两条路径的标称值在转换前报配置错误。"""
    with pytest.raises(ConfigError):
        resolve_source_fps(value, None, "r")
