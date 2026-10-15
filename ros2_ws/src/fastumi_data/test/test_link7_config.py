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
    """包默认配置。"""
    return write(tmp_path / "default.yaml", require_tracker_status=True)


def test_source_priority_is_explicit_snapshot_default(tmp_path, default) -> None:
    """显式配置优先于记录快照，快照优先于包默认。"""
    explicit = write(tmp_path / "c.yaml")
    snapshot = write(tmp_path / "s.yaml")
    assert resolve_processing(explicit, snapshot, default).source == SOURCE_EXPLICIT
    assert resolve_processing(None, snapshot, default).source == SOURCE_SNAPSHOT
    resolved = resolve_processing(None, tmp_path / "missing.yaml", default)
    assert resolved.source == SOURCE_PACKAGE_DEFAULT


@pytest.mark.parametrize("source", ["explicit", "snapshot"])
def test_legacy_rate_and_minimum_fields_are_ignored(tmp_path, default, source) -> None:
    """遗留的 sample_rate_hz、minimum_samples、max_image_delta_s 不报错、不产生调整，也不进入报告。"""
    path = write(
        tmp_path / "c.yaml", sample_rate_hz=20.0, minimum_samples=10, max_image_delta_s=0.03
    )
    resolved = (
        resolve_processing(path, None, default)
        if source == "explicit"
        else resolve_processing(None, path, default)
    )
    assert resolved.adjustments == []
    report = resolved.report()
    assert not {"sample_rate_hz", "minimum_samples", "max_image_delta_s"} & set(report)


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


def test_image_mode_is_optional_and_validated(tmp_path, default) -> None:
    """image_mode 缺省为 None（按 bag 话题类型识别），非法取值报错。"""
    assert resolve_processing(write(tmp_path / "a.yaml"), None, default).image_mode is None
    resolved = resolve_processing(write(tmp_path / "b.yaml", image_mode="h264"), None, default)
    assert resolved.image_mode == "h264"
    with pytest.raises(ConfigError, match="image_mode"):
        resolve_processing(write(tmp_path / "c.yaml", image_mode="mjpeg"), None, default)


def test_quality_thresholds_are_read_and_validated(tmp_path, default) -> None:
    """质量门限从所选文档读取，非法值报错。"""
    resolved = resolve_processing(
        write(tmp_path / "c.yaml", interval_tolerance_s=0.006), None, default
    )
    assert resolved.sync.selection.interval_tolerance_s == 0.006
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
