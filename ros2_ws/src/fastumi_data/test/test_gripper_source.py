"""验证 convert_mcap 夹爪数据来源的选择与 binarization.yaml 校验。"""

from pathlib import Path

import pytest
import yaml

from fastumi_data.gripper_source import (
    BINARIZATION_FILE,
    INTERPOLATION_LINEAR,
    INTERPOLATION_NEAREST,
    resolve_gripper_source,
)
from fastumi_data.link7_config import ConfigError


TOPIC = "/gripper/state"


def write_binarization(record: Path, **overrides) -> Path:
    """写一份最小 binarization.yaml。"""
    document = {"schema_version": 1, "topic": TOPIC, "frame_count": 4, "original_valid": "1011"}
    document.update(overrides)
    record.mkdir(parents=True, exist_ok=True)
    path = record / BINARIZATION_FILE
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "requested, effective, field, interpolation",
    [
        ("auto", "binary", "filtered_openness", INTERPOLATION_NEAREST),
        ("binary", "binary", "filtered_openness", INTERPOLATION_NEAREST),
        ("raw", "raw", "raw_openness", INTERPOLATION_LINEAR),
    ],
)
def test_binarized_record_sources(tmp_path, requested, effective, field, interpolation) -> None:
    """二值化记录：auto 选 binary；raw 恢复原 valid。"""
    write_binarization(tmp_path)
    source = resolve_gripper_source(requested, tmp_path, TOPIC)
    assert (source.effective, source.field, source.interpolation) == (effective, field, interpolation)
    assert source.binarized and source.frame_count == 4
    assert len(source.binarization_sha256) == 64
    expected_valid = (True, False, True, True) if effective == "raw" else None
    assert source.original_valid == expected_valid
    assert source.report()["effective"] == effective


@pytest.mark.parametrize(
    "requested, effective, field",
    [("auto", "filtered", "filtered_openness"), ("filtered", "filtered", "filtered_openness"),
     ("raw", "raw", "raw_openness")],
)
def test_plain_record_sources(tmp_path, requested, effective, field) -> None:
    """未二值化记录：auto 选 filtered，线性插值，沿用消息 valid。"""
    source = resolve_gripper_source(requested, tmp_path, TOPIC)
    assert (source.effective, source.field) == (effective, field)
    assert source.interpolation == INTERPOLATION_LINEAR
    assert not source.binarized and source.original_valid is None


def test_bare_bag_is_treated_as_not_binarized() -> None:
    """没有记录目录的裸 bag 按未二值化处理。"""
    assert resolve_gripper_source("auto", None, TOPIC).effective == "filtered"
    with pytest.raises(ConfigError, match="不能使用二值化"):
        resolve_gripper_source("binary", None, TOPIC)


def test_incompatible_choices_are_rejected(tmp_path) -> None:
    """二值化记录不能选 filtered；未二值化记录不能选 binary。"""
    with pytest.raises(ConfigError, match="不能使用二值化"):
        resolve_gripper_source("binary", tmp_path, TOPIC)
    write_binarization(tmp_path)
    with pytest.raises(ConfigError, match="已被二值结果覆盖"):
        resolve_gripper_source("filtered", tmp_path, TOPIC)
    with pytest.raises(ConfigError, match="未知"):
        resolve_gripper_source("other", tmp_path, TOPIC)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"topic": "/other"}, "话题"),
        ({"schema_version": 2}, "schema_version"),
        ({"frame_count": 5}, "长度"),
        ({"frame_count": 0}, "正整数"),
        ({"original_valid": "10x1"}, "0/1"),
    ],
)
def test_invalid_binarization_file_is_rejected(tmp_path, overrides, message) -> None:
    """binarization.yaml 内容与配置不符时报错。"""
    write_binarization(tmp_path, **overrides)
    with pytest.raises(ConfigError, match=message):
        resolve_gripper_source("auto", tmp_path, TOPIC)
