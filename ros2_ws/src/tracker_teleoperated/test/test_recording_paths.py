"""验证管理面板与记录节点共享的输出目录规则。"""

from pathlib import Path

import pytest

from tracker_teleoperated import recording_paths


def test_output_directory(tmp_path, monkeypatch):
    """相对路径、用户目录和中文空格任务名均解析为一致的绝对目录。"""
    monkeypatch.setattr(recording_paths, "repository_root", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert recording_paths.output_directory("dataset", " 采集任务 ", "测试 1") == (
        tmp_path / "dataset/采集任务/测试 1"
    )
    assert recording_paths.output_directory("~/dataset", "test", "sample") == (
        tmp_path / "dataset/test/sample"
    )
    assert recording_paths.output_directory(str(tmp_path / "data/../dataset"), "test", "sample") == (
        tmp_path / "dataset/test/sample"
    )


@pytest.mark.parametrize("part", ["", " ", ".", "..", "a/b", "/tmp"])
@pytest.mark.parametrize("position", [0, 1])
def test_invalid_task_names(part, position):
    """两层任务名都拒绝空值及跨目录路径，维持原有记录格式约束。"""
    names = ["task", "name"]
    names[position] = part
    with pytest.raises(ValueError, match="单层非空目录名"):
        recording_paths.output_directory("/tmp", *names)
