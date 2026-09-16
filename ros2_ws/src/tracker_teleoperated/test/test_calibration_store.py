"""验证 Tracker 工作空间标定文件的解析、校验和原子替换。"""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from tracker_teleoperated.calibration_store import (
    CalibrationStoreError,
    load_workspace_calibration,
    resolve_workspace_calibration_path,
    save_workspace_calibration,
)


def test_default_path_uses_ros_home(monkeypatch, tmp_path):
    """验证空参数将标定文件放在 ROS_HOME 的包级目录。"""
    monkeypatch.setenv("ROS_HOME", str(tmp_path))

    result = resolve_workspace_calibration_path("")

    assert result == tmp_path / "tracker_teleoperated/workspace_calibration.yaml"


def test_custom_path_is_expanded(monkeypatch, tmp_path):
    """验证显式路径覆盖默认位置并展开用户目录。"""
    monkeypatch.setenv("HOME", str(tmp_path))

    result = resolve_workspace_calibration_path("~/mapping.yaml")

    assert result == tmp_path / "mapping.yaml"


def test_save_and_reload_preserves_frames_and_mapping(tmp_path):
    """验证节点重启场景可从 YAML 恢复同一非单位映射方向。"""
    path = tmp_path / "nested/workspace.yaml"
    mapping = Rotation.from_euler("xyz", [10.0, 20.0, 30.0], degrees=True).as_matrix()

    save_workspace_calibration(path, "vive_tracker_odom", "base_link", mapping)
    loaded = load_workspace_calibration(
        path, "vive_tracker_odom", "base_link"
    )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert loaded == pytest.approx(mapping)
    assert document["version"] == 1
    assert document["odom_frame"] == "vive_tracker_odom"
    assert document["base_frame"] == "base_link"
    assert set(document) == {
        "version",
        "odom_frame",
        "base_frame",
        "mapping_basis",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 2, "版本"),
        ("odom_frame", "other_odom", "odom_frame"),
        ("base_frame", "other_base", "base_frame"),
        ("mapping_basis", np.ones((3, 3)).tolist(), "正交"),
    ],
)
def test_load_rejects_incompatible_or_invalid_file(
    tmp_path, field, value, message
):
    """验证版本、坐标系和矩阵不兼容时拒绝恢复标定。"""
    path = tmp_path / "workspace.yaml"
    document = {
        "version": 1,
        "odom_frame": "vive_tracker_odom",
        "base_frame": "base_link",
        "mapping_basis": np.eye(3).tolist(),
    }
    document[field] = value
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(CalibrationStoreError, match=message):
        load_workspace_calibration(path, "vive_tracker_odom", "base_link")


def test_load_rejects_malformed_yaml(tmp_path):
    """验证损坏的 YAML 不会被当作有效标定恢复。"""
    path = tmp_path / "workspace.yaml"
    path.write_text("mapping_basis: [\n", encoding="utf-8")

    with pytest.raises(CalibrationStoreError, match="无法读取"):
        load_workspace_calibration(path, "vive_tracker_odom", "base_link")


def test_failed_replace_keeps_previous_file(monkeypatch, tmp_path):
    """验证原子替换失败时保留旧文件并清理临时文件。"""
    path = tmp_path / "workspace.yaml"
    old_mapping = np.eye(3)
    new_mapping = Rotation.from_euler("z", 30.0, degrees=True).as_matrix()
    save_workspace_calibration(
        path, "vive_tracker_odom", "base_link", old_mapping
    )
    old_bytes = path.read_bytes()

    def fail_replace(_source, _destination):
        """模拟目标文件原子替换失败。"""
        raise OSError("磁盘只读")

    monkeypatch.setattr("tracker_teleoperated.calibration_store.os.replace", fail_replace)

    with pytest.raises(CalibrationStoreError, match="磁盘只读"):
        save_workspace_calibration(
            path, "vive_tracker_odom", "base_link", new_mapping
        )

    assert path.read_bytes() == old_bytes
    assert list(tmp_path.glob(".*.tmp")) == []
