"""验证 launch 将外参轴向校验交给使用最终 ROS 参数的节点。"""

import importlib.util
from pathlib import Path
import sys

import pytest

pytest.importorskip("launch")
from launch import LaunchContext  # noqa: E402

from dp_infer_lerobot.config import DiffusionConfig  # noqa: E402
from dp_infer_lerobot import deployment  # noqa: E402


def test_launch_defers_tcp_validation_to_node(tmp_path, monkeypatch):
    """launch 不再使用固定 10° 提前拒绝配置了更大阈值的节点。"""
    path = Path(__file__).resolve().parents[1] / "launch/dp_infer_lerobot.launch.py"
    spec = importlib.util.spec_from_file_location("lerobot_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "params.yaml"
    config.write_text("dp_infer_lerobot:\n  ros__parameters:\n    max_tcp_axis_deg: 15.0\n")
    context = LaunchContext()
    context.launch_configurations.update({name: "" for name in module.OVERRIDE_NAMES})
    context.launch_configurations.update({
        "checkpoint": str(tmp_path), "inference_python": sys.executable,
        **{name: str(config) for name in (
            "deployment_config", "urdf_path", "config_file", "end_camera_extrinsic",
            "deploy_tcp_extrinsic")},
    })
    monkeypatch.setattr(module, "get_package_share_directory", lambda _: str(tmp_path))
    monkeypatch.setattr(deployment, "load_profile", lambda _: {})
    monkeypatch.setattr(deployment, "validate_checkpoint", lambda *_: None)
    monkeypatch.setattr(deployment, "validate_urdf", lambda *_: None)
    monkeypatch.setattr(DiffusionConfig, "from_file", lambda *_: None)

    def reject_duplicate_validation(*args):
        """发现 launch 重复执行使用固定阈值的外参校验时令测试失败。"""
        pytest.fail("TCP axis validation must use the node's effective ROS parameters")

    monkeypatch.setattr(deployment, "load_tool_offset", reject_duplicate_validation)
    actions = module._start(context)
    assert len(actions) == 2
