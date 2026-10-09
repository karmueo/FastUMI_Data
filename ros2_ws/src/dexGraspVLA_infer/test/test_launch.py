"""启动契约检查：默认 dry-run、共享 GPU 调度和独立控制配置。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, ExecuteProcess


def module():
    spec = spec_from_file_location("dex_launch", Path(__file__).parents[1] / "launch/dexgraspvla_infer.launch.py")
    value = module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_default_launch_is_dry_and_combined(monkeypatch, tmp_path):
    value = module()
    monkeypatch.setattr(value, "get_package_share_directory", lambda name: str(tmp_path))
    description = value.generate_launch_description()
    defaults = {item.name: item.default_value[0].text for item in description.entities
                if isinstance(item, DeclareLaunchArgument)}
    assert defaults["dry_run"] == "true"
    assert defaults["mode"] == "combined"
    assert defaults["num_inference_steps"] == "16"
    assert defaults["image_topic"] == "/wrist_camera/image_decoded"


def test_launch_does_not_override_custom_home_or_start_hardware(monkeypatch, tmp_path):
    value = module()
    repo = Path(__file__).resolve().parents[4]
    shares = {"dexgraspvla_infer": Path(__file__).parents[1],
              "fastumi_rm75": repo / "ros2_ws/src/fastumi_rm75"}
    monkeypatch.setattr(value, "get_package_share_directory", lambda name: str(shares[name]))
    context = LaunchContext()
    context.launch_configurations.update({
        "repo_root": str(repo), "asset_root": "", "checkpoint": "", "urdf_path": "",
        "inference_python": "/usr/bin/python3", "controller_python": "/usr/bin/python3",
        "config_file": "", "controller_config": str(tmp_path / "custom.yaml"),
        "dry_run": "true", "start_controller": "true", "mode": "combined",
        "num_inference_steps": "4", "image_topic": "/wrist_camera/image_decoded", "image_type": "raw",
    })
    processes = [item for item in value.start(context) if isinstance(item, ExecuteProcess)]
    commands = [[part[0].text for part in item.cmd] for item in processes]
    assert len(commands) == 2
    assert "fastumi_rm75.rm75_joint_controller" in commands[1]
    assert str(tmp_path / "custom.yaml") in commands[1]
    assert not any("start_joint_positions:=" in word for word in commands[1])
    assert not any("hardware.launch" in word or "rm_driver" in word for command in commands for word in command)
