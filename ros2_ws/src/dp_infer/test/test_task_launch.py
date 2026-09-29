"""验证组合启动确实让推理和控制器从待命开始。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.utilities import perform_substitutions

from fastumi_bringup.initial_pose import INITIAL_JOINT_POSITIONS


def test_launch_enables_task_gate_and_shared_home(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / "launch/dp_infer.launch.py"
    spec = spec_from_file_location("task_launch", script)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    control_dir = tmp_path / "assets"
    control_dir.mkdir()
    (control_dir / "rm_75_kinematic.urdf").write_text("<robot/>")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "rm75_placo_controller.yaml").write_text("controller: {}")
    monkeypatch.setattr(module, "get_package_share_directory", lambda _: str(tmp_path))
    monkeypatch.setattr(module, "_path_argument", lambda *_: tmp_path / "model.ckpt")
    monkeypatch.setattr(module, "_python_argument", lambda *_: "/usr/bin/python3")
    monkeypatch.setattr(module, "validate_kinematic_equivalence", lambda *_: None)
    context = LaunchContext()
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            context.launch_configurations[action.name] = perform_substitutions(
                context, action.default_value)
    actions = module._start(context)
    commands = [
        [perform_substitutions(context, item) for item in action.cmd]
        for action in actions if isinstance(action, ExecuteProcess)
    ]
    assert len(commands) == 2
    assert all("task_control_enabled:=true" in command for command in commands)
    assert "set_inference_steps_service:=/fastumi/policy/set_inference_steps" in commands[0]
    target = "start_joint_positions:=[" + ",".join(
        str(value) for value in INITIAL_JOINT_POSITIONS) + "]"
    assert target in commands[1]
