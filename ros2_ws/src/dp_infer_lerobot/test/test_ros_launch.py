"""目标 ROS 环境的 launch 构造测试；仅查看命令，不启动任何进程。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

pytest.importorskip('launch')
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.utilities import perform_substitutions
from dp_infer_lerobot import deployment
from dp_infer_lerobot.config import DiffusionConfig


def test_launch_replaces_backend_preserves_interfaces_and_defaults_to_dry_run(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'launch/dp_infer_lerobot.launch.py'
    spec = spec_from_file_location('lerobot_launch_test', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'assets').mkdir()
    (tmp_path / 'assets/rm_75_kinematic.urdf').write_text('<robot/>')
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/rm75_placo_controller.yaml').write_text('{}')
    monkeypatch.setattr(module, 'get_package_share_directory', lambda _: str(tmp_path))
    monkeypatch.setattr(module, '_path_argument', lambda *_: tmp_path / 'assets/rm_75_kinematic.urdf')
    monkeypatch.setattr(module, '_python_argument', lambda *_: '/usr/bin/python3')
    monkeypatch.setattr(module, 'validate_kinematic_equivalence', lambda *_: None)
    for name in ('load_profile', 'validate_checkpoint', 'validate_urdf', 'load_correction'):
        monkeypatch.setattr(deployment, name, lambda *_: {})
    monkeypatch.setattr(DiffusionConfig, 'from_file', lambda *_: None)
    context = LaunchContext()
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            context.launch_configurations[action.name] = perform_substitutions(context, action.default_value)
    context.launch_configurations['checkpoint'] = str(tmp_path)
    commands = [[perform_substitutions(context, value) for value in action.cmd]
                for action in module._start(context) if isinstance(action, ExecuteProcess)]
    assert len(commands) == 2
    assert 'dp_infer_lerobot.node' in commands[0]
    assert 'num_inference_steps:=16' in commands[0]
    assert 'gripper_topic:=/fastumi/policy/gripper_state' in commands[0]
    assert 'dry_run:=true' in commands[1] and 'grasp_latch_enabled:=false' in commands[1]
    assert all('task_control_enabled:=true' in command for command in commands)
    assert 'require_close_approval:=false' in commands[1]
    context.launch_configurations['start_controller'] = 'false'
    commands = [[perform_substitutions(context, value) for value in action.cmd]
                for action in module._start(context) if isinstance(action, ExecuteProcess)]
    assert len(commands) == 1 and 'gripper_topic:=/motion_control/gripper_state' in commands[0]
    context.launch_configurations['checkpoint'] = str(tmp_path / 'assets/rm_75_kinematic.urdf')
    with pytest.raises(ValueError, match='directory'):
        module._start(context)
