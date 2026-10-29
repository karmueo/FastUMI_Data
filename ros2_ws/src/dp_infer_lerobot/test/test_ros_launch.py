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
    for name in ('load_profile', 'validate_checkpoint', 'validate_urdf'):
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
    assert 'num_inference_steps:=16' in commands[0] and 'noise_scheduler:=DDIM' in commands[0]
    assert any(item.startswith('end_camera_extrinsic:=') for item in commands[0])
    assert any(item.startswith('deploy_tcp_extrinsic:=') for item in commands[0])
    assert not any('link7_correction_file' in item for item in commands[0])
    assert 'gripper_topic:=/fastumi/policy/gripper_state' in commands[0]
    assert 'dry_run:=true' in commands[1] and 'grasp_latch_enabled:=false' in commands[1]
    assert all('task_control_enabled:=true' in command for command in commands)
    assert 'require_close_approval:=false' in commands[1]
    assert 'start_from_current_pose:=true' in commands[1]
    assert 'sequential_execution:=false' in commands[0]
    context.launch_configurations['start_from_current_pose'] = 'false'
    commands = [[perform_substitutions(context, value) for value in action.cmd]
                for action in module._start(context) if isinstance(action, ExecuteProcess)]
    assert 'start_from_current_pose:=false' in commands[1]
    context.launch_configurations['start_from_current_pose'] = 'yes'
    with pytest.raises(ValueError, match='start_from_current_pose'):
        module._start(context)
    context.launch_configurations['start_from_current_pose'] = 'true'
    context.launch_configurations['start_controller'] = 'false'
    commands = [[perform_substitutions(context, value) for value in action.cmd]
                for action in module._start(context) if isinstance(action, ExecuteProcess)]
    assert len(commands) == 1 and 'gripper_topic:=/motion_control/gripper_state' in commands[0]
    context.launch_configurations['checkpoint'] = str(tmp_path / 'assets/rm_75_kinematic.urdf')
    with pytest.raises(ValueError, match='directory'):
        module._start(context)


def test_launch_validates_tensorrt_package_and_passes_only_export_dir(tmp_path, monkeypatch):
    """TensorRT 后端改为校验本机模型包，只向节点传递 backend 与 export_dir。"""
    path = Path(__file__).resolve().parents[1] / 'launch/dp_infer_lerobot.launch.py'
    spec = spec_from_file_location('lerobot_launch_trt_test', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    params = tmp_path / 'params.yaml'
    params.write_text('dp_infer_lerobot:\n  ros__parameters:\n    backend: tensorrt_fp16\n')
    export = tmp_path / 'export'
    (export / 'assets').mkdir(parents=True)
    monkeypatch.setattr(module, 'get_package_share_directory', lambda _: str(tmp_path))
    monkeypatch.setattr(module, '_path_argument', lambda _, name: params if name == 'config_file' else params)
    monkeypatch.setattr(module, '_python_argument', lambda *_: '/usr/bin/python3')
    checked = []  # validate_export 收到的模型包与精度。
    monkeypatch.setattr(deployment, 'load_profile', lambda _: {})
    monkeypatch.setattr(deployment, 'validate_checkpoint', lambda *_: pytest.fail('checkpoint not used'))
    monkeypatch.setattr(deployment, 'validate_export', lambda root, _, precision: checked.append((root, precision)))
    monkeypatch.setattr(deployment, 'validate_urdf', lambda *_: None)
    monkeypatch.setattr(DiffusionConfig, 'from_file', lambda *_: None)
    context = LaunchContext()
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            context.launch_configurations[action.name] = perform_substitutions(context, action.default_value)
    context.launch_configurations.update({'export_dir': str(export), 'start_controller': 'false'})
    commands = [[perform_substitutions(context, value) for value in action.cmd]
                for action in module._start(context) if isinstance(action, ExecuteProcess)]
    assert checked == [(export.resolve(), 'fp16')]
    assert f'export_dir:={export.resolve()}' in commands[0] and 'backend:=tensorrt_fp16' in commands[0]
    assert not any(item.startswith('checkpoint:=') for item in commands[0])
    # 命令行 backend 覆盖参数文件：pytorch 需要 checkpoint。
    context.launch_configurations['backend'] = 'pytorch'
    with pytest.raises(ValueError, match='checkpoint'):
        module._start(context)
    context.launch_configurations['backend'] = 'onnx'
    with pytest.raises(ValueError, match='backend'):
        module._start(context)


@pytest.mark.parametrize('backend', ['pytorch', 'tensorrt_fp16'])
@pytest.mark.parametrize('variant', ['dp_umi_v4', 'dp_umi_v5'])
@pytest.mark.parametrize('scheduler', ['', 'DDIM', 'DDPM'])
def test_launch_passes_resolved_scheduler(tmp_path, monkeypatch, backend, variant, scheduler):
    """父 launch 的空调度器沿用训练配置，显式选择覆盖参数文件，两种模型后端保持一致。"""
    import json

    from dp_infer_lerobot.config import COMMON, VARIANTS

    path = Path(__file__).resolve().parents[1] / 'launch/dp_infer_lerobot.launch.py'
    spec = spec_from_file_location('lerobot_launch_scheduler_test', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assets = tmp_path if backend == 'pytorch' else tmp_path / 'assets'
    assets.mkdir(exist_ok=True)
    (assets / 'config.json').write_text(json.dumps(dict(COMMON, **VARIANTS[variant])))
    params = tmp_path / 'params.yaml'
    params.write_text('dp_infer_lerobot:\n  ros__parameters:\n    noise_scheduler: DDIM\n')
    monkeypatch.setattr(module, 'get_package_share_directory', lambda _: str(tmp_path))
    monkeypatch.setattr(module, '_path_argument', lambda *_: params)
    monkeypatch.setattr(module, '_python_argument', lambda *_: '/usr/bin/python3')
    for name in ('load_profile', 'validate_checkpoint', 'validate_export', 'validate_urdf'):
        monkeypatch.setattr(deployment, name, lambda *_: {})
    context = LaunchContext()
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            context.launch_configurations[action.name] = perform_substitutions(context, action.default_value)
    context.launch_configurations.update(
        backend=backend, checkpoint=str(tmp_path), export_dir=str(tmp_path),
        start_controller='false', noise_scheduler=scheduler)
    commands = [[perform_substitutions(context, value) for value in action.cmd]
                for action in module._start(context) if isinstance(action, ExecuteProcess)]
    expected = scheduler or VARIANTS[variant]['noise_scheduler_type']
    assert [arg for arg in commands[0] if arg.startswith('noise_scheduler:=')] == [
        f'noise_scheduler:={expected}']
