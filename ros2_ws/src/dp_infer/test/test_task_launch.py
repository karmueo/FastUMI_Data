"""验证组合启动确实让推理和控制器从待命开始。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.utilities import perform_substitutions

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
    # 控制器直接读取唯一的 initial_pose.yaml，launch 不再另设起始角度。
    assert not any("start_joint_positions:=" in item for command in commands for item in command)
    # 视觉闭合门控默认关闭，launch 参数可显式启用。
    assert "require_close_approval:=false" in commands[1]
    context.launch_configurations["require_close_approval"] = "true"
    enabled = [perform_substitutions(context, item) for item in next(
        action for action in module._start(context)
        if isinstance(action, ExecuteProcess) and "fastumi_rm75.rm75_placo_controller" in [
            perform_substitutions(context, item) for item in action.cmd]).cmd]
    assert "require_close_approval:=true" in enabled
    # 推理观测订阅控制器换算后的训练刻度夹爪反馈，控制器仍读宇树原始反馈。
    assert "gripper_topic:=/fastumi/policy/gripper_state" in commands[0]
    assert "gripper_state_topic:=/motion_control/gripper_state" in commands[1]
    assert "policy_gripper_state_topic:=/fastumi/policy/gripper_state" in commands[1]
    assert "grasp_latch_enabled:=true" in commands[1]
    # 推理过程录制默认关闭，record:=true 时推理节点开启任务录制。
    assert "record_task:=false" in commands[0]
    context.launch_configurations["record"] = "true"
    inference = [perform_substitutions(context, item) for item in next(
        action for action in module._start(context) if isinstance(action, ExecuteProcess)).cmd]
    assert "record_task:=true" in inference and "record_name:=rollout" in inference
    # Link7 部署修正默认传入随包配置文件；置空则不传参数（节点按单位阵处理）。
    assert f"link7_correction_file:={tmp_path / 'model.ckpt'}" in commands[0]
    context.launch_configurations["link7_correction_file"] = ""
    uncorrected = [perform_substitutions(context, item) for item in next(
        action for action in module._start(context) if isinstance(action, ExecuteProcess)).cmd]
    assert not any(item.startswith("link7_correction_file:=") for item in uncorrected)
    # 不启动控制器时没有换算话题，推理直接订阅宇树原始反馈。
    context.launch_configurations["start_controller"] = "false"
    alone = [perform_substitutions(context, item) for item in next(
        action for action in module._start(context) if isinstance(action, ExecuteProcess)).cmd]
    assert "gripper_topic:=/motion_control/gripper_state" in alone
