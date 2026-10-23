"""验证控制器直接读取唯一的 initial_pose.yaml，launch 和 YAML 不保存角度副本。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch.actions import ExecuteProcess
import pytest
import yaml


CONFIG = Path(__file__).parents[1] / "config/rm75_placo_controller.yaml"


def launch_command():
    """加载源码中的 launch 文件，返回控制器进程命令中的字面量参数。"""
    pytest.importorskip("fastumi_bringup.initial_pose")
    spec = spec_from_file_location(
        "rm75_placo_controller_launch",
        Path(__file__).parents[1] / "launch/rm75_placo_controller.launch.py")
    module = module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        description = module.generate_launch_description()
    except RuntimeError as error:
        pytest.skip(str(error))
    process = next(entity for entity in description.entities
                   if isinstance(entity, ExecuteProcess))
    return ["".join(getattr(part, "text", "") for part in item) for item in process.cmd]


def test_launch_does_not_set_start_joints():
    command = launch_command()
    assert not any(value.startswith("start_joint_positions:=") for value in command)


@pytest.mark.parametrize("filename, node_name", [
    ("rm75_placo_controller.yaml", "rm75_placo_controller"),
    ("rm75_joint_controller.yaml", "rm75_joint_controller"),
])
def test_controller_config_has_no_start_joints(filename, node_name):
    """起始关节角只能来自 initial_pose.yaml，控制器配置不得保留第二份副本。"""
    parameters = yaml.safe_load((CONFIG.parent / filename).read_text(encoding="utf-8"))
    assert "start_joint_positions" not in parameters[node_name]["ros__parameters"]


def test_controller_config_has_no_workspace_box():
    """位置保护由示教器虚拟墙负责，控制器配置不得再设一套可能与之冲突的工作区。"""
    parameters = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert not {"workspace_min", "workspace_max"} & set(
        parameters["rm75_placo_controller"]["ros__parameters"])


def test_controller_config_disables_close_gate_by_default():
    parameters = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert parameters["rm75_placo_controller"]["ros__parameters"]["require_close_approval"] is False


def test_controller_config_uses_measured_gripper_calibration_and_sequence_hold():
    """两种夹爪物理刻度一致，取恒等映射；夹空阈值位于空夹与夹住读数之间；序列末点短时保持。"""
    parameters = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    values = parameters["rm75_placo_controller"]["ros__parameters"]
    assert values["grasp_latch_enabled"] is True
    assert values["gripper_map_robot_ref"] == pytest.approx(values["gripper_map_policy_ref"])
    assert 0.031 < values["grasp_miss_state"] < 0.32
    assert 0.0 < values["sequence_hold_s"] <= 1.0


@pytest.mark.parametrize("module_name, class_name, filename", [
    ("rm75_placo_controller", "Rm75PlacoController", "rm75_placo_controller.yaml"),
    ("rm75_joint_controller", "Rm75JointController", "rm75_joint_controller.yaml"),
])
def test_configured_controllers_read_shared_start_joints(module_name, class_name, filename):
    """直接运行两个控制器均从源 YAML 读取起点，不创建或执行运动命令。"""
    from fastumi_bringup.initial_pose import load_initial_joint_positions

    rclpy = pytest.importorskip("rclpy")
    controller = pytest.importorskip(f"fastumi_rm75.{module_name}")
    rclpy.init(args=["--ros-args", "--params-file", str(CONFIG.parent / filename),
                     "-p", "dry_run:=true"])
    node = None
    try:
        node = getattr(controller, class_name)()
        assert node._start_joints == pytest.approx(load_initial_joint_positions())
        assert not node.has_parameter("start_joint_positions")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
