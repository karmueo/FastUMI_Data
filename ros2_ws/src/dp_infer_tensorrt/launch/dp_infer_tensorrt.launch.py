"""在 Jetson 上启动 DP 推理，并可选启动 RM75 Placo 控制器。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from fastumi_bringup.initial_pose import INITIAL_JOINT_POSITIONS
from dp_infer.urdf_contract import validate_kinematic_equivalence


def _candidate_python(package_share, relative):
    """在已安装工作区的祖先目录中寻找预置的 Jetson 环境。"""
    for base in (package_share, *package_share.parents):
        candidate = base / relative
        if candidate.is_file():
            # venv/bin/python 通常是系统 Python 的符号链接；保留 venv 路径。
            return candidate.absolute()
    raise FileNotFoundError(f"Cannot find {relative}; pass an explicit Python executable")


def _path_argument(context, name):
    value = LaunchConfiguration(name).perform(context).strip()
    if not value:
        raise ValueError(f"{name} must name an existing file")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _python_argument(context, name, package_share, relative):
    value = LaunchConfiguration(name).perform(context).strip()
    path = Path(value).expanduser().absolute() if value else _candidate_python(package_share, relative)
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return str(path)


def _start(context):
    """先验证模型文件与运动链，再创建两个受共同退出事件管理的进程。"""
    share = Path(get_package_share_directory("dp_infer_tensorrt"))
    engine_dir = Path(LaunchConfiguration("engine_dir").perform(context)).expanduser().resolve()
    if not engine_dir.is_dir() or not (engine_dir / "manifest.json").is_file():
        raise ValueError("engine_dir must name an existing TensorRT bundle")
    model_manifest = LaunchConfiguration("model_manifest").perform(context).strip()
    if model_manifest:
        model_manifest = str(_path_argument(context, "model_manifest"))
    training_urdf = _path_argument(context, "urdf_path")
    config_file = _path_argument(context, "config_file")
    enabled = LaunchConfiguration("start_controller").perform(context).lower()
    if enabled not in ("true", "false"):
        raise ValueError("start_controller must be true or false")
    dry_run = LaunchConfiguration("dry_run").perform(context).lower()
    if dry_run not in ("true", "false"):
        raise ValueError("dry_run must be true or false")
    image_type = LaunchConfiguration("image_type").perform(context)
    if image_type not in ("compressed", "raw"):
        raise ValueError("image_type must be compressed or raw")
    steps = int(LaunchConfiguration("num_inference_steps").perform(context))
    if not 1 <= steps <= 50:
        raise ValueError("num_inference_steps must be in [1, 50]")

    parameter_names = (
        "device", "precision", "image_topic", "image_type", "joint_topic", "gripper_topic",
        "output_topic", "close_guard_topic", "num_inference_steps", "postprocessors",
        "set_inference_steps_service",
    )
    overrides = {
        name: LaunchConfiguration(name).perform(context)
        for name in parameter_names
    }
    overrides["num_inference_steps"] = steps
    parameters = {**overrides, "engine_dir": str(engine_dir),
                  "model_manifest": model_manifest, "urdf_path": str(training_urdf),
                  "require_controller_reset": enabled == "true", "task_control_enabled": True,
                  "controller_reset_service": LaunchConfiguration("controller_reset_service").perform(context)}
    inference = Node(package="dp_infer_tensorrt", executable="dp_infer_tensorrt_node",
                     output="screen", parameters=[str(config_file), parameters])
    actions = [
        RegisterEventHandler(OnProcessExit(
            target_action=inference,
            on_exit=[EmitEvent(event=Shutdown(reason="DP inference process exited"))],
        )),
        inference,
    ]

    if enabled == "true":
        control_share = Path(get_package_share_directory("fastumi_rm75"))
        control_urdf_arg = LaunchConfiguration("control_urdf_path").perform(context).strip()
        control_urdf = (Path(control_urdf_arg).expanduser().resolve()
                        if control_urdf_arg else control_share / "assets/rm_75_kinematic.urdf")
        if not control_urdf.is_file():
            raise FileNotFoundError(f"RM75 control URDF does not exist: {control_urdf}")
        validate_kinematic_equivalence(training_urdf, control_urdf)
        controller_python = _python_argument(
            context, "controller_python", control_share,
            Path(".venv-numpy2/bin/python"))
        control_config = control_share / "config/rm75_placo_controller.yaml"
        if not control_config.is_file():
            raise FileNotFoundError(f"RM75 controller config does not exist: {control_config}")
        controller = ExecuteProcess(cmd=[
            controller_python, "-m", "fastumi_rm75.rm75_placo_controller",
            "--ros-args", "--params-file", str(control_config),
            "-p", f"dry_run:={dry_run}",
            "-p", "task_control_enabled:=true",
            "-p", "start_joint_positions:=[" + ",".join(
                str(value) for value in INITIAL_JOINT_POSITIONS) + "]",
            "-p", f"urdf_path:={control_urdf}",
            "-p", f"joint_state_topic:={overrides['joint_topic']}",
            "-p", f"gripper_state_topic:={overrides['gripper_topic']}",
            "-p", f"policy_topic:={overrides['output_topic']}",
            "-p", f"close_approval_topic:={overrides['close_guard_topic']}",
            "-p", "reset_service:="
            + LaunchConfiguration("controller_reset_service").perform(context),
            "-p", "max_start_displacement_m:="
            + LaunchConfiguration("max_start_displacement_m").perform(context),
            "-p", "max_start_rise_m:="
            + LaunchConfiguration("max_start_rise_m").perform(context),
        ], output="screen")
        actions.extend((
            RegisterEventHandler(OnProcessExit(
                target_action=controller,
                on_exit=[EmitEvent(event=Shutdown(reason="RM75 controller process exited"))],
            )),
            controller,
        ))
    return actions


def generate_launch_description():
    """暴露部署所需参数，缺省启动推理和实机控制。"""
    share = Path(get_package_share_directory("dp_infer_tensorrt"))
    defaults = {
        "engine_dir": "", "model_manifest": "", "urdf_path": "",
        "config_file": str(share / "config/dp_infer_tensorrt.yaml"),
        "controller_python": "",
        "control_urdf_path": "", "start_controller": "true", "dry_run": "false",
        "device": "cuda:0", "precision": "fp16", "num_inference_steps": "8",
        "image_topic": "/wrist_camera/image_raw/compressed",
        "image_type": "compressed", "joint_topic": "/joint_states",
        "gripper_topic": "/motion_control/gripper_state",
        "output_topic": "/fastumi/policy/action_sequence",
        "close_guard_topic": "/fastumi/policy/gripper_close_allowed",
        "set_inference_steps_service": "/fastumi/policy/set_inference_steps",
        "controller_reset_service": "/fastumi/rm75/placo/reset_episode",
        "max_start_displacement_m": "0.0", "max_start_rise_m": "0.0",
        "postprocessors": "",
    }
    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value=value)
          for name, value in defaults.items()),
        OpaqueFunction(function=_start),
    ])
