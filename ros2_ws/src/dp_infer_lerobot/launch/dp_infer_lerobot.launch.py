"""在 Jetson 上启动 DP 推理，并可选启动 RM75 Placo 控制器。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration

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
    share = Path(get_package_share_directory("dp_infer_lerobot"))
    value = LaunchConfiguration("checkpoint").perform(context).strip()
    if not value or not Path(value).expanduser().is_dir():
        raise ValueError("checkpoint must name an existing pretrained directory")
    checkpoint = Path(value).expanduser().resolve()
    deployment_config = _path_argument(context, "deployment_config")
    deployment_urdf = _path_argument(context, "urdf_path")
    config_file = _path_argument(context, "config_file")
    # 空字符串使用部署配置中绑定的 Link7 修正；非空时必须是已有文件。
    correction_file = ""
    if LaunchConfiguration("link7_correction_file").perform(context).strip():
        correction_file = str(_path_argument(context, "link7_correction_file"))
    inference_python = _python_argument(
        context, "inference_python", share, Path(".venv-dp-lerobot/bin/python"))
    enabled = LaunchConfiguration("start_controller").perform(context).lower()
    if enabled not in ("true", "false"):
        raise ValueError("start_controller must be true or false")
    dry_run = LaunchConfiguration("dry_run").perform(context).lower()
    if dry_run not in ("true", "false"):
        raise ValueError("dry_run must be true or false")
    record = LaunchConfiguration("record").perform(context).lower()
    if record not in ("true", "false"):
        raise ValueError("record must be true or false")
    close_approval = LaunchConfiguration("require_close_approval").perform(context).lower()
    if close_approval != "false":
        raise ValueError("This backend has no visual close guard; require_close_approval must be false")
    grasp_latch = LaunchConfiguration("grasp_latch").perform(context).lower()
    if grasp_latch not in ("true", "false"):
        raise ValueError("grasp_latch must be true or false")
    image_type = LaunchConfiguration("image_type").perform(context)
    if image_type not in ("compressed", "raw"):
        raise ValueError("image_type must be compressed or raw")
    sequential = LaunchConfiguration("sequential_execution").perform(context).lower()
    if sequential not in ("true", "false"):
        raise ValueError("sequential_execution must be true or false")
    settle = float(LaunchConfiguration("execution_settle_s").perform(context))
    if not 0.0 <= settle < float("inf"):
        raise ValueError("execution_settle_s must be finite and non-negative")
    action_steps = int(LaunchConfiguration("execution_action_steps").perform(context))
    if action_steps < 1:
        raise ValueError("execution_action_steps must be a positive integer")
    steps = int(LaunchConfiguration("num_inference_steps").perform(context))
    if not 1 <= steps <= 50:
        raise ValueError("num_inference_steps must be in [1, 50]")

    parameter_names = (
        "device", "image_topic", "image_type", "joint_topic", "gripper_topic",
        "output_topic", "close_guard_topic", "num_inference_steps", "postprocessors",
        "set_inference_steps_service",
    )
    overrides = {
        name: LaunchConfiguration(name).perform(context)
        for name in parameter_names
    }
    raw_gripper_topic = overrides["gripper_topic"]
    policy_gripper_topic = LaunchConfiguration("policy_gripper_topic").perform(context)
    if enabled == "true":
        # 控制器把宇树夹爪反馈换算到训练刻度后转发，推理观测改用换算后的话题。
        overrides["gripper_topic"] = policy_gripper_topic
    infer_cmd = [
        inference_python, "-m", "dp_infer_lerobot.node", "--ros-args",
        "--params-file", str(config_file),
        "-p", f"checkpoint:={checkpoint}",
        "-p", f"deployment_config:={deployment_config}",
        "-p", f"urdf_path:={deployment_urdf}",
        "-p", f"require_controller_reset:={enabled}",
        "-p", "task_control_enabled:=true",
        "-p", "controller_reset_service:="
        + LaunchConfiguration("controller_reset_service").perform(context),
        "-p", f"record_task:={record}",
        "-p", "record_dir_name:=" + LaunchConfiguration("record_dir_name").perform(context),
        "-p", "record_name:=" + LaunchConfiguration("record_name").perform(context),
        "-p", f"sequential_execution:={sequential}",
        # 显式写成浮点，避免 "1" 被 ROS 解析为整数参数。
        "-p", f"execution_settle_s:={settle!r}",
        "-p", f"execution_action_steps:={action_steps}",
    ]
    for name, value in overrides.items():
        if value:
            infer_cmd.extend(("-p", f"{name}:={value}"))
    if correction_file:
        infer_cmd.extend(("-p", f"link7_correction_file:={correction_file}"))
    from dp_infer_lerobot.config import DiffusionConfig
    from dp_infer_lerobot.deployment import (load_profile, validate_checkpoint,
                                            validate_urdf, load_correction)
    profile = load_profile(deployment_config)
    validate_checkpoint(checkpoint, profile)
    validate_urdf(deployment_urdf, profile)
    load_correction(correction_file, profile)
    DiffusionConfig.from_file(checkpoint / "config.json")
    inference = ExecuteProcess(cmd=infer_cmd, output="screen")
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
                        if control_urdf_arg else deployment_urdf)
        if not control_urdf.is_file():
            raise FileNotFoundError(f"RM75 control URDF does not exist: {control_urdf}")
        validate_urdf(control_urdf, profile)
        validate_kinematic_equivalence(deployment_urdf, control_urdf)
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
            "-p", f"urdf_path:={control_urdf}",
            "-p", f"joint_state_topic:={overrides['joint_topic']}",
            "-p", f"gripper_state_topic:={raw_gripper_topic}",
            "-p", f"policy_gripper_state_topic:={policy_gripper_topic}",
            "-p", f"grasp_latch_enabled:={grasp_latch}",
            "-p", f"policy_topic:={overrides['output_topic']}",
            "-p", f"require_close_approval:={close_approval}",
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
    """暴露部署所需参数，缺省待命并启动 dry-run 控制器。"""
    share = Path(get_package_share_directory("dp_infer_lerobot"))
    defaults = {
        "checkpoint": "",
        "urdf_path": str(Path(get_package_share_directory("fastumi_rm75")) / "assets/rm_75_kinematic.urdf"),
        "deployment_config": str(share / "config/jingbao_deployment.yaml"),
        "config_file": str(share / "config/dp_infer_lerobot.yaml"),
        # 与 jingbao_deployment.yaml 绑定；重装相机或更换模型后须重新审核生成。
        "link7_correction_file": str(share / "config/link7_correction.yaml"),
        "inference_python": "", "controller_python": "",
        "control_urdf_path": "", "start_controller": "true", "dry_run": "true",
        "require_close_approval": "false",
        "grasp_latch": "false", "policy_gripper_topic": "/fastumi/policy/gripper_state",
        "record": "false", "record_dir_name": "dp_infer_lerobot", "record_name": "rollout",
        "device": "cuda:0", "num_inference_steps": "16",
        "image_topic": "/wrist_camera/image_raw/compressed",
        "image_type": "compressed", "joint_topic": "/joint_states",
        "gripper_topic": "/motion_control/gripper_state",
        "output_topic": "/fastumi/policy/action_sequence",
        "close_guard_topic": "/fastumi/policy/gripper_close_allowed",
        "set_inference_steps_service": "/fastumi/policy/set_inference_steps",
        "controller_reset_service": "/fastumi/rm75/placo/reset_episode",
        "max_start_displacement_m": "0.0", "max_start_rise_m": "0.0",
        "postprocessors": "",
        # 串行模式：整段动作执行完并稳定 execution_settle_s 秒后才进行下一次观测+推理。
        "sequential_execution": "false", "execution_settle_s": "0.3",
        # 串行模式每段执行的动作数，轨迹另含起始锚点。
        "execution_action_steps": "8",
    }
    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value=value)
          for name, value in defaults.items()),
        OpaqueFunction(function=_start),
    ])
