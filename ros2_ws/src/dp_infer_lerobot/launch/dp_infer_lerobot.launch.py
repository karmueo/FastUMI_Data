"""在 Jetson 上启动 LeRobot TCP 扩散策略推理，并可选启动 RM75 Placo 控制器。

启动前校验模型指纹、训练 URDF、模型配置与部署外参文件；推理 FK 使用训练 URDF，
控制器使用运动学等价的 RM75 运动学 URDF。

``backend`` 为 ``pytorch`` 时校验 ``checkpoint``；为 ``tensorrt_fp32``/``tensorrt_fp16`` 时校验
``export_dir`` 本机模型包（manifest 摘要、部署档绑定与该精度的验证报告），不需要 checkpoint。
"""

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
    """把 launch 参数解析为已存在文件的绝对路径。"""
    value = LaunchConfiguration(name).perform(context).strip()
    if not value:
        raise ValueError(f"{name} must name an existing file")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _python_argument(context, name, package_share, relative):
    """解析显式解释器参数，空值时在工作区祖先目录中查找。"""
    value = LaunchConfiguration(name).perform(context).strip()
    path = Path(value).expanduser().absolute() if value else _candidate_python(package_share, relative)
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return str(path)


def _file_parameter(config_file, name, default):
    """读取参数文件中推理节点的某个 ROS 参数，缺失时返回默认值。"""
    import yaml

    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    node = raw.get("dp_infer_lerobot") if isinstance(raw, dict) else None
    parameters = node.get("ros__parameters") if isinstance(node, dict) else None
    return str(parameters.get(name, default)) if isinstance(parameters, dict) else default


def _start(context):
    """先验证模型文件与运动链，再创建两个受共同退出事件管理的进程。"""
    share = Path(get_package_share_directory("dp_infer_lerobot"))
    deployment_config = _path_argument(context, "deployment_config")
    # 训练数据记录的 URDF（推理 FK 用），须与部署档 urdf_sha256 一致。
    deployment_urdf = _path_argument(context, "urdf_path")
    config_file = _path_argument(context, "config_file")
    from dp_infer_lerobot.config import validate_backend

    # 未覆盖 backend 时取参数文件中的值，与节点实际生效的后端一致。
    backend = (LaunchConfiguration("backend").perform(context).strip()
               or _file_parameter(config_file, "backend", "pytorch"))
    precision = validate_backend(backend)
    model_key = "checkpoint" if precision is None else "export_dir"
    value = LaunchConfiguration(model_key).perform(context).strip()
    if not value or not Path(value).expanduser().is_dir():
        raise ValueError(f"{model_key} must name an existing directory for backend {backend}")
    model_dir = Path(value).expanduser().resolve()
    end_camera = _path_argument(context, "end_camera_extrinsic")
    deploy_tcp = _path_argument(context, "deploy_tcp_extrinsic")
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
    start_current = LaunchConfiguration("start_from_current_pose").perform(context).lower()
    if start_current not in ("true", "false"):
        raise ValueError("start_from_current_pose must be true or false")
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
    published_steps = int(LaunchConfiguration("published_action_steps").perform(context))
    if not 8 <= published_steps <= 15:
        raise ValueError("published_action_steps must be in [8, 15]")
    lookahead = float(LaunchConfiguration("lookahead_s").perform(context))
    if not 0.0 <= lookahead <= 0.5:
        raise ValueError("lookahead_s must be in [0, 0.5]")
    # 串行运行层会按 execution_action_steps 截断序列，无法保留流水线前瞻所需的后续点。
    if sequential == "true" and lookahead > 0:
        raise ValueError("lookahead_s must be 0 when sequential_execution is true")
    steps = int(LaunchConfiguration("num_inference_steps").perform(context))
    scheduler = LaunchConfiguration("noise_scheduler").perform(context).strip()

    parameter_names = (
        "device", "image_topic", "image_type", "joint_topic", "gripper_topic",
        "output_topic", "close_guard_topic", "num_inference_steps", "noise_scheduler", "postprocessors",
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
        "-p", f"backend:={backend}",
        "-p", f"{model_key}:={model_dir}",
        "-p", f"deployment_config:={deployment_config}",
        "-p", f"urdf_path:={deployment_urdf}",
        "-p", f"end_camera_extrinsic:={end_camera}",
        "-p", f"deploy_tcp_extrinsic:={deploy_tcp}",
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
        "-p", f"published_action_steps:={published_steps}",
    ]
    from dp_infer_lerobot.config import DiffusionConfig, validate_steps
    from dp_infer_lerobot.deployment import load_profile, validate_checkpoint, validate_export, validate_urdf
    validate_steps(steps)
    profile = load_profile(deployment_config)
    if precision is None:
        validate_checkpoint(model_dir, profile)
        assets = model_dir
    else:
        validate_export(model_dir, profile, precision)
        assets = model_dir / "assets"
    validate_urdf(deployment_urdf, profile)
    # 外参轴向校验由节点执行，使用参数文件中实际生效的 max_tcp_axis_deg。
    model_config = DiffusionConfig.from_file(assets / "config.json", scheduler or None, steps)
    # 空值表示训练调度器，须传递解析后的名称，避免节点回退到参数文件中的 DDIM。
    overrides["noise_scheduler"] = scheduler or model_config.noise_scheduler_type
    for name, value in overrides.items():
        if value:
            infer_cmd.extend(("-p", f"{name}:={value}"))
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
        control_urdf = (Path(control_urdf_arg).expanduser().resolve() if control_urdf_arg
                        else control_share / "assets/rm_75_kinematic.urdf")
        if not control_urdf.is_file():
            raise FileNotFoundError(f"RM75 control URDF does not exist: {control_urdf}")
        # 控制 URDF 与训练 URDF 文件不同，只要求关节链运动学等价，保证 Link7 目标含义一致。
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
            "-p", f"start_from_current_pose:={start_current}",
            # 显式写成浮点，避免 "0" 被 ROS 解析为整数参数。
            "-p", f"lookahead_s:={lookahead!r}",
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
        # 模型目录（pytorch 用 checkpoint，TensorRT 用 export_dir）、训练 URDF 与两份部署外参必填，
        # 路径与机器相关；backend 为空时沿用参数文件。
        "backend": "", "checkpoint": "", "export_dir": "", "urdf_path": "",
        "end_camera_extrinsic": "", "deploy_tcp_extrinsic": "",
        "deployment_config": str(share / "config/dp_umi_v4_deployment.yaml"),
        "config_file": str(share / "config/dp_infer_lerobot.yaml"),
        "inference_python": "", "controller_python": "",
        "control_urdf_path": "", "start_controller": "true", "dry_run": "true",
        "require_close_approval": "false",
        # 默认从当前静止关节位置起步，不要求回到 initial_pose.yaml（仍要求静止与夹爪张开）；
        # false 恢复必须回到起点。
        "start_from_current_pose": "true",
        "grasp_latch": "false", "policy_gripper_topic": "/fastumi/policy/gripper_state",
        "record": "false", "record_dir_name": "dp_infer_lerobot", "record_name": "rollout",
        "device": "cuda:0", "num_inference_steps": "16", "noise_scheduler": "DDIM",
        "image_topic": "/wrist_camera/image_raw/compressed",
        "image_type": "compressed", "joint_topic": "/joint_states",
        "gripper_topic": "/motion_control/gripper_state",
        "output_topic": "/fastumi/policy/action_sequence",
        "close_guard_topic": "/fastumi/policy/gripper_close_allowed",
        "set_inference_steps_service": "/fastumi/policy/set_inference_steps",
        "controller_reset_service": "/fastumi/rm75/placo/reset_episode",
        "max_start_displacement_m": "0.0", "max_start_rise_m": "0.0",
        "postprocessors": "",
        # 默认流水线：持续推理并只执行未过期的结果。TensorRT FP16 DDIM16 约 80 ms，在 8 个动作覆盖的
        # 233 ms 内；PyTorch 16 步约 650 ms 会全部过期，需改用 sequential_execution:=true 或 2～4 步。
        # true 为串行：整段动作执行完并稳定 execution_settle_s 秒后才进行下一次观测+推理。
        "sequential_execution": "false", "execution_settle_s": "0.3",
        # 串行模式每段执行的动作数，最多 8。
        "execution_action_steps": "8",
        # 低跟随提速：推理多发布 horizon 后续预测点（8～15），控制器按 lookahead_s 前瞻采样位姿。
        # 默认 8 / 0.0 保持原行为；建议 15 / 0.2 起逐步增大到 0.3。
        "published_action_steps": "8", "lookahead_s": "0.0",
    }
    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value=value)
          for name, value in defaults.items()),
        OpaqueFunction(function=_start),
    ])
