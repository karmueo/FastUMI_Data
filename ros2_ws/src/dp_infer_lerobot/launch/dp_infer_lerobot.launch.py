"""启动 LeRobot TCP 扩散策略推理节点；启动前校验模型指纹与 URDF，节点按生效参数校验部署外参。

``backend`` 为 ``pytorch`` 时校验 ``checkpoint``；为 ``tensorrt_fp32``/``tensorrt_fp16`` 时校验
``export_dir`` 模型包（manifest 摘要、部署档绑定与该精度的验证报告），不需要 checkpoint。
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration

# 可从命令行覆盖、原样转交节点的参数；空字符串表示沿用参数文件。
OVERRIDE_NAMES = (
    "backend", "device", "noise_scheduler", "num_inference_steps", "image_topic", "joint_topic", "gripper_topic",
    "output_topic", "reset_service", "postprocessors",
)


def _candidate_python(package_share, relative):
    """在已安装工作区的祖先目录中寻找预置的推理环境解释器。"""
    for base in (package_share, *package_share.parents):
        candidate = base / relative
        if candidate.is_file():
            # venv/bin/python 通常是系统 Python 的符号链接；保留 venv 路径。
            return candidate.absolute()
    raise FileNotFoundError(f"Cannot find {relative}; pass inference_python explicitly")


def _path_argument(context, name):
    """把 launch 参数解析为已存在文件的绝对路径。"""
    value = LaunchConfiguration(name).perform(context).strip()
    if not value:
        raise ValueError(f"{name} must name an existing file")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _file_parameter(config_file, name, default):
    """读取参数文件中节点的某个 ROS 参数，缺失时返回默认值。"""
    import yaml

    raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    return str(raw.get("dp_infer_lerobot", {}).get("ros__parameters", {}).get(name, default))


def _start(context):
    """校验部署输入后创建推理进程；进程退出时结束整个 launch。"""
    share = Path(get_package_share_directory("dp_infer_lerobot"))
    deployment_config = _path_argument(context, "deployment_config")
    urdf_path = _path_argument(context, "urdf_path")
    config_file = _path_argument(context, "config_file")
    end_camera = _path_argument(context, "end_camera_extrinsic")
    deploy_tcp = _path_argument(context, "deploy_tcp_extrinsic")
    python_value = LaunchConfiguration("inference_python").perform(context).strip()
    inference_python = (Path(python_value).expanduser().absolute() if python_value
                        else _candidate_python(share, Path(".venv-dp-lerobot/bin/python")))
    if not inference_python.is_file():
        raise FileNotFoundError(f"inference_python does not exist: {inference_python}")

    from dp_infer_lerobot.config import DiffusionConfig, validate_backend
    from dp_infer_lerobot.deployment import load_profile, validate_checkpoint, validate_export, validate_urdf

    overrides = {name: LaunchConfiguration(name).perform(context).strip() for name in OVERRIDE_NAMES}
    profile = load_profile(deployment_config)
    # 未覆盖 backend 时取参数文件中的值，与节点实际生效的后端一致。
    backend = overrides["backend"] or _file_parameter(config_file, "backend", "pytorch")
    precision = validate_backend(backend)
    model_key = "checkpoint" if precision is None else "export_dir"
    value = LaunchConfiguration(model_key).perform(context).strip()
    if not value or not Path(value).expanduser().is_dir():
        raise ValueError(f"{model_key} must name an existing directory for backend {backend}")
    model_dir = Path(value).expanduser().resolve()
    if precision is None:
        validate_checkpoint(model_dir, profile)
        assets = model_dir
    else:
        validate_export(model_dir, profile, precision)
        assets = model_dir / "assets"
    validate_urdf(urdf_path, profile)
    # 外参轴向校验由节点执行，使用参数文件中实际生效的 max_tcp_axis_deg。
    DiffusionConfig.from_file(assets / "config.json", overrides["noise_scheduler"] or None,
                              int(overrides["num_inference_steps"] or 0) or None,
                              profile["contract"].get("pose_frame", "current_tcp"))
    command = [
        str(inference_python), "-m", "dp_infer_lerobot.node", "--ros-args",
        "--params-file", str(config_file),
        "-p", f"{model_key}:={model_dir}",
        "-p", f"deployment_config:={deployment_config}",
        "-p", f"urdf_path:={urdf_path}",
        "-p", f"end_camera_extrinsic:={end_camera}",
        "-p", f"deploy_tcp_extrinsic:={deploy_tcp}",
    ]
    for name, override in overrides.items():
        if override:
            command.extend(("-p", f"{name}:={override}"))
    inference = ExecuteProcess(cmd=command, output="screen")
    return [
        RegisterEventHandler(OnProcessExit(
            target_action=inference,
            on_exit=[EmitEvent(event=Shutdown(reason="DP inference process exited"))],
        )),
        inference,
    ]


def generate_launch_description():
    """暴露部署所需参数；话题默认值与 dp_infer 一致。"""
    share = Path(get_package_share_directory("dp_infer_lerobot"))
    defaults = {
        "checkpoint": "", "export_dir": "",
        "deployment_config": str(share / "config/dp_umi_v4_deployment.yaml"),
        "config_file": str(share / "config/dp_infer_lerobot.yaml"),
        "urdf_path": "", "end_camera_extrinsic": "", "deploy_tcp_extrinsic": "",
        "inference_python": "",
        **{name: "" for name in OVERRIDE_NAMES},
    }
    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()),
        OpaqueFunction(function=_start),
    ])
