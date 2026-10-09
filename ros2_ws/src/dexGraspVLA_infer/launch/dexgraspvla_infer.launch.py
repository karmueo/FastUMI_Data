"""启动 Orin 感知/策略进程和独立关节执行器，缺省不发硬件指令。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration


CHECKPOINT = "dataset/h5dy_data/rm75_DexGraspVLA/2026.10.01/15.52_train_dexgraspvla_controller_grasp_rm75/checkpoints/latest.ckpt"


def start(context):
    """从已安装 share 寻找仓库，或接受显式 repo_root；不写机器绝对路径。"""
    share = Path(get_package_share_directory("dexgraspvla_infer"))
    value = lambda name: LaunchConfiguration(name).perform(context)
    configured_root = value("repo_root")
    root = Path(configured_root).expanduser().resolve() if configured_root else next(
        (parent for parent in (share, *share.parents) if (parent / "model/DexGraspVLA/controller").is_dir()), None)
    if root is None:
        raise ValueError("Cannot locate repository; pass repo_root")
    assets = Path(value("asset_root")).expanduser().resolve() if value("asset_root") else root / ".local/dexgraspvla"
    python = value("inference_python") or str(assets / "venv/bin/python")
    controller_python = value("controller_python") or str(root / "ros2_ws/.venv-numpy2/bin/python")
    checkpoint = Path(value("checkpoint")).expanduser().resolve() if value("checkpoint") else root / CHECKPOINT
    urdf = root / "model/DexGraspVLA/assets/rm75/rm_75.urdf"
    if value("urdf_path"):
        urdf = Path(value("urdf_path")).expanduser().resolve()
    control_share = Path(get_package_share_directory("fastumi_rm75"))
    control_urdf = control_share / "assets/rm_75_kinematic.urdf"
    for path in (Path(python), Path(controller_python), checkpoint, urdf, control_urdf):
        if not path.is_file():
            raise FileNotFoundError(path)
    # This independent contract checker does not import DP policy modules.
    from dexgraspvla_infer.urdf_contract import validate_kinematic_equivalence
    validate_kinematic_equivalence(urdf, control_urdf)
    if value("dry_run") not in ("true", "false") or value("start_controller") not in ("true", "false"):
        raise ValueError("dry_run and start_controller must be true/false")
    if value("mode") not in ("combined", "policy", "perception"):
        raise ValueError("invalid mode")
    config = value("config_file") or str(share / "config/dexgraspvla.yaml")
    cmd = [python, "-m", "dexgraspvla_infer.cli", "--mode", value("mode"), "--ros-args", "--params-file", config]
    parameters = {"checkpoint": checkpoint, "model_root": root / "model/DexGraspVLA", "asset_root": assets,
                  "urdf_path": urdf, "num_inference_steps": value("num_inference_steps"),
                  "image_topic": value("image_topic"), "image_type": value("image_type")}
    for key, item in parameters.items():
        cmd += ["-p", f"{key}:={item}"]
    gpu = ExecuteProcess(cmd=cmd, output="screen")
    processes = [gpu]
    if value("start_controller") == "true" and value("mode") != "perception":
        config = value("controller_config") or str(control_share / "config/rm75_joint_controller.yaml")
        processes.append(ExecuteProcess(cmd=[controller_python, "-m", "fastumi_rm75.rm75_joint_controller",
                         "--ros-args", "--params-file", config, "-p", f"dry_run:={value('dry_run')}",
                         "-p", f"urdf_path:={control_urdf}"], output="screen"))
    actions = []
    for process in processes:
        actions.append(RegisterEventHandler(OnProcessExit(target_action=process,
                       on_exit=[EmitEvent(event=Shutdown(reason="DexGraspVLA process exited"))])))
        actions.append(process)
    return actions


def generate_launch_description():
    """模型、配置与解释器均可通过 launch 参数覆盖。"""
    defaults = {"repo_root": "", "asset_root": "", "checkpoint": "", "urdf_path": "",
                "inference_python": "", "controller_python": "", "config_file": "", "controller_config": "",
                "mode": "combined", "start_controller": "true", "dry_run": "false", "num_inference_steps": "16",
                "image_topic": "/wrist_camera/image_decoded", "image_type": "raw"}
    return LaunchDescription([*(DeclareLaunchArgument(key, default_value=value) for key, value in defaults.items()),
                              OpaqueFunction(function=start)])
