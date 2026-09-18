"""统一启动 RM75、相机、夹爪、预测、Tracker 和遥操。"""

import math
import shlex
from pathlib import Path

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration

from tracker_teleoperated.bringup_checks import check_for_existing_publishers


def _numpy1_command(workspace, package, launch_file, arguments=()):
    """在独立 Bash 进程中加载 NumPy 1 环境并执行现有 launch。"""
    ros_setup = Path("/opt/ros/jazzy/setup.bash")
    venv_activate = workspace / ".venv-numpy1/bin/activate"
    workspace_setup = workspace / "install/setup.bash"
    command = " ".join(
        shlex.quote(part)
        for part in ("ros2", "launch", package, launch_file, *arguments)
    )
    script = (
        "set -e; "
        f"source {shlex.quote(str(ros_setup))}; "
        f"source {shlex.quote(str(venv_activate))}; "
        f"source {shlex.quote(str(workspace_setup))}; "
        f"exec {command}"
    )
    return ["bash", "-c", script]


def _shutdown_on_exit(action, name):
    """让受管理组件退出时结束整个一键启动。"""
    return RegisterEventHandler(
        OnProcessExit(
            target_action=action,
            on_exit=[EmitEvent(event=Shutdown(reason=f"{name} 已退出"))],
        )
    )


def _readiness_exit_actions(returncode, teleop):
    """根据就绪检查退出码决定启动遥操或关闭整组进程。"""
    if returncode == 0:
        return [teleop]
    return [EmitEvent(event=Shutdown(reason="前置输入检查失败"))]


def _start_components(context):
    """完成冲突预检，再启动前置设备和消息就绪检查。"""
    # 包前缀位于工作区 install/<package>；允许命令行覆盖工作区位置。
    workspace = Path(LaunchConfiguration("workspace_root").perform(context))
    workspace = workspace.expanduser().resolve()
    required = (
        workspace / ".venv-numpy1/bin/activate",
        workspace / "install/setup.bash",
        workspace / "src/unitree_gripper/run_gripper.sh",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("一键启动所需文件不存在：" + ", ".join(missing))

    start_arm = LaunchConfiguration("start_arm").perform(context) == "true"
    move_to_initial_pose = (
        LaunchConfiguration("move_to_initial_pose").perform(context) == "true"
    )
    timeout = LaunchConfiguration("startup_timeout_s").perform(context)
    try:
        timeout_seconds = float(timeout)
    except ValueError as error:
        raise RuntimeError("startup_timeout_s 必须为正的有限秒数") from error
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise RuntimeError("startup_timeout_s 必须为正的有限秒数")
    # 检查在任何设备进程创建前执行；外部机械臂模式允许已有 /joint_states。
    check_for_existing_publishers(start_arm)

    components = []
    if start_arm:
        pose_arg = "move_to_initial_pose:=" + str(move_to_initial_pose).lower()
        components.append((
            "RM75 机械臂",
            ExecuteProcess(
                cmd=_numpy1_command(
                    workspace, "rm_bringup", "rm_75_bringup.launch.py",
                    (pose_arg,),
                ),
                cwd=str(workspace), output="screen",
            ),
        ))
    components.extend((
        (
            "USB 相机",
            ExecuteProcess(
                cmd=_numpy1_command(
                    workspace, "fastumi_usb_camera", "usb_camera.launch.py"
                ),
                cwd=str(workspace), output="screen",
            ),
        ),
        (
            "Unitree 夹爪",
            ExecuteProcess(
                cmd=[str(workspace / "src/unitree_gripper/run_gripper.sh")],
                cwd=str(workspace), output="screen",
            ),
        ),
        (
            "VIVE Tracker",
            ExecuteProcess(
                cmd=_numpy1_command(
                    workspace, "vive_tracker", "vive_tracker.launch.py",
                    ("use_rviz:=false",),
                ),
                cwd=str(workspace), output="screen",
            ),
        ),
        (
            "夹爪开合度预测",
            ExecuteProcess(
                cmd=_numpy1_command(
                    workspace, "fastumi_gripper_estimator",
                    "gripper_openness.launch.py",
                ),
                cwd=str(workspace), output="screen",
            ),
        ),
    ))

    readiness_arguments = ["--timeout", timeout]
    if start_arm and move_to_initial_pose:
        readiness_arguments.append("--require-initial-pose")
    readiness = ExecuteProcess(
        cmd=[
            "ros2", "run", "tracker_teleoperated", "tracker_teleop_wait_ready",
            *readiness_arguments,
        ],
        cwd=str(workspace), output="screen",
    )
    teleop = ExecuteProcess(
        cmd=[
            "ros2", "launch", "tracker_teleoperated",
            "tracker_teleoperated.launch.py",
            "config_file:=" + LaunchConfiguration("config_file").perform(
                context
            ),
            "use_keyboard:=" + LaunchConfiguration("use_keyboard").perform(
                context
            ),
            "keyboard_prefix:=" + LaunchConfiguration(
                "keyboard_prefix"
            ).perform(context),
        ],
        cwd=str(workspace), output="screen",
    )

    def on_readiness_exit(event, _context):
        """仅在前置输入检查成功后启动遥操。"""
        return _readiness_exit_actions(event.returncode, teleop)

    handlers = [_shutdown_on_exit(action, name) for name, action in components]
    handlers.extend((
        RegisterEventHandler(
            OnProcessExit(target_action=readiness, on_exit=on_readiness_exit)
        ),
        _shutdown_on_exit(teleop, "遥操"),
    ))
    return handlers + [action for _name, action in components] + [readiness]


def generate_launch_description():
    """声明一键启动参数和设备编排入口。"""
    # 当前工作区使用 colcon isolated install 布局。
    package_prefix = Path(get_package_prefix("tracker_teleoperated"))
    default_workspace = package_prefix.parent.parent
    package_share = package_prefix / "share/tracker_teleoperated"
    default_config = package_share / "config/tracker_teleoperated.yaml"
    return LaunchDescription([
        DeclareLaunchArgument(
            "workspace_root", default_value=str(default_workspace),
            description="包含 src、install 和 .venv-numpy1 的工作区路径。",
        ),
        DeclareLaunchArgument(
            "start_arm", default_value="true", choices=["true", "false"]
        ),
        DeclareLaunchArgument(
            "move_to_initial_pose", default_value="true",
            choices=["true", "false"],
        ),
        DeclareLaunchArgument("startup_timeout_s", default_value="180"),
        DeclareLaunchArgument(
            "config_file", default_value=str(default_config)
        ),
        DeclareLaunchArgument(
            "use_keyboard", default_value="true", choices=["true", "false"]
        ),
        DeclareLaunchArgument(
            "keyboard_prefix", default_value="xterm -fa Monospace -fs 20 -e"
        ),
        OpaqueFunction(function=_start_components),
    ])
