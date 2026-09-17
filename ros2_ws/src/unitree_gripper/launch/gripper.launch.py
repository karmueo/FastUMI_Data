"""同时管理 Unitree 厂商服务端和 ROS 2 控制节点。"""

import os
import subprocess
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _server_environment(server_path: Path) -> dict:
    """优先加载与 C++ 库配套的 DDS C 库，并验证运行时符号。"""
    discovery = subprocess.run(
        ["ldd", str(server_path)], capture_output=True, text=True, check=False,
    )
    cxx_library = _resolved_library(discovery.stdout, "libddscxx.so.0")
    if cxx_library is None or cxx_library.name != "libddscxx.so.0.10.2":
        raise RuntimeError("服务端需要 libddscxx.so.0.10.2")
    library_dir = cxx_library.parent
    search_paths = [str(library_dir), os.environ.get("LD_LIBRARY_PATH", "")]
    library_path = ":".join(path for path in search_paths if path)
    environment = dict(os.environ, LD_LIBRARY_PATH=library_path)
    result = subprocess.run(
        ["ldd", "-r", str(server_path)], env=environment,
        capture_output=True, text=True, check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0 or "not found" in output or "undefined symbol" in output:
        raise RuntimeError(f"服务端动态库解析失败:\n{result.stdout}{result.stderr}")

    c_library = _resolved_library(result.stdout, "libddsc.so.0")
    if c_library is None or c_library.name != "libddsc.so.0.10.2":
        raise RuntimeError("服务端需要 libddsc.so.0.10.2")
    if _resolved_library(result.stdout, "libddscxx.so.0") != cxx_library:
        raise RuntimeError("服务端加载的 libddscxx 与预检结果不一致")
    return {"LD_LIBRARY_PATH": library_path}


def _resolved_library(ldd_output: str, soname: str):
    """从 ldd 输出读取共享库的实际文件路径。"""
    for line in ldd_output.splitlines():
        name, separator, target = line.strip().partition(" => ")
        if separator and name == soname and target != "not found":
            return Path(target.split(" (", 1)[0]).resolve()
    return None


def generate_launch_description() -> LaunchDescription:
    """任一进程退出时停止整个夹爪服务。"""
    share = Path(get_package_share_directory("unitree_gripper"))
    server_path = share / "vendor" / "dex1_1_gripper_server"
    sdk_library = share / "vendor" / "lib" / "libUnitreeMotorSDK_Arm64.so"
    config_path = share / "config" / "gripper.yaml"
    for path in (server_path, sdk_library, config_path):
        if not path.is_file():
            raise RuntimeError(f"夹爪安装文件缺失: {path}；请重新构建 unitree_gripper")
    if not server_path.stat().st_mode & 0o111:
        raise RuntimeError(f"厂商服务端没有执行权限: {server_path}")

    network = LaunchConfiguration("network_interface")
    server = ExecuteProcess(
        cmd=[str(server_path), "-n", network],
        output="screen",
        additional_env=_server_environment(server_path),
    )
    node = Node(
        package="unitree_gripper",
        executable="gripper_node",
        name="dex1_gripper_node",
        parameters=[str(config_path), {"network_interface": network}],
        output="screen",
        additional_env={"RMW_IMPLEMENTATION": "rmw_fastrtps_cpp"},
    )
    return LaunchDescription([
        DeclareLaunchArgument("network_interface", default_value="wlP1p1s0"),
        server,
        node,
        RegisterEventHandler(OnProcessExit(
            target_action=server,
            on_exit=[EmitEvent(event=Shutdown(reason="Unitree 服务端已退出"))],
        )),
        RegisterEventHandler(OnProcessExit(
            target_action=node,
            on_exit=[EmitEvent(event=Shutdown(reason="夹爪控制节点已退出"))],
        )),
    ])
