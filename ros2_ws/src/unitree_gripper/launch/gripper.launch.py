"""同时管理 Unitree 厂商服务端和 ROS 2 控制节点。"""

import os
import platform
import socket
import subprocess
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _server_environment(server_path: Path) -> dict:
    """优先加载与 C++ 库配套的 DDS C 库，并验证运行时符号。"""
    vendor_lib = str(server_path.parent / "lib")
    search_paths = [vendor_lib, os.environ.get("LD_LIBRARY_PATH", "")]
    library_path = ":".join(path for path in search_paths if path)
    environment = dict(os.environ, LD_LIBRARY_PATH=library_path)
    discovery = subprocess.run(
        ["ldd", str(server_path)], env=environment,
        capture_output=True, text=True, check=False,
    )
    cxx_library = _resolved_library(discovery.stdout, "libddscxx.so.0")
    bundled_cxx = Path(vendor_lib) / "libddscxx.so"
    if bundled_cxx.is_file():
        if cxx_library != bundled_cxx.resolve():
            raise RuntimeError("服务端未加载随包的 CycloneDDS C++ 库")
    elif cxx_library is None or cxx_library.name != "libddscxx.so.0.10.2":
        raise RuntimeError("服务端需要 libddscxx.so.0.10.2")
    library_dir = cxx_library.parent
    search_paths = [vendor_lib]
    if str(library_dir) != vendor_lib:
        search_paths.append(str(library_dir))
    search_paths.append(os.environ.get("LD_LIBRARY_PATH", ""))
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


def _node_environment() -> dict:
    """让 Python DDS 扩展与 CYCLONEDDS_HOME 使用同一份 C 库。"""
    home = os.environ.get("CYCLONEDDS_HOME")
    if not home:
        raise RuntimeError("未设置 CYCLONEDDS_HOME；请通过 run_gripper.sh 启动")
    library_dir = Path(home) / "lib"
    if not (library_dir / "libddsc.so.0.10.2").is_file():
        raise RuntimeError(f"缺少夹爪 CycloneDDS 0.10.2 动态库: {library_dir}")
    search_paths = [str(library_dir), os.environ.get("LD_LIBRARY_PATH", "")]
    return {"LD_LIBRARY_PATH": ":".join(path for path in search_paths if path),
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp"}


def _resolved_library(ldd_output: str, soname: str):
    """从 ldd 输出读取共享库的实际文件路径。"""
    for line in ldd_output.splitlines():
        name, separator, target = line.strip().partition(" => ")
        if separator and name == soname and target != "not found":
            return Path(target.split(" (", 1)[0]).resolve()
    return None


def _vendor_paths(share: Path, architecture: str):
    """按架构定位服务端和电机库，x86 构建产物由启动脚本提供。"""
    if architecture not in ("x86_64", "aarch64"):
        raise RuntimeError(f"不支持的架构: {architecture}")
    vendor_override = os.environ.get("UNITREE_GRIPPER_VENDOR_DIR")
    if architecture == "x86_64" and not vendor_override:
        raise RuntimeError("x86 服务端路径未设置；请通过 run_gripper.sh 启动")
    vendor = Path(vendor_override) if vendor_override else share / "vendor"
    motor_name = (
        "libUnitreeMotorSDK_Linux64.so" if architecture == "x86_64"
        else "libUnitreeMotorSDK_Arm64.so"
    )
    return vendor / "dex1_1_gripper_server", vendor / "lib" / motor_name


def _configured_network(config_path: Path) -> str:
    """从 ROS 参数 YAML 读取供服务端和节点共用的网卡。"""
    if not config_path.is_file():
        raise RuntimeError(f"夹爪配置文件不存在: {config_path}")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError(f"无法读取夹爪配置文件 {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError(f"夹爪配置文件格式无效: {config_path}")
    node_config = config.get("dex1_gripper_node")
    parameters = node_config.get("ros__parameters") if isinstance(node_config, dict) else None
    network = parameters.get("network_interface") if isinstance(parameters, dict) else None
    if not isinstance(network, str) or not network.strip():
        raise RuntimeError(f"配置文件缺少有效的 network_interface: {config_path}")
    return network.strip()


def _launch_gripper(context):
    """解析配置和命令行覆盖，再同时创建服务端与 ROS 节点。"""
    share = Path(get_package_share_directory("unitree_gripper"))
    config_path = Path(LaunchConfiguration("config_file").perform(context)).expanduser().resolve()
    configured_network = _configured_network(config_path)
    override = LaunchConfiguration("network_interface").perform(context).strip()
    network = override or configured_network
    try:
        socket.if_nametoindex(network)
    except OSError as exc:
        raise RuntimeError(f"网卡不存在: {network}；配置文件: {config_path}") from exc

    server_path, sdk_library = _vendor_paths(share, platform.machine())
    for path in (server_path, sdk_library):
        if not path.is_file():
            raise RuntimeError(f"夹爪安装文件缺失: {path}；请重新构建 unitree_gripper")
    if not server_path.stat().st_mode & 0o111:
        raise RuntimeError(f"厂商服务端没有执行权限: {server_path}")

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
        additional_env=_node_environment(),
    )
    return [
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
    ]


def generate_launch_description() -> LaunchDescription:
    """声明配置文件和可选网卡覆盖，统一管理夹爪进程。"""
    share = Path(get_package_share_directory("unitree_gripper"))
    default_config = share / "config" / "gripper.yaml"
    return LaunchDescription([
        DeclareLaunchArgument("config_file", default_value=str(default_config)),
        DeclareLaunchArgument("network_interface", default_value=""),
        OpaqueFunction(function=_launch_gripper),
    ])
