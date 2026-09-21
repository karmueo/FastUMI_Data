"""同时管理 Unitree 厂商服务端和 ROS 2 控制节点。"""

import os
import socket
import subprocess
from pathlib import Path

import yaml
from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
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


def _cyclonedds_home() -> Path:
    """解析启动脚本覆盖或当前工作区的夹爪私有 CycloneDDS。"""
    configured = os.environ.get("CYCLONEDDS_HOME")
    if configured:
        return Path(configured)
    prefix = Path(get_package_prefix("unitree_gripper"))
    return prefix.parents[1] / ".deps/unitree_gripper/install/cyclonedds"


def _server_environment(server_path: Path) -> dict:
    """使用厂商服务端自身可解析的 DDS 组合，并验证全部运行时符号。"""
    vendor_lib = str(server_path.parent / "lib")
    initial_paths = [vendor_lib, os.environ.get("LD_LIBRARY_PATH", "")]
    initial_library_path = ":".join(path for path in initial_paths if path)
    initial_environment = dict(os.environ, LD_LIBRARY_PATH=initial_library_path)
    discovery = subprocess.run(
        ["ldd", str(server_path)], env=initial_environment,
        capture_output=True, text=True, check=False,
    )
    cxx_library = _resolved_library(discovery.stdout, "libddscxx.so.0")
    if cxx_library is None or cxx_library.name != "libddscxx.so.0.10.2":
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
    if c_library is None:
        raise RuntimeError("服务端未解析到 libddsc.so.0")
    if _resolved_library(result.stdout, "libddscxx.so.0") != cxx_library:
        raise RuntimeError("服务端加载的 libddscxx 与预检结果不一致")
    return {"LD_LIBRARY_PATH": library_path}


def _node_environment() -> dict:
    """让 Python DDS 扩展优先使用夹爪私有 CycloneDDS C 库。"""
    home = _cyclonedds_home()
    library_dir = home / "lib"
    if not (library_dir / "libddsc.so.0.10.2").is_file():
        raise RuntimeError(f"缺少夹爪 CycloneDDS 0.10.2 动态库: {library_dir}")
    search_paths = [str(library_dir), os.environ.get("LD_LIBRARY_PATH", "")]
    return {
        "CYCLONEDDS_HOME": str(home),
        "LD_LIBRARY_PATH": ":".join(path for path in search_paths if path),
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
    }


def _resolved_library(ldd_output: str, soname: str):
    """从 ldd 输出读取共享库的实际文件路径。"""
    for line in ldd_output.splitlines():
        name, separator, target = line.strip().partition(" => ")
        if separator and name == soname and target != "not found":
            return Path(target.split(" (", 1)[0]).resolve()
    return None


def _configured_network(config_path: Path) -> str:
    """从 ROS 参数文件读取服务端和节点共用的网卡。"""
    if not config_path.is_file():
        raise RuntimeError(f"夹爪配置文件不存在: {config_path}")
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise RuntimeError(f"无法读取夹爪配置文件 {config_path}: {error}") from error
    node_config = config.get("dex1_gripper_node") if isinstance(config, dict) else None
    parameters = node_config.get("ros__parameters") if isinstance(node_config, dict) else None
    network = parameters.get("network_interface") if isinstance(parameters, dict) else None
    if not isinstance(network, str) or not network.strip():
        raise RuntimeError(f"配置文件缺少有效的 network_interface: {config_path}")
    return network.strip()


def _validate_serial_ports(dev_root: Path = Path("/dev")) -> None:
    """确认厂商服务端可访问至少一个支持的本地串口。"""
    ports = sorted({
        path
        for pattern in ("ttyUSB*", "ttyCH343USB*", "ttyACM*")
        for path in dev_root.glob(pattern)
        if path.is_char_device()
    })
    if not ports:
        raise RuntimeError("未发现夹爪串口（ttyUSB*、ttyCH343USB*、ttyACM*）")
    accessible = [path for path in ports if os.access(path, os.R_OK | os.W_OK)]
    if not accessible:
        raise RuntimeError(
            "当前用户无权读写任何夹爪串口: "
            + ", ".join(str(path) for path in ports)
        )


def _launch_gripper(context):
    """解析配置与可选网卡覆盖，创建 ARM64 服务端和 ROS 节点。"""
    share = Path(get_package_share_directory("unitree_gripper"))
    server_path = share / "vendor" / "dex1_1_gripper_server"
    sdk_library = share / "vendor" / "lib" / "libUnitreeMotorSDK_Arm64.so"
    config_path = Path(
        LaunchConfiguration("config_file").perform(context)
    ).expanduser().resolve()
    network = LaunchConfiguration("network_interface").perform(context).strip()
    network = network or _configured_network(config_path)
    _validate_serial_ports()
    try:
        socket.if_nametoindex(network)
    except OSError as error:
        raise RuntimeError(f"网卡不存在: {network}；配置文件: {config_path}") from error
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
    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file", default_value=str(share / "config" / "gripper.yaml")
        ),
        DeclareLaunchArgument("network_interface", default_value=""),
        OpaqueFunction(function=_launch_gripper),
    ])
