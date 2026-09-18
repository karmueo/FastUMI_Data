"""验证厂商服务端只使用匹配的 CycloneDDS C/C++ 运行库。"""

import importlib.util
import subprocess
from pathlib import Path

import pytest
from launch import LaunchContext


LAUNCH_FILE = Path(__file__).resolve().parents[1] / "launch" / "gripper.launch.py"


def _load_launch_module():
    """按文件路径加载 ROS 2 launch 文件。"""
    spec = importlib.util.spec_from_file_location("unitree_gripper_launch", LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("c_version,cxx_version,undefined_symbol,valid", [
    ("0.10.2", "0.10.2", False, True),
    ("0.10.5", "0.10.2", False, False),
    ("0.10.2", "0.10.5", False, False),
    ("0.10.2", "0.10.2", True, False),
])
def test_server_rejects_mixed_cyclonedds_versions(
    tmp_path, monkeypatch, c_version, cxx_version, undefined_symbol, valid,
):
    """ROS 路径抢占 C 库、C++ 版本不符或符号缺失时拒绝启动。"""
    module = _load_launch_module()
    library_dir = tmp_path / "system" / "lib"
    library_dir.mkdir(parents=True)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/ros/jazzy/lib")

    c_library = library_dir / f"libddsc.so.{c_version}"
    cxx_library = library_dir / f"libddscxx.so.{cxx_version}"
    output = (
        f"libddsc.so.0 => {c_library} (0x1234)\n"
        f"libddscxx.so.0 => {cxx_library} (0x5678)\n"
    )

    def fake_ldd(command, **kwargs):
        if command == ["ldd", "/tmp/dex1_server"]:
            assert kwargs["env"]["LD_LIBRARY_PATH"].split(":")[0] == "/tmp/lib"
            return subprocess.CompletedProcess(command, 0, output, "")
        assert command == ["ldd", "-r", "/tmp/dex1_server"]
        assert kwargs["env"]["LD_LIBRARY_PATH"].split(":")[:2] == [
            "/tmp/lib", str(library_dir),
        ]
        errors = "undefined symbol: shm_set_data_state\n" if undefined_symbol else ""
        return subprocess.CompletedProcess(command, 0, output + errors, "")

    monkeypatch.setattr(module.subprocess, "run", fake_ldd)
    if valid:
        assert module._server_environment(Path("/tmp/dex1_server")) == {
            "LD_LIBRARY_PATH": f"/tmp/lib:{library_dir}:/opt/ros/jazzy/lib"
        }
    else:
        with pytest.raises(RuntimeError):
            module._server_environment(Path("/tmp/dex1_server"))


def test_architecture_vendor_selection(tmp_path, monkeypatch):
    """x86 使用本机构建产物，ARM64 使用随包文件。"""
    module = _load_launch_module()
    monkeypatch.delenv("UNITREE_GRIPPER_VENDOR_DIR", raising=False)
    with pytest.raises(RuntimeError, match="x86 服务端路径未设置"):
        module._vendor_paths(tmp_path, "x86_64")
    monkeypatch.setenv("UNITREE_GRIPPER_VENDOR_DIR", str(tmp_path / "local"))
    server, motor = module._vendor_paths(tmp_path, "x86_64")
    assert server == tmp_path / "local/dex1_1_gripper_server"
    assert motor == tmp_path / "local/lib/libUnitreeMotorSDK_Linux64.so"
    monkeypatch.delenv("UNITREE_GRIPPER_VENDOR_DIR")
    _, arm_motor = module._vendor_paths(tmp_path, "aarch64")
    assert arm_motor == tmp_path / "vendor/lib/libUnitreeMotorSDK_Arm64.so"
    with pytest.raises(RuntimeError, match="不支持的架构"):
        module._vendor_paths(tmp_path, "unknown")


def test_bundled_x86_cyclonedds_library(tmp_path, monkeypatch):
    """x86 服务端优先加载固定版本 SDK 附带的 C++ 库。"""
    module = _load_launch_module()
    vendor_lib = tmp_path / "lib"
    vendor_lib.mkdir()
    cxx_library = vendor_lib / "libddscxx.so"
    cxx_library.touch()
    output = (
        f"libddscxx.so.0 => {cxx_library} (0x1234)\n"
        f"libddsc.so.0 => {vendor_lib / 'libddsc.so.0.10.2'} (0x5678)\n"
    )

    def fake_ldd(command, **kwargs):
        assert kwargs["env"]["LD_LIBRARY_PATH"].split(":")[0] == str(vendor_lib)
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(module.subprocess, "run", fake_ldd)
    environment = module._server_environment(tmp_path / "dex1_1_gripper_server")
    assert environment["LD_LIBRARY_PATH"].split(":")[0] == str(vendor_lib)
    assert environment["LD_LIBRARY_PATH"].split(":").count(str(vendor_lib)) == 1


def test_node_uses_private_cyclonedds(tmp_path, monkeypatch):
    """Python 扩展优先加载准备脚本构建的 DDS C 库。"""
    module = _load_launch_module()
    monkeypatch.setenv("CYCLONEDDS_HOME", str(tmp_path))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/ros/jazzy/lib")
    with pytest.raises(RuntimeError, match="缺少夹爪 CycloneDDS"):
        module._node_environment()
    library_dir = tmp_path / "lib"
    library_dir.mkdir()
    (library_dir / "libddsc.so.0.10.2").touch()
    environment = module._node_environment()
    assert environment["LD_LIBRARY_PATH"] == f"{library_dir}:/opt/ros/jazzy/lib"
    assert environment["RMW_IMPLEMENTATION"] == "rmw_fastrtps_cpp"


def test_configured_network_and_invalid_file(tmp_path):
    """从 ROS 参数文件读取网卡，并在文件缺失或参数无效时拒绝启动。"""
    module = _load_launch_module()
    config = tmp_path / "gripper.yaml"
    with pytest.raises(RuntimeError, match="配置文件不存在"):
        module._configured_network(config)
    config.write_text("dex1_gripper_node:\n  ros__parameters:\n    network_interface: test0\n")
    assert module._configured_network(config) == "test0"
    config.write_text("dex1_gripper_node:\n  ros__parameters:\n    network_interface: ''\n")
    with pytest.raises(RuntimeError, match="network_interface"):
        module._configured_network(config)
    config.write_text("dex1_gripper_node: [invalid\n")
    with pytest.raises(RuntimeError, match="无法读取"):
        module._configured_network(config)


@pytest.mark.parametrize("override,expected", [("", "yaml0"), ("cli0", "cli0")])
def test_launch_uses_one_network_and_custom_topics(
    tmp_path, monkeypatch, override, expected,
):
    """YAML 和 CLI 网卡依优先级传给两个进程，节点保留话题配置。"""
    module = _load_launch_module()
    config = tmp_path / "custom.yaml"
    config.write_text(
        "dex1_gripper_node:\n  ros__parameters:\n"
        "    network_interface: yaml0\n"
        "    cmd_topic_name: /custom/gripper_command\n"
        "    state_topic_name: /custom/gripper_state\n"
    )
    server = tmp_path / "dex1_1_gripper_server"
    server.touch(mode=0o755)
    motor = tmp_path / "libUnitreeMotorSDK_Linux64.so"
    motor.touch()
    monkeypatch.setattr(module, "get_package_share_directory", lambda _: str(tmp_path))
    monkeypatch.setattr(module, "_vendor_paths", lambda *_: (server, motor))
    monkeypatch.setattr(module, "_server_environment", lambda _: {})
    monkeypatch.setattr(module, "_node_environment", lambda: {})
    monkeypatch.setattr(module.socket, "if_nametoindex", lambda _: 1)

    class CapturedAction:
        """只记录启动参数，不创建进程。"""

        def __init__(self, **kwargs):
            """保存启动参数以便验证传递结果。"""
            self.kwargs = kwargs

    monkeypatch.setattr(module, "ExecuteProcess", CapturedAction)
    monkeypatch.setattr(module, "Node", CapturedAction)
    monkeypatch.setattr(module, "OnProcessExit", lambda **kwargs: kwargs)
    monkeypatch.setattr(module, "RegisterEventHandler", lambda handler: handler)
    context = LaunchContext()
    context.launch_configurations["config_file"] = str(config)
    context.launch_configurations["network_interface"] = override
    actions = module._launch_gripper(context)
    assert actions[0].kwargs["cmd"][-1] == expected
    assert actions[1].kwargs["parameters"] == [
        str(config), {"network_interface": expected},
    ]


def test_launch_rejects_missing_network_interface(tmp_path, monkeypatch):
    """网卡不存在时在创建服务端进程之前停止。"""
    module = _load_launch_module()
    config = tmp_path / "gripper.yaml"
    config.write_text("dex1_gripper_node:\n  ros__parameters:\n    network_interface: absent0\n")
    monkeypatch.setattr(module, "get_package_share_directory", lambda _: str(tmp_path))

    def missing_interface(_):
        """模拟系统中不存在所配置的网卡。"""
        raise OSError()

    monkeypatch.setattr(module.socket, "if_nametoindex", missing_interface)
    context = LaunchContext()
    context.launch_configurations["config_file"] = str(config)
    context.launch_configurations["network_interface"] = ""
    with pytest.raises(RuntimeError, match=f"网卡不存在: absent0；配置文件: {config}"):
        module._launch_gripper(context)
