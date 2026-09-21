"""验证厂商服务端 DDS 运行库解析及夹爪配置读取。"""

import importlib.util
import subprocess
from pathlib import Path

import pytest


LAUNCH_FILE = Path(__file__).resolve().parents[1] / "launch" / "gripper.launch.py"


def _load_launch_module():
    """按文件路径加载 ROS 2 launch 文件。"""
    spec = importlib.util.spec_from_file_location("unitree_gripper_launch", LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("c_version,cxx_version,undefined_symbol,valid", [
    ("0.10.2", "0.10.2", False, True),
    ("0.10.5", "0.10.2", False, True),
    ("0.10.2", "0.10.5", False, False),
    ("0.10.2", "0.10.2", True, False),
])
def test_server_validates_cyclonedds_runtime(
    tmp_path, monkeypatch, c_version, cxx_version, undefined_symbol, valid,
):
    """允许 Humble 的 C 库，并拒绝错误 C++ 版本或未解析符号。"""
    module = _load_launch_module()
    library_dir = tmp_path / "system" / "lib"
    library_dir.mkdir(parents=True)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/ros/humble/lib")

    c_library = library_dir / f"libddsc.so.{c_version}"
    cxx_library = library_dir / f"libddscxx.so.{cxx_version}"
    output = (
        f"libddsc.so.0 => {c_library} (0x1234)\n"
        f"libddscxx.so.0 => {cxx_library} (0x5678)\n"
    )

    def fake_ldd(command, **kwargs):
        if command == ["ldd", "/tmp/dex1_server"]:
            return subprocess.CompletedProcess(command, 0, output, "")
        assert command == ["ldd", "-r", "/tmp/dex1_server"]
        assert str(library_dir) in kwargs["env"]["LD_LIBRARY_PATH"].split(":")
        errors = "undefined symbol: shm_set_data_state\n" if undefined_symbol else ""
        return subprocess.CompletedProcess(command, 0, output + errors, "")

    monkeypatch.setattr(module.subprocess, "run", fake_ldd)
    if valid:
        paths = module._server_environment(Path("/tmp/dex1_server"))[
            "LD_LIBRARY_PATH"
        ].split(":")
        assert paths == ["/tmp/lib", str(library_dir), "/opt/ros/humble/lib"]
    else:
        with pytest.raises(RuntimeError):
            module._server_environment(Path("/tmp/dex1_server"))


def test_configured_network_reads_ros_parameter_file(tmp_path):
    """服务端与节点从同一个参数文件读取网卡。"""
    module = _load_launch_module()
    config = tmp_path / "gripper.yaml"
    config.write_text(
        "dex1_gripper_node:\n  ros__parameters:\n"
        "    network_interface: eno1\n",
        encoding="utf-8",
    )
    assert module._configured_network(config) == "eno1"


@pytest.mark.parametrize("content", ["{}", "dex1_gripper_node: {}"])
def test_configured_network_rejects_missing_value(tmp_path, content):
    """配置缺失网卡时在启动厂商进程前失败。"""
    module = _load_launch_module()
    config = tmp_path / "gripper.yaml"
    config.write_text(content, encoding="utf-8")
    with pytest.raises(RuntimeError, match="network_interface"):
        module._configured_network(config)


def test_serial_validation_requires_accessible_character_device(tmp_path, monkeypatch):
    """启动前拒绝没有设备或当前用户不可读写的串口。"""
    module = _load_launch_module()
    with pytest.raises(RuntimeError, match="未发现"):
        module._validate_serial_ports(tmp_path)
    port = tmp_path / "ttyUSB0"
    port.touch()
    monkeypatch.setattr(Path, "is_char_device", lambda self: self == port)
    monkeypatch.setattr(module.os, "access", lambda path, mode: False)
    with pytest.raises(RuntimeError, match="无权读写"):
        module._validate_serial_ports(tmp_path)
    monkeypatch.setattr(module.os, "access", lambda path, mode: True)
    module._validate_serial_ports(tmp_path)


def test_serial_validation_accepts_one_accessible_port(tmp_path, monkeypatch):
    """无关串口不可访问时，不应阻塞可读写的夹爪串口。"""
    module = _load_launch_module()
    gripper = tmp_path / "ttyUSB0"
    unrelated = tmp_path / "ttyACM0"
    gripper.touch()
    unrelated.touch()
    monkeypatch.setattr(
        Path, "is_char_device", lambda self: self in (gripper, unrelated)
    )
    monkeypatch.setattr(
        module.os, "access", lambda path, mode: path == gripper
    )
    module._validate_serial_ports(tmp_path)
