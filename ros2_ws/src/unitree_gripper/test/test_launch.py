"""验证厂商服务端只使用匹配的 CycloneDDS C/C++ 运行库。"""

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
        assert kwargs["env"]["LD_LIBRARY_PATH"].split(":")[0] == str(library_dir)
        errors = "undefined symbol: shm_set_data_state\n" if undefined_symbol else ""
        return subprocess.CompletedProcess(command, 0, output + errors, "")

    monkeypatch.setattr(module.subprocess, "run", fake_ldd)
    if valid:
        assert module._server_environment(Path("/tmp/dex1_server")) == {
            "LD_LIBRARY_PATH": f"{library_dir}:/opt/ros/humble/lib"
        }
    else:
        with pytest.raises(RuntimeError):
            module._server_environment(Path("/tmp/dex1_server"))
