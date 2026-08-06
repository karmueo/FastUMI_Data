"""测试 FastUMI 会话录制子进程的联动监管。"""

from pathlib import Path
import subprocess
from typing import List, Optional
from unittest.mock import Mock

import pytest
import yaml

from fastumi_data.session_recorder import (
    _TerminalKeyReader,
    _build_bag_command,
    _build_parser,
    _wait_for_service_readiness,
    _wait_for_recording_processes,
    main,
)


def test_reports_episode_manager_exit() -> None:
    """验证事件管理器退出会立即使整场录制失败。"""
    manager_process = Mock()
    manager_process.poll.return_value = 2
    bag_process = Mock()

    with pytest.raises(RuntimeError, match="episode_manager"):
        _wait_for_recording_processes(manager_process, bag_process)

    bag_process.wait.assert_not_called()


def test_returns_when_bag_finishes_cleanly() -> None:
    """验证录包进程正常结束时监管函数正常返回。"""
    manager_process = Mock()
    manager_process.poll.return_value = None
    bag_process = Mock()
    bag_process.wait.return_value = 0

    _wait_for_recording_processes(manager_process, bag_process)

    bag_process.wait.assert_called_once_with(timeout=0.2)


def test_retries_while_both_processes_are_running() -> None:
    """验证录包等待超时后会继续检查管理器状态。"""
    manager_process = Mock()
    manager_process.poll.side_effect = [None, 3]
    bag_process = Mock()
    bag_process.wait.side_effect = subprocess.TimeoutExpired("bag", 0.2)

    with pytest.raises(RuntimeError, match="episode_manager"):
        _wait_for_recording_processes(manager_process, bag_process)


class _KeyReader:
    """为录制监管测试提供顺序按键。"""

    def __init__(self, keys: List[str]) -> None:
        """保存待返回的按键序列。"""
        self._keys = iter(keys)

    def read_key(self, timeout_sec: float) -> Optional[str]:
        """忽略轮询时间并返回下一个模拟按键。"""
        del timeout_sec
        return next(self._keys, None)


def test_keyboard_keys_call_expected_episode_commands() -> None:
    """验证 s/e（大小写均可）映射到对应的 episode 服务命令。"""
    manager_process = Mock()
    manager_process.poll.return_value = None
    bag_process = Mock()
    bag_process.poll.side_effect = [None, None, None, None, None, 0]
    handled_commands: List[str] = []

    _wait_for_recording_processes(
        manager_process,
        bag_process,
        key_reader=_KeyReader(["s", "S", "x", "e", "E"]),
        command_handler=handled_commands.append,
    )

    assert handled_commands == ["start", "start", "stop", "stop"]


def test_keyboard_command_failure_does_not_stop_recording(capsys) -> None:
    """验证按键服务调用失败后仍继续监管录制进程。"""
    manager_process = Mock()
    manager_process.poll.return_value = None
    bag_process = Mock()
    bag_process.poll.side_effect = [None, 0]

    def raise_service_error(command: str) -> None:
        """模拟服务调用失败。"""
        raise RuntimeError(f"{command} 不可用")

    _wait_for_recording_processes(
        manager_process,
        bag_process,
        key_reader=_KeyReader(["s"]),
        command_handler=raise_service_error,
    )

    assert "控制失败" in capsys.readouterr().out


def test_terminal_key_reader_restores_terminal_settings(monkeypatch) -> None:
    """验证交互终端退出读取器后恢复原始终端设置。"""
    stream = Mock()
    stream.isatty.return_value = True
    stream.fileno.return_value = 7
    original_settings = ["original"]
    tcgetattr = Mock(return_value=original_settings)
    tcsetattr = Mock()
    setcbreak = Mock()
    monkeypatch.setattr(
        "fastumi_data.session_recorder.termios.tcgetattr", tcgetattr
    )
    monkeypatch.setattr(
        "fastumi_data.session_recorder.termios.tcsetattr", tcsetattr
    )
    monkeypatch.setattr(
        "fastumi_data.session_recorder.tty.setcbreak", setcbreak
    )

    with _TerminalKeyReader(stream):
        pass

    setcbreak.assert_called_once_with(7)
    tcsetattr.assert_called_once_with(
        7,
        pytest.importorskip("termios").TCSADRAIN,
        original_settings,
    )


def test_terminal_key_reader_sleeps_when_input_is_not_a_tty(
    monkeypatch,
) -> None:
    """验证非交互标准输入不会忙循环。"""
    stream = Mock()
    stream.isatty.return_value = False
    sleep = Mock()
    monkeypatch.setattr("fastumi_data.session_recorder.time.sleep", sleep)

    with _TerminalKeyReader(stream) as reader:
        assert reader.enabled is False
        assert reader.read_key(0.2) is None

    sleep.assert_called_once_with(0.2)


def test_bag_command_uses_fast_zstd_and_disables_keyboard_controls() -> None:
    """验证 rosbag 默认快速压缩，且不会与录制器竞争终端按键。"""
    command = _build_bag_command(
        Path("/tmp/session/raw"), ["/fastumi/episode/events"]
    )

    assert command == [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--storage-preset-profile",
        "zstd_fast",
        "--disable-keyboard-controls",
        "--output",
        "/tmp/session/raw/bag",
        "--topics",
        "/fastumi/episode/events",
    ]


def test_service_readiness_timeout_only_prints_warning(capsys) -> None:
    """验证初始服务发现超时不会终止录制会话。"""
    service_client = Mock()
    service_client.wait_until_ready.side_effect = RuntimeError(
        "服务暂不可用"
    )

    _wait_for_service_readiness(service_client)

    assert "服务暂不可用" in capsys.readouterr().out


def test_parser_defaults_to_portable_dataset_root() -> None:
    """验证未指定采集根目录时使用可移植的相对路径。"""
    arguments = _build_parser().parse_args(
        ["--task", "pick_place", "--extrinsic", "tracker_to_tcp.yaml"]
    )

    assert arguments.dataset_root == "dataset"


def test_parser_rejects_removed_unverified_calibration_option() -> None:
    """验证录制命令行不再接受跳过标定验收的开关。"""
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [
                "--task",
                "pick_place",
                "--extrinsic",
                "tracker_to_tcp.yaml",
                "--allow-unverified-calibration",
            ]
        )


def test_main_rejects_excessive_rmse_before_creating_session_or_processes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """验证正式录制在创建目录和进程前执行严格外参及残差门限。"""
    strict_loader = Mock(
        return_value=Mock(
            metadata={
                "translation_rmse_mm": 2.1,
                "rotation_rmse_deg": 0.1,
            }
        )
    )
    popen = Mock()
    monkeypatch.setattr(
        "fastumi_data.session_recorder.load_tracker_tcp_extrinsic",
        strict_loader,
    )
    monkeypatch.setattr(
        "fastumi_data.session_recorder.subprocess.Popen", popen
    )

    with pytest.raises(ValueError, match="2 mm、1°"):
        main(
            [
                "--task",
                "pick_place",
                "--session-id",
                "session",
                "--dataset-root",
                str(tmp_path),
                "--extrinsic",
                str(tmp_path / "tracker_to_tcp.yaml"),
            ]
        )

    strict_loader.assert_called_once_with(
        str(tmp_path / "tracker_to_tcp.yaml")
    )
    assert not (tmp_path / "pick_place" / "session").exists()
    popen.assert_not_called()


def test_main_writes_v2_manifest_without_calibration_verified(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """验证成功会话清单使用 v2 并省略已删除的验证状态。"""
    extrinsic_path = tmp_path / "tracker_to_tcp.yaml"
    processing_path = tmp_path / "processing.yaml"
    extrinsic_path.write_text("tracker_to_tcp: {}\n", encoding="utf-8")
    processing_path.write_text("topics: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        "fastumi_data.session_recorder.load_tracker_tcp_extrinsic",
        Mock(
            return_value=Mock(
                metadata={
                    "translation_rmse_mm": 2.0,
                    "rotation_rmse_deg": 1.0,
                }
            )
        ),
    )
    monkeypatch.setattr(
        "fastumi_data.session_recorder.subprocess.Popen",
        Mock(side_effect=[Mock(), Mock()]),
    )
    monkeypatch.setattr(
        "fastumi_data.session_recorder._wait_for_recording_processes",
        Mock(),
    )

    main(
        [
            "--task",
            "pick_place",
            "--session-id",
            "session",
            "--dataset-root",
            str(tmp_path),
            "--extrinsic",
            str(extrinsic_path),
            "--processing-config",
            str(processing_path),
        ]
    )

    manifest_path = tmp_path / "pick_place" / "session" / "session.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert "calibration_verified" not in manifest
