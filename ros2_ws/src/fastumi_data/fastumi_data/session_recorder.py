"""创建 FastUMI 会话目录、快照配置并连续录制 ROS 2 MCAP。"""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import termios
import time
import tty
from typing import Any, Callable, List, Mapping, Optional, TextIO

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import yaml

from fastumi_data.extrinsic import load_tracker_tcp_extrinsic


# 默认录制的数据和 episode 边界话题。
DEFAULT_TOPICS = [
    "/tof_stereo_camera/rgb/image_raw",
    "/tof_stereo_camera/rgb/frame_seqidx",
    "/gripper/state",
    "/vive_tracker/pose",
    "/vive_tracker/status",
    "/fastumi/episode/events",
    "/tf_static",
]


class _TerminalKeyReader:
    """在交互终端中以非阻塞方式读取单个按键。"""

    def __init__(self, stream: Optional[TextIO] = None) -> None:
        """保存输入流并初始化终端状态。"""
        self._stream = stream or sys.stdin
        self._file_descriptor: Optional[int] = None
        self._original_settings: Optional[list] = None
        self.enabled = False

    def __enter__(self) -> "_TerminalKeyReader":
        """进入 cbreak 模式，使空格键无需回车即可读取。"""
        if not self._stream.isatty():
            return self
        self._file_descriptor = self._stream.fileno()
        self._original_settings = termios.tcgetattr(self._file_descriptor)
        tty.setcbreak(self._file_descriptor)
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """恢复进入读取器前的终端设置。"""
        del exc_type, exc_value, traceback
        if (
            self._file_descriptor is not None
            and self._original_settings is not None
        ):
            termios.tcsetattr(
                self._file_descriptor,
                termios.TCSADRAIN,
                self._original_settings,
            )
        self.enabled = False

    def read_key(self, timeout_sec: float) -> Optional[str]:
        """在指定时间内读取一个按键，非交互输入时避免忙循环。"""
        if not self.enabled:
            time.sleep(timeout_sec)
            return None
        readable, _, _ = select.select([self._stream], [], [], timeout_sec)
        if not readable:
            return None
        return self._stream.read(1)


class _EpisodeServiceClient:
    """在录制器进程内调用 episode 的 Trigger 服务。"""

    def __init__(self, node: Node) -> None:
        """创建 start 和 stop 服务客户端。"""
        self._node = node
        self._clients = {
            command: node.create_client(
                Trigger, f"/fastumi/episode/{command}"
            )
            for command in ("start", "stop")
        }

    def wait_until_ready(self, timeout_sec: float = 5.0) -> None:
        """确认开始和结束服务已经可用。"""
        for command, client in self._clients.items():
            service_name = f"/fastumi/episode/{command}"
            if not client.wait_for_service(timeout_sec=timeout_sec):
                raise RuntimeError(
                    f"服务 {service_name} 在 {timeout_sec:g} 秒内不可用"
                )

    def call(self, command: str) -> str:
        """调用指定服务，并返回服务端的状态说明。"""
        client = self._clients[command]
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(
            self._node, future, timeout_sec=10.0
        )
        if future.result() is None:
            raise RuntimeError(f"服务 /fastumi/episode/{command} 调用超时")
        result = future.result()
        if not result.success:
            raise RuntimeError(result.message)
        return result.message


def _episode_command_from_key(
    key: Optional[str], episode_active: bool
) -> Optional[str]:
    """根据当前 episode 状态将空格键映射为开始或结束命令。"""
    if key != " ":
        return None
    return "stop" if episode_active else "start"


def _wait_for_service_readiness(
    service_client: _EpisodeServiceClient,
) -> None:
    """等待初始服务发现，超时时保留录制会话供后续按键重试。"""
    try:
        service_client.wait_until_ready()
    except RuntimeError as error:
        print(
            f"Episode 服务尚未就绪: {error}；"
            "录制继续，按空格键时将再次尝试调用"
        )


def _build_parser() -> argparse.ArgumentParser:
    """构造会话录制命令行解析器。"""
    parser = argparse.ArgumentParser(description="录制 FastUMI MCAP 会话")
    parser.add_argument("--task", required=True, help="任务名称")
    parser.add_argument("--session-id", default="", help="可选会话标识")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument(
        "--extrinsic",
        required=True,
        help="通过残差验收的 Tracker 到 TCP 外参 YAML",
    )
    parser.add_argument(
        "--processing-config",
        default="",
        help="默认使用 fastumi_data 包内的同步配置",
    )
    parser.add_argument(
        "--topic", action="append", dest="topics", help="覆盖默认录制话题"
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        default=[],
        help="需要复制到会话中的配置或标定文件",
    )
    return parser


def _sha256(path: Path) -> str:
    """计算配置快照的 SHA-256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_snapshot(
    source_text: str,
    snapshot_dir: Path,
    destination_name: Optional[str] = None,
) -> dict:
    """复制一份配置并返回相对路径和内容哈希。"""
    source = Path(source_text).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"快照文件不存在: {source}")
    destination = snapshot_dir / (destination_name or source.name)
    if destination.exists():
        raise FileExistsError(f"快照目标重名: {destination.name}")
    shutil.copy2(source, destination)
    return {
        "path": f"calibration_snapshot/{destination.name}",
        "sha256": _sha256(destination),
    }


def _calibration_passes_acceptance(metadata: Mapping[str, Any]) -> bool:
    """检查已严格校验外参元数据是否满足 2 mm、1° 正式门限。"""
    try:
        translation_rmse_mm = float(metadata["translation_rmse_mm"])
        rotation_rmse_deg = float(metadata["rotation_rmse_deg"])
    except (TypeError, ValueError, KeyError):
        return False
    return translation_rmse_mm <= 2.0 and rotation_rmse_deg <= 1.0


def _terminate_process(process: subprocess.Popen) -> None:
    """向子进程发送 SIGINT，并在必要时升级为终止。"""
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5.0)


def _build_bag_command(raw_dir: Path, topics: List[str]) -> List[str]:
    """构造关闭 rosbag 自带键盘控制的连续录制命令。"""
    return [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--storage-preset-profile",
        "zstd_fast",
        "--disable-keyboard-controls",
        "--output",
        str(raw_dir / "bag"),
        "--topics",
        *topics,
    ]


def _wait_for_recording_processes(
    manager_process: subprocess.Popen,
    bag_process: subprocess.Popen,
    key_reader: Optional[_TerminalKeyReader] = None,
    command_handler: Optional[Callable[[str], None]] = None,
) -> None:
    """监管录制进程，并在交互终端中处理 episode 快捷键。"""
    episode_active = False  # 当前录制器快捷键对应的 episode 状态。
    while True:
        manager_exit_code = manager_process.poll()
        if manager_exit_code is not None:
            raise RuntimeError(
                "episode_manager 意外退出，退出码为 "
                f"{manager_exit_code}"
            )
        if key_reader is None:
            try:
                bag_exit_code = bag_process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                continue
        else:
            bag_exit_code = bag_process.poll()
        if bag_exit_code is not None:
            if bag_exit_code != 0:
                raise RuntimeError(
                    f"ros2 bag record 退出码为 {bag_exit_code}"
                )
            return
        if key_reader is None:
            continue
        command = _episode_command_from_key(
            key_reader.read_key(0.2), episode_active
        )
        if command is None or command_handler is None:
            continue
        try:
            command_handler(command)
            episode_active = command == "start"
        except RuntimeError as error:
            print(f"Episode {command} 控制失败: {error}")


def main(argv: Optional[List[str]] = None) -> None:
    """启动 episode 管理节点和 ros2 bag MCAP 录制进程。"""
    arguments = _build_parser().parse_args(argv)
    extrinsic = load_tracker_tcp_extrinsic(arguments.extrinsic)
    if not _calibration_passes_acceptance(extrinsic.metadata):
        raise ValueError("Tracker 到 TCP 外参未通过 2 mm、1° 验收")
    session_id = arguments.session_id or datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    session_dir = (
        Path(arguments.dataset_root) / arguments.task / session_id
    ).resolve()
    raw_dir = session_dir / "raw"
    snapshot_dir = session_dir / "calibration_snapshot"
    topics = arguments.topics or DEFAULT_TOPICS

    raw_dir.mkdir(parents=True, exist_ok=False)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    processing_config = arguments.processing_config or str(
        Path(get_package_share_directory("fastumi_data"))
        / "config"
        / "processing.yaml"
    )
    extrinsic_snapshot = _copy_snapshot(
        arguments.extrinsic,
        snapshot_dir,
        "tracker_to_tcp.yaml",
    )
    processing_snapshot = _copy_snapshot(
        processing_config,
        snapshot_dir,
        "processing.yaml",
    )
    copied_snapshots = []
    for source_text in arguments.snapshot:
        copied_snapshots.append(
            _copy_snapshot(source_text, snapshot_dir)
        )
    manifest = {
        "schema_version": 2,
        "task_name": arguments.task,
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "storage_id": "mcap",
        "bag_uri": "raw/bag",
        "topics": topics,
        "tracker_to_tcp": extrinsic_snapshot,
        "processing_config": processing_snapshot,
        "additional_snapshots": copied_snapshots,
    }
    (session_dir / "session.yaml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    manager_command = [
        "ros2",
        "run",
        "fastumi_data",
        "episode_manager",
        "--ros-args",
        "-p",
        f"task_name:={arguments.task}",
        "-p",
        f"session_id:={session_id}",
    ]
    bag_command = _build_bag_command(raw_dir, topics)
    print(f"会话目录: {session_dir}")
    manager_process = None
    bag_process = None
    control_node = None
    try:
        manager_process = subprocess.Popen(manager_command)
        bag_process = subprocess.Popen(bag_command)
        with _TerminalKeyReader() as key_reader:
            if not key_reader.enabled:
                print(
                    "标准输入不是交互终端；使用 "
                    "`ros2 run fastumi_data episode_command start|stop|abort` "
                    "控制分段"
                )
                _wait_for_recording_processes(manager_process, bag_process)
            else:
                rclpy.init()
                control_node = Node("fastumi_record_session_control")
                service_client = _EpisodeServiceClient(control_node)
                _wait_for_service_readiness(service_client)
                print(
                    "快捷键: [空格] 开始/结束当前示范、"
                    "[Ctrl+C] 结束会话"
                )

                def handle_command(command: str) -> None:
                    """调用服务并在录制终端显示执行结果。"""
                    print(service_client.call(command))

                _wait_for_recording_processes(
                    manager_process,
                    bag_process,
                    key_reader=key_reader,
                    command_handler=handle_command,
                )
    except KeyboardInterrupt:
        pass
    finally:
        if control_node is not None:
            control_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if bag_process is not None:
            _terminate_process(bag_process)
        if manager_process is not None:
            _terminate_process(manager_process)


if __name__ == "__main__":
    main()
