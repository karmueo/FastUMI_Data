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
from typing import Callable, List, Optional, TextIO

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import yaml

from fastumi_data.extrinsic import (
    calibration_passes_acceptance,
    load_tracker_tcp_extrinsic,
)
from fastumi_data.image_modes import (
    DEFAULT_IMAGE_MODE,
    DEFAULT_IMAGE_QOS_DEPTH,
    DEFAULT_IMAGE_RELIABILITY,
    IMAGE_MODES,
    IMAGE_RELIABILITIES,
    MODE_RAW,
    message_type_for_mode,
    resolve_image_topic,
    validate_image_mode,
    validate_image_qos_depth,
    validate_image_reliability,
)


# 默认录制的非图像数据和 episode 边界话题；图像话题由所选模式决定。
NON_IMAGE_TOPICS = [
    "/gripper/state",
    "/vive_tracker/pose",
    "/vive_tracker/status",
    "/fastumi/episode/events",
    "/tf_static",
]


def default_topics(image_topic: str) -> List[str]:
    """返回默认录制列表，所选图像话题排在最前。"""
    return [image_topic, *NON_IMAGE_TOPICS]


def resolve_record_topics(
    explicit_topics: Optional[List[str]], image_topic: str
) -> List[str]:
    """确定录制话题：未显式给出时使用默认列表，否则必须包含图像话题。

    Raises:
        ValueError: 显式列表缺少所选图像话题。
    """
    if not explicit_topics:
        return default_topics(image_topic)
    if image_topic not in explicit_topics:
        raise ValueError(
            f"显式 --topic 列表必须包含所选图像话题 {image_topic}，"
            "否则录制的会话没有图像"
        )
    return list(explicit_topics)


# raw 模式的默认录制列表，保持既有名称供外部引用。
DEFAULT_TOPICS = default_topics(resolve_image_topic(DEFAULT_IMAGE_MODE))


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
        "--topic",
        action="append",
        dest="topics",
        help="覆盖默认录制话题；必须包含所选图像话题",
    )
    parser.add_argument(
        "--image-mode",
        choices=IMAGE_MODES,
        default=DEFAULT_IMAGE_MODE,
        help="图像消息模式：raw、compressed（JPEG/PNG）或 h264（无 B 帧）",
    )
    parser.add_argument(
        "--image-topic",
        default="",
        help="图像话题；留空按模式使用 /umi_camera/image_raw、.../compressed 或 .../ffmpeg",
    )
    parser.add_argument(
        "--image-reliability",
        choices=IMAGE_RELIABILITIES,
        default=DEFAULT_IMAGE_RELIABILITY,
        help="图像订阅可靠性；默认 best_effort 兼容现有相机发布端",
    )
    parser.add_argument(
        "--image-qos-depth",
        type=int,
        default=DEFAULT_IMAGE_QOS_DEPTH,
        help="图像订阅 KEEP_LAST 队列深度",
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


def _write_processing_snapshot(
    source_text: str, snapshot_dir: Path, image_mode: str, image_topic: str
) -> dict:
    """写入带实际图像模式和话题的处理配置快照，使离线转换与录制一致。"""
    source = Path(source_text).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"快照文件不存在: {source}")
    document = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ValueError(f"处理配置必须是 YAML 映射: {source}")
    document["image_mode"] = image_mode
    document["topics"] = {**(document.get("topics") or {}), "image": image_topic}
    destination = snapshot_dir / "processing.yaml"
    if destination.exists():
        raise FileExistsError(f"快照目标重名: {destination.name}")
    destination.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "path": f"calibration_snapshot/{destination.name}",
        "sha256": _sha256(destination),
    }


def _write_qos_overrides(
    path: Path, image_topic: str, reliability: str, depth: int
) -> Path:
    """写出 rosbag2 的 QoS 覆盖文件，使图像订阅使用配置的可靠性和深度。"""
    path.write_text(
        yaml.safe_dump(
            {
                image_topic: {
                    "history": "keep_last",
                    "depth": depth,
                    "reliability": reliability,
                    "durability": "volatile",
                }
            }
        ),
        encoding="utf-8",
    )
    return path


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


def _build_bag_command(
    raw_dir: Path, topics: List[str], qos_overrides: Optional[Path] = None
) -> List[str]:
    """构造关闭 rosbag 自带键盘控制的连续录制命令。"""
    command = [
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
    ]
    if qos_overrides is not None:
        command += ["--qos-profile-overrides-path", str(qos_overrides)]
    return [*command, "--topics", *topics]


def _wait_for_recording_processes(
    manager_process: Optional[subprocess.Popen],
    bag_process: subprocess.Popen,
    key_reader: Optional[_TerminalKeyReader] = None,
    command_handler: Optional[Callable[[str], None]] = None,
) -> None:
    """监管录制进程并处理快捷键；编码模式的边界服务与录包共用一个进程。"""
    episode_active = False  # 当前录制器快捷键对应的 episode 状态。
    while True:
        manager_exit_code = (
            manager_process.poll() if manager_process is not None else None
        )
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
                    f"MCAP 录制进程退出码为 {bag_exit_code}"
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
    """创建会话并启动录制；编码模式共享图像订阅和边界服务的执行器。"""
    arguments = _build_parser().parse_args(argv)
    # 先校验图像参数：配置错误不应在读取外参或创建会话目录之后才暴露。
    image_mode = validate_image_mode(arguments.image_mode)
    image_topic = resolve_image_topic(image_mode, arguments.image_topic)
    image_reliability = validate_image_reliability(arguments.image_reliability)
    image_qos_depth = validate_image_qos_depth(arguments.image_qos_depth)
    topics = resolve_record_topics(arguments.topics, image_topic)
    extrinsic = load_tracker_tcp_extrinsic(arguments.extrinsic)
    if not calibration_passes_acceptance(extrinsic.metadata):
        raise ValueError("Tracker 到 TCP 外参未通过 2 mm、1° 验收")
    session_id = arguments.session_id or datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    session_dir = (
        Path(arguments.dataset_root) / arguments.task / session_id
    ).resolve()
    raw_dir = session_dir / "raw"
    snapshot_dir = session_dir / "calibration_snapshot"
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
    processing_snapshot = _write_processing_snapshot(
        processing_config, snapshot_dir, image_mode, image_topic
    )
    qos_overrides = _write_qos_overrides(
        session_dir / "image_qos_overrides.yaml",
        image_topic,
        image_reliability,
        image_qos_depth,
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
        "image": {
            "mode": image_mode,
            "topic": image_topic,
            "message_type": message_type_for_mode(image_mode),
            "reliability": image_reliability,
            "qos_depth": image_qos_depth,
        },
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
        "-p",
        f"image_mode:={image_mode}",
        "-p",
        f"image_topic:={image_topic}",
        "-p",
        f"image_reliability:={image_reliability}",
        "-p",
        f"image_qos_depth:={image_qos_depth}",
    ]
    bag_command = _build_bag_command(raw_dir, topics, qos_overrides)
    if image_mode != MODE_RAW:
        # 共享录制进程在写入图像后更新就绪状态，消除独立订阅的关键帧竞态。
        bag_command = [
            sys.executable, "-m", "fastumi_data.continuous_recorder",
            "--bag-uri", str(raw_dir / "bag"),
        ]
        for topic in topics:
            bag_command += ["--topic", topic]
        bag_command += manager_command[4:]
    print(f"会话目录: {session_dir}")
    manager_process = None
    bag_process = None
    control_node = None
    try:
        if image_mode == MODE_RAW:
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
