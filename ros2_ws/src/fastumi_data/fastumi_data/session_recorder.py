"""创建 FastUMI 会话目录、快照配置并连续录制 ROS 2 MCAP。"""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import signal
import subprocess
from typing import List, Optional

from ament_index_python.packages import get_package_share_directory
import yaml


# 默认录制的数据和 episode 边界话题。
DEFAULT_TOPICS = [
    "/xv_sdk/SN250801DR48FB26001253/rgb/image",
    "/gripper/state",
    "/vive_tracker/pose",
    "/vive_tracker/status",
    "/fastumi/episode/events",
    "/tf_static",
]


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
        "--allow-unverified-calibration",
        action="store_true",
        help="仅用于降级试验；允许残差缺失或超过正式门限的外参",
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


def _calibration_passes_acceptance(path: str) -> bool:
    """检查外参是否满足 2 mm、1° 的正式数据门限。"""
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        translation_rmse_mm = float(document["translation_rmse_mm"])
        rotation_rmse_deg = float(document["rotation_rmse_deg"])
    except (OSError, TypeError, ValueError, KeyError, yaml.YAMLError):
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


def _wait_for_recording_processes(
    manager_process: subprocess.Popen,
    bag_process: subprocess.Popen,
) -> None:
    """同时监管事件管理器和录包进程，任一异常退出即停止会话。"""
    while True:
        manager_exit_code = manager_process.poll()
        if manager_exit_code is not None:
            raise RuntimeError(
                "episode_manager 意外退出，退出码为 "
                f"{manager_exit_code}"
            )
        try:
            bag_exit_code = bag_process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            continue
        if bag_exit_code != 0:
            raise RuntimeError(
                f"ros2 bag record 退出码为 {bag_exit_code}"
            )
        return


def main(argv: Optional[List[str]] = None) -> None:
    """启动 episode 管理节点和 ros2 bag MCAP 录制进程。"""
    arguments = _build_parser().parse_args(argv)
    session_id = arguments.session_id or datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")
    session_dir = (
        Path(arguments.dataset_root) / arguments.task / session_id
    ).resolve()
    raw_dir = session_dir / "raw"
    snapshot_dir = session_dir / "calibration_snapshot"
    topics = arguments.topics or DEFAULT_TOPICS

    calibration_verified = _calibration_passes_acceptance(
        arguments.extrinsic
    )
    if (
        not calibration_verified
        and not arguments.allow_unverified_calibration
    ):
        raise ValueError(
            "Tracker 到 TCP 外参未通过 2 mm、1° 验收；"
            "降级试验可显式使用 --allow-unverified-calibration"
        )
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
        "schema_version": 1,
        "task_name": arguments.task,
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "storage_id": "mcap",
        "bag_uri": "raw/bag",
        "topics": topics,
        "tracker_to_tcp": extrinsic_snapshot,
        "calibration_verified": calibration_verified,
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
    bag_command = [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--output",
        str(raw_dir / "bag"),
        "--topics",
        *topics,
    ]
    print(f"会话目录: {session_dir}")
    print("使用 `ros2 run fastumi_data episode_command start|stop|abort` 控制分段")
    manager_process = None
    bag_process = None
    try:
        manager_process = subprocess.Popen(manager_command)
        bag_process = subprocess.Popen(bag_command)
        _wait_for_recording_processes(manager_process, bag_process)
    except KeyboardInterrupt:
        pass
    finally:
        if bag_process is not None:
            _terminate_process(bag_process)
        if manager_process is not None:
            _terminate_process(manager_process)


if __name__ == "__main__":
    main()
