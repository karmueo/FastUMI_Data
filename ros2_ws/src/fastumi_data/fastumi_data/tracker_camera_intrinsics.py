"""为 Tracker–鱼眼外参标定提取图像并调用 Kalibr ROS 2 生成内参。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_prefix,
    get_package_share_directory,
)
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import yaml

from fastumi_data.tracker_camera_bag import _open_reader
from fastumi_data.tracker_camera_config import FisheyeCameraModel, load_kalibr_camera


KALIBR_REPOSITORY = "https://github.com/yuhang0131/kalibr_ros2.git"
KALIBR_COMMIT = "c79d1b0cf012fed63dcff5ab8c76778e8343190f"
IMAGE_MESSAGE_TYPE = "sensor_msgs/msg/Image"


class IntrinsicCalibrationError(RuntimeError):
    """表示可诊断且必须终止外参流水线的自动内参失败。"""

    def __init__(self, message: str, log_path: Path | None = None) -> None:
        """保存错误说明和可选的已保留 Kalibr 日志路径。"""
        super().__init__(message)
        self.log_path = log_path

def resolve_kalibr_package_prefix() -> str:
    """返回已 source 的 Kalibr 包前缀，缺失时给出可执行的安装说明。"""
    try:
        return get_package_prefix("kalibr_imu_camera")
    except PackageNotFoundError as error:
        raise IntrinsicCalibrationError(
            "未找到 kalibr_imu_camera。请在独立工作空间用 "
            "ros2_ws/kalibr_ros2.repos 固定版本导入 Kalibr，应用 "
            "ros2_ws/patches/kalibr_ros2-jazzy.patch，并按 Jazzy -> "
            "Kalibr overlay -> FastUMI 的顺序 source。"
        ) from error


def _kalibr_resource_path(filename: str) -> Path:
    """优先返回已安装包共享目录中的 Kalibr 资源，开发时回退源码树。"""
    try:
        installed = Path(get_package_share_directory("fastumi_data")) / "kalibr" / Path(filename).name
        if installed.is_file():
            return installed
    except PackageNotFoundError:
        pass
    source = Path(__file__).resolve().parents[3] / filename
    if source.is_file():
        return source
    raise IntrinsicCalibrationError(f"缺少 Kalibr 资源文件: {filename}")

def kalibr_compatibility_patch_path() -> Path:
    """返回 Kalibr Jazzy 兼容补丁的安装路径或开发回退路径。"""
    return _kalibr_resource_path("patches/kalibr_ros2-jazzy.patch")

def kalibr_repositories_path() -> Path:
    """返回固定 Kalibr 仓库清单的安装路径或开发回退路径。"""
    return _kalibr_resource_path("kalibr_ros2.repos")


@dataclass(frozen=True)
class ExtractedImageBag:
    """保存临时 SQLite3 bag 的图像话题、帧数与已验证分辨率。"""

    bag_uri: Path
    topic: str
    frame_count: int
    resolution: tuple[int, int]


@dataclass(frozen=True)
class KalibrArtifacts:
    """保存 Kalibr staging 目录内已验证的四个输出文件。"""

    yaml_path: Path
    results_path: Path
    report_path: Path
    log_path: Path


@dataclass(frozen=True)
class IntrinsicCalibrationResult:
    """保存已发布相机模型、文件路径及可写入报告的溯源信息。"""

    camera: FisheyeCameraModel
    artifacts: KalibrArtifacts
    provenance: Mapping[str, Any]


def _stamp_to_ns(stamp: object) -> int:
    """把 ROS header 时间转换为纳秒整数。"""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def sha256_path(path: Path | str) -> str:
    """计算单个文件内容的 SHA-256，用于产物可追溯性。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_image_topic_to_sqlite3(
    source_bag_uri: str,
    image_topic: str,
    destination_uri: Path | str,
    frequency_hz: float,
    sample_end_offset_s: float | None = None,
) -> ExtractedImageBag:
    """从 MCAP 按 header 时间抽取 Image CDR 原文到临时 SQLite3 bag。

    每条输入图像均检查严格递增的 header 时间和一致的正分辨率；写入时保留
    原序列化 CDR 字节及原 bag 写入时间，避免在 Kalibr 前发生转码。
    """
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("内参抽帧频率必须为有限正数")
    if sample_end_offset_s is not None and (
        not np.isfinite(sample_end_offset_s) or sample_end_offset_s < 0.0
    ):
        raise ValueError("sample_end_offset_s 必须为有限非负数")
    reader = _open_reader(source_bag_uri)
    topic_types = {
        metadata.name: metadata.type
        for metadata in reader.get_all_topics_and_types()
    }
    if image_topic not in topic_types:
        raise ValueError(f"MCAP 缺少话题: {image_topic}")
    if topic_types[image_topic] != IMAGE_MESSAGE_TYPE:
        raise ValueError(
            f"话题 {image_topic} 必须使用 {IMAGE_MESSAGE_TYPE}，"
            f"实际为 {topic_types[image_topic]}"
        )
    message_type = get_message(IMAGE_MESSAGE_TYPE)
    bag_uri = Path(destination_uri)
    bag_uri.parent.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag_uri), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            id=0,
            name=image_topic,
            type=IMAGE_MESSAGE_TYPE,
            serialization_format="cdr",
        )
    )
    interval_ns = int(round(1_000_000_000.0 / frequency_hz))
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    last_selected_timestamp_ns: int | None = None
    resolution: tuple[int, int] | None = None
    selected = 0
    while reader.has_next():
        topic, serialized, bag_timestamp_ns = reader.read_next()
        if topic != image_topic:
            continue
        message = deserialize_message(serialized, message_type)
        timestamp_ns = _stamp_to_ns(message.header.stamp)
        if last_timestamp_ns is not None and timestamp_ns <= last_timestamp_ns:
            raise ValueError(f"话题 {image_topic} 的 header 时间戳不是严格递增")
        last_timestamp_ns = timestamp_ns
        try:
            current_resolution = (int(message.width), int(message.height))
        except (TypeError, ValueError) as error:
            raise ValueError("图像分辨率无效") from error
        if current_resolution[0] <= 0 or current_resolution[1] <= 0:
            raise ValueError("图像分辨率必须为正数")
        if resolution is None:
            resolution = current_resolution
            first_timestamp_ns = timestamp_ns
        elif current_resolution != resolution:
            raise ValueError("图像分辨率在 MCAP 中不一致")
        assert first_timestamp_ns is not None
        if (
            sample_end_offset_s is not None
            and timestamp_ns - first_timestamp_ns > sample_end_offset_s * 1.0e9
        ):
            break
        if (
            last_selected_timestamp_ns is None
            or timestamp_ns - last_selected_timestamp_ns >= interval_ns
        ):
            writer.write(image_topic, serialized, int(bag_timestamp_ns))
            last_selected_timestamp_ns = timestamp_ns
            selected += 1
    del writer
    if selected == 0 or resolution is None:
        raise ValueError(f"话题 {image_topic} 没有可用于内参标定的图像帧")
    return ExtractedImageBag(bag_uri, image_topic, selected, resolution)


def build_kalibr_command(
    bag_uri: Path | str, image_topic: str, target_config: Path | str
) -> list[str]:
    """构造固定的单相机 pinhole-equi Kalibr 命令参数，不经 shell。"""
    return [
        "ros2", "run", "kalibr_imu_camera", "kalibr_calibrate_cameras",
        "--bag", str(bag_uri), "--topics", image_topic,
        "--models", "pinhole-equi", "--target", str(target_config),
        "--verbose", "--no-shuffle", "--dont-show-report",
    ]


def run_kalibr(
    command: Sequence[str], staging_dir: Path | str, log_path: Path | str
) -> subprocess.CompletedProcess[str]:
    """在 staging 目录运行 Kalibr，并把 stdout/stderr 完整写入日志。"""
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    argv_text = " ".join(str(item) for item in command)
    try:

        completed = subprocess.run(
            list(command), cwd=str(staging_dir), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            check=False,
        )
    except OSError as error:
        Path(log_path).write_text(
            f"argv: {argv_text}\nexit_status: launch_error\n无法启动 Kalibr: {error}\n",
            encoding="utf-8",
        )
        raise IntrinsicCalibrationError(f"无法启动 Kalibr: {error}", Path(log_path)) from error
    Path(log_path).write_text(
        f"argv: {argv_text}\nexit_status: {completed.returncode}\n{completed.stdout or ''}",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise IntrinsicCalibrationError(
            f"Kalibr 返回非零退出码 {completed.returncode}", Path(log_path)
        )
    return completed


def _find_single_artifact(staging_dir: Path, pattern: str) -> Path:
    """按 Kalibr 的前缀命名规则查找唯一且非空的约定产物。"""
    matches = sorted(path for path in staging_dir.glob(pattern) if path.is_file())
    if len(matches) != 1 or matches[0].stat().st_size == 0:
        raise ValueError(f"Kalibr 缺少唯一且非空的 {pattern} 输出")
    return matches[0]


def validate_kalibr_artifacts(
    staging_dir: Path | str, log_path: Path | str, image_topic: str,
    resolution: tuple[int, int],
) -> KalibrArtifacts:
    """严格检查 Kalibr 文件命名、YAML 模型、话题、分辨率和数值范围。"""
    staging = Path(staging_dir)
    log = Path(log_path)
    if not log.is_file() or log.stat().st_size == 0:
        raise ValueError("Kalibr 日志不存在或为空")
    yaml_path = _find_single_artifact(staging, "camchain-*.yaml")
    results_path = _find_single_artifact(staging, "results-cam-*.txt")
    report_path = _find_single_artifact(staging, "report-cam-*.pdf")
    try:
        document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取 Kalibr YAML: {error}") from error
    if not isinstance(document, Mapping) or set(document) != {"cam0"}:
        raise ValueError("Kalibr YAML 必须且只能包含 cam0")
    camera = document["cam0"]
    if not isinstance(camera, Mapping):
        raise ValueError("Kalibr cam0 必须是映射")
    if camera.get("camera_model") != "pinhole" or camera.get("distortion_model") != "equidistant":
        raise ValueError("Kalibr 输出必须是 pinhole/equidistant")
    if camera.get("rostopic") != image_topic:
        raise ValueError("Kalibr 输出 rostopic 与抽取话题不一致")
    if tuple(camera.get("resolution", ())) != tuple(resolution):
        raise ValueError("Kalibr 输出分辨率与抽取图像不一致")
    try:
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
        distortion = np.asarray(camera["distortion_coeffs"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Kalibr 输出缺少有效内参或畸变参数") from error
    if intrinsics.shape != (4,) or not np.all(np.isfinite(intrinsics)):
        raise ValueError("Kalibr 输出内参必须包含四个有限数值")
    if intrinsics[0] <= 0.0 or intrinsics[1] <= 0.0:
        raise ValueError("Kalibr 输出焦距必须为正数")
    if distortion.shape != (4,) or not np.all(np.isfinite(distortion)):
        raise ValueError("Kalibr 输出畸变参数必须包含四个有限数值")
    try:
        load_kalibr_camera(str(yaml_path))
    except ValueError as error:
        raise ValueError(f"Kalibr YAML 无法由本工具加载: {error}") from error
    return KalibrArtifacts(yaml_path, results_path, report_path, log)


def publish_kalibr_artifacts(
    artifacts: KalibrArtifacts, output_dir: Path | str
) -> KalibrArtifacts:
    """按 TXT/PDF/log/YAML 顺序原子发布 Kalibr 文件，YAML 是提交标记。"""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    targets = (
        (artifacts.results_path, destination / "camera_intrinsics_results.txt"),
        (artifacts.report_path, destination / "camera_intrinsics_report.pdf"),
        (artifacts.log_path, destination / "camera_intrinsics.log"),
        (artifacts.yaml_path, destination / "camera_intrinsics.yaml"),
    )
    published = []
    try:
        for source, target in targets:
            temporary = target.with_name(f".{target.name}.tmp")
            temporary.write_bytes(source.read_bytes())
            temporary.replace(target)
            published.append(target)
    except OSError as error:
        raise IntrinsicCalibrationError(f"无法发布 Kalibr 产物: {error}") from error
    return KalibrArtifacts(published[3], published[0], published[1], published[2])


def publish_kalibr_log(log_path: Path | str, output_dir: Path | str) -> Path | None:
    """在失败路径中尽力保留 Kalibr 日志，不发布未验证的其他产物。"""
    source = Path(log_path)
    destination = Path(output_dir) / "camera_intrinsics.log"
    if source.is_file():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(source.read_bytes())
        temporary.replace(destination)
        return destination
    return None


def kalibr_provenance(
    artifacts: KalibrArtifacts, frequency_hz: float, frame_count: int,
    command: Sequence[str], package_prefix: str,
    expected_compatibility_patch_sha256: str,
) -> dict[str, Any]:
    """创建自动内参的溯源记录，并区分运行时事实与期望版本。"""
    paths = {
        "yaml": artifacts.yaml_path, "results": artifacts.results_path,
        "report": artifacts.report_path, "log": artifacts.log_path,
    }
    return {
        "source": "kalibr_ros2",
        "frequency_hz": float(frequency_hz),
        "selected_frame_count": int(frame_count),
        "command_argv": list(command),
        "expected_kalibr_commit": KALIBR_COMMIT,
        "expected_compatibility_patch_sha256": expected_compatibility_patch_sha256,
        "kalibr_imu_camera_prefix": package_prefix,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_path(path)}
            for name, path in paths.items()
        },
        "final_yaml_sha256": sha256_path(artifacts.yaml_path),
    }
