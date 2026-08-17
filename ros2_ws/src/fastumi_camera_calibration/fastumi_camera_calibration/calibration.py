"""从 FastUMI MCAP 抽取图像并调用 Kalibr 生成单鱼眼相机内参。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
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


# 固定的 Kalibr ROS 2 移植仓库，用于复现当前标定环境。
KALIBR_REPOSITORY = "https://github.com/yuhang0131/kalibr_ros2.git"
# 已验证的 Kalibr ROS 2 提交。
KALIBR_COMMIT = "c79d1b0cf012fed63dcff5ab8c76778e8343190f"
# 内参输入必须使用的 ROS 2 图像消息类型。
IMAGE_MESSAGE_TYPE = "sensor_msgs/msg/Image"
# FastUMI 当前鱼眼相机的默认图像话题。
DEFAULT_IMAGE_TOPIC = "/tof_stereo_camera/rgb/image_raw"


class IntrinsicCalibrationError(RuntimeError):
    """表示可诊断且必须终止内参流水线的失败。"""

    def __init__(self, message: str, log_path: Path | None = None) -> None:
        """保存错误说明和可选的已保留 Kalibr 日志路径。

        Args:
            message: 面向用户的失败说明。
            log_path: 已发布到输出目录的本次运行日志。
        """
        super().__init__(message)
        self.log_path = log_path


@dataclass(frozen=True)
class ExtractedImageBag:
    """保存临时 SQLite3 bag 的话题、帧数与图像分辨率。"""

    bag_uri: Path
    topic: str
    frame_count: int
    resolution: tuple[int, int]


@dataclass(frozen=True)
class KalibrArtifacts:
    """保存 Kalibr 运行产生并通过校验的四个文件。"""

    yaml_path: Path
    results_path: Path
    report_path: Path
    log_path: Path


@dataclass(frozen=True)
class IntrinsicCalibrationResult:
    """保存已发布产物与可写入自动化记录的溯源信息。"""

    artifacts: KalibrArtifacts
    provenance: Mapping[str, Any]


def _resource_path(filename: str) -> Path:
    """返回已安装或源码树中的 Kalibr 版本资源。

    Args:
        filename: 相对于包内 vendor 目录的资源路径。

    Returns:
        存在的绝对资源路径。

    Raises:
        IntrinsicCalibrationError: 安装目录和源码目录都缺少资源。
    """
    try:
        # 已安装共享目录是正常运行时的首选资源来源。
        installed = (
            Path(get_package_share_directory("fastumi_camera_calibration"))
            / "vendor"
            / filename
        )
        if installed.is_file():
            return installed
    except PackageNotFoundError:
        pass
    # 源码回退路径允许在尚未安装包时运行单元测试。
    source = Path(__file__).resolve().parents[1] / "vendor" / filename
    if source.is_file():
        return source
    raise IntrinsicCalibrationError(f"缺少 Kalibr 资源文件: {filename}")


def kalibr_compatibility_patch_path() -> Path:
    """返回 Kalibr Jazzy 兼容补丁路径。"""
    return _resource_path("patches/kalibr_ros2-jazzy.patch")


def kalibr_repositories_path() -> Path:
    """返回固定版本的 Kalibr 仓库清单路径。"""
    return _resource_path("kalibr_ros2.repos")


def resolve_kalibr_package_prefix() -> str:
    """返回已 source 的 Kalibr 包前缀。"""
    try:
        return get_package_prefix("kalibr_imu_camera")
    except PackageNotFoundError as error:
        raise IntrinsicCalibrationError(
            "未找到 kalibr_imu_camera。请使用 "
            "fastumi_camera_calibration/vendor/kalibr_ros2.repos "
            "固定版本导入 Kalibr，应用 vendor/patches/"
            "kalibr_ros2-jazzy.patch，并按 Jazzy -> Kalibr overlay -> "
            "FastUMI 的顺序 source。"
        ) from error


def sha256_path(path: Path | str) -> str:
    """计算文件内容 SHA-256。"""
    # 摘要对象用于流式处理 Kalibr PDF 等较大产物。
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stamp_to_ns(stamp: object) -> int:
    """把 ROS 时间转换为纳秒整数。"""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _open_mcap_reader(bag_uri: str) -> rosbag2_py.SequentialReader:
    """以 MCAP 存储格式打开顺序读取器。"""
    # 输入目录必须在创建任何临时输出前得到验证。
    bag_path = Path(bag_uri)
    if not bag_path.exists():
        raise ValueError(f"MCAP 路径不存在: {bag_path}")
    # 顺序读取器保持内存占用与 bag 大小无关。
    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(
            rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap"),
            rosbag2_py.ConverterOptions("", ""),
        )
    except RuntimeError as error:
        raise ValueError(f"无法打开 MCAP {bag_path}: {error}") from error
    return reader


def extract_image_topic_to_sqlite3(
    source_bag_uri: str,
    image_topic: str,
    destination_uri: Path | str,
    frequency_hz: float,
    sample_end_offset_s: float | None = None,
) -> ExtractedImageBag:
    """从 MCAP 按 header 时间抽取 Image CDR 原文到临时 SQLite3 bag。

    每条输入图像均检查严格递增的 header 时间和一致的正分辨率；写入时
    保留原序列化 CDR 字节及原 bag 写入时间。

    Args:
        source_bag_uri: 输入 MCAP bag 目录。
        image_topic: 单相机图像话题。
        destination_uri: 临时 SQLite3 bag 目录。
        frequency_hz: 最大抽帧频率。
        sample_end_offset_s: 从首帧 header 时间起计算的可选时长上限。

    Returns:
        临时 bag 路径、抽取帧数和分辨率。

    Raises:
        ValueError: 参数、话题、时间戳或分辨率不符合约束。
    """
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("内参抽帧频率必须为有限正数")
    if sample_end_offset_s is not None and (
        not np.isfinite(sample_end_offset_s) or sample_end_offset_s < 0.0
    ):
        raise ValueError("sample_end_offset_s 必须为有限非负数")
    # MCAP reader 属于本包实现，避免依赖 fastumi_data 私有函数。
    reader = _open_mcap_reader(source_bag_uri)
    # 话题元数据用于在反序列化前验证消息契约。
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
    # 消息类型用于读取 header 时间与分辨率。
    message_type = get_message(IMAGE_MESSAGE_TYPE)
    # SQLite3 目标只存在于临时 staging 目录。
    bag_uri = Path(destination_uri)
    bag_uri.parent.mkdir(parents=True, exist_ok=True)
    # 写入器保留原消息字节和原 bag 写入时间。
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
    # 抽帧间隔以整数纳秒比较，保证边界行为确定。
    interval_ns = int(round(1_000_000_000.0 / frequency_hz))
    # 首帧时间用于应用可选采样时长。
    first_timestamp_ns: int | None = None
    # 上一输入时间用于拒绝乱序 header。
    last_timestamp_ns: int | None = None
    # 上一选中时间用于频率限制。
    last_selected_timestamp_ns: int | None = None
    # 分辨率在首帧确定，后续帧必须一致。
    resolution: tuple[int, int] | None = None
    # 已写入临时 bag 的图像帧数。
    selected = 0
    while reader.has_next():
        # 原始序列化消息必须原样交给 Kalibr。
        topic, serialized, bag_timestamp_ns = reader.read_next()
        if topic != image_topic:
            continue
        # 反序列化对象仅用于读取元数据。
        message = deserialize_message(serialized, message_type)
        # 图像 header 时间是抽帧与时间窗口的统一基准。
        timestamp_ns = _stamp_to_ns(message.header.stamp)
        if last_timestamp_ns is not None and timestamp_ns <= last_timestamp_ns:
            raise ValueError(f"话题 {image_topic} 的 header 时间戳不是严格递增")
        last_timestamp_ns = timestamp_ns
        try:
            # 当前帧分辨率按 Kalibr 的宽、高顺序记录。
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
    """构造固定的单相机 pinhole-equi Kalibr 命令。"""
    return [
        "ros2", "run", "kalibr_imu_camera", "kalibr_calibrate_cameras",
        "--bag", str(bag_uri), "--topics", image_topic,
        "--models", "pinhole-equi", "--target", str(target_config),
        "--verbose", "--no-shuffle", "--dont-show-report",
    ]


def run_kalibr(
    command: Sequence[str], staging_dir: Path | str, log_path: Path | str
) -> subprocess.CompletedProcess[str]:
    """在 staging 目录运行 Kalibr 并保存完整输出。"""
    # 子进程继承已 source 的 ROS 2 overlay 环境。
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["QT_QPA_PLATFORM"] = "offscreen"
    # 文本命令写入日志以便复现。
    argv_text = " ".join(str(item) for item in command)
    try:
        # 参数列表和禁用 shell 能避免路径被二次解释。
        completed = subprocess.run(
            list(command),
            cwd=str(staging_dir),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except OSError as error:
        Path(log_path).write_text(
            f"argv: {argv_text}\n"
            "exit_status: launch_error\n"
            f"无法启动 Kalibr: {error}\n",
            encoding="utf-8",
        )
        raise IntrinsicCalibrationError(
            f"无法启动 Kalibr: {error}", Path(log_path)
        ) from error
    Path(log_path).write_text(
        f"argv: {argv_text}\n"
        f"exit_status: {completed.returncode}\n"
        f"{completed.stdout or ''}",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise IntrinsicCalibrationError(
            f"Kalibr 返回非零退出码 {completed.returncode}", Path(log_path)
        )
    return completed


def _find_single_artifact(staging_dir: Path, pattern: str) -> Path:
    """按 Kalibr 命名规则查找唯一且非空的产物。"""
    # 排序让多个匹配时的诊断保持确定。
    matches = sorted(
        path for path in staging_dir.glob(pattern) if path.is_file()
    )
    if len(matches) != 1 or matches[0].stat().st_size == 0:
        raise ValueError(f"Kalibr 缺少唯一且非空的 {pattern} 输出")
    return matches[0]


def _validate_camera_document(
    yaml_path: Path, image_topic: str, resolution: tuple[int, int]
) -> None:
    """验证 Kalibr 单相机 YAML 的模型、话题、分辨率和数值。"""
    try:
        # SafeLoader 阻止标定文件构造任意 Python 对象。
        document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取 Kalibr YAML: {error}") from error
    if not isinstance(document, Mapping) or set(document) != {"cam0"}:
        raise ValueError("Kalibr YAML 必须且只能包含 cam0")
    # cam0 是首版唯一支持的相机。
    camera = document["cam0"]
    if not isinstance(camera, Mapping):
        raise ValueError("Kalibr cam0 必须是映射")
    if (
        camera.get("camera_model") != "pinhole"
        or camera.get("distortion_model") != "equidistant"
    ):
        raise ValueError("Kalibr 输出必须是 pinhole/equidistant")
    if camera.get("rostopic") != image_topic:
        raise ValueError("Kalibr 输出 rostopic 与抽取话题不一致")
    if tuple(camera.get("resolution", ())) != tuple(resolution):
        raise ValueError("Kalibr 输出分辨率与抽取图像不一致")
    try:
        # 内参与畸变参数必须满足固定的四参数模型。
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
        distortion = np.asarray(
            camera["distortion_coeffs"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Kalibr 输出缺少有效内参或畸变参数") from error
    if intrinsics.shape != (4,) or not np.all(np.isfinite(intrinsics)):
        raise ValueError("Kalibr 输出内参必须包含四个有限数值")
    if intrinsics[0] <= 0.0 or intrinsics[1] <= 0.0:
        raise ValueError("Kalibr 输出焦距必须为正数")
    if distortion.shape != (4,) or not np.all(np.isfinite(distortion)):
        raise ValueError("Kalibr 输出畸变参数必须包含四个有限数值")


def validate_kalibr_artifacts(
    staging_dir: Path | str,
    log_path: Path | str,
    image_topic: str,
    resolution: tuple[int, int],
) -> KalibrArtifacts:
    """检查 Kalibr 的 YAML、TXT、PDF 和日志产物。"""
    # 工作目录包含 Kalibr 按 bag 名生成的文件。
    staging = Path(staging_dir)
    # 日志独立于 Kalibr 默认产物命名。
    log = Path(log_path)
    if not log.is_file() or log.stat().st_size == 0:
        raise ValueError("Kalibr 日志不存在或为空")
    # 单相机调用必须分别产生唯一 YAML、文本结果和 PDF 报告。
    yaml_path = _find_single_artifact(staging, "camchain-*.yaml")
    results_path = _find_single_artifact(staging, "results-cam-*.txt")
    report_path = _find_single_artifact(staging, "report-cam-*.pdf")
    _validate_camera_document(yaml_path, image_topic, resolution)
    return KalibrArtifacts(yaml_path, results_path, report_path, log)


def publish_kalibr_artifacts(
    artifacts: KalibrArtifacts, output_dir: Path | str
) -> KalibrArtifacts:
    """按 TXT、PDF、日志、YAML 顺序原子发布 Kalibr 产物。"""
    # YAML 最后发布，作为整组产物成功提交的标记。
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    # 目标名称保持与旧集成流程兼容。
    targets = (
        (
            artifacts.results_path,
            destination / "camera_intrinsics_results.txt",
        ),
        (
            artifacts.report_path,
            destination / "camera_intrinsics_report.pdf",
        ),
        (artifacts.log_path, destination / "camera_intrinsics.log"),
        (artifacts.yaml_path, destination / "camera_intrinsics.yaml"),
    )
    # 已完成替换的目标按约定顺序保存。
    published: list[Path] = []
    try:
        for source, target in targets:
            # 同目录临时文件确保 replace 是原子操作。
            temporary = target.with_name(f".{target.name}.tmp")
            temporary.write_bytes(source.read_bytes())
            temporary.replace(target)
            published.append(target)
    except OSError as error:
        raise IntrinsicCalibrationError(
            f"无法发布 Kalibr 产物: {error}"
        ) from error
    return KalibrArtifacts(
        published[3], published[0], published[1], published[2]
    )


def publish_kalibr_log(
    log_path: Path | str, output_dir: Path | str
) -> Path | None:
    """在失败路径中尽力保留本次 Kalibr 日志。"""
    # 缺失源日志时不得返回历史日志路径。
    source = Path(log_path)
    if not source.is_file():
        return None
    # 失败日志同样使用原子替换。
    destination = Path(output_dir) / "camera_intrinsics.log"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)
    return destination


def kalibr_provenance(
    artifacts: KalibrArtifacts,
    frequency_hz: float,
    frame_count: int,
    command: Sequence[str],
    package_prefix: str,
    expected_compatibility_patch_sha256: str,
) -> dict[str, Any]:
    """创建独立内参标定的可复现溯源记录。"""
    # 四类产物均记录路径和内容摘要。
    paths = {
        "yaml": artifacts.yaml_path,
        "results": artifacts.results_path,
        "report": artifacts.report_path,
        "log": artifacts.log_path,
    }
    return {
        "source": "kalibr_ros2",
        "repository": KALIBR_REPOSITORY,
        "frequency_hz": float(frequency_hz),
        "selected_frame_count": int(frame_count),
        "command_argv": list(command),
        "expected_kalibr_commit": KALIBR_COMMIT,
        "expected_compatibility_patch_sha256": (
            expected_compatibility_patch_sha256
        ),
        "kalibr_imu_camera_prefix": package_prefix,
        "artifacts": {
            name: {"path": str(item), "sha256": sha256_path(item)}
            for name, item in paths.items()
        },
        "final_yaml_sha256": sha256_path(artifacts.yaml_path),
    }


def calibrate_camera_intrinsics(
    source_bag_uri: str,
    image_topic: str,
    target_config: Path | str,
    output_dir: Path | str,
    frequency_hz: float = 4.0,
    sample_end_offset_s: float | None = None,
) -> IntrinsicCalibrationResult:
    """执行完整的 MCAP 到 Kalibr 内参产物流水线。"""
    # 输出 YAML 是成功提交标记，开始新任务前必须使旧标记失效。
    destination = Path(output_dir)
    commit_marker = destination / "camera_intrinsics.yaml"
    log_marker = destination / "camera_intrinsics.log"
    commit_marker.unlink(missing_ok=True)
    log_marker.unlink(missing_ok=True)
    try:
        # 运行前确认 Kalibr overlay 已在当前环境中 source。
        package_prefix = resolve_kalibr_package_prefix()
        # 兼容补丁哈希用于确认期望的 Jazzy 构建版本。
        expected_patch_sha256 = sha256_path(
            kalibr_compatibility_patch_path()
        )
        # Kalibr 切换工作目录前必须把目标板路径绝对化。
        target_path = Path(target_config).resolve(strict=True)
    except (IntrinsicCalibrationError, OSError, ValueError) as error:
        if isinstance(error, IntrinsicCalibrationError):
            raise
        raise IntrinsicCalibrationError(str(error)) from error

    with tempfile.TemporaryDirectory(
        prefix="fastumi-camera-kalibr-"
    ) as temporary:
        # staging 在退出上下文后自动清理。
        staging = Path(temporary) / "staging"
        staging.mkdir()
        # 本次 Kalibr 日志先写入 staging。
        log_path = staging / "kalibr.log"
        try:
            # 输入 MCAP 被转换为 Kalibr ROS 2 移植支持的 SQLite3 bag。
            extracted = extract_image_topic_to_sqlite3(
                source_bag_uri,
                image_topic,
                staging / "images.sqlite3",
                frequency_hz,
                sample_end_offset_s,
            )
            # 模型和非交互参数由本包固定。
            command = build_kalibr_command(
                extracted.bag_uri, image_topic, target_path
            )
            run_kalibr(command, staging, log_path)
            # 只有全部产物通过验证后才进入最终目录。
            artifacts = validate_kalibr_artifacts(
                staging, log_path, image_topic, extracted.resolution
            )
            published = publish_kalibr_artifacts(
                artifacts, destination
            )
            # provenance 在最终路径上计算摘要。
            provenance = kalibr_provenance(
                published,
                frequency_hz,
                extracted.frame_count,
                command,
                package_prefix,
                expected_patch_sha256,
            )
            return IntrinsicCalibrationResult(published, provenance)
        except IntrinsicCalibrationError as error:
            # 优先保留异常携带的日志，否则检查 staging 日志。
            source_log = (
                error.log_path
                if error.log_path is not None and error.log_path.is_file()
                else log_path
            )
            retained_log = publish_kalibr_log(source_log, destination)
            raise IntrinsicCalibrationError(
                str(error), retained_log
            ) from error
        except (OSError, RuntimeError, ValueError) as error:
            # 抽取和产物校验失败也统一转换为稳定的公共异常。
            retained_log = publish_kalibr_log(log_path, destination)
            raise IntrinsicCalibrationError(
                str(error), retained_log
            ) from error
