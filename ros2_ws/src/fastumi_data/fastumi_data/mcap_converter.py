"""``convert_mcap``：把 MCAP 记录批量转换为 Link7 HDF5，并合并导出 30 Hz Zarr。

命令行入口执行 MCAP→``fastumi_ros2_link7_v1`` HDF5→``fastumi-rm75-link7-v1`` Zarr：
负责来源发现、配置与标称源帧率解析、外参设备身份预检、逐记录转换编排、
清单和发布。单条记录的读取与同步见 :mod:`fastumi_data.link7_record`。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from typing import Callable, Dict, List, Optional, Sequence, TextIO

from ament_index_python.packages import (
    PackageNotFoundError,
    get_package_share_directory,
)
import yaml

from fastumi_data.frame_selection import selection_path_for
from fastumi_data.gripper_source import (
    GRIPPER_SOURCE_CHOICES,
    GripperSource,
    SOURCE_AUTO,
    resolve_gripper_source,
)
from fastumi_data.json_report import write_json_report
from fastumi_data.link7_calibration import (
    CalibrationError,
    Link7Calibration,
    check_timestamp_source,
    load_link7_calibration,
    normalize_timestamp_source,
)
from fastumi_data.link7_config import (
    ConfigError,
    ResolvedProcessing,
    resolve_processing,
    resolve_source_fps,
)
from fastumi_data.link7_hdf5 import LINK7_SCHEMA_VERSION
from fastumi_data.link7_record import Link7RecordConverter, precheck_tracker_identity
from fastumi_data.link7_zarr import ZARR_FORMAT, ZarrEpisodeSource, export_link7_zarr


# 输出目录中的固定文件名。
OUTPUT_ZARR_NAME = "dataset.zarr"
MANIFEST_NAME = "manifest.json"
# 缺省 --urdf 时使用的包内部署 URDF（与 dp_infer 默认 expected_urdf_sha256 一致）。
DEFAULT_URDF_NAME = "rm_75.urdf"
EXTRINSIC_MIGRATION = (
    "--extrinsic 已移除：convert_mcap 现在输出 Link7 数据，请使用 --tracker-camera-extrinsic "
    "与 --end-camera-extrinsic（缺省为包内已绑定的默认外参），并提供 --output-dir"
)
# 逐 episode 写入 Zarr 根属性的元数据键。
_ZARR_EPISODE_KEYS = (
    "record_id", "episode_index", "task_name", "session_id", "collection_uuid",
    "nominal_source_fps", "nominal_source_fps_source", "measured_source_fps",
    "fitted_frame_rate_hz", "selection_path", "frame_drop_count", "image_mode",
    "image_timestamp_source", "expected_tracker_serial", "observed_tracker_serial",
    "tracker_serial_source", "tracker_binding_sha256", "tracker_camera_extrinsic_sha256",
    "end_camera_extrinsic_sha256", "tracker_time_offset_ms", "tracker_time_offset_source_s",
    "time_calibration_timestamp_source",
    "time_calibration_conditions",
    "gripper_source",
)
# --jobs 为 0 时自动选择的进程数上限；各进程独立解码，内存与磁盘暂存随进程数线性增长。
MAX_AUTO_JOBS = 8
# 并行时限制每个工作进程的数值库线程数，避免进程数 × 线程数超订 CPU。
_SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True)
class SourceRecord:
    """一条待转换来源：稳定标识、bag 路径及可选的采集记录快照。"""

    record_id: str
    bag_uri: Path
    record_dir: Optional[Path]
    session: Optional[Dict]


def _package_paths(*parts: str) -> list[Path]:
    """返回安装 share 目录与源码树中的候选路径。"""
    candidates = []
    try:
        candidates.append(Path(get_package_share_directory("fastumi_data")).joinpath(*parts))
    except PackageNotFoundError:
        pass
    candidates.append(Path(__file__).resolve().parents[1].joinpath(*parts))
    return candidates


def _first_existing(candidates: list[Path], label: str) -> Path:
    """返回第一个存在的候选文件。"""
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"找不到{label}：{', '.join(map(str, candidates))}")


def default_processing_path() -> Path:
    """包默认 processing.yaml。"""
    return _first_existing(_package_paths("config", "processing.yaml"), "包默认处理配置")


def default_calibration_path(name: str) -> Path:
    """包内安装的默认外参；源码树回退到仓库 ``config/calibration``。"""
    candidates = _package_paths("config", "calibration", name)
    candidates.append(Path(__file__).resolve().parents[4] / "config" / "calibration" / name)
    return _first_existing(candidates, f"默认外参 {name}")


def default_urdf_path() -> Path:
    """包内安装的默认部署 URDF；源码树回退到仓库 ``assets/rm_75.urdf``。"""
    candidates = _package_paths("assets", DEFAULT_URDF_NAME)
    candidates.append(Path(__file__).resolve().parents[4] / "assets" / DEFAULT_URDF_NAME)
    return _first_existing(candidates, f"默认 URDF {DEFAULT_URDF_NAME}")


def _is_bag_dir(path: Path) -> bool:
    """目录是否为 rosbag2 MCAP 目录。"""
    return path.is_dir() and (
        (path / "metadata.yaml").is_file() or any(path.glob("*.mcap"))
    )


def _load_session(record_dir: Path) -> Dict:
    """读取采集记录的 session.yaml。"""
    try:
        session = yaml.safe_load((record_dir / "session.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"无法读取 {record_dir / 'session.yaml'}: {error}") from error
    if not isinstance(session, dict):
        raise ValueError(f"{record_dir / 'session.yaml'} 必须是 YAML 映射")
    return session


def _record_from_dir(record_dir: Path) -> SourceRecord:
    """从含 session.yaml 的采集记录目录构造来源。"""
    session = _load_session(record_dir)
    bag_uri = record_dir / str(session.get("bag_uri") or "raw/bag")
    if not _is_bag_dir(bag_uri):
        raise ValueError(f"记录 {record_dir} 的 bag {bag_uri} 不存在")
    task = str(session.get("task_name") or record_dir.parent.name).strip()
    name = str(session.get("dir_name") or record_dir.name).strip()
    return SourceRecord(f"{task}/{name}", bag_uri.resolve(), record_dir.resolve(), session)


def discover_sources(input_path: Path) -> list[SourceRecord]:
    """发现单个 bag、单条已保存记录或任务目录下的全部来源，按来源路径稳定排序。

    任务目录内跳过 ``.staging`` 和 ``.trash``；目录符号链接不跟随。
    """
    path = Path(input_path).expanduser().resolve()
    if path.is_file() and path.suffix == ".mcap":
        return [SourceRecord(path.stem, path, None, None)]
    if not path.is_dir():
        raise ValueError(f"输入 {path} 不是目录或 .mcap 文件")
    if (path / "session.yaml").is_file():
        return [_record_from_dir(path)]
    if _is_bag_dir(path):
        record_dir = path.parents[1] if len(path.parents) > 1 else None
        if record_dir is not None and (record_dir / "session.yaml").is_file():
            record = _record_from_dir(record_dir)
            if record.bag_uri == path:
                return [record]
        return [SourceRecord(path.name, path, None, None)]
    records: list[SourceRecord] = []
    bags: list[Path] = []
    for directory, dirs, files in os.walk(path, followlinks=False):
        current = Path(directory)
        dirs[:] = sorted(
            name for name in dirs
            if name not in (".staging", ".trash") and not (current / name).is_symlink()
        )
        if "session.yaml" in files:
            records.append(_record_from_dir(current))
            dirs.clear()
        elif _is_bag_dir(current):
            bags.append(current)
            dirs.clear()
    for bag in bags:
        records.append(SourceRecord(bag.relative_to(path).as_posix(), bag, None, None))
    records.sort(key=lambda record: str(record.record_dir or record.bag_uri))
    identifiers = [record.record_id for record in records]
    duplicates = sorted({item for item in identifiers if identifiers.count(item) > 1})
    if duplicates:
        raise ValueError(f"来源记录标识重复: {', '.join(duplicates)}")
    return records


def _snapshot_processing(record: SourceRecord) -> Optional[Path]:
    """返回记录快照中的 processing.yaml 路径。"""
    if record.record_dir is None or not record.session:
        return None
    entry = record.session.get("processing_config")
    relative = entry.get("path") if isinstance(entry, dict) else None
    candidate = record.record_dir / (relative or "calibration_snapshot/processing.yaml")
    return candidate if candidate.is_file() else None


def _sha256_file(path: Path) -> str:
    """返回文件原始字节的 SHA-256。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy_snapshot(source: Path, target_dir: Path, expected_sha: str) -> str:
    """复制快照并核对摘要，返回快照相对路径名。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(source).name
    shutil.copy2(source, target)
    if _sha256_file(target) != expected_sha:
        raise CalibrationError(f"快照 {target} 摘要与源文件不一致")
    return target.name


def _build_parser() -> argparse.ArgumentParser:
    """命令行参数。"""
    parser = argparse.ArgumentParser(
        prog="convert_mcap",
        description=(
            "把 FastUMI MCAP 记录转换为 30 Hz Link7 HDF5，并合并导出 "
            "fastumi-rm75-link7-v1 Zarr"
        ),
    )
    parser.add_argument("input", help="单个 bag、单条已保存记录或任务目录")
    parser.add_argument("--output-dir", required=True, help="新输出目录，必须尚不存在")
    parser.add_argument(
        "--urdf", default="",
        help=f"实际部署使用的 URDF；缺省为包内 {DEFAULT_URDF_NAME}（仅记录摘要与快照，不参与计算）",
    )
    parser.add_argument(
        "--tracker-camera-extrinsic", default="",
        help="T_cam_tracker 外参；缺省为包内已绑定的默认外参",
    )
    parser.add_argument(
        "--end-camera-extrinsic", default="",
        help="T_cam_end 外参；缺省为包内默认外参",
    )
    parser.add_argument(
        "--tracker-serial", default="",
        help="自定义外参缺少设备绑定时声明标定所用 Tracker 序列号；不能覆盖既有绑定",
    )
    parser.add_argument(
        "--source-camera-fps", type=float, default=None,
        help="标称源帧率覆盖值（如 30、29.97、90）；缺省读取各记录快照",
    )
    parser.add_argument("--config", default="", help="可选处理配置；缺省用记录快照或包默认配置")
    parser.add_argument(
        "--gripper-source", choices=GRIPPER_SOURCE_CHOICES, default=SOURCE_AUTO,
        help=(
            "夹爪开度来源：auto（默认，记录含 binarization.yaml 时用 binary，否则用 filtered）、"
            "binary（二值化结果，最近邻取值）、filtered（采集时平滑开度）、raw（未平滑开度）"
        ),
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help=(
            "并行转换记录的进程数；1（默认）在当前进程串行；0 按 CPU 数和记录数自动选择，"
            f"上限 {MAX_AUTO_JOBS}。每条记录仍在单个进程内顺序处理，Zarr 在全部记录完成后统一合并"
        ),
    )
    return parser


def _episode_zarr_metadata(metadata: Dict) -> Dict:
    """挑选写入 Zarr 逐 episode 元数据的字段。"""
    return {key: metadata.get(key) for key in _ZARR_EPISODE_KEYS}


class _Progress:
    """向 stderr 绘制阶段进度条；仅在交互终端绘制，管道和日志里保持静默。

    阶段依次为“转换记录”（按记录数）、“写入图像”（按帧）、“校验数值”（按 episode）。
    ``write`` 打印异常行：先擦掉进度条再输出，随后重绘，因此异常行不会被覆盖。
    """

    _WIDTH = 24

    def __init__(self, total: int, stream: Optional[TextIO] = None) -> None:
        """``total`` 为待转换记录数；``stream`` 缺省为 stderr。"""
        self._label = "转换记录"
        self._total = total
        self._stream = stream if stream is not None else sys.stderr
        self._done = 0
        self._visible = False
        try:
            self._enabled = bool(self._stream.isatty())
        except (AttributeError, ValueError):
            self._enabled = False
        self._draw()

    def _draw(self) -> None:
        """原地重绘进度条。"""
        if not self._enabled:
            return
        filled = self._WIDTH * self._done // max(self._total, 1)
        bar = "#" * filled + "-" * (self._WIDTH - filled)
        self._stream.write(f"\r{self._label} [{bar}] {self._done}/{self._total}")
        self._stream.flush()
        self._visible = True

    def _erase(self) -> None:
        """擦掉当前进度条行。"""
        if self._visible:
            self._stream.write("\r\033[K")
            self._visible = False

    def advance(self) -> None:
        """完成数加一并重绘。"""
        self._done += 1
        self._draw()

    def report(self, label: str, done: int, total: int) -> None:
        """切换到（或更新）某个阶段：``done``/``total`` 为该阶段的进度。"""
        self._label, self._done, self._total = label, done, total
        self._draw()

    def write(self, line: str) -> None:
        """打印一行异常信息，进度条保持在其下方。"""
        self._erase()
        self._stream.write(line + "\n")
        self._draw()
        self._stream.flush()

    def close(self) -> None:
        """擦掉进度条，使后续输出从行首开始。"""
        self._erase()
        self._stream.flush()


@dataclass(frozen=True)
class _RecordTask:
    """一条记录的转换任务：只含可跨进程传递的不可变输入。"""

    record: SourceRecord
    processing: ResolvedProcessing
    fps: float
    fps_source: str
    gripper: GripperSource
    identity: Dict
    calibration: Link7Calibration
    output_dir: Path
    urdf_name: str
    urdf_sha256: str


@dataclass(frozen=True)
class _RecordResult:
    """一条记录的转换结果；``log`` 为需要打印的异常行（episode 被拒绝），由主进程按完成顺序输出。"""

    report: Dict
    accepted: List[ZarrEpisodeSource]
    log: List[str]


class _WorkerFailure(Exception):
    """工作进程内的转换异常；消息已含记录标识和原异常类型。"""


def resolve_jobs(requested: int, record_count: int) -> int:
    """0 表示按 CPU 数和记录数自动选择（不超过 ``MAX_AUTO_JOBS``）；显式值只受记录数限制。"""
    if requested == 0:
        return max(1, min(os.cpu_count() or 1, record_count, MAX_AUTO_JOBS))
    return max(1, min(requested, record_count))


def _convert_record(task: _RecordTask) -> _RecordResult:
    """转换一条记录，写 HDF5、报告与 ``record_report.json``；输出只落在该记录自己的目录。

    正常过程不输出；被拒绝的 episode 行收集在结果的 ``log`` 中。
    """
    rejected_lines: List[str] = []
    record, processing, gripper = task.record, task.processing, task.gripper
    record_dir = task.output_dir / "records" / record.record_id
    work_dir = task.output_dir / ".work" / record.record_id
    session_image = (record.session or {}).get("image") or {}
    base_metadata = dict(task.calibration.metadata())
    base_metadata.update(task.identity)
    base_metadata.update(
        {
            "record_id": record.record_id,
            "source_bag": str(record.bag_uri),
            "source_record_dir": str(record.record_dir or ""),
            "collection_uuid": str((record.session or {}).get("collection_uuid", "")),
            "nominal_source_fps": task.fps,
            "nominal_source_fps_source": task.fps_source,
            "selection_path": selection_path_for(task.fps),
            "image_timestamp_source": str(session_image.get("timestamp_source") or "unrecorded"),
            "image_timestamp_source_normalized": normalize_timestamp_source(
                session_image.get("timestamp_source")
            ),
            "tracker_status_topic": processing.topics["tracker_status"],
            "processing": processing.report(),
            "urdf_sha256": task.urdf_sha256,
            "urdf_file": task.urdf_name,
            "gripper_source": gripper.report(),
        }
    )
    summary = Link7RecordConverter(
        record.record_id, record.bag_uri, record_dir, work_dir,
        processing, task.calibration, task.fps, base_metadata,
        log=lambda _line: None, gripper_source=gripper, reject_log=rejected_lines.append,
    ).convert()
    record_report = {
        "record_id": record.record_id,
        "source_bag": str(record.bag_uri),
        "nominal_source_fps": task.fps,
        "nominal_source_fps_source": task.fps_source,
        "processing": processing.report(),
        "gripper_source": gripper.report(),
        "identity": task.identity,
        "decode": summary.stats,
        "accepted": [
            {key: item[key] for key in ("name", "hdf5", "report", "frames")}
            for item in summary.accepted
        ],
        "rejected": summary.rejected,
    }
    write_json_report(str(record_dir / "record_report.json"), record_report)
    accepted = [
        ZarrEpisodeSource(item["name"], Path(item["hdf5"]), _episode_zarr_metadata(item["metadata"]))
        for item in summary.accepted
    ]
    return _RecordResult(record_report, accepted, rejected_lines)


def _convert_in_worker(task: _RecordTask) -> _RecordResult:
    """工作进程入口：异常转为带记录标识的 ``_WorkerFailure``。"""
    try:
        return _convert_record(task)
    except Exception as error:
        raise _WorkerFailure(
            f"{task.record.record_id}: {type(error).__name__}: {error}"
        ) from None


def _convert_parallel(
    tasks: Sequence[_RecordTask], workers: int, finish: Callable[[int, _RecordResult], None],
) -> None:
    """在 spawn 进程池中并行转换，每条完成即交给 ``finish``。

    首个失败会取消尚未开始的记录并抛出；已在运行的记录会先跑完再退出。
    """
    for name in _SINGLE_THREAD_ENV:
        os.environ.setdefault(name, "1")
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
    )
    try:
        futures = {pool.submit(_convert_in_worker, task): index for index, task in enumerate(tasks)}
        for future in as_completed(futures):
            try:
                result = future.result()
            except BrokenProcessPool as error:
                raise _WorkerFailure(
                    f"工作进程异常退出（可能被系统终止，如内存不足）：{error}"
                ) from None
            finish(futures[future], result)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def run_batch(argv: Optional[list[str]] = None) -> int:
    """执行批量转换，返回进程退出码（0 成功，1 失败）。"""
    arguments_list = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if any(item == "--extrinsic" or item.startswith("--extrinsic=") for item in arguments_list):
        parser.error(EXTRINSIC_MIGRATION)
    arguments = parser.parse_args(arguments_list)
    if arguments.jobs < 0:
        parser.error("--jobs 不能为负数")
    output_dir = Path(arguments.output_dir).expanduser().resolve()
    if output_dir.exists():
        parser.error(f"--output-dir {output_dir} 已存在；请指定新的输出目录")
    if arguments.urdf:
        urdf_path = Path(arguments.urdf).expanduser().resolve()
        if not urdf_path.is_file():
            parser.error(f"--urdf {urdf_path} 不存在")
    else:
        try:
            urdf_path = default_urdf_path().resolve()
        except FileNotFoundError as error:
            parser.error(str(error))
    urdf_sha256 = _sha256_file(urdf_path)
    try:
        tracker_extrinsic = (
            Path(arguments.tracker_camera_extrinsic).expanduser()
            if arguments.tracker_camera_extrinsic
            else default_calibration_path("T_cam_tracker.yaml")
        )
        end_extrinsic = (
            Path(arguments.end_camera_extrinsic).expanduser()
            if arguments.end_camera_extrinsic
            else default_calibration_path("T_cam_end.yaml")
        )
        calibration = load_link7_calibration(
            str(tracker_extrinsic),
            str(end_extrinsic),
            arguments.tracker_serial,
        )
        sources = discover_sources(Path(arguments.input))
        if not sources:
            raise ValueError(f"{arguments.input} 下没有可转换的 MCAP 记录")
        explicit = Path(arguments.config).expanduser() if arguments.config else None
        default_config = default_processing_path()
        plans = []
        notes: list[str] = []
        for record in sources:
            processing = resolve_processing(explicit, _snapshot_processing(record), default_config)
            fps, fps_source = resolve_source_fps(
                arguments.source_camera_fps, record.session, record.record_id
            )
            image = (record.session or {}).get("image") or {}
            try:
                note = check_timestamp_source(calibration, image.get("timestamp_source"))
            except CalibrationError as error:
                raise CalibrationError(f"{record.record_id}: {error}") from error
            if note:
                notes.append(f"{record.record_id} {note}")
            try:
                gripper = resolve_gripper_source(
                    arguments.gripper_source, record.record_dir,
                    processing.topics["gripper_state"],
                )
            except ConfigError as error:
                raise ConfigError(f"{record.record_id}: {error}") from error
            plans.append((record, processing, fps, fps_source, gripper))
        workers = resolve_jobs(arguments.jobs, len(plans))
        zarr_workers = resolve_jobs(arguments.jobs, 1 << 30)
        identities = {}
        for record, processing, *_ in plans:
            identities[record.record_id] = precheck_tracker_identity(
                record.bag_uri, processing.topics["tracker_status"], calibration.tracker_serial
            )
    except (CalibrationError, ConfigError, ValueError, OSError, RuntimeError) as error:
        print(f"转换前检查失败：{error}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True)
    manifest: Dict = {
        "format": ZARR_FORMAT,
        "hdf5_schema": LINK7_SCHEMA_VERSION,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(arguments.input).expanduser().resolve()),
        "output_dir": str(output_dir),
        "arguments": arguments_list,
        "jobs": workers,
        "calibration": calibration.metadata(),
        "urdf": {"path": str(urdf_path), "sha256": urdf_sha256},
        "sources": [],
        "zarr": None,
        "warnings": notes,
    }
    manifest_path = output_dir / MANIFEST_NAME

    def save_manifest() -> None:
        write_json_report(str(manifest_path), manifest)

    accepted_sources: list[ZarrEpisodeSource] = []
    progress = _Progress(len(plans))
    try:
        snapshots = output_dir / "snapshots"
        manifest["calibration"]["snapshots"] = {
            "tracker_camera": _copy_snapshot(
                Path(calibration.tracker_camera.path), snapshots / "calibration",
                calibration.tracker_camera.sha256,
            ),
            "end_camera": _copy_snapshot(
                Path(calibration.end_camera.path), snapshots / "calibration",
                calibration.end_camera.sha256,
            ),
        }
        if calibration.binding is not None:
            manifest["calibration"]["snapshots"]["binding"] = _copy_snapshot(
                Path(calibration.binding.path), snapshots / "calibration",
                calibration.binding.sha256,
            )
        manifest["urdf"]["snapshot"] = "snapshots/urdf/" + _copy_snapshot(
            urdf_path, snapshots / "urdf", urdf_sha256
        )
        save_manifest()
        tasks = [
            _RecordTask(
                record, processing, fps, fps_source, gripper, identities[record.record_id],
                calibration, output_dir, urdf_path.name, urdf_sha256,
            )
            for record, processing, fps, fps_source, gripper in plans
        ]
        results: list[Optional[_RecordResult]] = [None] * len(tasks)

        def finish(index: int, result: _RecordResult) -> None:
            """登记一条记录的结果；清单中的来源始终按发现顺序排列。"""
            results[index] = result
            progress.advance()
            for line in result.log:
                progress.write(line)
            manifest["sources"] = [item.report for item in results if item is not None]
            save_manifest()

        if workers <= 1:
            for index, task in enumerate(tasks):
                finish(index, _convert_record(task))
        else:
            _convert_parallel(tasks, workers, finish)
        accepted_sources = [source for item in results for source in item.accepted]
        shutil.rmtree(output_dir / ".work", ignore_errors=True)
        if not accepted_sources:
            manifest["status"] = "failed"
            manifest["error"] = "没有任何有效 episode"
            save_manifest()
            progress.close()
            print("转换失败：没有任何有效 episode", file=sys.stderr)
            return 1
        manifest["zarr"] = export_link7_zarr(
            accepted_sources,
            output_dir / OUTPUT_ZARR_NAME,
            urdf_sha256,
            {
                "source_manifest": MANIFEST_NAME,
                "hdf5_schema": LINK7_SCHEMA_VERSION,
                "urdf_file": manifest["urdf"]["snapshot"],
                "expected_tracker_serial": calibration.tracker_serial,
                "tracker_camera_extrinsic_sha256": calibration.tracker_camera.sha256,
                "end_camera_extrinsic_sha256": calibration.end_camera.sha256,
            },
            jobs=zarr_workers,
            on_progress=progress.report,
        )
    except Exception as error:  # 硬错误：中止批次并保留诊断产物。
        manifest["status"] = "failed"
        manifest["error"] = (
            str(error) if isinstance(error, _WorkerFailure) else f"{type(error).__name__}: {error}"
        )
        save_manifest()
        progress.close()
        print(f"转换中止：{manifest['error']}", file=sys.stderr)
        return 1
    progress.close()
    if len(accepted_sources) < 2:
        message = "只有 1 条有效 episode：当前训练默认验证划分至少需要两条有效 episode"
        manifest["warnings"].append(message)
    manifest["status"] = "succeeded"
    save_manifest()
    rejected = sum(len(item["rejected"]) for item in manifest["sources"])
    print(
        f"完成：有效 {len(accepted_sources)} / 拒绝 {rejected} episode，"
        f"{manifest['zarr']['frames']} 帧 → {manifest['zarr']['path']}"
    )
    return 0


def main(argv: Optional[list[str]] = None) -> None:
    """``convert_mcap`` 命令行入口。"""
    sys.exit(run_batch(argv))


if __name__ == "__main__":
    main()
