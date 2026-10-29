"""编排 MCAP 分析缓存、可恢复复核和两套 LeRobot 数据集的事务式导出。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile

import h5py
import numpy as np

import binarize_gripper as gripper
from pick_place.core import (
    PARAMETERS, SCHEMA_VERSION, STAGES, binary_values, detect_events, fingerprint,
    initialize_annotation, load_manifest, rebase_qpos, reconcile_annotation,
    save_json, stage_issues, stage_rows,
)


def timestamp() -> str:
    """返回报告使用的 UTC 时间。"""
    return datetime.now(timezone.utc).isoformat()


def runtime_settings(options: dict):
    """延迟加载 ROS 转换依赖，沿用现有默认外参与设备绑定检查。"""
    from fastumi_data import mcap_converter as converter
    from fastumi_data.link7_calibration import load_tcp_calibration

    tracker = Path(options.get("tracker_camera_extrinsic") or converter.default_calibration_path("T_cam_tracker.yaml"))
    tcp = Path(options.get("umi_tcp_extrinsic") or converter.default_calibration_path("T_u_cam_tcp.yaml"))
    urdf = Path(options.get("urdf") or converter.default_urdf_path())
    calibration = load_tcp_calibration(str(tracker), str(tcp), options.get("tracker_serial") or "")
    signature = hashlib.sha256(json.dumps({
        "schema": SCHEMA_VERSION, "parameters": PARAMETERS,
        "calibration": calibration.metadata(),
        "tracker": fingerprint(tracker), "tcp": fingerprint(tcp), "urdf": fingerprint(urdf),
        "config": fingerprint(Path(options["config"])) if options.get("config") else
        fingerprint(converter.default_processing_path()),
        "source_camera_fps": options.get("source_camera_fps"),
    }, sort_keys=True).encode()).hexdigest()
    return converter, calibration, urdf, signature


def analyze_record(source, input_root: Path, output: Path, options: dict, previous=None) -> dict:
    """分析一条录制并缓存同步 HDF5；单条错误记录进清单，不中断其余来源。"""
    source_path = source.record_dir or source.bag_uri
    if not source_path.is_relative_to(input_root):
        source_path = source.bag_uri
    record = {
        "id": source.record_id,
        "source_rel": os.path.relpath(source_path, input_root),
        "bag_rel": os.path.relpath(source.bag_uri, input_root),
        "image_topic": ((source.session or {}).get("image") or {}).get("topic", "/umi_camera/image_raw"),
        "fingerprint": "", "processing_signature": "",
        "events": {"close_ns": None, "open_ns": None, "passed": False, "reasons": []},
        "image_stamps_ns": [], "source_image_index": [], "dropped_frames_before": [],
        "quality_errors": [], "hdf5": None,
    }
    try:
        converter, calibration, urdf, signature = runtime_settings(options)
        record["fingerprint"] = fingerprint(source_path)
        record["processing_signature"] = signature
        if previous and not options.get("force") and all(
            record[key] == previous.get(key) for key in ("fingerprint", "processing_signature")
        ) and all(previous.get(key) and (output / previous[key]).is_file()
                  for key in ("hdf5", "gripper_cache")):
            return deepcopy(previous)
        cache_key = hashlib.sha256((record["source_rel"] + record["fingerprint"] + signature).encode()).hexdigest()
        cache = output / ".cache" / cache_key
        record["cache"] = str(cache.relative_to(output))
        cache.mkdir(parents=True, exist_ok=True)
        processing = converter.resolve_processing(
            Path(options["config"]) if options.get("config") else None,
            converter._snapshot_processing(source), converter.default_processing_path(),
        )
        record["image_topic"] = processing.topics["image"]
        rt = gripper.load_runtime()
        stamps, raw, valid = gripper.read_gripper_samples(rt, source.bag_uri, processing.topics["gripper_state"])
        record["events"] = detect_events(stamps, raw, valid)
        np.savez(cache / "gripper.npz", stamps_ns=stamps, raw=raw, valid=valid)
        record["gripper_cache"] = str((cache / "gripper.npz").relative_to(output))
        fps, fps_source = converter.resolve_source_fps(options.get("source_camera_fps"), source.session, source.record_id)
        converter.check_timestamp_source(calibration, ((source.session or {}).get("image") or {}).get("timestamp_source"))
        identity = converter.precheck_tracker_identity(source.bag_uri, processing.topics["tracker_status"], calibration.tracker_serial)
        continuous = converter.resolve_gripper_source("raw", source.record_dir, processing.topics["gripper_state"])
        converted = cache / "converted"
        # 缓存目录只属于本次内容指纹；不删除原始数据或人工标注。
        if converted.exists():
            shutil.rmtree(converted)
        metadata = dict(calibration.metadata())
        metadata.update(identity)
        metadata.update(record_id=source.record_id, nominal_source_fps=fps,
                        nominal_source_fps_source=fps_source, processing=processing.report(),
                        gripper_source=continuous.report(), urdf_file=urdf.name,
                        urdf_sha256=converter._sha256_file(urdf))
        summary = converter.Link7RecordConverter(
            source.record_id, source.bag_uri, converted, cache / "work", processing,
            calibration, fps, metadata, log=lambda _: None, gripper_source=continuous,
            image_size=224,
        ).convert()
        record["conversion_rejections"] = summary.rejected
        if len(summary.accepted) != 1 or summary.rejected:
            record["quality_errors"].append("原录制必须有且仅有一个通过同步检查的完整 episode")
            record["quality_errors"].extend(
                f"同步拒绝: {','.join(item['rejections'])}" for item in summary.rejected
            )
        if len(summary.accepted) == 1:
            item = summary.accepted[0]
            path = Path(item["hdf5"])
            record["hdf5"] = str(path.relative_to(output))
            record["source_episode"] = item["name"]
            with h5py.File(path, "r") as handle:
                record["image_stamps_ns"] = handle["observations/image_timestamp_ns"][:].tolist()
                record["source_image_index"] = handle["observations/source_image_index"][:].tolist()
                record["dropped_frames_before"] = handle["observations/dropped_frames_before"][:].tolist()
            inside = (stamps >= record["image_stamps_ns"][0]) & (stamps <= record["image_stamps_ns"][-1])
            record["events"] = detect_events(stamps[inside], raw[inside], valid[inside])
        if fingerprint(source_path) != record["fingerprint"]:
            record["quality_errors"].append("读取过程中源文件发生变化，请重新分析")
        shutil.rmtree(cache / "work", ignore_errors=True)
    except Exception as error:
        record["quality_errors"].append(f"{type(error).__name__}: {error}")
    initialize_annotation(record)
    return reconcile_annotation(record, previous)


def write_report(output: Path, manifest: dict) -> dict:
    """汇总所有记录和阶段状态，同时保留逐条原因。"""
    counts = {stage: {key: 0 for key in ("auto", "accepted", "review", "excluded")} for stage in STAGES}
    entries = []
    for record in manifest["records"]:
        for stage in STAGES:
            counts[stage][record["annotation"]["stages"][stage]["status"]] += 1
        entries.append({"id": record["id"], "deleted": record["annotation"].get("deleted", False),
                        "quality_errors": record["quality_errors"],
                        "event_reasons": record["events"]["reasons"],
                        "stages": record["annotation"]["stages"]})
    report = {"updated_at": timestamp(), "records": len(entries), "counts": counts, "details": entries}
    save_json(output / "report.json", report)
    return report


def analyze(args) -> int:
    """遍历输入并逐条原子落盘；并行工作进程各写独立缓存目录。"""
    from fastumi_data.mcap_converter import discover_sources

    source_input, output = validate_paths(args.input, args.output)
    input_root = source_input if source_input.is_dir() else source_input.parent
    sources = discover_sources(source_input)
    if not sources:
        raise ValueError("没有发现可处理的录制")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "segments.json"
    old = load_manifest(path) if path.exists() else None
    if old and old["input_root"] != str(input_root):
        raise ValueError("该输出目录属于另一个输入目录，请指定新的 --output")
    previous = {record["id"]: record for record in (old or {}).get("records", [])}
    manifest = {"schema_version": SCHEMA_VERSION, "input_root": str(input_root),
                "created_at": (old or {}).get("created_at", timestamp()), "updated_at": timestamp(),
                "parameters": PARAMETERS, "options": vars(args), "records": [], "complete": False}
    results = {source.record_id: previous[source.record_id] for source in sources if source.record_id in previous}
    options = vars(args).copy()

    def finish(record: dict, number: int) -> None:
        """按来源稳定顺序保存进度，避免中断丢失已完成标注。"""
        results[record["id"]] = record
        manifest["records"] = [results[source.record_id] for source in sources if source.record_id in results]
        save_json(path, manifest)
        states = ", ".join(f"{stage}={record['annotation']['stages'][stage]['status']}" for stage in STAGES)
        print(f"[{number}/{len(sources)}] {record['id']}: {states}", flush=True)

    if args.jobs == 1:
        for number, source in enumerate(sources, 1):
            finish(analyze_record(source, input_root, output, options, previous.get(source.record_id)), number)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(analyze_record, source, input_root, output, options,
                                   previous.get(source.record_id)) for source in sources]
            for number, future in enumerate(as_completed(futures), 1):
                finish(future.result(), number)
    manifest["complete"] = True
    save_json(path, manifest)
    report = write_report(output, manifest)
    print(json.dumps({"records": report["records"], "counts": report["counts"]}, ensure_ascii=False))
    return 0


def prepare_stage(record: dict, stage: str, mode: str, output: Path, temporary: Path) -> dict:
    """对阶段独立归零并补齐短缺口，写出现有 LeRobot 写入器的 NPZ 契约。"""
    from fastumi_data.link7_lerobot import fill_rows

    issues = stage_issues(record, stage, mode)
    if issues:
        raise ValueError("；".join(issues))
    start, stop = stage_rows(record, stage)
    with h5py.File(output / record["hdf5"], "r") as handle:
        observed = handle["observations/image_timestamp_ns"][:]
        if observed.tolist() != record["image_stamps_ns"]:
            raise ValueError("同步缓存与标注时间轴不一致，请重新分析")
        qpos = rebase_qpos(handle["observations/qpos"][start:stop])
        if not np.allclose(handle["action"][start:stop], handle["observations/qpos"][start:stop]):
            raise ValueError("现有 state/action 同时刻契约不成立")
        if mode == "binary":
            qpos[:, 7] = binary_values(observed[start:stop], record["annotation"]["close_ns"], record["annotation"]["open_ns"])
        filled = fill_rows(qpos, np.asarray(record["dropped_frames_before"][start:stop]),
                           (0, stop - start), mode == "binary")
        if not np.isfinite(filled.state).all() or np.any((filled.state[:, 6] < 0) | (filled.state[:, 6] > 1)):
            raise ValueError("导出位姿或夹爪值不合法")
        images = handle["observations/images/front"][start:stop][filled.source_row]
        source_rows = filled.source_row + start
        source_indices = handle["observations/source_image_index"][:][source_rows]
        metadata = json.loads(handle.attrs["metadata_json"])
    temporary.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(record["id"].encode()).hexdigest()[:24] + f"_{stage}"
    npz = temporary / (name + ".npz")
    state = filled.state.astype(np.float32)
    np.savez(npz, images=images, state=state, action=state.copy(), source_row=source_rows,
             filled=filled.filled, source_image_index=source_indices, nominal_frame=filled.nominal_frame)
    return {
        "name": f"{record['id']}/{stage}", "npz": str(npz), "frames": len(state), "task": stage,
        "record_id": record["id"], "source_record": record["source_rel"],
        "source_episode": record.get("source_episode", record["id"]), "source_hdf5": record["hdf5"],
        "source_fingerprint": record["fingerprint"], "processing_signature": record["processing_signature"],
        "source_rows": [start, stop], "source_image_index": source_indices.tolist(),
        "source_image_timestamp_ns": observed[source_rows].tolist(),
        "segment_index": 0, "split": True, "stage": stage, "gripper_mode": mode,
        "filled_frames": np.flatnonzero(filled.filled).tolist(), "nominal_frame_start": 0,
        "annotation": deepcopy(record["annotation"]),
        "calibration": (
            {key: value for key, value in metadata.items()
             if key.endswith("sha256") or key.endswith("serial")}
            | {"umi_camera_from_tcp": metadata.get("T_Cu_TCP")}
        ),
    }


def publish_directory(prepared: Path, target: Path, overwrite: bool) -> None:
    """整体替换两阶段目录；发布失败恢复原目录，默认拒绝覆盖。"""
    backup = prepared.parent / "previous_dataset"
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} 已存在；需显式指定 --overwrite")
        os.rename(target, backup)
    try:
        os.rename(prepared, target)
    except BaseException:
        if backup.exists():
            os.rename(backup, target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def export(args) -> int:
    """检查来源未变，生成两套临时数据并经写入器回读后整体发布。"""
    from fastumi_data.lerobot_export import export_lerobot, preflight_lerobot, resolve_lerobot_python

    _, output = validate_paths(args.input, args.output)
    manifest = load_manifest(output / "segments.json")
    if not manifest.get("complete"):
        raise ValueError("分析尚未完成，请先继续 analyze")
    target = output / "lerobot"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"{target} 已存在；需显式指定 --overwrite")
    python = resolve_lerobot_python(args.lerobot_python)
    preflight_lerobot(python)
    temporary = Path(tempfile.mkdtemp(prefix=".export-", dir=output))
    prepared = temporary / "lerobot"
    prepared.mkdir()
    segments = {stage: [] for stage in STAGES}
    skipped = []
    input_root = Path(manifest["input_root"])
    try:
        for number, record in enumerate(manifest["records"], 1):
            if record["annotation"].get("deleted"):
                continue
            selected = [stage for stage in STAGES if record["annotation"]["stages"][stage]["status"] in ("auto", "accepted")]
            if not selected:
                continue
            print(f"准备阶段 [{number}/{len(manifest['records'])}] {record['id']}", flush=True)
            if fingerprint(input_root / record["source_rel"]) != record["fingerprint"]:
                for value in record["annotation"]["stages"].values():
                    value.update(status="review", reason="源文件变化，需要重新 analyze")
                save_json(output / "segments.json", manifest)
                raise ValueError(f"{record['id']} 源文件已变化，拒绝使用旧标注")
            for stage in selected:
                try:
                    segments[stage].append(prepare_stage(record, stage, args.gripper_mode, output, temporary / "npz"))
                except (ValueError, OSError, KeyError) as error:
                    skipped.append({"id": record["id"], "stage": stage, "reason": str(error)})
                    record["annotation"]["stages"][stage].update(status="review", reason=str(error))
        save_json(output / "segments.json", manifest)
        write_report(output, manifest)
        if not any(segments.values()):
            raise ValueError("没有可导出的合格阶段，请查看待核对清单")
        results = {}
        for stage in STAGES:
            if not segments[stage]:
                results[stage] = {"episodes": 0, "reason": "没有合格阶段，未创建空 LeRobot 数据集"}
                continue
            common = segments[stage][0]["calibration"]
            if any(item["calibration"] != common for item in segments[stage]):
                raise ValueError("同一数据集的标定契约不一致，请分别导出")
            results[stage] = export_lerobot(
                segments[stage], prepared / stage, temporary / stage, f"fastumi/{stage}", python,
                extra_metadata=common | {
                    "stage": stage, "gripper_mode": args.gripper_mode,
                    "source_manifest": "../../segments.json",
                    "frame_gap_policy": "fill_up_to_2_reject_long_gaps",
                    "split_parameters": PARAMETERS,
                },
                on_progress=lambda label, done, total: print(f"{label}: {done}/{total}", flush=True),
            )
            results[stage]["path"] = str(target / stage)
        report = {"created_at": timestamp(), "gripper_mode": args.gripper_mode, "datasets": results,
                  "skipped": skipped, "pending_or_excluded": [
                      {"id": record["id"], "stages": record["annotation"]["stages"]}
                      for record in manifest["records"] if any(
                          value["status"] in ("review", "excluded") for value in record["annotation"]["stages"].values())]}
        save_json(prepared / "export_report.json", report)
        publish_directory(prepared, target, args.overwrite)
        save_json(output / "export_report.json", report)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    except BaseException as error:
        save_json(temporary / "failure.json", {"error": str(error), "skipped": skipped})
        raise
    else:
        shutil.rmtree(temporary)
    return 0


def validate_paths(source: str, destination: str) -> tuple[Path, Path]:
    """禁止输出覆盖输入或位于输入内，避免后续遍历把缓存误认为原录制。"""
    source, destination = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("输入与输出必须是互不包含的独立路径")
    return source, destination


def build_parser() -> argparse.ArgumentParser:
    """构造三个子命令；帮助信息无需 ROS 运行时。"""
    parser = argparse.ArgumentParser(description="按首次合拢与随后打开的时间中点分割 pick/place")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("analyze", "review", "export"):
        command = commands.add_parser(name)
        command.add_argument("--input", default="dataset/raw_data")
        command.add_argument("--output", default="dataset/pick_place")
        if name == "analyze":
            command.add_argument("--jobs", type=int, default=4)
            command.add_argument("--force", action="store_true", help="重新分析缓存，保留未变化的人工标注")
            for flag in ("config", "tracker-camera-extrinsic", "umi-tcp-extrinsic", "tracker-serial", "urdf"):
                command.add_argument("--" + flag, default="")
            command.add_argument("--source-camera-fps", type=float)
        if name == "export":
            command.add_argument("--gripper-mode", choices=("binary", "continuous"), default="binary")
            command.add_argument("--lerobot-python", default="")
            command.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    """运行命令并打印可操作错误；不向用户展示无关 Python 堆栈。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "jobs", 1) < 1:
        parser.error("--jobs 必须为正整数")
    try:
        import fcntl

        _, output = validate_paths(args.input, args.output)
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("此输出目录正在分析、复核或导出，请结束该操作后重试") from error
            if args.command == "analyze":
                return analyze(args)
            if args.command == "export":
                return export(args)
            from pick_place.review import run_review
            return run_review(args)
    except Exception as error:
        print(f"处理失败：{error}", file=sys.stderr)
        return 2
