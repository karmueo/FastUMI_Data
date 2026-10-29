"""分段与标注的纯逻辑：时间为整数纳秒，位姿为米及 xyzw 四元数。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import binarize_gripper as gripper


SCHEMA_VERSION = 1
STAGES = ("pick", "place")
MIN_FRAMES = 16
MAX_FILL_GAP = 2
PARAMETERS = {
    "smooth_half_window_ns": gripper.SMOOTH_HALF_WINDOW_NS,
    "max_missing_ns": gripper.MAX_MISSING_NS,
    "min_plateau_ns": gripper.MIN_PLATEAU_NS,
    "min_state_ns": gripper.MIN_STATE_NS,
    "min_plateau_diff": gripper.MIN_PLATEAU_DIFF,
    "max_fill_gap": MAX_FILL_GAP,
    "minimum_frames": MIN_FRAMES,
    "split_rule": "first_close_open_time_midpoint_earlier_tie_shared_frame",
}


def save_json(path: Path, document: dict) -> None:
    """原子保存严格 JSON；写入中断时保留上一版清单。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(gripper.to_jsonable(document), ensure_ascii=False,
                                        indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(path: Path) -> dict:
    """读取清单并拒绝不兼容版本，避免误用旧标注。"""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("分段清单版本不兼容，请使用匹配的工具")
    return document


def fingerprint(path: Path) -> str:
    """流式计算录制文件内容及相对路径的 SHA-256，不依赖修改时间。"""
    path = Path(path)
    files = [path] if path.is_file() else sorted(
        item for item in path.rglob("*") if item.is_file()
        and item.suffix.lower() in (".mcap", ".yaml", ".yml", ".json")
    )
    digest = hashlib.sha256()
    for item in files:
        digest.update((item.name if path.is_file() else item.relative_to(path).as_posix()).encode())
        digest.update(b"\0")
        digest.update(str(item.stat().st_size).encode() + b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def midpoint_index(stamps_ns, close_ns: int, open_ns: int) -> int:
    """用整数比较到时间中点的距离；等距选择较早帧，不损失 Unix 纳秒精度。"""
    stamps = [int(value) for value in stamps_ns]
    if not stamps or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("有效图像时间必须非空且严格递增")
    if close_ns >= open_ns:
        raise ValueError("合拢必须早于随后打开")
    total = int(close_ns) + int(open_ns)
    return min(range(len(stamps)), key=lambda i: (abs(2 * stamps[i] - total), stamps[i]))


def detect_events(stamps, raw, valid) -> dict:
    """复用完整单周期检测；不把多周期的第一段猜测为可靠事件。"""
    result = gripper.binarize_openness(stamps, raw, valid)
    summary = gripper.build_summary("", result, "passed" if result.passed else "review")
    return {
        "close_ns": summary.get("close_stamp_ns"),
        "open_ns": summary.get("open_stamp_ns"),
        "reasons": result.reasons,
        "passed": result.passed,
    }


def initialize_annotation(record: dict) -> None:
    """建立原始首尾和自动中点候选；失败记录不伪造中点。"""
    stamps = record.get("image_stamps_ns", [])
    events = record["events"]
    split = None
    if stamps and events["close_ns"] is not None and events["open_ns"] is not None:
        split = midpoint_index(stamps, events["close_ns"], events["open_ns"])
    record["annotation"] = {
        "start": 0, "split": split, "end": len(stamps) - 1,
        "close_ns": events["close_ns"], "open_ns": events["open_ns"],
        "split_source": "automatic_midpoint", "manual": False,
        "stages": {stage: {"status": "review", "reason": ""} for stage in STAGES},
    }
    for stage in STAGES:
        errors = stage_issues(record, stage)
        if events["passed"] and not errors:
            record["annotation"]["stages"][stage]["status"] = "auto"
        record["annotation"]["stages"][stage]["reason"] = "; ".join(errors)


def stage_rows(record: dict, stage: str) -> tuple[int, int]:
    """返回阶段源行区间 [start, stop)，两个阶段共享 split 行。"""
    if stage not in STAGES:
        raise ValueError(f"未知阶段 {stage}")
    annotation = record["annotation"]
    start, split, end = (annotation[key] for key in ("start", "split", "end"))
    if any(type(value) is not int for value in (start, split, end)):
        raise ValueError("尚未确定有效切点")
    if not 0 <= start <= split <= end < len(record["image_stamps_ns"]):
        raise ValueError("阶段边界超出范围或顺序错误")
    return (start, split + 1) if stage == "pick" else (split, end + 1)


def stage_issues(record: dict, stage: str, mode: str = "binary") -> list[str]:
    """校验不可由人工确认绕过的同步、长度、边界及夹爪约束。"""
    errors = list(record.get("quality_errors", []))
    try:
        start, stop = stage_rows(record, stage)
    except (ValueError, KeyError) as error:
        return errors + [str(error)]
    dropped = record.get("dropped_frames_before", [])
    if len(dropped) != len(record["image_stamps_ns"]):
        return errors + ["缺少同步缺帧信息"]
    gaps = dropped[start + 1:stop]
    if any(value > MAX_FILL_GAP or value < 0 for value in gaps):
        errors.append("阶段内存在不可修复的同步缺口")
    if stop - start + sum(gaps) < MIN_FRAMES:
        errors.append(f"阶段不足 {MIN_FRAMES} 帧")
    annotation = record["annotation"]
    close, opened = annotation["close_ns"], annotation["open_ns"]
    if close is not None and opened is not None and close >= opened:
        errors.append("合拢时间必须早于打开时间")
    event = close if stage == "pick" else opened
    if event is not None and not record["image_stamps_ns"][start] <= event <= record["image_stamps_ns"][stop - 1]:
        errors.append("阶段范围未包含对应的合拢/打开事件")
    if mode == "binary":
        if close is None or (stage == "place" and opened is None):
            errors.append("二值模式缺少阶段所需的夹爪事件")
    return errors


def update_annotation(record: dict, *, events=None, bounds=None, decisions=None) -> None:
    """事务式修订事件、边界和阶段决策；事件变化先重算中点，手动切点后覆盖。"""
    edited = deepcopy(record)
    annotation = edited["annotation"]
    if events:
        annotation.update(events)
        close, opened = annotation["close_ns"], annotation["open_ns"]
        annotation["split"] = (
            midpoint_index(edited["image_stamps_ns"], close, opened)
            if close is not None and opened is not None else None
        )
        annotation["split_source"] = "edited_events_midpoint"
    if bounds:
        annotation.update(bounds)
        if "split" in bounds:
            annotation["split_source"] = "manual_override"
    if events or bounds:
        for state in annotation["stages"].values():
            if state["status"] != "excluded":
                state.update(status="review", reason="边界或事件已修改，请重新确认")
    for stage, decision in (decisions or {}).items():
        if stage not in STAGES or decision["status"] not in ("accepted", "excluded", "review"):
            raise ValueError("阶段决策无效")
        if decision["status"] == "accepted":
            # 连续模式也可确认缺失夹爪事件的片段，二值导出时另行严格检查。
            errors = stage_issues(edited, stage, "continuous")
            if errors:
                raise ValueError("；".join(errors))
        annotation["stages"][stage] = dict(decision)
    annotation["manual"] = True
    record["annotation"] = annotation


def delete_annotation(record: dict) -> None:
    """逻辑删除整条记录的两个阶段，保存删除前标注供恢复，不修改录制文件。"""
    annotation = record["annotation"]
    if annotation.get("deleted"):
        return
    previous = deepcopy(annotation)
    annotation.update(deleted=True, before_delete=previous, manual=True)
    for state in annotation["stages"].values():
        state.update(status="excluded", reason="已删除当前记录")


def restore_annotation(record: dict) -> None:
    """恢复逻辑删除前的标注；源变化后的快照已由重分析置为待核对。"""
    if record["annotation"].get("deleted"):
        record["annotation"] = deepcopy(record["annotation"]["before_delete"])


def reconcile_annotation(record: dict, previous: dict | None) -> dict:
    """保留未变化录制的标注；源或处理配置变化时存档并要求重新确认。"""
    if not previous:
        return record
    same = all(record.get(key) == previous.get(key) for key in ("fingerprint", "processing_signature"))
    if same and record.get("image_stamps_ns") == previous.get("image_stamps_ns"):
        record["annotation"] = deepcopy(previous["annotation"])
        if previous.get("invalidated_annotation"):
            record["invalidated_annotation"] = previous["invalidated_annotation"]
    else:
        record["invalidated_annotation"] = deepcopy(previous["annotation"])
        for value in record["annotation"]["stages"].values():
            value.update(status="review", reason="源数据或处理配置变化，旧标注已失效")
        if previous["annotation"].get("deleted"):
            delete_annotation(record)
    return record


def rebase_qpos(qpos: np.ndarray) -> np.ndarray:
    """将 (N,8) 的 xyz(m)、xyzw、夹爪轨迹变换到当前片段首帧 TCP 系。"""
    result = np.asarray(qpos, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 8 or not len(result) or not np.isfinite(result).all():
        raise ValueError("位姿必须是非空有限 (N,8) 数组")
    rotation = Rotation.from_quat(result[:, 3:7])
    inverse = rotation[0].inv()
    result[:, :3] = inverse.apply(result[:, :3] - result[0, :3])
    result[:, 3:7] = (inverse * rotation).as_quat()
    return result


def binary_values(stamps, close_ns: int, open_ns: int | None) -> np.ndarray:
    """按已确认事件生成开=1、闭=0；缺少释放时闭合持续至片段末尾。"""
    stamps = np.asarray(stamps, dtype=np.int64)
    closed = stamps >= close_ns
    if open_ns is not None:
        if open_ns <= close_ns:
            raise ValueError("打开事件必须晚于合拢")
        closed &= stamps < open_ns
    return (~closed).astype(np.float64)
