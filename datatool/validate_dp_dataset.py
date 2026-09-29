"""只读验证 RM75 关节或 Link7 UMI 训练用 Zarr 数据集。

作用：
    检查数据集的根属性、episode 边界、数组结构、chunk、数值和训练窗口，
    汇总不符合所选训练格式的问题。

输入：
    --input 为单个 Zarr 目录或包含 Zarr 目录的父目录；递归查找时不跟随符号链接。
    --type 必选，可选 rm75_joint 或 rm75_umi。--action-horizon 默认 16，
    --seed 默认 42，--val-ratio 默认 0.1，三者用于计算训练和验证窗口。
    输入路径按当前工作目录解析，数据集须包含 meta 和 data 中的必需数组。

输出：
    在标准输出打印每个数据集的检查摘要；--report 指定时，将完整结果写入
    该 JSON 路径，并在需要时创建父目录。报告路径不能位于被检查的 Zarr 目录内。
    返回码 0 表示全部通过，1 表示检查不通过或未找到数据集，2 表示参数或运行错误。

使用方法：
    在仓库根目录、具备 numpy 和 zarr 的 Python 环境中运行，例如：
    .venv/bin/python datatool/validate_dp_dataset.py --input dataset/vr_target/target.zarr --type rm75_joint
"""

import argparse
import itertools
import json
import os
import re
import sys
from pathlib import Path

try:
    import numpy as np
    import zarr
except ImportError as exc:
    print(f"缺少验证依赖：{exc}。请使用根目录 .venv 或安装 numpy、zarr。", file=sys.stderr)
    sys.exit(2)


# 两种训练数据类型允许的 Zarr 根属性 format 值。
FORMATS = {
    "rm75_joint": {"rm75-joint-image-v1", "rm75-joint-image-v2"},
    "rm75_umi": {"rm75-umi-pose-v1", "rm75-umi-pose-v2"},
}
# 数组形状只列出帧维之后的维度，帧数由 meta/episode_ends 确定。
COMMON = {"camera0_rgb": (224, 224, 3), "timestamp": (), "source_image_index": ()}
JOINT = {"robot0_joint_pos": (7,), "robot0_gripper_position": (1,), "action": (8,)}
UMI = {"robot0_eef_pos": (3,), "robot0_eef_rot_axis_angle": (3,),
       "robot0_gripper_width": (1,), "robot0_demo_start_pose": (6,), "action": (7,)}
JOINT_NAMES = [f"joint{index}" for index in range(1, 8)]
HEX_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
# 每类问题在报告中保留的定位样例上限。
MAX_EXAMPLES = 3


def discover_datasets(input_path):
    """从目录递归查找 Zarr 根目录，返回路径列表。

    不跟随目录符号链接，也不进入已发现的 Zarr；输入不是目录时抛出 ValueError。
    """
    input_path = Path(input_path)
    if not input_path.is_dir():
        raise ValueError(f"输入不是目录：{input_path}")
    found = []
    for directory, dirs, files in os.walk(input_path, followlinks=False):
        dirs[:] = sorted(name for name in dirs
                         if not (Path(directory) / name).is_symlink())
        if ".zgroup" in files or ".zarr.json" in files:
            found.append(Path(directory))
            dirs.clear()
    return found


class Findings:
    """按严重级别、错误码、字段和期望值汇总问题及少量定位样例。"""

    def __init__(self):
        """初始化按问题标识索引的汇总记录。"""
        self._items = {}

    def add(self, severity, code, field, expected, actual, episode=None, frame=None,
            count=1):
        """累计一类问题，并记录最多 MAX_EXAMPLES 个 episode 和帧样例。

        severity 为 errors 或 warnings；count 可一次累计多个同类问题。
        """
        key = (severity, code, field, str(expected))
        if key not in self._items:
            self._items[key] = {"code": code, "field": field, "expected": expected,
                                "count": 0, "examples": []}
        item = self._items[key]
        item["count"] += int(count)
        if len(item["examples"]) < MAX_EXAMPLES:
            item["examples"].append({"episode": episode, "frame": frame,
                                     "actual": str(actual)})

    def report(self):
        """返回按 errors 和 warnings 分组的问题列表。"""
        return {
            severity: [value for key, value in self._items.items() if key[0] == severity]
            for severity in ("errors", "warnings")
        }


def check_equal(attrs, name, expected, findings):
    """将根属性 name 与 expected 比较，并将缺失或不匹配项写入 findings。"""
    actual = attrs.get(name)
    if actual != expected:
        findings.add("errors", "ATTRIBUTE_MISMATCH", name, expected, actual)


def check_hash(attrs, findings):
    """检查根属性 urdf_sha256 是否为 64 位十六进制字符串。"""
    value = attrs.get("urdf_sha256")
    if not isinstance(value, str) or not HEX_SHA256.fullmatch(value):
        findings.add("errors", "ATTRIBUTE_INVALID", "urdf_sha256", "64 位十六进制 SHA-256", value)


def check_contract(attrs, kind, findings):
    """校验 kind 对应的根属性，并返回可用的关节限位或 None。

    kind 为 rm75_joint 或 rm75_umi；发现不匹配时向 findings 追加错误。
    关节限位仅在 v2 格式的相关属性全部有效时返回，形状均为 (7,)。
    """
    format_name = attrs.get("format")
    if format_name not in FORMATS[kind]:
        findings.add("errors", "FORMAT_MISMATCH", "format", sorted(FORMATS[kind]), format_name)
    check_equal(attrs, "complete", True, findings)
    check_equal(attrs, "frequency", 30, findings)
    check_equal(attrs, "image_size", 224, findings)
    check_equal(attrs, "action_layout", "joint8" if kind == "rm75_joint" else "pose10", findings)
    limits = None
    if kind == "rm75_umi":
        for key, expected in (("stored_action_layout", "xyz_rotvec_gripper"),
                              ("base_frame", "base_link"), ("end_frame", "Link7"),
                              ("position_unit", "m"), ("rotation_unit", "rad"),
                              ("gripper_representation", "normalized_0_1"),
                              ("joint_names", JOINT_NAMES)):
            check_equal(attrs, key, expected, findings)
        offset = attrs.get("tool_offset")
        try:
            valid_offset = np.asarray(offset, dtype=np.float64).shape == (4, 4) and np.allclose(
                offset, np.eye(4), atol=1e-10, rtol=0)
        except (TypeError, ValueError):
            valid_offset = False
        if not valid_offset:
            findings.add("errors", "ATTRIBUTE_INVALID", "tool_offset", "4×4 单位矩阵", offset)
        check_hash(attrs, findings)
    if format_name in ("rm75-joint-image-v2", "rm75-umi-pose-v2"):
        check_equal(attrs, "normalization_contract", "urdf-joint-limits-v1", findings)
        check_equal(attrs, "joint_names", JOINT_NAMES, findings)
        check_hash(attrs, findings)
        try:
            lower = np.asarray(attrs.get("joint_lower"), dtype=np.float64)
            upper = np.asarray(attrs.get("joint_upper"), dtype=np.float64)
            if lower.shape != (7,) or upper.shape != (7,) or not np.isfinite(lower).all() \
                    or not np.isfinite(upper).all() or not np.all(lower < upper):
                raise ValueError("invalid joint limits")
            gripper_lower = float(attrs.get("gripper_lower"))
            gripper_upper = float(attrs.get("gripper_upper"))
            if not np.isfinite([gripper_lower, gripper_upper]).all() \
                    or not (gripper_lower == 0 and gripper_upper == 1):
                raise ValueError("invalid gripper limits")
            limits = (lower, upper)
        except (TypeError, ValueError):
            findings.add("errors", "ATTRIBUTE_INVALID", "joint_lower/joint_upper/gripper_lower/gripper_upper",
                         "7 个有限且有序的关节限位、夹爪范围 [0,1]",
                         [attrs.get("joint_lower"), attrs.get("joint_upper"),
                          attrs.get("gripper_lower"), attrs.get("gripper_upper")])
    return limits


def read_meta(root, findings):
    """读取 episode 边界和名称，返回 (ends, names)。

    边界不可读取或无效时返回 (None, None) 并记录错误；名称无效时记录错误，
    但仍返回已读取的边界和名称。
    """
    try:
        ends = np.asarray(root["meta/episode_ends"][:])
        names = np.asarray(root["meta/episode_names"][:])
    except (KeyError, OSError, ValueError, TypeError) as exc:
        findings.add("errors", "META_READ_FAILED", "meta", "可读取的 episode_ends 与 episode_names", exc)
        return None, None
    if ends.ndim != 1 or len(ends) == 0 or ends.dtype.kind not in "iu" \
            or np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
        findings.add("errors", "EPISODE_ENDS_INVALID", "meta/episode_ends",
                     "非空、正整数、严格递增的一维数组", ends[:5].tolist())
        return None, None
    if names.ndim != 1 or len(names) != len(ends) or \
            any(not str(name).strip() for name in names) or \
            len(set(map(str, names))) != len(names):
        findings.add("errors", "EPISODE_NAMES_INVALID", "meta/episode_names",
                     f"{len(ends)} 个非空且唯一的名称", names[:5].tolist())
    return ends.astype(np.int64), [str(name) for name in names]


def describe_arrays(root, kind, count, findings):
    """检查必需数组的类型、形状和 dtype，返回 (arrays, shapes)。

    count 是总帧数；arrays 只包含形状符合预期、可继续扫描的数组，
    shapes 记录已打开数组的实际形状，发现的问题写入 findings。
    """
    required = dict(COMMON)
    required.update(JOINT if kind == "rm75_joint" else UMI)
    arrays = {}
    shapes = {}
    for name, tail in required.items():
        path = f"data/{name}"
        try:
            array = root[path]
        except KeyError:
            findings.add("errors", "ARRAY_MISSING", path, (count,) + tail, "缺失")
            continue
        except Exception as exc:
            findings.add("errors", "ARRAY_OPEN_FAILED", path, "可读取的 Zarr 数组",
                         f"{type(exc).__name__}: {exc}")
            continue
        if not isinstance(array, zarr.Array):
            findings.add("errors", "ARRAY_INVALID", path, "Zarr 数组", "非数组")
            continue
        shapes[name] = list(array.shape)
        expected_dtype = np.dtype("uint8" if name == "camera0_rgb" else
                                  "float64" if name == "timestamp" else
                                  "int64" if name == "source_image_index" else "float32")
        dtype_ok = (array.dtype == expected_dtype if name not in
                    ("timestamp", "source_image_index") else
                    array.dtype.kind == expected_dtype.kind)
        if not dtype_ok:
            findings.add("errors", "DTYPE_MISMATCH", path, str(expected_dtype), str(array.dtype))
        if array.shape != (count,) + tail:
            findings.add("errors", "SHAPE_MISMATCH", path, (count,) + tail, array.shape)
        if array.ndim == len(tail) + 1 and array.shape[1:] == tail and array.shape[0] == count:
            arrays[name] = array
    return arrays, shapes


def check_chunks(array, field, dataset, findings):
    """检查 dataset/data/field 下每个预期 chunk 的实体文件。

    缺失文件写入 findings；chunk 的读取与解码由 scan_array 检查。
    """
    counts = [(size + chunk - 1) // chunk for size, chunk in zip(array.shape, array.chunks)]
    separator = array._dimension_separator or "."
    missing = 0
    for indices in itertools.product(*(range(size) for size in counts)):
        relative = separator.join(map(str, indices))
        chunk_path = dataset / "data" / field / relative
        if not chunk_path.is_file():
            missing += 1
            if missing <= MAX_EXAMPLES:
                findings.add("errors", "CHUNK_MISSING", f"data/{field}", "chunk 文件", relative,
                             frame=indices[0] * array.chunks[0])
    if missing > MAX_EXAMPLES:
        findings.add("errors", "CHUNK_MISSING", f"data/{field}", "chunk 文件",
                     f"另外 {missing - MAX_EXAMPLES} 个缺失", count=missing - MAX_EXAMPLES)


def mark_bad(findings, code, field, expected, values, mask, start, ends):
    """根据逐帧布尔 mask 汇总异常，并用 ends 定位 episode 和全局帧号。

    values 与 mask 对应同一数据块，start 是该块在数据集中的起始帧号。
    """
    indexes = np.flatnonzero(mask)
    if not len(indexes):
        return
    for index in indexes[:MAX_EXAMPLES]:
        frame = start + int(index)
        episode = int(np.searchsorted(ends, frame, side="right"))
        findings.add("errors", code, f"data/{field}", expected,
                     np.asarray(values[index]).tolist(), episode, frame)
    if len(indexes) > MAX_EXAMPLES:
        findings.add("errors", code, f"data/{field}", expected,
                     f"另外 {len(indexes) - MAX_EXAMPLES} 帧", count=len(indexes) - MAX_EXAMPLES)


def scan_array(array, name, ends, findings, limits=None):
    """按首维 chunk 大小扫描数组，检查数值、来源索引和动作范围。

    limits 可选，为一对形状为 (7,) 的关节下限与上限；问题写入 findings。
    """
    count = array.shape[0]
    block = max(1, array.chunks[0])
    for start in range(0, count, block):
        stop = min(start + block, count)
        try:
            values = array[start:stop]
        except Exception as exc:
            findings.add("errors", "CHUNK_READ_FAILED", f"data/{name}", "可解码的 chunk",
                         f"{type(exc).__name__}: {exc}", frame=start)
            continue
        if name not in ("camera0_rgb", "source_image_index"):
            mark_bad(findings, "NONFINITE", name, "有限数值", values,
                     ~np.isfinite(values).reshape(len(values), -1).all(axis=1), start, ends)
        if name == "source_image_index":
            mark_bad(findings, "SOURCE_INDEX_INVALID", name, "非负整数", values,
                     values < 0, start, ends)
        if name in ("robot0_gripper_position", "robot0_gripper_width", "action"):
            gripper = values[:, -1] if name == "action" else values[:, 0]
            mark_bad(findings, "GRIPPER_OUT_OF_RANGE", name, "[0,1] ±1e-6", gripper,
                     np.isfinite(gripper) & ((gripper < -1e-6) | (gripper > 1 + 1e-6)),
                     start, ends)
        if limits is not None and name in ("robot0_joint_pos", "action"):
            joints = values[:, :7]
            lower, upper = limits
            invalid = np.isfinite(joints) & ((joints < lower - 1e-6) | (joints > upper + 1e-6))
            mark_bad(findings, "JOINT_OUT_OF_RANGE", name, "URDF 关节限位 ±1e-6",
                     joints, invalid.any(axis=1), start, ends)


def check_sequence(array, name, ends, findings, frequency):
    """按 episode 检查 timestamp 或 source_image_index 的相邻帧差值。

    timestamp 按 frequency（Hz）验证采样间隔；序列允许在 episode 之间重置。
    """
    starts = np.r_[0, ends[:-1]]
    for episode, (start, end) in enumerate(zip(starts, ends)):
        previous = None
        for offset in range(int(start), int(end), max(1, array.chunks[0])):
            stop = min(offset + max(1, array.chunks[0]), int(end))
            try:
                values = array[offset:stop]
            except Exception:
                continue  # scan_array 已报告 chunk 读取失败。
            if previous is not None:
                values = np.r_[previous, values]
                first_frame = offset - 1
            else:
                first_frame = offset
            deltas = np.diff(values.astype(np.float64))
            if name == "timestamp":
                bad = ~np.isfinite(deltas) | (deltas <= 0) | (np.abs(deltas - 1 / frequency) > 1e-5)
                code, expected = "TIMESTAMP_INVALID", f"严格递增，间隔 {1/frequency:.8f} 秒 ±1e-5"
            else:
                bad = deltas < 0
                code, expected = "SOURCE_INDEX_DECREASE", "episode 内不递减"
            indexes = np.flatnonzero(bad)
            for index in indexes[:MAX_EXAMPLES]:
                findings.add("errors", code, f"data/{name}", expected,
                             float(deltas[index]), episode, first_frame + int(index) + 1)
            if len(indexes) > MAX_EXAMPLES:
                findings.add("errors", code, f"data/{name}", expected,
                             f"另外 {len(indexes) - MAX_EXAMPLES} 处", count=len(indexes) - MAX_EXAMPLES)
            previous = values[-1]


def check_start_pose(data, ends, findings):
    """检查 Link7 起始姿态在每个 episode 内恒定且等于首帧观测。

    data 包含位置、旋转轴角及起始姿态数组；差异和读取错误写入 findings。
    """
    starts = np.r_[0, ends[:-1]]
    try:
        for episode, (start, end) in enumerate(zip(starts, ends)):
            start = int(start)
            end = int(end)
            reference = np.r_[data["robot0_eef_pos"][start],
                              data["robot0_eef_rot_axis_angle"][start]]
            for offset in range(start, end, max(1, data["robot0_demo_start_pose"].chunks[0])):
                stop = min(end, offset + max(1, data["robot0_demo_start_pose"].chunks[0]))
                values = data["robot0_demo_start_pose"][offset:stop]
                bad = ~np.isclose(values, reference, atol=1e-5, rtol=0).all(axis=1)
                mark_bad(findings, "START_POSE_MISMATCH", "robot0_demo_start_pose",
                         "与本段首帧观测相同，段内恒定，误差 ≤1e-5", values,
                         bad, offset, ends)
    except Exception as exc:
        findings.add("errors", "CHUNK_READ_FAILED", "data/robot0_demo_start_pose",
                     "可读取的起始位姿", f"{type(exc).__name__}: {exc}")


def check_windows(ends, horizon, seed, val_ratio, findings):
    """按 episode 长度和 horizon 计算无动作填充的训练与验证窗口。

    seed 控制按 episode 划分的随机选择，val_ratio 为验证集比例。
    返回 train、validation 和 total 窗口数，短 episode 与空划分写入 findings。
    """
    lengths = np.diff(np.r_[0, ends])
    windows = np.maximum(0, lengths - horizon + 1)
    for episode in np.flatnonzero(windows == 0)[:MAX_EXAMPLES]:
        findings.add("warnings", "EPISODE_TOO_SHORT", "meta/episode_ends",
                     f"至少 {horizon} 帧", int(lengths[episode]), int(episode))
    if len(np.flatnonzero(windows == 0)) > MAX_EXAMPLES:
        findings.add("warnings", "EPISODE_TOO_SHORT", "meta/episode_ends",
                     f"至少 {horizon} 帧", "其余短 episode",
                     count=len(np.flatnonzero(windows == 0)) - MAX_EXAMPLES)
    val_mask = np.zeros(len(ends), dtype=bool)
    if val_ratio > 0:
        n_val = min(max(1, round(len(ends) * val_ratio)), len(ends) - 1)
        val_mask[np.random.default_rng(seed).choice(len(ends), size=n_val, replace=False)] = True
    train_windows = int(windows[~val_mask].sum())
    validation_windows = int(windows[val_mask].sum())
    if train_windows == 0:
        findings.add("errors", "NO_TRAIN_WINDOWS", "meta/episode_ends", "至少 1 个训练窗口", 0)
    if val_ratio > 0 and validation_windows == 0:
        findings.add("errors", "NO_VALIDATION_WINDOWS", "meta/episode_ends", "至少 1 个验证窗口", 0)
    return {"train": train_windows, "validation": validation_windows,
            "total": int(windows.sum())}


def validate_dataset(path, kind, horizon=16, seed=42, val_ratio=0.1):
    """只读验证 path 指向的单个 Zarr，返回包含检查结果的字典。

    kind 选择训练格式；horizon、seed 和 val_ratio 控制窗口计算与划分。
    即使发现错误，也继续执行可用的检查并将问题汇入结果。
    """
    path = Path(path)
    findings = Findings()
    result = {"path": str(path.resolve()), "type": kind, "frames": None,
              "episodes": None, "shapes": {}, "windows": None}
    try:
        root = zarr.open_group(str(path), mode="r")
    except Exception as exc:
        findings.add("errors", "ZARR_OPEN_FAILED", "root", "Zarr v2 根组",
                     f"{type(exc).__name__}: {exc}")
        result.update(findings.report())
        result["valid"] = False
        return result
    try:
        attrs = dict(root.attrs)
    except Exception as exc:
        findings.add("errors", "ATTRIBUTE_READ_FAILED", ".zattrs", "有效的根属性",
                     f"{type(exc).__name__}: {exc}")
        result.update(findings.report())
        result["valid"] = False
        return result
    limits = check_contract(attrs, kind, findings)
    ends, names = read_meta(root, findings)
    if ends is not None:
        count = int(ends[-1])
        result["frames"] = count
        result["episodes"] = len(ends)
        result["windows"] = check_windows(ends, horizon, seed, val_ratio, findings)
        arrays, result["shapes"] = describe_arrays(root, kind, count, findings)
        for name, array in arrays.items():
            check_chunks(array, name, path, findings)
            scan_array(array, name, ends, findings,
                       limits if kind == "rm75_joint" else None)
        for name in ("timestamp", "source_image_index"):
            if name in arrays:
                check_sequence(arrays[name], name, ends, findings, 30)
        if kind == "rm75_umi" and all(name in arrays for name in
                                      ("robot0_demo_start_pose", "robot0_eef_pos",
                                       "robot0_eef_rot_axis_angle")):
            check_start_pose(arrays, ends, findings)
    result.update(findings.report())
    result["valid"] = not result["errors"]
    return result


def main(argv=None):
    """解析 argv 并验证指定目录中的数据集，返回 0、1 或 2。

    argv 为 None 时读取进程命令行；可选地写入 JSON 报告，并打印检查摘要。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="单个 Zarr 或其父目录")
    parser.add_argument("--type", required=True, choices=sorted(FORMATS))
    parser.add_argument("--report", type=Path, help="可选 JSON 报告路径")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args(argv)
    if args.action_horizon < 1 or not np.isfinite(args.val_ratio) or not 0 <= args.val_ratio < 1:
        parser.error("--action-horizon 必须为正整数，--val-ratio 必须在 [0,1) 内")
    try:
        paths = discover_datasets(args.input)
    except (OSError, ValueError) as exc:
        print(f"运行错误：{exc}", file=sys.stderr)
        return 2
    if args.report is not None and any(
            path.resolve() == args.report.resolve() or
            path.resolve() in args.report.resolve().parents for path in paths):
        print("运行错误：报告路径不能位于被检查的 Zarr 目录内。", file=sys.stderr)
        return 2
    datasets = []
    for path in paths:
        try:
            datasets.append(validate_dataset(path, args.type, args.action_horizon,
                                             args.seed, args.val_ratio))
        except Exception as exc:
            datasets.append({"path": str(path.resolve()), "type": args.type, "frames": None,
                             "episodes": None, "shapes": {}, "windows": None, "valid": False,
                             "warnings": [], "errors": [{"code": "DATASET_READ_FAILED",
                               "field": "root", "expected": "可读取的 Zarr 数据集", "count": 1,
                               "examples": [{"episode": None, "frame": None,
                                             "actual": f"{type(exc).__name__}: {exc}"}]}]})
    report = {"input": str(args.input.resolve()), "type": args.type,
              "action_horizon": args.action_horizon, "obs_horizon": 2,
              "frequency_hz": 30, "image_size": 224, "seed": args.seed,
              "val_ratio": args.val_ratio, "valid": bool(datasets) and all(
                  item["valid"] for item in datasets), "datasets": datasets}
    if not paths:
        print("未发现 Zarr 数据集。")
    for item in datasets:
        print(f"{'通过' if item['valid'] else '不通过'}：{item['path']}；"
              f"帧数 {item['frames']}；episode {item['episodes']}；窗口 {item['windows']}")
        for issue in item["errors"] + item["warnings"]:
            sample = issue["examples"][0]
            print(f"  {issue['code']} {issue['field']}：期望 {issue['expected']}，"
                  f"实际 {sample['actual']}；次数 {issue['count']}；"
                  f"episode {sample['episode']}，帧 {sample['frame']}")
    if args.report is not None:
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        except OSError as exc:
            print(f"报告写入失败：{exc}", file=sys.stderr)
            return 2
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
