"""将 FastUMI LeRobot v3 的视频和 (N, 7) TCP 示教数据导出为独立 RRD。

位置保留 meta/fastumi.json 的参考系，单位米；姿态为弧度旋转向量，
夹爪为 [0, 1]。仅在实际转换时加载第三方依赖，不依赖 ROS 或 LeRobot。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace
import uuid

POSE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
COLORS = {"state": [60, 170, 255], "action": [255, 150, 50]}
SKIP_DIRS = {".staging", ".trash", "_analysis"}


def load_runtime():
    """延迟加载离线转换依赖；缺失时提供独立环境安装提示。"""
    try:
        import av
        import cv2
        import numpy as np
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as error:
        raise RuntimeError(
            f"无法加载 LeRobot 转换依赖：{error}。请安装 requirements-rerun.txt；"
            "LeRobot 模式无需加载 ROS。"
        ) from error
    return SimpleNamespace(av=av, cv2=cv2, np=np, ds=ds, pq=pq, rr=rr, rrb=rrb)


def discover_datasets(source, output):
    """发现含 meta/info.json 的数据集，跳过输出、暂存目录与目录符号链接。"""
    if not source.is_dir():
        raise ValueError(f"输入不是目录：{source}")
    found = []
    for current, dirs, _ in os.walk(source, followlinks=False):
        path = Path(current)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS
                         and not (path / d).is_symlink()
                         and (path / d).resolve() != output)
        if (path / "meta/info.json").is_file():
            found.append(path)
            dirs[:] = []
    return found


def read_json(path):
    """读取 JSON 对象，拒绝非对象元数据。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"元数据不是对象：{path}")
    return value


def local_path(root, relative):
    """解析数据集内路径，禁止模板指向数据集外。"""
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"路径超出数据集：{relative}")
    return path


def read_metadata(rt, root):
    """校验 FastUMI 契约并读取 episode 索引，返回数据源与元数据。"""
    info = read_json(root / "meta/info.json")
    contract = read_json(root / "meta/fastumi.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("仅支持 LeRobot v3.0")
    for key, expected in {
        "state_layout": "xyz_rotvec_gripper", "action_layout": "xyz_rotvec_gripper",
        "position_unit": "m", "rotation_unit": "rad",
        "gripper_representation": "normalized_0_1",
    }.items():
        if contract.get(key) != expected:
            raise ValueError(f"不兼容的 {key}：{contract.get(key)!r}，需要 {expected}")
    if not isinstance(contract.get("pose_frame"), str) or not contract["pose_frame"].strip():
        raise ValueError("缺少 pose_frame 坐标系声明")
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps 必须是有限正数")
    features = info["features"]
    for key in ("observation.state", "action"):
        feature = features[key]
        if feature.get("shape") != [7] or feature.get("names") != POSE_NAMES:
            raise ValueError(f"{key} 必须是 [xyz, 旋转向量, gripper] 七维向量")
    cameras = [key for key, value in features.items() if value.get("dtype") == "video"]
    if not cameras:
        raise ValueError("没有 video 图像特征")
    episode_files = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise ValueError("没有 episode Parquet 索引")
    episodes = rt.pq.read_table(episode_files).to_pylist()
    indices = [int(row["episode_index"]) for row in episodes]
    if len(set(indices)) != len(indices) or len(indices) != int(info["total_episodes"]):
        raise ValueError("episode 索引重复或总数与 info.json 不一致")
    # 扫描模板匹配的所有分片，由 Arrow 按 episode 和全局行号过滤，支持跨文件 episode。
    pattern = re.sub(r"\{(?:chunk_index|file_index)(?::[^}]*)?\}", "*", info["data_path"])
    local_path(root, pattern)
    files = sorted(root.glob(pattern))
    if not files:
        raise ValueError("没有匹配 data_path 的 Parquet 数据")
    data = rt.ds.dataset([str(local_path(root, str(p.relative_to(root)))) for p in files],
                         format="parquet")
    return info, contract, cameras, sorted(episodes, key=lambda r: r["episode_index"]), data


def read_episode(rt, data, episode):
    """按全局行范围读取一个 episode，严格校验帧号、时间和 (N, 7) 数值。"""
    np = rt.np
    index = int(episode["episode_index"])
    start, stop = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
    rows = data.to_table(
        filter=(rt.ds.field("episode_index") == index),
        columns=["index", "episode_index", "frame_index", "timestamp", "observation.state", "action"],
    ).sort_by("frame_index").to_pydict()
    count = int(episode["length"])
    if count <= 0 or stop - start != count or rows["index"] != list(range(start, stop)):
        raise ValueError("episode 数据行数或全局索引不匹配")
    if rows["frame_index"] != list(range(count)):
        raise ValueError("episode 帧号必须从零连续递增")
    times = np.asarray(rows["timestamp"], dtype=np.float64)
    if not np.isfinite(times).all() or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("timestamp 必须有限、非负且严格递增")
    values = {}
    for column, name in (("observation.state", "state"), ("action", "action")):
        array = np.asarray(rows[column], dtype=np.float64)
        if array.shape != (count, 7) or not np.isfinite(array).all():
            raise ValueError(f"{column} 形状错误或含非有限值")
        if np.any((array[:, 6] < 0) | (array[:, 6] > 1)):
            raise ValueError(f"{column} 夹爪超出 [0, 1]")
        values[name] = array
    return times, values


def video_frames(rt, root, info, episode, key, times):
    """按 PTS 和 episode 偏移匹配视频帧；不重用帧、不越过 episode 边界。

    解码从目标前的关键帧开始；半帧容差吸收 float32 timestamp 的舍入误差，
    不通过重复帧或插值掩盖缺失帧。返回 uint8 BGR (H, W, 3)。
    """
    prefix = f"videos/{key}"
    path = local_path(root, info["video_path"].format(
        video_key=key, chunk_index=int(episode[f"{prefix}/chunk_index"]),
        file_index=int(episode[f"{prefix}/file_index"])))
    start = float(episode[f"{prefix}/from_timestamp"])
    stop = float(episode[f"{prefix}/to_timestamp"])
    if not (math.isfinite(start) and math.isfinite(stop) and 0 <= start < stop):
        raise ValueError(f"{key} 视频时间范围无效")
    tolerance = 0.49 / float(info["fps"])
    targets = start + times
    if targets[0] < start or targets[-1] >= stop:
        raise ValueError(f"{key} 数据时间超出 episode 视频范围")
    with rt.av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int(start / stream.time_base)), stream=stream, backward=True)
        matched = 0
        previous = -math.inf
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"{key} 视频帧缺少 PTS")
            stamp = float(frame.pts * frame.time_base)
            if stamp <= previous:
                raise ValueError(f"{key} 视频 PTS 不递增")
            previous = stamp
            if stamp < start - 1e-6 or stamp < targets[matched] - tolerance:
                continue
            if stamp >= stop - 1e-6 or abs(stamp - targets[matched]) > tolerance:
                raise ValueError(f"{key} 第 {matched} 帧没有对应视频帧")
            pixels = frame.to_ndarray(format="bgr24")
            if list(pixels.shape) != info["features"][key]["shape"]:
                raise ValueError(f"{key} 视频尺寸与 features 不一致")
            yield pixels
            matched += 1
            if matched == len(targets):
                return
        raise ValueError(f"{key} 视频提前结束：需要 {len(targets)} 帧，仅匹配 {matched}")


def rotvec_quaternion(np, vector):
    """弧度旋转向量 (3,) 转为 Rerun 使用的 xyzw 单位四元数 (4,)。"""
    angle = float(np.linalg.norm(vector))
    scale = 0.5 if angle < 1e-12 else math.sin(angle / 2) / angle
    return [*(vector * scale), math.cos(angle / 2)]


def blueprint(rt, cameras):
    """相机、TCP 位姿、夹爪和七维曲线的默认布局。"""
    b = rt.rrb
    return b.Blueprint(b.Horizontal(
        b.Vertical(*[b.Spatial2DView(name=key, origin=f"/camera/{key}") for key in cameras],
                   b.TextDocumentView(name="Episode metadata", origin="/metadata")),
        b.Vertical(b.Spatial3DView(name="TCP (m)", origin="/world"),
                   b.TimeSeriesView(name="Gripper [0, 1]", origin="/gripper"),
                   b.Tabs(*[b.TimeSeriesView(name=name, origin=f"/values/{name}")
                            for name in POSE_NAMES]))),
        b.TimePanel(state="expanded"), collapse_panels=False)


def convert_episode(rt, root, info, contract, cameras, data, episode, destination):
    """转换单个 episode，成功后原子发布且拒绝覆盖，失败删除临时 RRD。"""
    if destination.exists():
        raise FileExistsError(f"目标已存在，拒绝覆盖：{destination}")
    times, values = read_episode(rt, data, episode)
    rr = rt.rr
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.stem}.{uuid.uuid4().hex}.rrd"
    videos = {key: video_frames(rt, root, info, episode, key, times) for key in cameras}
    try:
        with rr.RecordingStream("fastumi_lerobot_episode") as recording:
            recording.save(stage, default_blueprint=blueprint(rt, cameras))
            metadata = {"dataset": root.name, "episode_index": episode["episode_index"],
                        "tasks": episode.get("tasks", []),
                        **{key: contract.get(key) for key in (
                            "pose_frame", "end_frame", "position_unit", "rotation_unit",
                            "state_layout", "action_layout", "action_source")}}
            recording.log("metadata", rr.TextDocument(
                json.dumps(metadata, ensure_ascii=False, indent=2)), static=True)
            # 坐标系的重力方向未由契约声明，因此不套用 MCAP 的 Z-up 约定。
            for name, array in values.items():
                recording.log(f"world/{name}/tcp", rr.TransformAxes3D(0.1), static=True)
                recording.log(f"world/{name}/path", rr.LineStrips3D(
                    [array[:, :3]], colors=[COLORS[name]]), static=True)
                for dimension in POSE_NAMES:
                    recording.log(f"values/{dimension}/{name}", rr.SeriesLines(
                        colors=[COLORS[name]], names=[name]), static=True)
                recording.log(f"gripper/{name}", rr.SeriesLines(
                    colors=[COLORS[name]], names=[name]), static=True)
            for i, stamp in enumerate(times):
                recording.set_time("episode_time", duration=float(stamp))
                recording.set_time("frame_index", sequence=i)
                for key, frames in videos.items():
                    pixels = next(frames)
                    ok, jpeg = rt.cv2.imencode(".jpg", pixels, [rt.cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok:
                        raise ValueError(f"{key} JPEG 编码失败")
                    recording.log(f"camera/{key}", rr.EncodedImage(contents=jpeg.tobytes(), media_type="image/jpeg"))
                for name, array in values.items():
                    row = array[i]
                    recording.log(f"world/{name}/tcp", rr.Transform3D(
                        translation=row[:3], quaternion=rotvec_quaternion(rt.np, row[3:6])))
                    recording.log(f"world/{name}/tcp", rr.Points3D(
                        [[0, 0, 0]], colors=[COLORS[name]], radii=0.005))
                    for dimension, value in zip(POSE_NAMES, row):
                        recording.log(f"values/{dimension}/{name}", rr.Scalars(float(value)))
                    recording.log(f"gripper/{name}", rr.Scalars(float(row[6])))
        # 同目录硬链接以排他方式发布，避免检查后 rename 覆盖并发生成的同名文件。
        os.link(stage, destination)
    finally:
        for frames in videos.values():
            frames.close()
        stage.unlink(missing_ok=True)
    return len(times)


def run(source, output, datasets, selected=None):
    """逐数据集、逐 episode 导出；汇总错误，保持其他 episode 可继续处理。"""
    try:
        rt = load_runtime()
    except RuntimeError as error:
        print(error)
        return 2
    failed = succeeded = 0
    for root in datasets:
        try:
            info, contract, cameras, episodes, data = read_metadata(rt, root)
        except Exception as error:
            failed += 1
            print(f"{root}: 数据集失败：{error}")
            continue
        available = {int(row["episode_index"]) for row in episodes}
        if selected is not None:
            for missing in sorted(set(selected) - available):
                failed += 1
                print(f"{root}: episode {missing} 不存在")
            episodes = [row for row in episodes if int(row["episode_index"]) in selected]
        for episode in episodes:
            index = int(episode["episode_index"])
            destination = output / root.relative_to(source) / f"episode_{index:06d}.rrd"
            try:
                count = convert_episode(rt, root, info, contract, cameras, data, episode, destination)
                succeeded += 1
                print(f"{root.name}/{index}: 完成 -> {destination}，{count} 帧，"
                      f"坐标系 {contract['pose_frame']}")
            except Exception as error:
                failed += 1
                print(f"{root.name}/{index}: 失败：{error}")
    print(f"总计成功 {succeeded}，失败 {failed}（含数据集错误与缺失 episode）")
    return 1 if failed else 0
