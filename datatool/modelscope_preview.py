"""从 FastUMI Zarr 抽样生成 ModelScope 可索引的轨迹及数据集卡片。

CSV 仅保存少量帧的数值字段；RGB 截图用于远端数据集卡片。完整训练数据
仍保存在原始 ZIP 中。生成文件放在源数据之外，不导出机器专用源路径。
"""

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import quote
import zipfile

import numpy as np
import yaml
import zarr


CONFIG_NAME = "fastumi_preview"
CSV_PATH = "fastumi_preview/trajectory.csv"
START_MARKER = "<!-- fastumi-preview:start -->"
END_MARKER = "<!-- fastumi-preview:end -->"
MAX_IMAGES = 6
NUMERIC_FIELDS = (
    "timestamp", "source_image_index", "robot0_eef_pos",
    "robot0_eef_rot_axis_angle", "robot0_gripper_width", "robot0_joint_pos",
    "robot0_gripper_position", "action",
)


@dataclass
class PreviewBundle:
    """生成的文件及抽样统计；files 中元数据和卡片排在数据文件之后。"""

    directory: Path
    files: list
    rows: int
    images: int


@contextmanager
def open_source(source):
    """只读打开 Zarr 目录或包含单个 Zarr 根目录的 ZIP。"""
    store = None
    try:
        if source.is_dir():
            root = zarr.open_group(str(source), mode="r")
        elif zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                names = set(archive.namelist())
            if ".zgroup" in names:
                prefix = ""
            else:
                candidates = [name.rsplit("/", 1)[0] for name in names
                              if name.count("/") == 1 and name.endswith("/.zgroup")]
                if len(candidates) != 1:
                    raise ValueError("预览 ZIP 必须包含单个顶层 Zarr v2 目录")
                prefix = candidates[0]
            store = zarr.ZipStore(str(source), mode="r")
            root = zarr.open_group(store=store, path=prefix, mode="r")
        else:
            raise ValueError("预览仅支持 FastUMI Zarr 目录或 Zarr ZIP")
        yield root
    finally:
        if store is not None:
            store.close()


def sample_indices(total_frames, max_rows):
    """等距抽样全局帧，包含首尾帧，不重复、不跨越数组边界。"""
    if total_frames <= 0 or max_rows <= 0:
        raise ValueError("预览帧数和 --preview-rows 必须大于零")
    return np.linspace(0, total_frames - 1, min(total_frames, max_rows), dtype=np.int64)


def merge_card(existing, section):
    """保留远端卡片正文及元数据，指定 CSV 配置并更新生成的预览段落。"""
    metadata = {}
    body = existing
    if existing.startswith("---\n") or existing.startswith("---\r\n"):
        boundaries = list(re.finditer(r"(?m)^---\r?$", existing))
        if len(boundaries) < 2:
            raise ValueError("远端 README.md 的 YAML 文件头没有闭合，无法安全合并")
        first, last = boundaries[:2]
        metadata = yaml.safe_load(existing[first.end():last.start()]) or {}
        body = existing[last.end():].lstrip("\r\n")
        if not isinstance(metadata, dict):
            raise ValueError("远端 README.md 的 YAML 文件头必须是对象")
    configs = metadata.get("configs", [])
    if not isinstance(configs, list) or any(not isinstance(item, dict) for item in configs):
        raise ValueError("远端卡片的 configs 格式无效，无法安全合并")
    configs = [dict(item) for item in configs if item.get("config_name") != CONFIG_NAME]
    for item in configs:
        item.pop("default", None)
    metadata["configs"] = [{"config_name": CONFIG_NAME, "default": True,
                            "data_files": [{"split": "train", "path": CSV_PATH}]}, *configs]
    generated = f"{START_MARKER}\n{section.rstrip()}\n{END_MARKER}"
    if START_MARKER in body or END_MARKER in body:
        if body.count(START_MARKER) != 1 or body.count(END_MARKER) != 1:
            raise ValueError("远端卡片的 FastUMI 预览标记重复或不完整")
        start, end = body.index(START_MARKER), body.index(END_MARKER) + len(END_MARKER)
        if end < start:
            raise ValueError("远端卡片的 FastUMI 预览标记顺序无效")
        body = body[:start] + generated + body[end:]
    else:
        body = body.rstrip() + "\n\n" + generated + "\n"
    return "---\n" + yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False) + "---\n\n" + body.lstrip("\n")


def check_output_path(path):
    """生成文件不得写入符号链接指向的位置。"""
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"预览输出不支持符号链接：{component}")


def build_preview(source, directory, archive_path, dataset_name, max_rows=1000,
                  existing_card="", existing_infos=None):
    """生成抽样 CSV、最多六张 RGB 截图及合并后的远端元数据。

    数值保留原字段含义和单位；二维数值字段按列展开。全局 frame_index 和
    episode_frame_index 均从零开始。CSV 的 train split 仅用于网页预览，
    不代表完整训练集或训练/验证划分。
    """
    source, directory = Path(source).absolute(), Path(directory).absolute()
    if directory == source or source in directory.parents:
        raise ValueError("预览输出目录不能位于源数据内部")
    check_output_path(directory)
    if existing_infos is not None and not isinstance(existing_infos, dict):
        raise ValueError("远端 dataset_infos.json 必须是对象，无法安全合并")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".modelscope-preview-",
                                     dir=directory.parent) as temporary, open_source(source) as root:
        staging = Path(temporary)
        if "data" not in root or "meta/episode_ends" not in root:
            raise ValueError("预览需要 data 数组组和 meta/episode_ends")
        data = root["data"]
        ends = np.asarray(root["meta/episode_ends"][:])
        if (ends.ndim != 1 or len(ends) == 0 or ends.dtype.kind not in "iu"
                or ends[0] <= 0 or np.any(ends[1:] <= ends[:-1])):
            raise ValueError("meta/episode_ends 必须是严格递增的正整数数组")
        total = int(ends[-1])
        indices = sample_indices(total, max_rows)
        episodes = np.searchsorted(ends, indices, side="right")
        starts = np.r_[0, ends[:-1]]
        names = None
        if "meta/episode_names" in root:
            names = root["meta/episode_names"][:]
            if names.shape != ends.shape:
                raise ValueError("episode_names 和 episode_ends 长度必须一致")
        headers = ["frame_index", "episode_index", "episode_frame_index"]
        features = {name: {"dtype": "int64", "_type": "Value"} for name in headers}
        if names is not None:
            headers.append("episode_name")
            features["episode_name"] = {"dtype": "string", "_type": "Value"}
        values = {}
        for name in NUMERIC_FIELDS:
            if name not in data:
                continue
            array = data[name]
            if array.ndim not in (1, 2) or array.shape[0] != total or array.dtype.kind not in "iuf":
                raise ValueError(f"data/{name} 必须是与帧数对齐的一维或二维数值数组")
            sampled = np.asarray(array.oindex[indices])
            if not np.isfinite(sampled).all():
                raise ValueError(f"data/{name} 的预览样本包含非有限数值")
            columns = [name] if array.ndim == 1 else [f"{name}_{i}" for i in range(array.shape[1])]
            for offset, column in enumerate(columns):
                values[column] = sampled if array.ndim == 1 else sampled[:, offset]
                features[column] = {"dtype": "float64" if array.dtype.kind == "f" else "int64",
                                    "_type": "Value"}
            headers.extend(columns)
        if not values and "camera0_rgb" not in data:
            raise ValueError("未找到可预览的轨迹数值或 camera0_rgb 图像")
        csv_file = staging / CSV_PATH
        csv_file.parent.mkdir(parents=True)
        with csv_file.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=headers)
            writer.writeheader()
            for row_index, frame in enumerate(indices):
                episode = int(episodes[row_index])
                row = {"frame_index": int(frame), "episode_index": episode,
                       "episode_frame_index": int(frame - starts[episode])}
                if names is not None:
                    value = names[episode]
                    row["episode_name"] = value.decode("utf-8") if isinstance(value, bytes) else str(value)
                row.update({name: column[row_index].item() for name, column in values.items()})
                writer.writerow(row)
        image_paths = []
        if "camera0_rgb" in data:
            import cv2

            camera = data["camera0_rgb"]
            if camera.ndim != 4 or camera.shape[0] != total or camera.shape[-1] != 3 or camera.dtype != np.uint8:
                raise ValueError("camera0_rgb 必须是 (帧数, 高, 宽, 3) 的 uint8 RGB 数组")
            for frame in indices[sample_indices(len(indices), MAX_IMAGES)]:
                relative = f"fastumi_preview/images/frame_{int(frame):09d}.jpg"
                path = staging / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image = cv2.cvtColor(camera[int(frame)], cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    raise OSError(f"无法保存预览图片：{relative}")
                image_paths.append(relative)
        infos = dict(existing_infos or {})
        infos[CONFIG_NAME] = {"config_name": CONFIG_NAME, "features": features,
                              "splits": {"train": {"name": "train", "num_examples": len(indices),
                                                    "dataset_name": dataset_name}},
                              "dataset_size": csv_file.stat().st_size}
        (staging / "dataset_infos.json").write_text(
            json.dumps(infos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        section = (f"## FastUMI 数据预览\n\n完整数据包含 {total:,} 帧、{len(ends):,} 个 episode。"
                   f"网页 `fastumi_preview` 配置仅展示等距抽样的 {len(indices):,} 帧；"
                   "预览中的 train 是展示分组，不代表训练数据划分。\n\n"
                   f"完整 Zarr ZIP：[下载文件]({quote(archive_path, safe='/')})，下载后解压使用。\n\n"
                   f"轨迹索引：[trajectory.csv]({CSV_PATH})。frame_index 是原始全局帧索引，"
                   "episode_frame_index 是 episode 内帧索引，二者从零开始。"
                   "向量字段后缀 `_0`、`_1` 等表示原数组列号。"
                   "robot0_eef_pos 对应 x/y/z，robot0_eef_rot_axis_angle 对应旋转向量，"
                   "action 保留原始动作列定义。\n\n")
        # 只发布描述数据含义的属性，排除 source_dataset 等本机绝对路径。
        for key in ("format", "frequency", "base_frame", "end_frame", "position_unit",
                    "rotation_unit", "gripper_representation", "stored_action_layout"):
            if key in root.attrs:
                section += f"- {key}: `{root.attrs[key]}`\n"
        if image_paths:
            section += "\n### 相机截图\n\n" + "\n\n".join(
                f"![RGB {Path(path).stem}]({path})" for path in image_paths) + "\n"
        (staging / "README.md").write_text(merge_card(existing_card, section), encoding="utf-8")
        files = [CSV_PATH, *image_paths, "dataset_infos.json", "README.md"]
        for relative in files:
            target = directory / relative
            check_output_path(target)
            if target.exists() and not target.is_file():
                raise ValueError(f"预览输出路径不是普通文件：{target}")
        directory.mkdir(parents=True, exist_ok=True)
        for relative in files:
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / relative).replace(target)
    return PreviewBundle(directory, files, len(indices), len(image_paths))


def publish_preview(api, source, archive_path, repo_id, revision, max_rows=1000,
                    require_archive=False):
    """合并远端已有卡片和字段信息，先上传预览数据，最后提交卡片配置。"""
    remote_files = {item.path for item in api.list_repo_files(
        repo_id=repo_id, repo_type="dataset", revision=revision)}
    if require_archive and archive_path not in remote_files:
        raise ValueError(f"远端未找到完整 ZIP：{archive_path}；请确认 --path-in-repo 和 --revision")
    existing_card = ""
    existing_infos = None
    with tempfile.TemporaryDirectory(prefix="modelscope-preview-meta-") as cache:
        for name in ("README.md", "dataset_infos.json"):
            if name not in remote_files:
                continue
            path = api.download_file(repo_id=repo_id, repo_type="dataset",
                                     file_path=name, revision=revision, cache_dir=cache,
                                     force=True)
            text = Path(path).read_text(encoding="utf-8")
            if name == "README.md":
                existing_card = text
            else:
                existing_infos = json.loads(text)
    source = Path(source)
    directory = source.with_name(source.name + ".modelscope-preview")
    bundle = build_preview(source, directory, archive_path, repo_id.split("/")[-1],
                           max_rows, existing_card, existing_infos)
    print(f"预览已生成：{bundle.rows:,} 行轨迹、{bundle.images} 张图片，保存在 {bundle.directory}")
    for relative in bundle.files:
        api.upload_file(repo_id=repo_id, repo_type="dataset",
                        path_or_fileobj=str(bundle.directory / relative),
                        path_in_repo=relative, revision=revision,
                        commit_message=f"Add FastUMI preview: {relative}", disable_tqdm=False)
    print("预览文件已上传；网页解析由 ModelScope 异步执行。若仍显示旧错误，请在预览页重试解析。")
    return bundle
