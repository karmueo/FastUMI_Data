"""把 FastUMI TCP HDF5 episodes 增量导出为 Diffusion Policy Zarr。"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import h5py
from imagecodecs_numcodecs import JpegXl, register_codecs
import numpy as np
from scipy.spatial.transform import Rotation
import zarr

from replay_buffer import ReplayBuffer


register_codecs()


def _episode_sort_key(path: Path) -> Tuple[str, int]:
    """按 session 路径和 episode 数字稳定排序。"""
    match = re.search(r"episode_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"无法从文件名解析 episode 编号: {path}")
    return str(path.parent), int(match.group(1))


def discover_hdf5_files(input_path: str) -> list[Path]:
    """发现目录下全部 episode HDF5，或接受单个 HDF5 文件。"""
    source = Path(input_path).resolve()
    if source.is_file():
        if source.suffix.lower() not in (".hdf5", ".h5"):
            raise ValueError(f"输入文件不是 HDF5: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {source}")
    files = sorted(source.rglob("episode_*.hdf5"), key=_episode_sort_key)
    if not files:
        raise ValueError(f"{source} 下没有 episode_*.hdf5")
    return files


def resize_rgb_image(image_rgb: np.ndarray, output_size: Tuple[int, int]):
    """居中裁剪并缩放 RGB 图像，保持目标宽高比。

    Args:
        image_rgb: 形状为 ``(H,W,3)`` 的 RGB 图像。
        output_size: 目标宽、高。
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("图像必须具有 HxWx3 形状")
    output_width, output_height = output_size
    input_height, input_width = image_rgb.shape[:2]
    input_ratio = input_width / input_height
    output_ratio = output_width / output_height
    if input_ratio > output_ratio:
        crop_width = int(round(input_height * output_ratio))
        x_start = (input_width - crop_width) // 2
        cropped = image_rgb[:, x_start:x_start + crop_width]
    else:
        crop_height = int(round(input_width / output_ratio))
        y_start = (input_height - crop_height) // 2
        cropped = image_rgb[y_start:y_start + crop_height, :]
    return cv2.resize(
        cropped,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )


def _decode_attribute(value) -> str:
    """把 HDF5 字节或字符串属性统一转换为普通字符串。"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _load_episode_metadata(
    path: Path,
) -> tuple[np.ndarray, float, str, tuple[int, ...]]:
    """读取并验证 episode 的轻量元数据和动作，不加载完整图像。"""
    with h5py.File(path, "r") as root:
        if "action" not in root or "observations/images/front" not in root:
            raise ValueError(f"{path} 缺少 FastUMI action 或 front 图像")
        action = np.asarray(root["action"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 8:
            raise ValueError(
                f"{path} action 必须为 (T,8)，实际为 {action.shape}"
            )
        if action.shape[0] == 0:
            raise ValueError(f"{path} 是空 episode")
        if not np.all(np.isfinite(action)):
            raise ValueError(f"{path} action 包含 NaN 或 Inf")
        quaternion_norms = np.linalg.norm(action[:, 3:7], axis=1)
        if np.max(np.abs(quaternion_norms - 1.0)) > 1.0e-3:
            raise ValueError(f"{path} action 四元数未归一化")
        if np.any(action[:, 7] < 0.0) or np.any(action[:, 7] > 1.0):
            raise ValueError(f"{path} 夹爪开度超出 [0,1]")
        image_dataset = root["observations/images/front"]
        image_shape = tuple(image_dataset.shape)
        if (
            len(image_shape) != 4
            or image_shape[0] != action.shape[0]
            or image_shape[-1] != 3
        ):
            raise ValueError(
                f"{path} front 图像必须为 (T,H,W,3)，实际为 {image_shape}"
            )
        image_encoding = _decode_attribute(
            root.attrs.get("image_encoding", "bgr8")
        ).lower()
        if image_encoding not in ("rgb8", "bgr8"):
            raise ValueError(f"{path} 不支持图像编码 {image_encoding}")
        sample_rate_hz = float(root.attrs.get("sample_rate_hz", 20.0))
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
            raise ValueError(f"{path} sample_rate_hz 无效")
    return action, sample_rate_hz, image_encoding, image_shape


def _build_numeric_episode(action: np.ndarray) -> dict[str, np.ndarray]:
    """把 TCP 四元数动作转换为 Diffusion Policy 数值键。"""
    positions = action[:, :3].astype(np.float32)
    rotations = Rotation.from_quat(action[:, 3:7]).as_rotvec().astype(
        np.float32
    )
    gripper_width = action[:, 7:8].astype(np.float32)
    pose_six = np.concatenate((positions, rotations), axis=1)
    return {
        "robot0_eef_pos": positions,
        "robot0_eef_rot_axis_angle": rotations,
        "robot0_gripper_width": gripper_width,
        "robot0_demo_start_pose": np.repeat(
            pose_six[0:1], pose_six.shape[0], axis=0
        ),
        "robot0_demo_end_pose": np.repeat(
            pose_six[-1:], pose_six.shape[0], axis=0
        ),
    }


def _append_array(
    data_group,
    key: str,
    values: np.ndarray,
    start_index: int,
) -> None:
    """创建或扩展一个时间维 Zarr 数组并写入当前 episode。"""
    new_length = start_index + values.shape[0]
    if key not in data_group:
        chunk_length = min(max(values.shape[0], 1), 1024)
        array = data_group.zeros(
            name=key,
            shape=(new_length,) + values.shape[1:],
            chunks=(chunk_length,) + values.shape[1:],
            dtype=values.dtype,
            compressor=ReplayBuffer.resolve_compressor("default"),
        )
    else:
        array = data_group[key]
        if array.shape[1:] != values.shape[1:]:
            raise ValueError(
                f"{key} episode 间形状不一致: "
                f"{array.shape[1:]} != {values.shape[1:]}"
            )
        array.resize((new_length,) + array.shape[1:])
    array[start_index:new_length] = values


def _prepare_image_array(
    data_group,
    start_index: int,
    episode_length: int,
    output_size: Tuple[int, int],
    image_compressor,
):
    """创建或扩展 RGB 图像 Zarr 数组并返回写入目标。"""
    sample_shape = (output_size[1], output_size[0], 3)
    new_length = start_index + episode_length
    if "camera0_rgb" not in data_group:
        return data_group.zeros(
            name="camera0_rgb",
            shape=(new_length,) + sample_shape,
            chunks=(1,) + sample_shape,
            dtype=np.uint8,
            compressor=image_compressor,
        )
    image_array = data_group["camera0_rgb"]
    if image_array.shape[1:] != sample_shape:
        raise ValueError(
            "camera0_rgb episode 间形状不一致: "
            f"{image_array.shape[1:]} != {sample_shape}"
        )
    image_array.resize((new_length,) + sample_shape)
    return image_array


def _append_episode(
    replay_buffer: ReplayBuffer,
    path: Path,
    output_size: Tuple[int, int],
    image_compressor,
    batch_size: int,
    legacy_bgr: bool,
) -> float:
    """分批转换一条 HDF5 episode 并追加到磁盘 Zarr。"""
    action, sample_rate_hz, image_encoding, _ = _load_episode_metadata(path)
    episode_length = action.shape[0]
    start_index = replay_buffer.n_steps
    for key, values in _build_numeric_episode(action).items():
        _append_array(replay_buffer.data, key, values, start_index)
    image_array = _prepare_image_array(
        replay_buffer.data,
        start_index,
        episode_length,
        output_size,
        image_compressor,
    )
    convert_bgr = legacy_bgr or image_encoding == "bgr8"
    with h5py.File(path, "r") as root:
        image_dataset = root["observations/images/front"]
        for batch_start in range(0, episode_length, batch_size):
            batch_stop = min(batch_start + batch_size, episode_length)
            resized_batch = np.empty(
                (
                    batch_stop - batch_start,
                    output_size[1],
                    output_size[0],
                    3,
                ),
                dtype=np.uint8,
            )
            for batch_offset, image_index in enumerate(
                range(batch_start, batch_stop)
            ):
                image = np.asarray(
                    image_dataset[image_index], dtype=np.uint8
                )
                if convert_bgr:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resized_batch[batch_offset] = resize_rgb_image(
                    image, output_size
                )
            image_array[
                start_index + batch_start:start_index + batch_stop
            ] = resized_batch
    episode_ends = replay_buffer.episode_ends
    episode_ends.resize(episode_ends.shape[0] + 1)
    episode_ends[-1] = start_index + episode_length
    return sample_rate_hz


def _create_working_store(
    output_path: Path, force: bool
) -> tuple[Path, bool]:
    """选择同目录临时 Zarr，同时保留尚未替换的正式目标。"""
    zip_output = output_path.suffix.lower() == ".zip"
    working_path = output_path.with_name(output_path.name + ".building")
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} 已存在；使用 --force 覆盖")
    if working_path.exists():
        if not force:
            raise FileExistsError(f"{working_path} 已存在；使用 --force 覆盖")
        if working_path.is_dir():
            shutil.rmtree(working_path)
        else:
            working_path.unlink()
    return working_path, zip_output


def _replace_completed_output(source: Path, destination: Path) -> None:
    """用完整产物替换目标，目录替换失败时恢复旧版本。"""
    if not destination.exists() or (
        source.is_file() and destination.is_file()
    ):
        source.replace(destination)
        return
    backup = destination.with_name(
        f"{destination.name}.backup-{uuid.uuid4().hex}"
    )
    destination.replace(backup)
    try:
        source.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    if backup.is_dir():
        shutil.rmtree(backup)
    else:
        backup.unlink()


def export_zarr(
    input_files: Iterable[Path],
    output_path: str,
    output_size: Tuple[int, int],
    compression_level: int,
    force: bool,
    legacy_bgr: bool,
    batch_size: int,
) -> None:
    """逐 episode 写入磁盘 Zarr，并可在完成后生成兼容 ZIP。"""
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    working_path, zip_output = _create_working_store(destination, force)
    archive_path = working_path.with_suffix(working_path.suffix + ".zip")
    if archive_path.exists():
        if not force:
            raise FileExistsError(
                f"{archive_path} 已存在；使用 --force 覆盖"
            )
        archive_path.unlink()
    store = zarr.DirectoryStore(str(working_path))
    replay_buffer = ReplayBuffer.create_empty_zarr(storage=store)
    image_compressor = JpegXl(level=compression_level, numthreads=1)
    sample_rates = set()
    episode_count = 0
    try:
        for path in input_files:
            sample_rate_hz = _append_episode(
                replay_buffer,
                path,
                output_size,
                image_compressor,
                batch_size,
                legacy_bgr,
            )
            sample_rates.add(sample_rate_hz)
            episode_count += 1
            print(f"已导出 {path}")
        if len(sample_rates) != 1:
            raise ValueError(
                f"episodes 的采样率不一致: {sorted(sample_rates)}"
            )
        replay_buffer.root.attrs.update(
            {
                "schema_version": "fastumi_dp_v1",
                "sample_rate_hz": float(next(iter(sample_rates))),
                "image_encoding": "rgb8",
                "image_width": output_size[0],
                "image_height": output_size[1],
                "episode_count": episode_count,
            }
        )
        total_steps = replay_buffer.n_steps
        if zip_output:
            with zarr.ZipStore(str(archive_path), mode="w") as zip_store:
                replay_buffer.save_to_store(zip_store)
                zip_root = zarr.group(store=zip_store)
                zip_root.attrs.update(dict(replay_buffer.root.attrs))
            store.close()
            shutil.rmtree(working_path)
            _replace_completed_output(archive_path, destination)
        else:
            store.close()
            _replace_completed_output(working_path, destination)
    except Exception:
        store.close()
        if working_path.exists():
            shutil.rmtree(working_path)
        if archive_path.exists():
            archive_path.unlink()
        raise
    print(
        f"完成: {episode_count} episodes, {total_steps} steps -> "
        f"{destination}"
    )


def _legacy_defaults() -> tuple[str, str, str, int]:
    """从原 config.json 读取无参数运行时的兼容默认值。"""
    with open("config/config.json", "r", encoding="utf-8") as config_file:
        config = json.load(config_file)["data_process_config"]
    return (
        config["output_tcp_dir"],
        config["dp_train_data_dir"],
        config.get("dp_data_res", "224, 224"),
        int(config.get("compression_level", 99)),
    )


def main(argv: Optional[list[str]] = None) -> None:
    """解析输入、输出、图像尺寸和覆盖策略后执行导出。"""
    default_input, default_output, default_resolution, default_level = (
        _legacy_defaults()
    )
    parser = argparse.ArgumentParser(
        description="把 FastUMI HDF5 episodes 导出为 DP Zarr"
    )
    parser.add_argument("--input", default=default_input)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--resolution", default=default_resolution)
    parser.add_argument("--compression-level", type=int, default=default_level)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="单次读取和缩放的图像帧数",
    )
    parser.add_argument("--legacy-bgr", action="store_true")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)
    resolution_values = [
        int(value.strip()) for value in arguments.resolution.split(",")
    ]
    if len(resolution_values) != 2 or any(
        value <= 0 for value in resolution_values
    ):
        raise ValueError("--resolution 必须为正整数宽高，例如 224,224")
    if arguments.batch_size <= 0:
        raise ValueError("--batch-size 必须为正整数")
    input_files = discover_hdf5_files(arguments.input)
    export_zarr(
        input_files,
        arguments.output,
        (resolution_values[0], resolution_values[1]),
        arguments.compression_level,
        arguments.force,
        arguments.legacy_bgr,
        arguments.batch_size,
    )


if __name__ == "__main__":
    main()
