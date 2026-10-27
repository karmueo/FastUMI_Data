"""在 LeRobot 的 Python 环境中写入并回读校验 v3.0 数据集（独立脚本）。

由 ``convert_mcap`` 以 ``<lerobot-python> -I lerobot_writer.py <job.json> <result.json>``
启动，运行时只依赖标准库、NumPy 和 LeRobot，**不得导入 fastumi_data 或任何 ROS 模块**。
输入的逐 episode ``.npz`` 已由 :mod:`fastumi_data.link7_lerobot` 完成全部数值和图像处理。

任务 JSON 字段：

- ``mode``：``preflight``（编码一段测试视频并读回）或 ``write``（写入全部 episode）。
- ``root``：数据集目录，必须尚不存在。
- ``repo_id``、``fps``、``robot_type``、``encoder``（``RGBEncoderConfig`` 参数）。
- ``episodes``：``[{"npz": 路径, "task": 任务描述, "name": 名称}]``，仅 ``write`` 模式。

进度以 ``FASTUMI_PROGRESS <已完成> <总数>`` 行写到 stdout；结果摘要写入 result JSON。
任何异常以非零退出码结束，错误信息写到 stderr 和 result JSON。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback


# 进度行前缀，主进程据此解析 stdout。
PROGRESS_PREFIX = "FASTUMI_PROGRESS"
# 图像特征键与 state/action 维度名称。
IMAGE_KEY = "observation.images.front"
POSE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
# 回读图像允许的平均绝对误差（0-1 灰度），H.264 有损编码。
MAX_IMAGE_MEAN_ABS_ERROR = 0.03


def _features(image_size: int) -> dict:
    """LeRobot features：7 维 state/action 与一路 RGB 视频。"""
    pose = {"dtype": "float32", "shape": (7,), "names": list(POSE_NAMES)}
    return {
        "observation.state": dict(pose),
        "action": dict(pose),
        IMAGE_KEY: {
            "dtype": "video",
            "shape": (image_size, image_size, 3),
            "names": ["height", "width", "channels"],
        },
    }


def _progress(done: int, total: int) -> None:
    """向 stdout 写一行进度。"""
    print(f"{PROGRESS_PREFIX} {done} {total}", flush=True)


def _create(job: dict, image_size: int):
    """按任务参数创建空数据集。"""
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=job["repo_id"],
        fps=int(job["fps"]),
        features=_features(image_size),
        root=job["root"],
        robot_type=job.get("robot_type"),
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(**job["encoder"]),
    )


def _add_episode(dataset, images, state, action, task: str) -> None:
    """逐帧加入一条 episode 并保存。"""
    for index in range(len(state)):
        dataset.add_frame(
            {
                "observation.state": state[index],
                "action": action[index],
                IMAGE_KEY: images[index],
                "task": task,
            }
        )
    dataset.save_episode()


def _image_error(decoded, expected) -> float:
    """解码帧（CHW float [0,1]）与原图（HWC uint8）的平均绝对误差。"""
    import numpy as np

    array = decoded.numpy() if hasattr(decoded, "numpy") else np.asarray(decoded)
    if array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    return float(np.mean(np.abs(array.astype(np.float64) - expected.astype(np.float64) / 255.0)))


def _verify(job: dict, expected: list) -> dict:
    """重新打开数据集，逐 episode 核对长度、数值与首/中/末帧图像。

    Args:
        expected: ``[(npz 路径或数组字典, 帧数)]``，顺序与写入一致。
    """
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(job["repo_id"], root=job["root"])
    if dataset.num_episodes != len(expected):
        raise RuntimeError(f"回读 episode 数 {dataset.num_episodes} 与写入 {len(expected)} 不一致")
    episodes = dataset.meta.episodes
    worst = 0.0
    lengths = []
    for index, (source, count) in enumerate(expected):
        arrays = np.load(source) if isinstance(source, str) else source
        start = int(episodes[index]["dataset_from_index"])
        stop = int(episodes[index]["dataset_to_index"])
        if stop - start != count:
            raise RuntimeError(f"episode {index} 回读 {stop - start} 帧，写入 {count} 帧")
        rows = dataset.hf_dataset.select(range(start, stop))
        for key, name in (("observation.state", "state"), ("action", "action")):
            values = np.stack([np.asarray(item, dtype=np.float32) for item in rows[key]])
            if not np.array_equal(values, arrays[name]):
                raise RuntimeError(f"episode {index} 的 {key} 回读不一致")
        frame_index = np.asarray(rows["frame_index"], dtype=np.int64)
        if not np.array_equal(frame_index, np.arange(count)):
            raise RuntimeError(f"episode {index} 的 frame_index 不连续")
        for offset in sorted({0, count // 2, count - 1}):
            error = _image_error(dataset[start + offset][IMAGE_KEY], arrays["images"][offset])
            worst = max(worst, error)
            if error > MAX_IMAGE_MEAN_ABS_ERROR:
                raise RuntimeError(
                    f"episode {index} 第 {offset} 帧解码误差 {error:.4f} 超过 {MAX_IMAGE_MEAN_ABS_ERROR}"
                )
        lengths.append(count)
    return {
        "episodes": len(lengths),
        "frames": int(sum(lengths)),
        "episode_lengths": lengths,
        "max_image_mean_abs_error": worst,
        "video_backend": str(getattr(dataset, "_video_backend", "unknown")),
    }


def _preflight(job: dict) -> dict:
    """编码一段 16 帧渐变测试视频并读回，验证编码器和解码后端可用。"""
    import numpy as np

    size = int(job.get("image_size", 224))
    count = 16
    ramp = np.linspace(0, 255, size, dtype=np.float64)
    images = np.stack(
        [
            np.stack(
                [
                    np.tile(ramp, (size, 1)),
                    np.tile(ramp[:, None], (1, size)),
                    np.full((size, size), 8.0 * index),
                ],
                axis=-1,
            ).astype(np.uint8)
            for index in range(count)
        ]
    )
    state = np.zeros((count, 7), dtype=np.float32)
    state[:, 0] = np.arange(count, dtype=np.float32)
    dataset = _create(job, size)
    _add_episode(dataset, images, state, state, "preflight")
    dataset.finalize()
    summary = _verify(job, [({"images": images, "state": state, "action": state}, count)])
    return summary


def _write(job: dict) -> dict:
    """写入全部 episode、finalize 并回读校验。"""
    import numpy as np

    episodes = job["episodes"]
    size = int(job.get("image_size", 224))
    dataset = _create(job, size)
    expected = []
    _progress(0, len(episodes))
    for number, episode in enumerate(episodes, start=1):
        with np.load(episode["npz"]) as arrays:
            images, state, action = arrays["images"], arrays["state"], arrays["action"]
            if images.shape[1:] != (size, size, 3) or images.dtype != np.uint8:
                raise ValueError(f"{episode['name']} 图像形状 {images.shape} 或类型 {images.dtype} 不符合")
            if state.shape != (len(images), 7) or action.shape != state.shape:
                raise ValueError(f"{episode['name']} 数值形状不符合")
            _add_episode(dataset, images, state, action, episode["task"])
            expected.append((episode["npz"], len(state)))
        _progress(number, len(episodes))
    dataset.finalize()
    return _verify(job, expected)


def _versions() -> dict:
    """记录 LeRobot 与 NumPy 版本。"""
    import numpy as np

    try:
        from importlib.metadata import version

        lerobot_version = version("lerobot")
    except Exception:  # 可编辑安装可能缺少分发信息。
        lerobot_version = "unknown"
    return {"lerobot": lerobot_version, "numpy": np.__version__, "python": sys.version.split()[0]}


def main(argv: list) -> int:
    """读取任务 JSON 执行写入或预检，结果写入 result JSON，返回退出码。"""
    if len(argv) != 2:
        print("用法：lerobot_writer.py <job.json> <result.json>", file=sys.stderr)
        return 2
    job_path, result_path = Path(argv[0]), Path(argv[1])
    result: dict = {"status": "failed"}
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if Path(job["root"]).exists():
            raise FileExistsError(f"{job['root']} 已存在，拒绝覆盖")
        mode = job.get("mode")
        if mode == "preflight":
            summary = _preflight(job)
        elif mode == "write":
            summary = _write(job)
        else:
            raise ValueError(f"未知模式 {mode!r}")
        result = {"status": "succeeded", "versions": _versions(), **summary}
        code = 0
    except Exception as error:
        traceback.print_exc()
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        code = 1
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
