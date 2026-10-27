"""在 ROS 主进程中调用 LeRobot 子进程写入数据集，并校验后原子发布。

LeRobot 安装在独立的 Python 环境中（NumPy 2），不能与 ROS 的 NumPy 1 混用。本模块：

- 解析 ``--lerobot-python``：命令行值 → 环境变量 ``FASTUMI_LEROBOT_PYTHON`` →
  ``~/work/lerobot/.venv/bin/python``，都不可用时报错，不回退到当前解释器；
- 以 ``-I`` 隔离模式启动 :mod:`fastumi_data.lerobot_writer` 脚本，并从环境中移除
  ``PYTHONPATH``/``PYTHONHOME``，避免继承 ROS 的 Python 路径；
- 写入临时目录并由子进程完成 ``finalize()`` 和回读校验后，写入 ``meta/fastumi.json``、
  确认 ``meta/`` 中没有临时目录的绝对路径，再整体重命名发布。失败时保留临时产物。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence

from fastumi_data.json_report import write_json_report
from fastumi_data.link7_arrays import IMAGE_SIZE, to_jsonable
from fastumi_data.link7_lerobot import LEROBOT_CONTRACT, LEROBOT_FORMAT, LEROBOT_FPS


# 指定 LeRobot Python 的环境变量。
LEROBOT_PYTHON_ENV = "FASTUMI_LEROBOT_PYTHON"
# 缺省 LeRobot Python：用户主目录下的 LeRobot 工作区虚拟环境。
DEFAULT_LEROBOT_PYTHON = Path("~/work/lerobot/.venv/bin/python")
# 子进程写入器脚本，与本模块同目录安装。
WRITER_SCRIPT = Path(__file__).resolve().with_name("lerobot_writer.py")
# 写入器的进度行前缀（与 lerobot_writer.PROGRESS_PREFIX 一致）。
PROGRESS_PREFIX = "FASTUMI_PROGRESS"
# 数据集中的 FastUMI 契约与溯源文件。
FASTUMI_META_NAME = "fastumi.json"
# LeRobot 元数据中记录的机器人类型。
ROBOT_TYPE = "fastumi_umi_tcp"
# CPU H.264 编码参数：GOP 为 2 便于随机访问解码，CRF 18 接近视觉无损。
H264_ENCODER: Dict[str, Any] = {"vcodec": "h264", "pix_fmt": "yuv420p", "g": 2, "crf": 18}
# 子进程环境中需要移除的变量：ROS 的 Python 路径会让 LeRobot 环境混入 NumPy 1。
_STRIPPED_ENV = ("PYTHONPATH", "PYTHONHOME")
# 进度回调：(阶段名, 已完成, 总数)。
ProgressCallback = Callable[[str, int, int], None]


class LeRobotExportError(RuntimeError):
    """LeRobot 环境不可用、写入失败或发布前校验失败。"""


def resolve_lerobot_python(
    explicit: str = "", environ: Optional[Dict[str, str]] = None,
) -> Path:
    """按命令行、环境变量、缺省路径的顺序解析 LeRobot Python。

    Raises:
        LeRobotExportError: 选中的路径不存在或不可执行；缺省路径也不可用时给出指引。
    """
    environ = os.environ if environ is None else environ
    if explicit:
        candidate, source = Path(explicit).expanduser(), "--lerobot-python"
    elif environ.get(LEROBOT_PYTHON_ENV):
        candidate, source = Path(environ[LEROBOT_PYTHON_ENV]).expanduser(), LEROBOT_PYTHON_ENV
    else:
        candidate, source = DEFAULT_LEROBOT_PYTHON.expanduser(), "缺省路径"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate.absolute()
    raise LeRobotExportError(
        f"{source} 指定的 LeRobot Python {candidate} 不存在或不可执行；"
        f"请用 --lerobot-python 或环境变量 {LEROBOT_PYTHON_ENV} 指向安装了 LeRobot 的解释器，"
        "或改用 --output-format zarr"
    )


def writer_environment(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """返回子进程环境：移除 Python 路径变量，并禁止访问 Hugging Face Hub。"""
    environment = dict(os.environ if base is None else base)
    for name in _STRIPPED_ENV:
        environment.pop(name, None)
    environment["HF_HUB_OFFLINE"] = "1"
    return environment


def run_writer(
    python: Path,
    job: Dict[str, Any],
    work_dir: Path,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """在 LeRobot 环境中执行写入器并返回结果摘要。

    子进程 stderr 与非进度 stdout 行写入 ``work_dir/writer_<mode>.log``，任务和结果 JSON
    也保留在 ``work_dir`` 供诊断。

    Raises:
        LeRobotExportError: 子进程启动失败、非零退出或结果缺失。
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    mode = str(job.get("mode"))
    job_path = work_dir / f"job_{mode}.json"
    result_path = work_dir / f"result_{mode}.json"
    log_path = work_dir / f"writer_{mode}.log"
    write_json_report(str(job_path), job)
    result_path.unlink(missing_ok=True)
    command = [str(python), "-I", str(WRITER_SCRIPT), str(job_path), str(result_path)]
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=log, text=True,
                env=writer_environment(), cwd=str(work_dir),
            )
            assert process.stdout is not None
            for line in process.stdout:
                if line.startswith(PROGRESS_PREFIX):
                    parts = line.split()
                    if on_progress is not None and len(parts) == 3:
                        on_progress(int(parts[1]), int(parts[2]))
                else:
                    log.write(line)
            code = process.wait()
    except OSError as error:
        raise LeRobotExportError(f"无法启动 LeRobot 写入器 {python}：{error}") from error
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result = {}
    if code != 0 or result.get("status") != "succeeded":
        detail = result.get("error") or f"退出码 {code}"
        raise LeRobotExportError(f"LeRobot 写入器（{mode}）失败：{detail}；日志见 {log_path}")
    return result


def preflight_lerobot(python: Path, work_dir: Optional[Path] = None) -> Dict[str, Any]:
    """在临时目录编码并解码一段测试视频，验证 LeRobot、编码器和解码后端可用。

    ``work_dir`` 缺省时使用系统临时目录，结束后删除；失败时保留并在错误信息中给出日志路径。
    """
    owned = work_dir is None
    directory = Path(tempfile.mkdtemp(prefix="fastumi_lerobot_preflight_")) if owned else Path(work_dir)
    job = {
        "mode": "preflight",
        "root": str(directory / "dataset"),
        "repo_id": "fastumi/preflight",
        "fps": LEROBOT_FPS,
        "robot_type": ROBOT_TYPE,
        "image_size": IMAGE_SIZE,
        "encoder": dict(H264_ENCODER),
    }
    try:
        result = run_writer(python, job, directory)
    except LeRobotExportError as error:
        raise LeRobotExportError(
            f"{error}。请确认该环境安装了当前 LeRobot、PyAV 支持 libx264 编码，"
            "且 torchcodec 或 PyAV 可以解码 H.264"
        ) from error
    if owned:
        shutil.rmtree(directory, ignore_errors=True)
    return result


def validate_repo_id(repo_id: str) -> str:
    """检查 ``<用户或组织>/<数据集名>`` 形式，返回去空白后的值。"""
    value = (repo_id or "").strip()
    parts = value.split("/")
    if len(parts) != 2 or not all(part.strip() for part in parts) or any(" " in part for part in parts):
        raise ValueError(f"--repo-id {repo_id!r} 必须是 <用户或组织>/<数据集名> 形式")
    return value


def _find_absolute_path(meta_dir: Path, needles: Sequence[str]) -> List[str]:
    """返回 ``meta_dir`` 下包含任一临时绝对路径字节串的文件（相对路径）。"""
    encoded = [needle.encode("utf-8") for needle in needles if needle]
    hits = []
    for path in sorted(meta_dir.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            if any(needle in data for needle in encoded):
                hits.append(path.relative_to(meta_dir).as_posix())
    return hits


def export_lerobot(
    segments: Sequence[Dict[str, Any]],
    output_root: Path,
    staging_dir: Path,
    repo_id: str,
    python: Path,
    extra_metadata: Optional[Dict[str, Any]] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """写入、校验并原子发布 LeRobot 数据集。

    Args:
        segments: :func:`fastumi_data.link7_lerobot.prepare_lerobot_episode` 返回的片段，按写入顺序。
        output_root: 发布目录（``<output-dir>/lerobot``），必须尚不存在。
        staging_dir: 临时工作目录，数据集先写到其下 ``dataset``。
        repo_id: LeRobot 数据集标识。
        python: LeRobot Python。
        extra_metadata: 合并写入 ``meta/fastumi.json`` 的批次元数据（如标定摘要）。

    Returns:
        路径、episode 数、帧数及写入器摘要。

    Raises:
        FileExistsError: 发布目录已存在。
        ValueError: 没有片段。
        LeRobotExportError: 写入、回读或路径检查失败；临时产物保留在 ``staging_dir``。
    """
    output_root = Path(output_root)
    staging_dir = Path(staging_dir)
    if output_root.exists():
        raise FileExistsError(f"{output_root} 已存在，拒绝覆盖")
    if not segments:
        raise ValueError("没有可导出的 episode")
    report = on_progress or (lambda _stage, _done, _total: None)
    temporary = staging_dir / "dataset"
    if temporary.exists():
        shutil.rmtree(temporary)
    job = {
        "mode": "write",
        "root": str(temporary),
        "repo_id": repo_id,
        "fps": LEROBOT_FPS,
        "robot_type": ROBOT_TYPE,
        "image_size": IMAGE_SIZE,
        "encoder": dict(H264_ENCODER),
        "episodes": [
            {"npz": item["npz"], "task": item["task"], "name": item["name"]} for item in segments
        ],
    }
    stage = "写入 LeRobot"
    report(stage, 0, len(segments))
    result = run_writer(python, job, staging_dir, lambda done, total: report(stage, done, total))
    expected_lengths = [int(item["frames"]) for item in segments]
    if result.get("episode_lengths") != expected_lengths:
        raise LeRobotExportError(
            f"写入器回读的 episode 长度 {result.get('episode_lengths')} 与预期 {expected_lengths} 不一致"
        )
    episodes = []
    for index, item in enumerate(segments):
        record = {key: value for key, value in item.items() if key != "npz"}
        record["episode_index"] = index
        episodes.append(record)
    fastumi_meta = dict(LEROBOT_CONTRACT)
    fastumi_meta.update(to_jsonable(extra_metadata or {}))
    fastumi_meta.update(
        {
            "format": LEROBOT_FORMAT,
            "repo_id": repo_id,
            "video_codec": H264_ENCODER["vcodec"],
            "video_encoder": dict(H264_ENCODER),
            "episodes": episodes,
        }
    )
    meta_dir = temporary / "meta"
    write_json_report(str(meta_dir / FASTUMI_META_NAME), to_jsonable(fastumi_meta))
    needles = {str(temporary), str(temporary.resolve()), str(staging_dir), str(staging_dir.resolve())}
    leaked = _find_absolute_path(meta_dir, sorted(needles, key=len, reverse=True))
    if leaked:
        raise LeRobotExportError(f"meta/ 中写入了临时目录绝对路径：{', '.join(leaked)}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.rename(temporary, output_root)
    return {
        "path": str(output_root),
        "repo_id": repo_id,
        "episodes": len(segments),
        "frames": int(sum(expected_lengths)),
        "writer": {key: result.get(key) for key in ("versions", "max_image_mean_abs_error", "video_backend")},
    }


def training_command(repo_id: str, dataset_root: Path) -> str:
    """返回在 LeRobot 工作区中训练相对位姿 Diffusion Policy 的参考命令。"""
    lines = [
        "uv run lerobot-train",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        "--policy.type=diffusion",
        "--policy.use_relative_pose=true",
        "--policy.relative_pose_state_starts='[0]'",
        "--policy.n_obs_steps=2",
        "--policy.horizon=16",
        "--policy.n_action_steps=8",
        "--policy.drop_n_last_frames=7",
        "--policy.resize_shape='[224,224]'",
        "--policy.crop_ratio=0.95",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--output_dir=outputs/train/dp_relpose",
        "--job_name=dp_relpose",
        "--batch_size=64",
        "--steps=200000",
    ]
    return " \\\n  ".join(lines)
