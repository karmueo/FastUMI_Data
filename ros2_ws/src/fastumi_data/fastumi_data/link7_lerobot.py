"""在 ROS 环境中把示教 HDF5 整理为 LeRobot 写入器使用的逐 episode ``.npz``。

本模块完成全部数值和图像处理：四元数→旋转向量、丢帧补齐、按超限缺口切分 episode
和 224×224 letterbox（转换器在 LeRobot 模式下已在暂存前 letterbox，此处对 224 输入是
逐像素恒等复制，保留调用以兼容原分辨率 HDF5）。LeRobot 子进程
（:mod:`fastumi_data.lerobot_writer`）只读取这些 ``.npz``，不再做任何变换。

补帧规则（依据 HDF5 的 ``dropped_frames_before``，即某帧之前缺失的名义帧数）：

- 缺失帧数 ≤ ``max_fill_gap`` 时在两侧有效帧之间补齐；超过时在缺口处切分 episode，
  ``0`` 表示所有缺口都切分。
- 图像取最近邻，与两侧等距时取较早帧；位置线性插值；姿态四元数 SLERP 后再转为
  旋转向量；二值夹爪（最近邻来源）取最近邻，规则同图像；连续夹爪线性插值。
- 补齐后片段少于 ``MINIMUM_LINK7_FRAMES`` 帧时丢弃。
- 切分出的片段重新从零计时，但位姿仍位于原 episode 起始 UMI TCP 坐标系，并保留
  每帧到源 HDF5 行号的映射。不外推首尾。

``.npz`` 字段（M 为补齐后帧数）::

    images               (M, 224, 224, 3) uint8 RGB
    state                (M, 7) float32 [x, y, z, rx, ry, rz, gripper]
    action               (M, 7) float32，与 state 相同（同一图像时刻的示教位姿）
    source_row           (M,) int64 图像取自的源 HDF5 行号
    filled               (M,) bool  是否为补齐帧
    source_image_index   (M,) int64 图像对应的源 bag 解码序号
    nominal_frame        (M,) int64 在原 episode 名义时间轴上的帧号（30 Hz）
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from fastumi_data.gripper_source import INTERPOLATION_NEAREST
from fastumi_data.link7_arrays import IMAGE_SIZE, letterbox_rgb_uint8, quaternion_pose_to_rotvec
from fastumi_data.link7_hdf5 import HDF5_SCHEMA_VERSION, iter_link7_images, read_link7_arrays
from fastumi_data.link7_sync import MINIMUM_LINK7_FRAMES


# 数据集格式标识与固定频率。
LEROBOT_FORMAT = "lerobot-v3.0"
LEROBOT_FPS = 30
# state/action 各维名称，写入 LeRobot features。
POSE_NAMES = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
# 片段过短时的舍弃原因代码。
SEGMENT_TOO_SHORT = "SEGMENT_TOO_SHORT"
# 写入 ``meta/fastumi.json`` 的固定契约。
LEROBOT_CONTRACT: Dict[str, Any] = {
    "source_format": HDF5_SCHEMA_VERSION,
    "fps": LEROBOT_FPS,
    "state_layout": "xyz_rotvec_gripper",
    "action_layout": "xyz_rotvec_gripper",
    "action_source": "demonstration_trajectory",
    "state_time_reference": "image_timestamp",
    "action_time_reference": "image_timestamp",
    "pose_frame": "episode_start_tcp",
    "end_frame": "tcp",
    "tcp_definition": "gripper_grasp_center",
    "deployment_tool_offset": "deployment_calibration",
    "position_unit": "m",
    "rotation_unit": "rad",
    "gripper_representation": "normalized_0_1",
    "image_size": IMAGE_SIZE,
    "image_preprocessing": "letterbox_inter_area_black_border",
    "frame_gap_policy": "fill_short_gaps_split_long_gaps",
}


@dataclass(frozen=True)
class FilledSegment:
    """一个补齐后的片段：数值和到源 HDF5 的映射，尚不含图像。

    Attributes:
        rows: 源 HDF5 行区间 ``(start, stop)``，左闭右开。
        split_index: 在原 episode 切分结果中的序号（含被舍弃片段）。
        state: ``(M, 7)`` float64 ``[xyz, 旋转向量, 夹爪]``。
        source_row: ``(M,)`` 图像取自的源行号。
        filled: ``(M,)`` 是否补齐帧。
        nominal_frame: ``(M,)`` 原 episode 名义帧号。
    """

    rows: Tuple[int, int]
    split_index: int
    state: np.ndarray
    source_row: np.ndarray
    filled: np.ndarray
    nominal_frame: np.ndarray

    def __len__(self) -> int:
        """补齐后的帧数。"""
        return int(len(self.state))


def split_rows(dropped_frames_before: np.ndarray, max_fill_gap: int) -> List[Tuple[int, int]]:
    """在缺失帧数超过 ``max_fill_gap`` 的位置切分，返回源行区间列表。

    Args:
        dropped_frames_before: ``(N,)`` 每行之前缺失的名义帧数。
        max_fill_gap: 允许补齐的最大连续缺失帧数，非负。

    Raises:
        ValueError: ``max_fill_gap`` 为负或缺失帧数为负。
    """
    if max_fill_gap < 0:
        raise ValueError("max_fill_gap 不能为负数")
    dropped = np.asarray(dropped_frames_before, dtype=np.int64)
    if np.any(dropped < 0):
        raise ValueError("dropped_frames_before 含负值")
    count = len(dropped)
    cuts = [int(index) for index in np.flatnonzero(dropped > max_fill_gap) if index > 0]
    bounds = [0] + cuts + [count]
    return [(start, stop) for start, stop in zip(bounds[:-1], bounds[1:]) if stop > start]


def fill_rows(
    qpos: np.ndarray,
    dropped_frames_before: np.ndarray,
    rows: Tuple[int, int],
    gripper_nearest: bool,
    split_index: int = 0,
) -> FilledSegment:
    """补齐 ``rows`` 区间内的全部缺口（调用方保证缺口不超过上限）。

    Args:
        qpos: ``(N, 8)`` ``[xyz, 四元数 xyzw, 夹爪]``，episode 起始 UMI TCP 系。
        dropped_frames_before: ``(N,)`` 每行之前缺失的名义帧数。
        rows: 源行区间 ``(start, stop)``。
        split_index: 片段在原 episode 切分结果中的序号。
        gripper_nearest: 夹爪是否按最近邻补齐（二值夹爪）；否则线性插值。
    """
    start, stop = rows
    values = np.asarray(qpos, dtype=np.float64)
    dropped = np.asarray(dropped_frames_before, dtype=np.int64)
    # 源行在原 episode 名义时间轴上的帧号：首行为 0，其后累加 1 + 缺失帧数。
    steps = np.ones(len(dropped), dtype=np.int64)
    steps[0] = 0
    steps[1:] += dropped[1:]
    row_frames = np.cumsum(steps)
    positions: List[np.ndarray] = []
    quaternions: List[np.ndarray] = []
    grippers: List[np.ndarray] = []
    source_rows: List[np.ndarray] = []
    filled_flags: List[np.ndarray] = []
    for row in range(start, stop):
        gap = int(dropped[row]) if row > start else 0
        if gap:
            previous = row - 1
            # 补齐帧在两侧有效帧之间的比例 j/(gap+1)，j = 1..gap。
            alpha = np.arange(1, gap + 1, dtype=np.float64) / (gap + 1)
            positions.append(
                values[previous, :3] + alpha[:, None] * (values[row, :3] - values[previous, :3])
            )
            slerp = Slerp([0.0, 1.0], Rotation.from_quat(values[[previous, row], 3:7]))
            quaternions.append(slerp(alpha).as_quat())
            # 等距（alpha=0.5）时取较早帧。
            nearest = np.where(alpha <= 0.5, previous, row)
            if gripper_nearest:
                grippers.append(values[nearest, 7])
            else:
                grippers.append(values[previous, 7] + alpha * (values[row, 7] - values[previous, 7]))
            source_rows.append(nearest.astype(np.int64))
            filled_flags.append(np.ones(gap, dtype=bool))
        positions.append(values[row:row + 1, :3])
        quaternions.append(values[row:row + 1, 3:7])
        grippers.append(values[row:row + 1, 7])
        source_rows.append(np.asarray([row], dtype=np.int64))
        filled_flags.append(np.zeros(1, dtype=bool))
    qpos_filled = np.concatenate(
        (
            np.concatenate(positions),
            np.concatenate(quaternions),
            np.concatenate(grippers)[:, None],
        ),
        axis=1,
    )
    count = len(qpos_filled)
    return FilledSegment(
        rows=(start, stop),
        split_index=split_index,
        state=quaternion_pose_to_rotvec(qpos_filled),
        source_row=np.concatenate(source_rows),
        filled=np.concatenate(filled_flags),
        nominal_frame=row_frames[start] + np.arange(count, dtype=np.int64),
    )


def plan_filled_segments(
    qpos: np.ndarray,
    dropped_frames_before: np.ndarray,
    max_fill_gap: int,
    gripper_nearest: bool,
    minimum_frames: int = MINIMUM_LINK7_FRAMES,
) -> Tuple[List[FilledSegment], List[Dict[str, Any]]]:
    """切分并补齐一条 episode，返回保留的片段和被舍弃片段的说明。

    片段长度在补齐之后检查，少于 ``minimum_frames`` 帧的片段被舍弃。
    """
    kept: List[FilledSegment] = []
    dropped_segments: List[Dict[str, Any]] = []
    for index, rows in enumerate(split_rows(dropped_frames_before, max_fill_gap)):
        segment = fill_rows(qpos, dropped_frames_before, rows, gripper_nearest, index)
        if len(segment) < minimum_frames:
            dropped_segments.append(
                {
                    "code": SEGMENT_TOO_SHORT,
                    "segment_index": index,
                    "source_rows": list(rows),
                    "frames_after_fill": len(segment),
                    "minimum_frames": minimum_frames,
                }
            )
            continue
        kept.append(segment)
    return kept, dropped_segments


def _check_segment(name: str, state: np.ndarray) -> None:
    """检查补齐结果为有限值且夹爪位于 [0,1]。"""
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{name} 的 state 含非有限值")
    gripper = state[:, 6]
    if np.any(gripper < 0.0) or np.any(gripper > 1.0):
        raise ValueError(f"{name} 的夹爪开度超出 [0,1]")


def resolve_task(episode_task: Any, record_task: Any, record_id: str) -> str:
    """按 episode 任务名、记录任务名、记录标识的顺序选取非空任务描述。"""
    for candidate in (episode_task, record_task):
        text = str(candidate or "").strip()
        if text:
            return text
    return record_id


def prepare_lerobot_episode(
    hdf5_path: Path,
    episode_name: str,
    npz_dir: Path,
    max_fill_gap: int,
    gripper_interpolation: str,
    task: str,
    relative_to: Optional[Path] = None,
) -> Dict[str, Any]:
    """把一条 HDF5 episode 写成一个或多个 ``.npz`` 片段。

    Args:
        hdf5_path: ``fastumi_ros2_tcp_v1`` episode 文件。
        episode_name: 原 episode 唯一名称，片段名在其后追加 ``/segment_XX``（仅切分时）。
        npz_dir: 片段输出目录，不存在时创建。
        max_fill_gap: 允许补齐的最大连续缺失帧数。
        gripper_interpolation: 夹爪来源的插值方式，``nearest`` 表示二值夹爪。
        task: LeRobot 任务描述。
        relative_to: 写入溯源信息时 HDF5 路径相对的目录（通常为输出根目录）。

    Returns:
        ``{"segments": [...], "dropped_segments": [...]}``；每个片段含 ``npz`` 路径、帧数、
        任务和溯源字段，均可跨进程传递和 JSON 序列化。
    """
    hdf5_path = Path(hdf5_path)
    arrays = read_link7_arrays(hdf5_path)
    qpos = np.asarray(arrays["qpos"], dtype=np.float64)
    dropped = np.asarray(arrays["dropped_frames_before"], dtype=np.int64)
    segments, dropped_segments = plan_filled_segments(
        qpos, dropped, max_fill_gap, gripper_interpolation == INTERPOLATION_NEAREST
    )
    split = len(split_rows(dropped, max_fill_gap)) > 1
    if relative_to is not None:
        try:
            source_hdf5 = os.path.relpath(hdf5_path, relative_to)
        except ValueError:
            source_hdf5 = str(hdf5_path)
    else:
        source_hdf5 = str(hdf5_path)
    for item in dropped_segments:
        item["source_episode"] = episode_name
    npz_dir = Path(npz_dir)
    npz_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(episode_name).name
    results: List[Dict[str, Any]] = []
    for segment in segments:
        number = segment.split_index
        name = f"{episode_name}/segment_{number:02d}" if split else episode_name
        _check_segment(name, segment.state)
        start, stop = segment.rows
        # 只解码本片段用到的源帧，逐帧 letterbox 后按 source_row 重复取用补齐帧。
        boxed = np.stack(
            [letterbox_rgb_uint8(frame) for _, frame in iter_link7_images(hdf5_path, start, stop)]
        )
        if len(boxed) != stop - start:
            raise ValueError(f"{name} 图像 {len(boxed)} 帧与源行区间 {stop - start} 帧不一致")
        state = segment.state.astype(np.float32)
        path = npz_dir / f"{stem}_segment_{number:02d}.npz"
        np.savez(
            path,
            images=boxed[segment.source_row - start],
            state=state,
            action=state.copy(),
            source_row=segment.source_row,
            filled=segment.filled,
            source_image_index=np.asarray(arrays["source_image_index"], dtype=np.int64)[
                segment.source_row
            ],
            nominal_frame=segment.nominal_frame,
        )
        results.append(
            {
                "name": name,
                "npz": str(path),
                "frames": len(segment),
                "task": task,
                "source_episode": episode_name,
                "source_hdf5": source_hdf5,
                "source_rows": [start, stop],
                "segment_index": number,
                "split": split,
                "filled_frames": np.flatnonzero(segment.filled).tolist(),
                "nominal_frame_start": int(segment.nominal_frame[0]),
            }
        )
    return {"segments": results, "dropped_segments": dropped_segments}
