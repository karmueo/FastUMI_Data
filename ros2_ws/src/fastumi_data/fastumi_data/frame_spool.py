"""把 episode 内解码后的 RGB 帧暂存到磁盘，避免整段原分辨率图像常驻内存。

30 Hz、1080p、60 秒仅源 RGB 就约 11 GB，因此转换时逐帧追加到工作目录下的临时
HDF5（每帧一个 chunk、LZF 压缩），同步完成后再按所选位置逐帧读回并写出。
内存占用只与单帧大小相关。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np


class FrameSpool:
    """单条 episode 的磁盘帧暂存；要求所有帧形状一致的 ``(H, W, 3)`` uint8。"""

    def __init__(self, path: Path) -> None:
        """在 ``path`` 创建新的暂存文件（已存在时覆盖）。"""
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: Optional[h5py.File] = h5py.File(self._path, "w")
        self._frames: Optional[h5py.Dataset] = None
        self._timestamps: List[int] = []
        self._source_indices: List[int] = []

    @property
    def path(self) -> Path:
        """暂存文件路径。"""
        return self._path

    @property
    def timestamps_ns(self) -> np.ndarray:
        """按追加顺序的源图像时间。"""
        return np.asarray(self._timestamps, dtype=np.int64)

    @property
    def source_indices(self) -> np.ndarray:
        """按追加顺序的源 bag 解码图像序号。"""
        return np.asarray(self._source_indices, dtype=np.int64)

    @property
    def frame_shape(self) -> Optional[tuple]:
        """帧形状；尚无帧时为 None。"""
        return None if self._frames is None else tuple(self._frames.shape[1:])

    def __len__(self) -> int:
        """已暂存帧数。"""
        return len(self._timestamps)

    def append(self, stamp_ns: int, source_index: int, image_rgb: np.ndarray) -> int:
        """追加一帧并返回其位置。

        Raises:
            ValueError: 图像不是 uint8 三通道，或与此前帧尺寸不同。
        """
        if self._file is None:
            raise RuntimeError("帧暂存已关闭")
        image = np.asarray(image_rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"暂存帧必须是 (H, W, 3) uint8，当前 {image.shape} {image.dtype}")
        if self._frames is None:
            self._frames = self._file.create_dataset(
                "frames",
                shape=(0,) + image.shape,
                maxshape=(None,) + image.shape,
                dtype=np.uint8,
                chunks=(1,) + image.shape,
                compression="lzf",
            )
        elif tuple(self._frames.shape[1:]) != image.shape:
            raise ValueError(
                f"同一 episode 的图像尺寸变化：{tuple(self._frames.shape[1:])} → {image.shape}"
            )
        position = len(self._timestamps)
        self._frames.resize(position + 1, axis=0)
        self._frames[position] = image
        self._timestamps.append(int(stamp_ns))
        self._source_indices.append(int(source_index))
        return position

    def read(self, position: int) -> np.ndarray:
        """读回指定位置的 RGB 帧。"""
        if self._frames is None:
            raise IndexError("帧暂存为空")
        return self._frames[int(position)]

    def close(self, delete: bool = True) -> None:
        """关闭文件，默认删除暂存数据。"""
        if self._file is not None:
            self._file.close()
            self._file = None
        if delete:
            self._path.unlink(missing_ok=True)

    def __enter__(self) -> "FrameSpool":
        """支持 with 语句。"""
        return self

    def __exit__(self, *_exc) -> None:
        """退出时关闭并删除暂存。"""
        self.close()
