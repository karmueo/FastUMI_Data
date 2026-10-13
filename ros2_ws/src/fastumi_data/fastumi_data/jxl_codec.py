"""加载仓库唯一的 ``imagecodecs_numcodecs`` 源文件并注册 JPEG XL 等 numcodecs 编解码器。

源文件位于仓库根目录，构建时作为数据文件安装到
``share/fastumi_data/codecs/imagecodecs_numcodecs.py``；源码树运行时回退到仓库根目录，
保证安装与开发使用同一份实现。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Optional


MODULE_NAME = "imagecodecs_numcodecs"
_module: Optional[ModuleType] = None


def _candidate_paths() -> list[Path]:
    """返回安装目录和源码树中可能的编解码器源文件路径。"""
    candidates = []
    try:
        from ament_index_python.packages import (
            PackageNotFoundError,
            get_package_share_directory,
        )

        try:
            candidates.append(
                Path(get_package_share_directory("fastumi_data")) / "codecs" / f"{MODULE_NAME}.py"
            )
        except PackageNotFoundError:
            pass
    except ImportError:
        pass
    # 源码树：ros2_ws/src/fastumi_data/fastumi_data/ → 仓库根目录。
    candidates.append(Path(__file__).resolve().parents[4] / f"{MODULE_NAME}.py")
    return candidates


def load_codecs() -> ModuleType:
    """加载并注册编解码器模块，重复调用返回同一模块。

    Raises:
        ImportError: 找不到编解码器源文件或缺少 imagecodecs/numcodecs。
    """
    global _module
    if _module is not None:
        return _module
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None and hasattr(existing, "JpegXl"):
        _module = existing
    else:
        for path in _candidate_paths():
            if not path.is_file():
                continue
            spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[MODULE_NAME] = module
            _module = module
            break
        else:
            raise ImportError(f"找不到 {MODULE_NAME}.py，请重新构建 fastumi_data")
    _module.register_codecs(verbose=False)
    return _module


def lossless_jpegxl():
    """返回无损 JPEG XL 编解码器：``lossless=True, effort=3, numthreads=1``。"""
    return load_codecs().JpegXl(lossless=True, effort=3, numthreads=1)
