"""仅添加源码路径；测试默认不加载 ROS、Torch 或真实模型。"""

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[2]
for package in ("dp_infer", "dp_infer_lerobot", "fastumi_rm75"):
    sys.path.insert(0, str(SRC / package))
