"""仅添加源码路径；离线测试默认不加载 ROS 或真实模型。"""

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[2]
for package in ("dp_infer", "dp_infer_lerobot"):
    sys.path.insert(0, str(SRC / package))
sys.path.insert(0, str(Path(__file__).resolve().parent))
