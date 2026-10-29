#!/usr/bin/env python3
"""分析、复核和导出抓取/放置两阶段数据；从仓库根目录运行。"""

from pathlib import Path
import sys

# 优先使用工作区源码，避免尚未重新构建的 ROS 安装副本遮蔽实现。
sys.path.insert(0, str(Path(__file__).resolve().parent / "ros2_ws/src/fastumi_data"))

from pick_place.pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
