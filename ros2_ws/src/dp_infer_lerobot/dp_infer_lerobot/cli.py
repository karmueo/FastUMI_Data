"""ROS 入口切换到独立 Python3.10/Jetson PyTorch 环境。"""

import os
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory


def main():
    """保留 ROS 参数；允许显式覆盖解释器而不改变旧 DP 环境。"""
    override = os.environ.get("DP_INFER_LEROBOT_PYTHON", "").strip()
    if override:
        executable = Path(override).expanduser().absolute()
    else:
        share = Path(get_package_share_directory("dp_infer_lerobot"))
        executable = next((base / ".venv-dp-lerobot/bin/python"
                           for base in (share, *share.parents)
                           if (base / ".venv-dp-lerobot/bin/python").is_file()), None)
    if executable is None or not executable.is_file():
        raise FileNotFoundError("Set DP_INFER_LEROBOT_PYTHON or prepare .venv-dp-lerobot")
    os.execv(str(executable), [str(executable), "-m", "dp_infer_lerobot.node", *sys.argv[1:]])
