"""从 ROS 入口切换到包含 Jetson PyTorch 的模型虚拟环境。"""

import os
from pathlib import Path
import sys

from ament_index_python.packages import get_package_share_directory


def main():
    """保持 ROS 参数，使用模型环境执行纯推理节点。"""
    override = os.environ.get("DP_INFER_PYTHON", "").strip()
    if override:
        executable = Path(override).expanduser().absolute()
    else:
        share = Path(get_package_share_directory("dp_infer"))
        executable = next((
            base / "model/dp/.venv/bin/python"
            for base in (share, *share.parents)
            if (base / "model/dp/.venv/bin/python").is_file()
        ), None)
    if executable is None or not executable.is_file():
        raise FileNotFoundError("model/dp/.venv/bin/python is missing; set DP_INFER_PYTHON")
    os.execv(str(executable), [str(executable), "-m", "dp_infer.node", *sys.argv[1:]])
