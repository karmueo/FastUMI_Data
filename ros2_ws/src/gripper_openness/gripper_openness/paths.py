"""统一使用 ROS 2 安装包共享目录中的夹爪标定路径。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory


def default_gripper_calibration_path() -> str:
    """返回当前 ROS 2 环境中 gripper_openness 包的默认夹爪标定路径。

    文件缺失时仍返回安装共享目录下的路径，由调用方报告缺失。
    """
    return str(
        Path(get_package_share_directory("gripper_openness"))
        / "config/gripper_calibration.yaml"
    )
