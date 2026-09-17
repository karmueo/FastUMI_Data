"""启动单个 RM75 Placo 关节控制节点。"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """使用工作区的 NumPy 2 解释器启动 Placo 节点。"""
    package_share = Path(get_package_share_directory("fastumi_rm75"))
    python_candidates = (
        parent / ".venv-numpy2" / "bin" / "python"
        for parent in (package_share, *package_share.parents)
    )
    python_executable = next(
        (candidate for candidate in python_candidates if candidate.is_file()),
        None,
    )
    if python_executable is None:
        raise RuntimeError("未找到 ros2_ws/.venv-numpy2；请先按工作区 README 创建环境")
    default_config = PathJoinSubstitution(
        [FindPackageShare("fastumi_rm75"), "config", "rm75_placo_controller.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument(
                "python_executable", default_value=str(python_executable)
            ),
            ExecuteProcess(
                cmd=[
                    LaunchConfiguration("python_executable"),
                    "-m",
                    "fastumi_rm75.rm75_placo_controller",
                    "--ros-args",
                    "--params-file",
                    LaunchConfiguration("config_file"),
                ],
                output="screen",
            ),
        ]
    )
