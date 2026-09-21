"""统一解析记录节点与管理面板使用的本机任务输出目录，不依赖 ROS 或数值库。"""

from pathlib import Path


def repository_root() -> Path:
    """从源码或符号链接安装路径定位仓库根目录，失败时提示使用绝对路径。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / "ros2_ws/src/tracker_teleoperated").is_dir():
            return parent
    raise RuntimeError("无法定位仓库根目录，请配置 dataset_root 的绝对路径")


def output_directory(dataset_root: str, dir_name: str, name: str) -> Path:
    """返回任务绝对目录；相对根目录基于仓库，非法任务名称抛出 ValueError。"""
    # 两层任务名称沿用记录节点的单层目录限制。
    parts = (str(dir_name).strip(), str(name).strip())
    if any(part in ("", ".", "..") or Path(part).name != part for part in parts):
        raise ValueError("dir_name 和 name 必须是单层非空目录名")
    # expanduser 保持原有用户目录语义，resolve 统一发布与实际写入路径。
    root = Path(str(dataset_root)).expanduser()
    if not root.is_absolute():
        root = repository_root() / root
    return (root / parts[0] / parts[1]).resolve()
