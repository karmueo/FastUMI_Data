"""验证 FastUMI 设备数据节点统一 launch 的结构和参数。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)


# 当前测试文件对应的 fastumi_data 包根目录。
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# 待验证的统一采集 launch 文件。
LAUNCH_FILE = PACKAGE_ROOT / "launch" / "fastumi_collection.launch.py"


def _load_launch_module():
    """加载文件名包含点号的 Python launch 模块。

    Returns:
        已执行并可调用 `generate_launch_description` 的模块。
    """
    # 独立模块规格，避免将 launch 文件当作普通包导入。
    module_spec = spec_from_file_location("fastumi_collection_launch", LAUNCH_FILE)
    assert module_spec is not None
    assert module_spec.loader is not None
    # 根据规格创建待执行的 launch 模块。
    launch_module = module_from_spec(module_spec)
    module_spec.loader.exec_module(launch_module)
    return launch_module


def test_collection_launch_only_includes_three_device_subsystems() -> None:
    """验证统一入口只包含三个设备子 launch。"""
    # 生成待检查的统一启动描述。
    launch_description = _load_launch_module().generate_launch_description()
    # 全部顶层 launch action。
    entities = launch_description.entities
    # 三个被复用的子系统 launch。
    included_launches = [
        entity
        for entity in entities
        if isinstance(entity, IncludeLaunchDescription)
    ]
    assert len(included_launches) == 3
    assert len(entities) == 7


def test_collection_launch_only_declares_device_arguments() -> None:
    """验证统一入口只声明设备节点参数。"""
    # 生成待检查的统一启动描述。
    launch_description = _load_launch_module().generate_launch_description()
    # 按参数名索引所有顶层参数声明。
    arguments = {
        entity.name: entity
        for entity in launch_description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }

    assert set(arguments) == {
        "camera_serial",
        "tracker_serial",
        "use_rviz",
        "publish_debug_image",
    }
    assert arguments["use_rviz"].choices == ["true", "false"]
