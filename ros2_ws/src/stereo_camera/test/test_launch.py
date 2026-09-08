"""验证 stereo_camera 启动文件的统一标定接口和参数转发。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
from launch_ros.actions import Node


# stereo_camera 源码包根目录。
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# 待检查的 ROS 2 启动文件。
LAUNCH_FILE = PACKAGE_ROOT / "launch" / "stereo_camera.launch.py"


def _load_launch_module():
    """加载启动文件并返回已执行的 Python 模块。"""
    # 使用文件路径构造模块加载规格。
    module_spec = spec_from_file_location("stereo_camera_launch", LAUNCH_FILE)
    assert module_spec is not None
    assert module_spec.loader is not None
    # 根据加载规格创建目标模块。
    launch_module = module_from_spec(module_spec)
    module_spec.loader.exec_module(launch_module)
    return launch_module


def _argument_default(argument: DeclareLaunchArgument) -> str:
    """解析不依赖运行时上下文的 launch 参数默认值。"""
    return perform_substitutions(LaunchContext(), argument.default_value)


def test_launch_defaults_and_node_interface() -> None:
    """验证默认设备、节点名称、命名空间和统一标定参数转发。"""
    # 生成待检查的启动描述。
    launch_description = _load_launch_module().generate_launch_description()
    # 按名称索引所有顶层启动参数。
    arguments = {
        entity.name: entity
        for entity in launch_description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    # 查找唯一的双目相机驱动节点。
    camera_node = next(
        entity
        for entity in launch_description.entities
        if isinstance(entity, Node)
        and entity.node_executable == "stereo_camera_node"
    )

    assert _argument_default(arguments["device_path"]) == "/dev/video0"
    # 默认标定必须指向包内保留的统一 YAML 文件。
    calibration_path = Path(_argument_default(arguments["calibration_file"]))
    assert calibration_path.name == "calib.yaml"
    assert calibration_path.is_file()
    assert camera_node.node_package == "stereo_camera"
    assert camera_node._Node__node_name == "stereo_camera_node"
    assert camera_node._Node__node_namespace == "stereo_camera"

    # 唯一参数项保存 launch 参数对节点参数的覆盖映射。
    assert len(camera_node._Node__parameters) == 1
    parameter_overrides = camera_node._Node__parameters[0]
    # 参数名称已经被 launch 规范化为 substitution 序列。
    parameter_names = {
        perform_substitutions(LaunchContext(), parameter_name)
        for parameter_name in parameter_overrides
    }
    assert parameter_names == {
        "device_path",
        "calibration_file",
    }

    # 启动文件不能再加载旧参数文件。
    assert "stereo_camera.yaml" not in LAUNCH_FILE.read_text(encoding="utf-8")
