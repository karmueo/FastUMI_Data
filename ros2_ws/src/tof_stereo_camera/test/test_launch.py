"""验证 ToF 相机 launch 的传感器 QoS 参数与环境边界。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.utilities import perform_substitutions
from launch_ros.actions import Node


# ToF 相机包源码根目录。
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# 待验证的相机 launch 文件。
LAUNCH_FILE = PACKAGE_ROOT / "launch" / "tof_stereo_camera.launch.py"


def _load_launch_module():
    """加载相机 launch 模块。

    Returns:
        已执行且可生成启动描述的模块。
    """
    # 文件名包含点号，需要通过文件规格加载。
    module_spec = spec_from_file_location(
        "tof_stereo_camera_launch", LAUNCH_FILE
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    # 根据规格创建并执行 launch 模块。
    launch_module = module_from_spec(module_spec)
    module_spec.loader.exec_module(launch_module)
    return launch_module


def _argument_default(argument: DeclareLaunchArgument) -> str:
    """解析纯文本 launch 参数默认值。

    Args:
        argument: 待解析的参数声明。

    Returns:
        参数的默认字符串。
    """
    return perform_substitutions(LaunchContext(), argument.default_value)


def test_launch_defaults_optimize_sensor_qos() -> None:
    """验证传感器默认采用深度 16 的最尽力 QoS。"""
    # 生成待检查的启动描述。
    launch_description = _load_launch_module().generate_launch_description()
    # 按名称索引顶层参数声明。
    arguments = {
        entity.name: entity
        for entity in launch_description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }

    assert _argument_default(arguments["sensor_qos_depth"]) == "16"
    assert (
        _argument_default(arguments["sensor_qos_reliability"])
        == "best_effort"
    )
    assert (
        _argument_default(arguments["timestamp_future_warning_threshold_us"])
        == "1000"
    )


def test_launch_forwards_qos_without_managing_environment() -> None:
    """验证 QoS 参数传入相机节点且 launch 不修改环境。"""
    # 生成待检查的启动描述。
    launch_description = _load_launch_module().generate_launch_description()
    # launch 不应包含任何进程环境设置动作。
    environment_actions = [
        entity
        for entity in launch_description.entities
        if isinstance(entity, SetEnvironmentVariable)
    ]
    # 相机主节点。
    camera_node = next(
        entity
        for entity in launch_description.entities
        if isinstance(entity, Node)
        and entity.node_executable == "tof_stereo_camera_node"
    )

    assert not environment_actions
    # Node 的首个参数字典保存所有 launch substitutions。
    node_parameters = camera_node._Node__parameters[0]
    # ROS 2 会将参数名规范化为 substitution 序列。
    parameter_names = {
        perform_substitutions(LaunchContext(), parameter_name)
        for parameter_name in node_parameters
    }
    assert "sensor_qos_depth" in parameter_names
    assert "sensor_qos_reliability" in parameter_names
    assert "timestamp_future_warning_threshold_us" in parameter_names
