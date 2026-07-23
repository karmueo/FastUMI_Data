"""验证 VIVE Tracker launch 文件的节点参数合并规则."""

import importlib.util
from pathlib import Path

from launch import LaunchContext


def _load_launch_module():
    """从源码路径加载待测试的 launch 模块."""
    # launch 文件的源码路径。
    launch_path = (
        Path(__file__).parents[1] / 'launch' / 'vive_tracker.launch.py'
    )
    # 用于加载 launch 文件的模块描述。
    module_spec = importlib.util.spec_from_file_location(
        'vive_tracker_launch', launch_path
    )
    # 待测试的 launch 模块。
    launch_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(launch_module)
    return launch_module


def test_config_file_serial_is_preserved_without_override():
    """未传入序列号时应仅使用参数文件，保留文件中的 serial."""
    # 待测试的 launch 模块。
    launch_module = _load_launch_module()
    # 模拟未显式传入 serial 的 launch 上下文。
    launch_context = LaunchContext()
    launch_context.launch_configurations['config_file'] = '/tmp/custom.yaml'
    launch_context.launch_configurations['serial'] = ''

    # 节点最终收到的参数来源列表。
    tracker_parameters = launch_module._build_tracker_parameters(
        launch_context
    )

    assert len(tracker_parameters) == 1
    assert tracker_parameters[0].perform(launch_context) == '/tmp/custom.yaml'


def test_explicit_serial_overrides_config_file():
    """显式传入非空序列号时应追加高优先级覆盖参数."""
    # 待测试的 launch 模块。
    launch_module = _load_launch_module()
    # 模拟显式传入 serial 的 launch 上下文。
    launch_context = LaunchContext()
    launch_context.launch_configurations['config_file'] = '/tmp/custom.yaml'
    launch_context.launch_configurations['serial'] = 'LHR-CUSTOM'

    # 节点最终收到的参数来源列表。
    tracker_parameters = launch_module._build_tracker_parameters(
        launch_context
    )

    assert len(tracker_parameters) == 2
    assert tracker_parameters[1] == {'serial': 'LHR-CUSTOM'}
