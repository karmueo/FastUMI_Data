"""验证 VIVE Tracker launch 文件的节点参数合并规则."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
    launch_context.launch_configurations['publish_rate_hz'] = ''
    launch_context.launch_configurations['path_publish_rate_hz'] = ''

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
    launch_context.launch_configurations['publish_rate_hz'] = ''
    launch_context.launch_configurations['path_publish_rate_hz'] = ''

    # 节点最终收到的参数来源列表。
    tracker_parameters = launch_module._build_tracker_parameters(
        launch_context
    )

    assert len(tracker_parameters) == 2
    assert tracker_parameters[1] == {'serial': 'LHR-CUSTOM'}


def test_empty_reorder_override_preserves_config_file():
    """未传入坐标重排开关时应保留参数文件中的设置."""
    # 待测试的 launch 模块。
    launch_module = _load_launch_module()
    # 模拟仅使用自定义参数文件的 launch 上下文。
    launch_context = LaunchContext()
    launch_context.launch_configurations['config_file'] = '/tmp/custom.yaml'
    launch_context.launch_configurations['serial'] = ''
    launch_context.launch_configurations['publish_rate_hz'] = ''
    launch_context.launch_configurations['path_publish_rate_hz'] = ''
    launch_context.launch_configurations['reorder_pose_axes'] = ''

    # 节点最终收到的参数来源列表。
    tracker_parameters = launch_module._build_tracker_parameters(
        launch_context
    )

    assert len(tracker_parameters) == 1
    assert tracker_parameters[0].perform(launch_context) == '/tmp/custom.yaml'


def test_explicit_true_enables_pose_axis_reordering():
    """显式传入 true 时应使用布尔值启用坐标重排."""
    # 待测试的 launch 模块。
    launch_module = _load_launch_module()
    # 模拟显式启用坐标重排的 launch 上下文。
    launch_context = LaunchContext()
    launch_context.launch_configurations['config_file'] = '/tmp/custom.yaml'
    launch_context.launch_configurations['serial'] = ''
    launch_context.launch_configurations['publish_rate_hz'] = ''
    launch_context.launch_configurations['path_publish_rate_hz'] = ''
    launch_context.launch_configurations['reorder_pose_axes'] = 'true'

    # 节点最终收到的参数来源列表。
    tracker_parameters = launch_module._build_tracker_parameters(
        launch_context
    )

    assert len(tracker_parameters) == 2
    assert tracker_parameters[1] == {'reorder_pose_axes': True}


def test_explicit_false_disables_configured_pose_axis_reordering():
    """显式传入 false 时应使用布尔值覆盖参数文件设置."""
    # 待测试的 launch 模块。
    launch_module = _load_launch_module()
    # 模拟显式关闭坐标重排的 launch 上下文。
    launch_context = LaunchContext()
    launch_context.launch_configurations['config_file'] = '/tmp/custom.yaml'
    launch_context.launch_configurations['serial'] = ''
    launch_context.launch_configurations['publish_rate_hz'] = ''
    launch_context.launch_configurations['path_publish_rate_hz'] = ''
    launch_context.launch_configurations['reorder_pose_axes'] = 'false'

    # 节点最终收到的参数来源列表。
    tracker_parameters = launch_module._build_tracker_parameters(
        launch_context
    )

    assert len(tracker_parameters) == 2
    assert tracker_parameters[1] == {'reorder_pose_axes': False}


def test_terminal_interrupt_is_forwarded_to_isolated_tracker():
    """终端 Ctrl+C 应生成一个发往独立 Tracker 会话的信号事件."""
    # 待测试的 launch 模块。
    launch_module = _load_launch_module()
    # 模拟由终端 SIGINT 触发的 launch 关闭事件。
    shutdown_event = SimpleNamespace(due_to_sigint=True)
    # 交互模式下 launch 假定子进程已从终端接收 SIGINT。
    launch_context = SimpleNamespace(noninteractive=False)
    # 匹配信号目标时使用的占位节点动作。
    tracker_node = object()

    # 关闭处理器返回的 launch 动作。
    shutdown_actions = launch_module._forward_terminal_interrupt(
        shutdown_event, launch_context, tracker_node
    )

    assert len(shutdown_actions) == 1


def test_nonterminal_shutdown_does_not_duplicate_tracker_signal():
    """程序化关闭应沿用 launch 默认信号，避免向节点重复发送."""
    # 待测试的 launch 模块。
    launch_module = _load_launch_module()
    # 模拟由程序逻辑触发的 launch 关闭事件。
    shutdown_event = SimpleNamespace(due_to_sigint=False)
    # 交互状态不影响程序化关闭的默认信号路径。
    launch_context = SimpleNamespace(noninteractive=False)

    # 程序化关闭不需要附加信号动作。
    shutdown_actions = launch_module._forward_terminal_interrupt(
        shutdown_event, launch_context, object()
    )

    assert shutdown_actions == []


def test_noninteractive_interrupt_uses_launch_default_signal():
    """非交互启动应沿用 launch 已有的 SIGINT 转发，避免重复发送."""
    # 待测试的 launch 模块。
    launch_module = _load_launch_module()
    # 模拟非交互模式收到 SIGINT 的关闭事件和上下文。
    shutdown_event = SimpleNamespace(due_to_sigint=True)
    launch_context = SimpleNamespace(noninteractive=True)

    # launch 默认处理器会在非交互模式发送 SIGINT。
    shutdown_actions = launch_module._forward_terminal_interrupt(
        shutdown_event, launch_context, object()
    )

    assert shutdown_actions == []


def test_empty_rate_overrides_preserve_config_file():
    """未传入频率覆盖值时应只保留配置文件参数来源."""
    # 待测试的 launch 模块。
    launch_module = _load_launch_module()
    # 模拟全部覆盖项均为空的 launch 上下文。
    launch_context = LaunchContext()
    launch_context.launch_configurations.update({
        'config_file': '/tmp/custom.yaml',
        'serial': '',
        'reorder_pose_axes': '',
        'publish_rate_hz': '',
        'path_publish_rate_hz': '',
    })

    # 节点最终收到的参数来源列表。
    tracker_parameters = launch_module._build_tracker_parameters(
        launch_context
    )

    assert len(tracker_parameters) == 1


def test_explicit_rate_overrides_are_numeric():
    """显式频率覆盖应以数值参数覆盖配置文件并支持 60 Hz/10 Hz."""
    # 待测试的 launch 模块。
    launch_module = _load_launch_module()
    # 模拟调用者显式指定采样和 Path 更新频率。
    launch_context = LaunchContext()
    launch_context.launch_configurations.update({
        'config_file': '/tmp/custom.yaml',
        'serial': '',
        'reorder_pose_axes': '',
        'publish_rate_hz': '60.0',
        'path_publish_rate_hz': '10.0',
    })

    # 节点最终收到的参数来源列表。
    tracker_parameters = launch_module._build_tracker_parameters(
        launch_context
    )

    assert tracker_parameters[1] == {'publish_rate_hz': 60.0}
    assert tracker_parameters[2] == {'path_publish_rate_hz': 10.0}
