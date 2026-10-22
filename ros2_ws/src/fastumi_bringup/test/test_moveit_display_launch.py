"""验证 MoveIt2 显示入口保持只读：move_group 不允许执行轨迹且不声明控制器。"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch_ros.actions import Node
import pytest


def module():
    spec = spec_from_file_location(
        'moveit_display_launch', Path(__file__).parents[1] / 'launch/moveit_display.launch.py')
    result = module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def move_group_parameters():
    pytest.importorskip('moveit_configs_utils')
    try:
        get_package_share_directory('rm_75_config')
    except PackageNotFoundError:
        pytest.skip('未构建 rm_75_config')
    mod = module()
    captured = {}

    def fake_node(**kwargs):
        captured[kwargs['executable']] = kwargs
        return Node(**kwargs)

    mod.Node = fake_node
    mod.generate_launch_description()
    merged = {}
    for parameters in captured['move_group']['parameters']:
        merged.update(parameters)
    return merged


def test_move_group_cannot_execute(move_group_parameters):
    assert move_group_parameters['allow_trajectory_execution'] is False
    assert move_group_parameters['moveit_manage_controllers'] is False
    assert 'moveit_simple_controller_manager' not in move_group_parameters
