"""验证初始位姿配置文件的读取与校验。"""
from importlib.util import module_from_spec, spec_from_file_location
import math
from pathlib import Path

import pytest


@pytest.fixture
def initial_pose(tmp_path):
    """模拟安装后的模块及过期的 share 副本，唯一有效配置位于 src。"""
    workspace = tmp_path / 'ros2_ws'
    config = workspace / 'src/fastumi_bringup/config/initial_pose.yaml'
    config.parent.mkdir(parents=True)
    config.write_text(
        'initial_joint_positions_deg: [5, 10, 15, 20, 25, 30, 35]\n',
        encoding='utf-8')
    installed = workspace / 'install/fastumi_bringup/share/fastumi_bringup/config/initial_pose.yaml'
    installed.parent.mkdir(parents=True)
    installed.write_text('initial_joint_positions_deg: [0, 0, 0, 0, 0, 0, 0]\n')
    source = Path(__file__).parents[1] / 'fastumi_bringup' / 'initial_pose.py'
    spec = spec_from_file_location('isolated_initial_pose', source)
    module = module_from_spec(spec)
    module.__file__ = str(workspace / 'install/fastumi_bringup/lib/python3.10/site-packages/'
                          'fastumi_bringup/initial_pose.py')
    spec.loader.exec_module(module)
    return module


def test_installed_module_reads_source_config(initial_pose):
    """安装后的模块必须读取源文件，忽略过期的 install/share 配置。"""
    expected = [math.radians(value) for value in (5, 10, 15, 20, 25, 30, 35)]
    assert list(initial_pose.load_initial_joint_positions()) == pytest.approx(expected)
    assert list(initial_pose.INITIAL_JOINT_POSITIONS) == pytest.approx(expected)


def test_missing_source_config_does_not_fall_back(initial_pose, tmp_path, monkeypatch):
    """源文件缺失时明确失败，不能使用安装副本或内置角度。"""
    monkeypatch.setattr(initial_pose, '__file__', str(tmp_path / 'other/install/module.py'))
    with pytest.raises(FileNotFoundError, match='initial_pose.yaml'):
        initial_pose.load_initial_joint_positions()


def test_source_config_is_reread(initial_pose):
    """配置修改后下一次加载立即生效，不必重建安装副本。"""
    path = initial_pose.default_config_path()
    path.write_text('initial_joint_positions_deg: [7, 8, 9, 10, 11, 12, 13]\n')
    assert initial_pose.load_initial_joint_positions() == pytest.approx(
        tuple(math.radians(value) for value in range(7, 14)))


def test_shipped_default_config(initial_pose):
    """源码随包配置须能加载，允许用户调整合法的回位角度。"""
    config = Path(__file__).parents[1] / 'config' / 'initial_pose.yaml'
    positions = initial_pose.load_initial_joint_positions(config)
    assert len(positions) == 7
    assert all(math.isfinite(value) for value in positions)


def test_custom_config(tmp_path, initial_pose):
    """可读取任意路径的配置。"""
    path = tmp_path / 'pose.yaml'
    path.write_text('initial_joint_positions_deg: [10, 0, 0, 0, 0, 0, -90.5]\n')
    assert initial_pose.load_initial_joint_positions(path) == pytest.approx(
        (math.radians(10), 0, 0, 0, 0, 0, math.radians(-90.5)))


@pytest.mark.parametrize('body', [
    'initial_joint_positions_deg: [0, 0, 0]\n',
    'initial_joint_positions_deg: [0, 0, 0, 0, 0, 0, a]\n',
    'initial_joint_positions_deg: [0, 0, 0, 0, 0, 0, .nan]\n',
    'other: 1\n',
    '',
])
def test_invalid_config(tmp_path, body, initial_pose):
    """长度不对、非数值或缺少字段时报错。"""
    path = tmp_path / 'pose.yaml'
    path.write_text(body)
    with pytest.raises(ValueError):
        initial_pose.load_initial_joint_positions(path)
