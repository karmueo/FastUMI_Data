"""验证初始位姿配置文件的读取与校验。"""
from importlib.util import module_from_spec, spec_from_file_location
import math
from pathlib import Path

import pytest


@pytest.fixture
def initial_pose(tmp_path, monkeypatch):
    """在独立 share 目录中加载模块，避免读取本机安装的回位目标。"""
    from ament_index_python import packages

    share = tmp_path / 'share'
    config = share / 'config' / 'initial_pose.yaml'
    config.parent.mkdir(parents=True)
    config.write_text(
        'initial_joint_positions_deg: [5, 10, 15, 20, 25, 30, 35]\n',
        encoding='utf-8')
    monkeypatch.setattr(packages, 'get_package_share_directory', lambda _: str(share))
    source = Path(__file__).parents[1] / 'fastumi_bringup' / 'initial_pose.py'
    spec = spec_from_file_location('isolated_initial_pose', source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installed_config_sets_initial_pose(initial_pose):
    """默认路径和模块常量均使用隔离安装目录中的自定义角度。"""
    expected = [math.radians(value) for value in (5, 10, 15, 20, 25, 30, 35)]
    assert list(initial_pose.load_initial_joint_positions()) == pytest.approx(expected)
    assert list(initial_pose.INITIAL_JOINT_POSITIONS) == pytest.approx(expected)


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
