"""RM75 初始关节目标，按 joint1 至 joint7 排列，单位为弧度。

数值来自 config/initial_pose.yaml（单位为度），优先读取安装后的 share 目录，
包未安装时回退到源码目录。
"""

import math
from pathlib import Path

import yaml


CONFIG_NAME = 'initial_pose.yaml'
JOINT_COUNT = 7


def default_config_path():
    """返回安装后的配置路径；找不到时返回源码目录中的配置。"""
    try:
        from ament_index_python.packages import get_package_share_directory
        installed = Path(get_package_share_directory('fastumi_bringup')) / 'config' / CONFIG_NAME
        if installed.is_file():
            return installed
    except Exception:
        pass
    return Path(__file__).resolve().parents[1] / 'config' / CONFIG_NAME


def load_initial_joint_positions(path=None):
    """读取 YAML 中 7 个关节角（度），返回弧度 tuple。"""
    path = Path(path) if path is not None else default_config_path()
    with open(path, encoding='utf-8') as stream:
        data = yaml.safe_load(stream) or {}
    degrees = data.get('initial_joint_positions_deg') if isinstance(data, dict) else None
    if not isinstance(degrees, list) or len(degrees) != JOINT_COUNT:
        raise ValueError(f'{path}: initial_joint_positions_deg 必须是 {JOINT_COUNT} 个数值')
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) for value in degrees):
        raise ValueError(f'{path}: initial_joint_positions_deg 必须全部为有限数值')
    return tuple(math.radians(value) for value in degrees)


INITIAL_JOINT_POSITIONS = load_initial_joint_positions()
