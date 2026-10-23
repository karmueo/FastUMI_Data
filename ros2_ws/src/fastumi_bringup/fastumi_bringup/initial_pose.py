"""RM75 初始关节目标，按 joint1 至 joint7 排列，单位为弧度。

数值只来自工作区 src/fastumi_bringup/config/initial_pose.yaml（单位为度）。
安装后的节点也定位该源文件，避免读取过期的 install/share 副本。
"""

import math
from pathlib import Path

import yaml


CONFIG_NAME = 'initial_pose.yaml'
JOINT_COUNT = 7


def default_config_path():
    """从源码或 install 中定位工作区唯一的起始姿态配置；缺失即报错。"""
    for parent in Path(__file__).resolve().parents:
        source = parent / 'src' / 'fastumi_bringup' / 'config' / CONFIG_NAME
        if source.is_file():
            return source
    raise FileNotFoundError('找不到工作区 src/fastumi_bringup/config/initial_pose.yaml')


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
