"""验证夹爪估计节点内置参数与默认 YAML 配置保持一致。"""

import ast
from pathlib import Path

import yaml


def test_node_distance_defaults_match_yaml() -> None:
    """验证直接启动节点与通过 launch 加载 YAML 使用相同标定端点。"""
    # 节点源码用于读取无需 ROS 运行时依赖的常量字面值。
    package_path = Path(__file__).parents[1]
    node_path = (
        package_path
        / "fastumi_gripper_estimator/gripper_openness_node.py"
    )
    syntax_tree = ast.parse(node_path.read_text(encoding="utf-8"))
    # 收集节点模块顶层的简单常量赋值。
    node_defaults = {
        statement.targets[0].id: ast.literal_eval(statement.value)
        for statement in syntax_tree.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
    }

    # 默认 YAML 位于当前测试目录的同级 config 目录。
    config_path = package_path / "config/gripper_openness.yaml"
    # 提取节点使用的夹爪距离参数子项。
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gripper_range = document["gripper_openness_estimator"]["ros__parameters"][
        "gripper_range"
    ]

    assert node_defaults["DEFAULT_MIN_MARKER_DIST_MM"] == gripper_range[
        "min_marker_dist_mm"
    ]
    assert node_defaults["DEFAULT_MAX_MARKER_DIST_MM"] == gripper_range[
        "max_marker_dist_mm"
    ]
