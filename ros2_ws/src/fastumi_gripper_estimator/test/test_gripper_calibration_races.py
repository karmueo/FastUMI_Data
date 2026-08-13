"""测试夹爪标定输出路径竞态、符号链接与实时窗口隔离。"""

import math
import os
from pathlib import Path
import sys
import types

from fastumi_gripper_estimator import gripper_calibration_cli as calibration_cli
from fastumi_gripper_estimator import range_calibration
from fastumi_gripper_estimator.range_calibration import (
    calculate_live_statistics,
    preflight_output_path,
    write_calibrated_config,
)
import pytest
import yaml


def _statistics():
    """创建可安全写入标定文件的固定端点统计。"""
    return calculate_live_statistics(
        [48.1] * 20, 20, [126.2] * 20, 20
    )


def _write_config(path: Path) -> None:
    """写入最小 ROS 参数 YAML。"""
    path.write_text(yaml.safe_dump({
        'node': {'ros__parameters': {
            'image_topic': '/image', 'marker_size_mm': 16.0,
            'dictionary_name': 'DICT_4X4_50',
            'roi_ratios': [0.1, 0.2, 0.8, 0.9],
            'gripper_range': {
                'left_finger_tag_id': 0, 'right_finger_tag_id': 1,
                'min_marker_dist_mm': 40.0, 'max_marker_dist_mm': 130.0,
            },
        }},
    }), encoding='utf-8')


@pytest.mark.parametrize('dangling', [False, True])
def test_output_symlink_is_rejected_in_preflight_and_commit(
    tmp_path: Path, dangling: bool
) -> None:
    """验证有效和悬挂最终符号链接均不会被当作输出文件。"""
    source = tmp_path / 'input.yaml'
    target = tmp_path / 'output.yaml'
    linked = tmp_path / 'linked.yaml'
    _write_config(source)
    if not dangling:
        linked.write_text('keep', encoding='utf-8')
    os.symlink(linked, target)

    with pytest.raises(ValueError, match='符号链接'):
        preflight_output_path(str(source), str(target), force=True)
    with pytest.raises(ValueError, match='符号链接'):
        write_calibrated_config(str(source), str(target), _statistics(), True)
    assert source.is_file()
    assert target.is_symlink()


def test_no_force_race_never_clobbers_new_target(tmp_path: Path, monkeypatch) -> None:
    """验证预检后才出现的文件会令 hard-link 提交失败且保留其内容。"""
    source = tmp_path / 'input.yaml'
    target = tmp_path / 'output.yaml'
    _write_config(source)
    original_link = range_calibration.os.link

    def create_racing_target(temporary_name, target_name):
        """在 no-clobber 链接操作前模拟其他进程创建输出。"""
        Path(target_name).write_text('raced', encoding='utf-8')
        return original_link(temporary_name, target_name)

    monkeypatch.setattr(range_calibration.os, 'link', create_racing_target)
    with pytest.raises(ValueError, match='已存在'):
        write_calibrated_config(str(source), str(target), _statistics())

    assert target.read_text(encoding='utf-8') == 'raced'
    assert source.is_file()
    assert not list(tmp_path.glob('.output.yaml.*.tmp'))


def test_committed_hard_link_ignores_temp_unlink_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """验证硬链接已提交后临时文件删除失败不改变成功结果。"""
    source = tmp_path / 'input.yaml'
    target = tmp_path / 'output.yaml'
    _write_config(source)
    original_unlink = range_calibration.os.unlink

    def fail_only_for_temporary(path):
        """保留输出硬链接，模拟临时目录短暂不可删除。"""
        if Path(path).name.startswith('.output.yaml.'):
            raise OSError('unlink')
        return original_unlink(path)

    monkeypatch.setattr(range_calibration.os, 'unlink', fail_only_for_temporary)
    result = write_calibrated_config(str(source), str(target), _statistics())

    assert result == target.resolve()
    assert target.is_file()
    assert yaml.safe_load(target.read_text(encoding='utf-8'))


def test_committed_hard_link_ignores_keyboard_interrupt_after_link(
    tmp_path: Path, monkeypatch
) -> None:
    """验证硬链接成功后紧邻 Ctrl+C 仍完成并报告输出成功。"""
    source = tmp_path / 'input.yaml'
    target = tmp_path / 'output.yaml'
    _write_config(source)
    original_link = range_calibration.os.link

    def link_then_interrupt(temporary_name, target_name):
        """在内核硬链接成功后模拟解释器收到中断。"""
        original_link(temporary_name, target_name)
        raise KeyboardInterrupt()

    monkeypatch.setattr(range_calibration.os, 'link', link_then_interrupt)
    result = write_calibrated_config(str(source), str(target), _statistics())

    assert result == target.resolve()
    assert target.is_file()
    assert yaml.safe_load(target.read_text(encoding='utf-8'))


def test_force_replaces_distinct_regular_output_without_touching_input(
    tmp_path: Path
) -> None:
    """验证 --force 仅替换独立常规输出，输入内容保持不变。"""
    source = tmp_path / 'input.yaml'
    target = tmp_path / 'output.yaml'
    _write_config(source)
    source_contents = source.read_text(encoding='utf-8')
    target.write_text('old', encoding='utf-8')

    write_calibrated_config(str(source), str(target), _statistics(), force=True)

    assert source.read_text(encoding='utf-8') == source_contents
    assert 'old' not in target.read_text(encoding='utf-8')


@pytest.mark.parametrize('duration', [math.nan, math.inf, 0.0, -1.0])
def test_live_duration_must_be_positive_and_finite(duration: float) -> None:
    """验证实时采样拒绝 NaN、无穷、零和负持续时间。"""
    with pytest.raises(ValueError, match='有限数'):
        calibration_cli.collect_live_statistics('/image', 0, duration, object())


def test_live_subscriptions_are_stage_local(monkeypatch) -> None:
    """验证提示和阶段间不存在订阅，两个窗口各自只接收当前消息。"""
    class FakeNode:
        """保存活动订阅并模拟 ROS 节点。"""

        instance = None

        def __init__(self, *_args) -> None:
            """初始化空订阅列表。"""
            self.subscriptions = []
            FakeNode.instance = self

        def create_subscription(self, _type, _topic, callback, _qos):
            """创建可销毁的当前窗口订阅。"""
            subscription = types.SimpleNamespace(callback=callback)
            self.subscriptions.append(subscription)
            return subscription

        def destroy_subscription(self, subscription) -> None:
            """销毁窗口订阅，丢弃任何后续排队消息。"""
            self.subscriptions.remove(subscription)

        def destroy_node(self) -> None:
            """模拟节点销毁。"""

    stage_messages = iter([48.2, 126.3])
    fake_rclpy = types.ModuleType('rclpy')
    fake_rclpy.init = lambda: None
    fake_rclpy.shutdown = lambda: None
    fake_rclpy.ok = lambda: True
    fake_rclpy.spin_once = lambda node, **_kwargs: (
        node.subscriptions[-1].callback(next(stage_messages))
    )
    node_module = types.ModuleType('rclpy.node')
    node_module.Node = FakeNode
    qos_module = types.ModuleType('rclpy.qos')
    qos_module.DurabilityPolicy = types.SimpleNamespace(VOLATILE=1)
    qos_module.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=1)
    qos_module.QoSProfile = lambda **_kwargs: object()
    image_module = types.ModuleType('sensor_msgs.msg')
    image_module.Image = object
    monkeypatch.setitem(sys.modules, 'rclpy', fake_rclpy)
    monkeypatch.setitem(sys.modules, 'rclpy.node', node_module)
    monkeypatch.setitem(sys.modules, 'rclpy.qos', qos_module)
    monkeypatch.setitem(sys.modules, 'sensor_msgs.msg', image_module)
    monkeypatch.setattr(calibration_cli, 'CvBridge', lambda: object())
    monkeypatch.setattr(calibration_cli, '_estimate_distance',
                        lambda _bridge, _estimator, message: message)
    monotonic = iter([0.0, 0.0, 1.0, 2.0, 2.0, 3.0])
    monkeypatch.setattr(calibration_cli.time, 'monotonic', lambda: next(monotonic))
    prompts = []

    def prompt(text: str) -> str:
        """确认任何预窗口或阶段间消息均没有活动订阅可投递。"""
        assert FakeNode.instance is not None
        assert not FakeNode.instance.subscriptions
        prompts.append(text)
        return ''

    statistics = calibration_cli.collect_live_statistics(
        '/image', 0, 0.1, object(), input_function=prompt
    )

    assert len(prompts) == 2
    assert statistics.closed.raw_count == 1
    assert statistics.opened.raw_count == 1
    assert statistics.closed.estimate_mm == 48.2
    assert statistics.opened.estimate_mm == 126.3
    assert not FakeNode.instance.subscriptions
