"""覆盖夹爪端点标定的复审质量门、文件安全与 ROS I/O 边界。"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import types

from fastumi_gripper_estimator import \
    gripper_calibration_cli as calibration_cli
from fastumi_gripper_estimator import range_calibration
from fastumi_gripper_estimator.range_calibration import \
    calculate_bag_statistics
from fastumi_gripper_estimator.range_calibration import \
    calculate_live_statistics
from fastumi_gripper_estimator.range_calibration import CalibrationQualityError
from fastumi_gripper_estimator.range_calibration import \
    validate_calibration_quality
from fastumi_gripper_estimator.range_calibration import write_calibrated_config
import pytest
import yaml


def _samples(center: float, count: int) -> list:
    """生成具有固定小幅扰动的确定性端点样本。"""
    return [center + (index % 3 - 1) * 0.1 for index in range(count)]


def _passing_statistics():
    """创建满足所有默认质量门的实时统计对象。"""
    return calculate_live_statistics(
        _samples(48.2, 24), 24, _samples(126.3, 24), 24
    )


@pytest.mark.parametrize(
    ('statistics', 'is_bag', 'condition'),
    [
        (
            replace(_passing_statistics(), closed=replace(
                _passing_statistics().closed, retained_count=19
            )),
            False, 'retained',
        ),
        (
            replace(_passing_statistics(), opened=replace(
                _passing_statistics().opened, robust_sigma_mm=2.1
            )),
            False, 'robust sigma',
        ),
        (
            replace(_passing_statistics(), opened=replace(
                _passing_statistics().opened, estimate_mm=60.0
            )),
            False, 'endpoint gap',
        ),
        (
            replace(_passing_statistics(), closed=replace(
                _passing_statistics().closed, raw_count=2
            )),
            True, 'raw bag cluster',
        ),
    ],
)
def test_quality_gates_name_each_failed_condition(
    statistics, is_bag: bool, condition: str
) -> None:
    """验证每一项端点质量门都返回可定位的错误说明。"""
    with pytest.raises(CalibrationQualityError, match=condition):
        validate_calibration_quality(statistics, is_bag=is_bag)


def test_bag_statistics_is_finite_deterministic_and_sorted() -> None:
    """验证非有限值被丢弃，聚类稳定且闭合中心始终较小。"""
    samples = [126.4, 48.1, float('nan'), 126.2, 48.3, float('inf')]

    first = calculate_bag_statistics(samples, len(samples))
    second = calculate_bag_statistics(list(reversed(samples)), len(samples))

    assert first.valid_count == 4
    assert first.closed.estimate_mm < first.opened.estimate_mm
    assert first.closed.estimate_mm == second.closed.estimate_mm
    assert first.opened.estimate_mm == second.opened.estimate_mm


def _write_config(path: Path) -> None:
    """写入最小完整 ROS 参数文件。"""
    document = {
        'node': {
            'ros__parameters': {
                'image_topic': '/image',
                'marker_size_mm': 16.0,
                'dictionary_name': 'DICT_4X4_50',
                'roi_ratios': [0.1, 0.2, 0.8, 0.9],
                'gripper_range': {
                    'left_finger_tag_id': 0,
                    'right_finger_tag_id': 1,
                    'min_marker_dist_mm': 40.0,
                    'max_marker_dist_mm': 130.0,
                },
            },
        },
    }
    path.write_text(yaml.safe_dump(document), encoding='utf-8')


def test_yaml_force_nested_parent_and_failed_write_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    """验证输出覆盖策略、嵌套父目录与原子写失败后的清理。"""
    source = tmp_path / 'input.yaml'
    target = tmp_path / 'nested' / 'range.yaml'
    _write_config(source)
    statistics = _passing_statistics()
    source_contents = source.read_text(encoding='utf-8')

    write_calibrated_config(str(source), str(target), statistics)
    assert target.is_file()
    assert source.read_text(encoding='utf-8') == source_contents
    with pytest.raises(ValueError, match='已存在'):
        write_calibrated_config(str(source), str(target), statistics)
    target.write_text('old', encoding='utf-8')
    write_calibrated_config(str(source), str(target), statistics, force=True)
    assert 'old' not in target.read_text(encoding='utf-8')

    failed_target = tmp_path / 'failed.yaml'
    monkeypatch.setattr(range_calibration.os, 'link',
                        lambda *_: (_ for _ in ()).throw(OSError('link')))
    with pytest.raises(ValueError, match='原子写入'):
        write_calibrated_config(str(source), str(failed_target), statistics)
    assert not failed_target.exists()
    assert not list(tmp_path.glob('.failed.yaml.*.tmp'))


class _FakeReader:
    """提供可控的顺序 bag 读取器。"""

    def __init__(self, topics, records) -> None:
        """保存话题类型和按时间排列的模拟记录。"""
        self._topics = topics
        self._records = iter(records)
        self._next = None

    def open(self, *_args) -> None:  # noqa: A003
        """模拟打开成功。"""

    def get_all_topics_and_types(self):
        """返回带 name/type 属性的话题描述。"""
        return [types.SimpleNamespace(name=name, type=type_name)
                for name, type_name in self._topics.items()]

    def has_next(self) -> bool:
        """缓存下一条记录以符合 rosbag2 顺序读取接口。"""
        if self._next is None:
            try:
                self._next = next(self._records)
            except StopIteration:
                return False
        return True

    def read_next(self):
        """返回缓存记录。"""
        record = self._next
        self._next = None
        return record


def _install_fake_bag_modules(monkeypatch, reader) -> None:
    """安装 bag 读取路径所需的最小 ROS 模块替身。"""
    bag_module = types.ModuleType('rosbag2_py')
    bag_module.SequentialReader = lambda: reader
    bag_module.StorageOptions = lambda **kwargs: kwargs
    bag_module.ConverterOptions = lambda *_args: None
    serialization = types.ModuleType('rclpy.serialization')
    serialization.deserialize_message = lambda data, _type: data
    image_module = types.ModuleType('sensor_msgs.msg')
    image_module.Image = object
    monkeypatch.setitem(sys.modules, 'rosbag2_py', bag_module)
    monkeypatch.setitem(sys.modules, 'rclpy.serialization', serialization)
    monkeypatch.setitem(sys.modules, 'sensor_msgs.msg', image_module)


def _write_metadata(path: Path, storage_identifier: str = 'mcap') -> None:
    """写入最小 bag metadata.yaml。"""
    (path / 'metadata.yaml').write_text(yaml.safe_dump({
        'rosbag2_bagfile_information': {
            'storage_identifier': storage_identifier,
        },
    }), encoding='utf-8')


def test_bag_validation_and_stride_selection(tmp_path: Path, monkeypatch) -> None:
    """验证存储、话题类型校验及 frame stride 的采样总数。"""
    _write_metadata(tmp_path)
    reader = _FakeReader(
        {'/image': 'sensor_msgs/msg/Image'},
        [('/image', 48.0, 0), ('/image', 48.1, 1),
         ('/image', 126.0, 2), ('/image', 126.1, 3)],
    )
    _install_fake_bag_modules(monkeypatch, reader)
    monkeypatch.setattr(calibration_cli, 'CvBridge', lambda: object())
    monkeypatch.setattr(calibration_cli, '_estimate_distance',
                        lambda _bridge, _estimator, message: message)

    statistics = calibration_cli.collect_bag_statistics(
        str(tmp_path), '/image', 2, estimator=None
    )

    assert statistics.total_count == 2
    assert statistics.valid_count == 2
    assert statistics.closed.estimate_mm == pytest.approx(48.0)
    assert statistics.opened.estimate_mm == pytest.approx(126.0)


@pytest.mark.parametrize(
    ('topics', 'message'),
    [({}, '不存在目标图像话题'), ({'/image': 'std_msgs/msg/String'}, '类型必须')],
)
def test_bag_rejects_missing_or_wrong_type_topic(
    tmp_path: Path, monkeypatch, topics, message: str
) -> None:
    """验证读取器打开后仍严格验证目标话题与 Image 类型。"""
    _write_metadata(tmp_path)
    _install_fake_bag_modules(monkeypatch, _FakeReader(topics, []))

    with pytest.raises(ValueError, match=message):
        calibration_cli.collect_bag_statistics(str(tmp_path), '/image', 1, None)


def test_bag_rejects_unsupported_storage_before_open(tmp_path: Path) -> None:
    """验证不支持的 storage_identifier 不会进入 ROS bag 读取器。"""
    _write_metadata(tmp_path, 'unsupported')

    with pytest.raises(ValueError, match='仅支持'):
        calibration_cli.collect_bag_statistics(str(tmp_path), '/image', 1, None)


def test_image_conversion_and_estimation_failures_are_invalid_samples(monkeypatch) -> None:
    """验证图像转换或估计失败均返回无效距离而不终止采样。"""
    bridge = types.SimpleNamespace(
        imgmsg_to_cv2=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError('conversion')
        )
    )
    assert calibration_cli._estimate_distance(bridge, object(), object()) is None
    bridge = types.SimpleNamespace(imgmsg_to_cv2=lambda *_args, **_kwargs: 1)
    estimator = types.SimpleNamespace(
        estimate=lambda _image: (_ for _ in ()).throw(RuntimeError('estimate'))
    )
    assert calibration_cli._estimate_distance(bridge, estimator, object()) is None


def test_main_preflight_does_not_collect_when_output_exists(
    tmp_path: Path, monkeypatch
) -> None:
    """验证已有输出在读取配置和采集前被拒绝。"""
    source = tmp_path / 'input.yaml'
    target = tmp_path / 'exists.yaml'
    source.write_text('{}', encoding='utf-8')
    target.write_text('{}', encoding='utf-8')
    arguments = argparse.Namespace(
        config=str(source), output=str(target), force=False, command='bag',
        image_topic=None, camera_calibration=None, bag_uri='never-open',
        frame_stride=1,
    )
    monkeypatch.setattr(calibration_cli, '_build_parser', lambda:
                        types.SimpleNamespace(parse_args=lambda _argv: arguments))
    monkeypatch.setattr(calibration_cli, 'collect_bag_statistics',
                        lambda *_args: pytest.fail('不应开始采样'))
    monkeypatch.setattr(calibration_cli, 'load_gripper_parameters',
                        lambda *_args: pytest.fail('不应读取配置'))

    assert calibration_cli.main([]) == 2


def test_main_cancellation_does_not_write_output(tmp_path: Path, monkeypatch) -> None:
    """验证实时采样收到 Ctrl+C 后返回取消状态且不产生 YAML。"""
    source = tmp_path / 'input.yaml'
    target = tmp_path / 'cancelled.yaml'
    source.write_text('{}', encoding='utf-8')
    arguments = argparse.Namespace(
        config=str(source), output=str(target), force=False, command='live',
        image_topic='/image', camera_calibration='/camera.yaml', countdown=0,
        sample_duration=1.0,
    )
    monkeypatch.setattr(calibration_cli, '_build_parser', lambda:
                        types.SimpleNamespace(parse_args=lambda _argv: arguments))
    monkeypatch.setattr(calibration_cli, 'load_gripper_parameters',
                        lambda *_args: ({}, {'image_topic': '/image'}))
    monkeypatch.setattr(calibration_cli, 'create_estimator', lambda *_args: object())
    monkeypatch.setattr(calibration_cli, 'collect_live_statistics',
                        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt()))

    assert calibration_cli.main([]) == 130
    assert not target.exists()


def test_live_guided_capture_order_and_counts(monkeypatch) -> None:
    """验证模拟实时采样严格按闭合、张开顺序并返回两个阶段计数。"""
    class FakeNode:
        """提供订阅回调容器。"""

        def __init__(self, *_args) -> None:
            """初始化回调槽。"""
            self.subscription_callback = None

        def create_subscription(self, _type, _topic, callback, _qos):
            """记录节点回调。"""
            self.subscription_callback = callback
            return object()

        def destroy_subscription(self, _subscription) -> None:
            """模拟窗口订阅销毁成功。"""
            self.subscription_callback = None

        def destroy_node(self) -> None:
            """模拟销毁成功。"""

    messages = iter([48.2, 126.3])
    fake_rclpy = types.ModuleType('rclpy')
    fake_rclpy.init = lambda: None
    fake_rclpy.shutdown = lambda: None
    fake_rclpy.spin_once = lambda node, **_kwargs: node.subscription_callback(next(messages))
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

    statistics = calibration_cli.collect_live_statistics(
        '/image', 0, 0.1, object(), input_function=prompts.append
    )

    assert '闭合' in prompts[0]
    assert '张开' in prompts[1]
    assert statistics.closed.raw_count == 1
    assert statistics.opened.raw_count == 1
    assert statistics.closed.estimate_mm == 48.2
    assert statistics.opened.estimate_mm == 126.3
