"""测试夹爪开合端点的纯统计、质量阈值和 YAML 输出安全性。"""

from pathlib import Path

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


def make_distances(center: float, count: int) -> list:
    """生成带确定性微小扰动的一组距离样本。"""
    return [center + ((index % 5) - 2) * 0.05 for index in range(count)]


def write_config(path: Path) -> None:
    """写入含无关值的完整 ROS 参数 YAML 用于副本校验。"""
    document = {
        'gripper_openness_estimator': {
            'ros__parameters': {
                'image_topic': '/image',
                'marker_size_mm': 16.0,
                'dictionary_name': 'DICT_4X4_50',
                'roi_ratios': [0.1, 0.2, 0.8, 0.9],
                'unrelated_value': {'keep': True},
                'gripper_range': {
                    'left_finger_tag_id': 0,
                    'right_finger_tag_id': 1,
                    'min_marker_dist_mm': 45.0,
                    'max_marker_dist_mm': 130.0,
                },
            },
        },
    }
    path.write_text(yaml.safe_dump(document), encoding='utf-8')


def test_bag_statistics_rejects_transition_outliers() -> None:
    """验证二均值加 MAD 会剔除端点外的明显过渡异常值。"""
    distances = make_distances(48.2, 40) + make_distances(126.3, 45)
    distances += [49.0, 125.3, 80.0, 95.0, float('nan'), float('inf')]

    statistics = calculate_bag_statistics(distances, total_count=len(distances))

    assert statistics.total_count == 91
    assert statistics.valid_count == 89
    assert statistics.closed.estimate_mm == pytest.approx(48.2)
    assert statistics.opened.estimate_mm == pytest.approx(126.3)
    validate_calibration_quality(statistics, is_bag=True)


def test_live_statistics_uses_operator_endpoint_order() -> None:
    """验证实时采样直接将闭合、张开阶段分别进行稳健剔除。"""
    statistics = calculate_live_statistics(
        make_distances(48.2, 24) + [80.0], 25,
        make_distances(126.3, 24) + [90.0], 25,
    )

    assert statistics.closed.estimate_mm == pytest.approx(48.2)
    assert statistics.opened.estimate_mm == pytest.approx(126.3)
    validate_calibration_quality(statistics, is_bag=False)


def test_quality_error_names_failed_condition() -> None:
    """验证低有效率错误包含明确的质量条件名称。"""
    statistics = calculate_live_statistics(
        make_distances(48.2, 20), 50,
        make_distances(126.3, 20), 50,
    )

    with pytest.raises(CalibrationQualityError, match='valid/total'):
        validate_calibration_quality(statistics, is_bag=False)


def test_output_only_changes_range_and_never_overwrites_input(
    tmp_path: Path,
) -> None:
    """验证完整配置副本只变更两个范围值，--force 也不能覆盖输入。"""
    input_path = tmp_path / 'input.yaml'
    output_path = tmp_path / 'output.yaml'
    write_config(input_path)
    statistics = calculate_live_statistics(
        make_distances(48.2345, 20), 20,
        make_distances(126.3456, 20), 20,
    )

    result = write_calibrated_config(
        str(input_path), str(output_path), statistics
    )

    assert result == output_path.resolve()
    original = yaml.safe_load(input_path.read_text(encoding='utf-8'))
    output = yaml.safe_load(output_path.read_text(encoding='utf-8'))
    original_range = original['gripper_openness_estimator']['ros__parameters'][
        'gripper_range'
    ]
    output_range = output['gripper_openness_estimator']['ros__parameters'][
        'gripper_range'
    ]
    assert output_range['min_marker_dist_mm'] == 48.234
    assert output_range['max_marker_dist_mm'] == 126.346
    original_range['min_marker_dist_mm'] = output_range['min_marker_dist_mm']
    original_range['max_marker_dist_mm'] = output_range['max_marker_dist_mm']
    assert output == original
    with pytest.raises(ValueError, match='不能与输入'):
        write_calibrated_config(
            str(input_path), str(input_path), statistics, force=True
        )
