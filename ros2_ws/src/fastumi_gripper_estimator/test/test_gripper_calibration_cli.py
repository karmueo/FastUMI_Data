"""测试夹爪标定命令行参数、bag 元数据校验与实时采样参数保护。"""

from pathlib import Path

from fastumi_gripper_estimator import \
    gripper_calibration_cli as calibration_cli
import pytest
import yaml


def write_metadata(path: Path, storage_identifier: str) -> None:
    """写入最小 ROS 2 bag 元数据文件。"""
    document = {
        'rosbag2_bagfile_information': {
            'storage_identifier': storage_identifier,
        },
    }
    path.write_text(yaml.safe_dump(document), encoding='utf-8')


def test_read_storage_identifier_accepts_metadata(tmp_path: Path) -> None:
    """验证 CLI 从 metadata.yaml 获取 mcap 存储插件名称。"""
    write_metadata(tmp_path / 'metadata.yaml', 'mcap')

    assert calibration_cli._read_storage_identifier(str(tmp_path)) == 'mcap'


def test_read_storage_identifier_rejects_missing_metadata(tmp_path: Path) -> None:
    """验证不存在 metadata.yaml 时不会尝试打开 bag。"""
    with pytest.raises(ValueError, match='metadata.yaml'):
        calibration_cli._read_storage_identifier(str(tmp_path))


@pytest.mark.parametrize(
    ('countdown', 'sample_duration', 'message'),
    [(-1, 5.0, 'countdown'), (0, 0.0, 'sample-duration')],
)
def test_live_options_are_validated_before_ros_initialization(
    countdown: int, sample_duration: float, message: str
) -> None:
    """验证非法实时选项在创建 ROS 节点前被明确拒绝。"""
    with pytest.raises(ValueError, match=message):
        calibration_cli.collect_live_statistics(
            '/image', countdown, sample_duration, estimator=None
        )
