"""测试 FastUMI HDF5 主结构、属性和质量数据。"""

import numpy as np
import pytest

from fastumi_data.models import SynchronizedEpisode


try:
    # 采集机通过 package.xml 安装 h5py；精简开发环境可跳过该单项。
    import h5py
    from fastumi_data.hdf5_writer import write_episode_hdf5
except ModuleNotFoundError:
    h5py = None
    write_episode_hdf5 = None


@pytest.mark.skipif(h5py is None, reason="当前系统 Python 未安装 h5py")
def test_writes_versioned_fastumi_hdf5(tmp_path) -> None:
    """验证输出包含训练兼容主结构和时间质量字段。"""
    episode = SynchronizedEpisode(
        timestamp_ns=np.arange(10, dtype=np.int64),
        images_rgb=np.zeros((10, 8, 8, 3), dtype=np.uint8),
        qpos=np.tile(
            np.asarray([0, 0, 0, 0, 0, 0, 1, 0.5], dtype=np.float32),
            (10, 1),
        ),
        gripper_observed=np.ones(10, dtype=np.bool_),
        tracker_tracking_ok=np.ones(10, dtype=np.bool_),
        pose_gap_ms=np.zeros(10, dtype=np.float32),
        gripper_gap_ms=np.zeros(10, dtype=np.float32),
    )
    output = tmp_path / "episode_0000.hdf5"

    write_episode_hdf5(
        str(output), episode, "task", "session", 0, 20.0, "hash"
    )

    with h5py.File(output, "r") as root:
        assert root.attrs["schema_version"] == "fastumi_ros2_v1"
        assert root.attrs["pose_frame"] == "episode_start_tcp"
        assert root["observations/qpos"].shape == (10, 8)
        assert root["action"].shape == (10, 8)
        assert root["observations/images/front"].shape == (10, 8, 8, 3)
        assert root["observations/timestamp_ns"].dtype == np.int64
        assert np.all(
            root["observations/quality/tracker_tracking_ok"][:]
        )
