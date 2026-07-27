"""测试 FastUMI 共享 SE(3) 位姿转换和 SLERP。"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fastumi_data.pose_math import (
    interpolate_pose,
    matrix_to_pose,
    pose_to_matrix,
    transform_series_to_episode_frame,
)


def test_pose_matrix_round_trip() -> None:
    """验证位置与四元数经过矩阵往返后保持同一位姿。"""
    position = np.asarray([0.2, -0.1, 0.5])
    quaternion = Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_quat()
    transform = pose_to_matrix(position, quaternion)

    recovered_position, recovered_quaternion = matrix_to_pose(transform)

    assert recovered_position == pytest.approx(position)
    assert abs(float(np.dot(recovered_quaternion, quaternion))) == pytest.approx(
        1.0
    )


def test_slerp_handles_opposite_quaternion_signs() -> None:
    """验证等价异号四元数不会导致插值绕远路。"""
    quaternion = Rotation.from_euler("z", 0.5).as_quat()

    _, interpolated = interpolate_pose(
        np.zeros(3), quaternion, np.ones(3), -quaternion, 0.5
    )

    assert abs(float(np.dot(interpolated, quaternion))) == pytest.approx(1.0)


def test_episode_frame_starts_at_identity() -> None:
    """验证 Tracker 外参应用后首帧 TCP 相对位姿为单位变换。"""
    positions = np.asarray([[1.0, 2.0, 3.0], [1.1, 2.0, 3.0]])
    quaternions = np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2)
    tracker_to_tcp = pose_to_matrix(
        np.asarray([0.2, 0.0, 0.0]), np.asarray([0.0, 0.0, 0.0, 1.0])
    )

    relative_position, relative_quaternion = (
        transform_series_to_episode_frame(
            positions, quaternions, tracker_to_tcp
        )
    )

    assert relative_position[0] == pytest.approx(np.zeros(3))
    assert relative_quaternion[0] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert relative_position[1] == pytest.approx([0.1, 0.0, 0.0])
