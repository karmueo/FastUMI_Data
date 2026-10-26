"""无 ROS/GPU 的几何、时间、颜色及夹爪锚点回归。"""

from io import BytesIO

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

from dp_infer.common import InferenceContext, Observation
from dp_infer_lerobot.core import (build_observations, compressed_image_to_rgb, correct_sequence,
                                   decode_actions, letterbox_rgb, result_is_usable)
from fastumi_rm75.placo_control import GripperScale, gripper_trajectory_anchor


def actions():
    values = np.tile(np.r_[np.zeros(3), [1, 0, 0, 0, 1, 0], 0.5], (8, 1))
    values[:, 9] = np.arange(8) / 10
    return values


def context(pose=None):
    return InferenceContext(np.eye(4) if pose is None else pose, 1_000_000_000, 2, 3)


def test_cumulative_noncommuting_local_steps_and_gripper_times():
    values = actions()
    turn = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    values[0, 3:9] = turn[:2].ravel()
    values[1, 0] = 0.1
    values[2, 3:9] = Rotation.from_euler("x", 35, degrees=True).as_matrix()[:2].ravel()
    sequence = decode_actions(values, context())
    np.testing.assert_allclose(sequence.positions[2], [0, 0.1, 0], atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(sequence.quaternions[3]).as_matrix(),
                               turn @ Rotation.from_euler("x", 35, degrees=True).as_matrix(), atol=1e-12)
    np.testing.assert_array_equal(sequence.gripper_openness, np.r_[np.arange(8) / 10, .7])
    np.testing.assert_allclose(sequence.time_from_start, np.arange(9) / 30)


def test_correction_anchor_roundtrip_and_exact_expiry_boundary():
    correction = np.eye(4)
    correction[:3, :3] = Rotation.from_euler("xyz", [.1, -.2, .3]).as_matrix()
    correction[:3, 3] = [.01, -.02, .03]
    real = np.eye(4)
    real[:3, 3] = [.2, .3, .4]
    ctx = context(real @ np.linalg.inv(correction))
    sequence = correct_sequence(decode_actions(actions(), ctx), correction)
    np.testing.assert_allclose(sequence.positions[0], real[:3, 3], atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(sequence.quaternions[0]).as_matrix(), np.eye(3), atol=1e-12)
    end = ctx.stamp_ns + round(8e9 / 30)
    assert result_is_usable(sequence, ctx, end - 1)
    assert not result_is_usable(sequence, ctx, end)
    assert not result_is_usable(sequence, ctx, ctx.stamp_ns - 1)


@pytest.mark.parametrize("fault", ["shape", "nan", "zero", "parallel", "gripper"])
def test_invalid_actions_rejected(fault):
    values = actions()
    if fault == "shape": values = values[:7]
    if fault == "nan": values[0, 0] = np.nan
    if fault == "zero": values[0, 3:6] = 0
    if fault == "parallel": values[0, 6:9] = values[0, 3:6]
    if fault == "gripper": values[0, 9] = 1.1
    with pytest.raises(ValueError):
        decode_actions(values, context())


def test_rgb_box_letterbox_and_two_frame_input():
    rgb = np.zeros((20, 40, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    stream = BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    decoded = compressed_image_to_rgb(stream.getvalue())
    np.testing.assert_array_equal(decoded, rgb)
    result = letterbox_rgb(decoded)
    assert result.shape == (3, 224, 224) and result.dtype == np.float32
    assert not result[:, :56].any() and not result[:, 168:].any()
    assert (result[0, 56:168] == 1).all() and not result[1:].any()
    history = [Observation(10, result, np.eye(4), .6), Observation(20, result, np.eye(4), .8)]
    obs = build_observations(history)
    assert obs['observation.images.cam0'].shape == (1, 2, 3, 224, 224)
    np.testing.assert_allclose(obs['observation.state'], [[[.6], [.8]]])


def test_first_gripper_anchor_uses_measured_training_scale():
    scale = GripperScale(.6, .5)
    assert gripper_trajectory_anchor(None, .5, scale) == pytest.approx(.6)
    assert gripper_trajectory_anchor(.8, .5, scale) == .8
    with pytest.raises(ValueError, match="Fresh"):
        gripper_trajectory_anchor(None, None, scale)


def test_controller_discards_past_points_after_full_integration():
    from types import SimpleNamespace
    from fastumi_rm75.placo_control import decode_trajectory

    def stamp(ns):
        sec, nano = divmod(ns, 1_000_000_000)
        return SimpleNamespace(sec=sec, nanosec=nano)

    values = actions()
    values[:, 0] = .01
    sequence = decode_actions(values, context())
    poses = [SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=z),
                              orientation=SimpleNamespace(x=qx, y=qy, z=qz, w=qw))
             for (x, y, z), (qx, qy, qz, qw) in zip(sequence.positions, sequence.quaternions)]
    message = SimpleNamespace(header=SimpleNamespace(stamp=stamp(context().stamp_ns), frame_id='base_link'),
                              end_frame='Link7', episode_id=2, sequence_id=3, poses=poses,
                              gripper_openness=sequence.gripper_openness,
                              time_from_start=[stamp(round(t * 1e9)) for t in sequence.time_from_start])
    trajectory = decode_trajectory(message, context().stamp_ns + 100_000_000, np.eye(4), anchor_gripper=.9)
    # 前3个增量已过期，但必须保留在第4个姿态的累计值中。
    assert trajectory.positions[0, 0] == pytest.approx(.04)
    assert trajectory.grippers[0] == pytest.approx(.4)
    with pytest.raises(ValueError, match='expired'):
        decode_trajectory(message, context().stamp_ns + round(8e9 / 30), np.eye(4))
