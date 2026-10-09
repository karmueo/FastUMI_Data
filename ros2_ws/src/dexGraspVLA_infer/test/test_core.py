"""同帧同步、物理状态归一化和端到端滚动时限的确定性测试。"""

from pathlib import Path

import numpy as np
import pytest

from dexgraspvla_infer.benchmark import evaluate
from dexgraspvla_infer.core import ObservationBuffer, denormalize_actions, normalization_bounds, normalize_state

ROOT = Path(__file__).resolve().parents[4]


def test_state_and_action_normalization_roundtrip():
    lower, upper = normalization_bounds(ROOT / "model/DexGraspVLA/assets/rm75/rm_75.urdf")
    state = np.r_[np.deg2rad([0, 20, 0, 70, 0, 90, 90]), 0.4]
    normalized = normalize_state(state, lower, upper)
    restored = denormalize_actions(np.tile(normalized, (64, 1)), lower, upper)
    np.testing.assert_allclose(restored, np.tile(state, (64, 1)), atol=1e-6)
    with pytest.raises(ValueError):
        denormalize_actions(np.full((64, 8), 1.1), lower, upper)


def populated_buffer():
    buffer = ObservationBuffer()
    stamp = 10_000_000_000
    buffer.image(stamp, "camera", np.zeros((20, 30, 3), np.uint8))
    buffer.mask(stamp, "camera", 7, np.full((20, 30), 255, np.uint8))
    buffer.joints.append((stamp + 10_000_000, np.zeros(7)))
    buffer.grippers.append((stamp - 20_000_000, 0.9))
    return buffer, stamp


def test_mask_matches_source_stamp_not_arrival_or_latest_rgb():
    buffer, stamp = populated_buffer()
    buffer.image(stamp + 30_000_000, "camera", np.ones((20, 30, 3), np.uint8))
    observation = buffer.latest(stamp + 100_000_000, tracking_id=7)
    assert observation.stamp_ns == stamp and not observation.bgr.any()
    assert buffer.latest(stamp + 100_000_000, tracking_id=8) is None
    assert buffer.latest(stamp + 2_000_000_000) is None


@pytest.mark.parametrize("failure", ["zero", "encoding", "shape", "frame", "joint_slop", "gripper_slop"])
def test_invalid_observation_cannot_infer(failure):
    buffer, stamp = populated_buffer()
    frame, identity, mask = buffer.masks[stamp]
    if failure == "zero":
        mask[:] = 0
    elif failure == "encoding":
        mask[:] = 1
    elif failure == "shape":
        mask = mask[:10]
    elif failure == "frame":
        frame = "another_camera"
    elif failure == "joint_slop":
        buffer.joints[0] = (stamp + 51_000_000, np.zeros(7))
    else:
        buffer.grippers[0] = (stamp - 51_000_000, 1.0)
    buffer.masks[stamp] = frame, identity, mask
    assert buffer.latest(stamp + 100_000_000) is None


def test_performance_checks_age_plus_interval_and_horizon_gaps():
    good = [{"age_s": 0.45, "interval_s": None if i == 0 else 0.5} for i in range(100)]
    assert evaluate(good)["passed"]
    slow = [{"age_s": 0.8, "interval_s": None if i == 0 else 0.6} for i in range(100)]
    assert not evaluate(slow)["passed"]
    good[50]["interval_s"] = 0.9
    assert evaluate(good)["horizon_gaps"] == 1
    assert not evaluate(good)["passed"]
    assert not evaluate(good[:99])["passed"]
