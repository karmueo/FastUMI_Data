"""验证实机回位脚本只接受起点附近的有效七轴反馈。"""

import numpy as np
import pytest

from return_to_start import HOME, validate_homing_start


def _safe_forward(joints):
    transform = np.eye(4)
    delta = joints[5] - HOME[5]
    transform[:3, 3] = [0.3, delta * 0.2, 0.337 + delta * 0.05]
    return transform


def test_homing_preflight_covers_latest_trial_but_rejects_larger_departures():
    joints = HOME.copy()
    joints[5] -= np.deg2rad(34.0)
    validate_homing_start(joints, _safe_forward)
    joints[5] -= np.deg2rad(12.0)
    with pytest.raises(RuntimeError, match="limit 45"):
        validate_homing_start(joints, _safe_forward)
    joints[5] = np.nan
    with pytest.raises(RuntimeError, match="invalid"):
        validate_homing_start(joints, _safe_forward)


def test_homing_preflight_rejects_path_outside_trial_region():
    joints = HOME.copy()
    joints[5] -= np.deg2rad(34.0)

    def far_forward(angles):
        transform = _safe_forward(angles)
        transform[1, 3] *= 2
        return transform

    with pytest.raises(RuntimeError, match="return path"):
        validate_homing_start(joints, far_forward)
