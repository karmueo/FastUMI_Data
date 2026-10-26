"""归一化对零跨度、小跨度及训练范围外状态的确定性检查。"""

import numpy as np
import pytest

from dp_infer_lerobot.processors import IMAGE, STATE, normalize_numpy, unnormalize_numpy


def test_normalization_keeps_out_of_training_range_and_color_statistics():
    stats = {f'{IMAGE}.mean': np.array([.2, .3, .4], dtype=np.float32)[:, None, None],
             f'{IMAGE}.std': np.ones((3, 1, 1), dtype=np.float32),
             f'{STATE}.min': np.array([.5], dtype=np.float32),
             f'{STATE}.max': np.array([1], dtype=np.float32)}
    obs = {IMAGE: np.ones((1, 2, 3, 2, 2), dtype=np.float32), STATE: np.array([[[0.], [1.]]], dtype=np.float32)}
    result = normalize_numpy(obs, stats)
    np.testing.assert_allclose(result[STATE], [[[-3], [1]]])
    np.testing.assert_allclose(result[IMAGE][0, 0, :, 0, 0], [.8, .7, .6])


def test_only_exact_zero_span_uses_epsilon():
    minimum = np.zeros(10, dtype=np.float32)
    maximum = np.ones(10, dtype=np.float32)
    maximum[:2] = [0, 1e-10]
    result = unnormalize_numpy(np.ones((8, 10), dtype=np.float32), {'action.min': minimum, 'action.max': maximum})
    np.testing.assert_allclose(result[:, 0], 1e-8, rtol=1e-6, atol=0)
    np.testing.assert_allclose(result[:, 1], 1e-10, rtol=1e-6, atol=0)


def test_nonfinite_and_wrong_shape_statistics_rejected():
    stats = {'action.min': np.zeros(10), 'action.max': np.ones(9)}
    with pytest.raises(ValueError):
        unnormalize_numpy(np.zeros((8, 10)), stats)
    stats['action.max'] = np.full(10, np.nan)
    with pytest.raises(ValueError):
        unnormalize_numpy(np.zeros((8, 10)), stats)
