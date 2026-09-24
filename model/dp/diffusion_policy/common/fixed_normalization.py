"""Build reversible fixed-range normalizers from URDF and gripper bounds."""

import numpy as np
import torch

from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer


def fixed_range_normalizer(lower, upper):
    """Map physical values in [lower, upper] to [-1, 1] without fitting data."""
    lower = np.atleast_1d(np.asarray(lower, dtype=np.float32))
    upper = np.atleast_1d(np.asarray(upper, dtype=np.float32))
    if lower.shape != upper.shape or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Invalid normalization bounds")
    if np.any(lower >= upper):
        raise ValueError("Normalization lower bounds must be less than upper bounds")
    lo = torch.from_numpy(lower.copy())
    hi = torch.from_numpy(upper.copy())
    scale = 2 / (hi - lo)
    offset = -1 - scale * lo
    stats = {"min": lo, "max": hi, "mean": (lo + hi) / 2,
             "std": (hi - lo) / (12 ** 0.5)}
    normalizer = SingleFieldLinearNormalizer.create_manual(scale, offset, stats)
    for parameter in normalizer.parameters():
        parameter.requires_grad_(False)
    return normalizer
