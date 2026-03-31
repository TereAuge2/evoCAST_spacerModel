"""
Scale-aware rounding for randomized hyperparameter search (sklearn + W&B sweep entrypoints).

Use before passing floats into models and when recording trial tables so plots stay readable.
Torch MLP/CNN search uses ``train_torch.round_lr_wd`` for lr / weight_decay (5 decimals).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, uniform


def round_log_space_param(x: float) -> float:
    """Round positive-ish floats spanning orders of magnitude (C, gamma, lr-scale, reg, etc.)."""
    x = float(x)
    ax = abs(x)
    if not np.isfinite(ax) or ax == 0.0:
        return x
    if ax >= 1.0:
        return round(x, 5)
    if ax >= 0.01:
        return round(x, 7)
    if ax >= 1e-4:
        return round(x, 9)
    return round(x, 12)


class RoundedLogUniform:
    """``scipy.stats.loguniform`` with ``rvs`` rounded via :func:`round_log_space_param`."""

    __slots__ = ("_dist",)

    def __init__(self, low: float, high: float):
        self._dist = loguniform(low, high)

    def rvs(self, random_state=None) -> float:
        return round_log_space_param(float(self._dist.rvs(random_state=random_state)))


class RoundedUniform:
    """``scipy.stats.uniform(loc, scale)`` with fixed decimal rounding (e.g. nu, subsample)."""

    __slots__ = ("_dist", "_ndigits")

    def __init__(self, loc: float, scale: float, *, ndigits: int = 4):
        self._dist = uniform(loc, scale)
        self._ndigits = ndigits

    def rvs(self, random_state=None) -> float:
        return round(float(self._dist.rvs(random_state=random_state)), self._ndigits)


def round_dropout(x: float) -> float:
    """Dropout in [0, 1); typical grid {0.1, 0.2, 0.3} — two decimals."""
    return round(float(x), 2)
