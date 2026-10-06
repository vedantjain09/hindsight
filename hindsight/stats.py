"""Paired comparison statistics for per-item pass/fail vectors on a shared held-out set."""
from __future__ import annotations

import numpy as np
from scipy.stats import binom


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Difference in pass rate (b - a) with a percentile CI, resampling items jointly."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.ndim == 1
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(d.mean()), "ci": (float(lo), float(hi)), "significant": bool(lo > 0 or hi < 0),
            "p_boot": float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))}


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact two-sided McNemar test on discordant pairs."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01 = int((~a & b).sum())   # b right, a wrong
    n10 = int((a & ~b).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * binom.cdf(min(n01, n10), n, 0.5))
    return {"b_only": n01, "a_only": n10, "p": float(p)}
