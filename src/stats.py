"""Statistical tests for comparing old vs new distance distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class BootstrapResult:
    mean: float
    ci_low: float
    ci_high: float
    n: int


def bootstrap_mean_ci(
    sample: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapResult:
    """Percentile bootstrap CI for the sample mean."""
    rng = np.random.default_rng(seed)
    n = sample.size
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = sample[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        mean=float(sample.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        n=int(n),
    )


@dataclass
class ComparisonResult:
    u_statistic: float
    p_value: float
    cliffs_delta: float
    direction: str  # "new > old", "old > new", or "equal"


def compare_distributions(old: np.ndarray, new: np.ndarray) -> ComparisonResult:
    """One-sided Mann-Whitney U (H1: new > old) plus Cliff's delta effect size."""
    u, p = stats.mannwhitneyu(new, old, alternative="greater")
    delta = _cliffs_delta(new, old)
    if delta > 0.05:
        direction = "new > old"
    elif delta < -0.05:
        direction = "old > new"
    else:
        direction = "equal"
    return ComparisonResult(
        u_statistic=float(u),
        p_value=float(p),
        cliffs_delta=float(delta),
        direction=direction,
    )


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a > b) - P(a < b). Range [-1, 1].

    Standard thresholds (Romano et al., 2006): negligible < 0.147, small < 0.33,
    medium < 0.474, otherwise large.
    """
    # Vectorized; O(|a| * |b|) memory — fine for n ≈ 78 here.
    diff = a[:, None] - b[None, :]
    gt = (diff > 0).sum()
    lt = (diff < 0).sum()
    return (gt - lt) / (a.size * b.size)
