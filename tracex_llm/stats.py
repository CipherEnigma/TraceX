"""Statistics for paired evaluation (Member 3). numpy only, no scipy.

Every method is scored on the same test instances, so comparisons are paired (brief):
- `bootstrap_ci`        : percentile CI of a mean (instances resampled with replacement).
- `paired_bootstrap`    : CI and two-sided p-value of mean(a) - mean(b) over the same instances.
- `mcnemar`             : McNemar test on paired binary outcomes (exact binomial when the number
                          of discordant pairs is < 25, else chi-square with continuity correction).
- `holm`                : Holm-Bonferroni adjustment for a family of p-values.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C

N_BOOT = 10_000
ALPHA = 0.05


def _idx(n: int, n_boot: int, seed: int):
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(x, n_boot: int = N_BOOT, seed: int = C.SEED, alpha: float = ALPHA) -> dict:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    means = x[_idx(len(x), n_boot, seed)].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(x))}


def paired_bootstrap(a, b, n_boot: int = N_BOOT, seed: int = C.SEED, alpha: float = ALPHA) -> dict:
    """Difference mean(a) - mean(b) for per-instance scores of two methods on the same instances."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired_bootstrap needs scores for the same instances")
    if len(a) == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "n": 0}
    d = a - b
    boots = d[_idx(len(d), n_boot, seed)].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    # two-sided p: how often the resampled difference falls on the other side of zero
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean()) if d.any() else 1.0
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "p": float(min(1.0, p)), "n": int(len(d))}


def mcnemar(a_correct, b_correct) -> dict:
    """b = a right & b wrong, c = a wrong & b right."""
    a, b = np.asarray(a_correct, dtype=bool), np.asarray(b_correct, dtype=bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0, "test": "none"}
    if n < 25:
        k = min(n10, n01)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
        test = "exact"
    else:
        stat = (abs(n10 - n01) - 1) ** 2 / n
        p = math.erfc(math.sqrt(stat / 2))  # chi-square survival function, 1 df
        test = "chi2-cc"
    return {"a_only": n10, "b_only": n01, "p": float(p), "test": test}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input). NaNs are left as NaN."""
    idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvals[i])
    m, adj, running = len(order), [float("nan")] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj
