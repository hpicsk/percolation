"""Autocorrelation-robust inference for overlapping-window percolation series.

The headline gradient (paper subsec:main_result) is tested on per-window paired
differences ``Delta_t = rho_c(Q_hi, t) - rho_c(Q_lo, t)``. The 60-day estimation
windows advance ``STEP = 20`` trading days, so windows up to two steps apart
share returns (``WINDOW - 2*STEP = 20`` days overlap) and the ``Delta_t`` series
is autocorrelated. A binomial sign test or Wilcoxon signed-rank test treats the
windows as independent and therefore overstates significance.

This module supplies the two autocorrelation-robust statistics the manuscript
reads the headline significance from: a Newey-West (1987) HAC ``t``-statistic on
the mean of ``Delta_t`` and a moving-block bootstrap one-sided ``p``-value. Both
the Korea pipeline (``scripts/headline/verify_claim.py``) and the Taiwan re-run
(``scripts/headline/run_headline.py``) import them so the inference is computed
once.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Moving-block bootstrap resamples and locked seed.
BLOCK_BOOT_DRAWS = 10000
BLOCK_BOOT_SEED = 20260517


def newey_west_lag(n: int) -> int:
    """Newey-West (1994) automatic Bartlett bandwidth ``L = floor(4 (n/100)^(2/9))``.

    At the headline ``n`` this exceeds the two-step (3-window) overlap span, so
    the HAC variance absorbs the overlap-induced autocorrelation of the
    paired-difference series (paper subsec:main_result).
    """
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def overlap_span(window: int, step: int) -> int:
    """Largest lag, in windows, at which two windows of *window* days whose
    starts lie *step* days apart share returns."""
    return int(np.ceil(window / step)) - 1


def hac_mean_se(x: np.ndarray, *, min_lag: int = 0) -> dict:
    """Newey-West HAC standard error of the mean of a serially correlated series.

    Shared by the paired-difference inference of ``hac_block_stats`` and by the
    per-quintile level error bars of ``fig:headline_gradient``, so both read the
    same Bartlett-kernel long-run variance rather than two copies of it.

    The bandwidth is the automatic one, raised to *min_lag* where a caller's
    windows share returns over more lags than the automatic bandwidth covers.
    Returns the sample mean, its HAC standard error, the bandwidth,
    and the effective sample size
    ``n_eff = n * gamma_0 / sigma^2_LR`` (the iid-equivalent number of windows
    once the overlap autocorrelation is discounted).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    nan = float("nan")
    if n < 3:
        return {"mean": nan, "se": nan, "nw_lag": 0, "n_eff": nan, "n": n}
    mean = float(x.mean())
    e = x - mean
    gamma0 = float(np.mean(e * e))
    lag = max(1, newey_west_lag(n), min_lag)
    # Long-run variance via the Bartlett-kernel HAC estimator.
    lrv = gamma0
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        gamma_k = float(np.mean(e[k:] * e[:-k]))
        lrv += 2.0 * weight * gamma_k
    lrv = max(lrv, 1e-24)
    return {"mean": mean, "se": float(np.sqrt(lrv / n)), "nw_lag": int(lag),
            "n_eff": float(n * gamma0 / lrv), "n": n}


def block_length(n: int) -> int:
    """Moving-block bootstrap block length for a series of ``n`` windows.

    About n^(1/3), and never below 3, the overlap span of consecutive windows,
    so each block carries the local dependence intact (subsec:inference)."""
    return max(3, int(round(n ** (1.0 / 3.0))))


def hac_block_stats(diff: np.ndarray, *, min_lag: int = 0,
                    min_block: int = 0) -> dict:
    """Autocorrelation-robust inference on a per-window paired-difference series.

    Returns the Newey-West HAC ``t``-statistic and one-sided ``p``-value for
    ``H0: E[Delta] <= 0`` (Bartlett kernel, automatic bandwidth raised to
    *min_lag*), a moving-block bootstrap one-sided ``p``-value (blocks of
    :func:`block_length` windows raised to *min_block*), and the effective sample
    size ``n_eff = n * gamma_0 / sigma^2_LR`` (the iid-equivalent number of
    windows once the overlap autocorrelation is discounted).
    """
    x = np.asarray(diff, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    nan = float("nan")
    if n < 3:
        return {"mean_diff": nan, "nw_lag": 0, "nw_t": nan,
                "nw_p_one_sided": nan, "block_boot_p_one_sided": nan,
                "n_eff": nan, "block_len": 0}
    hac = hac_mean_se(x, min_lag=min_lag)
    mean, se, lag = hac["mean"], hac["se"], hac["nw_lag"]
    t = mean / se
    p = float(norm.sf(t))                 # one-sided H0: E[Delta] <= 0
    n_eff = hac["n_eff"]
    block_len = max(block_length(n), min_block)
    n_blocks = int(np.ceil(n / block_len))
    max_start = n - block_len
    rng = np.random.default_rng(BLOCK_BOOT_SEED)
    offsets = np.arange(block_len)
    boot_means = np.empty(BLOCK_BOOT_DRAWS)
    for b in range(BLOCK_BOOT_DRAWS):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:n]
        boot_means[b] = x[idx].mean()
    block_p = float((np.sum(boot_means <= 0.0) + 1) / (BLOCK_BOOT_DRAWS + 1))
    return {"mean_diff": mean, "nw_lag": int(lag), "nw_t": float(t),
            "nw_p_one_sided": p, "block_boot_p_one_sided": block_p,
            "n_eff": n_eff, "block_len": int(block_len)}
