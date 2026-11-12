"""Rolling-window x size-quintile iteration for the percolation pipeline.

Every script that scans the 60-day rolling windows and samples size quintiles
shares the same bookkeeping: build the window grid, assign quintiles at each
window's end date, draw up to ``PER_QUINTILE`` tickers with the seed
``seed_base + g + start``, drop columns that are too sparse, and build the
correlation matrix. :func:`iter_window_cells` is that loop, exposed once so the
consumer scripts no longer copy it (and no longer import private helpers from
``pipeline``). A script runs the loop once per seed_base of the committed set
``DRAW_SEEDS`` through :func:`over_draw_seeds`, which returns each cell's mean
over the draws (subsec:methods).

The generator yields a :class:`WindowCell` per (window, quintile): the cleaned
return sub-frame plus a ready correlation matrix. Consumers that need an extra
aligned panel slice it with ``cell.start``;
consumers that transform the correlation matrix (RMT filtering) or the returns
(shuffling) work from ``cell.corr_unit_diag()`` / ``cell.sub``; consumers with a
custom universe (within-sector) pass a custom ``assigner``.

Callers that sub-group a single window themselves (the bootstrap 5-year draws)
use :func:`percolation_one_window` instead -- the
single-window counterpart that cleans a raw ticker list and returns the same
metrics dict.

The per-cell conventions every arm shares live here too: :func:`headline_corr`
builds the zero-diagonal correlation the scan reads, :func:`threshold_metrics`
turns one such matrix into the ``rho_c`` / ``giant_frac`` row the residual,
filtered, and null arms write, and :func:`n_windows` sizes the grid for progress
reporting.

Paper references: rolling-window estimation is subsec:methods / tab:percolation; the
susceptibility-peak threshold is subsec:methods.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from .constants import (
    DRAW_SEEDS,
    MIN_STOCKS_FOR_PERCOLATION,
    MIN_TICKERS_FOR_QUINTILE,
    MIN_VALID_FRACTION,
    N_QUINTILES,
    PER_QUINTILE,
    QUINTILES,
    RHO_RANGE,
    SEED_BASE,
    STEP,
    WINDOW,
)
from .network_utils import (
    find_percolation_threshold,
    find_susceptibility_peak,
    percolation_and_susceptibility_scan,
    threshold_from_corr,
)

# An assigner maps a window end-date to a Series {ticker -> group label}.
QuintileAssigner = Callable[[pd.Timestamp], pd.Series]


def sample_tickers(stocks: list[str], n: int, seed: int) -> list[str]:
    """Deterministic draw of up to *n* tickers (seeded; locked across scripts)."""
    if len(stocks) <= n:
        return list(stocks)
    rng = np.random.default_rng(seed)
    return list(rng.choice(stocks, n, replace=False))


# The job of the running map_draw_seeds call. Forked workers read it from here,
# so a closure over the loaded panels serves as the job without being pickled.
_SEED_JOB: Callable[[int], object] | None = None


def _run_seed_job(seed_base: int):
    if _SEED_JOB is None:
        raise RuntimeError("map_draw_seeds set no job before forking")
    return _SEED_JOB(seed_base)


def map_draw_seeds(job: Callable[[int], object],
                   seeds: Sequence[int] = DRAW_SEEDS) -> list:
    """``[job(s) for s in seeds]``, with one forked worker per seed.

    Results come back in the order of *seeds*. Each worker inherits the
    parent's BLAS thread setting, so launch the script with one BLAS thread
    (``OMP_NUM_THREADS=1``, as reproduce.sh does) to keep the workers from
    oversubscribing the cores.
    """
    global _SEED_JOB
    _SEED_JOB = job
    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(len(seeds), os.cpu_count() or 1)) as pool:
            return pool.map(_run_seed_job, list(seeds), chunksize=1)
    finally:
        _SEED_JOB = None


def seed_mean(frames: Sequence[pd.DataFrame], keys: list[str],
              seeds: Sequence[int] = DRAW_SEEDS
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stack the per-seed row frames and average them over the seeds.

    Returns ``(per_seed, mean)``. ``per_seed`` stacks the frames with a ``seed``
    column that holds each frame's seed_base. ``mean`` holds one row per value
    of *keys*: the mean over the seeds of every numeric column, and ``n_seeds``,
    the number of draws that produced the row. A row that a filter dropped
    under some draws is the mean over the draws that kept it.
    """
    per_seed = pd.concat([f.assign(seed=s) for f, s in zip(frames, seeds)
                          if not f.empty], ignore_index=True)
    values = [c for c in per_seed.columns
              if c not in keys and c != "seed"
              and pd.api.types.is_numeric_dtype(per_seed[c])]
    grouped = per_seed.groupby(keys, sort=True, dropna=False)
    mean = grouped[values].mean()
    mean["n_seeds"] = grouped["seed"].nunique()
    return per_seed, mean.reset_index()


def over_draw_seeds(job: Callable[[int], pd.DataFrame], keys: list[str],
                    seeds: Sequence[int] = DRAW_SEEDS
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run *job* under each seed_base of *seeds* and average its rows over the
    draws (:func:`seed_mean`). *job* takes a seed_base and returns one row per
    unit, keyed by *keys*."""
    return seed_mean(map_draw_seeds(job, seeds), keys, seeds)


def n_windows(n_dates: int, window: int = WINDOW, step: int = STEP) -> int:
    """Number of rolling windows on the grid ``range(0, n_dates - window + 1, step)``."""
    return max(0, (n_dates - window) // step + 1)


def headline_corr(frame: pd.DataFrame) -> np.ndarray:
    """Zero-diagonal correlation of *frame*'s columns, headline convention.

    ``corr() -> fill_diag(0) -> nan_to_num`` -- the matrix the headline
    pipeline feeds to the percolation scan, and the one every residualized,
    filtered, or shuffled arm rebuilds from its own transformed returns.
    """
    c = frame.corr().to_numpy(copy=True)
    np.fill_diagonal(c, 0.0)
    return np.nan_to_num(c)


def assign_quintiles_at(date: pd.Timestamp, marcap: pd.DataFrame) -> pd.Series:
    """Point-in-time size-quintile labels (1..5, 1=smallest) at one date.

    Uses the most recent prior market-cap row (ffill) so callers never peek at
    future market cap. Returns an empty Series if fewer than
    ``MIN_TICKERS_FOR_QUINTILE`` tickers are observable on that date.
    """
    pos = marcap.index.get_indexer([date], method="ffill")[0]
    if pos < 0:
        return pd.Series(dtype="int64")
    row = marcap.iloc[pos].dropna()
    if len(row) < MIN_TICKERS_FOR_QUINTILE:
        return pd.Series(dtype="int64")
    return pd.qcut(
        row.rank(method="first"), N_QUINTILES, labels=list(QUINTILES)
    ).astype(int)


def quintile_assigner(marcap: pd.DataFrame) -> QuintileAssigner:
    """Default assigner: point-in-time size quintiles from *marcap*."""
    return lambda end_date: assign_quintiles_at(end_date, marcap)


@dataclass(frozen=True)
class WindowCell:
    """One (rolling window, group) cell after sampling and column cleaning."""

    w: int                      # window ordinal (0-based) over the grid
    start: int                  # window start index into the date axis
    end_date: pd.Timestamp      # label date = date_idx[start + window - 1]
    quintile: int               # group label emitted by the assigner
    tickers: list[str]          # kept tickers, in sub-column order
    sub: pd.DataFrame           # cleaned return sub-frame

    @property
    def corr(self) -> np.ndarray:
        """Zero-diagonal correlation, headline convention (:func:`headline_corr`)."""
        return headline_corr(self.sub)

    def corr_unit_diag(self) -> np.ndarray:
        """Unit-diagonal correlation for RMT eigendecomposition.

        ``corr() -> nan_to_num(0) -> fill_diag(1)``. The off-diagonal entries
        equal :attr:`corr`; the caller zeroes the diagonal after filtering.
        """
        c = self.sub.corr().to_numpy(copy=True)
        c = np.nan_to_num(c, nan=0.0)
        np.fill_diagonal(c, 1.0)
        return c


def iter_window_cells(
    returns: pd.DataFrame,
    *,
    assigner: QuintileAssigner,
    groups: Iterable[int] = QUINTILES,
    window: int = WINDOW,
    step: int = STEP,
    per_group: int = PER_QUINTILE,
    starts: Iterable[int] | None = None,
    seed_base: int = SEED_BASE,
    min_valid_fraction: float = MIN_VALID_FRACTION,
    min_stocks: int = MIN_STOCKS_FOR_PERCOLATION,
    skip_before: pd.Timestamp | None = None,
) -> Iterator[WindowCell]:
    """Yield one :class:`WindowCell` per (window, group).

    Reproduces the canonical rolling loop exactly: the window grid
    ``range(0, len - window + 1, step)`` (override with *starts* for the
    network-size sub-sample or the bootstrap draws), the draw seed
    ``seed_base + g + start``, and the ``> min_valid_fraction`` / ``< min_stocks``
    column filters. Windows whose end date precedes *skip_before* are skipped
    (the flow / sector-coverage guards). Cells failing any filter are skipped.
    """
    date_idx = returns.index
    grid = range(0, len(date_idx) - window + 1, step) if starts is None else starts
    for w, start in enumerate(grid):
        end = start + window
        end_date = date_idx[end - 1]
        if skip_before is not None and end_date < skip_before:
            continue
        q_series = assigner(end_date)
        if q_series.empty:
            continue
        chunk = returns.iloc[start:end]
        for g in groups:
            tickers = q_series[q_series == g].index.tolist()
            tickers = sample_tickers(tickers, per_group, seed=seed_base + g + start)
            tickers = [t for t in tickers if t in chunk.columns]
            if not tickers:
                continue
            sub = chunk[tickers]
            valid = sub.notna().mean()
            sub = sub.loc[:, valid > min_valid_fraction]
            if sub.shape[1] < min_stocks:
                continue
            yield WindowCell(
                w=w, start=int(start), end_date=end_date, quintile=int(g),
                tickers=list(sub.columns), sub=sub,
            )


def percolation_metrics(corr: np.ndarray, n_stocks: int) -> dict:
    """Percolation + susceptibility summary for one correlation matrix.

    Runs the combined giant-component / susceptibility scan once and returns the
    curvature-inflection threshold ``rho_c`` (subsec:methods) alongside the
    susceptibility-peak threshold ``rho_c_chi`` and companion diagnostics
    (subsec:methods).
    """
    rhos, giants, chi = percolation_and_susceptibility_scan(corr, RHO_RANGE)
    rho_c = find_percolation_threshold(rhos, giants)
    rho_c_chi = find_susceptibility_peak(rhos, chi)
    idx = int(np.argmin(np.abs(rhos - rho_c)))
    if np.isfinite(rho_c_chi):
        idx_chi = int(np.argmin(np.abs(rhos - rho_c_chi)))
        chi_peak = float(chi[idx_chi])
        giant_at_chi = float(giants[idx_chi])
    else:
        chi_peak = float("nan")
        giant_at_chi = float("nan")
    return {
        "rho_c": float(rho_c),
        "rho_c_chi": float(rho_c_chi),
        "chi_peak": chi_peak,
        "giant_at_chi": giant_at_chi,
        "giant_frac": float(giants[idx]),
        "n_stocks": int(n_stocks),
    }


def threshold_metrics(corr: np.ndarray, n_stocks: int) -> dict | None:
    """``rho_c`` and the giant-component fraction at ``rho_c`` for one
    zero-diagonal correlation matrix, without the susceptibility columns of
    :func:`percolation_metrics`: the row every residualized, filtered, or
    shuffled arm writes (tab:residual_steps, subsec:null_robustness).
    ``None`` when the scan finds no threshold.
    """
    rhos, giants, rho_c = threshold_from_corr(corr, RHO_RANGE)
    if not np.isfinite(rho_c):
        return None
    idx = int(np.argmin(np.abs(rhos - rho_c)))
    return {
        "rho_c": float(rho_c),
        "giant_frac": float(giants[idx]),
        "n_stocks": int(n_stocks),
    }


def percolation_one_window(chunk: pd.DataFrame, tickers: list[str]) -> dict | None:
    """Clean *tickers* against *chunk*, build the headline correlation, and
    return :func:`percolation_metrics`; ``None`` if the cell is too sparse.

    The window-iteration counterpart is :func:`iter_window_cells`; this helper
    serves callers that sub-group a single window themselves (e.g. the
    bootstrap 5-year draws).
    """
    tickers = [t for t in tickers if t in chunk.columns]
    if not tickers:
        return None
    sub = chunk[tickers]
    valid = sub.notna().mean()
    sub = sub.loc[:, valid > MIN_VALID_FRACTION]
    if sub.shape[1] < MIN_STOCKS_FOR_PERCOLATION:
        return None
    return percolation_metrics(headline_corr(sub), sub.shape[1])
