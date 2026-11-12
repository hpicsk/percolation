"""
Network analysis utilities: percolation scan, threshold detection, null model.

Implements the percolation primitives of paper subsec:percolation_definition (eq:adjacency adjacency,
eq:giant_component giant-component fraction, eq:percolation_threshold curvature-inflection rho_c) and the
shuffled-returns null of paper subsec:null_robustness.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.sparse.csgraph import connected_components
from scipy.stats import linregress
from typing import Optional

from .constants import MIN_WINDOW_OBS

# Spline-mesh resolution for threshold detection (paper subsec:methods: 2000-point
# mesh over [0.05, 0.9], approx 4x10^-4 resolution).
SPLINE_MESH_SIZE = 2000
# Minimum number of grid points required to fit a cubic spline.
MIN_GRID_POINTS = 4


def percolation_scan(corr_matrix: np.ndarray, rho_range: np.ndarray,
                     labels: Optional[list[str]] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Scan giant component size as a function of correlation threshold.

    Uses scipy.sparse.csgraph.connected_components on the boolean adjacency
    |corr_ij| > rho* per threshold. ``labels`` is accepted but ignored; the
    function returns only the giant-component fraction, which is label-agnostic.

    Returns:
        (rho_values, giant_sizes): arrays of thresholds and corresponding giant component fractions.
    """
    N = corr_matrix.shape[0]
    if N == 0:
        return rho_range, np.zeros(len(rho_range))

    abs_corr = np.abs(corr_matrix)
    sizes = np.zeros(len(rho_range))
    for i, rho in enumerate(rho_range):
        adj = abs_corr > rho
        n_comp, labels_arr = connected_components(adj, directed=False)
        if n_comp > 0:
            sizes[i] = np.bincount(labels_arr).max() / N
    return rho_range, sizes


def percolation_and_susceptibility_scan(
    corr_matrix: np.ndarray,
    rho_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan giant-component fraction G(rho*) and percolation susceptibility chi(rho*).

    chi(rho*) = (1/N) * sum_{C != C_max} |C|^2 is the standard mean cluster size
    excluding the giant component (Stauffer & Aharony, Sec. 2.4). Its peak at
    rho_c is the textbook percolation observable and complements the
    curvature-inflection rho_c of find_percolation_threshold().

    Both G and chi are computed from the same connected_components call per
    threshold, so this combined scan replaces percolation_scan() when both
    diagnostics are wanted.
    """
    N = corr_matrix.shape[0]
    if N == 0:
        z = np.zeros(len(rho_range))
        return rho_range, z, z.copy()

    abs_corr = np.abs(corr_matrix)
    G = np.zeros(len(rho_range))
    chi = np.zeros(len(rho_range))
    for i, rho in enumerate(rho_range):
        adj = abs_corr > rho
        n_comp, labels_arr = connected_components(adj, directed=False)
        if n_comp == 0:
            continue
        comp_sizes = np.bincount(labels_arr)
        s_max = int(comp_sizes.max())
        G[i] = s_max / N
        chi[i] = float((comp_sizes.astype(np.int64) ** 2).sum() - s_max ** 2) / N
    return rho_range, G, chi


def off_diagonal_mean_abs(corr: np.ndarray) -> float:
    """Mean of |rho_ij| over the n*(n-1) off-diagonal entries (subsec:main_result, tab:percolation
    final row). ``cell.corr`` carries a zeroed diagonal and nan->0 off-diagonals,
    so the sum of |.| over the whole matrix divided by the off-diagonal count is
    the off-diagonal mean."""
    n = corr.shape[0]
    if n < 2:
        return float("nan")
    return float(np.abs(corr).sum() / (n * (n - 1)))


def find_percolation_threshold(rho_values: np.ndarray, giant_sizes: np.ndarray,
                               mesh_size: int = SPLINE_MESH_SIZE) -> float:
    """
    Find the percolation threshold as the rho of steepest decline in the
    giant component: argmin of the first derivative of a cubic spline
    interpolant of G(rho*).

    The spline is fit to (rho_values, giant_sizes) without smoothing; its
    analytic first derivative is evaluated on a fine uniform mesh (paper
    subsec:methods: 2000-point mesh over [0.05, 0.9]), and the threshold is the
    mesh argmin (G' is nonpositive everywhere since G is monotone decreasing
    in rho, so the argmin coincides with the steepest drop).
    """
    if len(rho_values) < MIN_GRID_POINTS:
        return np.nan

    spline = CubicSpline(rho_values, giant_sizes)
    mesh = np.linspace(rho_values[0], rho_values[-1], mesh_size)
    d1 = spline(mesh, 1)
    return float(mesh[np.argmin(d1)])


def threshold_from_corr(
    corr_matrix: np.ndarray,
    rho_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Percolation scan plus curvature-inflection threshold for one correlation
    matrix (subsec:methods).

    Composes :func:`percolation_scan` and :func:`find_percolation_threshold` in
    the single order every per-window caller uses. Returns
    ``(rho_values, giant_sizes, rho_c)`` so callers that also need the giant
    fraction or the full G(rho*) curve reuse the scan output instead of
    re-scanning; callers wanting only the threshold take ``*_, rho_c``. ``rho_c``
    is NaN when the grid is too coarse to fit the spline.
    """
    rhos, giants = percolation_scan(corr_matrix, rho_range)
    rho_c = find_percolation_threshold(rhos, giants)
    return rhos, giants, rho_c


def find_susceptibility_peak(rho_values: np.ndarray, chi_values: np.ndarray,
                             mesh_size: int = SPLINE_MESH_SIZE) -> float:
    """Return rho_c as the spline-interpolated argmax of chi(rho*).

    Uses the same cubic-spline interpolation strategy as
    find_percolation_threshold() so that the peak is not snapped to a grid
    node when the grid spacing is comparable to the precision of interest.
    """
    if len(rho_values) < MIN_GRID_POINTS:
        return float("nan")
    chi = np.asarray(chi_values, dtype=float)
    if not np.any(np.isfinite(chi)) or float(np.nanmax(chi)) <= 0.0:
        return float("nan")
    spline = CubicSpline(rho_values, chi)
    mesh = np.linspace(rho_values[0], rho_values[-1], mesh_size)
    chi_mesh = spline(mesh)
    return float(mesh[int(np.argmax(chi_mesh))])


def shuffle_returns_within_stock(returns_chunk: pd.DataFrame,
                                 seed: int) -> pd.DataFrame:
    """Independent time-permutation per stock column.

    Each column's non-NaN observations are permuted in time; NaN positions are
    preserved. Marginal distributions (volatility, skewness, tails) are
    unchanged, while all cross-sectional correlation structure is destroyed.

    Used as the shuffled-returns null of paper subsec:null_robustness, tab:robustness_summary: if the
    gradient survives on shuffled returns, it is an artifact of differing
    noise properties across size bins rather than genuine cross-sectional
    signal.
    """
    rng = np.random.default_rng(seed)
    arr = returns_chunk.values.copy()
    for j in range(arr.shape[1]):
        col = arr[:, j]
        mask = ~np.isnan(col)
        if mask.sum() < 2:
            continue
        arr[mask, j] = rng.permutation(col[mask])
    return pd.DataFrame(arr, index=returns_chunk.index,
                        columns=returns_chunk.columns)


def value_weighted_market_return(returns: pd.DataFrame,
                                 marcap: pd.DataFrame) -> pd.Series:
    """Cap-weighted daily log-return across the full common-stock universe.

    Weights are the previous day's market capitalization (``marcap.shift(1)``)
    restricted to tickers with a valid return that day, so the series never
    weights a stock's return by a market cap that return itself helped set.

    Shared by the market-model residual arm (paper tab:residual_steps) and
    its market+sector two-factor extension, both of which regress each
    stock's return on this series within a rolling window.
    """
    weights = marcap.shift(1).where(returns.notna())
    numer = (returns * weights).sum(axis=1, skipna=True)
    denom = weights.sum(axis=1, skipna=True)
    return (numer / denom).rename("market_return")


def equal_weighted_market_return(returns: pd.DataFrame) -> pd.Series:
    """Equal-weighted daily return across the full common-stock universe.

    Robustness alternative to :func:`value_weighted_market_return`: every
    stock with a valid return that day contributes ``1/n`` regardless of
    market cap, so no single large-cap name can dominate the factor the way
    it can under cap-weighting.

    Cross-check for the market-model residual arm of paper
    tab:residual_steps: whether the single-factor result depends on
    cap-weighting specifically.
    """
    return returns.mean(axis=1, skipna=True).rename("market_return")


def residualize_against_market(returns_chunk: pd.DataFrame,
                               market_chunk: pd.Series) -> pd.DataFrame:
    """Per-stock market-model residuals over one rolling window.

    For each column, fits ``r_i = alpha_i + beta_i r_mkt + epsilon_i`` by
    univariate OLS on the dates where both the stock and the market return are
    observed within the window, and returns epsilon_i at those dates (NaN
    elsewhere). Columns with fewer than ``MIN_WINDOW_OBS`` jointly-valid dates
    are left all-NaN.

    Used by the market-model residual arms of paper tab:residual_steps:
    if the cross-quintile rho_c gradient survives computing correlations on
    epsilon instead of on raw returns, the gradient is not solely a
    continuous common-factor-loading difference across size quintiles (a
    confound the within-sector ranking of subsec:size_confounds cannot rule
    out, since it controls sector membership but not a within-sector beta or
    R^2 gradient).
    """
    out = pd.DataFrame(np.nan, index=returns_chunk.index,
                       columns=returns_chunk.columns)
    mkt = market_chunk.reindex(returns_chunk.index).to_numpy(dtype=float)
    for col in returns_chunk.columns:
        y = returns_chunk[col].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(mkt)
        if mask.sum() < MIN_WINDOW_OBS:
            continue
        fit = linregress(mkt[mask], y[mask])
        if not np.isfinite(fit.slope):
            continue
        resid = y[mask] - (fit.intercept + fit.slope * mkt[mask])
        out.loc[returns_chunk.index[mask], col] = resid
    return out


def residualize_against_market_and_sector(
    returns_chunk: pd.DataFrame,
    market_chunk: pd.Series,
    sector_returns_chunk: pd.DataFrame,
    ticker_sector: pd.Series,
) -> pd.DataFrame:
    """Per-stock market+sector two-factor residuals over one rolling window.

    Extends :func:`residualize_against_market` with a second regressor: each
    column's own KRX sector's value-weighted return (``sector_returns_chunk``,
    from :func:`fn_percolation.sector_loader.sector_weighted_returns`). Fits
    ``r_i = alpha_i + beta_mkt,i r_mkt + beta_sec,i r_sec(i) + epsilon_i`` by
    least squares on the dates where the stock, the market return, and its
    sector's return are all observed within the window. A column whose sector
    is unmapped, or was below ``MIN_SECTOR_SIZE`` at the window's end date (so
    absent from ``sector_returns_chunk``'s columns), is left all-NaN -- same
    convention as the ``MIN_WINDOW_OBS`` guard below.

    Two-factor extension of the market-model residual arms (paper
    tab:residual_steps): tests whether the single-factor result is
    specific to the pure market factor, or whether a within-sector size
    gradient also fails to survive once a sector-level common factor is
    removed too.
    """
    out = pd.DataFrame(np.nan, index=returns_chunk.index,
                       columns=returns_chunk.columns)
    mkt = market_chunk.reindex(returns_chunk.index).to_numpy(dtype=float)
    for col in returns_chunk.columns:
        sector = ticker_sector.get(col)
        if sector is None or sector not in sector_returns_chunk.columns:
            continue
        sec = sector_returns_chunk[sector].reindex(
            returns_chunk.index).to_numpy(dtype=float)
        y = returns_chunk[col].to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(mkt) & np.isfinite(sec)
        if mask.sum() < MIN_WINDOW_OBS:
            continue
        design = np.column_stack(
            [np.ones(int(mask.sum())), mkt[mask], sec[mask]])
        beta, *_ = np.linalg.lstsq(design, y[mask], rcond=None)
        resid = y[mask] - design @ beta
        out.loc[returns_chunk.index[mask], col] = resid
    return out
