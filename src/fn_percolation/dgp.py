"""Data-generating-process simulator for the toy return model.

Implements the stylised return-generating process of paper subsec:toy_model (eq:toy_rgp,
eq:toy_rgp),

    r_i = beta_i * f + delta * u_i + (1 - delta) * sum_{j != i} W_ij * u_j,

together with the closed-form percolation-threshold approximation of eq:toy_closed_form
(eq:toy_closed_form),

    rho_c(delta, N, T) ~= mu(delta) + sigma_delta * sqrt(2 * ln(N - 1)),

and a wrapper that runs the simulated returns through the same percolation
machinery used for empirical pipelines (network_utils.percolation_and_susceptibility_scan,
find_percolation_threshold, find_susceptibility_peak). Identification of rho_c
on simulated returns is therefore byte-identical to the empirical identification.

Used by scripts/dgp/run_dgp_validation.py to populate si:dgp_validation (DGP validation).

Default DGP parameters:
    sigma_f = 0.007  (factor std)
    sigma_u = 0.02   (idiosyncratic std, plausible daily-return scale; not
                      calibrated to data, see paper si:dgp_validation)
    beta ~ Normal(1.0, 0.3) clipped to [0.2, 2.0]

The factor std is set to 0.007 rather than the naive 0.01 (a plausible ~1%
index daily-return scale) so the DGP's rho_c_sim range brackets the empirical
Q1-Q5 range of tab:percolation. This empirical anchor is
documented in the appendix as DGP_SIGMA_F_SCALE = 0.7 (where 1.0 would be
the un-scaled factor std). The remaining DGP parameters are fixed at
plausible daily-return scales without further tuning.
"""

from __future__ import annotations

import numpy as np

from .constants import RHO_RANGE
from .network_utils import (
    find_percolation_threshold,
    find_susceptibility_peak,
    percolation_and_susceptibility_scan,
)

DEFAULT_SIGMA_F = 0.007
DEFAULT_SIGMA_F_UNSCALED = 0.01
DEFAULT_SIGMA_F_SCALE = DEFAULT_SIGMA_F / DEFAULT_SIGMA_F_UNSCALED
DEFAULT_SIGMA_U = 0.02
DEFAULT_BETA_MEAN = 1.0
DEFAULT_BETA_STD = 0.3
DEFAULT_BETA_CLIP = (0.2, 2.0)


def draw_betas(N: int, rng: np.random.Generator) -> np.ndarray:
    """Factor loadings: Normal(1.0, 0.3) clipped to [0.2, 2.0].

    Clipping produces atoms at the bounds rather than a true truncated
    normal; at sigma=0.3 the clipped tail mass is <0.5% on each side, so
    the qualitative behaviour matches a truncated normal but the
    distribution is technically a rectified normal. The SI prose reflects
    this.
    """
    lo, hi = DEFAULT_BETA_CLIP
    beta = rng.normal(DEFAULT_BETA_MEAN, DEFAULT_BETA_STD, size=N)
    return np.clip(beta, lo, hi)


def build_peer_matrix_knn(N: int, k: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Random peer-coupling matrix, k links per row at random, with row sums normalised to 1.

    Each row i receives k entries at uniformly-chosen columns drawn without
    replacement from {0..N-1}\\{i}, each with weight 1/k. Diagonals are zero.
    """
    if k >= N:
        raise ValueError(f"k ({k}) must be < N ({N})")
    W = np.zeros((N, N), dtype=np.float64)
    weight = 1.0 / k
    for i in range(N):
        choices = rng.choice(N - 1, size=k, replace=False)
        # Map indices to skip row i (so column never equals i).
        cols = np.where(choices >= i, choices + 1, choices)
        W[i, cols] = weight
    return W


def build_peer_matrix_sector(N: int, n_sectors: int, intra: float,
                             inter: float,
                             rng: np.random.Generator) -> np.ndarray:
    """Sector-block peer-coupling matrix with row sums normalised to 1.

    Each stock is assigned uniformly to one of ``n_sectors`` sectors.
    Off-diagonal entry W_ij carries weight ``intra`` if i and j share a
    sector, and weight ``inter`` otherwise. Rows are then divided by their
    row sum so each row sums to 1. Diagonals are zero.
    """
    if intra <= 0 or inter < 0:
        raise ValueError("intra > 0 and inter >= 0 required")
    sectors = rng.integers(0, n_sectors, size=N)
    same = sectors[:, None] == sectors[None, :]
    W = np.where(same, intra, inter).astype(np.float64)
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return W / row_sums


def simulate_returns(N: int, T: int, delta,
                     W: np.ndarray, beta: np.ndarray,
                     sigma_f: float, sigma_u: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Generate T x N returns under eq:toy_rgp.

    ``delta`` may be a scalar (segment-uniform absorption) or a length-N
    array (stock-level absorption). The factor channel adds a column-shared
    term beta_i * f_t; the local-absorption term carries delta * u_{i,t};
    the propagation term carries (1 - delta) * sum_j W_ij * u_{j,t}.
    """
    delta_arr = np.asarray(delta, dtype=np.float64)
    if delta_arr.ndim == 0:
        delta_arr = np.full(N, float(delta_arr))
    if delta_arr.shape != (N,):
        raise ValueError(
            f"delta must be scalar or length-N; got shape {delta_arr.shape}"
        )

    f = rng.standard_normal(T) * sigma_f                       # (T,)
    u = rng.standard_normal((T, N)) * sigma_u                   # (T, N)
    factor = np.outer(f, beta)                                 # (T, N)
    propagation = u @ W.T                                      # (T, N)
    return factor + delta_arr * u + (1.0 - delta_arr) * propagation


def rho_c_closed_form(N: int, T: int, mu_delta: float,
                      sigma_delta: float) -> float:
    """Evaluate the eq:toy_closed_form Mills-ratio approximation.

    rho_c ~= mu_delta + sigma_delta * sqrt(2 * ln(N - 1))

    T is accepted for interface symmetry with the empirical pipeline but
    does not enter eq:toy_closed_form explicitly (its effect is absorbed into the
    estimated mu_delta and sigma_delta from the simulated bulk).
    """
    return float(mu_delta + sigma_delta * np.sqrt(2.0 * np.log(N - 1)))


def upper_tri_abs(corr: np.ndarray) -> np.ndarray:
    """Off-diagonal absolute correlations as a flat vector."""
    iu = np.triu_indices_from(corr, k=1)
    return np.abs(corr[iu])


def percolation_one_sim(
    N: int,
    T: int,
    delta,
    W: np.ndarray,
    beta: np.ndarray,
    sigma_f: float,
    sigma_u: float,
    rng: np.random.Generator,
    rho_range: np.ndarray = RHO_RANGE,
) -> dict:
    """Simulate eq:toy_rgp, build the correlation matrix, run the scan.

    Returns a dict with keys:
        rho_c           curvature-inflection threshold (paper eq:percolation_threshold)
        rho_c_chi       susceptibility-peak threshold (Stauffer textbook)
        mu_bulk         mean of off-diagonal |corr| (the eq:toy_closed_form mu_delta proxy)
        sigma_bulk      std of off-diagonal |corr| (the eq:toy_closed_form sigma_delta proxy)
        sigma_bulk_iqr  robust scale: (P84 - P16) / 2, alternative to std
        rho_c_closed    eq:toy_closed_form evaluated at (mu_bulk, sigma_bulk)
        giant_at_chi    G(rho*) at rho_c_chi
        n_pairs         off-diagonal pair count (sanity check)
    """
    R = simulate_returns(N, T, delta, W, beta, sigma_f, sigma_u, rng)
    # Pearson correlation of columns. ddof=1 matches pandas .corr() up to
    # sample-mean centring.
    Rc = R - R.mean(axis=0, keepdims=True)
    std = Rc.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    Rn = Rc / std
    corr = (Rn.T @ Rn) / (T - 1)
    np.fill_diagonal(corr, 0.0)

    bulk = upper_tri_abs(corr)
    mu_bulk = float(bulk.mean())
    sigma_bulk = float(bulk.std(ddof=1))
    p16, p84 = np.percentile(bulk, [16.0, 84.0])
    sigma_bulk_iqr = float((p84 - p16) / 2.0)
    # eq:toy_threshold degree-one criterion solved directly from the
    # empirical bulk CDF: rho_c = F^{-1}(1 - 1/(N-1)). This is the principled
    # comparator for the percolation pipeline's rho_c. eq:toy_closed_form below is the
    # Mills-ratio Gaussian asymptote of the same criterion and is
    # systematically biased by the non-Gaussian shape of the bulk.
    q = 1.0 - 1.0 / (N - 1)
    rho_c_degree_one = float(np.quantile(bulk, q))

    rhos, giants, chi = percolation_and_susceptibility_scan(corr, rho_range)
    rho_c = find_percolation_threshold(rhos, giants)
    rho_c_chi = find_susceptibility_peak(rhos, chi)
    rho_c_closed = rho_c_closed_form(N, T, mu_bulk, sigma_bulk)

    if np.isfinite(rho_c_chi):
        idx = int(np.argmin(np.abs(rhos - rho_c_chi)))
        giant_at_chi = float(giants[idx])
    else:
        giant_at_chi = float("nan")

    return {
        "rho_c": float(rho_c),
        "rho_c_chi": float(rho_c_chi),
        "rho_c_degree_one": rho_c_degree_one,
        "mu_bulk": mu_bulk,
        "sigma_bulk": sigma_bulk,
        "sigma_bulk_iqr": sigma_bulk_iqr,
        "rho_c_closed": rho_c_closed,
        "giant_at_chi": giant_at_chi,
        "n_pairs": int(len(bulk)),
    }


def make_peer_matrix(W_type: str, N: int,
                     rng: np.random.Generator,
                     k_knn: int = 5,
                     n_sectors: int = 10,
                     intra: float = 1.0,
                     inter: float = 0.1) -> np.ndarray:
    """Dispatch on W_type in {"knn", "sector"}.

    Keeps the experiment driver free of peer-matrix-construction branching.
    """
    if W_type == "knn":
        return build_peer_matrix_knn(N, k_knn, rng)
    if W_type == "sector":
        return build_peer_matrix_sector(N, n_sectors, intra, inter, rng)
    raise ValueError(f"unknown W_type: {W_type!r}")
