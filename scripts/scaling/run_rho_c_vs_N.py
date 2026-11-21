#!/usr/bin/env python3
"""Network-size dependence of rho_c (subsec:resampling_method, subsec:threshold_vs_mean).

The closed-form criterion (paper eq:toy_closed_form) puts rho_c at the
upper-tail quantile F^{-1}(1 - 1/(N-1)) of the absolute-correlation
distribution, so rho_c rises with the network size N as sigma_delta *
sqrt(2 ln(N-1)). Each Korean quintile is resampled at N in N_GRID, rho_c is
averaged over the draws of DRAW_SEEDS in each cell, and its cross-window mean is
regressed on sqrt(2 ln(N-1)) within each quintile and pooled across quintiles
after the quintile means are subtracted. generate_params.py applies the pooled
slope to the network sizes of Q1 and Q5 in each window, which gives the share of
the endpoint contrast that the network size produces (subsec:threshold_vs_mean).

Writes
    data/network_size_resampling/rho_c_vs_N.csv         (per N, quintile cell)
    data/network_size_resampling/rho_c_vs_N_slopes.csv  (per quintile + pooled)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

from fn_percolation.constants import RHO_RANGE, STEP, WINDOW
from fn_percolation.data_loader import load_panels
from fn_percolation.network_utils import threshold_from_corr
from fn_percolation.windows import (
    iter_window_cells, over_draw_seeds, quintile_assigner)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "network_size_resampling"

# Network-size grid and window sub-sampling of subsec:resampling_method.
N_GRID: tuple[int, ...] = (50, 100, 150, 200, 250, 300)
SUBSAMPLE_STEP = 5


def _sqrt_2_ln_n_minus_1(n: np.ndarray) -> np.ndarray:
    """The network-size location term sqrt(2 ln(N-1)) of the closed-form rho_c
    criterion (paper eq:toy_closed_form)."""
    return np.sqrt(2.0 * np.log(n.astype(float) - 1.0))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading returns + market-cap panels ...")
    returns, marcap = load_panels()

    date_idx = returns.index
    all_starts = list(range(0, len(date_idx) - WINDOW + 1, STEP))
    starts = all_starts[::SUBSAMPLE_STEP]
    print(f"N grid: {N_GRID}; {len(starts)} sub-sampled windows per (N, quintile) cell")

    def job(seed_base: int) -> pd.DataFrame:
        rows: list[dict] = []
        for n in N_GRID:
            for cell in iter_window_cells(returns, assigner=quintile_assigner(marcap),
                                          per_group=n, starts=starts,
                                          seed_base=seed_base):
                corr = cell.corr
                _, _, rho_c = threshold_from_corr(corr, RHO_RANGE)
                if not np.isfinite(rho_c):
                    continue
                rows.append({
                    "N": int(n),
                    "date": cell.end_date,
                    "quintile": int(cell.quintile),
                    "n_nodes": int(cell.sub.shape[1]),
                    "rho_c": float(rho_c),
                })
        return pd.DataFrame(rows)

    # Each (N, date, quintile) cell is its mean over the draws of DRAW_SEEDS.
    _, cells = over_draw_seeds(job, ["N", "date", "quintile"])

    # --- Per (N, quintile) cross-window means of rho_c ---
    summary = (cells.groupby(["N", "quintile"])
               .agg(n_windows=("rho_c", "size"),
                    mean_rho_c=("rho_c", "mean"))
               .reset_index()
               .sort_values(["quintile", "N"]))
    summary["sqrt2_ln_N_minus_1"] = _sqrt_2_ln_n_minus_1(summary["N"].to_numpy())
    cell_path = OUT_DIR / "rho_c_vs_N.csv"
    summary.to_csv(cell_path, index=False)
    print(f"\nWrote {cell_path}")
    print(summary.to_string(index=False))

    # --- Per-quintile and pooled OLS slopes on sqrt(2 ln(N-1)) ---
    # The closed form predicts a positive rho_c slope (= sigma_delta).
    def _slope(sub: pd.DataFrame, col: str) -> dict:
        x = sub["sqrt2_ln_N_minus_1"].to_numpy(dtype=float)
        y = sub[col].to_numpy(dtype=float)
        fit = linregress(x, y)
        drift = float(sub.sort_values("N")[col].iloc[-1]
                      - sub.sort_values("N")[col].iloc[0])
        return {"slope": float(fit.slope), "r2": float(fit.rvalue ** 2),
                "p": float(fit.pvalue), "drift_50_to_300": drift}

    slope_rows: list[dict] = []
    for g, sub in summary.groupby("quintile"):
        rc = _slope(sub, "mean_rho_c")
        slope_rows.append({
            "quintile": int(g),
            "rho_c_slope": rc["slope"], "rho_c_r2": rc["r2"],
            "rho_c_p": rc["p"], "rho_c_drift_50_to_300": rc["drift_50_to_300"],
        })
    # Pooled across quintiles (quintile-demeaned, so each quintile contributes its
    # own within-quintile N-slope to one pooled estimate).
    pooled = summary.copy()
    pooled["mean_rho_c_dm"] = (pooled["mean_rho_c"]
                               - pooled.groupby("quintile")["mean_rho_c"].transform("mean"))
    rc_p = linregress(pooled["sqrt2_ln_N_minus_1"], pooled["mean_rho_c_dm"])
    slope_rows.append({
        "quintile": 0,  # 0 = pooled (quintile-demeaned)
        "rho_c_slope": float(rc_p.slope), "rho_c_r2": float(rc_p.rvalue ** 2),
        "rho_c_p": float(rc_p.pvalue), "rho_c_drift_50_to_300": float("nan"),
    })
    slopes = pd.DataFrame(slope_rows)
    slope_path = OUT_DIR / "rho_c_vs_N_slopes.csv"
    slopes.to_csv(slope_path, index=False)
    print(f"\nWrote {slope_path}")
    print(slopes.to_string(index=False))


if __name__ == "__main__":
    main()
