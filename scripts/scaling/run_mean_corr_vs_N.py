#!/usr/bin/env python3
"""Network-size divergence of rho_c from the segment mean correlation (subsec:threshold_vs_mean).

A size gradient in rho_c could be a monotone relabeling of the mean pairwise
correlation that adds nothing to the first moment. The cross-section cannot
settle this because rho_c and the segment mean |rho_ij| co-move across quintiles
(Spearman ~0.90; see scripts/scaling/run_mean_corr_diagnostic.py). The network-size
dimension separates them.

The closed-form criterion (paper eq:toy_closed_form) makes rho_c
the upper-tail quantile F^{-1}(1 - 1/(N-1)) of the absolute-correlation
distribution, so rho_c rises with the network size N as sigma_delta *
sqrt(2 ln(N-1)). The segment mean |rho_ij| is the first moment of the same
distribution and carries no such N term, so it is flat in N. Measuring both on
the same sampled cells over the network-size grid N in {50,...,300} therefore
gives a falsifiable separation: a positive rho_c slope on sqrt(2 ln(N-1)) and a
near-zero mean-|rho| slope. A monotone relabeling could not produce different
N-slopes.

The sampling mirrors scripts/scaling/run_finite_size_scaling.py byte-for-byte (same
N_GRID, same SUBSAMPLE_STEP, same draw seeds DRAW_SEEDS, and the same mean of each
cell over the draws), so the rho_c(N) recomputed here matches, at both ends of
the grid, the means in data/finite_size/rho_c_vs_N.csv.

Writes
    data/mean_corr_diagnostic/mean_corr_vs_N.csv         (per N, quintile cell)
    data/mean_corr_diagnostic/mean_corr_vs_N_slopes.csv  (per quintile + pooled)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

from fn_percolation.constants import RHO_RANGE, STEP, WINDOW
from fn_percolation.data_loader import load_panels
from fn_percolation.network_utils import off_diagonal_mean_abs, threshold_from_corr
from fn_percolation.windows import (
    iter_window_cells, over_draw_seeds, quintile_assigner)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "mean_corr_diagnostic"

# Mirror the network-size grid and window sub-sampling of
# run_finite_size_scaling.py exactly, so the recomputed rho_c(N) matches
# data/finite_size/rho_c_vs_N.csv.
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
                    "mean_abs_corr": off_diagonal_mean_abs(corr),
                })
        return pd.DataFrame(rows)

    # Each (N, date, quintile) cell is its mean over the draws of DRAW_SEEDS.
    _, cells = over_draw_seeds(job, ["N", "date", "quintile"])

    # --- Per (N, quintile) cross-window means of rho_c and mean |rho| ---
    summary = (cells.groupby(["N", "quintile"])
               .agg(n_windows=("rho_c", "size"),
                    mean_rho_c=("rho_c", "mean"),
                    mean_abs_corr=("mean_abs_corr", "mean"))
               .reset_index()
               .sort_values(["quintile", "N"]))
    summary["sqrt2_ln_N_minus_1"] = _sqrt_2_ln_n_minus_1(summary["N"].to_numpy())
    cell_path = OUT_DIR / "mean_corr_vs_N.csv"
    summary.to_csv(cell_path, index=False)
    print(f"\nWrote {cell_path}")
    print(summary.to_string(index=False))

    # --- Per-quintile and pooled OLS slopes on sqrt(2 ln(N-1)) ---
    # The closed form predicts a positive rho_c slope (= sigma_delta) and a
    # near-zero mean-|rho| slope.
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
        mc = _slope(sub, "mean_abs_corr")
        slope_rows.append({
            "quintile": int(g),
            "rho_c_slope": rc["slope"], "rho_c_r2": rc["r2"],
            "rho_c_p": rc["p"], "rho_c_drift_50_to_300": rc["drift_50_to_300"],
            "mean_corr_slope": mc["slope"], "mean_corr_r2": mc["r2"],
            "mean_corr_p": mc["p"], "mean_corr_drift_50_to_300": mc["drift_50_to_300"],
        })
    # Pooled across quintiles (quintile-demeaned, so each segment contributes its
    # own within-quintile N-slope to one pooled estimate).
    pooled = summary.copy()
    for col in ("mean_rho_c", "mean_abs_corr"):
        pooled[col + "_dm"] = pooled[col] - pooled.groupby("quintile")[col].transform("mean")
    rc_p = linregress(pooled["sqrt2_ln_N_minus_1"], pooled["mean_rho_c_dm"])
    mc_p = linregress(pooled["sqrt2_ln_N_minus_1"], pooled["mean_abs_corr_dm"])
    slope_rows.append({
        "quintile": 0,  # 0 = pooled (quintile-demeaned)
        "rho_c_slope": float(rc_p.slope), "rho_c_r2": float(rc_p.rvalue ** 2),
        "rho_c_p": float(rc_p.pvalue), "rho_c_drift_50_to_300": float("nan"),
        "mean_corr_slope": float(mc_p.slope), "mean_corr_r2": float(mc_p.rvalue ** 2),
        "mean_corr_p": float(mc_p.pvalue), "mean_corr_drift_50_to_300": float("nan"),
    })
    slopes = pd.DataFrame(slope_rows)
    slope_path = OUT_DIR / "mean_corr_vs_N_slopes.csv"
    slopes.to_csv(slope_path, index=False)
    print(f"\nWrote {slope_path}")
    print(slopes.to_string(index=False))


if __name__ == "__main__":
    main()
