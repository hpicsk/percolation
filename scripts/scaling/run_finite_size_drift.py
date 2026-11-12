#!/usr/bin/env python3
"""Drift of the percolation threshold with the network size, rho_c(N).

The closed form of eq:toy_closed_form makes rho_c rise with the network size N
at fixed delta. This script measures that shift between the ends of the N grid
and estimates no scaling exponent.

Method:
1. Load data/finite_size/giant_curves_per_N.parquet (output of
   scripts/scaling/run_finite_size_scaling.py).
2. Per (draw seed, quintile, date, N), compute rho_c(N) as the curvature
   inflection, the same definition the headline pipeline uses
   (eq:percolation_threshold), and average each cell over the draws.
3. Per (quintile, N), average rho_c across windows, and report the drift
   from the smallest to the largest N on the grid.
4. Save data/finite_size/rho_c_vs_N.csv.

Compute budget: ~1 min, dominated by step 2.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fn_percolation.constants import QUINTILES
from fn_percolation.network_utils import find_percolation_threshold

REPO_ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = REPO_ROOT / "data" / "finite_size"

# Endpoints of the network-size grid run by run_finite_size_scaling.py.
# subsec:threshold_vs_mean quotes the drift between them.
N_SMALL = 50
N_LARGE = 300


def _per_cell_rho_c(curve_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rho_c per (N, date, quintile) cell via curvature inflection, as
    the mean over the draws of the cell's per-draw rho_c.

    The curve_df is the long-format parquet output by
    run_finite_size_scaling.py with columns (seed, N, date, quintile,
    rho_grid_idx, rho, G).
    """
    rows: list[dict] = []
    for (seed, n, date, q), sub in curve_df.groupby(
            ["seed", "N", "date", "quintile"]):
        sub = sub.sort_values("rho_grid_idx")
        rho_c = find_percolation_threshold(
            sub["rho"].to_numpy(), sub["G"].to_numpy()
        )
        if np.isfinite(rho_c):
            rows.append({"seed": int(seed), "N": int(n), "date": date,
                         "quintile": int(q), "rho_c": float(rho_c)})
    return (pd.DataFrame(rows).groupby(["N", "date", "quintile"])["rho_c"]
            .mean().reset_index())


def main() -> None:
    curves_path = OUT_DIR / "giant_curves_per_N.parquet"
    print(f"Loading {curves_path} ...")
    curve_df = pd.read_parquet(curves_path)
    print(f"  curves: {len(curve_df):,} rows, "
          f"{curve_df['quintile'].nunique()} quintiles, "
          f"{curve_df['N'].nunique()} N values, "
          f"{curve_df['date'].nunique()} dates")

    print("\nComputing rho_c per (N, date, quintile) cell ...")
    rho_c_df = _per_cell_rho_c(curve_df)
    print(f"  {len(rho_c_df):,} valid cells")

    summary: list[dict] = []
    for q in QUINTILES:
        q_rho_c = rho_c_df[rho_c_df["quintile"] == q]
        by_n = q_rho_c.groupby("N")["rho_c"].mean()
        small = float(by_n.get(N_SMALL, np.nan))
        large = float(by_n.get(N_LARGE, np.nan))
        print(f"  Q{q}: rho_c(N={N_SMALL})={small:.3f}, "
              f"rho_c(N={N_LARGE})={large:.3f}, drift={large - small:+.3f}")
        summary.append({
            "quintile": q,
            "n_N_values": int(by_n.size),
            f"rho_c_N_{N_SMALL}": small,
            f"rho_c_N_{N_LARGE}": large,
            "rho_c_drift_50_to_300": large - small,
        })

    out_csv = OUT_DIR / "rho_c_vs_N.csv"
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
