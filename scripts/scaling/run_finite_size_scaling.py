#!/usr/bin/env python3
"""Giant-component curves re-sampled over the network size N.

For each N in N_GRID, the script re-samples N stocks per quintile per window
under each seed of DRAW_SEEDS (point-in-time, seed = seed_base + g + start as in
the headline pipeline) and persists, with its seed, the giant-component curve
G(rho*) of every cell whose rho_c is finite.
scripts/scaling/run_finite_size_drift.py reads the curves back into the drift of
rho_c between the ends of the grid, which subsec:threshold_vs_mean quotes.

Compute budget. The scan loops over 6 N values; windows are sub-sampled by
SUBSAMPLE_STEP (every k-th window along the 20-day rolling grid) to keep the
runtime to a few minutes.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fn_percolation.constants import QUINTILES, RHO_RANGE, STEP, WINDOW
from fn_percolation.data_loader import load_panels
from fn_percolation.network_utils import threshold_from_corr
from fn_percolation.windows import (
    iter_window_cells, map_draw_seeds, quintile_assigner)

REPO_ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = REPO_ROOT / "data" / "finite_size"

# Network-size grid. 6 N values span the practical range we sample in the headline
# pipeline; the smallest (N=50) is at the edge of where rho_c is even well
# defined, and the largest (N=300) matches the headline.
N_GRID: tuple[int, ...] = (50, 100, 150, 200, 250, 300)

# Use every SUBSAMPLE_STEP-th rolling window from the headline grid
# (nWindowsHeadline windows). scripts/scaling/run_mean_corr_vs_N.py mirrors it.
SUBSAMPLE_STEP = 5


def main() -> None:
    print("Loading panels ...")
    returns, marcap = load_panels()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  returns: {returns.shape}")

    date_idx = returns.index
    all_starts = list(range(0, len(date_idx) - WINDOW + 1, STEP))
    starts = all_starts[::SUBSAMPLE_STEP]
    n_windows = len(starts)
    print(f"\nN grid: {N_GRID}")
    print(f"Sub-sampling every {SUBSAMPLE_STEP}-th of {len(all_starts)} rolling windows "
          f"= {n_windows} windows per (N, quintile) cell")
    print(f"Total cells: {len(N_GRID) * len(QUINTILES) * n_windows}")

    def job(seed_base: int) -> pd.DataFrame:
        curve_rows: list[dict] = []
        for n in N_GRID:
            for cell in iter_window_cells(returns, assigner=quintile_assigner(marcap),
                                          per_group=n, starts=starts,
                                          seed_base=seed_base):
                rhos, giants, rho_c = threshold_from_corr(cell.corr, RHO_RANGE)
                if not np.isfinite(rho_c):
                    continue

                # Persist the full G(rho*) curve, from which
                # run_finite_size_drift.py recomputes rho_c without a second
                # scan of the correlation matrix.
                for idx, (r, gval) in enumerate(zip(rhos, giants)):
                    curve_rows.append({
                        "seed": int(seed_base),
                        "N": int(n),
                        "date": cell.end_date,
                        "quintile": int(cell.quintile),
                        "rho_grid_idx": int(idx),
                        "rho": float(r),
                        "G": float(gval),
                    })
        print(f"  seed_base {seed_base} done", flush=True)
        return pd.DataFrame(curve_rows)

    # Persisted as parquet: one row per rho-grid point per cell and draw.
    curves_df = pd.concat(map_draw_seeds(job), ignore_index=True)
    curves_path = OUT_DIR / "giant_curves_per_N.parquet"
    curves_df.to_parquet(curves_path, index=False)
    print(f"Wrote {curves_path} ({len(curves_df):,} rows)")


if __name__ == "__main__":
    main()
