#!/usr/bin/env python3
"""Shuffled-returns null for the cross-quintile percolation gradient.

For each of `N_SEEDS` independent shuffles, every stock's return series within
the 60-day window is permuted in time independently of all other stocks. The
marginal distribution of each series is preserved; all cross-sectional
correlation is destroyed. Shuffle k runs on the draws of the k-th seed_base of
DRAW_SEEDS, so the shuffles also span the draws that the headline averages
over.

If the cross-quintile rho_c gradient survives this null, it is a finite-sample
artifact of differing noise properties across size bins (e.g. heavier-tailed
small-cap returns producing more spurious correlations). If the gradient
vanishes, the headline gradient is signal.

The output aggregates per-(seed, quintile) mean rho_c across the 245 rolling
windows so the cross-quintile gradient can be read off directly.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fn_percolation.constants import DRAW_SEEDS, RHO_RANGE
from fn_percolation.data_loader import load_panels
from fn_percolation.network_utils import (
    shuffle_returns_within_stock,
    threshold_from_corr,
)
from fn_percolation.windows import (
    headline_corr,
    iter_window_cells,
    map_draw_seeds,
    n_windows,
    quintile_assigner,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = REPO_ROOT / "data" / "null_shuffled"

# Paper subsec:null_robustness, tab:robustness_summary (shuffled row): one
# independent shuffle per draw seed; per-stock seed = 10_000 * shuffle_seed +
# window_start + quintile, with shuffle_seed the index k of the draw seed.
N_SEEDS = len(DRAW_SEEDS)


def main() -> None:
    print("Loading panels ...")
    returns, marcap = load_panels()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_windows_total = n_windows(len(returns.index))
    print(f"\nShuffled-returns null: {N_SEEDS} seeds x {n_windows_total} windows "
          f"x 5 quintiles ...")

    def job(seed_base: int) -> pd.DataFrame:
        seed = DRAW_SEEDS.index(seed_base)
        seed_rows: list[dict] = []
        for cell in iter_window_cells(returns, assigner=quintile_assigner(marcap),
                                      seed_base=seed_base):
            # Permute each stock's series in time independently (per-stock seed
            # = 10_000 * shuffle_seed + window_start + quintile), then rebuild
            # the correlation matrix from the shuffled returns.
            shuffled = shuffle_returns_within_stock(
                cell.sub, seed=10_000 * seed + cell.start + cell.quintile
            )
            *_, rho_c = threshold_from_corr(headline_corr(shuffled), RHO_RANGE)
            if not np.isfinite(rho_c):
                continue
            seed_rows.append({
                "seed": seed,
                "date": cell.end_date,
                "quintile": cell.quintile,
                "rho_c": float(rho_c),
            })
        print(f"  seed {seed + 1}/{N_SEEDS} done ({len(seed_rows):,} rows)",
              flush=True)
        return pd.DataFrame(seed_rows)

    long_df = pd.concat(map_draw_seeds(job), ignore_index=True)
    long_path = OUT_DIR / "rho_c_distribution.csv"
    long_df.to_csv(long_path, index=False)
    print(f"\nWrote {long_path} ({len(long_df):,} rows)")

    summary_rows = []
    for (seed, g), sub in long_df.groupby(["seed", "quintile"]):
        summary_rows.append({
            "seed": int(seed),
            "quintile": int(g),
            "n_windows": int(len(sub)),
            "mean_rho_c": float(sub["rho_c"].mean()),
            "std_rho_c": float(sub["rho_c"].std()),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "summary_per_seed_per_quintile.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    cross = summary.groupby("quintile")["mean_rho_c"].agg(["mean", "std"]).reset_index()
    cross.columns = ["quintile", "across_seed_mean", "across_seed_std"]
    cross_path = OUT_DIR / "cross_quintile_summary.csv"
    cross.to_csv(cross_path, index=False)
    print(f"Wrote {cross_path}")
    print(cross.to_string(index=False))


if __name__ == "__main__":
    main()
