#!/usr/bin/env python3
"""Per-quintile excess kurtosis of raw daily returns.

Diagnostic for why the shuffled-returns null of tab:robustness_summary
(subsec:null_robustness) has a reversed cross-quintile rho_c slope (Q1 > Q5).
Shuffling is a per-stock time permutation (network_utils.shuffle_returns_
within_stock): it destroys cross-sectional correlation but preserves each
stock's own marginal distribution exactly, so the null's rho_c level is set
by the marginals' tail behavior, not by anything shuffling removes.

If small-cap quintiles carry heavier-tailed (higher-excess-kurtosis) daily
returns, their sample correlations under independence have larger sampling
variance, which raises the null's rho_c (a standardized-Gaussian upper-tail
threshold scales with the bulk sigma, eq:toy_closed_form). This script
measures excess kurtosis directly, without touching the shuffle or the
percolation scan. Each cell's kurtosis is its mean over the draws of
DRAW_SEEDS, the cells on which the null runs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from fn_percolation.constants import MIN_WINDOW_OBS
from fn_percolation.data_loader import load_panels
from fn_percolation.windows import (
    iter_window_cells, over_draw_seeds, quintile_assigner)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "pca_kurtosis_diagnostic"


def main() -> None:
    print("Loading panels ...")
    returns, marcap = load_panels()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def job(seed_base: int) -> pd.DataFrame:
        rows: list[dict] = []
        for cell in iter_window_cells(returns, assigner=quintile_assigner(marcap),
                                      seed_base=seed_base):
            per_stock = []
            for col in cell.sub.columns:
                y = cell.sub[col].to_numpy(dtype=float)
                y = y[np.isfinite(y)]
                if y.size < MIN_WINDOW_OBS:
                    continue
                per_stock.append(kurtosis(y, fisher=True, bias=False))
            if not per_stock:
                continue
            rows.append({
                "date": cell.end_date, "quintile": cell.quintile,
                "n_stocks": len(per_stock),
                "mean_excess_kurtosis": float(np.mean(per_stock)),
                "median_excess_kurtosis": float(np.median(per_stock)),
            })
        return pd.DataFrame(rows)

    _, df = over_draw_seeds(job, ["date", "quintile"])
    path = OUT_DIR / "excess_kurtosis_by_cell.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df):,} rows)")

    summary = (df.groupby("quintile")[["mean_excess_kurtosis", "median_excess_kurtosis"]]
               .agg(["mean", "std"]))
    summary_path = OUT_DIR / "excess_kurtosis_summary.csv"
    summary.to_csv(summary_path)
    print(f"\nWrote {summary_path}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
