#!/usr/bin/env python3
"""Rolling percolation on Marchenko-Pastur-filtered correlation matrices.

Mirrors the headline pointintime pipeline but inserts an eigenvalue-clipping
RMT filter (Laloux et al. 1999) between the empirical correlation matrix and
the percolation scan. Writes one CSV per quintile to data/rmt_filtered/ in the
same schema as data/korea_headline/ so downstream plotting code can read either source.

Used to test whether the cross-quintile rho_c gradient survives the removal of
noise eigenvalues - i.e. whether the gradient lives in the signal subspace or
arises from the bulk-noise spectrum.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fn_percolation.constants import WINDOW
from fn_percolation.data_loader import load_panels
from fn_percolation.pipeline import (
    collect_quintile_frames,
    progress_hook,
    write_quintile_results,
)
from fn_percolation.rmt import rmt_filter
from fn_percolation.windows import (
    WindowCell,
    n_windows,
    quintile_assigner,
    threshold_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = REPO_ROOT / "data" / "rmt_filtered"


def main() -> None:
    print("Loading panels ...")
    returns, marcap = load_panels()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_total = n_windows(len(returns.index))
    print(f"\nRMT-filtered percolation: {n_total} windows x 5 quintiles ...")

    def rmt_metrics(cell: WindowCell) -> dict | None:
        # Marchenko-Pastur eigenvalue clipping on the unit-diagonal correlation,
        # then zero the diagonal for the percolation scan (Laloux et al. 1999).
        cleaned = rmt_filter(cell.corr_unit_diag(), T=WINDOW)
        np.fill_diagonal(cleaned, 0.0)
        return threshold_metrics(cleaned, cell.sub.shape[1])

    dfs = collect_quintile_frames(
        returns, assigner=quintile_assigner(marcap),
        cell_fn=rmt_metrics, on_cell=progress_hook(n_total),
    )
    # RMT-filtered cells carry no susceptibility (rho_c_chi) column and no
    # un-suffixed Q1 alias, so both extras stay off.
    write_quintile_results(dfs, OUT_DIR, write_q1_alias=False)

    summary = pd.read_csv(OUT_DIR / "rolling_percolation_summary.csv")
    summary_path = OUT_DIR / "rolling_percolation_summary.csv"
    print(f"\nWrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
