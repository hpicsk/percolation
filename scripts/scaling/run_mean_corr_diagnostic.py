#!/usr/bin/env python3
r"""Diagnostic: is rho_c a monotone relabeling of the mean pairwise correlation?

A size gradient in rho_c could be nothing more than the known fact that
large-cap segments are more correlated. This script extracts, at the *same*
byte-identical sampled (window, quintile) cells as the headline pipeline
(``run_pointintime``), the off-diagonal mean |rho_ij| alongside the stored
curvature-inflection rho_c, and compares the cross-quintile gradient of each.

If the two gradients track each other across all five quintiles, rho_c and the
mean pairwise correlation co-move cross-sectionally.

Reuses ``iter_window_cells`` with the default point-in-time ``quintile_assigner``
under each seed of DRAW_SEEDS, so the draws and column filters match
tab:percolation exactly, and averages each cell over the draws as the headline
does. The per-cell ``mean_abs_corr`` is the off-diagonal mean of ``cell.corr``,
which is the identical zero-diagonal matrix the percolation scan consumes.

Result on the headline panel (subsec:main_result): rho_c and mean |rho| co-move (pooled
Spearman ~0.90) and both rise monotonically Q1->Q5, so the manuscript makes no
claim that rho_c is independent of the segment correlation level. The pooled
Spearman is persisted to ``coupling.csv`` and surfaced as ``\meanCorrSpearman``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fn_percolation.data_loader import load_panels
from fn_percolation.network_utils import off_diagonal_mean_abs
from fn_percolation.windows import (
    iter_window_cells, over_draw_seeds, quintile_assigner)
from fn_percolation.constants import QUINTILES

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "mean_corr_diagnostic"
HEADLINE_DIR = REPO_ROOT / "data" / "korea_headline"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading returns + market-cap panels ...")
    returns, marcap = load_panels()

    def job(seed_base: int) -> pd.DataFrame:
        return pd.DataFrame([{
            "date": cell.end_date,
            "quintile": cell.quintile,
            "n_stocks": cell.sub.shape[1],
            "mean_abs_corr": off_diagonal_mean_abs(cell.corr),
        } for cell in iter_window_cells(returns, assigner=quintile_assigner(marcap),
                                        seed_base=seed_base)])

    _, cells = over_draw_seeds(job, ["date", "quintile"])
    cells.to_csv(OUT_DIR / "per_window_mean_corr.csv", index=False)

    # --- Faithful-sampling check: the seed-mean n_stocks per (date, quintile)
    # must match the stored headline per-window CSVs exactly. If it does, the
    # draws are the headline's and merging rho_c by date is valid. ---
    merged_per_q: dict[int, pd.DataFrame] = {}
    for g in QUINTILES:
        head = pd.read_csv(HEADLINE_DIR / f"rolling_percolation_Q{g}.csv",
                           parse_dates=["date"])
        mine = cells[cells["quintile"] == g].copy()
        mine["date"] = pd.to_datetime(mine["date"])
        m = head.merge(mine, on="date", suffixes=("_head", "_diag"))
        mismatch = int((m["n_stocks_head"] != m["n_stocks_diag"]).sum())
        if mismatch:
            raise SystemExit(
                f"Q{g}: {mismatch} windows where diagnostic n_stocks != headline "
                f"n_stocks -- sampling is NOT byte-identical, aborting.")
        merged_per_q[g] = m

    # --- Cross-quintile summary: cross-window means of rho_c and mean |rho| ---
    summary = []
    for g in QUINTILES:
        m = merged_per_q[g]
        summary.append({
            "quintile": g,
            "n_windows": len(m),
            "rho_c_mean": m["rho_c"].mean(),
            "mean_abs_corr_mean": m["mean_abs_corr"].mean(),
        })
    summ = pd.DataFrame(summary).set_index("quintile")
    summ.to_csv(OUT_DIR / "cross_quintile_summary.csv")

    # --- Within-cell coupling: how tightly does rho_c track mean |rho| across
    # the pooled windows? (Spearman, all cells pooled.) ---
    pooled = pd.concat(merged_per_q.values(), ignore_index=True)
    spearman = pooled[["rho_c", "mean_abs_corr"]].corr(method="spearman").iloc[0, 1]

    # Persist the pooled coupling so generate_params.py can emit
    # \meanCorrSpearman (subsec:main_result co-movement disclosure).
    rho_c_gap = float(summ.loc[5, "rho_c_mean"] - summ.loc[1, "rho_c_mean"])
    mean_corr_gap = float(
        summ.loc[5, "mean_abs_corr_mean"] - summ.loc[1, "mean_abs_corr_mean"])
    # --- What survives conditioning: fit rho_c on mean |rho| across the pooled
    # cells with a flexible (cubic) pooled fit, then read the per-quintile means
    # of the residual. The cubic is a shape allowance rather than a model: a
    # linear fit would attribute curvature in the rho_c-vs-mean relation to the
    # quintile. What subsec:threshold_vs_mean quotes is the residual endpoint
    # gradient as a share of the raw one.
    design = np.vander(pooled["mean_abs_corr"].to_numpy(float), 4)
    beta, *_ = np.linalg.lstsq(design, pooled["rho_c"].to_numpy(float), rcond=None)
    pooled = pooled.assign(resid=pooled["rho_c"].to_numpy(float) - design @ beta)
    resid_means = pooled.groupby("quintile")["resid"].mean().sort_index()
    resid_gap = float(resid_means.iloc[-1] - resid_means.iloc[0])
    pd.DataFrame([{
        "spearman": float(spearman),
        "rho_c_gap_q1q5": rho_c_gap,
        "mean_abs_corr_gap_q1q5": mean_corr_gap,
        "resid_gap_q1q5": resid_gap,
        "resid_gap_share_of_rho_c_gap": resid_gap / rho_c_gap,
        "resid_monotone": bool(all(a < b for a, b in zip(resid_means, resid_means[1:]))),
        "n_cells": int(len(pooled)),
    }]).to_csv(OUT_DIR / "coupling.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n=== Cross-quintile cross-window means ===")
    print(summ.to_string())

    def step(col: str, hi: int, lo: int) -> float:
        return float(summ.loc[hi, col] - summ.loc[lo, col])

    print("\n=== Adjacent-pair steps ===")
    for lo, hi in [(1, 2), (2, 3), (3, 4), (4, 5)]:
        print(f"  Q{lo}->Q{hi}:  d(rho_c) = {step('rho_c_mean', hi, lo):+.4f}"
              f"   d(mean|rho|) = {step('mean_abs_corr_mean', hi, lo):+.4f}")
    print(f"\n  Q1->Q5 endpoint:  d(rho_c) = {step('rho_c_mean', 5, 1):+.4f}"
          f"   d(mean|rho|) = {step('mean_abs_corr_mean', 5, 1):+.4f}")

    print(f"\nPooled Spearman(rho_c, mean|rho|) across all "
          f"{len(pooled)} cells = {spearman:.4f}")
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
