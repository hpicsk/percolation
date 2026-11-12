#!/usr/bin/env python3
"""Susceptibility-peak rho_c robustness alongside the curvature-inflection threshold (subsec:researcher_choices).

Step 1. Rerun the headline point-in-time pipeline emitting two thresholds in
every output CSV:
    * ``rho_c``      - the curvature inflection of G(rho*), the headline threshold.
    * ``rho_c_chi``  - the spline-interpolated argmax of the percolation
      susceptibility chi(rho*) = (1/N) sum_{C != C_max} |C|^2, the textbook
      observable.

Step 2. Reproduce the subsec:main_result paired-difference statistical battery under
both definitions on the common-date subsample, writing the comparison table
to ``data/susceptibility/comparison.csv``.

Parameter provenance
--------------------
CHOSEN  nothing beyond the headline settings inherited from run_headline.py;
        both thresholds are read off the same scan.
MEASURED  rho_c and rho_c_chi, by the scan.
BASELINE: this script is the discharge of run_headline.py's estimator choice.
        The two observables agree on the small-cap gradient and disagree on
        where the top of the distribution resolves, and subsec:researcher_choices
        reports both verdicts rather than the one that flatters the headline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from fn_percolation.data_loader import load_panels
from fn_percolation.pipeline import align_quintile_panel, run_pointintime

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data" / "susceptibility"
OUT_CSV = DATA_DIR / "comparison.csv"
QUINTILES = (1, 2, 3, 4, 5)


def _load() -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for g in QUINTILES:
        out[g] = pd.read_csv(
            DATA_DIR / f"rolling_percolation_Q{g}.csv",
            index_col="date",
            parse_dates=["date"],
        )
    return out


def pair_tests(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi in [(1, 2), (2, 3), (3, 4), (4, 5)]:
        a = panel[f"Q{lo}"].to_numpy()
        b = panel[f"Q{hi}"].to_numpy()
        diff = b - a
        n_pos = int((diff > 0).sum())
        n_neg = int((diff < 0).sum())
        n_eff = n_pos + n_neg
        binom = binomtest(n_pos, n_eff, p=0.5, alternative="greater")
        w = wilcoxon(diff, alternative="greater", zero_method="wilcox")
        rows.append({
            "pair": f"Q{hi}-Q{lo}",
            "mean_diff": float(np.mean(diff)),
            "n_windows": int(len(diff)),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "frac_pos": n_pos / n_eff if n_eff else float("nan"),
            "binomial_p_one_sided": float(binom.pvalue),
            "wilcoxon_stat": float(w.statistic),
            "wilcoxon_p_one_sided": float(w.pvalue),
        })
    return pd.DataFrame(rows)


def gap(panel: pd.DataFrame) -> dict:
    diff = panel["Q5"] - panel["Q1"]
    n_pos = int((diff > 0).sum())
    n_eff = int((diff != 0).sum())
    binom = binomtest(n_pos, n_eff, p=0.5, alternative="greater")
    w = wilcoxon(diff.to_numpy(), alternative="greater", zero_method="wilcox")
    return {
        "mean_gap": float(diff.mean()),
        "median_gap": float(diff.median()),
        "n_windows": int(len(diff)),
        "frac_pos": n_pos / n_eff if n_eff else float("nan"),
        "binomial_p_one_sided": float(binom.pvalue),
        "wilcoxon_p_one_sided": float(w.pvalue),
    }


def main() -> None:
    print("Loading returns + market-cap panels ...")
    returns, marcap = load_panels()
    print(f"  returns: {returns.shape[0]:,} dates x {returns.shape[1]:,} tickers")
    print(f"  marcap:  {marcap.shape[0]:,} dates x {marcap.shape[1]:,} tickers")

    print(f"\n=== Susceptibility-peak rerun -> {DATA_DIR} ===")
    run_pointintime(returns, marcap, DATA_DIR)

    print("\n=== Comparing curvature-inflection vs susceptibility-peak ===")
    dfs = _load()
    curv_panel = align_quintile_panel(dfs, "rho_c")
    chi_panel = align_quintile_panel(dfs, "rho_c_chi")

    print()
    print("Per-quintile mean rho_c (common-date subsample per definition)")
    side_by_side = pd.DataFrame({
        "rho_c_curvature": curv_panel.mean(),
        "rho_c_chi_peak":  chi_panel.mean(),
        "delta_chi_minus_curv": chi_panel.mean() - curv_panel.mean(),
    })
    print(side_by_side.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nQ5 - Q1 gap")
    g_curv = gap(curv_panel)
    g_chi = gap(chi_panel)
    gap_table = pd.DataFrame({"curvature_rho_c": g_curv, "chi_peak_rho_c": g_chi})
    print(gap_table.to_string(float_format=lambda x: f"{x:.4g}"))

    print("\nAdjacent-pair paired tests under CURVATURE rho_c")
    pc = pair_tests(curv_panel)
    print(pc.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\nAdjacent-pair paired tests under CHI-PEAK rho_c")
    px = pair_tests(chi_panel)
    print(px.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    pc.insert(0, "definition", "curvature")
    px.insert(0, "definition", "chi_peak")
    pd.concat([pc, px], ignore_index=True).to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
