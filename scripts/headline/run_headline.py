#!/usr/bin/env python3
"""Headline point-in-time rolling percolation on both markets.

Korea feeds subsec:main_result, tab:percolation, and fig:rolling_rho_c; the
Taiwan re-run feeds subsec:taiwan and tab:taiwan_rerun. Both markets go
through the one driver, ``pipeline.run_pointintime``, under the locked
constants (WINDOW, STEP, RHO_RANGE, PER_QUINTILE, MIN_VALID_FRACTION,
DRAW_SEEDS), so the cross-market comparison is between two rho_c gradients
produced under byte-identical estimation, each cell averaged over the same
draw seeds.

Parameter provenance
--------------------
CHOSEN  the window length, step, per-quintile sampling cap, and observation
        requirement (WINDOW, STEP, PER_QUINTILE, MIN_VALID_FRACTION in
        fn_percolation.constants). Structural, and each is swept:
        run_pipeline_sensitivity.py over the window and the requirement, the
        network-size grid over N. None is fitted to a target.
CHOSEN  the threshold scan grid (RHO_RANGE) and the spline mesh
        (SPLINE_MESH_SIZE), resolution settings rather than parameters of the
        result.
MEASURED  every rho_c this script writes, by the percolation scan itself.
        Nothing is fitted, so no statistic is spent: the gradient the manuscript
        reports is the whole of the output, not a residual after a fit.
BASELINE: the curvature inflection of G(rho*) as the rho_c estimator |
        obvious alternative: the textbook susceptibility peak chi(rho*) |
        tuning: none, both estimators read the identical scan output |
        discharge: run -- scripts/headline/run_susceptibility.py,
        subsec:researcher_choices, which reports the gradient under both.

Writes, per market, rolling_percolation_Q{1..5}.csv, the un-suffixed Q1
alias, and rolling_percolation_summary.csv to data/korea_headline/ and
data/taiwan/. For Taiwan this script also writes pair_significance.csv, the
adjacent-pair battery (binomial sign test, Wilcoxon signed-rank, Newey-West
HAC, moving-block bootstrap) on both observables; for Korea and every other
rolling arm, verify_claim.py writes that file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from fn_percolation.data_loader import load_panels
from fn_percolation.overlap_inference import hac_block_stats
from fn_percolation.pipeline import align_quintile_panel, run_pointintime
from fn_percolation.tw_data_loader import load_tw_panels

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
MARKETS = {
    "korea": (load_panels, DATA / "korea_headline"),
    "taiwan": (load_tw_panels, DATA / "taiwan"),
}


def _pair_tests(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi in [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]:
        a = panel[f"Q{lo}"].to_numpy()
        b = panel[f"Q{hi}"].to_numpy()
        diff = b - a
        n_pos = int((diff > 0).sum())
        n_neg = int((diff < 0).sum())
        n_eff = n_pos + n_neg
        binom = binomtest(n_pos, n_eff, p=0.5, alternative="greater")
        w = wilcoxon(diff, alternative="greater", zero_method="wilcox")
        hac = hac_block_stats(diff)
        rows.append({
            "pair": f"Q{hi}-Q{lo}",
            "mean_diff": float(np.mean(diff)),
            "n_windows": int(len(diff)),
            "frac_positive": n_pos / n_eff if n_eff else float("nan"),
            "binomial_p_one_sided": float(binom.pvalue),
            "wilcoxon_p_one_sided": float(w.pvalue),
            "nw_lag": hac["nw_lag"],
            "nw_t": hac["nw_t"],
            "nw_p_one_sided": hac["nw_p_one_sided"],
            "block_boot_p_one_sided": hac["block_boot_p_one_sided"],
            "n_eff": hac["n_eff"],
        })
    return pd.DataFrame(rows)


def write_taiwan_significance(results: dict[int, pd.DataFrame], out_dir: Path) -> None:
    """The adjacent-pair significance battery on both observables, persisted
    as pair_significance.csv next to the rolling outputs."""
    print("\n=== Adjacent-pair significance (sign + Wilcoxon + HAC) ===")
    rows_rho = _pair_tests(align_quintile_panel(results, "rho_c")).assign(observable="rho_c")
    rows_chi = _pair_tests(align_quintile_panel(results, "rho_c_chi")).assign(observable="rho_c_chi")
    out = pd.concat([rows_rho, rows_chi], ignore_index=True)
    out_csv = out_dir / "pair_significance.csv"
    out.to_csv(out_csv, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {out_csv}")


def main() -> None:
    for market, (load, out_dir) in MARKETS.items():
        print(f"\nLoading {market} returns + market-cap panels ...")
        returns, marcap = load()
        print(f"  returns: {returns.shape[0]:,} dates x {returns.shape[1]:,} tickers")
        print(f"  marcap:  {marcap.shape[0]:,} dates x {marcap.shape[1]:,} tickers")

        print(f"\n=== {market}: point-in-time rolling percolation ===")
        results = run_pointintime(returns, marcap, out_dir)
        print(f"Wrote {out_dir}")
        if market == "taiwan":
            write_taiwan_significance(results, out_dir)


if __name__ == "__main__":
    main()
