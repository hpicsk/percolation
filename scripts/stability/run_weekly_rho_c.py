#!/usr/bin/env python3
"""rho_c from weekly returns on long windows (subsec:trading_arrangements).

Stale prices lower the correlations measured from daily returns, more for
thinly traded stocks than for others (Epps 1979). A 60-day window holds about
thirteen weekly returns, too few to locate rho_c. This script lengthens the
window to each length in WEEKLY_WINDOWS, so that a window holds enough weekly
returns to locate rho_c, and estimates rho_c from the daily and from the weekly
returns of the same cells. At one window length, the two estimates differ only
in the sampling frequency.

The cells are those of iter_window_cells at window length T and the headline
step, under each seed of DRAW_SEEDS. The observation requirement applies to the
daily returns, and a week's return is the sum of its daily log returns (W-FRI,
min_count=1). A cell's rho_c at each frequency is its mean over the draws.

Criteria, committed on 2026-11-03 before the first run of this script. A
single-draw version of this check (window lengths 250 and 500, step 60) ran on
2026-10-31 outside this repository, so the criteria below are not blind to its
results in either market.

- Inference. For each market, window length T, frequency, and step (Q1-Q2 to
  Q4-Q5, and the endpoint Q1-Q5), the test of subsec:inference runs on the
  per-window differences of the seed-mean rho_c. Windows that lie up to
  overlap_span(T, STEP) steps apart share returns, so the HAC bandwidth and the
  bootstrap block are raised to overlap_span + 1 windows. A step is significant
  when both statistics reject at the 5% level, one-sided.
- Resolution. The weekly test resolves a step at T only if 2.49 times the HAC
  standard error of the weekly step is at most the daily step at T, where
  2.49 = 1.645 + 0.842 sets a one-sided 5% test at 80% power.
- Reading. A step that is not significant in daily returns at T returns no
  verdict on stale prices at T. For a step that is significant in daily returns
  at T: a weekly step that is significant means that weekly returns reproduce
  the step; a weekly step that is resolved and not significant means that
  weekly returns do not reproduce the step, so stale prices could produce it;
  an unresolved weekly step returns no verdict.
- Report. The Supplementary Material prints, for both markets and both window
  lengths, each step's daily and weekly mean with its HAC t and bootstrap p,
  whichever way they fall. The Results state, for each step that the paper
  calls significant at 60 days, the reading at both window lengths, and report
  both readings where they differ. The abstract and the conclusion state, for
  each market, whether weekly returns reproduce the rise that they report.

Degrees of freedom
------------------
CHOSEN    WEEKLY_WINDOWS: one and two trading years, the shortest lengths that
          hold about 50 and 100 weekly returns; 252 days is also the longest
          length of the window sweep of run_pipeline_sensitivity.py.
CHOSEN    WEEK_RULE: a weekly return sums the daily returns of a week that
          ends on Friday.
MEASURED  every rho_c, by the headline scan. Nothing is fitted.

Writes data/weekly_rho_c/cells_{market}_T{T}.csv (seed-mean rho_c per window
and quintile at both frequencies), per_seed_{market}_T{T}.csv (the same per
draw), and steps.csv (the tests, the resolution, and the reading of every
step).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fn_percolation.constants import QUINTILES, STEP
from fn_percolation.data_loader import load_panels
from fn_percolation.overlap_inference import (
    hac_block_stats, hac_mean_se, overlap_span)
from fn_percolation.tw_data_loader import load_tw_panels
from fn_percolation.windows import (
    headline_corr, iter_window_cells, over_draw_seeds, quintile_assigner,
    threshold_metrics)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "weekly_rho_c"

WEEKLY_WINDOWS: tuple[int, ...] = (252, 504)
WEEK_RULE = "W-FRI"
SIG_LEVEL = 0.05
Z_ALPHA_ONE_SIDED = 1.645          # one-sided 5% test
Z_POWER = 0.842                    # 80% power
PAIRS = ((1, 2), (2, 3), (3, 4), (4, 5), (1, 5))
MARKETS = {"korea": load_panels, "taiwan": load_tw_panels}


def _rho_c(frame: pd.DataFrame) -> float:
    m = threshold_metrics(headline_corr(frame), frame.shape[1])
    return float("nan") if m is None else m["rho_c"]


def cells_at(returns: pd.DataFrame, marcap: pd.DataFrame, window: int
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-draw and seed-mean daily and weekly rho_c of every cell at *window*."""
    def job(seed_base: int) -> pd.DataFrame:
        rows: list[dict] = []
        for cell in iter_window_cells(returns, assigner=quintile_assigner(marcap),
                                      window=window, seed_base=seed_base):
            weekly = cell.sub.resample(WEEK_RULE).sum(min_count=1)
            rows.append({
                "date": cell.end_date, "quintile": cell.quintile,
                "n_stocks": cell.sub.shape[1], "n_weeks": weekly.shape[0],
                "daily_rho_c": _rho_c(cell.sub),
                "weekly_rho_c": _rho_c(weekly),
            })
        print(f"  T={window} seed_base {seed_base}: {len(rows)} cells", flush=True)
        return pd.DataFrame(rows)

    return over_draw_seeds(job, ["date", "quintile"])


def step_tests(cells: pd.DataFrame, window: int) -> pd.DataFrame:
    """The tests, the resolution and the reading of every step at *window*."""
    span = overlap_span(window, STEP) + 1
    out: dict[tuple[int, int], dict] = {}
    for freq in ("daily", "weekly"):
        panel = cells.pivot(index="date", columns="quintile",
                            values=f"{freq}_rho_c").dropna()
        for lo, hi in PAIRS:
            diff = (panel[hi] - panel[lo]).to_numpy()
            s = hac_block_stats(diff, min_lag=span, min_block=span)
            row = out.setdefault((lo, hi), {"pair": f"Q{lo}-Q{hi}"})
            row.update({
                f"{freq}_mean": s["mean_diff"], f"{freq}_hac_t": s["nw_t"],
                f"{freq}_hac_se": hac_mean_se(diff, min_lag=span)["se"],
                f"{freq}_block_p": s["block_boot_p_one_sided"],
                f"{freq}_significant": bool(
                    s["nw_p_one_sided"] < SIG_LEVEL
                    and s["block_boot_p_one_sided"] < SIG_LEVEL),
                "nw_lag": s["nw_lag"], "block_len": s["block_len"],
                "n_windows": int(len(panel)),
            })
    rows = []
    for row in out.values():
        detection = (Z_ALPHA_ONE_SIDED + Z_POWER) * row["weekly_hac_se"]
        row["weekly_detection_limit"] = detection
        row["weekly_resolved"] = bool(detection <= row["daily_mean"])
        if not row["daily_significant"]:
            row["reading"] = "no verdict: daily step not significant"
        elif row["weekly_significant"]:
            row["reading"] = "weekly returns reproduce the step"
        elif row["weekly_resolved"]:
            row["reading"] = "weekly returns do not reproduce the step"
        else:
            row["reading"] = "no verdict: weekly test does not resolve the step"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = []
    for market, loader in MARKETS.items():
        print(f"\n=== {market} ===", flush=True)
        returns, marcap = loader()
        for window in WEEKLY_WINDOWS:
            per_seed, cells = cells_at(returns, marcap, window)
            per_seed.sort_values(["seed", "quintile", "date"]).to_csv(
                OUT_DIR / f"per_seed_{market}_T{window}.csv", index=False,
                float_format="%.6g")
            cells.to_csv(OUT_DIR / f"cells_{market}_T{window}.csv", index=False)
            steps = step_tests(cells, window)
            steps.insert(0, "window", window)
            steps.insert(0, "market", market)
            tables.append(steps)
            means = cells.groupby("quintile")[["daily_rho_c", "weekly_rho_c"]].mean()
            print(means.reindex(list(QUINTILES)).round(4).to_string())
    steps = pd.concat(tables, ignore_index=True)
    steps.to_csv(OUT_DIR / "steps.csv", index=False)
    pd.set_option("display.width", 200)
    print(steps[["market", "window", "pair", "daily_mean", "daily_hac_t",
                 "weekly_mean", "weekly_hac_t", "weekly_block_p",
                 "weekly_resolved", "reading"]].round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
