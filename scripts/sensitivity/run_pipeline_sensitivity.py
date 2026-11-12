#!/usr/bin/env python3
"""Sensitivity of the cross-quintile rho_c gradient to the two researcher-chosen
settings of the headline pipeline: the estimation window and the
observation requirement (subsec:researcher_choices, subsec:methods).

Each sweep re-runs the point-in-time pipeline with one setting varied and
every other locked constant held, and records the per-quintile summary, to
verify that the ordering Q1<...<Q5 and the positive Q5-Q1 gap are preserved
across the grid and are not an artifact of the headline value:

  window_sensitivity            W in {30, 60, 120, 252} trading days; the sweep
                                also supplies the horizon-dependent rho_c level
                                band of subsec:limits.
  min_valid_fraction_sensitivity  MIN_VALID_FRACTION in {0.7, 0.8, 0.9}, the
                                observation requirement: a stock's window
                                column is kept only if its within-window
                                non-missing rate exceeds MIN_VALID_FRACTION.

The headline setting is re-computed inside each sweep for byte-identical
comparability rather than re-using data/korea_headline/, since the
SEED_BASE+g+start ticker draws remain deterministic across reruns.

Outputs, per sweep under data/<sweep>/:
  <prefix><value>/rolling_percolation_*.csv   one tree per grid point
  summary.csv                                  long aggregate
  summary_wide.csv                             value x Q1..Q5 + Q5-Q1 gap
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fn_percolation.data_loader import load_panels
from fn_percolation.pipeline import run_pointintime

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"


@dataclass(frozen=True)
class Sweep:
    column: str                 # summary column naming the swept setting
    prefix: str                 # per-value output directory prefix
    values: tuple               # the grid
    kwarg: str                  # run_pointintime keyword the value goes to


SWEEPS = {
    "window_sensitivity": Sweep("window_days", "W", (30, 60, 120, 252), "window"),
    "min_valid_fraction_sensitivity": Sweep(
        "min_valid_fraction", "F", (0.7, 0.8, 0.9), "min_valid_fraction"),
}


def main() -> None:
    print("Loading returns + market-cap panels ...")
    returns, marcap = load_panels()
    print(f"  returns: {returns.shape[0]:,} dates x {returns.shape[1]:,} tickers")

    for name, sweep in SWEEPS.items():
        out_root = DATA / name
        out_root.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []
        for value in sweep.values:
            out_dir = out_root / f"{sweep.prefix}{value}"
            print(f"\n=== {name}: {sweep.column} = {value} ===")
            run_pointintime(returns, marcap, out_dir, **{sweep.kwarg: value})
            summary = pd.read_csv(out_dir / "rolling_percolation_summary.csv")
            for _, row in summary.iterrows():
                rows.append({
                    sweep.column: value,
                    "size_group": int(row["size_group"]),
                    "n_windows": int(row["n_windows"]),
                    "n_stocks": int(row["n_stocks"]),
                    "rho_c_mean": float(row["rho_c_mean"]),
                    "rho_c_std": float(row["rho_c_std"]),
                })

        out = pd.DataFrame(rows).sort_values([sweep.column, "size_group"])
        out.to_csv(out_root / "summary.csv", index=False)
        print(f"\nWrote {out_root / 'summary.csv'} ({len(out)} rows)")

        wide = out.pivot(index=sweep.column, columns="size_group", values="rho_c_mean")
        wide.columns = [f"Q{c}" for c in wide.columns]
        wide["Q5_minus_Q1"] = wide["Q5"] - wide["Q1"]
        wide.to_csv(out_root / "summary_wide.csv")
        print(wide.round(3))


if __name__ == "__main__":
    main()
