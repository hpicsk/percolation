#!/usr/bin/env python3
"""Within-sector size-quintile rolling percolation, at the
published sector-size floor and at the two floors bracketing it.

Confound control for sector mix (subsec:size_confounds,
tab:robustness_summary): rather than assigning quintiles by raw market cap
(which mechanically over-represents heavy-cap sectors in Q5 and light-cap
sectors in Q1), each stock is ranked by its market cap *within its own
sector* at the window's end date and assigned a size quintile from the
within-sector rank. Pooling across sectors yields a cross-section where every
quintile has roughly proportional sector mix. KRX sector mapping is daily
business-day snapshots from 2005-01-03 onward, so the within-sector analysis
runs over the same rolling-window grid as the headline pipeline, skipping the
windows that end before the first snapshot.

Sectors with fewer than ``MIN_SECTOR_SIZE`` stocks at a window's end date are
dropped before the ranking: too low and a sector with a handful of names
yields a degenerate size ranking, too high and mid-size sectors are
discarded. That floor is a researcher-chosen value, so the scan is repeated
at every floor in ``FLOORS`` and the per-quintile mean rho_c recorded, which
is what shows the published Q1<...<Q5 ordering robust to the choice.

Writes the per-quintile rolling_percolation_Q{g}.csv and summary for the
published floor to data/within_sector/ (rho_c only, no susceptibility column,
no Q1 alias), and one row per (min_sector_size, size_group) for every floor
to data/within_sector/sector_size_sensitivity.csv.
"""

from pathlib import Path

import pandas as pd

from fn_percolation.constants import MIN_SECTOR_SIZE, QUINTILES
from fn_percolation.data_loader import load_panels
from fn_percolation.pipeline import (
    collect_quintile_frames,
    progress_hook,
    write_quintile_results,
)
from fn_percolation.sector_loader import (
    FIRST_SNAPSHOT,
    load_sector_panel,
    sector_as_of,
    within_sector_quintiles,
)
from fn_percolation.windows import n_windows

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "within_sector"

# The published floor and the two bracketing it by +/- 5 stocks.
FLOORS = (15, MIN_SECTOR_SIZE, 25)


def main() -> None:
    print("Loading panels ...")
    returns, marcap = load_panels()
    sector_panel = load_sector_panel()
    print(f"  returns: {returns.shape}")
    print(f"  sector panel: {sector_panel['date'].min().date()} -> "
          f"{sector_panel['date'].max().date()} "
          f"({sector_panel['date'].nunique()} snapshots, "
          f"{sector_panel['ticker'].nunique()} tickers)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_total = n_windows(len(returns.index))

    rows: list[dict] = []
    for floor in FLOORS:
        print(f"\nSector-size floor = {floor}: scanning {n_total} windows "
              f"(only those ending >= {FIRST_SNAPSHOT.date()} are processed) ...")

        def assigner(end_date: pd.Timestamp, floor: int = floor) -> pd.Series:
            """Within-sector size quintiles; empty until a sector snapshot exists."""
            sectors = sector_as_of(end_date, sector_panel)
            if sectors.empty:
                return pd.Series(dtype="int64")
            return within_sector_quintiles(end_date, marcap, sectors,
                                           min_sector_size=floor)

        dfs = collect_quintile_frames(
            returns, assigner=assigner, on_cell=progress_hook(n_total),
            skip_before=FIRST_SNAPSHOT,
        )
        if floor == MIN_SECTOR_SIZE:
            write_quintile_results(dfs, OUT_DIR, write_q1_alias=False, with_chi=False)
            print(f"Wrote {OUT_DIR}")
            print(pd.read_csv(OUT_DIR / "rolling_percolation_summary.csv").to_string(index=False))

        means = []
        for g in QUINTILES:
            series = dfs[g]["rho_c"] if not dfs[g].empty else pd.Series(dtype=float)
            mean = float(series.mean()) if not series.empty else float("nan")
            means.append(mean)
            rows.append({
                "min_sector_size": floor,
                "size_group": g,
                "rho_c_mean": mean,
                "n_windows": int(len(series)),
            })
        gap = means[4] - means[0]
        monotone = all(a < b for a, b in zip(means, means[1:]))
        print(f"  Q1..Q5 mean rho_c: {[round(m, 3) for m in means]}")
        print(f"  Q5-Q1 gap: {gap:.3f}  | strictly monotone Q1<..<Q5: {monotone}")

    out_path = OUT_DIR / "sector_size_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
