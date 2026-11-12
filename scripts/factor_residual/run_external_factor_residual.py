#!/usr/bin/env python3
"""Rolling percolation on external-factor residual returns: the four arms of
tab:residual_steps (paper subsec:main_result), one pass over the same cells per
arm.

Every arm regresses each stock's return, within each rolling window, on one
or two factors built outside the cell (``fn_percolation.network_utils``
residualizers), runs the percolation scan on the residual correlation matrix,
and writes ``rolling_percolation_Q{g}.csv`` plus the summary to
``data/<arm>/`` in the schema of ``data/korea_headline/`` (no susceptibility
column, no un-suffixed Q1 alias). The arms differ only in how the factor is
built; ``ARMS`` is that table, and ``main`` runs it top to bottom.

What the arms measure (paper subsec:main_result, tab:residual_steps): whether
the cross-quintile gradient survives the removal of each stock's exposure to a
market factor, or to a market and a sector factor. The within-sector ranking of
subsec:size_confounds controls sector membership but not a continuous beta or
R^2 gradient within a sector, so a large-cap segment with systematically
higher common-factor exposure than a small-cap segment in the *same* sector
would still pass that check.

Parameter provenance
--------------------
CHOSEN  the factor constructions in ``ARMS``: a market return over the full
        common-stock universe, weighted by the previous day's capitalization
        so a stock's own realized return never determines its own weight, and
        for the two-factor arms each stock's own KRX sector return built the
        same way, with sector membership frozen at each window's end date
        (``sector_loader.sector_as_of``, the convention of
        run_within_sector.py). Each factor set is also run equal-weighted.
        Structural; nothing is chosen to hit a target.
MEASURED  the per-stock, per-window OLS betas and the residual rho_c.
BASELINE: cap-weighted market and sector factors (data/market_residual,
        data/market_sector_residual) | obvious alternative: that
        cap-weighting makes each factor substantially the largest stocks' own
        returns, so residualizing against it cleans the largest segment
        hardest | tuning: none, the construction has no free parameter |
        discharge: run -- the data/market_residual_ew and
        data/market_sector_residual_ew arms below equal-weight both factors,
        and tab:residual_steps reports all four side by side.
CHOSEN  MIN_SECTOR_SIZE (constants.py), the member floor below which a KRX
        sector gets no sector factor, so its stocks drop out of both
        two-factor arms. The table note of tab:residual_steps states the
        floor, and the manuscript attributes no difference between a one- and
        a two-factor row to the sector factor alone, since the rows differ in
        population as well.

Arms
----
market_residual
    One cap-weighted market factor (``value_weighted_market_return``).
market_residual_ew
    The market factor equal-weighted (``equal_weighted_market_return``), so
    no single large-cap name can dominate it.
market_sector_residual
    Market factor plus each stock's own cap-weighted sector return
    (``residualize_against_market_and_sector``). Only windows ending on or
    after the first sector snapshot (2005-01-03) are processed. A stock with
    no mapped sector, or a sub-MIN_SECTOR_SIZE one, is left as an all-NaN
    residual column.
market_sector_residual_ew
    Both factors equal-weighted (``sector_equal_weighted_returns``), so no
    single-name-dominated sector can make that stock's own return the bulk
    of the factor it is then regressed against.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from fn_percolation.data_loader import load_panels
from fn_percolation.network_utils import (
    equal_weighted_market_return,
    residualize_against_market,
    residualize_against_market_and_sector,
    value_weighted_market_return,
)
from fn_percolation.pipeline import (
    collect_quintile_frames,
    progress_hook,
    write_quintile_results,
)
from fn_percolation.sector_loader import (
    FIRST_SNAPSHOT,
    load_sector_panel,
    sector_as_of,
    sector_equal_weighted_returns,
    sector_weighted_returns,
)
from fn_percolation.windows import (
    WindowCell,
    headline_corr,
    n_windows,
    quintile_assigner,
    threshold_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"


@dataclass(frozen=True)
class Factors:
    """Every factor series the arms regress on, built once from the panels."""

    returns: pd.DataFrame
    lagged_marcap: pd.DataFrame       # previous day's capitalization, the weights
    market: pd.Series                 # cap-weighted market return
    market_ew: pd.Series              # equal-weighted market return
    sector_panel: pd.DataFrame        # KRX sector snapshots


def build_factors(returns: pd.DataFrame, marcap: pd.DataFrame) -> Factors:
    print("Building market-return series (cap-weighted, equal-weighted) ...")
    return Factors(
        returns=returns,
        lagged_marcap=marcap.shift(1),
        market=value_weighted_market_return(returns, marcap),
        market_ew=equal_weighted_market_return(returns),
        sector_panel=load_sector_panel(),
    )


# --- Per-arm residualizers: (factors, cell) -> residual returns, or None -----

def _market(f: Factors, cell: WindowCell) -> pd.DataFrame:
    return residualize_against_market(cell.sub, f.market.reindex(cell.sub.index))


def _market_ew(f: Factors, cell: WindowCell) -> pd.DataFrame:
    return residualize_against_market(cell.sub, f.market_ew.reindex(cell.sub.index))


def _sector_inputs(f: Factors, cell: WindowCell):
    """Sector labels at the window's end date and the window's full-universe
    returns and weights the sector factors are built from; None before the
    first sector snapshot."""
    sectors = sector_as_of(cell.end_date, f.sector_panel)
    if sectors.empty:
        return None
    return sectors, f.returns.loc[cell.sub.index], f.lagged_marcap.loc[cell.sub.index]


def _market_sector(f: Factors, cell: WindowCell) -> pd.DataFrame | None:
    inputs = _sector_inputs(f, cell)
    if inputs is None:
        return None
    sectors, returns_window, weights_window = inputs
    sector_returns = sector_weighted_returns(returns_window, weights_window, sectors)
    return residualize_against_market_and_sector(
        cell.sub, f.market.reindex(cell.sub.index), sector_returns, sectors)


def _market_sector_ew(f: Factors, cell: WindowCell) -> pd.DataFrame | None:
    inputs = _sector_inputs(f, cell)
    if inputs is None:
        return None
    sectors, returns_window, _ = inputs
    sector_returns = sector_equal_weighted_returns(returns_window, sectors)
    return residualize_against_market_and_sector(
        cell.sub, f.market_ew.reindex(cell.sub.index), sector_returns, sectors)



@dataclass(frozen=True)
class Arm:
    residual: Callable[[Factors, WindowCell], pd.DataFrame | None]
    sectors: bool = False          # needs the sector panel: skip pre-snapshot windows


ARMS: dict[str, Arm] = {
    "market_residual": Arm(_market),
    "market_residual_ew": Arm(_market_ew),
    "market_sector_residual": Arm(_market_sector, sectors=True),
    "market_sector_residual_ew": Arm(_market_sector_ew, sectors=True),
}


def run_arm(name: str, arm: Arm, factors: Factors, marcap: pd.DataFrame) -> None:
    out_dir = DATA / name
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = n_windows(len(factors.returns.index))
    print(f"\n[{name}] {n_total} windows x 5 quintiles"
          + (f" (only windows ending >= {FIRST_SNAPSHOT.date()})" if arm.sectors else "")
          + " ...")

    def cell_fn(cell: WindowCell) -> dict | None:
        resid = arm.residual(factors, cell)
        if resid is None:
            return None
        corr = headline_corr(resid)
        return threshold_metrics(corr, cell.sub.shape[1])

    dfs = collect_quintile_frames(
        factors.returns, assigner=quintile_assigner(marcap),
        cell_fn=cell_fn, on_cell=progress_hook(n_total),
        skip_before=FIRST_SNAPSHOT if arm.sectors else None,
    )
    write_quintile_results(dfs, out_dir, write_q1_alias=False)
    summary_path = out_dir / "rolling_percolation_summary.csv"
    print(f"Wrote {summary_path}")
    print(pd.read_csv(summary_path).to_string(index=False))


def main() -> None:
    print("Loading panels ...")
    returns, marcap = load_panels()
    factors = build_factors(returns, marcap)
    for name, arm in ARMS.items():
        run_arm(name, arm, factors, marcap)


if __name__ == "__main__":
    main()
