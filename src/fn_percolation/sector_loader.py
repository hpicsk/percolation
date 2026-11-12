"""KRX sector mapping loader and within-sector size quintiles.

The mapping at ``data/external/krx_supplement/sector_mapping.parquet`` is a
daily business-day snapshot panel of (date, ticker, sector_krx) running
from 2005-01-03 onward. ``as_of_snapshot(date, panel)`` selects the most recent
snapshot at or before a date (used by ``sector_as_of``), and
``within_sector_quintiles`` assigns the within-sector size quintiles for the
subsec:size_confounds confound controls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths
from .constants import MIN_SECTOR_SIZE, MIN_TICKERS_FOR_QUINTILE


def load_sector_panel() -> pd.DataFrame:
    """Long-form (date, ticker, sector_krx).

    KRX assigns a small number of conglomerates to multiple sub-sectors on the
    same date (e.g. ticker 007720 carries four sub-sector tags). Keep the first
    listing per (date, ticker) so downstream code can index by ticker with a
    single sector per name.
    """
    df = pd.read_parquet(paths.SECTOR_MAPPING,
                         columns=["date", "ticker", "sector_krx"])
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df = df.drop_duplicates(subset=["date", "ticker"], keep="first")
    return df.sort_values(["date", "ticker"]).reset_index(drop=True)


def as_of_snapshot(date: pd.Timestamp, panel: pd.DataFrame) -> pd.DataFrame:
    """Rows of *panel* at the most recent snapshot date at or before *date*.

    The daily sector-mapping snapshot panel carries a ``date`` column; this
    returns the single-date slice the "as of" lookup uses, or an empty frame if
    no snapshot exists at or before ``date``.
    """
    snap_dates = panel["date"].drop_duplicates().sort_values()
    eligible = snap_dates[snap_dates <= pd.Timestamp(date)]
    if eligible.empty:
        return panel.iloc[0:0]
    return panel.loc[panel["date"] == eligible.iloc[-1]]


def sector_as_of(date: pd.Timestamp, panel: pd.DataFrame) -> pd.Series:
    """Most recent prior snapshot assignment {ticker -> sector_krx}.

    Returns empty Series if no snapshot exists at or before ``date``.
    """
    snap = as_of_snapshot(date, panel)
    if snap.empty:
        return pd.Series(dtype=object)
    return snap.set_index("ticker")["sector_krx"]


def within_sector_quintiles(
    end_date: pd.Timestamp,
    marcap: pd.DataFrame,
    sectors: pd.Series,
    *,
    min_sector_size: int = MIN_SECTOR_SIZE,
) -> pd.Series:
    """Within-sector size quintiles {ticker -> 1..5, 1=smallest in sector}.

    At ``end_date`` take the most recent prior market-cap row (ffill),
    require at least ``MIN_TICKERS_FOR_QUINTILE`` names overall, then within every
    sector carrying at least ``min_sector_size`` stocks rank by market cap and cut
    into size quintiles; sectors below the floor are dropped (subsec:size_confounds, tab:robustness_summary).

    ``min_sector_size`` (default ``MIN_SECTOR_SIZE``) is exposed so the
    sector-size sensitivity check can sweep it.
    """
    pos = marcap.index.get_indexer([end_date], method="ffill")[0]
    if pos < 0:
        return pd.Series(dtype="int64")
    row = marcap.iloc[pos].dropna()
    if len(row) < MIN_TICKERS_FOR_QUINTILE:
        return pd.Series(dtype="int64")
    df = row.to_frame("marcap")
    df["sector"] = sectors.reindex(df.index)
    df = df.dropna(subset=["sector"])
    out: dict[str, int] = {}
    for _, sub in df.groupby("sector"):
        if len(sub) < min_sector_size:
            continue
        q = pd.qcut(sub["marcap"].rank(method="first"), 5,
                    labels=[1, 2, 3, 4, 5]).astype(int)
        out.update(dict(zip(sub.index, q.values)))
    return pd.Series(out, dtype="int64")


def sector_weighted_returns(
    returns_window: pd.DataFrame,
    weights_window: pd.DataFrame,
    sectors: pd.Series,
    *,
    min_sector_size: int = MIN_SECTOR_SIZE,
) -> pd.DataFrame:
    """Value-weighted daily return per sector, over one window's dates.

    Groups the full common-stock universe's returns by ``sectors`` (a single
    {ticker -> sector_krx} snapshot, e.g. from :func:`sector_as_of` at the
    window's end date -- the same "freeze membership at end_date" convention
    :func:`within_sector_quintiles` uses) and value-weights within each sector
    by ``weights_window`` (typically lagged market cap, so a stock's own
    realized return never sets its own weight). Sectors with fewer than
    ``min_sector_size`` members that day are dropped, matching the floor
    :func:`within_sector_quintiles` already applies to cross-sectional sector
    groups.

    Used by the market+sector two-factor residual arms of paper
    tab:residual_steps to build the sector regressor that
    :func:`fn_percolation.network_utils.residualize_against_market_and_sector`
    fits alongside the market return.
    """
    common = returns_window.columns.intersection(sectors.index)
    sec_of = sectors.loc[common].rename("sector").to_frame()
    r = returns_window[common]
    w = weights_window[common].where(r.notna())

    out: dict[str, pd.Series] = {}
    for sector, sub in sec_of.groupby("sector"):
        cols = sub.index
        if len(cols) < min_sector_size:
            continue
        numer = (r[cols] * w[cols]).sum(axis=1, skipna=True)
        denom = w[cols].sum(axis=1, skipna=True)
        out[sector] = numer / denom
    return pd.DataFrame(out, index=returns_window.index)


def sector_equal_weighted_returns(
    returns_window: pd.DataFrame,
    sectors: pd.Series,
    *,
    min_sector_size: int = MIN_SECTOR_SIZE,
) -> pd.DataFrame:
    """Equal-weighted daily return per sector, over one window's dates.

    Robustness alternative to :func:`sector_weighted_returns`: every member
    of a sector contributes ``1/n_sector`` regardless of market cap, so a
    single-name-dominated sector (one stock holding a majority of its
    sector's market cap) cannot make that stock's own return the bulk of the
    factor it is then regressed against. Sectors below ``min_sector_size``
    are dropped, matching :func:`sector_weighted_returns`.

    Cross-check for the market+sector two-factor residual arm of paper
    tab:residual_steps: whether the reversal depends on cap-weighting
    specifically.
    """
    common = returns_window.columns.intersection(sectors.index)
    sec_of = sectors.loc[common].rename("sector").to_frame()
    r = returns_window[common]

    out: dict[str, pd.Series] = {}
    for sector, sub in sec_of.groupby("sector"):
        cols = sub.index
        if len(cols) < min_sector_size:
            continue
        out[sector] = r[cols].mean(axis=1, skipna=True)
    return pd.DataFrame(out, index=returns_window.index)


FIRST_SNAPSHOT = pd.Timestamp("2005-01-03")
