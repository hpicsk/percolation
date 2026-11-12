"""Returns and market-cap panels for Taiwan, on FinMind's own series.

Everything the Taiwanese panel reads is a FinMind endpoint as the vendor serves
it. Nothing here reconstructs a price, patches an event, or re-derives a
listing: what the endpoints answer is what the panel carries, and
subsec:data_description discloses the delisted names they do not answer for.

Two price series come back for one stock. ``TaiwanStockPrice`` (``ohlcv/``) is
the traded print, and the market capitalization the size quintiles are cut on
is a level rather than a return, so it is formed against that print --
mirroring marcap's raw ``Marcap`` on the Korean side
(:mod:`fn_percolation.data_loader`). ``TaiwanStockPriceAdj`` (``price_adj/``)
is the 還原股價 series, which is total return -- cash dividends come out along
with 無償配股, 現增 and 減資 -- and the log-returns are formed on it alone. It is
the symmetric counterpart of the Korean FnGuide ``adj_close_tr``, which puts
both markets on one convention: an ex-dividend day is a return to the holder in
one market and a step in the price series in the other, and the cross-market
comparison of subsec:taiwan is between two rho_c gradients rather than between
two price conventions. Taiwan forces the choice besides -- 還原股價 is the only
adjusted series the vendor sells at any tier, so there is no price-return
endpoint to mirror a price-return Korea against.

The universe is the vendor's live listing registry, ``TaiwanStockInfo``. It
keeps some of the companies the exchange has delisted and drops the rest. That
is the survivorship exposure subsec:data_description states, and
:func:`delisted_in_window` measures it against the vendor's own
``TaiwanStockDelisting`` record. The vendor's adjusted endpoint is what bounds
the exposure rather than the registry: FinMind serves raw prices for most of
the names the registry drops but 還原股價 for few of them, so a panel formed on
the published adjusted series could carry few of those names, whatever
universe it is drawn from. The Korean panel is survivorship-bias-free because
marcap keeps a delisted code's snapshots; Taiwan has no counterpart.

The price endpoints also serve the sessions a stock traded on the TPEx emerging
board (興櫃), before it listed or between two listings, and those are not TWSE
or TPEx sessions. The shareholding endpoint (``shares/``) carries a stock only
while it is listed, and it usually first carries a listing on the session
before the listing's first. A session therefore enters the panel only when the endpoint
carries the stock on it and, if the stock printed a price on its previous
session, on that session too (si:settings). The same endpoint supplies the
share count the market capitalization is formed on.

The panel spans the same 2005--2025 window Korea does. 還原股價 handles 減資
itself, so the Taiwanese opening date is the first session the vendor prices
rather than a limit inherited from any filing series.

``build_tw_panels`` materializes wide ``date x stock_id`` panels (log-returns
and market cap) once and caches them as parquet. ``load_tw_panels`` reads them
back. Mirrors the public surface of :mod:`fn_percolation.data_loader` so the
percolation pipeline is reused unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from . import paths
from .data_loader import PAPER_END, PAPER_START, _write_build_stats

# Instrument types the registry names that are not common equity. A stock is
# dropped if any of its rows carries one: the type says what an instrument is,
# and none of these is a status a name holds for part of a window and not the
# rest. 創新版股票 / 創新板股票 is the TWSE Innovation Board, a relaxed-disclosure
# tier rather than a main-board listing.
EXCLUDED_TYPES = frozenset({
    "ETF", "ETN", "受益證券", "存託憑證", "臺灣存託憑證",
    "創新版股票", "創新板股票",
})
# The two boards the panel covers, pooled as KOSPI and KOSDAQ are on the Korean
# side. 興櫃 (emerging) is the third the registry carries and is not a listing.
LISTED_MARKETS = ("twse", "tpex")
# A common stock's code is four digits. ETFs are numbered 00xx, warrants carry
# six alphanumeric characters, and preferred lines end in a letter.
COMMON_CODE = re.compile(r"\d{4}")


def _registry() -> pd.DataFrame:
    raw = pd.read_parquet(paths.TWN_STOCK_INFO)
    raw["stock_id"] = raw["stock_id"].astype(str)
    return raw


def load_universe() -> list[str]:
    """The four-digit TWSE and TPEx common stocks the live registry lists.

    The registry serves one row per (market, industry) a stock has been
    classified under rather than one row per stock, and a classification the
    exchange has retired keeps the date it was retired on while one still in
    force carries the date the registry was pulled. So the exclusions drop the
    *stock* on the evidence of any of its rows: dropping rows and deduplicating
    afterwards would keep a name alive on whichever classification the response
    happened to list first, and every Innovation Board name also carries an
    ordinary industry row.

    A name the registry lists here may have spent part of the window on the
    emerging board; :func:`build_tw_panels` keeps only its listed sessions.
    """
    raw = _registry()
    excluded = set(raw.loc[raw["industry_category"].isin(EXCLUDED_TYPES), "stock_id"])
    listed = set(raw.loc[raw["type"].isin(LISTED_MARKETS), "stock_id"])
    return sorted(s for s in (listed - excluded) if COMMON_CODE.fullmatch(s))


def _served(directory: Path, ids: list[str]) -> list[str]:
    """The files of *ids* under *directory* the vendor actually filled.

    The download writes a zero-row file for every stock the endpoint returned
    nothing for, and such a file carries no schema at all, so a column
    projection cannot be pushed down on it: the row count settles from the
    parquet metadata first.
    """
    out = []
    for sid in ids:
        p = directory / f"{sid}.parquet"
        if p.exists() and pq.read_metadata(p).num_rows:
            out.append(str(p))
    return out


def _scan(directory: Path, ids: list[str], columns: list[str],
          start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """One windowed pyarrow scan over the served files of *ids*."""
    df = ds.dataset(_served(directory, ids)).to_table(columns=columns).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    df["stock_id"] = df["stock_id"].astype(str)
    return df.loc[(df["date"] >= start) & (df["date"] <= end)]


def delisted_in_window(start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """The delisted names the live registry no longer lists, and their reach.

    ``TaiwanStockDelisting`` records that a name left a board and on what date.
    A four-digit code it names, absent from the live registry and still printing
    inside the window, is a stock this panel cannot reach: the universe is drawn
    from the registry, so the name is gone from it while its sessions are not
    gone from the tape. Counting them is what puts a magnitude on the
    survivorship exposure subsec:data_description states.

    Reports how many of those names the vendor serves each tree for. The
    ``price_adj`` count is the one that bounds the exposure: a name the adjusted
    endpoint does not answer for cannot enter a returns panel formed on that
    endpoint, however the universe is drawn.
    """
    delisted = pd.read_parquet(paths.TWN_DELISTING)
    delisted["stock_id"] = delisted["stock_id"].astype(str)
    delisted["date"] = pd.to_datetime(delisted["date"])
    exits = delisted.loc[delisted["stock_id"].str.fullmatch(COMMON_CODE)
                         & (delisted["date"] >= start) & (delisted["date"] <= end)]
    missing = sorted(set(exits["stock_id"]) - set(_registry()["stock_id"]))

    px = _scan(paths.FINMIND_OHLCV_DIR, missing, ["date", "stock_id"], start, end)
    reached = sorted(px["stock_id"].unique())
    adj = _scan(paths.FINMIND_PRICE_ADJ_DIR, reached,
                ["date", "stock_id"], start, end)
    return {
        "delisted_absent_from_registry": len(reached),
        "delisted_absent_sessions": int(len(px)),
        "delisted_served_price_adj": int(adj["stock_id"].nunique()),
    }


def build_tw_panels(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize wide TW returns + market-cap panels under data/cache/."""
    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if (paths.TW_RETURNS_PARQUET.exists() and paths.TW_MARCAP_PARQUET.exists()
            and not force):
        return load_tw_panels()

    print(f"[tw_data_loader] window {PAPER_START.date()} -> {PAPER_END.date()}")
    ids = load_universe()
    print(f"[tw_data_loader] common-stock universe size: {len(ids):,}")

    px = _scan(paths.FINMIND_OHLCV_DIR, ids,
               ["date", "stock_id", "close", "Trading_Volume"],
               PAPER_START, PAPER_END)
    adj = _scan(paths.FINMIND_PRICE_ADJ_DIR, ids, ["date", "stock_id", "close"],
                PAPER_START, PAPER_END).rename(columns={"close": "adj_close_tr"})
    sh = _scan(paths.FINMIND_SHARES_DIR, ids,
               ["date", "stock_id", "NumberOfSharesIssued"],
               PAPER_START, PAPER_END)

    # The pyarrow scan keys rows on each file's internal ``stock_id`` column
    # while the universe is the filename. Those coincide for the current
    # FinMind dump, but a refresh with a mislabeled internal id would silently
    # produce a different panel, so anchor to the registry and fail loud.
    unexpected = set(px["stock_id"]) - set(ids)
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} TW tickers have an internal stock_id outside "
            f"the registry universe (e.g. {sorted(unexpected)[:5]}); the "
            f"parquet stock_id column disagrees with its filename.")

    # The price endpoints also serve a stock's emerging-board (興櫃) sessions.
    # The shareholding endpoint carries a stock only while it is listed, and
    # it usually first carries a listing on the session before the listing's
    # first, so a session stays only when the endpoint carries the stock on it
    # and, if the stock printed a price on its previous session, on that
    # session too (si:settings).
    px = px.sort_values(["stock_id", "date"]).reset_index(drop=True)
    carried = pd.Series(
        pd.MultiIndex.from_frame(px[["stock_id", "date"]]).isin(
            pd.MultiIndex.from_frame(sh[["stock_id", "date"]])),
        index=px.index)
    listed = carried & carried.groupby(px["stock_id"]).shift().ne(False)
    print(f"[tw_data_loader] dropping {int((~listed).sum()):,} unlisted sessions "
          f"({(~listed).mean():.3%} of priced sessions) in "
          f"{px.loc[~listed, 'stock_id'].nunique():,} stocks")
    px = px[listed]

    # The two price endpoints do not agree on which sessions exist. The raw
    # series carries a session the stock did not trade as ``close == 0``, and
    # the adjusted series fills the same session with the last traded price, so
    # filtering the adjusted close alone would keep a price nobody could
    # transact at. The join is therefore left onto the raw series, and the
    # no-trade mask below is read off the traded print rather than the filled
    # one. A session the adjusted endpoint does not serve at all arrives with
    # ``adj_close_tr`` missing and leaves the panel a NaN.
    px = px.merge(adj, on=["date", "stock_id"], how="left")

    # A 停牌 / no-trade session carries the last traded price forward, so the
    # return across it is a zero the stock never printed. Blank the price
    # instead, which drops both the return into that session and the one out of
    # it. ``Trading_Volume <= 0`` is the Taiwanese counterpart of the Korean
    # ``volume <= 0`` test (:func:`fn_percolation.data_loader.build_panels`);
    # the two markets carry the same kind of session, so measuring one and not
    # the other would put a stale-price bias into the cross-market comparison
    # of subsec:taiwan rather than into both sides of it.
    no_trade = px["Trading_Volume"].fillna(0.0) <= 0.0
    print(f"[tw_data_loader] blanking {int(no_trade.sum()):,} no-trade sessions "
          f"({no_trade.mean():.3%} of admitted cells)")
    px.loc[no_trade, "adj_close_tr"] = np.nan

    n_adj_missing = int(px["adj_close_tr"].isna().sum() - no_trade.sum())
    print(f"[tw_data_loader] {n_adj_missing:,} traded sessions carry no "
          f"adjusted close ({n_adj_missing / len(px):.3%})")

    # Market capitalization is formed against the last *traded* close. The raw
    # endpoint prints a no-trade session as ``close == 0``, and a market cap
    # formed against that zero ranks the stock smallest and puts it in Q1,
    # which is a size the stock does not have: it is the one that did not
    # trade, not the one worth nothing. Carrying the last traded close forward
    # within the stock is also what the Korean side already does, because KRX
    # publishes a 거래정지 session's Marcap at the last traded price, so the two
    # markets cut their quintiles on one convention. A stock with no traded
    # close yet takes NaN and receives no quintile that day.
    raw = px.merge(sh, on=["date", "stock_id"], how="inner"
                   ).sort_values(["stock_id", "date"])
    traded_close = raw["close"].astype("float64").where(raw["close"] > 0)
    raw["marcap"] = (traded_close.groupby(raw["stock_id"]).ffill()
                     * raw["NumberOfSharesIssued"].astype("float64"))
    # The shareholding endpoint serves a zero share count on a handful of
    # sessions, which produces the same zero market cap the no-trade close
    # produced and would rank the stock smallest for the same wrong reason. A
    # non-positive market capitalization is not a size, so it leaves the panel
    # as missing rather than as a number the quintile cut would rank.
    n_nonpositive = int((raw["marcap"] <= 0).sum())
    raw["marcap"] = raw["marcap"].where(raw["marcap"] > 0)
    mc_wide = raw.pivot(index="date", columns="stock_id",
                        values="marcap").sort_index()

    adj_wide = px.pivot(index="date", columns="stock_id",
                        values="adj_close_tr").sort_index()
    # Guard the log against non-positive closes: a zero price would make
    # ``log(0) = -inf`` and, since +/-inf is not NaN, slip past the >80%
    # non-missing window filter and isolate that stock as a degree-0 node.
    # Mapping non-positive prices to NaN instead drops only those days, so the
    # stock's valid returns enter the correlation pairwise -- identical to the
    # Korean path, whose adjusted closes are strictly positive.
    returns = np.log(adj_wide.where(adj_wide > 0)).diff()

    print(f"[tw_data_loader] returns panel: {returns.shape[0]:,} dates x "
          f"{returns.shape[1]:,} tickers")
    print(f"[tw_data_loader] writing {paths.TW_RETURNS_PARQUET}")
    returns.to_parquet(paths.TW_RETURNS_PARQUET)
    print(f"[tw_data_loader] writing {paths.TW_MARCAP_PARQUET}")
    mc_wide.to_parquet(paths.TW_MARCAP_PARQUET)
    _write_build_stats("taiwan", {
        "vendor_cell_coverage": 1.0 - n_adj_missing / len(px),
        "codes_in_universe": len(ids),
        "codes_absent_from_vendor": len(ids) - int(px["stock_id"].nunique()),
        "no_trade_share": float(no_trade.mean()),
        "no_trade_cells": int(no_trade.sum()),
        # The denominator the survivorship exposure is a share of: the traded,
        # adjusted stock-days the panel actually carries.
        "adj_sessions": int(px["adj_close_tr"].notna().sum()),
        "nonpositive_marcap_cells": n_nonpositive,
        **delisted_in_window(PAPER_START, PAPER_END),
    })
    return returns, mc_wide


def load_tw_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously-materialized TW returns + market-cap panels."""
    if not paths.TW_RETURNS_PARQUET.exists() or not paths.TW_MARCAP_PARQUET.exists():
        raise FileNotFoundError(
            "Taiwan returns/marcap panels not built yet. "
            "Run `python scripts/build_panels.py` first."
        )
    return (
        pd.read_parquet(paths.TW_RETURNS_PARQUET),
        pd.read_parquet(paths.TW_MARCAP_PARQUET),
    )
