"""Survivorship-bias-free returns and market-cap panels for Korea.

Two trees under ``data/external/`` supply the inputs. ``marcap`` is the KRX
daily-snapshot archive, which keeps a code after its delisting and so carries
every stock that was actually listed on each historical date;
:mod:`fn_percolation.kr_universe` reads the point-in-time common-stock universe
off it, so a ticker's series spans only its actual listing window and a code
that was ``common`` only in an earlier era does not leak its later history in.
``fnguide`` is the FnGuide DataGuide 수정주가 export, which supplies the
adjusted close the returns are built from.

Returns come from FnGuide's total-return series ``adj_close_tr``
(수정주가(현금배당포함), item ``S410007700``) rather than from the price-return
``adj_close_pr`` -- capital changes and cash dividends both removed. An
ex-dividend price drop is a distribution to the holder, not a move in what the
firm is worth, and Korean
ex-days cluster on the last sessions of December, so on a price-return series
most of the universe steps down together on one date and the correlation
estimator reads that step as co-movement. The total-return series is also what
puts Korea and Taiwan on one convention: 還原股價 is the only adjusted series
TWSE's vendor sells, so a price-return Korea would have nothing to be compared
against in subsec:taiwan.

BASELINE: FnGuide's total-return adjusted close (adj_close_tr) | obvious
alternative: the price-return series of the same export | tuning: none, a
vendor-published series | discharge: run, the price-return rebuild (strictly
ordered, endpoint gap 0.060 against 0.051), held outside the published tree;
subsec:data_description states the convention
without the two reasons above.

Market cap stays on marcap's raw ``Marcap`` (the actual traded value on that
date); FnGuide's export does not carry it.

``build_panels`` materializes wide ``date x code`` panels (log-returns and
market cap) once and caches them as parquet. ``load_panels`` reads them back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths
from .kr_universe import load_common_panel
from .paths import (  # re-exported: every caller reads these from here
    BUILD_STATS_CSV,
    CACHE_DIR,
    MARCAP_PARQUET,
    MARKET_FLAG_CSV,
    RETURNS_PARQUET,
)

PAPER_START = pd.Timestamp("2005-01-01")
PAPER_END = pd.Timestamp("2025-12-31")


def _fnguide_price_panel() -> pd.DataFrame:
    """Long ``(date, ticker, adj_close_tr, adj_close_pr, segment)`` FnGuide frame.

    Read from the export tree's own long-form table rather than from the 220 MB
    xlsx it was parsed out of. The export is a paid DataGuide subscription this
    repository neither declares as a dependency nor is licensed to
    redistribute, while the returns panel built from it is committed under
    ``data/cache``. Every published table and figure therefore reproduces from
    :func:`load_panels` with the export absent, and an absent export must not
    break that path: the read is deferred to the one caller that rebuilds.

    ``segment`` numbers each code's listing spells. KRX reissues a six-digit
    code once its first occupant is delisted and FnGuide keys a series by code,
    so both occupants arrive in one column; :func:`build_panels` asserts that no
    return differences across a change in it.
    """
    if not paths.FNGUIDE_PRICE.exists():
        raise SystemExit(
            f"Rebuilding the Korean returns panel needs the FnGuide "
            f"adjusted-close table at {paths.FNGUIDE_PRICE}, which is not "
            f"there. That tree carries a FnGuide DataGuide export (a paid "
            f"subscription) and is not part of this repository. The committed "
            f"panels under {CACHE_DIR} reproduce every published result "
            f"without it -- delete them only if you hold a DataGuide "
            f"subscription and intend to rebuild from your own export.")
    return pd.read_parquet(paths.FNGUIDE_PRICE)


def _load_market_long(fg: pd.DataFrame) -> pd.DataFrame:
    """Canonical long-format common-stock panel from the marcap snapshots.

    Single source of truth for the returns / market-cap panels.
    :func:`fn_percolation.kr_universe.load_common_panel` admits a ``(date,
    code)`` cell only while that code trades as a common stock. The security
    type is read off the sessions the FnGuide frame *fg* prices. No-trade days
    are retained (``volume`` may be zero) so each series spans its full listing
    window; :func:`build_panels` blanks the price on them rather than dropping
    the row, which leaves the market-cap panel and the quintile assignment
    untouched.
    """
    return load_common_panel(
        paths.MARCAP_DIR, PAPER_START, PAPER_END,
        priced=fg[["ticker", "date"]].rename(columns={"ticker": "code"}))


def build_panels(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize wide returns + market-cap panels under data/cache/.

    :func:`_load_market_long` defines which ``(date, code)`` cells are admitted
    -- the survivorship-bias-free, point-in-time common-stock universe -- and
    supplies market cap. FnGuide's total-return adjusted close supplies the
    price the log-returns are differenced from (module docstring), joined on
    ``(date, code)``. The universe admits only sessions the export prices, so
    every cell finds its close and the size cut ranks only cells that can be
    sampled (si:settings).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if RETURNS_PARQUET.exists() and MARCAP_PARQUET.exists() and not force:
        return load_panels()

    print("[data_loader] loading FnGuide adjusted closes ...")
    fg = _fnguide_price_panel()

    print("[data_loader] loading common-stock market panel from marcap ...")
    df = _load_market_long(fg)
    print(f"[data_loader] market rows: {len(df):,} "
          f"(dates {df['date'].min().date()} -> {df['date'].max().date()})")

    fg = fg.loc[
        (fg["date"] >= PAPER_START) & (fg["date"] <= PAPER_END),
        ["date", "ticker", "adj_close_tr", "segment"],
    ]
    px = df[["date", "code", "volume"]].merge(
        fg, left_on=["date", "code"], right_on=["date", "ticker"], how="inner"
    )
    if len(px) != len(df):
        raise RuntimeError(
            f"FnGuide prices {len(px):,} of the {len(df):,} universe cells, but "
            f"the universe admits only sessions the export prices; the two "
            f"reads of the export disagree")

    # A 거래정지 / no-trade session carries the last traded price forward, so the
    # return across it is a zero the stock never printed. Blank the price
    # instead, which drops both the return into that session and the one out of
    # it. ``Volume <= 0`` is marcap's own record of a session that did not
    # trade, applied here to the price rather than to the row so the
    # point-in-time universe interval -- and with it the market-cap panel and
    # the quintile assignment -- is unchanged. This is the Korean counterpart
    # of Taiwan's ``Trading_Volume <= 0`` mask
    # (:mod:`fn_percolation.tw_data_loader`); the two markets carry the same
    # kind of session, so measuring one of them and not the other would put a
    # stale-price bias into the cross-market comparison of subsec:taiwan rather
    # than into both sides of it.
    no_trade = px["volume"].fillna(0.0) <= 0.0
    print(f"[data_loader] blanking {int(no_trade.sum()):,} no-trade sessions "
          f"({no_trade.mean():.3%} of admitted cells)")
    px.loc[no_trade, "adj_close_tr"] = np.nan

    adj_wide = px.pivot(index="date", columns="code",
                        values="adj_close_tr").sort_index()
    mc_wide = df.pivot(index="date", columns="code", values="market_cap").sort_index()

    returns = np.log(adj_wide).diff()

    # KRX reissues a 6-digit code once its first occupant is delisted and
    # FnGuide keys a series by code, so both occupants arrive in one column,
    # numbered by the export's ``segment``. Differencing across the handover
    # would compare two companies. The code leaves the marcap snapshots between
    # the two listings, so the point-in-time universe leaves a missing price
    # between them and this raises on no cell of the current export.
    seg_wide = px.pivot(index="date", columns="code",
                        values="segment").sort_index()
    crossed = seg_wide.ffill().diff().fillna(0.0).ne(0.0) & returns.notna()
    if crossed.to_numpy().any():
        raise ValueError(
            f"{int(crossed.to_numpy().sum())} Korean returns difference across a "
            f"code-reuse boundary (FnGuide ``segment`` changes within a column): "
            f"{sorted(crossed.columns[crossed.any()])[:5]}. The point-in-time "
            "universe is supposed to cut the earlier occupant's history."
        )

    print(f"[data_loader] returns panel: {returns.shape[0]:,} dates x "
          f"{returns.shape[1]:,} tickers")
    print(f"[data_loader] writing {RETURNS_PARQUET}")
    returns.to_parquet(RETURNS_PARQUET)
    print(f"[data_loader] writing {MARCAP_PARQUET}")
    mc_wide.to_parquet(MARCAP_PARQUET)
    # The board each code last traded on, for the KOSPI-only arm. Written here
    # because the marcap snapshots are the only place it is recorded, and read
    # back from the cache so that arm reproduces without them
    # (:func:`load_market_flag`).
    (df.sort_values("date").groupby("code")["market"].last()
       .rename("market").reset_index().to_csv(MARKET_FLAG_CSV, index=False))
    print(f"[data_loader] writing {MARKET_FLAG_CSV}")
    _write_build_stats("korea", {
        "codes_in_universe": df["code"].nunique(),
        "no_trade_share": float(no_trade.mean()),
        "no_trade_cells": int(no_trade.sum()),
    })
    return returns, mc_wide


def _write_build_stats(market: str, stats: dict) -> None:
    """Record one market's panel-construction shares under data/cache/.

    tab:settings quotes the no-trade share, which is a property of the build
    rather than of any analysis output, so the builder is the only place it can
    be measured. Written per market and merged on read, so rebuilding one
    market does not erase the other's row.
    """
    rows = {}
    if BUILD_STATS_CSV.exists():
        rows = {r["market"]: r
                for r in pd.read_csv(BUILD_STATS_CSV).to_dict("records")}
    rows[market] = {"market": market, **stats}
    # A statistic no builder writes any more survives the read-back above as
    # an empty column; dropping those keeps the file to what the builders say.
    out = pd.DataFrame([rows[k] for k in sorted(rows)]).dropna(axis=1, how="all")
    out.to_csv(BUILD_STATS_CSV, index=False)
    print(f"[data_loader] wrote {market} build stats to {BUILD_STATS_CSV}")


def load_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously-materialized returns + market-cap panels."""
    if not RETURNS_PARQUET.exists() or not MARCAP_PARQUET.exists():
        raise FileNotFoundError(
            "Returns/marcap panels not built yet. "
            "Run `python scripts/build_panels.py` first."
        )
    return pd.read_parquet(RETURNS_PARQUET), pd.read_parquet(MARCAP_PARQUET)


def load_market_flag() -> pd.Series:
    """Series of {code -> 'KOSPI' | 'KOSDAQ' | ...} for the common-stock universe.

    Written by :func:`build_panels` and committed beside the panels, so the
    KOSPI-only arm of subsec:size_confounds reproduces from the cache with
    the marcap archive absent, as every other published arm does. A code that
    transferred between the boards carries the last market it traded on.
    """
    if not MARKET_FLAG_CSV.exists():
        raise FileNotFoundError(
            "Korean market flag not built yet. "
            "Run `python scripts/build_panels.py` first."
        )
    flag = pd.read_csv(MARKET_FLAG_CSV, dtype={"code": str})
    return flag.set_index("code")["market"]
