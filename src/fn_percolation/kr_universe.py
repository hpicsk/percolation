"""Point-in-time Korean common-stock universe, built from the marcap archive.

si:settings defines the panel's universe as the common stocks of KOSPI
and KOSDAQ, read point in time: a code enters on the session it first
trades as a common stock and leaves on the session it stops, so no later
reclassification reaches back over a stock's history. The marcap archive is a
daily snapshot of every code KRX quoted that session and keeps a code after its
delisting, which is what makes the universe survivorship-free without a
delisting calendar beside it.

A security type is not a column marcap carries, and the listed name does not
settle it: 맥쿼리인프라, an infrastructure investment company, carries no type
word, and 메리츠금융지주 carries 리츠 inside an operating company's name. The
type is read instead off FnGuide's all-codes 수정주가 export, the series the
returns are formed on. The export prices no preferred line, REIT, fund or ETF,
and it prices every other KOSPI and KOSDAQ issue on each session it is listed,
so a session the export does not price is filed as a fund's. On the current
export every such session that the code and market tests leave belongs to a
REIT or a listed investment company (ship, infrastructure, resource, mutual).

The export also prices SPACs, so a SPAC is read off its name: 스팩 or SPAC, or
the shell number before 호 that two shells' listed names keep in place of the
word (케이프이에스제4호). The name test runs after the export test, so a
numbered ship fund, which the export does not price, is filed as a fund. KRX
gives a preferred line a terminal character other than ``0``, and KONEX is a
market. The classification is applied per session, so a shell that becomes an
operating company on its merger enters the universe on the merger rather than
carrying its latest kind back over the shell's own prices.

Degrees of freedom
------------------
MEASURED  the security type, read off the sessions FnGuide's all-codes export
          prices.
CHOSEN    the rules the export leaves to this module: the SPAC name test, the
          terminal-character test for preferred lines (the exchange's own
          numbering), and the KONEX market. Each reads a KRX naming or
          numbering convention rather than a threshold.
MEASURED  every span and count in the output, over the sessions marcap holds.
FITTED    nothing.

BASELINE: FnGuide's all-codes export as the security-type source | obvious
          alternative: a referee says an exchange's own listing register, not a
          vendor's coverage, is what defines a universe | discharge:
          ``test_kr_universe_excludes_preferred_lines``, which asserts on the
          published panel that no preferred line reached it, and the build
          itself, which raises on a universe session the export does not price.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# A SPAC shell's listed name carries 스팩 or SPAC, or only the shell number
# before 호.
_SPAC_RE = re.compile(r"스팩|SPAC|\d호$", re.IGNORECASE)
# KOSDAQ GLOBAL is a disclosure tier of KOSDAQ rather than a separate market,
# so its names are KOSDAQ commons.
_COMMON_MARKETS = ("KOSPI", "KOSDAQ", "KOSDAQ GLOBAL")

_MARCAP_COLS = ["Date", "Code", "Name", "Market", "Volume", "Amount", "Marcap"]


def classify_session(code: str, name: str, market: str, priced: bool) -> str:
    """The security kind of one Korean listing on one session, first match winning.

    1. the KONEX market                                     -> ``konex``
    2. a terminal character other than ``0``                -> ``preferred``
    3. a session FnGuide's all-codes export leaves unpriced -> ``fund``
    4. 스팩 / SPAC, or a shell number before 호               -> ``spac``
    5. KOSPI / KOSDAQ / KOSDAQ GLOBAL                       -> ``common``
    6. anything else                                        -> ``other``

    KRX issues six-character alphanumeric codes as well as the familiar
    six-digit ones (``00088K`` 한화3우B, ``0001A0`` 덕양에너젠). The terminal
    character catches the K/L/M-suffixed preferred lines among them, and an
    alphanumeric common ends in ``0`` and falls through to its market.
    """
    code = str(code).strip()
    name = str(name).strip()
    if market == "KONEX":
        return "konex"
    if code[-1] != "0":
        return "preferred"
    if not priced:
        return "fund"
    if _SPAC_RE.search(name):
        return "spac"
    if market in _COMMON_MARKETS:
        return "common"
    return "other"


def _classify_frame(df: pd.DataFrame) -> pd.Series:
    """``classify_session`` over a frame, evaluated once per distinct state.

    A session-by-session call would classify the same (code, name, market,
    priced) state on every one of its sessions. The distinct states number in
    the thousands against tens of millions of rows, so the classification is
    taken on them and joined back.
    """
    keys = ["Code", "Name", "Market", "priced"]
    states = df[keys].drop_duplicates()
    states["kind"] = [classify_session(c, n, m, p)
                      for c, n, m, p in states.itertuples(index=False)]
    return df.merge(states, on=keys, how="left")["kind"]


def load_common_panel(
    marcap_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    priced: pd.DataFrame,
) -> pd.DataFrame:
    """Long ``(date, code, market, volume, amount, market_cap)`` common-stock panel.

    Every marcap session between ``start`` and ``end`` that the classifier calls
    ``common``. ``priced`` holds the ``(code, date)`` pairs FnGuide's all-codes
    export prices, which the classifier reads the security type off.
    ``volume``, ``amount`` and ``market_cap`` are marcap's own raw
    columns: the traded value on that date, which is the convention the market
    cap the quintiles are cut on has to be on.

    No-trade sessions are retained (``volume`` may be zero), so each series
    spans its full listing window and :func:`build_panels` can blank the price
    on them without moving the quintile assignment.
    """
    frames = []
    for year in range(start.year, end.year + 1):
        f = Path(marcap_dir) / f"marcap-{year}.parquet"
        if not f.exists():
            raise FileNotFoundError(f"marcap year file missing: {f}")
        frames.append(pd.read_parquet(f, columns=_MARCAP_COLS))
    df = pd.concat(frames, ignore_index=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= start) & (df["Date"] <= end)]
    df = df.dropna(subset=["Code", "Date"])
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    # A missing name or market must reach the classifier as a string it can
    # test rather than as a NaN it would raise on; both branches fall through
    # to the market test, which is what an unnamed row should get.
    df["Name"] = df["Name"].where(df["Name"].map(lambda v: isinstance(v, str)), "")
    df["Market"] = df["Market"].where(
        df["Market"].map(lambda v: isinstance(v, str)), "")

    # Outside the export's span every session would read as a fund's, and the
    # universe would end there without saying so.
    first, last = priced["date"].min(), priced["date"].max()
    if df["Date"].min() < first or df["Date"].max() > last:
        raise ValueError(
            f"the marcap sessions run {df['Date'].min().date()} to "
            f"{df['Date'].max().date()}, past the FnGuide export's {first.date()} "
            f"to {last.date()}; the security type cannot be read outside it")
    served = pd.MultiIndex.from_frame(priced[["code", "date"]])
    df["priced"] = pd.MultiIndex.from_arrays([df["Code"], df["Date"]]).isin(served)
    df = df.loc[_classify_frame(df).to_numpy() == "common"]

    return pd.DataFrame({
        "date": df["Date"].to_numpy(),
        "code": df["Code"].to_numpy(),
        "market": df["Market"].to_numpy(),
        "volume": df["Volume"].astype("float64").to_numpy(),
        "amount": df["Amount"].astype("float64").to_numpy(),
        "market_cap": df["Marcap"].astype("float64").to_numpy(),
    }).sort_values(["date", "code"]).reset_index(drop=True)
