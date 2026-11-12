"""Every path the pipeline reads, resolved in one place.

Two kinds of thing sit under ``data/external/``. The registry tables and the
KRX sector mapping are small and are committed, so a clone carries them. The
three data trees are large and are not: each is a directory the replicator
supplies, by cloning the public archive in the case of marcap and by placing
(or symlinking) the vendor download in the case of the other two.
``FNP_DATA_ROOT`` relocates the whole root for a replicator who keeps the trees
elsewhere.

Expected layout under ``data/external/``:

    marcap/data/        the marcap daily-snapshot archive's
                        ``marcap-YYYY.parquet`` files, read for the quantities
                        KRX publishes and the closing print they are formed at
                        (volume, traded value, market cap) and for the
                        point-in-time common-stock universe
                        :mod:`fn_percolation.kr_universe` builds from the same
                        snapshots
    fnguide/            the FnGuide DataGuide 수정주가 export tree, read for the
                        long-form adjusted-close table at
                        ``cache/fnguide_price.parquet`` the Korean returns are
                        formed on
    finmind/            one directory per FinMind Taiwan dataset, each holding
                        one parquet per stock: ``ohlcv/`` (TaiwanStockPrice,
                        the traded print), ``price_adj/`` (TaiwanStockPriceAdj,
                        the 還原股價 series) and ``shares/``
                        (TaiwanStockShareholding)
    registry/           the committed tables: FinMind's Taiwanese listing and
                        delisting registries
    krx_supplement/     the KRX sector mapping the size-confound controls read

Paths under ``data/external/`` resolve lazily on attribute access, so a checker
importing this module for one pure function does not need the trees. A tree
that is not there raises ``RuntimeError`` naming it and what supplies it.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache"

# Cached panels the analysis reads. Committed, so every published table and
# figure reproduces with the external trees absent.
RETURNS_PARQUET = CACHE_DIR / "returns_panel.parquet"
MARCAP_PARQUET = CACHE_DIR / "marketcap_panel.parquet"
TW_RETURNS_PARQUET = CACHE_DIR / "tw_returns_panel.parquet"
TW_MARCAP_PARQUET = CACHE_DIR / "tw_marketcap_panel.parquet"
BUILD_STATS_CSV = CACHE_DIR / "panel_build_stats.csv"
MARKET_FLAG_CSV = CACHE_DIR / "kr_market_flag.csv"

_DATA_ROOT_DEFAULT = REPO_ROOT / "data" / "external"

# Logical name -> subpath under the data root.
_EXTERNAL_ROOTS = {
    "MARCAP_DIR": "marcap/data",
    "FNGUIDE_DIR": "fnguide",
    "FINMIND_DIR": "finmind",
    "REGISTRY_DIR": "registry",
    "KRX_SUPPLEMENT_DIR": "krx_supplement",
}

# What supplies each root, quoted back in the error a missing one raises.
_ROOT_SOURCE = {
    "MARCAP_DIR": "clone https://github.com/FinanceData/marcap into "
                  "data/external/marcap",
    "FNGUIDE_DIR": "place the FnGuide DataGuide 수정주가 export at "
                   "data/external/fnguide (subscription required; see README, "
                   "Data requirements)",
    "FINMIND_DIR": "download the three FinMind Taiwan datasets into "
                   "data/external/finmind (README, Data requirements)",
    "REGISTRY_DIR": "committed with the repository; restore it from git",
    "KRX_SUPPLEMENT_DIR": "committed with the repository; restore it from git",
}

_DERIVED = {
    # FinMind (Taiwan): one directory per dataset, one parquet per stock.
    "FINMIND_OHLCV_DIR": ("FINMIND_DIR", "ohlcv"),
    "FINMIND_PRICE_ADJ_DIR": ("FINMIND_DIR", "price_adj"),
    "FINMIND_SHARES_DIR": ("FINMIND_DIR", "shares"),
    # Committed registries.
    "TWN_STOCK_INFO": ("REGISTRY_DIR", "twn_stock_info.parquet"),
    "TWN_DELISTING": ("REGISTRY_DIR", "twn_delisting.parquet"),
    "SECTOR_MAPPING": ("KRX_SUPPLEMENT_DIR", "sector_mapping.parquet"),
    # The 수정주가 export parsed once into long form, as the export tree ships
    # it: ``date, ticker, adj_close_pr, adj_close_tr, market, segment``.
    "FNGUIDE_PRICE": ("FNGUIDE_DIR", "cache/fnguide_price.parquet"),
}

_resolved: dict[str, Path] = {}


def _data_root() -> Path:
    root = Path(os.environ.get("FNP_DATA_ROOT", _DATA_ROOT_DEFAULT)).expanduser()
    if not root.is_dir():
        raise RuntimeError(
            f"Data root {root} does not exist. The repository carries it as "
            f"data/external/; set FNP_DATA_ROOT to your own root if the trees "
            f"live elsewhere. See paths.py docstring for the expected layout.")
    return root


def _resolve_root(name: str) -> Path:
    p = _data_root() / _EXTERNAL_ROOTS[name]
    if not p.is_dir():
        raise RuntimeError(
            f"{p} does not exist. To supply it: {_ROOT_SOURCE[name]}.")
    return p


def __getattr__(name: str) -> Path:
    if name in _resolved:
        return _resolved[name]
    if name in _EXTERNAL_ROOTS:
        p = _resolve_root(name)
        _resolved[name] = p
        return p
    if name in _DERIVED:
        parent_name, suffix = _DERIVED[name]
        p = __getattr__(parent_name) / suffix
        _resolved[name] = p
        return p
    raise AttributeError(f"module 'paths' has no attribute {name!r}")
