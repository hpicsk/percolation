#!/usr/bin/env python3
"""Build every cached input panel once (idempotent; reads data/external/*).

Materializes all wide ``date x code`` parquet panels under ``data/cache/`` that
the analysis pipeline consumes (paper subsec:data_description panel description):

  - Korean returns + market cap (``build_panels``)
  - Taiwan returns + market cap (``build_tw_panels``; subsec:taiwan re-run)

Each builder reuses its existing parquet if present (``force=False``), so
re-running this script is cheap once the caches exist.
"""

from fn_percolation.data_loader import build_panels
from fn_percolation.tw_data_loader import build_tw_panels


def main() -> None:
    returns, marcap = build_panels(force=False)
    print(f"  KR returns: {returns.shape}, market cap: {marcap.shape}")

    tw_returns, tw_marcap = build_tw_panels(force=False)
    print(f"  TW returns: {tw_returns.shape}, market cap: {tw_marcap.shape}")

    print("\nDone. All input panels cached under data/cache/.")


if __name__ == "__main__":
    main()
