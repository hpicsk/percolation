#!/usr/bin/env python3
"""Re-run the point-in-time rolling percolation analysis on KOSPI-only stocks.

Confound control for the KOSPI vs KOSDAQ venue effect: if the Q1<Q5 gradient
survives when KOSDAQ stocks are removed, the gradient cannot be a pure
venue-mix artefact.
"""

from pathlib import Path

from fn_percolation.data_loader import load_market_flag, load_panels
from fn_percolation.pipeline import run_pointintime

REPO_ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = REPO_ROOT / "data" / "kospi_only"


def main() -> None:
    print("Loading returns + market-cap panels ...")
    returns, marcap = load_panels()
    print(f"  returns: {returns.shape[0]:,} dates x {returns.shape[1]:,} tickers")

    print("Loading market flags ...")
    market = load_market_flag()
    kospi_codes = set(market[market == "KOSPI"].index)
    print(f"  KOSPI common stocks: {len(kospi_codes):,}")

    cols_kospi = [c for c in marcap.columns if c in kospi_codes]
    marcap_kospi = marcap[cols_kospi]
    print(f"  marcap restricted to KOSPI: {marcap_kospi.shape[1]:,} tickers")

    print("\n=== Rolling percolation: KOSPI-only ===")
    run_pointintime(returns, marcap_kospi, OUT_DIR)

    # Persist the KOSPI common-stock universe size for the manuscript macro
    # \nKospiTickers (subsec:size_confounds prose + tab:robustness_summary note:
    # "restricted to KOSPI common stocks (N tickers)"). The market flag is
    # written from the panel's own window, so len(kospi_codes) and the
    # marcap-restricted subset are the same set: the count the prose reports is
    # the universe the arm ran on.
    (OUT_DIR / "universe_size.csv").write_text(
        f"n_kospi_tickers\n{len(kospi_codes)}\n"
    )
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
