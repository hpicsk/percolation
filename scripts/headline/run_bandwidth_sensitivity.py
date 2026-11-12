#!/usr/bin/env python3
"""The tests of subsec:inference at longer bandwidths (subsec:inference).

The automatic Newey-West bandwidth of subsec:inference, floor(4 (n/100)^(2/9)),
is four windows at the headline n, and the bootstrap block is the integer
nearest n^(1/3) windows. Both exceed the lags at which two windows share
returns. A step whose mean drifts over the sample has autocorrelation at lags
longer than that overlap, which the automatic bandwidth leaves out of the
standard error. This script reruns both tests of each headline step and of the
endpoint contrast with the bandwidth and the block length raised to each length of
BANDWIDTHS, in both markets, on the seed-mean rho_c of every window.

Criteria, committed on 2026-11-03 before the first run of this script. On
2026-10-31, the Korean Q2-Q3 step of the single-draw headline was computed at
several bandwidths outside this repository, so the criteria below are not blind
to that step.

- Grid. The automatic bandwidth and block of subsec:inference, and 12 and 24
  windows (about one and two trading years of window ends) for both.
- Report. The Supplementary Material prints the HAC t and the bootstrap p of
  every step and of the endpoint contrast, in both markets, at each bandwidth,
  whichever way they fall.
- Reading. A step that subsec:inference finds significant, but whose HAC t or
  bootstrap p misses the 5% level at 12 or 24 windows, is named in the Results
  with its verdict at each bandwidth. Each sentence of the abstract and of the
  conclusion that calls such a step significant states that its significance
  holds at the automatic bandwidth.

Writes data/bandwidth_sensitivity/steps.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fn_percolation.overlap_inference import hac_block_stats

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
OUT_DIR = DATA / "bandwidth_sensitivity"
MARKETS = {"korea": DATA / "korea_headline", "taiwan": DATA / "taiwan"}
BANDWIDTHS: tuple[int, ...] = (0, 12, 24)     # 0: the automatic bandwidth
PAIRS = ((1, 2), (2, 3), (3, 4), (4, 5), (1, 5))
SIG_LEVEL = 0.05


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for market, data_dir in MARKETS.items():
        panel = pd.concat(
            {g: pd.read_csv(data_dir / f"rolling_percolation_Q{g}.csv",
                            index_col="date", parse_dates=True)["rho_c"]
             for g in (1, 2, 3, 4, 5)}, axis=1).dropna()
        for lo, hi in PAIRS:
            diff = (panel[hi] - panel[lo]).to_numpy()
            for bw in BANDWIDTHS:
                s = hac_block_stats(diff, min_lag=bw, min_block=bw)
                rows.append({
                    "market": market, "pair": f"Q{lo}-Q{hi}",
                    "bandwidth": "automatic" if bw == 0 else str(bw),
                    "nw_lag": s["nw_lag"], "block_len": s["block_len"],
                    "mean_diff": s["mean_diff"], "hac_t": s["nw_t"],
                    "hac_p": s["nw_p_one_sided"],
                    "block_p": s["block_boot_p_one_sided"],
                    "significant": bool(s["nw_p_one_sided"] < SIG_LEVEL
                                        and s["block_boot_p_one_sided"] < SIG_LEVEL),
                    "n_windows": int(len(diff)),
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "steps.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'steps.csv'}")


if __name__ == "__main__":
    main()
