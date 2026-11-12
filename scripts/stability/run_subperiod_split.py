#!/usr/bin/env python3
"""Subperiod stability of the endpoint percolation gradient (paper subsec:subperiod).

Figure~\\ref{fig:rolling_rho_c} shows the Q1 and Q5 rolling series drawing
closer in the second half of the sample. This script tests whether the
endpoint Q1--Q5 gradient is concentrated in the first half by splitting the
aligned rolling windows at their median into a first half and a second half
and re-running the headline autocorrelation-robust inference
(``hac_block_stats``) on each half. The split is parameter-free (the median
window), so there is no hand-picked calendar cut-point to defend; the 5-year
calendar periods are reported alongside as a sensitivity that shows the
attenuation is a gradual glide rather than a discrete break at any one date.

A fourth cut reads the full cross-quintile ordering over that same second half.
The endpoint split resolves Q1 against Q5 alone, so the shape of the three
interior means over the half is what this cut adds, and it is the row
tab:robustness_contrast carries.

The endpoint HAC split runs for Korea and Taiwan, each at its own median
window. The two panels are built from different vendors on different spans, so
a compression that appears in one and not the other is a property of that
market rather than of a step shared by the two constructions.

Post-processes the committed rolling outputs under data/korea_headline and
data/taiwan; it does not recompute panels and so carries no upstream data
dependency.
"""

from pathlib import Path

import pandas as pd

from fn_percolation.constants import QUINTILES
from fn_percolation.overlap_inference import hac_block_stats

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
OUT_DIR = DATA / "subperiod"

# Calendar-block width for the cut-point sensitivity (subsec:subperiod).
# Descriptive bins only -- the load-bearing split is the parameter-free median
# window, and these blocks show the attenuation does not turn on at one date.
BLOCK_YEARS = 5

def _year_blocks(years: "list[int]") -> tuple[tuple[int, int], ...]:
    """Successive calendar periods of :data:`BLOCK_YEARS` spanning ``years``.

    Derived from the windows the two markets hold rather than listed, so
    extending the sample re-bins it instead of leaving its last years in no
    block at all. The grid is anchored on the earliest year either market
    covers and is shared by both, because the blocks are read across markets
    (the one Korean block that leaves the endpoint unresolved, against a
    Taiwanese endpoint t that does not fall across the same blocks) and a
    per-market anchor would put two different spans under one label.

    A trailing remainder shorter than a full block joins the block before it. A
    one- or two-year bin would carry an HAC $t$ over a handful of overlapping
    windows, and the manuscript reads these blocks as a sequence of comparable
    ones.
    """
    lo_year, hi_year = min(years), max(years)
    blocks = [(lo, min(lo + BLOCK_YEARS - 1, hi_year))
              for lo in range(lo_year, hi_year + 1, BLOCK_YEARS)]
    if len(blocks) > 1 and blocks[-1][1] - blocks[-1][0] + 1 < BLOCK_YEARS:
        blocks[-2] = (blocks[-2][0], blocks[-1][1])
        blocks.pop()
    return tuple(blocks)


def _endpoint_diff(base: Path) -> pd.Series:
    """Per-window endpoint difference rho_c(Q5) - rho_c(Q1), aligned on the
    common windows, matching the headline endpoint of scripts/headline/verify_claim.py."""
    q1 = pd.read_csv(base / "rolling_percolation_Q1.csv",
                     index_col=0, parse_dates=True)["rho_c"]
    q5 = pd.read_csv(base / "rolling_percolation_Q5.csv",
                     index_col=0, parse_dates=True)["rho_c"]
    aligned = pd.concat({1: q1, 5: q5}, axis=1).dropna()
    return (aligned[5] - aligned[1]).sort_index()


def _seg_row(market: str, segment: str, diff: pd.Series) -> dict:
    stats = hac_block_stats(diff.to_numpy())
    return {
        "market": market,
        "segment": segment,
        "n": int(diff.size),
        "start_date": diff.index.min().date().isoformat(),
        "end_date": diff.index.max().date().isoformat(),
        "mean_diff": stats["mean_diff"],
        "nw_t": stats["nw_t"],
        "nw_p_one_sided": stats["nw_p_one_sided"],
        "block_boot_p_one_sided": stats["block_boot_p_one_sided"],
        "n_eff": stats["n_eff"],
        "frac_positive": float((diff > 0).mean()),
    }


def _endpoint_split(market: str, diff: pd.Series,
                    year_blocks: tuple) -> tuple[list[dict], list[dict]]:
    """Full / first-half / second-half endpoint HAC stats plus the calendar-block
    sensitivity for one market."""
    half = diff.size // 2
    rows = [
        _seg_row(market, "full", diff),
        _seg_row(market, "early", diff.iloc[:half]),
        _seg_row(market, "late", diff.iloc[half:]),
    ]
    blocks = []
    for lo, hi in year_blocks:
        seg = diff[(diff.index.year >= lo) & (diff.index.year <= hi)]
        if seg.size < 3:
            continue
        s = hac_block_stats(seg.to_numpy())
        blocks.append({"market": market, "block": f"{lo}-{hi}", "n": int(seg.size),
                       "mean_diff": s["mean_diff"], "nw_t": s["nw_t"]})
    return rows, blocks


def _second_half_quintile_cut() -> pd.DataFrame:
    """Per-quintile mean rho_c over the second half of the Korean windows, with
    the HAC statistics of the adjacent step into each quintile
    (tab:robustness_contrast, second-half row).

    Split at the same median window the endpoint stats above are split at, so
    the row carries no cut-point of its own to defend. The endpoint split
    resolves Q1 against Q5 only. This cut carries the full cross-quintile
    ordering, which is what the "Ordering" column of tab:robustness_contrast
    reports, and the per-step inference that says which steps survive the
    restriction."""
    panel = pd.concat(
        {g: pd.read_csv(DATA / "korea_headline" / f"rolling_percolation_Q{g}.csv",
                        index_col=0, parse_dates=True)["rho_c"]
         for g in QUINTILES},
        axis=1,
    ).dropna().sort_index()
    post = panel.iloc[len(panel) // 2:]

    rows = []
    for g in QUINTILES:
        row = {
            "quintile": g,
            "mean_rho_c": float(post[g].mean()),
            "n_windows": int(len(post)),
            "start_date": post.index.min().date().isoformat(),
            "end_date": post.index.max().date().isoformat(),
        }
        if g > QUINTILES[0]:
            step = (post[g] - post[g - 1]).to_numpy()
            s = hac_block_stats(step)
            row.update({
                "step_mean_diff": s["mean_diff"],
                "step_nw_t": s["nw_t"],
                "step_nw_p_one_sided": s["nw_p_one_sided"],
                "step_block_boot_p_one_sided": s["block_boot_p_one_sided"],
                "step_frac_positive": float((step > 0).mean()),
            })
        rows.append(row)
    return pd.DataFrame(rows)



def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    diffs = {market: _endpoint_diff(DATA / sub)
             for market, sub in (("korea", "korea_headline"), ("taiwan", "taiwan"))}
    year_blocks = _year_blocks(
        [y for d in diffs.values() for y in (d.index.min().year, d.index.max().year)]
    )
    print(f"  calendar blocks: {', '.join(f'{lo}-{hi}' for lo, hi in year_blocks)}")

    endpoint_rows: list[dict] = []
    block_rows: list[dict] = []
    for market, diff in diffs.items():
        rows, blocks = _endpoint_split(market, diff, year_blocks)
        endpoint_rows.extend(rows)
        block_rows.extend(blocks)

    endpoint = pd.DataFrame(endpoint_rows)
    blocks = pd.DataFrame(block_rows)
    second_half = _second_half_quintile_cut()

    endpoint.to_csv(OUT_DIR / "subperiod_endpoint.csv", index=False)
    blocks.to_csv(OUT_DIR / "subperiod_yearblocks.csv", index=False)
    second_half.to_csv(OUT_DIR / "subperiod_second_half.csv", index=False)

    print("=== Endpoint Q5-Q1 HAC split ===")
    print(endpoint.to_string(index=False))
    print("\n=== Calendar-block sensitivity (HAC t) ===")
    print(blocks.to_string(index=False))
    print("\n=== Second-half cross-quintile cut (Korea) ===")
    print(second_half.to_string(index=False))
    print(f"\nWrote 3 CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()
