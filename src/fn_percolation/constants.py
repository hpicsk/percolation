"""Single source of truth for the rolling-percolation pipeline constants.

Every magic number that controls the rolling-window percolation estimation
lives here, so the headline pipeline, the robustness reruns, and the
mechanism-proxy scripts all anchor to identical values rather than redefining
them. Paper references are subsec:methods / tab:percolation unless noted otherwise.
"""

from __future__ import annotations

import numpy as np

# --- Rolling-window estimation (paper subsec:methods, tab:percolation) ---
WINDOW = 60                              # 60-day rolling estimation window
STEP = 20                                # 20-day step between windows
RHO_RANGE = np.linspace(0.05, 0.9, 200)  # 200-point rho* grid on [0.05, 0.9]
PER_QUINTILE = 300                       # up to 300 stocks per quintile per window
QUINTILES: tuple[int, ...] = (1, 2, 3, 4, 5)  # Q1 smallest .. Q5 largest
N_QUINTILES = 5

# --- Per-(window, quintile) cell admissibility (paper subsec:methods) ---
MIN_TICKERS_FOR_QUINTILE = 50    # min tickers observable on a date to form quintiles
# Observation requirement: column kept iff within-window non-missing rate > 0.8
# (subsec:methods).
# Justified by sensitivity check: scripts/sensitivity/run_pipeline_sensitivity.py
# sweeps the requirement over {0.7, 0.8, 0.9}; strict Q1<...<Q5 holds at every value and
# the Q5-Q1 gap barely moves (subsec:researcher_choices,
# test_min_valid_fraction_not_load_bearing).
MIN_VALID_FRACTION = 0.8
MIN_STOCKS_FOR_PERCOLATION = 10  # min stocks in a cell to run percolation

# --- Within-sector confound controls (paper subsec:size_confounds, tab:robustness_summary) ---
# Sectors with fewer than this many stocks at a window end-date are dropped
# (too few to support a within-sector size ranking).
# Justified by sensitivity check: scripts/confounds/run_within_sector.py
# sweeps the floor over {15, 20, 25}; the ordering the manuscript reports
# (rising through Q1<Q2<Q3, a flat Q3-Q4 step, rising again into Q5) holds at
# every floor and the endpoint Q5-Q1 gap barely moves
# (data/within_sector/sector_size_sensitivity.csv, test_within_sector_
# sector_size_robust).
MIN_SECTOR_SIZE = 20

# --- Reproducibility: the committed draw-seed set (subsec:methods) ---
# Each (window, quintile) cell draws up to PER_QUINTILE tickers with the seed
# ``seed_base + g + start_index``. The draw changes rho_c, so a single seed_base
# gives one draw per cell, and a verdict near the 5% level can move with it.
# Every stage that draws therefore runs under each seed_base of DRAW_SEEDS, and
# a cell's statistic is its mean over the draws. The tests of subsec:inference
# pair the windows on these means.
#
# Committed on 2026-11-03, before the first run under the set, with these
# criteria:
# - The set is SEED_BASE + SEED_STRIDE * k for k = 0 .. N_DRAW_SEEDS - 1. Its
#   first member is the single draw that the paper reported before the set.
#   SEED_STRIDE exceeds every start_index + g, so no two members share a seed in
#   any cell.
# - N_DRAW_SEEDS is sized from six draws run on 2026-10-31 outside this
#   repository (seed_base 42, 1042, ..., 5042). Across those draws, the SD of
#   each Korean step mean was 0.0023 to 0.0041, and the HAC standard error of
#   each step over the six-draw mean was 0.0030 to 0.0088. Twenty draws bring
#   the SD that the draw adds to each step mean to at most one fifth of that
#   standard error (Q3-Q4: 0.0027 / sqrt(20) = 0.0006 against 0.0030).
# - The tests and the 5% rule of subsec:inference are unchanged. The paper
#   states each step and each ordering from the verdicts on the seed means,
#   whichever way they fall, and prints the SD across the draws of each step
#   mean. The seed means replace the single-draw results in every table.
SEED_BASE = 42
N_DRAW_SEEDS = 20
SEED_STRIDE = 100_000
DRAW_SEEDS: tuple[int, ...] = tuple(
    SEED_BASE + SEED_STRIDE * k for k in range(N_DRAW_SEEDS))

# --- Minimum usable daily observations per stock and window ---
# The factor-residual regressions (network_utils.residualize_against_market and
# residualize_against_market_and_sector) and the per-stock kurtosis of
# run_pca_kurtosis_diagnostic.py skip a stock with fewer usable days than this.
MIN_WINDOW_OBS = 20

