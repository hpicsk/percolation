#!/usr/bin/env bash
#
# Reproduce every numerical result and figure in
#   Kang (2026), "Firm Size and the Percolation Threshold of Stock-Return
#   Correlation Networks: Evidence from Korea and Taiwan".
#
# Prerequisites:
#   - Python 3.11+ with `pip install -r requirements.txt`
#   - The fn_percolation package importable: `pip install -e .` from the repo
#     root (editable install; replaces the former sys.path bootstrap).
#
# Step 0 is the ONLY step that reads the vendor trees, and it is the only one
# most users can skip. It rebuilds the cached input panels from the three trees
# under data/external/, none of which is redistributed here:
#   - marcap    the Korean universe, market cap and turnover, over the
#               open-source KRX daily-snapshot archive
#   - fnguide   Korean adjusted closes, a paid FnGuide DataGuide export
#   - finmind   Taiwanese prices and share counts from a FinMind download
#               (free tier, token)
# The panels those trees produce are committed under data/cache/, so every step
# from 1 onward runs without them and reproduces every published table and
# figure. build_panels.py raises naming the missing tree rather than
# substituting if one is absent.
#
# Every step that draws stocks runs once per seed of the committed set
# DRAW_SEEDS (src/fn_percolation/constants.py), one forked worker per seed, so
# a run takes about four hours on a machine with twenty or more cores (steps
# within a numbered group are independent and can be run concurrently). The workers inherit one BLAS thread
# each from the exports below. Each stage is
# idempotent: re-running the script skips stages whose outputs already
# exist (the build_*.py scripts honor cache files; the run_*.py scripts
# overwrite their outputs deterministically). To force a full regeneration,
# delete data/ except data/cache/ first -- the checkpointed scripts return
# early when their output is already present.

set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 0. Cache every input panel, KR + TW (one-time; reads data/external/*).
python scripts/build_panels.py

# 1. Headline point-in-time pipeline, Korea (tab:percolation, fig:rolling_rho_c)
#    and Taiwan (subsec:taiwan, tab:taiwan_rerun), under one driver.
python scripts/headline/run_headline.py

# 2. Confound controls (tab:robustness_summary): the venue control and the
#    within-sector ranking at the published sector-size floor and the two
#    bracketing it (subsec:size_confounds).
python scripts/confounds/run_kospi_only.py
python scripts/confounds/run_within_sector.py

# 3. Null models (tab:robustness_summary).
python scripts/nulls/run_null_shuffled.py
python scripts/nulls/run_rmt_filtered.py

# 4. The giant-component curve re-sampled over the network size N and the
#    rho_c drift it gives (subsec:threshold_vs_mean), the mean-correlation
#    diagnostic (subsec:threshold_vs_mean, si:rho_c_vs_mean) measuring the
#    cross-quintile rho_c-vs-mean|rho| coupling, and the divergence script
#    (subsec:threshold_vs_mean) re-using the same N grid to show rho_c rises
#    with sqrt(2 ln(N-1)) while the segment mean |rho| stays flat.
python scripts/scaling/run_finite_size_scaling.py
python scripts/scaling/run_finite_size_drift.py
python scripts/scaling/run_mean_corr_diagnostic.py
python scripts/scaling/run_mean_corr_vs_N.py

# 5. Window-length and column-admissibility-floor sensitivity
#    (subsec:researcher_choices, subsec:methods; the window sweep also supplies
#    the horizon-dependent rho_c level band of subsec:limits).
python scripts/sensitivity/run_pipeline_sensitivity.py

# 6. Susceptibility-peak robustness (tab:percolation chi-peak row, subsec:researcher_choices).
python scripts/headline/run_susceptibility.py

# 7. Stability of the endpoint gradient (subsec:subperiod, tab:subperiod) and
#    the non-synchronous-trading check (subsec:trading_arrangements): rho_c
#    from weekly returns on long windows (KR + TW).
python scripts/stability/run_subperiod_split.py
python scripts/stability/run_weekly_rho_c.py

# 8. Factor residuals (subsec:main_result, tab:residual_steps): the four
#    external-factor residual arms, and the per-quintile excess kurtosis the
#    shuffled null's asymmetry is read against (subsec:null_robustness).
python scripts/factor_residual/run_external_factor_residual.py
python scripts/factor_residual/run_pca_kurtosis_diagnostic.py

# 9. Data-generating-process validation of the toy model (si:dgp_validation).
python scripts/dgp/run_dgp_validation.py

# 10. Verify the headline claim, which also writes the per-adjacent-pair
#     HAC/bootstrap files (pair_significance.csv) for every arm data/ holds --
#     the arm set is discovered there, not listed, so an arm added above gets
#     its per-pair statistics without anyone remembering to add it here --
#     then figures + PDF.
python scripts/headline/verify_claim.py
python scripts/headline/run_bandwidth_sensitivity.py
pytest test_assertions.py -q
python generate_figures.py
python scripts/generate_params.py
# The two documents cross-reference each other through xr-hyper, so each needs
# the other's .aux and one pass leaves the supplement's references to the paper
# unresolved on a clean checkout. Three alternating rounds reach a fixed point.
(cd paper && for _ in 1 2 3; do
   latexmk -pdf paper.tex
   latexmk -pdf paper_si.tex
 done)

echo
echo "Done. Outputs in data/*, figures/, reports/, paper/paper.pdf."
