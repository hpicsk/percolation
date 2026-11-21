"""Manuscript-claim assertions for fn_percolation.

Every quantitative number printed in paper/paper.tex is asserted here
against the regenerated CSV/Parquet outputs under data/. Error messages
quote the manuscript sentence being violated. Absolute tolerance is 0.005 for rho_c-level values
(three-decimal paper rounding) unless a tighter tolerance is justified.

Steps of the arguments of subsec:toy_model, subsec:network_size and
si:variance_derivative that no computation decides, each with its reason. The
tests at the end of this file decide the other steps.

- subsec:toy_model and si:variance_derivative: "sum_k W_ik^2 <= 1", the
  non-negativity of the factors (W_ij + W_ji) sigma_u^2 and
  sum_k W_ik W_jk sigma_u^2 that multiply the coefficients of delta, "delta >=
  1-delta >= (1-delta) sum_k W_ik^2", "If all loadings beta_i share a sign,
  every covariance is non-negative", the rise with delta of each divided
  variance as a sum of the two rising functions with the coefficients
  beta_i^2 sigma_f^2 and sigma_u^2, and the fall with delta of the absolute
  correlation of every pair on [1/2, 1). Each is an inequality at every point of
  a set of two or more real variables. Only a decision procedure for polynomial
  inequalities, such as quantifier elimination, decides it, and the paper
  proves it by hand, not with such a procedure.
- subsec:toy_model: "F_delta_1 then first-order stochastically dominates
  F_delta_2", "first-order stochastic dominance on [1/2,1) gives
  tilde-rho_c(delta_2) <= tilde-rho_c(delta_1)", and its strictness where the
  pairs near the quantile are coupled only through shared peers. Each ranges
  over every distribution of the pair correlations, which no finite
  computation enumerates.
- subsec:toy_model: "Below delta = 1/2, the sign of the response of
  tilde-rho_c to delta depends on W". eq:toy_threshold has a root only where
  F_delta is continuous, whereas F_delta over the pairs of a finite W is a step
  function, so a pair of example matrices would decide the claim only under a
  convention for the quantile that the paper does not state.
- subsec:toy_model: "Because delta falls with firm size ... the model predicts
  that tilde-rho_c rises with firm size". The premise is the empirical claim of
  subsec:dissipation_framework, which the cited studies support and no
  computation of this repository measures.
- subsec:network_size: "The leading-order solution of phi(z)/z = 1/(N-1) is
  then z approx sqrt(2 ln(N-1))", the Mills-ratio asymptote, and the growth of
  the largest sample correlation as sqrt(4 ln N / T). Each is a limit as N
  grows, which no computation at finitely many N decides; the second is the
  cited theorem of jiang2004largest.

Run: pytest test_assertions.py -q
"""

from __future__ import annotations

import math
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from scipy.stats import norm

from fn_percolation.constants import RHO_RANGE, STEP, WINDOW
from fn_percolation.dgp import DEFAULT_SIGMA_F_SCALE
from fn_percolation.overlap_inference import hac_block_stats, hac_mean_se, newey_west_lag

DATA = Path(__file__).resolve().parent / "data"

ABS_TOL_RHO = 0.001      # paper rounds rho_c to three decimals
ABS_TOL_FRAC = 0.006     # paper rounds fractions to two decimals
ABS_TOL_TAU = 0.006      # paper rounds tau to two decimals
ABS_TOL_SLOPE = 0.001    # paper rounds slopes to three decimals
ABS_TOL_T = 0.08         # paper rounds t-stats to one decimal


def _summary(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name / "rolling_percolation_summary.csv")


def _q_means(name: str) -> list[float]:
    df = _summary(name).sort_values("size_group")
    return df["rho_c_mean"].tolist()


def _per_q(name: str, g: int) -> pd.DataFrame:
    return pd.read_csv(DATA / name / f"rolling_percolation_Q{g}.csv",
                       parse_dates=["date"]).set_index("date")


# ---------------------------------------------------------------------------
# tab:percolation / subsec:main_result: headline rolling percolation
# ---------------------------------------------------------------------------

def test_tab1_headline_means():
    """tab:percolation, subsec:main_result: 'Mean rho_c' row 0.524, 0.556, 0.567,
    0.576, 0.584; 'the cross-window mean bar(rho_c) rises at every step, from
    0.524 in Q1 to 0.584 in Q5'."""
    expected = [0.524, 0.556, 0.567, 0.576, 0.584]
    actual = _q_means("korea_headline")
    for g, (e, a) in enumerate(zip(expected, actual), start=1):
        assert math.isclose(a, e, abs_tol=ABS_TOL_RHO), (
            f"Manuscript claims bar(rho_c)(Q{g})={e:.3f} (tab:percolation) "
            f"but data shows {a:.4f}"
        )


def test_tab1_headline_gap():
    """subsec:main_result: 'an endpoint contrast of 0.060'."""
    means = _q_means("korea_headline")
    gap = means[4] - means[0]
    assert math.isclose(gap, 0.060, abs_tol=ABS_TOL_RHO), (
        f"Manuscript claims Q5-Q1 gap 0.060 (subsec:main_result) "
        f"but data shows {gap:.4f}"
    )


def test_si_rho_c_range_matches_macros():
    """si:dgp_validation: the simulated rho_c range is anchored to bracket the
    empirical bar(rho_c)(Q1)--bar(rho_c)(Q5) band. The Q5 mean rounds to 0.58,
    not the stale 0.60 upper bound the SI prose carried before macro-ization."""
    q5 = _q_means("korea_headline")[4]
    assert round(q5, 2) == 0.58, (
        f"SI brackets the empirical band at bar(rho_c)(Q5) (si:dgp_validation); the Q5 "
        f"mean {q5:.4f} must round to 0.58 (the pre-fix '0.60' bound was stale)"
    )
    assert q5 < 0.60, (
        f"SI must not imply a 0.60 upper bound (si:dgp_validation) but Q5 mean is {q5:.4f}"
    )


def test_tab1_headline_monotone():
    """subsec:main_result: 'the cross-window mean bar(rho_c) rises at every step'."""
    means = _q_means("korea_headline")
    assert all(a < b for a, b in zip(means, means[1:])), (
        f"Manuscript claims strict monotone Q1<Q2<...<Q5 (subsec:main_result) "
        f"but data shows {means}"
    )


def test_tab1_n_windows_matches_the_macro():
    """tab:percolation prints \\nWindowsHeadline in all five columns. Read the
    macro rather than a literal: the count moves with the sample, and a literal
    here would have to be found and retyped every time it does."""
    claimed = int(_params_macros()["nWindowsHeadline"])
    df = _summary("korea_headline")
    assert (df["n_windows"] == claimed).all(), (
        f"tab:percolation prints {claimed} rolling windows per quintile "
        f"but data shows {df['n_windows'].tolist()}"
    )


def _adjacent_diff(g: int) -> np.ndarray:
    """Per-window paired difference rho_c(Q_{g+1}, t) - rho_c(Q_g, t)."""
    a = _per_q("korea_headline", g)["rho_c"]
    b = _per_q("korea_headline", g + 1)["rho_c"]
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    joined.columns = ["a", "b"]
    return (joined["b"] - joined["a"]).values


def _endpoint_diff() -> np.ndarray:
    """Per-window endpoint difference rho_c(Q5, t) - rho_c(Q1, t)."""
    a = _per_q("korea_headline", 1)["rho_c"]
    b = _per_q("korea_headline", 5)["rho_c"]
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    joined.columns = ["a", "b"]
    return (joined["b"] - joined["a"]).values


def test_tab1_primary_inference_per_pair():
    """subsec:main_result: the gradient is read from the primary autocorrelation-robust
    inference (Newey-West HAC t and moving-block bootstrap). 'Tests robust to
    the overlap of the windows find the three steps through Q4 and the endpoint
    contrast significant and the Q4--Q5 step not';
    tab:pair_significance prints the endpoint HAC t = 6.8 and
    prints the endpoint bootstrap p < 10^-4. HAC and bootstrap p-values are read
    from the persisted pair_significance.csv (deterministic)."""
    sig = pd.read_csv(DATA / "korea_headline" / "pair_significance.csv")
    by_pair = {row["pair"]: row for _, row in sig.iterrows()}
    ep_boot = float(by_pair["Q5-Q1"]["block_boot_p_one_sided"])
    assert ep_boot < 1.5e-4, (
        f"Manuscript claims endpoint Q1-Q5 bootstrap p < 1e-4 (tab:pair_significance) "
        f"but data shows p={ep_boot:.2e}"
    )
    # The three steps through Q4 clear the one-sided level under HAC and bootstrap
    for g in (1, 2, 3):
        row = by_pair[f"Q{g+1}-Q{g}"]
        hac_p = float(row["nw_p_one_sided"])
        boot_p = float(row["block_boot_p_one_sided"])
        assert hac_p < 0.05 and boot_p < 0.05, (
            f"Manuscript claims the Q{g}-Q{g+1} step clears the conventional one-sided "
            f"level under the autocorrelation-robust inference (subsec:main_result) but "
            f"HAC p={hac_p:.3f}, bootstrap p={boot_p:.3f}"
        )
    # Q4-Q5 rejects on neither statistic, and Q3-Q4 and Q4-Q5 run in the
    # predicted direction.
    for pair in ("Q5-Q4",):
        for col, label in (("nw_p_one_sided", "HAC"),
                           ("block_boot_p_one_sided", "block bootstrap")):
            assert float(by_pair[pair][col]) >= 0.05, (
                f"Manuscript claims the {pair} step rejects on neither "
                f"statistic (tab:pair_significance) but it clears the {label} at "
                f"p={float(by_pair[pair][col]):.3f}"
            )
    for g in (3, 4):
        row = by_pair[f"Q{g+1}-Q{g}"]
        assert float(row["mean_diff"]) > 0, (
            f"Manuscript claims Q{g}-Q{g+1} runs in the predicted direction "
            f"(subsec:main_result) but the mean difference is {float(row['mean_diff']):+.4f}"
        )


def test_tab1_per_pair_fraction_holds():
    """subsec:main_result: 'Each adjacent pair is ordered in
    \\fracWindowsAdjPairLo to \\fracWindowsAdjPairHi of the windows.' Read off the
    macros rather than retyped, so the bound moves with the sentence.
    Tolerance is half the last printed digit."""
    macros = _params_macros()
    claimed = [float(macros[m].replace(r"\%", "")) / 100
               for m in ("fracWindowsAdjPairLo", "fracWindowsAdjPairHi")]
    fracs = []
    for g in (1, 2, 3, 4):
        diff = _adjacent_diff(g)
        fracs.append(float((diff >= 0).mean()))
    lo, hi = min(fracs), max(fracs)
    assert lo >= claimed[0] - 0.005 and hi <= claimed[1] + 0.005, (
        f"Manuscript claims per-pair window-level fractions in "
        f"[{claimed[0]:.2f}, {claimed[1]:.2f}] (subsec:main_result) but data "
        f"shows min={lo:.3f}, max={hi:.3f} across pairs {fracs}"
    )


def test_tab1_hac_endpoint_significance():
    """tab:pair_significance prints the endpoint HAC t = 6.8 and bootstrap
    p < 10^-4. The per-pair HAC t-statistics and
    bootstrap p-values that the body cites are checked in
    test_tab1_primary_inference_per_pair against the persisted
    pair_significance.csv."""
    ep = hac_block_stats(_endpoint_diff())
    assert math.isclose(ep["nw_t"], 6.8, abs_tol=ABS_TOL_T), (
        f"Manuscript claims endpoint Q1-Q5 Newey-West HAC t = 6.8 (subsec:main_result) "
        f"but data shows t={ep['nw_t']:.2f}"
    )
    assert ep["block_boot_p_one_sided"] < 1.5e-4, (
        f"Manuscript claims the endpoint moving-block bootstrap rejects in every "
        f"resample, p < 1e-4 (tab:pair_significance), but data shows "
        f"p={ep['block_boot_p_one_sided']:.2e}"
    )


def test_ordering_through_q4_holds_step_by_step_and_through_q5_does_not():
    """subsec:main_result: "The three steps through Q4 and the endpoint contrast
    are significant, whereas the Q4--Q5 step is not ... The ordering Q1 < Q2 < Q3 < Q4 is therefore significant at the
    5% level, whereas the ordering through Q5 is not." The abstract: "In both
    markets, the cross-window mean threshold rises significantly at each step
    from the smallest to the fourth quintile but not to the largest."
    sec:introduction: "Tests that allow for the overlap of the windows find each step through the fourth quintile significant at the 5%
    level, whereas the step to the largest quintile is significant in neither
    market." The highlights: "In Korea and Taiwan, thresholds rise
    significantly quintile by quintile to the fourth", whose Taiwanese half
    test_taiwan_every_step_through_q4_significant_and_not_q4_to_q5 covers.

    subsec:inference reads an ordering as an intersection of one-sided step
    hypotheses, which holds at the 5% level only when every step rejects there
    on its own (Berger 1982), so both halves are asserted step by step on both
    autocorrelation-robust statistics."""
    pairs = pd.read_csv(DATA / "korea_headline" / "pair_significance.csv").set_index("pair")
    for pair in ("Q2-Q1", "Q3-Q2", "Q4-Q3"):
        t, p_boot = float(pairs.loc[pair, "nw_t"]), float(pairs.loc[pair, "block_boot_p_one_sided"])
        assert t > 1.645 and p_boot < 0.05, (
            "Manuscript claims the three steps through Q4 reject on the HAC t and "
            f"the block bootstrap (subsec:main_result) but {pair} gives t={t:.2f}, "
            f"bootstrap p={p_boot:.3f}"
        )
    t, p_boot = float(pairs.loc["Q5-Q4", "nw_t"]), float(pairs.loc["Q5-Q4", "block_boot_p_one_sided"])
    assert not (t > 1.645 or p_boot < 0.05), (
        "Manuscript claims the Q4--Q5 step rejects on neither "
        f"autocorrelation-robust statistic (subsec:main_result) but it gives "
        f"t={t:.2f}, bootstrap p={p_boot:.3f}"
    )


def test_newey_west_lag_follows_the_rule_and_covers_the_overlap():
    """subsec:inference: "[the Newey--West HAC standard error], which sums the
    autocovariances of the differences up to the lag floor(4(n/100)^(2/9)) of
    Newey and West (1994), where n is the number of windows. For the 257 Korean
    windows, the lag is 4 windows, above the largest lag at which two windows
    share returns, 60/20 - 1." The lag is read from the persisted inference, not recomputed
    from the rule alone."""
    pairs = pd.read_csv(DATA / "korea_headline" / "pair_significance.csv").set_index("pair")
    n = int(pairs.loc["Q5-Q1", "n_windows"])
    lag = int(pairs.loc["Q5-Q1", "nw_lag"])
    assert lag == newey_west_lag(n) == 4, (
        f"Manuscript claims a Newey-West lag of 4 windows at n = {n} "
        f"(subsec:inference) but the inference used {lag} and the rule gives "
        f"{newey_west_lag(n)}"
    )
    assert lag > WINDOW // STEP - 1, (
        f"Manuscript claims the Newey-West lag exceeds the largest overlap lag "
        f"{WINDOW // STEP - 1} (subsec:inference) but the lag is {lag}"
    )


def test_rho_c_grid_edge_draws_are_few():
    """tab:settings, notes: "rho_c lies at the upper edge of the cutoff grid in
    \\nDrawsRhoCAtGridEdge of the \\nDrawCellsHeadline Korean draws, one for each
    seed, window, and quintile, in which the edge bounds rho_c rather than
    measuring it." Both counts are read off the per-draw file
    the means are taken over."""
    per_seed = pd.read_csv(DATA / "korea_headline" / "rolling_percolation_per_seed.csv")
    macros = _params_macros()
    at_edge = int((per_seed["rho_c"] >= float(RHO_RANGE[-1]) - 1e-9).sum())
    assert at_edge == int(macros["nDrawsRhoCAtGridEdge"]), (
        f"Manuscript claims {macros['nDrawsRhoCAtGridEdge']} Korean draws reach the "
        f"grid edge (subsec:methods) but the per-draw file holds {at_edge}"
    )
    assert len(per_seed) == int(macros["nDrawCellsHeadline"].replace("{,}", "")), (
        f"Manuscript counts {macros['nDrawCellsHeadline']} Korean draws "
        f"(subsec:methods) but the per-draw file holds {len(per_seed)}"
    )


def test_taiwan_every_step_through_q4_significant_and_not_q4_to_q5():
    """The abstract: "The rise continues significantly to the fourth quintile in
    Taiwan". sec:introduction: "In Taiwan, each of the three steps up to the
    fourth quintile is a significant rise, and the step to the largest quintile
    is not significant." subsec:taiwan: "Each of the three steps through Q4 is
    significant ... The Q4--Q5 step, -0.001, is not significant"; "bar(rho_c)
    rises from 0.520 in Q1 to 0.635 in Q4 and is 0.633 in Q5".
    subsec:inference calls a step significant when both the HAC t and the
    block bootstrap reject, and the Q4-Q5 step, which runs the other way, must
    reject on neither."""
    pairs = pd.read_csv(DATA / "taiwan" / "pair_significance.csv")
    pairs = pairs[pairs["observable"] == "rho_c"].set_index("pair")
    for pair in ("Q2-Q1", "Q3-Q2", "Q4-Q3"):
        t, p_boot = float(pairs.loc[pair, "nw_t"]), float(pairs.loc[pair, "block_boot_p_one_sided"])
        assert t > 1.645 and p_boot < 0.05, (
            "Manuscript claims every Taiwanese step through the fourth quintile "
            f"is significant (abstract, sec:introduction) but {pair} gives "
            f"t={t:.2f}, bootstrap p={p_boot:.3f}"
        )
    t, p_boot = float(pairs.loc["Q5-Q4", "nw_t"]), float(pairs.loc["Q5-Q4", "block_boot_p_one_sided"])
    assert not (t > 1.645 or p_boot < 0.05), (
        "Manuscript claims the Taiwanese rise reaches only the fourth quintile "
        f"(abstract, subsec:taiwan) but the Q4-Q5 step is significant: t={t:.2f}, "
        f"bootstrap p={p_boot:.3f}"
    )
    assert float(pairs.loc["Q5-Q4", "mean_diff"]) < 0, (
        "Manuscript claims the Taiwanese Q4--Q5 contrast is negative "
        f"(subsec:taiwan) but it is {float(pairs.loc['Q5-Q4', 'mean_diff']):+.4f}"
    )


def test_taiwan_hac_endpoint_significant():
    """subsec:taiwan: 'The Taiwanese endpoint contrast of 0.112 is significant
    (HAC t = 15.6).' Significant is read as in subsec:inference, on both the
    HAC t and the bootstrap."""
    sig = pd.read_csv(DATA / "taiwan" / "pair_significance.csv")
    row = sig[(sig["pair"] == "Q5-Q1")
              & (sig["observable"] == "rho_c")].iloc[0]
    assert math.isclose(float(row["nw_t"]), 15.6, abs_tol=ABS_TOL_T), (
        f"Manuscript claims Taiwan endpoint Newey-West HAC t = 15.6 (subsec:taiwan) "
        f"but data shows t={float(row['nw_t']):.2f}"
    )
    assert float(row["block_boot_p_one_sided"]) < 1.5e-4, (
        f"Manuscript claims the Taiwan endpoint moving-block bootstrap rejects "
        f"in every resample, p < 1e-4 (tab:taiwan_rerun), but data shows "
        f"p={float(row['block_boot_p_one_sided']):.2e}"
    )


def _n_implied_gap_component(name: str) -> float:
    """Q1-Q5 rho_c gap component the unequal quintile universes explain: the
    pooled rho_c slope on sqrt(2 ln(N-1)) evaluated across the per-window
    stock counts the two endpoint quintiles hold (subsec:threshold_vs_mean)."""
    slopes = pd.read_csv(DATA / "network_size_resampling" / "rho_c_vs_N_slopes.csv")
    slope = float(slopes[slopes["quintile"] == 0].iloc[0]["rho_c_slope"])
    n1, n5 = (_per_q(name, g)["n_stocks"] for g in (1, 5))
    idx = n1.index.intersection(n5.index)
    loc = lambda n: float(np.sqrt(2.0 * np.log(n.loc[idx].astype(float) - 1.0)).mean())
    return slope * (loc(n5) - loc(n1))


def test_taiwan_unequal_universes_bound_endpoint_gap():
    """subsec:threshold_vs_mean: 'The Korean pooled slope accounts for ... and,
    if the Taiwanese rho_c depends on N as the Korean rho_c does, for 0.006 (5%)
    of the Taiwanese one.' subsec:resampling_method applies the slope to the
    network sizes of Q1 and Q5 in each window."""
    implied = _n_implied_gap_component("taiwan")
    assert math.isclose(implied, 0.006, abs_tol=0.0005), (
        f"Manuscript claims the unequal Taiwanese universes contribute 0.006 to "
        f"the Q1-Q5 contrast (subsec:threshold_vs_mean) but data shows {implied:.4f}"
    )
    gap = _q_means("taiwan")[4] - _q_means("taiwan")[0]
    pct = implied / gap * 100
    assert math.isclose(pct, 5.0, abs_tol=0.5), (
        f"Manuscript claims the Korean pooled slope accounts for 5% of the "
        f"Taiwanese endpoint contrast (subsec:threshold_vs_mean) but data shows {pct:.1f}%"
    )


def test_korea_network_size_component_matches_taiwans_share():
    """subsec:threshold_vs_mean: 'The Korean pooled slope accounts for 0.003 (5%)
    of the Korean endpoint contrast and, if the Taiwanese rho_c depends on N as
    the Korean rho_c does, for 0.006 (5%) of the Taiwanese one.'"""
    implied = _n_implied_gap_component("korea_headline")
    assert implied > 0, (
        "Manuscript claims the Korean network-size component carries the same "
        f"sign as Taiwan's (subsec:threshold_vs_mean) but data shows {implied:+.4f}"
    )
    means = _q_means("korea_headline")
    pct = implied / (means[4] - means[0]) * 100
    assert math.isclose(pct, 5.0, abs_tol=0.5), (
        f"Manuscript claims the Korean pooled slope accounts for 5% of the Korean "
        f"endpoint contrast (subsec:threshold_vs_mean) but data shows {pct:.1f}%"
    )
    tw_pct = (_n_implied_gap_component("taiwan")
              / (_q_means("taiwan")[4] - _q_means("taiwan")[0]) * 100)
    assert abs(pct - tw_pct) < 5.0, (
        f"Manuscript prints the two markets' network-size shares as comparable "
        f"(subsec:threshold_vs_mean) but Korea shows {pct:.1f}% against Taiwan's {tw_pct:.1f}%"
    )


def test_q1_network_smaller_than_q5_in_every_window():
    """subsec:threshold_vs_mean: 'Because Q1 holds fewer stocks than Q5 in each
    window, the network size alone produces part of the endpoint contrast.'
    Asserted over every window of both markets, as the network-size component
    that follows is computed in both."""
    for name in ("korea_headline", "taiwan"):
        sizes = _per_q(name, 1)[["n_stocks"]].join(
            _per_q(name, 5)[["n_stocks"]], lsuffix="_q1", rsuffix="_q5", how="outer")
        assert sizes.notna().all().all(), (
            f"{name}: Q1 and Q5 do not share their windows, so 'in each window' "
            "(subsec:threshold_vs_mean) cannot be checked")
        bad = sizes[sizes["n_stocks_q1"] >= sizes["n_stocks_q5"]]
        assert bad.empty, (
            "Manuscript claims Q1 holds fewer stocks than Q5 in each window "
            f"(subsec:threshold_vs_mean) but {name} has {len(bad)} windows "
            f"where it does not, first at {bad.index[0].date()}")


def _subperiod_endpoint() -> pd.DataFrame:
    return pd.read_csv(DATA / "subperiod" / "subperiod_endpoint.csv").set_index(
        ["market", "segment"])


def test_subperiod_korea_endpoint_attenuates():
    """subsec:subperiod: 'The Korean endpoint contrast in the second half of the
    windows ... is +0.023, 23% of its first-half value of +0.097. The
    second-half contrast stays significant (Table tab:subperiod), has a block
    bootstrap p-value of 2.1e-3, and is positive in 71% of the second-half
    windows.' tab:subperiod prints the HAC t, 8.3 in the first half and 3.0 in
    the second. Significant is read as in subsec:inference, on both the
    HAC t and the bootstrap."""
    ep = _subperiod_endpoint()
    early = ep.loc[("korea", "early")]
    late = ep.loc[("korea", "late")]
    assert math.isclose(early["nw_t"], 8.3, abs_tol=ABS_TOL_T), (
        f"Manuscript claims first-half HAC t = 8.3 (tab:subperiod) "
        f"but data shows {early['nw_t']:.2f}"
    )
    assert math.isclose(late["nw_t"], 3.0, abs_tol=ABS_TOL_T), (
        f"Manuscript claims second-half HAC t = 3.0 (tab:subperiod) "
        f"but data shows {late['nw_t']:.2f}"
    )
    assert late["nw_t"] < early["nw_t"], (
        f"Manuscript claims the Korean HAC t falls with the gap (tab:subperiod) "
        f"but late t={late['nw_t']:.2f} >= early t={early['nw_t']:.2f}"
    )
    assert math.isclose(early["mean_diff"], 0.097, abs_tol=ABS_TOL_RHO), (
        f"Manuscript claims first-half gap = +0.097 (subsec:subperiod) "
        f"but data shows {early['mean_diff']:.4f}"
    )
    assert math.isclose(late["mean_diff"], 0.023, abs_tol=ABS_TOL_RHO), (
        f"Manuscript claims second-half gap = +0.023 (subsec:subperiod) "
        f"but data shows {late['mean_diff']:.4f}"
    )
    share = late["mean_diff"] / early["mean_diff"] * 100
    claimed = float(_params_macros()["rhoCGapLateShareOfEarlyPct"].replace(r"\%", ""))
    assert abs(share - claimed) <= 0.5, (
        f"Manuscript claims the second-half gap is {claimed:.0f}% of its first-half "
        f"size (subsec:subperiod) but data shows {share:.1f}%"
    )
    assert 0 < share < 100, (
        f"Manuscript claims the Korean gap compresses and keeps its sign "
        f"(subsec:subperiod) but the second half holds {share:.1f}% of the first"
    )
    assert late["nw_t"] > 1.645 and late["block_boot_p_one_sided"] < 0.05, (
        f"Manuscript claims the second-half contrast stays significant "
        f"(subsec:subperiod) but HAC t={late['nw_t']:.2f}, "
        f"bootstrap p={late['block_boot_p_one_sided']:.4f}"
    )
    assert math.isclose(late["frac_positive"], 0.71, abs_tol=ABS_TOL_FRAC), (
        f"Manuscript claims the endpoint holds in 71% of second-half windows "
        f"(subsec:subperiod) but data shows {late['frac_positive']:.3f}"
    )


def test_subperiod_taiwan_endpoint_stays_significant():
    """subsec:taiwan: 'The Taiwanese endpoint contrast falls from +0.120 in the
    first half of the windows to +0.104 in the second, 86% of its first-half
    value, whereas the Korean endpoint contrast falls to 23% of its first-half
    value'. tab:subperiod
    prints its HAC t, 10.2 in the first half and 12.9 in the second."""
    ep = _subperiod_endpoint()
    tw_early = ep.loc[("taiwan", "early")]
    tw_late = ep.loc[("taiwan", "late")]
    assert math.isclose(tw_early["nw_t"], 10.2, abs_tol=ABS_TOL_T), (
        f"Manuscript claims Taiwan first-half HAC t = 10.2 (tab:subperiod) "
        f"but data shows {tw_early['nw_t']:.2f}"
    )
    assert math.isclose(tw_late["nw_t"], 12.9, abs_tol=ABS_TOL_T), (
        f"Manuscript claims Taiwan second-half HAC t = 12.9 (tab:subperiod) "
        f"but data shows {tw_late['nw_t']:.2f}"
    )
    assert tw_late["nw_t"] > tw_early["nw_t"], (
        f"Manuscript claims the Taiwan HAC t rises across its own median split "
        f"(tab:subperiod) but early t={tw_early['nw_t']:.2f} >= late "
        f"t={tw_late['nw_t']:.2f}"
    )
    share = 100 * tw_late["mean_diff"] / tw_early["mean_diff"]
    printed = float(_params_macros()["rhoCGapLateShareOfEarlyTwPct"].replace("\\%", ""))
    assert math.isclose(share, printed, abs_tol=0.5), (
        f"Manuscript claims the Taiwanese second-half contrast is {printed:.0f}% "
        f"of the first-half one (subsec:subperiod) but data shows {share:.1f}%"
    )
    assert math.isclose(tw_early["mean_diff"], 0.120, abs_tol=ABS_TOL_RHO), (
        f"Manuscript claims Taiwan first-half gap = +0.120 (subsec:subperiod) "
        f"but data shows {tw_early['mean_diff']:.4f}"
    )
    assert math.isclose(tw_late["mean_diff"], 0.104, abs_tol=ABS_TOL_RHO), (
        f"Manuscript claims Taiwan second-half gap = +0.104 (subsec:subperiod) "
        f"but data shows {tw_late['mean_diff']:.4f}"
    )
    assert float(tw_late["block_boot_p_one_sided"]) < 0.05, (
        f"Manuscript claims the Taiwanese second-half contrast is significant "
        f"(tab:subperiod) but its moving-block bootstrap gives "
        f"p={float(tw_late['block_boot_p_one_sided']):.4f}"
    )
    share = tw_late["mean_diff"] / tw_early["mean_diff"] * 100
    claimed = float(_params_macros()["rhoCGapLateShareOfEarlyTwPct"].replace(r"\%", ""))
    assert abs(share - claimed) <= 0.5, (
        f"Manuscript claims the Taiwanese rise keeps {claimed:.0f}% of its magnitude "
        f"(rhoCGapLateShareOfEarlyTwPct) but data shows {share:.1f}%"
    )
    assert share > 50, (
        f"Manuscript claims the Taiwan endpoint keeps most of its magnitude "
        f"(subsec:subperiod) but its second half holds {share:.1f}% of the first"
    )


def test_subperiod_yearblock_korea_declines_taiwan_does_not():
    """subsec:subperiod: 'The Korean HAC t-statistic falls from each five-year
    block to the next through 2015--2019, the one block in which the endpoint
    contrast is not significant. The endpoint contrast is significant again in
    the last block.'
    Of Taiwan: 'Its HAC t-statistic is at least \\tHacYearBlockTwA in every
    five-year period' (tab:subperiod).
    subsec:time_method: 'five-year calendar periods, the last of which also
    covers the sample's final year'.
    subsec:limits names the same block.

    A block's contrast is not significant when its HAC t misses the one-sided 5% level of
    subsec:inference, the criterion generate_params derives
    \\krUnresolvedBlock with."""
    blk = pd.read_csv(DATA / "subperiod" / "subperiod_yearblocks.csv")
    kr = blk[blk["market"] == "korea"].set_index("block")
    tw = blk[blk["market"] == "taiwan"].set_index("block")
    # The blocks are derived from the sample span (run_subperiod_split._year_blocks),
    # so read which ones exist off the file rather than naming them here: a block
    # a longer sample adds is one more the claim has to hold over.
    named = _params_macros()["krUnresolvedBlock"].replace("--", "-")
    unresolved = [b for b in kr.index if kr.loc[b, "nw_t"] <= 1.645]
    assert unresolved == [named], (
        f"Manuscript claims {named} is the only Korean block that leaves the "
        f"endpoint not significant (subsec:subperiod) but those blocks are "
        f"{unresolved} over HAC t {kr['nw_t'].round(2).to_dict()}"
    )
    upto = list(kr.index[:list(kr.index).index(named) + 1])
    ts = kr.loc[upto, "nw_t"].tolist()
    assert all(a > b for a, b in zip(ts, ts[1:])), (
        f"Manuscript claims the Korean endpoint t falls from each block to the "
        f"next up to {named} (subsec:subperiod) but the blocks {upto} show {ts}"
    )
    last = kr.index[-1]
    end_year = pd.Timestamp(_subperiod_endpoint().loc[("korea", "full"), "end_date"]).year
    assert int(last.split("-")[1]) == end_year and kr.loc[last, "nw_t"] > 1.645, (
        f"Manuscript claims the last block runs to the sample's end and "
        f"is significant again (subsec:time_method, subsec:subperiod) but {last} shows HAC "
        f"t = {kr.loc[last, 'nw_t']:.2f} against a sample ending in {end_year}"
    )
    tw_ts = tw["nw_t"].tolist()
    printed = float(_params_macros()["tHacYearBlockTwA"])
    assert math.isclose(tw_ts[0], printed, abs_tol=0.05) and min(tw_ts) == tw_ts[0], (
        f"Manuscript claims the Taiwan endpoint HAC t is at least {printed} in "
        f"every block, the value of the first block (tab:subperiod), but the "
        f"blocks show {tw_ts}"
    )


def test_subperiod_second_half_matches_median_split():
    """subsec:subperiod: 'The Korean endpoint contrast falls from +0.087 in the
    first half of the windows to +0.022 in the second half (the 129 windows from
    2015-07-22)' and 'In the second half, the Q1--Q2 step of +0.033 ... exceeds
    the whole endpoint contrast', so the steps and the endpoint contrast are read
    on one half of the median split of subsec:time_method. The cross-quintile cut aligns all five quintiles and the
    endpoint split aligns Q1 against Q5, so the two select the same windows only
    where no window drops an interior quintile; this checks that they do."""
    post = pd.read_csv(DATA / "subperiod" / "subperiod_second_half.csv")
    late = _subperiod_endpoint().loc[("korea", "late")]
    n_post = int(post["n_windows"].iloc[0])
    assert n_post == int(late["n"]), (
        f"Manuscript claims the second-half steps are read on the median split's second "
        f"half (subsec:subperiod) but the cross-quintile cut "
        f"has {n_post} windows against {int(late['n'])} in the late segment"
    )
    assert post["start_date"].iloc[0] == late["start_date"], (
        f"Manuscript claims the second-half subsample starts where the median "
        f"split's second half starts (subsec:subperiod) but the "
        f"cross-quintile cut starts {post['start_date'].iloc[0]} against "
        f"{late['start_date']}"
    )
    means = post.sort_values("quintile")["mean_rho_c"].tolist()
    endpoint = means[4] - means[0]
    assert math.isclose(endpoint, late["mean_diff"], abs_tol=ABS_TOL_RHO), (
        f"Manuscript compares the second-half steps with the second-half "
        f"endpoint contrast (subsec:subperiod) but "
        f"the cross-quintile means give {endpoint:.4f} against "
        f"{late['mean_diff']:.4f}"
    )


def test_subperiod_second_half_ordering_and_steps():
    """subsec:subperiod: 'In the second half, the Q1--Q2 step of +0.030 is
    significant (HAC t = 6.3) and exceeds the second-half endpoint contrast, so
    the three later steps sum to a fall. The second-half Q4--Q5 step is -0.012
    (HAC t = -2.2)'. subsec:korean_middle_steps: 'In the second half of the
    windows, neither the Q2--Q3 step (HAC t = -0.1) nor the Q3--Q4 step (HAC
    t = 1.2) is significant. In the second half, the means of Q2, Q3, and Q4
    lie within 0.005 of each other.'
    subsec:limits: 'The Korean rise through Q4 holds for all windows taken
    together rather than for every period', which the second-half Q2--Q3 and
    Q3--Q4 steps show."""
    post = pd.read_csv(DATA / "subperiod" / "subperiod_second_half.csv").set_index(
        "quintile")
    means = post.sort_index()["mean_rho_c"].tolist()
    assert means[4] < means[3], (
        f"Manuscript claims the second-half Q4--Q5 step is negative "
        f"(subsec:subperiod) but data shows Q4={means[3]:.4f} "
        f"Q5={means[4]:.4f}"
    )
    q2 = post.loc[2]
    assert math.isclose(q2["step_mean_diff"], 0.030, abs_tol=ABS_TOL_RHO), (
        f"Manuscript claims 'the Q1--Q2 step of +0.030' in the second half "
        f"(subsec:subperiod) but data shows {q2['step_mean_diff']:.4f}"
    )
    assert q2["step_mean_diff"] > means[4] - means[0], (
        f"Manuscript claims the second-half Q1--Q2 step 'exceeds the second-half "
        f"endpoint contrast' (subsec:subperiod) but the step {q2['step_mean_diff']:+.4f} "
        f"falls short of the endpoint {means[4] - means[0]:+.4f}"
    )
    assert math.isclose(q2["step_nw_t"], 6.3, abs_tol=ABS_TOL_T), (
        f"Manuscript claims the second-half Q1-Q2 step has '(HAC t = 6.3)' "
        f"(subsec:subperiod) but data shows {q2['step_nw_t']:.2f}"
    )
    assert q2["step_block_boot_p_one_sided"] < 0.05, (
        f"Manuscript claims the second-half 'Q1--Q2 step stays "
        f"significant' (subsec:subperiod) but its moving-block "
        f"bootstrap p is {q2['step_block_boot_p_one_sided']:.3f}"
    )
    q5 = post.loc[5]
    assert math.isclose(q5["step_mean_diff"], -0.012, abs_tol=ABS_TOL_RHO), (
        f"Manuscript claims the second-half Q4--Q5 step 'is -0.012' "
        f"(subsec:subperiod) but data shows {q5['step_mean_diff']:.4f}"
    )
    assert math.isclose(q5["step_nw_t"], -2.2, abs_tol=ABS_TOL_T), (
        f"Manuscript claims the second-half Q4-Q5 step has '(HAC t = -2.2)' "
        f"(subsec:subperiod) but data shows {q5['step_nw_t']:.2f}"
    )
    for g, expected in ((3, -0.1), (4, 1.2)):
        actual = float(post.loc[g, "step_nw_t"])
        assert math.isclose(actual, expected, abs_tol=ABS_TOL_T), (
            f"Manuscript claims 'the Q2--Q3 step (HAC t = -0.1)' and 'the Q3--Q4 "
            f"step (HAC t = 1.2)' (subsec:subperiod) but the step into Q{g} shows {actual:.2f}"
        )
        assert actual <= 1.645, (
            f"Manuscript claims neither interior step is significant "
            f"(subsec:subperiod) but the step into Q{g} has HAC t = {actual:.2f}"
        )
    span = float(post.loc[[2, 3, 4], "mean_rho_c"].max()
                 - post.loc[[2, 3, 4], "mean_rho_c"].min())
    assert math.isclose(span, 0.005, abs_tol=ABS_TOL_RHO), (
        f"Manuscript claims 'The means of Q2, Q3, and Q4 lie within "
        f"0.005 of each other' (subsec:korean_middle_steps) but data shows {span:.4f}"
    )


def _adjacent_hac_t(subdir: str, col: str) -> dict[str, float]:
    """HAC t of each adjacent-pair per-window difference in a rolling output."""
    panel = pd.concat(
        {g: _per_q(subdir, g)[col] for g in range(1, 6)}, axis=1
    ).dropna().sort_index()
    return {f"Q{g-1}-Q{g}": float(
        hac_block_stats((panel[g] - panel[g - 1]).to_numpy())["nw_t"])
        for g in range(2, 6)}


def test_robustness_inversions_significance():
    """tab:robustness_contrast caption: '"Ordering" ... names the step whose two
    means reverse or are equal at three decimals'. Of the KOSPI-only Q3--Q4
    step, subsec:main_result says: 'Neither its Q2--Q3 step ... nor its Q3--Q4
    step, at which its two means reverse (HAC t = -0.87), is significant'. The second-half Q4--Q5 step, no longer a row of the table, is
    'a fall' (subsec:subperiod). The check enumerates the reversals
    rather than naming a count, because a count goes stale the next time a row
    is added.
    """
    for subdir, col in (("kospi_only", "rho_c"),):
        t = _adjacent_hac_t(subdir, col)["Q3-Q4"]
        assert t < 0, (
            f"tab:robustness_contrast prints 'Q3--Q4 reversed' for the "
            f"{subdir} row but its HAC t is {t:.2f}"
        )
        assert abs(t) < 1.96, (
            f"Manuscript claims the Q3--Q4 reversal 'is not significant' "
            f"(subsec:main_result) but the "
            f"{subdir} Q3--Q4 step has HAC t = {t:.2f}"
        )

    post = pd.read_csv(DATA / "subperiod" / "subperiod_second_half.csv").set_index(
        "quintile")
    t_post = float(post.loc[5, "step_nw_t"])
    assert t_post < -2.0, (
        f"Manuscript calls the second-half Q4--Q5 step 'a significant "
        f"fall' (subsec:subperiod) but its HAC t is "
        f"{t_post:.2f}"
    )


def test_every_check_but_the_null_keeps_the_endpoint_positive():
    """subsec:main_result: "The shuffled-returns null, which leaves only noise,
    reverses the endpoint contrast, whereas Marchenko--Pastur clipping, which
    removes noise, widens it (Table tab:robustness_contrast), so noise does not
    produce the rise." subsec:dissipation_mechanism: "None of the exchange
    board, the sector mix, and the estimation settings accounts for the
    endpoint contrast". The set of
    checks is read off tab:robustness_contrast itself, so a row added to the
    table is covered without editing this test. Each row's endpoint cell prints
    one macro, whose value carries the sign (a retention share is positive
    when the weekly gap keeps the daily gap's sign)."""
    tex = (Path(__file__).resolve().parent / "paper" / "paper.tex").read_text()
    table = tex[tex.index(r"\label{tab:robustness_contrast}"):]
    body = table[table.index(r"\midrule"):table.index(r"\bottomrule")]
    macros = _params_macros()
    rows = {}
    for line in body.splitlines():
        cells = line.split("&")
        if len(cells) != 3:
            continue
        label = re.sub(r"\\tnote\{\w\}", "", cells[0]).strip()
        names = re.findall(r"\\(\w+)", cells[1])
        assert len(names) == 1, f"tab:robustness_contrast row {label!r} prints {names}"
        rows[label] = float(macros[names[0]].replace("\\%", "").replace("%", ""))
    assert "Shuffled-returns null" in rows and len(rows) >= 10, (
        f"tab:robustness_contrast parsed into {sorted(rows)}"
    )
    assert rows.pop("Shuffled-returns null") < 0, (
        "tab:robustness_contrast reports the shuffled-returns null as reversing "
        "the endpoint contrast"
    )
    negative = {k: v for k, v in rows.items() if v <= 0}
    assert not negative, (
        "Manuscript claims that none of the exchange board, the sector mix, and "
        "the estimation settings accounts for the endpoint contrast "
        f"(subsec:dissipation_mechanism) but {negative}"
    )


def test_interior_ordering_reverses_only_at_q3q4_and_not_significant():
    """subsec:korean_middle_steps: 'In the KOSPI-only panel, the mean of Q4
    lies below the mean of Q3 (Table tab:robustness_contrast). Neither the
    KOSPI-only Q2--Q3 step nor the KOSPI-only Q3--Q4 step is significant'.
    Marchenko--Pastur clipping 'keeps the five means ordered', ranking size
    within sectors keeps 'the ordering of all five means', and rho_c^chi's 'Q3
    and Q4 means are equal at three decimals' (subsec:korean_middle_steps;
    tab:robustness_contrast marks the row 'Q3--Q4 equal'), a tie that test_chi_peak_monotone_ordering reads at three
    decimals and that this test reads unrounded. The interior ordering is Q1 < Q2 < Q3 < Q4 on the
    cross-window means. Two claims: which step reverses, and that the reversal
    is not significant. The check the sentence names is checked against the set
    the artifacts give, not against a list repeated here."""
    checks = {
        "korea_headline": "rho_c", "rmt_filtered": "rho_c",
        "within_sector": "rho_c", "kospi_only": "rho_c",
        "susceptibility": "rho_c_chi",
    }
    inverted = {}
    for subdir, col in checks.items():
        means = [_per_q(subdir, g)[col].mean() for g in range(1, 5)]
        steps = [i for i, (a, b) in enumerate(zip(means, means[1:]), start=1)
                 if b <= a]
        if steps:
            inverted[subdir] = steps
    assert all(steps == [3] for steps in inverted.values()), (
        f"Manuscript claims the interior ordering reverses at the Q3--Q4 step "
        f"(subsec:main_result) but the reversing steps are {inverted}"
    )
    for subdir in inverted:
        t = _adjacent_hac_t(subdir, checks[subdir])["Q3-Q4"]
        assert abs(t) < 1.96, (
            f"Manuscript claims the interior reversal 'is not "
            f"significant' (subsec:main_result) but {subdir} shows HAC t = {t:.2f}"
        )
    assert set(inverted) == {"kospi_only"}, (
        f"Manuscript names the KOSPI-only universe as the check under which the "
        f"interior ordering reverses (subsec:main_result) but the checks that "
        f"break it are {sorted(inverted)}"
    )


def test_rho_c_slope_on_network_size():
    """subsec:threshold_vs_mean: "In the regression on sqrt(2 ln(N-1)), the
    pooled slope of rho_c is \\rhoCSlopeSqrtLogN. The slope of each quintile is
    positive." """
    sl = pd.read_csv(DATA / "network_size_resampling" / "rho_c_vs_N_slopes.csv")
    pooled = sl[sl["quintile"] == 0].iloc[0]
    printed = float(_params_macros()["rhoCSlopeSqrtLogN"])
    assert math.isclose(float(pooled["rho_c_slope"]), printed, abs_tol=ABS_TOL_SLOPE), (
        f"Manuscript prints the pooled rho_c slope on sqrt(2 ln(N-1)) as {printed} "
        f"(subsec:threshold_vs_mean) but data shows {float(pooled['rho_c_slope']):.3f}"
    )
    per_q = sl[sl["quintile"] != 0]
    assert len(per_q) == 5 and (per_q["rho_c_slope"] > 0).all(), (
        "Manuscript claims the rho_c slope is positive in every quintile "
        f"(subsec:threshold_vs_mean) but slopes are {per_q['rho_c_slope'].tolist()}"
    )


# ---------------------------------------------------------------------------
# subsec:size_confounds: the KOSPI-only control arm (\nKospiTickers).
# ---------------------------------------------------------------------------

def test_kospi_universe_size_matches_the_arm_it_describes():
    """tab:robustness_summary note a: 'Quintiles formed from the
    \\nKospiTickers KOSPI common stocks alone'.

    The count must be the universe the arm ran on, not a wider one. It was
    previously read off a listing panel spanning marcap's full history from
    1995, so it counted KOSPI codes that never entered the 2005 sample and
    printed a universe larger than the arm used -- the design-time-frame
    crossing of a macro's population. Asserted against the market-cap panel the arm
    restricts, so the two cannot drift apart again."""
    n = int(pd.read_csv(DATA / "kospi_only" / "universe_size.csv"
                        ).iloc[0]["n_kospi_tickers"])
    flag = pd.read_csv(DATA / "cache" / "kr_market_flag.csv", dtype={"code": str})
    kospi = set(flag.loc[flag["market"] == "KOSPI", "code"])
    marcap_cols = set(pd.read_parquet(DATA / "cache"
                                      / "marketcap_panel.parquet").columns.astype(str))
    realized = len(kospi & marcap_cols)
    assert n == realized, (
        f"subsec:checks prints {n} KOSPI common stocks but the arm ran on "
        f"{realized}: the printed count is a wider population than the sample"
    )
    assert int(_params_macros()["nKospiTickers"].replace("{,}", "")) == n, (
        f"\\nKospiTickers disagrees with the count run_kospi_only.py persisted ({n})"
    )


def test_taiwan_chi_steps_through_q4_significant():
    """subsec:taiwan: 'The susceptibility-peak threshold rho_c^chi also rises
    through Q4, and each of its three steps through Q4 is significant, with HAC
    t-statistics between \\tHacTwChiStepMin and \\tHacTwChiStepMax.'
    Significant is read as in subsec:inference, on both statistics."""
    csv = pd.read_csv(DATA / "taiwan" / "pair_significance.csv")
    chi = csv[csv["observable"] == "rho_c_chi"].set_index("pair")
    steps = chi.loc[["Q2-Q1", "Q3-Q2", "Q4-Q3"]]
    for pair, r in steps.iterrows():
        assert r["mean_diff"] > 0 and r["nw_p_one_sided"] < 0.05 and r["block_boot_p_one_sided"] < 0.05, (
            f"subsec:taiwan claims the Taiwanese rho_c^chi {pair} step is a "
            f"significant rise, but its mean is {r['mean_diff']:.4f}, HAC p "
            f"{r['nw_p_one_sided']:.3g} and bootstrap p {r['block_boot_p_one_sided']:.3g}")
    macros = _params_macros()
    for name, value in (("tHacTwChiStepMin", steps["nw_t"].min()),
                        ("tHacTwChiStepMax", steps["nw_t"].max())):
        assert math.isclose(float(macros[name]), value, abs_tol=0.05), (
            f"subsec:taiwan prints \\{name} = {macros[name]} but the file gives {value:.2f}")


def test_kosdaq_is_the_board_of_smaller_stocks():
    """subsec:checks: 'The KOSPI-only check forms the quintiles from the KOSPI
    common stocks alone, which removes KOSDAQ, the board of smaller and more
    thinly traded stocks.' The size half is asserted here on the median over
    stocks of each stock's median market capitalization; a KOSDAQ segment
    (KOSDAQ GLOBAL) counts as KOSDAQ."""
    flag = pd.read_csv(DATA / "cache" / "kr_market_flag.csv",
                       dtype={"code": str}).set_index("code")["market"]
    mc = pd.read_parquet(DATA / "cache" / "marketcap_panel.parquet")
    per_stock = mc.median()
    per_stock.index = per_stock.index.astype(str)
    board = flag.reindex(per_stock.index).str.startswith("KOSDAQ").map(
        {True: "KOSDAQ", False: "KOSPI"})
    med = per_stock.groupby(board).median()
    assert med["KOSDAQ"] < med["KOSPI"], (
        f"subsec:checks calls KOSDAQ the board of smaller stocks, but the median "
        f"stock's market capitalization is {med['KOSDAQ']:.3g} on KOSDAQ and "
        f"{med['KOSPI']:.3g} on KOSPI")


def test_within_sector_steps_significance():
    """subsec:size_confounds: 'Of the within-sector steps, the Q1--Q2, Q2--Q3,
    and Q4--Q5 steps are significant, whereas the Q3--Q4 step is not.'
    tab:size_confound_tests prints the statistics. subsec:main_result:
    'Ranking size within sectors keeps 81% of the endpoint contrast and the
    ordering of all five means.' That Q3--Q4 is the smallest step is read off
    the five means of tab:robustness_summary. Significant is read as in subsec:inference, on both
    the one-sided HAC t and the block bootstrap."""
    csv = pd.read_csv(DATA / "within_sector" / "pair_significance.csv")
    rows = {r["pair"]: r for _, r in csv.iterrows()}
    for pair in ("Q2-Q1", "Q3-Q2", "Q5-Q4"):
        assert rows[pair]["nw_p_one_sided"] < 0.05 and rows[pair]["block_boot_p_one_sided"] < 0.05, (
            f"Manuscript claims the within-sector {pair} step is significant "
            f"(subsec:size_confounds) but its one-sided HAC p is "
            f"{rows[pair]['nw_p_one_sided']:.3f} and bootstrap p "
            f"{rows[pair]['block_boot_p_one_sided']:.3f}"
        )
    q34 = rows["Q4-Q3"]
    assert q34["nw_p_one_sided"] >= 0.05 and q34["block_boot_p_one_sided"] >= 0.05, (
        f"Manuscript claims the within-sector Q3--Q4 step is significant by "
        f"neither statistic (subsec:size_confounds) but the HAC p is "
        f"{q34['nw_p_one_sided']:.3f} and the bootstrap p {q34['block_boot_p_one_sided']:.3f}"
    )
    steps = {pair: float(rows[pair]["mean_diff"]) for pair in ("Q2-Q1", "Q3-Q2", "Q4-Q3", "Q5-Q4")}
    assert min(steps.values()) > 0 and min(steps, key=steps.get) == "Q4-Q3", (
        f"Manuscript claims the within-sector means rise at every step and "
        f"Q3--Q4 is the smallest step (subsec:main_result, tab:robustness_summary) but the steps are {steps}"
    )


# ---------------------------------------------------------------------------
# subsec:threshold_vs_mean / si:rho_c_vs_mean: rho_c co-moves with segment mean |rho|
# ---------------------------------------------------------------------------

def _pooled_rho_c_mean_corr() -> pd.DataFrame:
    """Per-(window, quintile) rho_c joined to the same cell's off-diagonal mean
    |rho_ij|, pooled over all cells. Reproduces the merge in
    scripts/scaling/run_mean_corr_diagnostic.py from its persisted per-window file, so
    the assertion runs on primitive outputs rather than the derived coupling.csv."""
    mc = pd.read_csv(DATA / "mean_corr_diagnostic" / "per_window_mean_corr.csv",
                     parse_dates=["date"])
    parts = []
    for g in (1, 2, 3, 4, 5):
        head = _per_q("korea_headline", g)["rho_c"].reset_index()
        mine = mc[mc["quintile"] == g][["date", "mean_abs_corr"]]
        parts.append(head.merge(mine, on="date"))
    return pd.concat(parts, ignore_index=True)


def test_meancorr_spearman_coupling():
    """subsec:threshold_vs_mean: 'Over all (window, quintile) cells, the Spearman
    rank correlation between rho_c and the mean |rho_ij| is 0.95' (macro
    meanCorrSpearman)."""
    pooled = _pooled_rho_c_mean_corr()
    s = pooled[["rho_c", "mean_abs_corr"]].corr(method="spearman").iloc[0, 1]
    assert math.isclose(s, 0.95, abs_tol=ABS_TOL_FRAC), (
        f"Manuscript claims pooled Spearman(rho_c, mean|rho|)=0.95 "
        f"(subsec:threshold_vs_mean, macro meanCorrSpearman) but data shows {s:.4f}"
    )


def test_meancorr_comovement_monotone():
    """subsec:main_result: 'the cross-window mean bar(rho_c) rises at every
    step'. subsec:threshold_vs_mean: 'The mean |rho_ij| rises from
    \\meanAbsCorrQone in Q1 to \\meanAbsCorrQfive in Q5 (HAC t =
    \\tHacMeanCorrQoneQfive; Table tab:percolation).' Significant is read as
    in subsec:inference, on both the HAC t and the bootstrap of the per-window
    Q5-Q1 difference."""
    df = pd.read_csv(
        DATA / "mean_corr_diagnostic" / "cross_quintile_summary.csv"
    ).sort_values("quintile")
    rho_c = df["rho_c_mean"].tolist()
    assert all(a < b for a, b in zip(rho_c, rho_c[1:])), (
        f"Manuscript claims rho_c rises monotonically Q1->Q5 (subsec:main_result) "
        f"but data shows {rho_c}"
    )
    mc = pd.read_csv(DATA / "mean_corr_diagnostic" / "per_window_mean_corr.csv")
    per_window = mc.pivot(index="date", columns="quintile", values="mean_abs_corr")
    ep = hac_block_stats((per_window[5] - per_window[1]).dropna().to_numpy())
    assert ep["nw_t"] > 1.645 and ep["block_boot_p_one_sided"] < 0.05, (
        "Manuscript claims the mean |rho_ij| rises from Q1 to Q5 "
        f"(subsec:threshold_vs_mean) but HAC t={ep['nw_t']:.2f}, "
        f"block bootstrap p={ep['block_boot_p_one_sided']:.3f}"
    )
    printed_t = float(_params_macros()["tHacMeanCorrQoneQfive"])
    assert math.isclose(ep["nw_t"], printed_t, abs_tol=0.05), (
        f"Manuscript prints the mean |rho_ij| endpoint HAC t as {printed_t} "
        f"(tHacMeanCorrQoneQfive, subsec:threshold_vs_mean) but data gives "
        f"{ep['nw_t']:.2f}"
    )


def test_meancorr_per_quintile_levels():
    """tab:percolation 'Mean |rho_ij|' row: 0.182, 0.198, 0.204, 0.206, 0.207
    (macros meanAbsCorrQone..Qfive)."""
    expected = [0.182, 0.198, 0.204, 0.206, 0.207]
    df = pd.read_csv(
        DATA / "mean_corr_diagnostic" / "cross_quintile_summary.csv"
    ).sort_values("quintile")
    means = df["mean_abs_corr_mean"].tolist()
    for g, (e, a) in enumerate(zip(expected, means), start=1):
        assert math.isclose(a, e, abs_tol=ABS_TOL_FRAC), (
            f"Manuscript tab:percolation 'Mean |rho_ij|' row claims mean |rho_ij| (Q{g})={e:.3f} "
            f"(macro meanAbsCorr) but data shows {a:.4f}"
        )


def test_meancorr_residual_significance_at_every_polynomial_degree():
    """subsec:threshold_vs_mean: 'Polynomials of degree one to five leave
    residual endpoint contrasts between 27% and 53% of that of rho_c. At every
    degree, the residual endpoint contrast and the residual Q1--Q2 step are
    significant, whereas each of the three residual steps above Q2 is not
    significant at one degree or more. The residual Q2--Q3 step is significant
    at no degree.' A contrast is significant as subsec:inference defines it: the
    one-sided HAC t and the block bootstrap both reject at 5%, on the
    per-window differences of the residuals. Recomputed from the per-window
    cells."""
    mc = pd.read_csv(DATA / "mean_corr_diagnostic" / "per_window_mean_corr.csv",
                     parse_dates=["date"])
    parts = []
    for g in (1, 2, 3, 4, 5):
        head = _per_q("korea_headline", g)["rho_c"].reset_index()
        parts.append(head.merge(mc[mc["quintile"] == g][["date", "mean_abs_corr"]],
                                on="date").assign(quintile=g))
    long = pd.concat(parts, ignore_index=True).dropna()
    raw = long.groupby("quintile")["rho_c"].mean().sort_index()
    def significant(diff: np.ndarray) -> bool:
        s = hac_block_stats(diff)
        return s["nw_t"] > 1.645 and s["block_boot_p_one_sided"] < 0.05

    shares = []
    upper_sig = {(2, 3): [], (3, 4): [], (4, 5): []}
    for deg in range(1, 6):
        fit = np.polyval(np.polyfit(long["mean_abs_corr"], long["rho_c"], deg),
                         long["mean_abs_corr"])
        rm = (long["rho_c"] - fit).groupby(long["quintile"]).mean().sort_index().tolist()
        shares.append((rm[-1] - rm[0]) / float(raw.iloc[-1] - raw.iloc[0]))
        cells = long.assign(resid=long["rho_c"] - fit).pivot(
            index="date", columns="quintile", values="resid")
        for lo, hi in ((1, 5), (1, 2)):
            assert significant((cells[hi] - cells[lo]).to_numpy()), (
                f"Manuscript claims the residual Q{lo}-Q{hi} contrast is significant "
                f"at every degree (subsec:threshold_vs_mean) but degree {deg} "
                "does not reject on both statistics"
            )
        for lo, hi in upper_sig:
            upper_sig[(lo, hi)].append(significant((cells[hi] - cells[lo]).to_numpy()))
    for (lo, hi), sig in upper_sig.items():
        assert not all(sig), (
            f"Manuscript claims no residual step above Q2 is significant at every "
            f"degree (subsec:threshold_vs_mean) but Q{lo}-Q{hi} is at all five"
        )
    assert not any(upper_sig[(2, 3)]), (
        "Manuscript claims the residual Q2-Q3 step is significant at no degree "
        "(subsec:threshold_vs_mean) but it is at degrees "
        f"{[d + 1 for d, s in enumerate(upper_sig[(2, 3)]) if s]}"
    )
    for claimed, got in ((27, min(shares)), (53, max(shares))):
        assert math.isclose(got * 100, claimed, abs_tol=0.5), (
            f"Manuscript prints the degree range of the residual share as 27% to "
            f"53% (subsec:threshold_vs_mean) but data gives {got:.1%}"
        )


def test_meancorr_residual_gradient_is_a_fraction_of_the_raw_one():
    """subsec:threshold_vs_mean: 'A cubic regression of rho_c on the mean
    |rho_ij| over all cells leaves residuals whose endpoint contrast is
    rhoCResidGapPctOfRaw of that of rho_c and significant (HAC t =
    tHacMeanCorrResidQoneQfive).' The abstract: 'a cubic regression of the
    threshold on that mean leaves 33% of the threshold's difference, a
    significant remainder'. sec:introduction and
    sec:conclusion state the same remainder.
    Recomputed here from the per-window cells, independently of
    scripts/scaling/run_mean_corr_diagnostic.py, which writes the same quantity into
    coupling.csv for the macro."""
    mc = pd.read_csv(DATA / "mean_corr_diagnostic" / "per_window_mean_corr.csv",
                     parse_dates=["date"])
    parts = []
    for g in (1, 2, 3, 4, 5):
        head = _per_q("korea_headline", g)["rho_c"].reset_index()
        mine = mc[mc["quintile"] == g][["date", "mean_abs_corr"]]
        parts.append(head.merge(mine, on="date").assign(quintile=g))
    long = pd.concat(parts, ignore_index=True).dropna()

    X = np.vander(long["mean_abs_corr"].to_numpy(), 4)  # cubic in mean|rho|
    beta, *_ = np.linalg.lstsq(X, long["rho_c"].to_numpy(), rcond=None)
    long = long.assign(resid=long["rho_c"].to_numpy() - X @ beta)

    rm = long.groupby("quintile")["resid"].mean().sort_index()
    raw = long.groupby("quintile")["rho_c"].mean().sort_index()
    raw_gap = float(raw.iloc[-1] - raw.iloc[0])
    resid_gap = float(rm.iloc[-1] - rm.iloc[0])

    per_window = long.pivot(index="date", columns="quintile", values="resid")
    ep = hac_block_stats((per_window[5] - per_window[1]).to_numpy())
    assert ep["nw_t"] > 1.645 and ep["block_boot_p_one_sided"] < 0.05, (
        "Manuscript claims the cubic's residual endpoint contrast is significant "
        f"(subsec:threshold_vs_mean, abstract) but HAC t={ep['nw_t']:.2f}, "
        f"block bootstrap p={ep['block_boot_p_one_sided']:.3f}"
    )
    printed_t = float(_params_macros()["tHacMeanCorrResidQoneQfive"])
    assert math.isclose(ep["nw_t"], printed_t, abs_tol=0.05), (
        f"Manuscript prints the residual endpoint HAC t as {printed_t} "
        f"(tHacMeanCorrResidQoneQfive, subsec:threshold_vs_mean) but data "
        f"gives {ep['nw_t']:.2f}"
    )
    share = resid_gap / raw_gap
    claimed = float(_params_macros()["rhoCResidGapPctOfRaw"].replace(r"\%", ""))
    assert math.isclose(share * 100, claimed, abs_tol=0.5), (
        f"Manuscript prints the residual endpoint gradient as {claimed}% of the "
        f"raw endpoint gap (rhoCResidGapPctOfRaw, subsec:threshold_vs_mean) but "
        f"data shows {share:.1%}"
    )
    coupling = pd.read_csv(DATA / "mean_corr_diagnostic" / "coupling.csv").iloc[0]
    assert math.isclose(float(coupling["resid_gap_share_of_rho_c_gap"]), share,
                        abs_tol=0.005), (
        f"coupling.csv reports a residual share of "
        f"{float(coupling['resid_gap_share_of_rho_c_gap']):.4f} but recomputing "
        f"it from the per-window cells gives {share:.4f}; the macro "
        f"subsec:threshold_vs_mean prints would be stale"
    )


# ---------------------------------------------------------------------------
# tab:percolation / subsec:researcher_choices: susceptibility-peak robustness (chi-peak rho_c)
# ---------------------------------------------------------------------------

def _susceptibility_summary() -> pd.DataFrame:
    return pd.read_csv(DATA / "susceptibility" / "rolling_percolation_summary.csv")


def _susceptibility_per_q(g: int) -> pd.DataFrame:
    return pd.read_csv(DATA / "susceptibility" / f"rolling_percolation_Q{g}.csv",
                       parse_dates=["date"]).set_index("date")


def test_tab1_chi_peak_means():
    """tab:percolation 'Mean rho_c^chi' row: 0.583, 0.614, 0.623, 0.623, 0.631."""
    expected = [0.583, 0.614, 0.623, 0.623, 0.631]
    df = _susceptibility_summary().sort_values("size_group")
    actual = df["rho_c_chi_mean"].tolist()
    for g, (e, a) in enumerate(zip(expected, actual), start=1):
        assert math.isclose(a, e, abs_tol=ABS_TOL_RHO), (
            f"Manuscript claims chi-peak bar(rho_c^chi)(Q{g})={e:.3f} (tab:percolation) "
            f"but data shows {a:.4f}"
        )


def test_chi_peak_monotone_ordering():
    """subsec:korean_middle_steps: 'The Q3--Q4 step is not significant ...
    when the threshold is rho_c^chi, whose Q3 and Q4 means are equal at three
    decimals'; tab:robustness_contrast prints its endpoint contrast, 0.047;
    tab:robustness_summary and
    tab:robustness_contrast mark the row 'Q3--Q4 equal', read at the three
    decimals the summary table prints."""
    df = _susceptibility_summary().sort_values("size_group")
    means = [float(f"{m:.3f}") for m in df["rho_c_chi_mean"]]
    others = [(a, b) for i, (a, b) in enumerate(zip(means, means[1:]), start=1) if i != 3]
    assert all(a < b for a, b in others) and means[2] == means[3], (
        f"Manuscript claims the chi-peak means rise at every step except Q3--Q4, "
        f"where they are equal at three decimals (tab:robustness_contrast), but they "
        f"are {means}"
    )


def test_chi_peak_q5_q1_gap_significance():
    """subsec:main_result: rho_c^chi has 'an endpoint contrast of 0.045'. Earlier
    wording that no current sentence carries: 'the Q5-Q1 endpoint preserved
    under the autocorrelation-robust inference of subsec:main_result'. Read from the same Newey-West HAC statistic as the headline."""
    a = _susceptibility_per_q(1)["rho_c_chi"]
    b = _susceptibility_per_q(5)["rho_c_chi"]
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    joined.columns = ["a", "b"]
    diff = (joined["b"] - joined["a"]).values
    nw_t = hac_block_stats(diff)["nw_t"]
    assert nw_t > 1.96, (
        f"Manuscript claims the chi-peak Q5-Q1 endpoint is preserved under "
        f"autocorrelation-robust inference (subsec:researcher_choices) but HAC t={nw_t:.2f}"
    )


def test_chi_peak_top_flattens():
    """subsec:researcher_choices: 'The Q3--Q4 and Q4--Q5 steps of rho_c^chi are not
    significant, at HAC t = 0.25 and t = 0.63' (\\tHacChiQthreeQfour,
    \\tHacChiQfourQfive). A step is significant only when both the HAC t and
    the block bootstrap reject, so a HAC t below the one-sided 5% critical value
    suffices."""
    for lo, hi in ((3, 4), (4, 5)):
        a = _susceptibility_per_q(lo)["rho_c_chi"]
        b = _susceptibility_per_q(hi)["rho_c_chi"]
        joined = pd.concat([a, b], axis=1, join="inner").dropna()
        joined.columns = ["a", "b"]
        diff = (joined["b"] - joined["a"]).values
        nw_t = hac_block_stats(diff)["nw_t"]
        assert nw_t <= 1.645, (
            f"Manuscript claims the chi-peak Q{lo}-Q{hi} step is not significant "
            f"(subsec:researcher_choices) but HAC t={nw_t:.2f}"
        )


def test_chi_peak_offset_above_curvature():
    """subsec:researcher_choices: 'The susceptibility-peak threshold rho_c^chi
    lies \\rhoCChiOffsetMin to \\rhoCChiOffsetMax above rho_c, more in the two
    smallest quintiles than in the two largest, which flattens its rise relative
    to that of rho_c.'"""
    df = _susceptibility_summary().sort_values("size_group")
    offsets = (df["rho_c_chi_mean"] - df["rho_c_mean"]).tolist()
    for g, off in enumerate(offsets, start=1):
        assert 0.03 <= off <= 0.07, (
            f"Manuscript claims chi-peak offset is 0.047-0.059 in every quintile "
            f"(subsec:researcher_choices) but Q{g} shows offset={off:.4f}"
        )
    assert min(offsets[:2]) > max(offsets[3:]), (
        f"Manuscript claims the chi-peak offset is larger in the two smallest "
        f"quintiles than in the two largest (subsec:researcher_choices) but "
        f"the offsets are {[round(o, 4) for o in offsets]}"
    )


# ---------------------------------------------------------------------------
# subsec:null_robustness: noise checks (raw, Marchenko-Pastur clipping, shuffled)
# ---------------------------------------------------------------------------

def test_null_rmt_cleaned_means():
    """tab:robustness_summary: 'Marchenko-Pastur clipping: 0.497, 0.533, 0.543,
    0.553, 0.565'."""
    expected = [0.497, 0.533, 0.543, 0.553, 0.565]
    actual = _q_means("rmt_filtered")
    for g, (e, a) in enumerate(zip(expected, actual), start=1):
        assert math.isclose(a, e, abs_tol=ABS_TOL_RHO), (
            f"Manuscript claims Marchenko-Pastur-clipped bar(rho_c)(Q{g})={e:.3f} (tab:robustness_summary) "
            f"but data shows {a:.4f}"
        )


def test_null_rmt_gap():
    """tab:robustness_contrast: 'Marchenko--Pastur clipping +0.068', and
    subsec:main_result: 'Marchenko--Pastur clipping keeps the five means ordered
    and widens the endpoint contrast'."""
    means = _q_means("rmt_filtered")
    gap = means[4] - means[0]
    assert math.isclose(gap, 0.068, abs_tol=ABS_TOL_RHO), (
        f"Manuscript claims Marchenko-Pastur-clipped Q5-Q1 gap +0.068 "
        f"(tab:robustness_contrast) but data shows {gap:.4f}"
    )
    head = _q_means("korea_headline")
    head_gap = head[4] - head[0]
    assert gap > head_gap, (
        f"Manuscript claims the clipping 'widens the endpoint contrast' "
        f"(subsec:main_result) but the clipped gap {gap:.4f} does not exceed "
        f"the headline gap {head_gap:.4f}"
    )


def test_null_shuffled_band_and_reversal():
    """tab:robustness_summary: 'Shuffled-returns null: 0.376, 0.376, 0.374, 0.371, 0.368';
    tab:robustness_contrast: 'Shuffled-returns null -0.008 reversed'.
    subsec:null_robustness: 'the mean rho_c of every
    quintile lies between 0.368 and 0.376'."""
    expected = [0.376, 0.376, 0.374, 0.371, 0.368]
    df = pd.read_csv(DATA / "null_shuffled" / "cross_quintile_summary.csv").sort_values("quintile")
    actual = df["across_seed_mean"].tolist()
    for g, (e, a) in enumerate(zip(expected, actual), start=1):
        assert math.isclose(a, e, abs_tol=ABS_TOL_RHO), (
            f"Manuscript claims shuffled bar(rho_c)(Q{g})={e:.3f} (tab:robustness_summary) "
            f"but data shows {a:.4f}"
        )
    gap = actual[4] - actual[0]
    assert math.isclose(gap, -0.008, abs_tol=ABS_TOL_RHO), (
        f"Manuscript claims shuffled Q5-Q1 gap ~ -0.008 (tab:robustness_contrast) "
        f"but data shows {gap:.4f}"
    )


# ---------------------------------------------------------------------------
# tab:robustness_summary / subsec:size_confounds: KOSPI-only confound control
# ---------------------------------------------------------------------------

def test_kospi_only_means():
    """tab:robustness_summary: 'KOSPI-only universe: 0.514, 0.539, 0.540, 0.535, 0.582'."""
    expected = [0.514, 0.539, 0.540, 0.535, 0.582]
    actual = _q_means("kospi_only")
    for g, (e, a) in enumerate(zip(expected, actual), start=1):
        assert math.isclose(a, e, abs_tol=ABS_TOL_RHO), (
            f"Manuscript claims KOSPI-only bar(rho_c)(Q{g})={e:.3f} (tab:robustness_summary) "
            f"but data shows {a:.4f}"
        )


def test_appB1_kospi_gap_matches_headline():
    """tab:robustness_contrast: 'KOSPI-only universe +0.068 Q3--Q4 reversed'."""
    means = _q_means("kospi_only")
    gap = means[4] - means[0]
    assert math.isclose(gap, 0.068, abs_tol=ABS_TOL_RHO), (
        f"Manuscript claims KOSPI-only Q5-Q1 gap = 0.068 (tab:robustness_contrast) "
        f"but data shows {gap:.4f}"
    )


# ---------------------------------------------------------------------------
# tab:robustness_summary / subsec:size_confounds: within-sector confound control
# ---------------------------------------------------------------------------

def test_within_sector_means():
    """tab:robustness_summary: 'Within-sector quintiles: 0.529, 0.554, 0.566, 0.569, 0.577'."""
    expected = [0.529, 0.554, 0.566, 0.569, 0.577]
    actual = _q_means("within_sector")
    for g, (e, a) in enumerate(zip(expected, actual), start=1):
        assert math.isclose(a, e, abs_tol=ABS_TOL_RHO), (
            f"Manuscript claims within-sector bar(rho_c)(Q{g})={e:.3f} (tab:robustness_summary) "
            f"but data shows {a:.4f}"
        )


def test_appB2_within_sector_gap():
    """tab:robustness_contrast: 'Within-sector quintiles +0.048'."""
    means = _q_means("within_sector")
    gap = means[4] - means[0]
    assert math.isclose(gap, 0.048, abs_tol=ABS_TOL_RHO), (
        f"Manuscript claims within-sector Q5-Q1 gap = 0.048 (tab:robustness_contrast) "
        f"but data shows {gap:.4f}"
    )


def test_within_sector_gap_pct_of_headline_macro():
    """subsec:main_result: 'Ranking size within sectors keeps
    \\rhoCGapWithinSectorPctOfHeadline of the endpoint contrast'. The generated percent
    macro must equal round(rhoCGapWithinSector / rhoCGapHeadline * 100)."""
    gap_head = _q_means("korea_headline")[4] - _q_means("korea_headline")[0]
    gap_ws = _q_means("within_sector")[4] - _q_means("within_sector")[0]
    expected_pct = round(gap_ws / gap_head * 100)
    params = (Path(__file__).resolve().parent / "paper"
              / "params.tex").read_text()
    m = re.search(r"\\newcommand\{\\rhoCGapWithinSectorPctOfHeadline\}\{(\d+)\\%", params)
    assert m, "macro \\rhoCGapWithinSectorPctOfHeadline not found in params.tex"
    actual_pct = int(m.group(1))
    assert actual_pct == expected_pct, (
        f"Manuscript says the within-sector gap retains "
        f"\\rhoCGapWithinSectorPctOfHeadline (={actual_pct}%) of the headline "
        f"(subsec:main_result) but the data ratio = {expected_pct}%"
    )


def test_appB2_within_sector_n_windows():
    """tab:robustness_summary note (b): the within-sector control runs over
    \nWindowsWithinSector windows, the headline grid. Both the macro and the
    headline count are read rather than retyped, so the claim that the control
    runs on the headline grid is what is asserted."""
    claimed = int(_params_macros()["nWindowsWithinSector"])
    df = _summary("within_sector")
    assert (df["n_windows"] == claimed).all(), (
        f"tab:robustness_summary note (b) prints {claimed} within-sector windows "
        f"but data shows {df['n_windows'].tolist()}"
    )
    assert claimed == int(_params_macros()["nWindowsHeadline"]), (
        f"Manuscript calls the within-sector grid the headline grid "
        f"(tab:robustness_summary note (b)) but it has {claimed} windows against "
        f"the headline's {_params_macros()['nWindowsHeadline']}"
    )


def test_within_sector_sector_size_robust():
    """subsec:size_confounds: 'The five within-sector means stay ordered at
    sector-size floors of 15 and 25 stocks. The endpoint contrast stays between
    \\rhoCGapWithinSectorFloorMin and \\rhoCGapWithinSectorFloorMax at these two
    floors and at the check's own floor.' The band is read
    off the macros the sentence prints rather than retyped here."""
    df = pd.read_csv(DATA / "within_sector" / "sector_size_sensitivity.csv")
    gaps = []
    for floor in sorted(df["min_sector_size"].unique()):
        means = (df[df["min_sector_size"] == floor]
                 .sort_values("size_group")["rho_c_mean"].tolist())
        assert all(a < b for a, b in zip(means, means[1:])), (
            f"Manuscript claims the within-sector design leaves the five means "
            f"ordered at every sector-size floor (subsec:size_confounds) "
            f"but floor={floor} gives {means}"
        )
        gaps.append(means[-1] - means[0])
    lo, hi = min(gaps), max(gaps)
    macros = _params_macros()
    claimed_lo = float(macros["rhoCGapWithinSectorFloorMin"])
    claimed_hi = float(macros["rhoCGapWithinSectorFloorMax"])
    assert math.isclose(lo, claimed_lo, abs_tol=ABS_TOL_RHO) and \
        math.isclose(hi, claimed_hi, abs_tol=ABS_TOL_RHO), (
        f"Manuscript prints a within-sector Q5-Q1 gap band of "
        f"[{claimed_lo}, {claimed_hi}] across sector-size floors "
        f"(subsec:size_confounds) but data shows [{lo:.4f}, {hi:.4f}]"
    )


# ---------------------------------------------------------------------------
# subsec:residuals, tab:residual_steps: the gradient on factor residuals
# ---------------------------------------------------------------------------

_STEP_PAIRS = ("Q2-Q1", "Q3-Q2", "Q4-Q3", "Q5-Q4")


def _residual_arms() -> list[str]:
    """Every residualization arm data/ holds, derived rather than listed: the
    paragraph of subsec:residuals quantifies over this set."""
    import generate_params

    return [a.name for a in generate_params._residual_arm_dirs()]


def _steps(name: str) -> dict[str, tuple[float, float]]:
    """Pair -> (cross-window mean paired difference, Newey-West t)."""
    pairs = pd.read_csv(DATA / name / "pair_significance.csv").set_index("pair")
    return {pair: (float(pairs.loc[pair, "mean_diff"]), float(pairs.loc[pair, "nw_t"]))
            for pair in (*_STEP_PAIRS, "Q5-Q1")}


def _rise_q1_to_q4(name: str) -> float:
    st = _steps(name)
    return sum(st[pair][0] for pair in _STEP_PAIRS[:3])


def test_residualization_removes_most_of_the_q1_to_q4_rise():
    """subsec:residuals: "Removing common-factor returns removes most of the
    rise from Q1 to Q4, whereas the residual Q4--Q5 step depends on the factor
    construction (Table tab:residual_steps). The rise from Q1 to Q4 is +0.047 in
    raw returns and lies between -0.012 and +0.011 in the residuals of each of
    the four factor constructions." Earlier
    wording that no current sentence carries: the rise from Q1 to Q4 'carries
    the gradient'.

    Three claims, each over the arm set data/ holds: the Q1-to-Q4 rise carries
    most of the raw-return endpoint contrast, every residual rise keeps less
    than half of it, and the printed range covers every arm (its ends round away
    from the data)."""
    macros = _params_macros()
    head = _steps("korea_headline")
    rise_head = _rise_q1_to_q4("korea_headline")
    assert rise_head > 0.5 * head["Q5-Q1"][0], (
        "Manuscript claims the rise from Q1 to Q4 carries the gradient "
        f"(subsec:residuals) but it is {rise_head:+.4f} of an endpoint "
        f"contrast of {head['Q5-Q1'][0]:+.4f}"
    )
    assert math.isclose(rise_head, float(macros["rhoCRiseQoneQfourHeadline"]),
                        abs_tol=0.0005), (
        f"Manuscript prints a total-return Q1-to-Q4 rise of "
        f"{macros['rhoCRiseQoneQfourHeadline']} (subsec:residuals) but data "
        f"shows {rise_head:+.4f}"
    )
    rises = {arm: _rise_q1_to_q4(arm) for arm in _residual_arms()}
    assert rises, "subsec:residuals reports residualization arms, but data/ holds none"
    for arm, rise in rises.items():
        assert rise < 0.5 * rise_head, (
            "Manuscript claims removing common factors removes most of the rise "
            f"from Q1 to Q4 (subsec:residuals) but {arm} keeps {rise:+.4f} of "
            f"{rise_head:+.4f}"
        )
    lo, hi = (float(macros["rhoCRiseQoneQfourResidMin"]),
              float(macros["rhoCRiseQoneQfourResidMax"]))
    assert lo <= min(rises.values()) and max(rises.values()) <= hi, (
        f"Manuscript prints the residual Q1-to-Q4 rise as lying between {lo:+.3f} "
        f"and {hi:+.3f} (subsec:residuals) but the arms span {rises}"
    )
    assert min(rises.values()) - lo < 0.001 and hi - max(rises.values()) < 0.001, (
        f"The printed range [{lo:+.3f}, {hi:+.3f}] is wider than the arms' span "
        f"{rises} by more than its rounding"
    )


def test_residual_endpoint_depends_on_the_construction():
    """subsec:residuals: "It is +0.006 (HAC t = 1.3) after the cap-weighted
    market return and -0.011 (HAC t = -3.8) after the cap-weighted market and
    sector returns. The contrast after the cap-weighted market return is not
    significant. The endpoint contrast is +0.058
    after the equal-weighted market return and +0.021 after the equal-weighted
    market and sector returns. The Q4--Q5 step of the equal-weighted residuals
    is +0.049 after the market return and +0.026 after the market and sector
    returns, compared with +0.008 in raw returns. The endpoint contrast of each
    residual is the sum of its Q4--Q5 step and its rise from Q1 to Q4."
    subsec:main_result: "and the Q4--Q5 step not".

    The equal-weighted half is a quantifier over the arms whose directory name
    carries the ew token, derived from data/. "Each residual" ranges over every
    arm data/ holds."""
    macros = _params_macros()
    cw_market, cw_sector = _steps("market_residual")["Q5-Q1"], _steps("market_sector_residual")["Q5-Q1"]
    assert cw_market[0] > 0 and cw_market[1] <= 1.645, (
        "Manuscript claims the cap-weighted market residual leaves a positive "
        f"endpoint contrast that is not significant (subsec:residuals) but data "
        f"shows {cw_market[0]:+.4f} at HAC t={cw_market[1]:.2f}"
    )
    assert cw_sector[0] < 0 and cw_sector[1] < -1.645, (
        "Manuscript prints a negative endpoint contrast for the cap-weighted "
        f"market-and-sector residual (subsec:residuals) but data shows "
        f"{cw_sector[0]:+.4f} at HAC t={cw_sector[1]:.2f}"
    )
    for name, (gap, t) in (("MktResid", cw_market), ("MktSecResid", cw_sector)):
        assert math.isclose(gap, float(macros[f"rhoCGap{name}"]), abs_tol=0.0005) and \
            math.isclose(t, float(macros[f"tHac{name}QoneQfive"]), abs_tol=0.05), (
            f"Manuscript prints {macros[f'rhoCGap{name}']} (t = "
            f"{macros[f'tHac{name}QoneQfive']}) for the {name} arm "
            f"(subsec:residuals) but data shows {gap:+.4f} (t = {t:.2f})"
        )
    head_top = _steps("korea_headline")["Q5-Q4"]
    assert abs(head_top[1]) < 1.645, (
        "Manuscript claims the raw returns do not show a Q4-Q5 step "
        f"(subsec:main_result) but the headline step is {head_top[0]:+.4f} at "
        f"HAC t={head_top[1]:.2f}"
    )
    ew = [arm for arm in _residual_arms() if "ew" in arm.split("_")]
    assert ew, "subsec:residuals reports equal-weighted arms, but data/ holds none"
    for arm in _residual_arms():
        st = _steps(arm)
        gap, top, rise = st["Q5-Q1"][0], st["Q5-Q4"][0], _rise_q1_to_q4(arm)
        assert math.isclose(gap, top + rise, abs_tol=0.0005), (
            "Manuscript claims the endpoint contrast of each residual is the sum "
            "of its Q4--Q5 step and its rise from Q1 to Q4 "
            f"(subsec:residuals) but {arm} gives an endpoint of {gap:+.4f} with "
            f"a Q4--Q5 step of {top:+.4f} and a rise of {rise:+.4f}"
        )


def test_residual_step_table_matches_the_pair_files():
    """tab:residual_steps prints every step and endpoint of every residualization
    arm data/ holds, and the raw-return row's steps. Each printed macro must
    match the pair file the generator reads and be typeset: an arm whose row is
    missing is an arm the paragraph's quantifiers range over unseen."""
    import generate_params

    macros = _params_macros()
    text = _manuscript_text()
    names = ("QoneQtwo", "QtwoQthree", "QthreeQfour", "QfourQfive")
    problems = []
    rows = {"Headline": "korea_headline",
            **{generate_params._external_factor_stem(a): a for a in _residual_arms()}}
    for stem, arm in rows.items():
        st = _steps(arm)
        checks = [(f"rhoCStep{stem}{n}", st[pair][0], 0.0005)
                  for pair, n in zip(_STEP_PAIRS, names)]
        if stem != "Headline":
            checks += [(f"tHac{stem}{n}", st[pair][1], 0.05)
                       for pair, n in zip(_STEP_PAIRS, names)]
            checks += [(f"rhoCGap{stem}", st["Q5-Q1"][0], 0.0005),
                       (f"tHac{stem}QoneQfive", st["Q5-Q1"][1], 0.05)]
        for macro, value, tol in checks:
            if macro not in macros:
                problems.append(f"{arm}: \\{macro} not generated")
            elif not _is_printed(macro, text):
                problems.append(f"{arm}: \\{macro} generated but not printed")
            elif not math.isclose(float(macros[macro]), value, abs_tol=tol):
                problems.append(f"{arm}: \\{macro} = {macros[macro]} but data "
                                f"shows {value:+.4f}")
    assert not problems, (
        "tab:residual_steps does not carry every arm's steps as data holds them:\n  "
        + "\n  ".join(problems)
    )


# ---------------------------------------------------------------------------
# subsec:limits (Limitations): the rho_c level band is horizon-dependent
# ---------------------------------------------------------------------------

def test_rho_c_level_band_shifts_with_horizon():
    """subsec:limits: 'The claim concerns the direction of the rise at a fixed
    window length and not the level of rho_c, which falls as the window
    lengthens'; tab:robustness_summary prints the T = 30 band
    barRhoCWthirtyQone--barRhoCWthirtyQfive and the T = 252 band
    barRhoCWtwofiftyQone--barRhoCWtwofiftyQfive. The two bands are read off
    the window-length sweep, which carries ~240 windows behind each horizon, and the
    claim is that they do not overlap: the level moves with the estimation horizon by
    more than the cross-quintile spread at either end."""
    df = pd.read_csv(DATA / "window_sensitivity" / "summary.csv")
    short = df[df["window_days"] == 30].sort_values("size_group")["rho_c_mean"]
    long = df[df["window_days"] == 252].sort_values("size_group")["rho_c_mean"]
    assert not short.empty and not long.empty, (
        "Manuscript quotes the T=30 and T=252 rho_c bands (tab:robustness_summary) but "
        "window_sensitivity/summary.csv holds no such horizon"
    )
    assert float(long.max()) < float(short.min()), (
        f"Manuscript claims the rho_c level band shifts with the estimation horizon "
        f"(subsec:limits) but the T=252 band "
        f"[{long.min():.3f}, {long.max():.3f}] overlaps the T=30 band "
        f"[{short.min():.3f}, {short.max():.3f}]"
    )


def test_shuffled_null_across_seed_sd_bound():
    """tab:robustness_summary note: 'The shuffled-returns row averages 20
    shuffles, one on each draw. The largest standard deviation of a quintile's
    mean across the shuffles is 0.0004.' The note is a resolution claim about the shipped
    ensemble -- it is what licenses reading the null's residual gap as a
    measurement rather than as seed noise -- so it is asserted on the per-seed
    file the macro is generated from."""
    df = pd.read_csv(DATA / "null_shuffled" / "summary_per_seed_per_quintile.csv")
    n_seeds = df["seed"].nunique()
    assert n_seeds == int(_params_macros()["nDrawSeeds"]), (
        f"Manuscript claims one shuffle on each of the {_params_macros()['nDrawSeeds']} draws "
        f"(tab:robustness_summary note) but data holds {n_seeds}"
    )
    sd_max = float(df.groupby("quintile")["mean_rho_c"].std(ddof=1).max())
    assert math.isclose(sd_max, 0.0004, abs_tol=0.00005), (
        f"Manuscript prints a largest per-quintile across-seed standard deviation "
        f"of 0.0004 (tab:robustness_summary note) but data shows {sd_max:.5f}"
    )


# ---------------------------------------------------------------------------
# subsec:taiwan: Taiwan re-run
# ---------------------------------------------------------------------------

def test_taiwan_q1_lt_q5():
    """subsec:taiwan: 'The Taiwanese endpoint contrast is 0.112'."""
    means = _q_means("taiwan")
    assert means[4] > means[0], (
        f"Manuscript claims Taiwan rho_c(Q5) > rho_c(Q1) (subsec:taiwan) "
        f"but data shows Q1={means[0]:.4f}, Q5={means[4]:.4f}"
    )
    gap = means[4] - means[0]
    assert math.isclose(gap, 0.112, abs_tol=ABS_TOL_RHO), (
        f"Manuscript claims Taiwan Q5-Q1 gap = 0.112 (subsec:taiwan) "
        f"but data shows gap = {gap:.4f}"
    )


def test_taiwan_rerun_cross_market_comparisons_hold():
    """subsec:taiwan: 'Q1's rho_c is below Q5's in \\fracWindowsQoneLtQfiveTw of
    the Taiwanese windows and in \\fracWindowsQoneLtQfiveKr of the Korean ones.'
    The sentence orders the two markets' shares, a comparison rather than a
    bound number, so no macro check reaches it."""
    macros = _params_macros()
    fracs = {}
    for market, macro in (("taiwan", "fracWindowsQoneLtQfiveTw"),
                          ("korea_headline", "fracWindowsQoneLtQfiveKr")):
        a = _per_q(market, 1)["rho_c"]
        b = _per_q(market, 5)["rho_c"]
        joined = pd.concat([a, b], axis=1, join="inner").dropna()
        joined.columns = ["a", "b"]
        fracs[market] = float((joined["b"] > joined["a"]).mean())
        printed = float(macros[macro].rstrip("%").replace("\\", ""))
        assert math.isclose(printed, fracs[market] * 100, abs_tol=0.5), (
            f"subsec:taiwan prints \\{macro} = {macros[macro]} but "
            f"{market} shows {fracs[market]:.1%}"
        )
    assert fracs["taiwan"] > fracs["korea_headline"], (
        "subsec:taiwan claims the Taiwanese ordering holds in a higher fraction "
        f"of windows than Korea's but Taiwan shows {fracs['taiwan']:.1%} "
        f"against Korea's {fracs['korea_headline']:.1%}"
    )


def test_taiwan_interior_monotone_q1_to_q4():
    """subsec:taiwan: 'Each of the three steps through Q4 is significant';
    tab:taiwan_rerun prints the means 0.520, 0.579, 0.607, 0.635."""
    means = _q_means("taiwan")
    for g in range(3):
        assert means[g] < means[g + 1], (
            f"Manuscript claims Taiwan Q1<Q2<Q3<Q4 (subsec:taiwan) "
            f"but Q{g+1}={means[g]:.4f} >= Q{g+2}={means[g+1]:.4f}"
        )


def test_taiwan_chi_peak_row_matches_the_curvature_shape():
    """tab:taiwan_rerun now carries a Mean rho_c^chi row, as tab:percolation
    does for Korea, and subsec:taiwan claims 'The susceptibility-peak threshold
    rho_c^chi also rises through Q4 and falls slightly into Q5'."""
    df = pd.read_csv(DATA / "taiwan" / "rolling_percolation_summary.csv"
                     ).sort_values("size_group")
    chi = df["rho_c_chi_mean"].tolist()
    expected = [0.579, 0.641, 0.657, 0.671, 0.666]
    for g, (a, e) in enumerate(zip(chi, expected), start=1):
        assert math.isclose(a, e, abs_tol=ABS_TOL_RHO), (
            f"tab:taiwan_rerun prints bar(rho_c^chi)(Q{g})={e:.3f} but "
            f"data shows {a:.4f}"
        )
    for g in range(3):
        assert chi[g] < chi[g + 1], (
            f"Manuscript claims the Taiwanese chi-peak means rise 'through Q4' "
            f"(subsec:taiwan) but Q{g+1}={chi[g]:.4f} >= Q{g+2}={chi[g+1]:.4f}"
        )
    assert chi[4] < chi[3], (
        f"Manuscript claims the Taiwanese chi-peak means 'also ... [fall] "
        f"slightly into Q5' (subsec:taiwan) but "
        f"Q4={chi[3]:.4f} <= Q5={chi[4]:.4f}"
    )


def test_window_sensitivity_monotone_at_every_W():
    """subsec:main_result: 'The five means stay ordered at each window length and
    observation requirement.' tab:robustness_summary marks every window-length row
    'ordered', which it reads at the three decimals it prints."""
    df = pd.read_csv(DATA / "window_sensitivity" / "summary.csv")
    for w, sub in df.groupby("window_days"):
        sub = sub.sort_values("size_group")
        rhos = [float(f"{r:.3f}") for r in sub["rho_c_mean"]]
        for g in range(4):
            assert rhos[g] < rhos[g + 1], (
                f"Manuscript claims the cross-quintile ordering Q1<Q2<Q3<Q4<Q5 holds "
                f"at every window length (subsec:main_result, tab:robustness_summary) "
                f"but at W={w} the printed Q{g+1}={rhos[g]:.3f} >= Q{g+2}={rhos[g+1]:.3f}"
            )


def test_window_sensitivity_T60_row_is_the_headline():
    """tab:robustness_summary prints the main estimate's row, 'Main estimate,
    T = 60 days', in the slot of the 60-day window length, so the
    window-length family ships no T=60 macros of its own."""
    ws = pd.read_csv(DATA / "window_sensitivity" / "summary.csv")
    sub = ws[ws["window_days"] == 60].sort_values("size_group")["rho_c_mean"].tolist()
    headline = _q_means("korea_headline")
    for g, (a, b) in enumerate(zip(sub, headline), start=1):
        assert math.isclose(a, b, abs_tol=ABS_TOL_RHO), (
            f"Manuscript claims the T=60 window-length row reproduces the headline "
            f"exactly (subsec:researcher_choices, tab:robustness_summary) but Q{g} is "
            f"{a:.4f} in window_sensitivity against {b:.4f} in korea_headline"
        )


def test_window_sensitivity_gap_widens_with_W():
    """subsec:main_result: 'The endpoint contrast widens with the window length,
    from \\rhoCGapWthirty at \\windowDaysWthirty days to \\rhoCGapWtwofifty at
    \\windowDaysWtwofifty days, so the choice of the \\paperWindowDays-day window
    for the main estimate does not maximize the contrast.'
    tab:robustness_contrast prints the contrast at each length."""
    df = pd.read_csv(DATA / "window_sensitivity" / "summary.csv")
    wide = df.pivot(index="window_days", columns="size_group", values="rho_c_mean")
    gap = (wide[5] - wide[1]).sort_index()
    windows = gap.index.tolist()
    for i in range(len(windows) - 1):
        assert gap.iloc[i] < gap.iloc[i + 1], (
            f"Manuscript claims Q5-Q1 gap widens monotonically with W (subsec:main_result) "
            f"but gap(W={windows[i]})={gap.iloc[i]:.4f} >= "
            f"gap(W={windows[i+1]})={gap.iloc[i+1]:.4f}"
        )


def _setting_panel(folder: Path, column: str = "rho_c") -> pd.DataFrame:
    return pd.concat({g: pd.read_csv(folder / f"rolling_percolation_Q{g}.csv",
                                     index_col="date")[column] for g in range(1, 6)},
                     axis=1).dropna()


def _significant(stats: dict) -> bool:
    """subsec:inference: both the HAC t and the block bootstrap reject at 5%."""
    return stats["nw_t"] > 1.645 and stats["block_boot_p_one_sided"] < 0.05


def test_setting_step_verdicts():
    """subsec:main_result, Estimation settings: 'At each window length and
    observation requirement, the five means stay ordered. At each of these
    settings, the endpoint contrast and the Q1--Q2 and Q2--Q3 steps are
    significant ... The Q3--Q4 step
    is significant at each of these settings except the windows of
    \\windowDaysWthirty days. The Q4--Q5 step is significant only on the windows
    of \\windowDaysWhundredtwenty and \\windowDaysWtwofifty days.' The settings
    are derived from the sweep folders, and the lag rule is subsec:checks':
    the bandwidth and the block length rise to one more than the largest lag at
    which two windows share returns. tab:setting_steps prints each HAC t."""
    from fn_percolation.overlap_inference import overlap_span
    macros = _params_macros()
    names = {30: "Wthirty", 120: "Whundredtwenty", 252: "Wtwofifty"}
    settings = []
    for folder in sorted((DATA / "window_sensitivity").glob("W*")):
        w = int(folder.name[1:])
        if w != WINDOW:
            settings.append((names[w], folder, w))
    fracs = sorted(float(f.name[1:]) for f in (DATA / "min_valid_fraction_sensitivity").glob("F*"))
    settings += [("ObsReqLo", DATA / "min_valid_fraction_sensitivity" / f"F{fracs[0]}", WINDOW),
                 ("ObsReqHi", DATA / "min_valid_fraction_sensitivity" / f"F{fracs[-1]}", WINDOW)]
    assert len(settings) == 5, f"tab:setting_steps prints five settings, the sweeps hold {settings}"
    pairs = {"QoneQtwo": (1, 2), "QtwoQthree": (2, 3), "QthreeQfour": (3, 4),
             "QfourQfive": (4, 5), "QoneQfive": (1, 5)}
    for name, folder, w in settings:
        panel = _setting_panel(folder)
        lag = overlap_span(w, STEP) + 1
        verdict = {}
        for key, (lo, hi) in pairs.items():
            st = hac_block_stats((panel[hi] - panel[lo]).to_numpy(), min_lag=lag, min_block=lag)
            verdict[key] = _significant(st)
            printed = float(macros[f"tHac{name}{key}"])
            assert math.isclose(st["nw_t"], printed, abs_tol=0.05), (
                f"tab:setting_steps prints HAC t {printed} for {name} {key} "
                f"but data gives {st['nw_t']:.2f}")
        assert verdict["QoneQfive"] and verdict["QoneQtwo"] and verdict["QtwoQthree"], (
            f"subsec:main_result claims the endpoint contrast and the Q1--Q2 and "
            f"Q2--Q3 steps are significant at every setting, but not at {name}: {verdict}")
        assert verdict["QthreeQfour"] == (name != "Wthirty"), (
            f"subsec:main_result claims the Q3--Q4 step is significant at every setting "
            f"except the 30-day windows, but at {name} it is {verdict['QthreeQfour']}")
        assert verdict["QfourQfive"] == (name in ("Whundredtwenty", "Wtwofifty")), (
            f"subsec:main_result claims the Q4--Q5 step is significant only on the "
            f"120- and 252-day windows, but at {name} it is {verdict['QfourQfive']}")


def test_significant_steps_exceed_their_draw_spread():
    """subsec:researcher_choices (Supplementary Material): 'Every significant
    step is at least \\stepDrawSpreadRatioMin times its spread across the
    draws, so the random draws of stocks alone do not produce it.' Asserted
    over every step and endpoint contrast of both markets that subsec:inference
    finds significant, derived from data/bandwidth_sensitivity/steps.csv."""
    verdicts = pd.read_csv(DATA / "bandwidth_sensitivity" / "steps.csv")
    verdicts = verdicts[verdicts["bandwidth"] == "automatic"]
    printed = float(_params_macros()["stepDrawSpreadRatioMin"])
    ratios = {}
    for market, sub in (("korea", "korea_headline"), ("taiwan", "taiwan")):
        ps = pd.read_csv(DATA / sub / "rolling_percolation_per_seed.csv")
        wide = ps.pivot_table(index=["seed", "date"], columns="quintile", values="rho_c")
        for _, row in verdicts[verdicts["market"] == market].iterrows():
            if not row["significant"]:
                continue
            lo, hi = (int(q[1:]) for q in row["pair"].split("-"))
            sm = (wide[hi] - wide[lo]).groupby(level="seed").mean()
            ratios[(market, row["pair"])] = float(sm.mean() / sm.std(ddof=1))
    assert len(ratios) >= 8, f"expected the significant steps of both markets, got {ratios}"
    worst = min(ratios, key=ratios.get)
    assert printed <= ratios[worst] < printed + 0.1, (
        f"The Supplementary Material prints every significant step as at least "
        f"{printed} times its spread across the draws, but {worst} gives "
        f"{ratios[worst]:.3f}")


def test_within_sector_floor_places_stocks_in_every_quintile():
    """subsec:checks: 'The within-sector check keeps the KRX sectors of at
    least \\minSectorSize stocks, so that each kept sector places at least
    \\minSectorStocksPerQuintile stocks in every quintile.' Decided on the
    assignment that sector_loader.within_sector_quintiles uses, a cut of the
    sector's size ranks into five equal groups, at the smallest kept sector."""
    from fn_percolation.constants import MIN_SECTOR_SIZE
    ranks = pd.Series(range(1, MIN_SECTOR_SIZE + 1), dtype=float)
    groups = pd.qcut(ranks.rank(method="first"), 5, labels=False)
    printed = int(_params_macros()["minSectorStocksPerQuintile"])
    assert groups.value_counts().min() == printed, (
        f"subsec:checks claims a sector of {MIN_SECTOR_SIZE} stocks places at least "
        f"{printed} in every quintile, but the cut gives {groups.value_counts().to_dict()}")


def test_q3q4_power_rule_per_check():
    """subsec:korean_middle_steps: 'The Q3--Q4 step is not significant when the
    threshold is rho_c^chi ... or on windows of \\windowDaysWthirty days, and
    each of these two tests detects a step of the main estimate's size with
    \\detectPowerPct power ... The Q3--Q4 step is not significant either when
    size is ranked within sectors, and neither the Q2--Q3 step nor the Q3--Q4
    step is significant in the KOSPI-only panel, but neither of these two tests
    detects a step of the main estimate's size'. subsec:researcher_choices
    prints the 30-day and rho_c^chi limits as upper bounds below the main
    step. The detection limit is run_weekly_rho_c.py's: (z_0.05 + z_0.80) HAC
    standard errors; the KOSPI-only limits are test_kospi_only_interior_steps_are_unresolved's."""
    from fn_percolation.overlap_inference import overlap_span
    z = 1.645 + 0.842
    macros = _params_macros()
    main = _setting_panel(DATA / "korea_headline")
    main_step = float((main[4] - main[3]).mean())
    printed_main = float(macros["rhoCStepHeadlineQthreeQfour"])
    w30 = _setting_panel(DATA / "window_sensitivity" / "W30")
    lag = overlap_span(30, STEP) + 1
    st30 = hac_block_stats((w30[4] - w30[3]).to_numpy(), min_lag=lag, min_block=lag)
    chi = _setting_panel(DATA / "susceptibility", "rho_c_chi")
    st_chi = hac_block_stats((chi[4] - chi[3]).to_numpy())
    ws = _setting_panel(DATA / "within_sector")
    st_ws = hac_block_stats((ws[4] - ws[3]).to_numpy())
    for label, st, macro in (("30-day", st30, "detectWthirtyQthreeQfour"),
                             ("rho_c^chi", st_chi, "detectChiQthreeQfour")):
        limit = z * st["mean_diff"] / st["nw_t"]
        assert not _significant(st) and limit < main_step, (
            f"subsec:korean_middle_steps claims the {label} Q3--Q4 test is not "
            f"significant and detects the main step {main_step:.4f} with 80% power, "
            f"but its limit is {limit:.4f}")
        printed = float(macros[macro])
        assert limit <= printed < printed_main, (
            f"subsec:researcher_choices prints the {label} limit as at most {printed}, "
            f"below the main step {printed_main}, but the limit is {limit:.4f}")
    limit_ws = z * st_ws["mean_diff"] / st_ws["nw_t"]
    printed_ws = float(macros["detectWithinSectorQthreeQfour"])
    printed_fine = float(macros["rhoCStepHeadlineQthreeQfourFine"])
    assert printed_ws <= limit_ws and main_step <= printed_fine < printed_ws, (
        f"subsec:size_confounds prints the within-sector limit as at least {printed_ws}, "
        f"above the main Q3--Q4 step of {printed_fine}, but the limit is "
        f"{limit_ws:.5f} and the step {main_step:.5f}")
    assert not _significant(st_ws) and limit_ws > main_step, (
        "subsec:korean_middle_steps claims the within-sector Q3--Q4 test does not detect "
        f"the main step {main_step:.4f} with 80% power, but its limit is {limit_ws:.4f}")


def test_window_sensitivity_level_and_top_steps():
    """subsec:researcher_choices: 'The level of $\\bar\\rho_c$ falls as the window
    lengthens'. A claim over the window lengths the sweep ran, read off the
    sweep rather than a list of lengths: every quintile's mean falls from each
    length to the next."""
    df = pd.read_csv(DATA / "window_sensitivity" / "summary.csv")
    wide = df.pivot(index="window_days", columns="size_group",
                    values="rho_c_mean").sort_index()
    lengths = wide.index.tolist()
    for g in wide.columns:
        col = wide[g].tolist()
        assert all(a > b for a, b in zip(col, col[1:])), (
            f"Manuscript claims the level of rho_c falls as the window lengthens "
            f"(subsec:researcher_choices) but Q{g} gives {col} at {lengths}"
        )


def test_min_valid_fraction_not_load_bearing():
    """tab:robustness_contrast prints the endpoint contrast at observation requirements of
    \\minValidFractionLoPct and \\minValidFractionHiPct beside the main
    estimate's at \\minValidFractionPct. subsec:main_result: 'The five means stay ordered at each window length and
    observation requirement.' Strict Q1<...<Q5 and the positive Q5-Q1 gap survive at every
    swept requirement, and the gap is essentially invariant to the requirement. Guards
    scripts/sensitivity/run_pipeline_sensitivity.py ->
    data/min_valid_fraction_sensitivity/summary.csv."""
    df = pd.read_csv(DATA / "min_valid_fraction_sensitivity" / "summary.csv")
    gaps = {}
    for f, sub in df.groupby("min_valid_fraction"):
        sub = sub.sort_values("size_group")
        rhos = sub["rho_c_mean"].tolist()
        for g in range(4):
            assert rhos[g] < rhos[g + 1], (
                f"Headline ordering Q1<...<Q5 (subsec:main_result) should survive the "
                f"observation requirement, but at MIN_VALID_FRACTION={f}, "
                f"Q{g+1}={rhos[g]:.4f} >= Q{g+2}={rhos[g+1]:.4f}"
            )
        gaps[f] = rhos[4] - rhos[0]
        assert gaps[f] > 0, (
            f"Manuscript reports a positive Q5-Q1 rho_c gap (tab:robustness_contrast) but at "
            f"MIN_VALID_FRACTION={f}, gap={gaps[f]:.4f}"
        )
    macros = _params_macros()
    # tab:robustness_contrast and tab:robustness_summary print the lowest and
    # highest swept requirements as rows of their own.
    for f, fname in ((min(gaps), "ObsReqLo"), (max(gaps), "ObsReqHi")):
        sub = df[df["min_valid_fraction"] == f].sort_values("size_group")
        for q, want in zip(("Qone", "Qtwo", "Qthree", "Qfour", "Qfive"), sub["rho_c_mean"]):
            assert math.isclose(float(macros[f"barRhoC{fname}{q}"]), want, abs_tol=0.0005), (
                f"Supplementary Material tab:robustness_summary prints \\barRhoC{fname}{q} = "
                f"{macros[f'barRhoC{fname}{q}']} but data shows {want:.4f}"
            )
        assert math.isclose(float(macros[f"rhoCGap{fname}"]), gaps[f], abs_tol=0.0005), (
            f"tab:robustness_contrast prints \\rhoCGap{fname} = {macros[f'rhoCGap{fname}']} "
            f"but data shows {gaps[f]:.4f}"
        )
    for macro, want in (("minValidFractionLoPct", min(gaps)),
                        ("minValidFractionHiPct", max(gaps))):
        printed = float(macros[macro].replace(r"\%", ""))
        assert math.isclose(printed, 100 * want), (
            f"Manuscript prints \\{macro} = {macros[macro]} "
            f"(subsec:researcher_choices) but the sweep ran requirement {want}"
        )
    # "essentially invariant": the gap moves <0.005 across the swept requirements.
    spread = max(gaps.values()) - min(gaps.values())
    assert spread < 0.005, (
        f"Manuscript treats MIN_VALID_FRACTION=0.8 as a non-load-bearing observation requirement "
        f"(subsec:researcher_choices) but the Q5-Q1 gap moves {spread:.4f} across requirements "
        f"{sorted(gaps)} ({gaps})"
    )


# ---------------------------------------------------------------------------
# si:dgp_validation: DGP validation appendix
# ---------------------------------------------------------------------------

def test_dgp_eq8_random_graph_criterion_holds():
    """subsec:toy_model: 'In the simulation, the cutoff of fastest change lies within
    \\dgpEqEightMaxGap of tilde-rho_c at every simulated delta from
    \\dgpDeltaGridMin to \\dgpDeltaGridMax.' si:eq9_verification: 'Table tab:dgp_eq9 gives, in each of
    the \\nDgpEqEightCells cells of delta and peer-coupling matrix, ...'

    The cell count is checked against the cells the file holds.

    Threshold (0.012) is set just above the current macro value (0.009) so
    a small re-tuning of MC reps does not break the test, but a drift larger
    than ~30% above the published value does."""
    df = pd.read_csv(DATA / "dgp_validation" / "eq9_verification.csv")
    assert int(_params_macros()["nDgpEqEightCells"]) == len(df), (
        "Manuscript claims the criterion holds across all "
        f"\\nDgpEqEightCells = {_params_macros()['nDgpEqEightCells']} cells "
        f"(si:eq9_verification) but the file holds {len(df)}"
    )
    # subsec:toy_model: 'whereas tilde-rho_c falls by at least
    # \\dgpDegOneMinFall between adjacent simulated values of delta, so the
    # cutoff of fastest change also falls with delta at every step'. The 'so'
    # holds only while every gap is below half the smallest fall.
    falls = pd.concat([-g.sort_values("delta")["rho_c_degree_one_mean"].diff().dropna()
                       for _, g in df.groupby("W_type")])
    printed_fall = float(_params_macros()["dgpDegOneMinFall"])
    assert printed_fall <= falls.min() < printed_fall + 0.001, (
        f"subsec:toy_model prints 'falls by at least {printed_fall}', a lower bound "
        f"at three decimals, but the smallest fall is {falls.min():.4f}")
    max_gap = (df["rho_c_sim_mean"] - df["rho_c_degree_one_mean"]).abs().max()
    printed_gap = float(_params_macros()["dgpEqEightMaxGap"])
    assert printed_gap - 0.001 < max_gap <= printed_gap, (
        f"subsec:toy_model prints 'lies within {printed_gap}', an upper bound at "
        f"three decimals, but the largest gap is {max_gap:.4f}")
    assert max_gap < falls.min() / 2, (
        f"subsec:toy_model infers from the gap ({max_gap:.4f}) and the smallest "
        f"fall ({falls.min():.4f}) that the simulated rho_c falls at every step, "
        f"which needs the gap below half the fall")
    for _, row in df.iterrows():
        gap = abs(float(row["rho_c_sim_mean"]) - float(row["rho_c_degree_one_mean"]))
        assert gap < 0.012, (
            f"Manuscript claims eq:toy_threshold random-graph criterion holds in "
            f"simulation within \\dgpEqEightMaxGap (si:eq9_verification) but at "
            f"delta={row['delta']}, W_type={row['W_type']} the gap is "
            f"{gap:.4f}"
        )


def test_dgp_rho_c_falls_across_delta_grid():
    """subsec:toy_model: 'The two thresholds fall with delta at every step of
    the simulated grid', the cutoff of fastest change and the model's
    threshold. The grid runs from \\dgpDeltaGridMin to
    \\dgpDeltaGridMax (si:dgp_validation).

    Each step must clear twice the combined Monte-Carlo standard error of its
    two cells, and the
    grid endpoints and the matrix count are read from the file rather than
    from the members the sentence names."""
    df = pd.read_csv(DATA / "dgp_validation" / "eq9_verification.csv")
    macros = _params_macros()
    for macro, value in (("dgpDeltaGridMin", df["delta"].min()),
                         ("dgpDeltaGridMax", df["delta"].max())):
        assert math.isclose(float(macros[macro]), float(value), abs_tol=0.005), (
            f"subsec:toy_model prints the simulated delta-grid endpoint "
            f"\\{macro} = {macros[macro]} but eq9_verification.csv holds {value}"
        )
    assert df["W_type"].nunique() == 2, (
        "subsec:toy_model cites 'each of two peer-coupling matrices' but "
        f"eq9_verification.csv holds W types {sorted(set(df['W_type']))}"
    )
    for wt, sub in df.groupby("W_type"):
        sub = sub.sort_values("delta").reset_index(drop=True)
        for k in range(len(sub) - 1):
            a, b = sub.iloc[k], sub.iloc[k + 1]
            for col, se_col in (("rho_c_sim_mean", "rho_c_sim_se"),
                                ("rho_c_degree_one_mean", "rho_c_degree_one_se")):
                step = float(b[col] - a[col])
                se = math.hypot(float(a[se_col]), float(b[se_col]))
                assert step < -2 * se, (
                    f"subsec:toy_model claims that the cutoff of fastest change "
                    f"and the model's threshold both fall with delta at every "
                    f"step of the simulated grid, but under W={wt} the {col} "
                    f"step {a['delta']:.2f}->{b['delta']:.2f} is {step:+.4f} "
                    f"against 2 SE = {2 * se:.4f}"
                )


def test_dgp_eq9_overshoot_range():
    """si:eq9_verification: 'The closed form lies above rho_c^sim by
    \\dgpEqNineOvershootLo to \\dgpEqNineOvershootHi, because the upper tail of
    the simulated bulk is not normal.'

    Asserts (a) the overshoot is strictly positive in every cell and
    (b) the min/max round-trip to the macro values within tolerance."""
    df = pd.read_csv(DATA / "dgp_validation" / "eq9_verification.csv")
    overshoot = df["rho_c_closed_mean"] - df["rho_c_sim_mean"]
    assert overshoot.min() > 0.0, (
        f"Manuscript claims eq:toy_closed_form sits *above* rho_c_sim (si:eq9_verification) "
        f"but min overshoot = {overshoot.min():+.4f}"
    )
    lo, hi = float(overshoot.min()), float(overshoot.max())
    # The macro values are emitted at 3-decimal precision from the same
    # data; tolerance covers MC-rep-driven fluctuation across regenerations.
    assert math.isclose(lo, 0.035, abs_tol=0.010), (
        f"Manuscript claims eq:toy_closed_form overshoot lower bound \\dgpEqNineOvershootLo "
        f"= 0.035 (si:eq9_verification) but data shows {lo:.4f}"
    )
    assert math.isclose(hi, 0.186, abs_tol=0.010), (
        f"Manuscript claims eq:toy_closed_form overshoot upper bound \\dgpEqNineOvershootHi "
        f"= 0.186 (si:eq9_verification) but data shows {hi:.4f}"
    )


def test_dgp_sigma_f_scale_matches_code():
    """si:dgp_validation: \\dgpSigmaFScale must equal the code-level
    DEFAULT_SIGMA_F_SCALE in src.fn_percolation.dgp."""
    expected = float(DEFAULT_SIGMA_F_SCALE)
    params = (Path(__file__).resolve().parent / "paper"
              / "params.tex").read_text()
    m = re.search(r"\\newcommand\{\\dgpSigmaFScale\}\{([^\\]+)\\xspace", params)
    assert m, "macro \\dgpSigmaFScale not found in params.tex"
    actual = float(m.group(1).strip())
    assert math.isclose(actual, expected, abs_tol=0.005), (
        f"params.tex \\dgpSigmaFScale = {actual:.3f} but "
        f"dgp.DEFAULT_SIGMA_F_SCALE = {expected:.3f}"
    )


# ---------------------------------------------------------------------------
# Pipeline-integrity guards on the cached input panels
# ---------------------------------------------------------------------------

def test_returns_panels_finite():
    """No +/-inf in either cached returns panel. A zero/halted close gives
    log(0) = -inf, which (not being NaN) slips past the >80% non-missing window
    filter and silently isolates a stock as a degree-0 node.
    The loaders map non-positive prices to NaN so only valid returns survive."""
    for name in ("returns_panel.parquet", "tw_returns_panel.parquet"):
        arr = pd.read_parquet(DATA / "cache" / name).to_numpy()
        n_inf = int(np.isinf(arr).sum())
        assert n_inf == 0, (
            f"{name} contains {n_inf} +/-inf returns (zero/non-positive price "
            f"slipped through the log)"
        )


def test_tw_panel_stock_ids_unique():
    """Cached TW panel columns are unique, non-empty stock ids. The build-time
    loader guard pins them to the filename universe; this catches gross
    structural corruption from the committed cache alone."""
    cols = pd.read_parquet(DATA / "cache" / "tw_returns_panel.parquet").columns
    assert len(cols) > 0, "TW returns panel has no columns"
    assert len(cols) == len(set(cols)), "TW returns panel has duplicate stock_id columns"


def test_korea_window_fraction_q1_lt_q5():
    """subsec:taiwan: 'Q1's rho_c is below Q5's in 96% of the Taiwanese windows
    and in 83% of the Korean ones' (fracWindowsQoneLtQfiveTw,
    fracWindowsQoneLtQfiveKr)."""
    # The sentence states the strict ordering, so ties do not count (unlike
    # pair_significance.csv's frac_positive, the sign-test convention).
    both = pd.concat({1: _per_q("korea_headline", 1)["rho_c"],
                      5: _per_q("korea_headline", 5)["rho_c"]}, axis=1).dropna()
    frac = float((both[1] < both[5]).mean())
    assert math.isclose(frac, 0.83, abs_tol=ABS_TOL_FRAC), (
        f"Manuscript claims Q1's rho_c is below Q5's in 83% of the Korean windows "
        f"(subsec:taiwan) but data shows {frac:.4f}"
    )


def test_no_dangling_helper_file_references():
    """Code and docstrings name no file that the published tree does not
    hold. Two exploratory helper files are not part of it: neither may exist,
    and neither the suite nor the loader may name them."""
    root = Path(__file__).resolve().parent
    # Build the needles at runtime so this guard never matches its own source.
    pruned = ["diagnostic_" + "2011.py", "ADJUSTMENT_" + "DIAGNOSIS.md"]
    for name in pruned:
        assert not (root / "scripts" / name).exists() and not (root / name).exists(), (
            f"Pruned helper {name} should not exist on main"
        )
    sources = [root / "test_assertions.py",
               root / "src" / "fn_percolation" / "data_loader.py"]
    for src in sources:
        text = src.read_text()
        for name in pruned:
            assert name not in text, (
                f"{src.name} still references pruned file {name}"
            )


# ---------------------------------------------------------------------------
# sec:robustness: the checks and the verdict each returns
# ---------------------------------------------------------------------------

def test_robustness_contrast_chi_gap():
    """tab:robustness_contrast: 'Susceptibility peak rho_c^chi +0.047'."""
    df = _susceptibility_summary().sort_values("size_group")
    chi = df["rho_c_chi_mean"].tolist()
    gap = chi[4] - chi[0]
    assert math.isclose(gap, 0.047, abs_tol=ABS_TOL_RHO), (
        f"Manuscript claims the chi-peak endpoint contrast is +0.047 "
        f"(tab:robustness_contrast) but data shows {gap:.4f}"
    )


def test_robustness_summary_ordering_column():
    """tab:robustness_summary, 'Ordering' column: which checks preserve the
    strict cross-quintile ordering Q1 < Q2 < Q3 < Q4 < Q5 and which do not.

    The headline, Marchenko-Pastur and within-sector rows claim 'ordered'; the
    KOSPI-only row claims 'Q3--Q4 reversed'; the shuffled-returns row claims
    'reversed'. Each is asserted on the per-quintile means the
    corresponding pipeline wrote.
    """
    def strictly_increasing(v: list[float]) -> bool:
        return all(b > a for a, b in zip(v, v[1:]))

    for name in ("korea_headline", "rmt_filtered", "within_sector"):
        means = _q_means(name)
        assert strictly_increasing(means), (
            f"Manuscript claims the {name} row is 'ordered' "
            f"(tab:robustness_summary) but data shows {means}"
        )

    kospi = _q_means("kospi_only")
    assert kospi[3] < kospi[2], (
        f"Manuscript claims the KOSPI-only row shows 'Q3--Q4 reversed' "
        f"(tab:robustness_summary) but data shows Q3={kospi[2]:.4f} "
        f"Q4={kospi[3]:.4f}"
    )
    assert strictly_increasing([kospi[0], kospi[1], kospi[2]]) and kospi[4] > kospi[3], (
        f"Manuscript claims the KOSPI-only endpoint still rises around the "
        f"Q3--Q4 reversal (tab:robustness_summary) but data shows {kospi}"
    )

    shuffled = (pd.read_csv(DATA / "null_shuffled" / "cross_quintile_summary.csv")
                .sort_values("quintile")["across_seed_mean"].tolist())
    assert all(b < a for a, b in zip(shuffled, shuffled[1:])), (
        f"Manuscript claims the shuffled-returns row is 'reversed' "
        f"(tab:robustness_summary) but data shows {shuffled}"
    )


# ---------------------------------------------------------------------------
# fig:headline_gradient: the paired-difference error bars
# ---------------------------------------------------------------------------

def test_hac_mean_se_matches_block_stats():
    """fig:headline_gradient caption: 'Error bars are one Newey--West HAC
    standard error of the mean difference between the quintile's rho_c and
    Q1's in the same window'; tab:pair_significance tests the same per-window
    differences.

    The figure calls hac_mean_se and the significance table calls
    hac_block_stats, so the two must return the same standard error on the same
    series or the bars would not match the tabulated t-statistics.
    """
    for g in (2, 3, 4, 5):
        diff = _adjacent_diff(g - 1) if g < 5 else _endpoint_diff()
        block = hac_block_stats(diff)
        mean_se = hac_mean_se(diff)
        implied = block["mean_diff"] / block["nw_t"]
        assert math.isclose(mean_se["se"], implied, rel_tol=1e-12), (
            f"fig:headline_gradient error bars use hac_mean_se but "
            f"tab:pair_significance uses hac_block_stats; the two disagree at "
            f"Q{g} ({mean_se['se']:.6g} vs {implied:.6g})"
        )
        assert mean_se["nw_lag"] == block["nw_lag"], (
            f"fig:headline_gradient and tab:pair_significance disagree on the "
            f"Newey-West bandwidth at Q{g}"
        )


# ---------------------------------------------------------------------------
# Arm-coverage gates: every arm reaches the reader, and a quantifier over the
# paper's own arms holds over every arm. The arm set these range over is derived from
# data/ at check time. A written-out arm list is true on the day it is written
# and silently false afterwards: it passes the first time an arm is added,
# which is the failure it exists to prevent.
# ---------------------------------------------------------------------------

PAPER = Path(__file__).resolve().parent / "paper"
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))


def _params_macros() -> dict[str, str]:
    """Macro name -> value, parsed from the committed params.tex."""
    text = (PAPER / "params.tex").read_text()
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*?)\\xspace\}", text))


def _strip_tex_comments(text: str) -> str:
    """Drop everything from an unescaped % to end of line.

    A macro sitting in a commented-out row is present in the file and absent
    from the typeset page, so a "reaches the reader" check that reads the raw
    source passes on a row someone commented out -- the artifact-to-reader gap
    one level down."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _manuscript_text() -> str:
    return _strip_tex_comments((PAPER / "paper.tex").read_text()
                               + "\n" + (PAPER / "paper_si.tex").read_text())


def _is_printed(macro: str, text: str) -> bool:
    return re.search(r"\\" + macro + r"(?![A-Za-z])", text) is not None


def test_generator_opens_every_arm_the_pipeline_scored():
    """Every arm the pipeline scored must
    reach the reader as some number, and nothing is bound if nothing is printed,
    so no macro-binding assertion can see an arm the generator never opens. Six
    external-factor and six own-PCA residualization arms sat in data/ unopened
    for exactly this reason: the generator held a literal arm list.

    Both sides are derived -- the arms from data/, the opens from the paths the
    generator actually read while building the macro set."""
    import generate_params

    generate_params.build_macros()
    opened = {Path(rel).parts[0] for rel in generate_params.READ_PATHS}
    scored = {d.name for d in generate_params._rolling_arm_dirs()}
    unopened = sorted(scored - opened)
    assert not unopened, (
        "Every arm the pipeline scored reaches the reader as "
        "some number, but scripts/generate_params.py never opens these arms' "
        f"output, so nothing they produced is printed: {unopened}"
    )


def test_taiwan_interior_hac_range_covers_every_interior_pair():
    """subsec:taiwan: "Each of the three steps through Q4 is significant, with
    HAC t-statistics between \\tHacTwQtwoQthree and \\tHacTwQoneQtwo." "Between
    X and Y" ranges over the
    interior pairs, so the two macros the sentence prints have to be that
    range's endpoints rather than two members of it, and the range is derived
    from the panel's own per-pair file."""
    pairs = pd.read_csv(DATA / "taiwan" / "pair_significance.csv")
    rho_c = pairs[pairs["observable"] == "rho_c"]
    interior = {r["pair"]: float(r["nw_t"])
                for _, r in rho_c.iterrows() if r["pair"] in ("Q2-Q1", "Q3-Q2", "Q4-Q3")}
    assert len(interior) == 3, f"expected three interior pairs, got {interior}"
    macros = _params_macros()
    lo = float(macros["tHacTwQtwoQthree"])
    hi = float(macros["tHacTwQoneQtwo"])
    assert math.isclose(lo, min(interior.values()), abs_tol=0.05), (
        f"Manuscript prints {lo} as the low end of the Taiwanese interior HAC "
        f"range (subsec:taiwan) but the minimum over {interior} is "
        f"{min(interior.values()):.2f}"
    )
    assert math.isclose(hi, max(interior.values()), abs_tol=0.05), (
        f"Manuscript prints {hi} as the high end of the Taiwanese interior HAC "
        f"range (subsec:taiwan) but the maximum over {interior} is "
        f"{max(interior.values()):.2f}"
    )
    for pair, t in interior.items():
        assert t > 2.0, (
            "Manuscript claims each Taiwanese step through Q4 "
            f"is significant (subsec:taiwan) but {pair} has HAC "
            f"t={t:.2f}"
        )


def test_params_generated_region_matches_a_fresh_regeneration():
    """A regeneration is byte-identical to what is committed,
    or the difference is one this commit owes. This also checks the design
    constants, because the generator reads WINDOW,
    STEP, PER_QUINTILE, MIN_VALID_FRACTION, MIN_SECTOR_SIZE, FLOORS, and
    MA_WINDOW from the modules that fix them, so a constant that moves in
    code without params.tex following it fails here rather than printing the old
    value in the manuscript indefinitely."""
    import generate_params

    fresh = generate_params.render_block(generate_params.build_macros())
    committed = (PAPER / "params.tex").read_text()
    start = committed.index(generate_params.BEGIN)
    end = committed.index(generate_params.END) + len(generate_params.END)
    assert committed[start:end] == fresh, (
        "paper/params.tex's generated region does not match a fresh run of "
        "scripts/generate_params.py. Re-run it and ship the regenerated file "
        "with whatever moved."
    )


def test_multi_observable_pair_files_are_read_on_one_observable():
    """Two populations that print identically are the ones a
    reader cannot catch. This repo has one inside a single file --
    data/taiwan/pair_significance.csv carries rho_c and rho_c_chi under the same
    five pair labels -- so a read that does not filter on `observable` mixes
    them, and the mixed value is a plausible t-statistic. It is not a
    hypothetical: an audit gate written against this file reported the Taiwanese
    Q3-Q4 contrast as unresolved by silently keeping the chi rows.

    The set of files this ranges over is derived: every pair file whose label
    column repeats. Each must resolve to one row per label once its discriminator
    is applied, and the macros the manuscript prints must match that
    sub-population rather than a mixture."""
    macros = _params_macros()
    checked = 0
    for path in sorted(DATA.glob("*/pair_significance.csv")):
        df = pd.read_csv(path)
        if not df["pair"].duplicated().any():
            continue
        checked += 1
        assert "observable" in df.columns, (
            f"{path} repeats a pair label with no column separating the rows; "
            "any read of it mixes two populations under one name"
        )
        primary = df[df["observable"] == "rho_c"]
        assert not primary["pair"].duplicated().any(), (
            f"{path}: filtering on observable=='rho_c' still leaves duplicate "
            "pair labels, so a third population is present and undeclared"
        )
        if path.parent.name != "taiwan":
            continue
        # The Taiwanese macros must carry the curvature observable's values, not
        # the chi observable's and not a mixture of the two.
        by_pair = {r["pair"]: float(r["nw_t"]) for _, r in primary.iterrows()}
        for lo, hi, macro in [(1, 2, "tHacTwQoneQtwo"), (2, 3, "tHacTwQtwoQthree"),
                              (3, 4, "tHacTwQthreeQfour"), (4, 5, "tHacTwQfourQfive"),
                              (1, 5, "tHacTwQoneQfive")]:
            assert math.isclose(float(macros[macro]), by_pair[f"Q{hi}-Q{lo}"],
                                abs_tol=0.06), (
                f"\\{macro} = {macros[macro]} does not match the rho_c row of "
                f"{path} ({by_pair[f'Q{hi}-Q{lo}']:.2f}); it may have been read "
                "off the rho_c_chi rows, which carry the same pair labels"
            )
    assert checked >= 1, "expected at least one multi-observable pair file"


def test_universe_cap_is_never_printed_as_a_realized_count():
    """A design-time frame and the realized one are two
    populations. \\paperPerQuintile is the sampling cap PER_QUINTILE fixes, not
    the number of stocks any window actually held -- the realized per-window
    medians (\\medianStocks*) are a different population and sit below it. Every
    print of the cap must carry the hedge that says so, or the sentence claims a
    realized count the data does not support."""
    hedges = ("cap", "Up to", "up to", r"\leq", r"\approx", "N = ", "N =",
              r"$N = \paperPerQuintile$")
    offenders = []
    for name in ("paper.tex", "paper_si.tex"):
        text = (PAPER / name).read_text()
        for m in re.finditer(r"\\paperPerQuintile(?![A-Za-z])", text):
            window = text[max(0, m.start() - 120):m.end() + 40]
            if not any(h in window for h in hedges):
                offenders.append(f"{name}: ...{window[-100:].strip()}...")
    assert not offenders, (
        "\\paperPerQuintile is the design-time sampling cap, not a realized "
        "per-window count, but it is printed with no hedge "
        "at:\n  " + "\n  ".join(offenders)
    )


def test_highlights_file_is_a_fresh_render_of_the_manuscript_block():
    """paper/highlights.txt ships with the article and duplicates paper.tex's
    own highlights environment, which is the block that renders. Two files under
    one meaning with nothing comparing them is a second copy of one block, in
    which a hand edit can state a claim that the paper does not. The file is
    generated, so a divergence is a diff."""
    import generate_params

    before = (PAPER / "highlights.txt").read_text()
    generate_params.write_highlights()
    after = (PAPER / "highlights.txt").read_text()
    assert before == after, (
        "paper/highlights.txt is not what paper.tex's highlights environment "
        "renders to. Re-run scripts/generate_params.py; the manuscript block is "
        "the source of truth"
    )
    for name in ("barRhoCQone", "barRhoCQfive"):
        value = _params_macros()[name]
        assert value in after, (
            f"the rendered highlights no longer carry \\{name} = {value}, so "
            f"one of the file's two numbers is unbound"
        )
    # Elsevier's limits on highlights: three to five items, each at most 85
    # characters including spaces. The renders are measured after macro
    # substitution, because a number that grows by a digit can push an item over.
    items = [line[2:] for line in after.splitlines() if line.startswith("- ")]
    assert 3 <= len(items) <= 5, (
        f"The journal takes three to five highlights but paper.tex's highlights "
        f"environment renders {len(items)}"
    )
    long_items = [f"{len(i)}: {i}" for i in items if len(i) > 85]
    assert not long_items, (
        "The journal caps a highlight at 85 characters including spaces, and "
        "these render longer:\n  " + "\n  ".join(long_items)
    )


def test_the_cross_population_macro_is_registered_and_names_both_markets():
    """A deliberate crossing is registered with its reason in
    the place the check reads. \\rhoCGapTwNImplied is the one macro here whose
    value spans two populations by design -- a slope estimated on the Korean
    panel, applied to Taiwan's realized network sizes -- so it is registered in
    generate_params.REGISTERED_CROSSINGS, and the sentence printing it must name
    both markets rather than reading as one measurement."""
    import generate_params

    assert "rhoCGapTwNImplied" in generate_params.REGISTERED_CROSSINGS, (
        "\\rhoCGapTwNImplied crosses the Korean and Taiwanese populations and "
        "must be registered with its reason in generate_params."
    )
    # Every paragraph printing the crossing, in either document, in either form
    # (the value or its share of the Taiwanese gap).
    text = _manuscript_text()
    hits = [m.start() for m in re.finditer(r"\\rhoCGapTwNImplied(?:Pct)?(?![A-Za-z])", text)]
    assert hits, "\\rhoCGapTwNImplied is registered as a crossing but never printed"
    for idx in hits:
        para_start = text.rfind("\n\n", 0, idx)
        para = text[para_start:text.index("\n\n", idx)]
        assert "Korea" in para and ("Taiwan" in para or "Taiwanese" in para), (
            "The paragraph printing \\rhoCGapTwNImplied applies a Korean-measured "
            "slope to Taiwanese network sizes and must name both markets, "
            "but names only one."
        )


REPO = Path(__file__).resolve().parent


def _baseline_tags() -> dict[str, str]:
    """Script path -> the text of its BASELINE tag, derived by scanning
    scripts/ rather than from a list of the scripts known to carry one."""
    tags = {}
    for path in sorted(REPO.glob("scripts/**/*.py")):
        text = path.read_text()
        i = text.find("BASELINE:")
        if i == -1:
            continue
        end = text.find('"""', i)
        tags[str(path.relative_to(REPO))] = text[i:end if end != -1 else i + 2000]
    return tags


def test_no_baseline_choice_is_left_pending():
    """A `discharge: pending` tag under a shipped claim is an
    audit finding. Every BASELINE tag in the repo names how its choice was
    discharged, and the tag set is derived from scripts/, not listed."""
    pending = [p for p, tag in _baseline_tags().items() if "discharge: pending" in tag]
    assert not pending, (
        "These fixed choices ship under a published claim with their obvious "
        f"alternative undischarged: {pending}"
    )


def test_every_baseline_discharge_names_a_script_the_pipeline_runs():
    """A discharge by run keeps its evidence attached, and a discharge living
    in a branch reaches no reader -- so a `discharge: run --
    <script>` must name a script that exists and that reproduce.sh executes.
    Both ends are derived: the tags from scripts/, the run set from
    reproduce.sh."""
    reproduce = (REPO / "reproduce.sh").read_text()
    broken = []
    for owner, tag in _baseline_tags().items():
        for name in re.findall(r"(run_[a-z0-9_]+\.py)", tag):
            matches = list(REPO.glob(f"scripts/**/{name}"))
            if not matches:
                broken.append(f"{owner}: BASELINE names {name}, which does not exist")
            elif name not in reproduce:
                broken.append(
                    f"{owner}: BASELINE discharges to {name}, which reproduce.sh "
                    "does not run, so the evidence is not attached")
    assert not broken, (
        "BASELINE discharges must point at runs the pipeline executes"
        ":\n  " + "\n  ".join(broken)
    )


def test_the_residual_battery_arms_named_as_discharges_are_all_reported():
    """A discharge that confirms the result ships as an
    in-scope check carrying its own label, and one that overturns it ships as
    the finding. The cap-weighted residual arms name their equal-weighted
    siblings, by data/ directory, as their discharge; each must reach the
    reader, which is the same obligation test_residual_step_table_matches_the_pair_files
    enforces from the artifact side. Checked from the tag side here, so deleting a tag cannot
    quietly retire it."""
    import generate_params

    macros = _params_macros()
    text = _manuscript_text()
    named = set()
    for tag in _baseline_tags().values():
        for name in re.findall(r"data/(market_[a-z0-9_]+)", tag):
            named.add(name)
    assert named, "expected the residual scripts' BASELINE tags to name their discharges"
    for arm in sorted(named):
        assert (DATA / arm).is_dir(), f"BASELINE names {arm}, which wrote no data/ directory"
        macro = "rhoCGap" + generate_params._external_factor_stem(arm)
        assert macro in macros and _is_printed(macro, text), (
            f"{arm} is named as a BASELINE discharge but its endpoint contrast "
            f"(\\{macro}) does not reach the reader"
        )


def test_ai_declaration_pins_no_model_version_the_history_denies():
    """paper.tex, Declaration of generative AI: 'the author used Claude
    (Anthropic) in order to draft manuscript prose ...'.

    A version pinned in that sentence is dereferenced by nothing, so it goes
    stale the day the next release does the work and every check stays green:
    the declaration named Opus 4.6 while six later versions carried the commits.
    Derive the versions that did the work from the co-authorship trailers
    and require any version the sentence names to be among
    them. Naming none passes, and the tool itself must still be named."""
    import subprocess

    decl = re.search(r"\\section\*\{Declaration of generative AI[^}]*\}"
                     r"(.*?)(?=\\(?:section|bibliography|end\{document\}))",
                     (PAPER / "paper.tex").read_text(), re.S)
    assert decl, "paper.tex no longer carries a 'Declaration of generative AI' section"
    sentence = decl.group(1)
    assert "Claude" in sentence and "Anthropic" in sentence, (
        "The generative-AI declaration must name the tool and its maker; "
        f"paper.tex now reads: {' '.join(sentence.split())[:200]}"
    )

    named = set(re.findall(r"\b(Opus|Sonnet|Haiku|Fable)\s+([0-9][0-9.]*)", sentence))
    trailers = subprocess.run(["git", "log", "--format=%b"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout
    did_the_work = set(re.findall(r"Co-Authored-By: Claude (Opus|Sonnet|Haiku|Fable) "
                                  r"([0-9][0-9.]*)", trailers))
    unsupported = named - did_the_work
    assert not unsupported, (
        "The generative-AI declaration pins model versions no commit in this "
        f"repository was co-authored by: {sorted(' '.join(v) for v in unsupported)}. "
        f"The versions that did the work are "
        f"{sorted(' '.join(v) for v in did_the_work)}"
    )


def _tracked_python_files() -> list[Path]:
    """Every .py file version control holds, so a module added to the repository
    is checked without anyone remembering to list it."""
    import subprocess

    out = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    return [REPO / rel for rel in out.split()]


def test_every_name_a_latex_comment_cites_resolves():
    """A LaTeX comment naming a
    script, a constant, a test, or a \\label slug is dereferenced by nothing --
    the typesetter never reads it and no import breaks -- so a rename leaves it
    pointing at nothing with the suite green. params.tex's manual region named
    test_iv_lag_horizon_robust for the commit that replaced it.

    These comments ship: a preprint server publishes the sources it is given, so
    a dangling pointer here reaches the reader. Both ends derived: the citing set from the comments
    themselves, the defined set from the tree and from the suite."""
    import re

    tex = [PAPER / n for n in ("paper.tex", "paper_si.tex", "params.tex")]
    comments, labels = [], set()
    for path in tex:
        text = path.read_text()
        labels |= set(re.findall(r"\\label\{([^}]+)\}", text))
        for lineno, line in enumerate(text.splitlines(), 1):
            m = re.search(r"(?<!\\)%(.*)$", line)
            if m:
                comments.append((path.name, lineno, m.group(1)))

    suite_defs = set(re.findall(r"^def (test_[a-z0-9_]+)",
                                Path(__file__).read_text(), re.M))
    dangling = []
    for name, lineno, comment in comments:
        for rel in re.findall(r"[A-Za-z0-9_][A-Za-z0-9_./-]*\.(?:py|sh|tex|csv|bib)",
                              comment):
            candidates = [REPO / rel] if "/" in rel else list(REPO.rglob(rel))
            if not any(c.exists() for c in candidates):
                dangling.append(f"{name}:{lineno} path {rel}")
        for test in re.findall(r"\btest_[a-z0-9_]+\b", comment):
            if test not in suite_defs:
                dangling.append(f"{name}:{lineno} test {test}")
        for slug in re.findall(r"\b(?:sec|subsec|si|tab|fig|eq|app):[A-Za-z0-9_]+",
                               comment):
            if slug not in labels:
                dangling.append(f"{name}:{lineno} label {slug}")

    assert not dangling, (
        "These names are cited in a LaTeX comment and resolve to nothing"
        f": {dangling}")


def test_no_tracked_file_names_one_machines_home_directory():
    """A path naming a home directory fails on every other
    machine pointing at a stranger's filesystem instead of at the tree to
    supply, whereas the loaders resolve the vendor trees under ``~``.

    The repository publishes at the URL the data-availability statement names,
    so this is what a first run on another machine hits. Derived: the tracked
    set from version control, not a list of files to check."""
    import os
    import subprocess

    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout.split()
    offenders = []
    for rel in out:
        path = REPO / rel
        if path.suffix.lower() in {".pdf", ".parquet", ".png", ".jpg", ".bst",
                                   ".cls", ".sty"}:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        # Assembled rather than written out, so this file does not match itself.
        needles = [f"{os.sep}{root}{os.sep}" for root in ("home", "Users")]
        for lineno, line in enumerate(text.splitlines(), 1):
            if any(needle in line for needle in needles):
                offenders.append(f"{rel}:{lineno}")
    assert not offenders, (
        "These tracked files name an absolute home directory, which resolves "
        "only on the machine that wrote it. Resolve the "
        f"path under ``~`` instead: {offenders}")


def test_every_package_name_is_imported_from_the_module_that_defines_it():
    """No dead code on the publication line, and the check reads the artifact
    rather than a comment asserting it.

    pipeline.py carried a seventeen-name backward-compatibility shim re-exporting
    constants.py and windows.py, for a package whose only consumers are in this
    repository. Two of the seventeen had a caller. A shim is invisible to every
    other gate here -- it moves no number and breaks no import -- and it gives a
    name two import paths, so an edit to one leaves the other stale.

    Both ends derived: the defining module from each module's own bindings, the
    referring set from the import statements the repository actually contains."""
    import ast

    src_root = REPO / "src" / "fn_percolation"
    trees = {p.stem: ast.parse(p.read_text()) for p in src_root.glob("*.py")}

    # What each module binds by importing it from somewhere else, rather than
    # by defining it.
    reexported = {}
    for mod, tree in trees.items():
        names = set()
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    if a.name != "*":
                        names.add(a.asname or a.name.split(".")[0])
        reexported[mod] = names

    offenders = []
    for path in _tracked_python_files():
        if path.parent == src_root and path.stem in trees:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            mod = node.module.split(".")[-1]
            if mod not in trees:
                continue
            for a in node.names:
                if a.name in reexported.get(mod, ()):
                    offenders.append(
                        f"{path.relative_to(REPO)} takes {a.name} from "
                        f"fn_percolation.{mod}, which imports it rather than "
                        f"defining it")
    assert not offenders, (
        "A package name has two import paths, so the repository publishes a "
        "re-export shim no caller needs: "
        + "; ".join(offenders))


def test_internal_working_documents_are_untracked():
    """Audit instruments stay out of the manuscript, and "in the repo" is
    safe only while the repo is private -- once the availability
    statement points at a public URL, whatever is tracked there is published.
    .gitignore's block records that decision, and a pattern added to .gitignore
    is inert on a path already tracked, so a file that the block names can
    keep publishing while the block reads as enforced.

    Both ends derived: the patterns from .gitignore, the tracked set from git.
    The block is found by its BEGIN marker rather than by its prose, which an
    edit to the comment would otherwise silently move out from under this
    gate."""
    import subprocess

    gitignore = (REPO / ".gitignore").read_text()
    marker = "# BEGIN not-published-by-the-availability-statement"
    assert marker in gitignore, (
        f".gitignore no longer carries the {marker!r} marker this gate reads; "
        "the block it names is where the availability statement's exclusions live"
    )
    block = gitignore.split(marker)[1].split("\n\n")[0]
    patterns = [l.strip() for l in block.splitlines()
                if l.strip() and not l.strip().startswith("#")]
    assert patterns, "the internal-documents block of .gitignore names no paths"
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout.split()
    still_tracked = [f for f in tracked
                     if any(Path(f).match(pat) for pat in patterns)]
    assert not still_tracked, (
        ".gitignore's internal-documents block names these paths, but they are "
        "still tracked and would publish with the repo the data-availability "
        f"statement points at. `git rm --cached` them: {still_tracked}"
    )


# A directory nothing publishes directly, whose numbers reach the reader through
# the directory named beside it. An escape hatch that a check enumerates is a
# decision, a comment is not -- so each entry is verified below
# against its downstream, and a stale one fails when that downstream stops being
# read.
INTERMEDIATE_RESULT_DIRS = {}


def _directly_read_result_dirs() -> set[str]:
    """The data/ directories some reader opens: the macro generator, the figure
    script, this suite, or either document naming one."""
    import generate_params

    generate_params.build_macros()
    opened = {Path(rel).parts[0] for rel in generate_params.READ_PATHS}
    surfaces = ((REPO / "generate_figures.py").read_text()
                + _suite_source_without_its_exemptions()
                + _manuscript_text())
    return opened | {name for name in _committed_result_dirs() if name in surfaces}


def _suite_source_without_its_exemptions() -> str:
    """This file's own source, minus INTERMEDIATE_RESULT_DIRS.

    A directory named in the exemption table is declared, not read, and counting
    the declaration as a read would let the table satisfy itself."""
    return re.sub(r"INTERMEDIATE_RESULT_DIRS = \{.*?\}", "",
                  (REPO / "test_assertions.py").read_text(), flags=re.S)


def _committed_result_dirs() -> set[str]:
    import subprocess

    tracked = subprocess.run(["git", "ls-files", "data"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.split()
    return {Path(p).parts[1] for p in tracked if len(Path(p).parts) > 2}


def test_every_committed_result_directory_reaches_the_reader():
    """Every arm, split, or condition the
    results data contains reaches the reader as some number, or is deleted.

    test_generator_opens_every_arm_the_pipeline_scored derives over directories
    holding a rolling-percolation output set, so a diagnostic writing any other
    shape is invisible to it -- the same defect one directory over. This gate
    derives over every committed data/ directory instead, and each must be read
    by the generator, a figure, this suite, or a document, or be declared an
    intermediate feeding one that is."""
    unread = sorted(_committed_result_dirs()
                    - _directly_read_result_dirs()
                    - set(INTERMEDIATE_RESULT_DIRS))
    assert not unread, (
        "These directories are committed results that reach no "
        "reader -- no macro, figure, assertion, or section takes a number from "
        f"them. Report them or delete them: {unread}"
    )


def test_declared_intermediates_feed_a_published_directory():
    """The escape hatch above is only a decision while what it claims stays
    true: an intermediate earns its exemption from the directory it feeds, so
    that directory must itself reach a reader."""
    published = _directly_read_result_dirs()
    orphaned = {src: dst for src, dst in INTERMEDIATE_RESULT_DIRS.items()
                if dst not in published}
    assert not orphaned, (
        "INTERMEDIATE_RESULT_DIRS exempts each key because the directory it "
        "names carries its numbers to the reader, but that directory is itself "
        f"unread, so the exemption rests on nothing: {orphaned}"
    )


def test_shuffled_null_asymmetry_tracks_the_marginals_the_shuffle_preserves():
    """subsec:dissipation_mechanism sources the shuffled null's reversed
    Q1-above-Q5 asymmetry in the return marginals, which the per-stock time
    permutation preserves exactly: "Q1's returns have the heaviest tails, which
    can raise the upper quantile of the sampling distribution of the
    correlations of independent series, where rho_c lies, and so can produce
    the reversed endpoint contrast of the shuffled-returns null";
    subsec:null_robustness: "Averaged over the stocks of each window's quintile
    and then over the windows, the excess kurtosis of the Korean daily returns
    falls at every step across the quintiles, from 3.8 in Q1 to 2.0 in Q5, so the returns
    of the smaller stocks have heavier tails, which can raise the upper quantile
    of the sampling distribution of the correlations of independent series and
    so reverse the endpoint contrast of the shuffled-returns null."

    Monotonicity is a quantifier over all five quintiles, so it is asserted over
    the set rather than over the two the sentence names, and the per-quintile
    summary is checked against the cell-level file it aggregates."""
    summary = pd.read_csv(REPO / "data" / "pca_kurtosis_diagnostic"
                          / "excess_kurtosis_summary.csv", header=[0, 1], index_col=0)
    means = summary[("mean_excess_kurtosis", "mean")].sort_index()

    cells = pd.read_csv(REPO / "data" / "pca_kurtosis_diagnostic"
                        / "excess_kurtosis_by_cell.csv")
    recomputed = cells.groupby("quintile")["mean_excess_kurtosis"].mean().sort_index()
    assert np.allclose(means.to_numpy(), recomputed.to_numpy(), atol=5e-4), (
        "excess_kurtosis_summary.csv is the per-quintile mean of "
        f"excess_kurtosis_by_cell.csv, but they disagree: {means.to_numpy()} vs "
        f"{recomputed.to_numpy()}"
    )

    steps = means.diff().dropna()
    assert (steps < 0).all(), (
        "subsec:null_robustness claims mean excess kurtosis falls at every step "
        f"across the quintiles, but the steps are {steps.to_dict()}"
    )

    macros = _params_macros()
    for g, name in ((1, "excessKurtQone"), (5, "excessKurtQfive")):
        assert math.isclose(float(macros[name]), float(means.loc[g]), abs_tol=0.05), (
            f"subsec:null_robustness prints \\{name} = {macros[name]} but the "
            f"Q{g} mean excess kurtosis is {means.loc[g]:.4f}"
        )


def test_no_pipeline_intermediate_is_read_by_the_paper_chain():
    """.gitignore's pipeline-intermediates block claims its entries are "read by
    no test, no figure, and no parameter macro", and warns in its own words that
    a stale entry silently untracks a published artifact.

    That warning is the only thing standing behind the list, and a comment is
    not the property it asserts. The block states its own
    re-derivation; this runs it. It is also the one staleness
    test_every_committed_result_directory_reaches_the_reader cannot see, because
    a directory this block untracks never appears in `git ls-files data`."""
    import generate_params

    generate_params.build_macros()
    opened = set(generate_params.READ_PATHS)
    surfaces = ((REPO / "generate_figures.py").read_text()
                + _suite_source_without_its_exemptions())

    block = (REPO / ".gitignore").read_text().split(
        "# Pipeline intermediates:")[1].split("\n\n")[0]
    entries = [l.strip() for l in block.splitlines()
               if l.strip() and not l.strip().startswith(("#", "-"))]
    entries = [e for e in entries if e.startswith("data/")]
    assert entries, "the pipeline-intermediates block of .gitignore names no paths"

    published = []
    for entry in entries:
        rel = entry[len("data/"):].rstrip("/")
        if any(o == rel or o.startswith(rel + "/") for o in opened) or rel in surfaces:
            published.append(entry)
    assert not published, (
        ".gitignore's pipeline-intermediates block untracks these, but the paper "
        "chain now opens them, so the block would drop a published artifact from "
        f"the repository the availability statement points at: {published}"
    )


# ---------------------------------------------------------------------------
# Panel construction: the Korean universe rules and the Taiwanese survivorship
# exposure (subsec:data_description), and the cleaning steps (si:settings).
# ---------------------------------------------------------------------------

def _build_stats(market: str) -> pd.Series:
    """One market's row of the builder-written panel-construction shares."""
    stats = pd.read_csv(DATA / "cache" / "panel_build_stats.csv").set_index("market")
    return stats.loc[market]


def test_kr_universe_excludes_preferred_lines():
    """si:settings: 'The universe excludes preferred shares, ...', the rule
    behind 'the common stocks of the KOSPI and KOSDAQ boards' of
    subsec:data_description.

    KRX numbers a preferred line with a terminal character other than '0', so
    the claim is checkable on the published panel itself rather than on the
    classifier that produced it. Asserted over the panel's own columns
    (derived, not declared), not over a list of the codes the
    classifier happened to drop."""
    for panel in ("returns_panel.parquet", "marketcap_panel.parquet"):
        cols = pd.read_parquet(DATA / "cache" / panel).columns
        offenders = [c for c in cols if not str(c).endswith("0")]
        assert not offenders, (
            f"si:settings claims the Korean universe excludes preferred shares, "
            f"but {panel} carries {len(offenders)} column(s) whose code does "
            f"not end in 0, the mark of a preferred line: {offenders[:5]}"
        )


def test_kr_universe_admits_only_the_sessions_fnguide_prices():
    """si:settings: 'FnGuide prices no REIT, fund, or ETF, so the universe
    admits only the sessions that FnGuide prices.'

    The build raises on a universe session the export does not price, and the
    export is not in the repository. Asserted here on the published panels:
    the market-cap panel the size cut ranks and the board flag carry exactly the codes the returns panel carries, so no code without
    a return reaches the ranking."""
    ret = set(pd.read_parquet(DATA / "cache" / "returns_panel.parquet").columns)
    held = {
        "marketcap_panel.parquet":
            set(pd.read_parquet(DATA / "cache" / "marketcap_panel.parquet").columns),
        "kr_market_flag.csv":
            set(pd.read_csv(DATA / "cache" / "kr_market_flag.csv",
                            dtype={"code": str})["code"]),
    }
    for name, codes in held.items():
        assert codes == ret, (
            f"si:settings claims the Korean universe admits only the sessions "
            f"FnGuide prices, but {name} carries {len(codes - ret)} code(s) "
            f"with no returns column and lacks {len(ret - codes)} that have one"
        )


def test_taiwan_panel_omits_the_delisted_names_it_says_it_omits():
    """subsec:data_description: 'The Taiwanese panel holds the common stocks
    ... except some delisted stocks whose prices the Taiwanese data lack' and
    'FinMind ... provides no adjusted close for those delisted stocks.'

    Asserted as the property rather than as the wording: of the delisted names
    the live registry no longer lists, the vendor serves an adjusted series for
    fewer than all, and none of them reaches the published returns panel. A grep for the phrase would pass on a panel that
    quietly changed and fail on a sentence that merely names both markets."""
    tw = _build_stats("taiwan")
    absent = int(tw["delisted_absent_from_registry"])
    served = int(tw["delisted_served_price_adj"])
    assert served < absent, (
        "subsec:data_description claims FinMind provides no adjusted close for "
        f"some delisted Taiwanese stocks, but it serves one for all {absent} "
        f"delisted names the registry no longer lists"
    )

    delisting = pd.read_parquet(REPO / "data" / "external" / "registry"
                                / "twn_delisting.parquet")
    gone = set(delisting["stock_id"].astype(str))
    cols = set(pd.read_parquet(DATA / "cache"
                               / "tw_returns_panel.parquet").columns.astype(str))
    live = set(pd.read_parquet(REPO / "data" / "external" / "registry"
                               / "twn_stock_info.parquet")["stock_id"].astype(str))
    leaked = sorted((gone & cols) - live)
    assert not leaked, (
        f"The Taiwanese universe is FinMind's live registry, but the panel "
        f"carries {len(leaked)} delisted code(s) the registry no longer lists: "
        f"{leaked[:5]}"
    )


def test_taiwan_returns_fall_on_listed_sessions():
    """si:settings: 'The panel therefore admits a Taiwanese trading day only
    if FinMind carries the stock's share count on that day. If the stock has a
    price on an earlier day, FinMind must also carry the share count on the
    latest earlier day with a price. [...] No return of the panel therefore runs from a price
    set on the emerging board, because no return spans a dropped day.'

    The market cap is formed on the share count, so on the published panels
    every Taiwanese return must have a market cap on its own session and on
    the session it is differenced from. The registry records the date on
    which it retired a stock's emerging-board row, and a stock that listed
    from the emerging board carries no return on or before that date."""
    ret = pd.read_parquet(DATA / "cache" / "tw_returns_panel.parquet")
    mc = pd.read_parquet(DATA / "cache" / "tw_marketcap_panel.parquet")
    mc = mc.reindex(index=ret.index, columns=ret.columns)
    has_ret = ret.notna()
    unsized = has_ret & (mc.isna() | mc.shift().isna())
    # A session on which FinMind reports a zero share count is listed but
    # carries no market cap (test_taiwan_market_cap_carries_no_nonpositive_size),
    # which leaves the return into it and the one out of it unsized.
    allowed = 2 * int(_build_stats("taiwan")["nonpositive_marcap_cells"])
    n_unsized = int(unsized.to_numpy().sum())
    assert n_unsized <= allowed, (
        f"si:settings claims every Taiwanese session in the panel carries a "
        f"share count, but {n_unsized} returns lack a market cap on their own "
        f"session or the one before it"
    )

    reg = pd.read_parquet(REPO / "data" / "external" / "registry"
                          / "twn_stock_info.parquet")
    em = reg[reg["type"] == "emerging"]
    retired = pd.to_datetime(em["date"]).groupby(em["stock_id"].astype(str)).max()
    early = {s: ret.loc[:d, s].notna().sum() for s, d in retired.items()
             if s in ret.columns}
    early = {s: int(n) for s, n in early.items() if n}
    assert not early, (
        f"si:settings excludes emerging-board sessions, but {len(early)} "
        f"Taiwanese stocks carry returns on or before the date the registry "
        f"retired their emerging-board row: {dict(list(early.items())[:5])}"
    )


def test_taiwan_panel_spans_the_korean_sample():
    """subsec:data_description: the Taiwanese panel covers the common stocks
    'listed over the same period' as the Korean one, which the abstract also
    prints.

    The two spans were once different and the supplement carried a note saying
    why. Asserted on the panels rather than on the macros, so a rebuild that
    shortened one market would fail here rather than silently re-date the
    abstract."""
    kr = pd.read_parquet(DATA / "cache" / "returns_panel.parquet").index
    tw = pd.read_parquet(DATA / "cache" / "tw_returns_panel.parquet").index
    assert kr.min().year == tw.min().year and kr.max().year == tw.max().year, (
        "The abstract prints one span for both panels (subsec:data_description) "
        f"but Korea runs {kr.min().date()}--{kr.max().date()} and Taiwan "
        f"{tw.min().date()}--{tw.max().date()}"
    )


def test_taiwan_market_cap_carries_no_nonpositive_size():
    """Pipeline integrity: every Taiwanese market capitalization the quintile
    cut reads is positive.

    A zero reaches the quintile cut where a NaN does not, because
    ``assign_quintiles_at`` drops missing values and ranks everything else. A
    market cap of zero therefore ranks its stock smallest and puts it in Q1,
    which is a size the stock does not have. The vendor produces zeros two
    ways, an untraded session's close and a zero share count, and the builder
    removes both; this asserts the published panel carries none, which is the
    property, rather than a bound on how many the builder had to remove."""
    mc = pd.read_parquet(DATA / "cache" / "tw_marketcap_panel.parquet").to_numpy()
    present = np.isfinite(mc)
    offenders = int((present & (mc <= 0)).sum())
    assert offenders == 0, (
        f"The Taiwanese market-cap panel carries {offenders:,} non-positive "
        f"cells, which the quintile cut would rank smallest"
    )
    # The same property on the Korean panel, whose market cap is the exchange's
    # own published Marcap rather than a product this repository forms.
    kr = pd.read_parquet(DATA / "cache" / "marketcap_panel.parquet").to_numpy()
    present_kr = np.isfinite(kr)
    offenders_kr = int((present_kr & (kr <= 0)).sum())
    assert offenders_kr == 0, (
        f"The Korean market-cap panel carries {offenders_kr:,} non-positive "
        f"cells, which the quintile cut would rank smallest"
    )


# ---------------------------------------------------------------------------
# subsec:methods, subsec:inference: the committed draw-seed set
# ---------------------------------------------------------------------------

_STEPS = ((1, 2), (2, 3), (3, 4), (4, 5), (1, 5))
_Q_NAME = {1: "Qone", 2: "Qtwo", 3: "Qthree", 4: "Qfour", 5: "Qfive"}
_MARKET_DIRS = {"Kr": "korea_headline", "Tw": "taiwan"}


def test_every_cell_is_the_mean_of_the_committed_draws():
    """subsec:methods: "up to \\paperPerQuintile stocks are drawn from each
    quintile without replacement. [...] Every statistic of a window and quintile
    is the mean of its values over \\nDrawSeeds draws." The draws are the
    committed set DRAW_SEEDS of constants.py, and each cell of both markets is checked to be
    the mean of its own draws."""
    from fn_percolation.constants import DRAW_SEEDS, N_DRAW_SEEDS
    assert int(_params_macros()["nDrawSeeds"]) == N_DRAW_SEEDS == len(DRAW_SEEDS)
    for subdir in _MARKET_DIRS.values():
        per_seed = pd.read_csv(DATA / subdir / "rolling_percolation_per_seed.csv",
                               parse_dates=["date"])
        assert sorted(per_seed["seed"].unique()) == sorted(DRAW_SEEDS), (
            f"subsec:methods averages each cell over the committed draws, but "
            f"{subdir} holds the draws {sorted(per_seed['seed'].unique())}"
        )
        means = per_seed.groupby(["quintile", "date"])["rho_c"].mean()
        for g in range(1, 6):
            cell = _per_q(subdir, g)
            assert (cell["n_seeds"] == N_DRAW_SEEDS).all(), (
                f"subsec:methods averages every cell over {N_DRAW_SEEDS} draws, "
                f"but {subdir} Q{g} has cells over {sorted(cell['n_seeds'].unique())}"
            )
            # The per-draw file keeps six significant digits (pipeline.py), so
            # the mean of its values matches the cell to about 1e-6.
            gap = (cell["rho_c"] - means.loc[g].reindex(cell.index)).abs().max()
            assert gap < 5e-6, (
                f"subsec:methods: the rho_c of a cell is the mean over its draws, "
                f"but {subdir} Q{g} departs from that mean by {gap:.2e}"
            )


def test_spread_of_each_step_across_the_draws():
    """tab:step_robustness prints each step's SD across the \\nDrawSeeds draws. These SDs are the spread that the seed set yields,
    which the suite asserts, recomputed here from
    the per-draw files."""
    macros = _params_macros()
    for mk, subdir in _MARKET_DIRS.items():
        per_seed = pd.read_csv(DATA / subdir / "rolling_percolation_per_seed.csv")
        wide = per_seed.pivot_table(index=["seed", "date"], columns="quintile",
                                    values="rho_c")
        for lo, hi in _STEPS:
            sd = float((wide[hi] - wide[lo]).groupby(level="seed").mean().std(ddof=1))
            name = f"sdDraw{mk}{_Q_NAME[lo]}{_Q_NAME[hi]}"
            assert math.isclose(sd, float(macros[name]), abs_tol=0.00005), (
                f"tab:step_robustness prints {macros[name]} as the SD across the "
                f"draws of the {mk} Q{lo}--Q{hi} step, but the draws give {sd:.5f}"
            )


def test_every_verdict_holds_at_longer_bandwidths():
    """subsec:main_result: "In Korea, the tests with both the bandwidth and the
    block length set to \\bwShort windows, and with both set to \\bwLong
    windows, find the same steps significant". subsec:taiwan: the same of "the
    same Taiwanese steps". subsec:inference: "each test
    is repeated with the bandwidth and the block length both raised from
    \\nwLagHeadline and \\blockBootLenHeadline windows to \\bwShort windows, and
    again with both raised to \\bwLong windows".
    Significant is read as in subsec:inference, on both
    statistics, and one cell of the file is recomputed from the headline."""
    steps = pd.read_csv(DATA / "bandwidth_sensitivity" / "steps.csv",
                        dtype={"bandwidth": str})
    macros = _params_macros()
    assert sorted(steps.loc[steps["bandwidth"] != "automatic", "bandwidth"]
                  .astype(int).unique()) == [int(macros["bwShort"]), int(macros["bwLong"])]
    for (market, pair), grp in steps.groupby(["market", "pair"]):
        verdicts = grp.set_index("bandwidth")["significant"]
        assert verdicts.nunique() == 1, (
            f"Manuscript claims every verdict holds at the longer bandwidths "
            f"(subsec:main_result) but the {market} {pair} verdicts are "
            f"{verdicts.to_dict()}"
        )
    row = steps[(steps["market"] == "korea") & (steps["pair"] == "Q2-Q3")
                & (steps["bandwidth"] == macros["bwLong"])].iloc[0]
    redo = hac_block_stats(_adjacent_diff(2), min_lag=int(macros["bwLong"]),
                           min_block=int(macros["bwLong"]))
    assert math.isclose(redo["nw_t"], float(row["hac_t"]), abs_tol=1e-9), (
        "data/bandwidth_sensitivity/steps.csv does not match the headline windows"
    )


# ---------------------------------------------------------------------------
# subsec:checks, subsec:main_result: rho_c from weekly returns on long windows
# ---------------------------------------------------------------------------

def _weekly_steps() -> pd.DataFrame:
    return pd.read_csv(DATA / "weekly_rho_c" / "steps.csv")


def test_weekly_threshold_design():
    """subsec:checks: "The weekly-threshold check locates rho_c itself in weekly
    returns, on windows of \\wkWindowShort and \\wkWindowLong trading days, which
    hold a median of \\wkMedianWeeksShort and \\wkMedianWeeksLong weekly returns
    ... the bandwidth and the block are raised to \\wkLagShort and \\wkLagLong
    windows, one more than the largest lag at which two windows share
    returns." """
    from fn_percolation.overlap_inference import overlap_span
    macros = _params_macros()
    steps = _weekly_steps()
    for wn in ("Short", "Long"):
        window = int(macros[f"wkWindow{wn}"])
        cells = pd.read_csv(DATA / "weekly_rho_c" / f"cells_korea_T{window}.csv")
        assert int(round(cells["n_weeks"].median())) == int(macros[f"wkMedianWeeks{wn}"])
        rows = steps[steps["window"] == window]
        span = overlap_span(window, STEP) + 1
        assert (rows["nw_lag"] == span).all() and (rows["block_len"] == span).all() \
            and span == int(macros[f"wkLag{wn}"]), (
            f"subsec:checks raises the bandwidth and the block to {span} windows "
            f"at T = {window}, but the check used {rows['nw_lag'].unique()} and "
            f"{rows['block_len'].unique()}"
        )


def test_weekly_threshold_readings():
    """subsec:main_result: "rho_c located in Korean weekly returns on windows of
    \\wkWindowShort and \\wkWindowLong days keeps a significant endpoint
    contrast at both lengths ... Weekly returns reproduce the Q1--Q2 step at
    both lengths, the Q2--Q3 step only at \\wkWindowLong days, and the Q3--Q4
    step only at \\wkWindowShort days ... At each length at which weekly returns
    do not reproduce a step, the weekly test detects a step of the daily step's
    size with \\detectPowerPct power ... On the windows of \\wkWindowShort and \\wkWindowLong
    days, the daily Korean Q4--Q5 step is significant, unlike on the
    \\paperWindowDays-day windows of the main estimate. The weekly Q4--Q5 step is
    not significant at either length, although the weekly test detects a step
    of the daily step's size with \\detectPowerPct power." subsec:taiwan:
    "Weekly returns reproduce the Taiwanese endpoint contrast and each
    Taiwanese step through Q4 on both window lengths of the weekly-threshold
    check." The abstract: "In weekly returns, which stale prices affect less,
    each Taiwanese step through the fourth and, in both markets, the
    difference between the two extreme quintiles are significant, whereas not
    every Korean step is significant there or with other estimation settings."
    sec:introduction: "In weekly returns, each Taiwanese step through the
    fourth quintile and, in both markets, the difference between the smallest
    and the largest quintile are significant, but not every Korean step is, so weekly
    returns do not exclude stale prices as the source of the Korean steps from
    the second to the fourth quintile." Each daily step that is significant and that weekly
    returns do not reproduce must exceed the weekly detection limit at the same
    length, because the main text says the weekly test detects it. A step is reproduced when the weekly step is significant
    on both statistics (run_weekly_rho_c.py, criteria)."""
    macros = _params_macros()
    short, long_ = int(macros["wkWindowShort"]), int(macros["wkWindowLong"])
    st = _weekly_steps().set_index(["market", "window", "pair"])

    def weekly(market, window, pair):
        return bool(st.loc[(market, window, pair), "weekly_significant"])

    for market in ("korea", "taiwan"):
        for window in (short, long_):
            assert weekly(market, window, "Q1-Q5"), (
                f"Manuscript claims a significant weekly endpoint contrast in "
                f"{market} at T = {window} (subsec:main_result)"
            )
    for window in (short, long_):
        for pair in ("Q1-Q2", "Q2-Q3", "Q3-Q4"):
            assert weekly("taiwan", window, pair), (
                f"Manuscript claims weekly returns reproduce the Taiwanese {pair} "
                f"step at T = {window} (subsec:main_result)"
            )
    both = {pair for pair in ("Q1-Q2", "Q2-Q3", "Q3-Q4", "Q4-Q5", "Q1-Q5")
            if weekly("korea", short, pair) and weekly("korea", long_, pair)}
    assert both == {"Q1-Q2", "Q1-Q5"}, (
        f"Manuscript claims that only the first Korean step and the endpoint "
        f"contrast are reproduced at both lengths (subsec:main_result) but the set is {both}"
    )
    assert weekly("korea", long_, "Q2-Q3") and not weekly("korea", short, "Q2-Q3"), (
        "Manuscript claims the Korean Q2--Q3 step is reproduced only on the longer windows"
    )
    assert weekly("korea", short, "Q3-Q4") and not weekly("korea", long_, "Q3-Q4"), (
        "Manuscript claims the Korean Q3--Q4 step is reproduced only on the shorter windows"
    )
    for window in (short, long_):
        row = st.loc[("korea", window, "Q4-Q5")]
        assert row["daily_significant"] and not row["weekly_significant"], (
            f"The Supplementary Material claims the Korean Q4--Q5 step is significant "
            f"in daily but not in weekly returns at T = {window}"
        )
    missed = st[st["daily_significant"] & ~st["weekly_significant"]]
    assert missed["weekly_resolved"].all(), (
        "Manuscript claims every daily step that weekly returns do not reproduce "
        "exceeds the weekly detection limit, but these do not: "
        f"{missed[~missed['weekly_resolved']].index.tolist()}"
    )


def test_weekly_threshold_rows_of_the_contrast_table():
    """tab:robustness_contrast: "Weekly threshold, T = \\wkWindowShort" reads
    'Q4--Q5 equal' and "Weekly threshold, T = \\wkWindowLong" reads 'ordered', the
    five weekly means read at three decimals as the caption reads every row."""
    macros = _params_macros()
    for wn, label in (("Short", "Q4--Q5 equal"), ("Long", "ordered")):
        cells = pd.read_csv(DATA / "weekly_rho_c"
                            / f"cells_korea_T{macros[f'wkWindow{wn}']}.csv")
        means = [float(f"{m:.3f}") for m in
                 cells.groupby("quintile")["weekly_rho_c"].mean().sort_index()]
        steps = list(zip(means, means[1:]))
        if label == "ordered":
            ok = all(a < b for a, b in steps)
        else:
            ok = all(a < b for a, b in steps[:3]) and means[3] == means[4]
        assert ok, (
            f"tab:robustness_contrast marks the weekly-threshold row at T = "
            f"{macros[f'wkWindow{wn}']} '{label}' but its means are {means}"
        )


def test_kospi_only_interior_steps_are_unresolved():
    """subsec:korean_middle_steps: "Neither the KOSPI-only Q2--Q3 step nor
    the KOSPI-only Q3--Q4 step is significant, but the KOSPI-only test detects
    with 80% power only a step larger than the Q2--Q3 and Q3--Q4 steps of the
    main estimate".
    subsec:size_confounds: "The KOSPI-only test detects
    either of these two steps with 80% power only if the step is at least
    \\kospiDetectMin, which exceeds the main estimate's Q2--Q3 and Q3--Q4 steps
    of \\rhoCStepHeadlineQtwoQthree and \\rhoCStepHeadlineQthreeQfour". The detection limit is that of
    run_weekly_rho_c.py, 2.49 HAC standard errors."""
    macros = _params_macros()
    sig = pd.read_csv(DATA / "kospi_only" / "pair_significance.csv").set_index("pair")
    for pair in ("Q3-Q2", "Q4-Q3"):
        assert not (sig.loc[pair, "nw_p_one_sided"] < 0.05
                    and sig.loc[pair, "block_boot_p_one_sided"] < 0.05), (
            f"Manuscript claims the KOSPI-only {pair} step is not significant"
        )
    panel = pd.concat({g: _per_q("kospi_only", g)["rho_c"] for g in range(1, 6)},
                      axis=1).dropna()
    limits = [2.487 * hac_mean_se((panel[hi] - panel[lo]).to_numpy())["se"]
              for lo, hi in ((2, 3), (3, 4))]
    printed = float(macros["kospiDetectMin"])
    assert printed <= min(limits) < printed + 0.001, (
        f"Manuscript prints {printed} as the smaller KOSPI-only detection limit "
        f"but the limits are {limits}"
    )
    head = [float(macros[m]) for m in ("rhoCStepHeadlineQtwoQthree",
                                        "rhoCStepHeadlineQthreeQfour")]
    assert min(limits) > max(head), (
        f"Manuscript claims the KOSPI-only detection limits exceed the main "
        f"estimate's steps {head}, but the limits are {limits}"
    )


def test_abstract_fits_the_journal_limit():
    """Physica A's guide for authors limits the abstract to 250 words. Each
    whitespace-separated token of the source counts as a word, so a range such
    as \\paperSampleStart--\\paperSampleEnd counts once, as it prints."""
    text = (PAPER / "paper.tex").read_text()
    body = text[text.index(r"\begin{abstract}") + len(r"\begin{abstract}"):
                text.index(r"\end{abstract}")].replace(r"\noindent", "")
    n = len(body.split())
    assert n <= 250, f"The abstract holds {n} words, above Physica A's limit of 250"


# ---------------------------------------------------------------------------
# Steps of the arguments of subsec:toy_model, subsec:network_size and
# si:variance_derivative. The module docstring names the steps that no
# computation decides.
# ---------------------------------------------------------------------------

# Network sizes at which each identity in W is decided: an identity in the N x N
# matrix W is decided at each N, not for all N at once (subsec:toy_model).
ARG_N_RANGE = range(2, 6)


def _toy_model(n: int):
    """Symbols of eq:toy_rgp for n stocks, and the covariance that the model
    gives by bilinearity: r_i = beta_i f + sum_k C_ik u_k, with C_ii = delta and
    C_ik = (1 - delta) W_ik, where f and the u_k are independent with variances
    sigma_f^2 and sigma_u^2 (subsec:toy_model)."""
    d, sf, su = sp.symbols("delta sigma_f sigma_u", positive=True)
    beta = sp.symbols(f"beta0:{n}", real=True)
    W = sp.Matrix(n, n, lambda i, j: 0 if i == j else sp.Symbol(f"W_{i}_{j}", nonnegative=True))
    C = sp.Matrix(n, n, lambda i, k: d if i == k else (1 - d) * W[i, k])
    cov = sp.Matrix(n, n, lambda i, j: beta[i] * beta[j] * sf**2
                    + su**2 * sum(C[i, k] * C[j, k] for k in range(n)))
    return d, sf, su, beta, W, cov


def _rises_on(expr, x, interval) -> bool:
    """True where the derivative of expr in x is positive at every point of the interval."""
    return sp.solveset(sp.diff(expr, x) > 0, x, interval) == interval


def _falls_on(expr, x, interval) -> bool:
    """True where the derivative of expr in x is negative at every point of the interval."""
    return sp.solveset(sp.diff(expr, x) < 0, x, interval) == interval


def test_toy_covariance_and_variance_follow_from_the_model():
    """subsec:toy_model, eq:toy_cov and eq:toy_var: the covariance of two
    returns of eq:toy_rgp and the variance of each return, decided as
    polynomial identities at each N of ARG_N_RANGE."""
    for n in ARG_N_RANGE:
        d, sf, su, beta, W, cov = _toy_model(n)
        for i in range(n):
            var = (beta[i]**2 * sf**2 + d**2 * su**2
                   + (1 - d)**2 * sum(W[i, k]**2 for k in range(n)) * su**2)
            assert sp.expand(cov[i, i] - var) == 0, (
                f"eq:toy_var does not give the variance of eq:toy_rgp at N = {n}")
            for j in range(i + 1, n):
                printed = (beta[i] * beta[j] * sf**2
                           + d * (1 - d) * (W[i, j] + W[j, i]) * su**2
                           + (1 - d)**2 * sum(W[i, k] * W[j, k] for k in range(n)
                                              if k not in (i, j)) * su**2)
                assert sp.expand(cov[i, j] - printed) == 0, (
                    f"eq:toy_cov does not give the covariance of eq:toy_rgp at N = {n}")


def test_toy_covariance_parts_depend_on_delta_as_printed():
    """subsec:toy_model, after eq:toy_cov: 'a factor part that does not depend
    on delta, a direct-link part that rises with delta up to delta = 1/2 and
    falls beyond it, and a shared-peer part that falls with delta', read on the
    coefficient of delta in each part. 'Above delta = 1/2, no part of the
    covariance rises with delta' reads the same coefficients on (1/2, 1)."""
    x = sp.Symbol("x", real=True)
    d, sf, su, beta, W, cov = _toy_model(3)
    assert sp.diff(beta[0] * beta[1] * sf**2, d) == 0, (
        "subsec:toy_model claims the factor part does not depend on delta")
    below, above = sp.Interval.open(0, sp.Rational(1, 2)), sp.Interval.open(sp.Rational(1, 2), 1)
    assert _rises_on(x * (1 - x), x, below) and _falls_on(x * (1 - x), x, above), (
        "subsec:toy_model claims the direct-link coefficient delta(1-delta) rises "
        "up to delta = 1/2 and falls beyond it")
    assert _falls_on((1 - x)**2, x, sp.Interval.open(0, 1)), (
        "subsec:toy_model claims the shared-peer coefficient (1-delta)^2 falls with delta")


def test_toy_bulk_correlation_numerator_and_division():
    """subsec:toy_model, eq:toy_bulk: a pair with no direct link and no factor
    part has covariance (1-delta)^2 b_ij sigma_u^2 with b_ij = sum_k W_ik W_jk,
    decided at each N of ARG_N_RANGE. 'Dividing the numerator and each variance
    by (1-delta)^2 leaves the numerator at b_ij sigma_u^2 and turns each
    variance into sum_k W_ik^2 sigma_u^2 + (beta_i^2 sigma_f^2 + delta^2
    sigma_u^2)/(1-delta)^2', a rational identity. The two functions of delta in
    that sum, 1/(1-delta)^2 and delta^2/(1-delta)^2, rise on (0, 1)."""
    for n in ARG_N_RANGE:
        d, sf, su, beta, W, cov = _toy_model(n)
        no_link = {W[0, 1]: 0, W[1, 0]: 0, beta[0]: 0, beta[1]: 0}
        b01 = sum(W[0, k] * W[1, k] for k in range(n)).subs(no_link)
        assert sp.expand(cov[0, 1].subs(no_link) - (1 - d)**2 * b01 * su**2) == 0, (
            f"eq:toy_bulk does not give the covariance of a pair without a direct "
            f"link and factor part at N = {n}")
        divided = sp.simplify(cov[0, 0] / (1 - d)**2
                              - (sum(W[0, k]**2 for k in range(n)) * su**2
                                 + (beta[0]**2 * sf**2 + d**2 * su**2) / (1 - d)**2))
        assert divided == 0, (
            f"subsec:toy_model's division of the variance by (1-delta)^2 fails at N = {n}")
    x = sp.Symbol("x", real=True)
    unit = sp.Interval.open(0, 1)
    assert _rises_on(1 / (1 - x)**2, x, unit) and _rises_on(x**2 / (1 - x)**2, x, unit), (
        "subsec:toy_model claims each divided variance rises with delta on (0, 1)")


def test_variance_derivative_of_si():
    """si:variance_derivative: 'Each variance of main-paper Eq. eq:toy_var has
    the derivative 2 sigma_u^2 [delta - (1-delta) sum_k W_ik^2] in delta',
    decided at each N of ARG_N_RANGE, and 'For delta >= 1/2, ... delta >=
    1-delta', decided on [1/2, 1)."""
    for n in ARG_N_RANGE:
        d, sf, su, beta, W, cov = _toy_model(n)
        printed = 2 * su**2 * (d - (1 - d) * sum(W[0, k]**2 for k in range(n)))
        assert sp.expand(sp.diff(cov[0, 0], d) - printed) == 0, (
            f"si:variance_derivative prints a derivative of eq:toy_var that fails at N = {n}")
    x = sp.Symbol("x", real=True)
    half_open = sp.Interval.Ropen(sp.Rational(1, 2), 1)
    assert sp.solveset(x - (1 - x) >= 0, x, half_open) == half_open, (
        "si:variance_derivative claims delta >= 1 - delta for delta >= 1/2")


def test_gram_identity_of_propagation_and_factor_model():
    """subsec:toy_model: 'The part of the covariance that Eq. eq:toy_rgp
    produces without f is sigma_u^2 M M^T with M = delta I + (1-delta) W, which
    is the covariance that a linear factor model with loading matrix sigma_u M
    produces', decided at each N of ARG_N_RANGE."""
    for n in ARG_N_RANGE:
        d, sf, su, beta, W, cov = _toy_model(n)
        M = d * sp.eye(n) + (1 - d) * W
        without_f = cov - sf**2 * sp.Matrix(n, 1, list(beta)) * sp.Matrix(1, n, list(beta))
        assert sp.expand(without_f - su**2 * M * M.T) == sp.zeros(n, n), (
            f"subsec:toy_model's identity sigma_u^2 M M^T fails at N = {n}")
        loading = su * M
        assert sp.expand(loading * loading.T - su**2 * M * M.T) == sp.zeros(n, n), (
            f"subsec:toy_model's factor model with loading sigma_u M fails at N = {n}")


def test_threshold_quantile_and_network_size_rate():
    """subsec:toy_model: 'Eq. eq:toy_threshold puts tilde-rho_c at the fixed
    quantile 1 - 1/(N-1) of F_delta', and with a normal F_delta it gives
    eq:toy_gauss, 1 - Phi(z) = 1/(N-1). subsec:network_size: 'At fixed delta,
    tilde-rho_c therefore rises with N at the rate sigma_delta per unit of
    sqrt(2 ln(N-1))', read on eq:toy_closed_form, whose sqrt(2 ln(N-1)) rises
    with N for N > 2."""
    N, F, Phi, x, mu, s = sp.symbols("N F Phi x mu sigma_delta", positive=True)
    (root,) = sp.solve(sp.Eq((N - 1) * (1 - F), 1), F)
    assert sp.simplify(root - (1 - 1 / (N - 1))) == 0, (
        "subsec:toy_model claims eq:toy_threshold puts tilde-rho_c at the quantile 1 - 1/(N-1)")
    # A normal F_delta gives F_delta(tilde-rho_c) = Phi(z); Phi stands for Phi(z).
    (phi_root,) = sp.solve(sp.Eq((N - 1) * (1 - Phi), 1), Phi)
    assert sp.simplify((1 - phi_root) - 1 / (N - 1)) == 0, (
        "subsec:network_size claims eq:toy_threshold becomes eq:toy_gauss")
    assert sp.diff(mu + s * x, x) == s, (
        "subsec:network_size claims eq:toy_closed_form rises at the rate sigma_delta "
        "per unit of sqrt(2 ln(N-1))")
    n = sp.Symbol("n", real=True)
    above_two = sp.Interval.open(2, sp.oo)
    assert _rises_on(sp.sqrt(2 * sp.log(n - 1)), n, above_two), (
        "subsec:network_size claims sqrt(2 ln(N-1)) rises with N")


def test_leading_order_root_lies_above_the_normal_quantile():
    """si:eq9_verification: the closed form 'keeps only the leading-order root
    of main-paper Eq. eq:toy_gauss, which lies above the exact normal quantile',
    decided at each network size from 3 to 5000, which holds every network size
    that the paper uses (the simulation's \\dgpN included)."""
    sizes = np.arange(3, 5001)
    leading = np.sqrt(2 * np.log(sizes - 1))
    exact = norm.isf(1 / (sizes - 1))
    used = [int(_params_macros()["dgpN"])] + [
        int(np.ceil(_per_q(name, g)["n_stocks"].max()))
        for name in ("korea_headline", "taiwan") for g in range(1, 6)]
    assert max(used) <= sizes[-1], (
        f"A network of {max(used)} stocks lies outside the sizes that this test decides")
    below = sizes[leading <= exact]
    assert below.size == 0, (
        "si:eq9_verification claims the leading-order root sqrt(2 ln(N-1)) lies "
        f"above the exact normal quantile, but it does not at N = {below[:5]}")
