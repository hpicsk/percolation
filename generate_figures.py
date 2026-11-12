#!/usr/bin/env python3
"""Generate figures for the percolation paper."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from fn_percolation.overlap_inference import hac_mean_se

RESULTS = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0.1,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

GRAY1, GRAY2, GRAY3, GRAY4, GRAY5 = "#1a1a1a", "#444444", "#666666", "#999999", "#bbbbbb"
GRAYS5 = [GRAY1, GRAY2, GRAY3, GRAY4, GRAY5]


# fig:rolling_rho_c smoothing window: number of rolling steps (each ~20 trading days,
# subsec:researcher_choices) in the centred moving average. 6 steps ~ 120 trading days ~ half a
# year. Mirrors \rollingMASteps in params.tex (manual region).
MA_WINDOW = 6


def fig_headline_gradient():
    """Cross-window mean rho_c by size quintile, Korea and Taiwan side by side.

    The headline claim of subsec:main_result and subsec:taiwan as one picture.
    Korea rises strictly across all five quintiles; Taiwan rises more steeply
    through Q4 and turns over at Q5, the top-of-distribution step that does not
    replicate across the two markets.

    Error bars are one Newey-West HAC standard error of the *paired* per-window
    difference from Q1, which is the quantity the significance tests of
    tab:pair_significance read (same hac_mean_se long-run variance). Pairing
    matters here: the five quintiles share rolling windows, so the common
    time-variation cancels in the difference, and a standard error on each
    quintile's level would be several times wider than the tested contrast and
    would understate the resolution of the gradient. Q1 carries no bar by
    construction, being the reference.

    The two panels share a y-axis so the Korean and Taiwanese gradients are
    directly comparable in level and in slope.
    """
    markets = [("korea_headline", "Korea (KOSPI and KOSDAQ)", "o", "-"),
               ("taiwan", "Taiwan (TWSE and TPEx)", "s", "--")]
    qx = np.arange(1, 6)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, (folder, title, marker, ls) in zip(axes, markets):
        series = {}
        for g in qx:
            path = RESULTS / folder / f"rolling_percolation_Q{g}.csv"
            if not path.exists():
                plt.close(fig)
                raise FileNotFoundError(
                    f"fig_headline_gradient (fig:headline_gradient) needs {path}, "
                    f"which does not exist. Run the stage that writes it before "
                    f"this script; skipping would leave the committed PDF in "
                    f"place and republish an older run.")
            df = pd.read_csv(path, parse_dates=["date"])
            series[g] = df.set_index("date")["rho_c"]
        means, errs = [], []
        for g in qx:
            means.append(float(series[g].mean()))
            # Paired difference from Q1 on the shared window grid.
            paired = pd.concat([series[g], series[1]], axis=1, join="inner")
            errs.append(hac_mean_se((paired.iloc[:, 0] - paired.iloc[:, 1]).to_numpy())["se"]
                        if g > 1 else 0.0)
        ax.errorbar(qx, means, yerr=errs, color=GRAY1, marker=marker,
                    linestyle=ls, linewidth=1.6, markersize=6, capsize=4,
                    elinewidth=1.0)
        ax.set_xticks(qx)
        ax.set_xticklabels(["Q1\n(small)", "Q2", "Q3", "Q4", "Q5\n(large)"])
        ax.set_xlim(0.5, 5.5)
        ax.set_xlabel("Market capitalization quintile")
        ax.set_title(title, fontsize=11)
    axes[0].set_ylabel(r"Cross-window mean $\bar\rho_c$")
    fig.tight_layout()
    fig.savefig(OUT / "fig_headline_gradient.pdf")
    plt.close(fig)
    print("  Saved fig_headline_gradient.pdf")


def fig_rolling_rho_c():
    """Rolling percolation threshold for Q1 and Q5 over the sample, with crisis
    bands. Faint traces are the raw 60-day rolling series; bold lines are a
    centred MA_WINDOW-step (~half-year) moving average. Only the endpoint
    quintiles are drawn, matching the caption; the full per-quintile gradient
    is in Table 1 (tab:percolation)."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    # Q1 solid, Q5 dashed: the line style (not color) carries the contrast, so
    # the figure stays legible in grayscale (paper figure-style convention).
    styles = {1: ("-", "Q1 (small)"), 5: ("--", "Q5 (large)")}
    for q in (1, 5):
        path = RESULTS / "korea_headline" / f"rolling_percolation_Q{q}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        ls, label = styles[q]
        ma = df["rho_c"].rolling(MA_WINDOW, center=True, min_periods=3).mean()
        ax.plot(df.index, df["rho_c"], color=GRAY1, linewidth=0.6, alpha=0.20)
        ax.plot(df.index, ma, color=GRAY1, linewidth=2.0, linestyle=ls,
                label=label)

    ax.set_xlabel("Date")
    ax.set_ylabel(r"Percolation threshold $\rho_c$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              frameon=False)
    fig.savefig(OUT / "fig_rolling_rho_c.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_rolling_rho_c.pdf")


def fig_rho_c_vs_mean():
    """rho_c against segment mean |rho_ij|, and what conditioning on it leaves.

    Left panel: per-(window, quintile) scatter of rho_c on the same cell's
    off-diagonal mean |rho_ij|, one marker per quintile, with the pooled cubic
    fit (the conditioning model used in the SI) and the pooled Spearman rank
    correlation annotated. Right panel: per-quintile mean rho_c demeaned by the
    grand mean (rising Q1->Q5, the headline gradient) against the per-quintile
    mean residual after the cubic fit. The contrast is the point: the size
    ordering keeps its direction after conditioning but shrinks to a fraction of
    the raw span, so most of the cross-quintile gradient sits in the first
    moment (SI Sec. on rho_c vs mean correlation,
    test_meancorr_residual_gradient_is_a_fraction_of_the_raw_one). Mirrors that
    test's merge exactly."""
    mc_path = RESULTS / "mean_corr_diagnostic" / "per_window_mean_corr.csv"
    if not mc_path.exists():
        raise FileNotFoundError(
            f"fig_rho_c_vs_mean (fig:si_rho_c_vs_mean) needs {mc_path}, which "
            f"does not exist. Run the stage that writes it before this script; "
            f"skipping would leave the committed PDF in place and republish an "
            f"older run.")
    mc = pd.read_csv(mc_path, parse_dates=["date"])
    parts = []
    for g in (1, 2, 3, 4, 5):
        head_path = RESULTS / "korea_headline" / f"rolling_percolation_Q{g}.csv"
        head = pd.read_csv(head_path, parse_dates=["date"])[["date", "rho_c"]]
        mine = mc[mc["quintile"] == g][["date", "mean_abs_corr"]]
        parts.append(head.merge(mine, on="date").assign(quintile=g))
    long = pd.concat(parts, ignore_index=True).dropna()

    # Pooled cubic fit of rho_c on mean|rho|, identical to the SI test.
    X = np.vander(long["mean_abs_corr"].to_numpy(), 4)
    beta, *_ = np.linalg.lstsq(X, long["rho_c"].to_numpy(), rcond=None)
    long = long.assign(resid=long["rho_c"].to_numpy() - X @ beta)
    spearman = long[["rho_c", "mean_abs_corr"]].corr(method="spearman").iloc[0, 1]

    markers = {1: "o", 2: "s", 3: "^", 4: "D", 5: "v"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))

    for q, color in zip([1, 2, 3, 4, 5], GRAYS5):
        sub = long[long["quintile"] == q]
        label = f"Q{q}" + (" (small)" if q == 1 else
                           " (large)" if q == 5 else "")
        axL.scatter(sub["mean_abs_corr"], sub["rho_c"], s=10, color=color,
                    marker=markers[q], alpha=0.55, edgecolors="none",
                    label=label)
    xs = np.linspace(long["mean_abs_corr"].min(), long["mean_abs_corr"].max(), 200)
    ys = np.vander(xs, 4) @ beta
    axL.plot(xs, ys, color=GRAY1, linewidth=1.8, label="pooled cubic fit")
    axL.text(0.04, 0.95, rf"Spearman $\rho={spearman:.2f}$",
             transform=axL.transAxes, va="top", fontsize=10)
    axL.set_xlabel(r"Segment mean off-diagonal $|\rho_{ij}|$")
    axL.set_ylabel(r"Percolation threshold $\rho_c$")
    axL.legend(loc="lower right", framealpha=0.9, fontsize=8.5, ncol=2)

    raw = long.groupby("quintile")["rho_c"].mean().sort_index()
    rm = long.groupby("quintile")["resid"].mean().sort_index()
    qx = np.arange(1, 6)
    axR.axhline(0.0, color=GRAY4, linewidth=0.8, linestyle=":")
    axR.plot(qx, (raw - raw.mean()).to_numpy(), color=GRAY1, marker="o",
             linewidth=1.6, label=r"raw $\bar\rho_c$ (demeaned)")
    axR.plot(qx, rm.to_numpy(), color=GRAY3, marker="s", linestyle="--",
             linewidth=1.6, label="residual of the cubic")
    axR.set_xticks(qx)
    axR.set_xticklabels([f"Q{q}" for q in qx])
    axR.set_xlabel("Market capitalization quintile")
    axR.set_ylabel(r"Cross-quintile $\rho_c$ component")
    axR.legend(loc="best", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fig_rho_c_vs_mean.pdf")
    plt.close(fig)
    print("  Saved fig_rho_c_vs_mean.pdf")


if __name__ == "__main__":
    print("Generating percolation paper figures...")
    fig_headline_gradient()
    fig_rolling_rho_c()
    fig_rho_c_vs_mean()
    print("Done.")
