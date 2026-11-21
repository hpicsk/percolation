#!/usr/bin/env python3
"""Regenerate the GENERATED region of paper/params.tex from data/ CSVs.

The single source of truth for every numeric value cited in paper.tex is the
regenerated CSV/parquet output under data/. This script reads the relevant
CSVs, formats each value at the precision the paper rounds to, and emits a
deterministically-ordered block of `\\newcommand` macros between the
GENERATED region markers in paper/params.tex.

Run via reproduce.sh; the MANUAL region of params.tex is preserved.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from fn_percolation.constants import (
    MIN_SECTOR_SIZE,
    N_DRAW_SEEDS,
    MIN_VALID_FRACTION,
    PER_QUINTILE,
    RHO_RANGE,
    STEP,
    WINDOW,
)
from fn_percolation.network_utils import SPLINE_MESH_SIZE
from fn_percolation.overlap_inference import (BLOCK_BOOT_DRAWS, block_length,
                                               hac_block_stats, hac_mean_se,
                                               overlap_span)
from fn_percolation.rmt import marchenko_pastur_bounds
from fn_percolation.dgp import (
    DEFAULT_BETA_CLIP,
    DEFAULT_BETA_MEAN,
    DEFAULT_BETA_STD,
    DEFAULT_SIGMA_F,
    DEFAULT_SIGMA_F_SCALE,
    DEFAULT_SIGMA_F_UNSCALED,
    DEFAULT_SIGMA_U,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
PARAMS = REPO_ROOT / "paper" / "params.tex"
HIGHLIGHTS = REPO_ROOT / "paper" / "highlights.txt"

BEGIN = "% === BEGIN GENERATED REGION ==="
END = "% === END GENERATED REGION ==="

QUINTILE_NAMES = {1: "Qone", 2: "Qtwo", 3: "Qthree", 4: "Qfour", 5: "Qfive"}

# The one-sided level at which subsec:inference rejects every contrast.
ONE_SIDED_LEVEL = 0.05


def _fmt(value: float, digits: int) -> str:
    """Format ``value`` to fixed ``digits`` decimal places."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return r"\text{NaN}"
    return f"{value:.{digits}f}"


def _fmt_bound(value: float, digits: int, side: str) -> str:
    """Format a value the manuscript quotes as a strict bound, rounding AWAY from
    the data rather than to nearest.

    Round-to-nearest makes a printed bound tighter than the quantity it bounds
    half the time -- a minimum of 87.151 printed as 87.2 turns "never below
    87.2" into a false sentence -- and the error is invisible because the macro
    is correctly bound to its artifact. ``side`` is "lo" for a lower bound
    (round down) and "hi" for an upper bound (round up)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return r"\text{NaN}"
    scale = 10.0 ** digits
    rounded = (math.floor(value * scale) if side == "lo"
               else math.ceil(value * scale)) / scale
    return f"{rounded:.{digits}f}"


# The 5% level of subsec:inference.
SIGNIFICANCE_LEVEL = 0.05


def _fmt_pval(p: float) -> str:
    """Format a p-value for body text / table notes: two-decimal fixed for
    ``p >= 0.01`` (e.g. 0.74, 0.08, 0.02), one-decimal-mantissa scientific
    (``m.x \\times 10^{e}``) below that. The 0.01 cutoff matches the manuscript
    convention of printing only sub-percent p-values in scientific form; the
    body is used inside math mode. A p-value that two decimals round onto
    SIGNIFICANCE_LEVEL takes a third decimal, so the printed value falls on the
    side of the level on which the value falls."""
    if p >= 0.01:
        body = f"{p:.2f}"
        if float(body) == SIGNIFICANCE_LEVEL and p != SIGNIFICANCE_LEVEL:
            body = f"{p:.3f}"
        return body
    exp = int(np.floor(np.log10(p)))
    mant = round(p / (10.0 ** exp), 1)
    # A mantissa that rounds up to 10.0 (a bootstrap p of 1/(B+1) = 9.999e-5)
    # moves to the next exponent, so the body never prints 10.0 x 10^{e}.
    if mant >= 10.0:
        mant, exp = mant / 10.0, exp + 1
    return f"{mant:.1f} \\times 10^{{{exp}}}"


def _fmt_block_p(p: float) -> str:
    """Format a moving-block bootstrap one-sided p-value (subsec:main_result). At the
    resampling resolution (zero violations in 10,000 resamples the reported
    p ~ 1e-4) report the bound ``10^{-4}``, which prose reads as ``p < ...``;
    otherwise use the standard scientific p-value format."""
    if p < 1.5e-4:
        return "10^{-4}"
    return _fmt_pval(p)


def _fmt_block_p_relation(p: float) -> str:
    """A moving-block bootstrap one-sided p-value with its relation, for prose
    that writes ``$p \\macro$``: ``< 10^{-4}`` at the resampling resolution,
    where _fmt_block_p prints the bound, and ``= `` before the value otherwise.
    The relation is derived here, so a p-value that leaves the resolution
    changes the relation that the prose prints."""
    body = _fmt_block_p(p)
    return ("< " if body == "10^{-4}" else "= ") + body


def _fmt_block_p_cell(p: float) -> str:
    """A moving-block bootstrap p-value for a table cell: ``< 10^{-4}`` at the
    resampling resolution and the value otherwise, so a cell never prints the
    bound without its relation."""
    body = _fmt_block_p(p)
    return "< 10^{-4}" if body == "10^{-4}" else body


def _macro(name: str, body: str) -> str:
    return "\\newcommand{\\" + name + "}{" + body + "\\xspace}"


# Every data/ path this generator successfully opens, recorded at read time and
# emitted as a provenance list at the foot of the generated region. The list is
# what test_assertions.py checks the pipeline's arm set against: an arm whose
# output the generator never opens reaches no reader and no macro-binding
# assertion can see it, because nothing is printed to bind.
READ_PATHS: set[str] = set()


def _record_read(path: Path) -> None:
    """Record a data/ read. Called for every artifact the generator opens,
    whatever its format."""
    if path.is_relative_to(DATA):
        READ_PATHS.add(path.relative_to(DATA).as_posix())


def _read_parquet(path: Path) -> pd.DataFrame:
    """pandas.read_parquet with the read recorded, so a block whose numbers come
    from a cached panel declares that panel as its population like any other."""
    _record_read(path)
    return pd.read_parquet(path)


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    _record_read(path)
    return pd.read_csv(path)


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read an input whose macros have no defined value without it.

    _safe_read_csv returns None and its caller emits nothing, which is the right
    shape for a block the manuscript reads conditionally. A missing input fails
    rather than skips wherever the manuscript prints the
    macro unconditionally: skipping there regenerates params.tex without the
    definition and moves the failure to pdflatex, which names a control
    sequence rather than the file that went missing."""
    if not path.exists():
        sys.exit(f"generate_params.py: missing required input {path}. Run "
                 f"reproduce.sh to regenerate it.")
    _record_read(path)
    return pd.read_csv(path, **kwargs)


def _rolling_arm_dirs() -> list[Path]:
    """Every data/ directory holding a complete rolling-percolation output set.

    Derived from the artifacts at generation time rather than listed: a
    written-out arm list is true on the day it is written and silently false
    afterwards, passing the first time an arm is added, which is the failure it
    exists to prevent."""
    return sorted(
        d for d in DATA.iterdir()
        if d.is_dir()
        and (d / "rolling_percolation_summary.csv").exists()
        and all((d / f"rolling_percolation_Q{g}.csv").exists()
                for g in QUINTILE_NAMES)
    )


# Directory-name token -> macro-name fragment for the residualization arms,
# whose set is discovered rather than listed. An unmapped token is a hard
# failure, not a skip: a new arm must be named before it can ship, and the
# alternative is the arm silently producing no macro.
_EXTERNAL_FACTOR_TOKENS = {
    "market": "Mkt", "sector": "Sec", "residual": "Resid", "ew": "Ew",
}


def _external_factor_stem(dirname: str) -> str:
    """``market_sector_residual_ew`` -> ``MktSecResidEw``."""
    parts = []
    for token in dirname.split("_"):
        if token not in _EXTERNAL_FACTOR_TOKENS:
            sys.exit(f"data/{dirname}: unmapped arm-name token {token!r}. Add it "
                     f"to _EXTERNAL_FACTOR_TOKENS in {Path(__file__).name} so the "
                     f"arm gets a macro family; an unnamed arm reaches no reader.")
        parts.append(_EXTERNAL_FACTOR_TOKENS[token])
    return "".join(parts)


def _residual_arm_dirs() -> list[Path]:
    """The residualization arms of tab:residual_steps, every rolling arm whose
    directory name carries the ``residual`` token. Discovered from data/ rather
    than listed."""
    return [a for a in _rolling_arm_dirs() if "residual" in a.name.split("_")]


def _module_constant(relpath: str, name: str):
    """Read a module-level constant out of a sibling script by importing it.

    A design constant already fixed in code -- a window length, a draw count, a
    floor -- has its source of truth in the script, so the prose follows the
    constant from here rather than the parameter file's manual region fixing the
    same number a second time. A missing name is a hard failure: the macro would
    otherwise keep printing the value the script no longer holds."""
    path = REPO_ROOT / relpath
    spec = importlib.util.spec_from_file_location(f"_paramsrc_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    # A module defining a dataclass must be importable by name while it runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, name):
        sys.exit(f"{relpath} no longer defines {name}, which params.tex reads "
                 f"through scripts/generate_params.py. The macro would go stale "
                 f"silently.")
    return getattr(module, name)


def _design_constant_macros() -> list[tuple[str, str, str]]:
    """Design constants whose source of truth is the code, not this file.

    These are not measurements, so a sweep for results does not find them, and
    they were hand-typed into the manual region with the constant's name and its
    value written out beside each other -- the arrangement that goes stale the
    day the constant moves. Nothing dereferences the comment, so no check saw it.
    Each value is now read from the module that fixes it."""
    header = ("Design constants, read from the code that fixes them "
              "(subsec:methods, subsec:data_description, tab:settings)")
    floors = _module_constant(
        "scripts/confounds/run_within_sector.py", "FLOORS")
    out = [
        (header, "paperWindowDays", f"{WINDOW}"),
        (header, "paperStepDays", f"{STEP}"),
        (header, "paperPerQuintile", f"{PER_QUINTILE}"),
        (header, "minValidFractionPct", _fmt(100.0 * MIN_VALID_FRACTION, 0) + r"\%"),
        (header, "minSectorSize", f"{MIN_SECTOR_SIZE}"),
        # The fewest stocks that a kept sector places in each within-sector
        # quintile: sector_loader cuts the sector's size ranks into equal
        # quintiles, so a sector of MIN_SECTOR_SIZE stocks places
        # MIN_SECTOR_SIZE // 5 in each (subsec:checks).
        (header, "minSectorStocksPerQuintile", f"{MIN_SECTOR_SIZE // len(QUINTILE_NAMES)}"),
        # The sensitivity sweep brackets the published floor; the two bracketing
        # values are its extremes, not a third and fourth constant to fix here.
        (header, "minSectorSizeLo", f"{min(floors)}"),
        (header, "minSectorSizeHi", f"{max(floors)}"),
        # fig:rolling_rho_c display smoothing: a centred moving average over this
        # many steps of \paperStepDays trading days each.
        (header, "rollingMASteps",
         f"{_module_constant('generate_figures.py', 'MA_WINDOW')}"),
    ]
    # The moving-block bootstrap of subsec:inference: its draw count, and the
    # block length its rule gives at the Korean window count.
    out.append((header, "blockBootDraws", f"{BLOCK_BOOT_DRAWS:,}".replace(",", "{,}")))
    # The network-size resampling of subsec:resampling_method: its grid of N
    # and the stride at which it sub-samples the rolling windows.
    n_grid = sorted(_module_constant("scripts/scaling/run_rho_c_vs_N.py", "N_GRID"))
    spacing = {b - a for a, b in zip(n_grid, n_grid[1:])}
    if len(spacing) != 1:
        sys.exit("subsec:methods prints the network-size grid as evenly spaced, "
                 f"but run_rho_c_vs_N.py's N_GRID is {n_grid}.")
    out.append((header, "networkSizeNMin", f"{n_grid[0]}"))
    out.append((header, "networkSizeNMax", f"{n_grid[-1]}"))
    out.append((header, "networkSizeNStep", f"{spacing.pop()}"))
    out.append((header, "networkSizeWindowStride",
                f"{_module_constant('scripts/scaling/run_rho_c_vs_N.py', 'SUBSAMPLE_STEP')}"))
    # Korean panel coverage, read off the realized window dates rather than
    # typed: the panel's bounds are what the loader returned, not a convention.
    head = _safe_read_csv(DATA / "korea_headline" / "rolling_percolation_Q1.csv")
    if head is not None and not head.empty:
        dates = pd.to_datetime(head["date"])
        out.append((header, "paperSampleStart", f"{dates.min().year}"))
        out.append((header, "paperSampleEnd", f"{dates.max().year}"))
        out.append((header, "blockBootLenHeadline", f"{block_length(len(head))}"))
    return out


def _scan_grid_macros() -> list[tuple[str, str, str]]:
    """The rho* scan grid and the spline derivative mesh (subsec:methods,
    tab:settings). Both are design constants whose source of truth is the code
    (RHO_RANGE in constants.py, SPLINE_MESH_SIZE in network_utils.py), so the
    prose follows them from here rather than fixing them a second time."""
    group = "Threshold scan grid (subsec:methods, tab:settings)"
    return [
        (group, "rhoGridPoints", f"{len(RHO_RANGE):,}"),
        (group, "rhoGridLo", _fmt(float(RHO_RANGE[0]), 2)),
        (group, "rhoGridHi", _fmt(float(RHO_RANGE[-1]), 2)),
        (group, "splineMeshPoints", f"{SPLINE_MESH_SIZE:,}"),
    ]


def _headline_macros() -> list[tuple[str, str, str]]:
    """Headline per-quintile rho_c, the Q5-Q1 gap, and per-quintile
    dispersion / median stocks (tab:percolation, subsec:main_result)."""
    out: list[tuple[str, str, str]] = []

    # Headline: per-quintile mean rho_c and Q5-Q1 gap (tab:percolation, subsec:main_result).
    summary = _safe_read_csv(DATA / "korea_headline" / "rolling_percolation_summary.csv")
    if summary is not None:
        summary = summary.sort_values("size_group").reset_index(drop=True)
        for _, row in summary.iterrows():
            g = int(row["size_group"])
            out.append((
                "Headline rho_c (tab:percolation, subsec:main_result)",
                f"barRhoC{QUINTILE_NAMES[g]}",
                _fmt(row["rho_c_mean"], 3),
            ))
        gap = float(summary.iloc[-1]["rho_c_mean"] - summary.iloc[0]["rho_c_mean"])
        out.append((
            "Headline rho_c (tab:percolation, subsec:main_result)",
            "rhoCGapHeadline",
            _fmt(gap, 3),
        ))
        n_windows_q1 = int(summary.iloc[0]["n_windows"])
        out.append((
            "Headline rho_c (tab:percolation, subsec:main_result)",
            "nWindowsHeadline",
            f"{n_windows_q1}",
        ))

    # Headline per-quintile median stocks per window + std/min/max rho_c (tab:percolation).
    if summary is not None:
        for _, row in summary.iterrows():
            g = int(row["size_group"])
            qn = QUINTILE_NAMES[g]
            n_stk = row.get("n_stocks_median", row.get("n_stocks"))
            out.append((
                "Headline rho_c (tab:percolation, subsec:main_result)",
                f"medianStocksHeadline{qn}",
                _fmt(float(n_stk), 0),
            ))
            for col, prefix in [("rho_c_std", "rhoCStdHeadline"),
                                ("rho_c_min", "rhoCMinHeadline"),
                                ("rho_c_max", "rhoCMaxHeadline")]:
                if col in row.index:
                    out.append((
                        "Headline rho_c (tab:percolation, subsec:main_result)",
                        f"{prefix}{qn}",
                        _fmt(float(row[col]), 3),
                    ))

    return out


# The adjacent steps and the endpoint contrast, in the order the tables print them.
PAIRS: tuple[tuple[int, int], ...] = ((1, 2), (2, 3), (3, 4), (4, 5), (1, 5))
MARKET_NAMES = {"korea": "Kr", "taiwan": "Tw"}


def _pair_name(lo: int, hi: int) -> str:
    return QUINTILE_NAMES[lo] + QUINTILE_NAMES[hi]


def _pair_bounds(pair: str) -> tuple[int, int]:
    """``"Q1-Q2"`` -> ``(1, 2)``."""
    lo, hi = pair.split("-")
    return int(lo[1:]), int(hi[1:])


def _draw_seed_macros() -> list[tuple[str, str, str]]:
    """The committed draw-seed set and the spread of each step across its
    draws (subsec:methods, subsec:inference, tab:step_robustness)."""
    header = ("Draw seeds and the spread of each step across the draws "
              "(subsec:methods, subsec:inference, tab:step_robustness)")
    out = [(header, "nDrawSeeds", f"{N_DRAW_SEEDS}")]
    # Which steps subsec:inference finds significant at the automatic bandwidth.
    verdicts = _read_csv(DATA / "bandwidth_sensitivity" / "steps.csv")
    verdicts = verdicts[verdicts["bandwidth"] == "automatic"].set_index(
        ["market", "pair"])["significant"]
    ratios = []
    for market, mk in MARKET_NAMES.items():
        subdir = "korea_headline" if market == "korea" else market
        per_seed = _read_csv(DATA / subdir / "rolling_percolation_per_seed.csv")
        wide = per_seed.pivot_table(index=["seed", "date"], columns="quintile",
                                    values="rho_c")
        for lo, hi in PAIRS:
            # The cross-window mean of the step under each draw, and its SD
            # across the draws.
            step_means = (wide[hi] - wide[lo]).groupby(level="seed").mean()
            sd = float(step_means.std(ddof=1))
            out.append((header, f"sdDraw{mk}{_pair_name(lo, hi)}", _fmt(sd, 4)))
            if bool(verdicts[(market, f"Q{lo}-Q{hi}")]):
                ratios.append(float(step_means.mean()) / sd)
    # tab:step_robustness's text bounds every significant step from below as a
    # multiple of its spread across the draws, so the bound is rounded down.
    out.append((header, "stepDrawSpreadRatioMin", _fmt_bound(min(ratios), 1, "lo")))
    # Draws whose rho_c reaches the upper edge of the rho* scan grid, where the
    # edge bounds rho_c rather than measuring it (subsec:methods).
    kr = _read_csv(DATA / "korea_headline" / "rolling_percolation_per_seed.csv")
    out.append((header, "nDrawsRhoCAtGridEdge",
                str(int((kr["rho_c"] >= float(RHO_RANGE[-1]) - 1e-9).sum()))))
    out.append((header, "nDrawCellsHeadline", f"{len(kr):,}".replace(",", "{,}")))
    return out


def _bandwidth_macros() -> list[tuple[str, str, str]]:
    """The headline tests at the longer bandwidths of
    run_bandwidth_sensitivity.py (subsec:inference, tab:step_robustness)."""
    header = ("Headline tests at longer bandwidths "
              "(subsec:inference, subsec:main_result, tab:step_robustness)")
    lengths = sorted(b for b in _module_constant(
        "scripts/headline/run_bandwidth_sensitivity.py", "BANDWIDTHS") if b)
    if len(lengths) != 2:
        sys.exit("tab:step_robustness prints two longer bandwidths, but "
                 f"run_bandwidth_sensitivity.py's BANDWIDTHS gives {lengths}.")
    names = dict(zip((str(b) for b in lengths), ("Short", "Long")))
    out = [(header, "bwShort", str(lengths[0])), (header, "bwLong", str(lengths[1]))]
    steps = _read_csv(DATA / "bandwidth_sensitivity" / "steps.csv",
                      dtype={"bandwidth": str})
    for _, r in steps[steps["bandwidth"] != "automatic"].iterrows():
        name = (MARKET_NAMES[r["market"]] + names[r["bandwidth"]]
                + _pair_name(*_pair_bounds(r["pair"])))
        out.append((header, f"tHacBw{name}", _fmt(float(r["hac_t"]), 1)))
        out.append((header, f"pBlockBootBw{name}",
                    _fmt_block_p_cell(float(r["block_p"]))))
    return out


def _weekly_macros() -> list[tuple[str, str, str]]:
    """rho_c from daily and weekly returns on the long windows of
    run_weekly_rho_c.py (subsec:checks, subsec:main_result, tab:weekly_rho_c)."""
    header = ("rho_c from weekly returns on long windows "
              "(subsec:checks, subsec:main_result, tab:weekly_rho_c)")
    windows = sorted(_module_constant("scripts/stability/run_weekly_rho_c.py",
                                      "WEEKLY_WINDOWS"))
    if len(windows) != 2:
        sys.exit("tab:weekly_rho_c prints two window lengths, but "
                 f"run_weekly_rho_c.py's WEEKLY_WINDOWS gives {windows}.")
    names = dict(zip(windows, ("Short", "Long")))
    # The resolution rule of the check: a one-sided 5% test at the power that
    # Z_POWER sets, so a detection limit of (Z_ALPHA_ONE_SIDED + Z_POWER) HAC
    # standard errors (subsec:checks, tab:weekly_rho_c).
    z_alpha = _module_constant("scripts/stability/run_weekly_rho_c.py", "Z_ALPHA_ONE_SIDED")
    z_power = _module_constant("scripts/stability/run_weekly_rho_c.py", "Z_POWER")
    out: list[tuple[str, str, str]] = [
        (header, "detectPowerPct", _fmt(100.0 * norm.cdf(z_power), 0) + r"\%"),
        (header, "detectMultiplier", _fmt(z_alpha + z_power, 2)),
    ]
    steps = _read_csv(DATA / "weekly_rho_c" / "steps.csv")
    for window, wn in names.items():
        cells = _read_csv(DATA / "weekly_rho_c" / f"cells_korea_T{window}.csv")
        out.append((header, f"wkWindow{wn}", str(window)))
        out.append((header, f"wkMedianWeeks{wn}",
                    _fmt(float(cells["n_weeks"].median()), 0)))
        lags = steps.loc[steps["window"] == window, "nw_lag"].unique()
        if len(lags) != 1:
            sys.exit(f"run_weekly_rho_c.py used several bandwidths at T={window}: {lags}")
        out.append((header, f"wkLag{wn}", str(int(lags[0]))))
    for _, r in steps.iterrows():
        base = MARKET_NAMES[r["market"]] + names[int(r["window"])]
        pair = _pair_name(*_pair_bounds(r["pair"]))
        for freq, fn in (("daily", "Daily"), ("weekly", "Weekly")):
            out.append((header, f"wk{base}{fn}Step{pair}",
                        f"{float(r[f'{freq}_mean']):+.3f}"))
            out.append((header, f"wk{base}{fn}T{pair}",
                        _fmt(float(r[f"{freq}_hac_t"]), 1)))
            out.append((header, f"wk{base}{fn}P{pair}",
                        _fmt_block_p_cell(float(r[f"{freq}_block_p"]))))
        out.append((header, f"wk{base}Detect{pair}",
                    _fmt(float(r["weekly_detection_limit"]), 3)))
    return out


def _kospi_resolution_macros() -> list[tuple[str, str, str]]:
    """The smallest step that the KOSPI-only check detects at each interior
    step, by the resolution rule of run_weekly_rho_c.py: a one-sided 5% test at
    80% power, on the step's HAC standard error (subsec:main_result)."""
    header = "KOSPI-only resolution of the interior steps (subsec:main_result)"
    z = (_module_constant("scripts/stability/run_weekly_rho_c.py", "Z_ALPHA_ONE_SIDED")
         + _module_constant("scripts/stability/run_weekly_rho_c.py", "Z_POWER"))
    panel = pd.concat(
        {g: _read_csv(DATA / "kospi_only" / f"rolling_percolation_Q{g}.csv",
                      index_col="date")["rho_c"] for g in QUINTILE_NAMES},
        axis=1).dropna()
    # subsec:main_result prints the smaller of the two limits as the size from
    # which the check detects either step.
    limits = [z * hac_mean_se((panel[hi] - panel[lo]).to_numpy())["se"]
              for lo, hi in ((2, 3), (3, 4))]
    return [(header, "kospiDetectMin", _fmt_bound(min(limits), 3, "lo"))]


def _setting_step_macros() -> list[tuple[str, str, str]]:
    """One-sided tests of each Korean step and of the endpoint contrast at each
    window length and observation requirement of the estimation-settings check
    (subsec:researcher_choices, tab:setting_steps). Where the windows of a
    setting share returns at more lags than the automatic bandwidth or block
    length covers, both are raised to one more than the largest such lag, the
    rule of the weekly-threshold check (subsec:checks). The detection limits are
    the steps that the 30-day test and the test of rho_c^chi detect with 80%
    power at the 5% level, by the resolution rule of run_weekly_rho_c.py, which
    subsec:researcher_choices compares with the Q3-Q4 step of the main
    estimate. Each is rounded up, because the comparison needs an upper bound."""
    header = "Steps under the estimation settings (subsec:researcher_choices, tab:setting_steps)"
    z = (_module_constant("scripts/stability/run_weekly_rho_c.py", "Z_ALPHA_ONE_SIDED")
         + _module_constant("scripts/stability/run_weekly_rho_c.py", "Z_POWER"))
    sweeps = _module_constant("scripts/sensitivity/run_pipeline_sensitivity.py", "SWEEPS")
    windows = sweeps["window_sensitivity"].values
    fractions = sweeps["min_valid_fraction_sensitivity"].values
    names = {30: "Wthirty", 120: "Whundredtwenty", 252: "Wtwofifty"}
    if sorted(w for w in windows if w != WINDOW) != sorted(names):
        sys.exit(f"tab:setting_steps prints the window lengths {sorted(names)}, "
                 f"but run_pipeline_sensitivity.py sweeps {windows}.")
    settings = [(names[w], DATA / "window_sensitivity" / f"W{w}", w)
                for w in windows if w != WINDOW]
    settings += [("ObsReqLo", DATA / "min_valid_fraction_sensitivity" / f"F{min(fractions)}", WINDOW),
                 ("ObsReqHi", DATA / "min_valid_fraction_sensitivity" / f"F{max(fractions)}", WINDOW)]
    pairs = ((1, 2), (2, 3), (3, 4), (4, 5), (1, 5))
    out: list[tuple[str, str, str]] = []
    for name, folder, window in settings:
        panel = pd.concat(
            {g: _read_csv(folder / f"rolling_percolation_Q{g}.csv",
                          index_col="date")["rho_c"] for g in QUINTILE_NAMES},
            axis=1).dropna()
        lag = overlap_span(window, STEP) + 1
        for lo, hi in pairs:
            stats = hac_block_stats((panel[hi] - panel[lo]).to_numpy(),
                                    min_lag=lag, min_block=lag)
            pair = f"{QUINTILE_NAMES[lo]}{QUINTILE_NAMES[hi]}"
            out.append((header, f"tHac{name}{pair}", _fmt(stats["nw_t"], 1)))
            out.append((header, f"pBoot{name}{pair}",
                        _fmt_block_p_cell(stats["block_boot_p_one_sided"])))
            if name == "Wthirty" and (lo, hi) == (3, 4):
                se = stats["mean_diff"] / stats["nw_t"]
                out.append((header, "detectWthirtyQthreeQfour", _fmt_bound(z * se, 3, "hi")))
    chi = pd.concat(
        {g: _read_csv(DATA / "susceptibility" / f"rolling_percolation_Q{g}.csv",
                      index_col="date")["rho_c_chi"] for g in (3, 4)},
        axis=1).dropna()
    se = hac_mean_se((chi[4] - chi[3]).to_numpy())["se"]
    out.append((header, "detectChiQthreeQfour", _fmt_bound(z * se, 3, "hi")))
    # subsec:size_confounds compares the within-sector limit with the main
    # Q3-Q4 step at four decimals, where three decimals cannot order them; the
    # limit is rounded down and the step up, so the printed order holds.
    ws = pd.concat(
        {g: _read_csv(DATA / "within_sector" / f"rolling_percolation_Q{g}.csv",
                      index_col="date")["rho_c"] for g in (3, 4)},
        axis=1).dropna()
    out.append((header, "detectWithinSectorQthreeQfour",
                _fmt_bound(z * hac_mean_se((ws[4] - ws[3]).to_numpy())["se"], 4, "lo")))
    main = pd.concat(
        {g: _read_csv(DATA / "korea_headline" / f"rolling_percolation_Q{g}.csv",
                      index_col="date")["rho_c"] for g in (3, 4)},
        axis=1).dropna()
    out.append((header, "rhoCStepHeadlineQthreeQfourFine",
                _fmt_bound(float((main[4] - main[3]).mean()), 4, "hi")))
    return out


def _control_arm_hac_macros() -> list[tuple[str, str, str]]:
    """Per-pair Newey-West HAC t and moving-block bootstrap p for the KOSPI-only
    and within-sector controls (subsec:size_confounds, tab:robustness_summary
    notes), routed from the pair_significance.csv files persisted by
    verify_claim.py. The headline arm's are named without a prefix by
    _hac_macros below."""
    specs = [
        ("kospi_only", "Kospi",
         "KOSPI-only HAC t and block-bootstrap p (subsec:size_confounds, tab:robustness_summary)"),
        ("within_sector", "WithinSector",
         "Within-sector HAC t and block-bootstrap p (subsec:size_confounds, tab:robustness_summary)"),
    ]
    out: list[tuple[str, str, str]] = []
    for subdir, prefix, header in specs:
        pairs = _safe_read_csv(DATA / subdir / "pair_significance.csv")
        if pairs is None or pairs.empty:
            continue
        by_pair = {row["pair"]: row for _, row in pairs.iterrows()}
        for lo, hi in [(1, 2), (2, 3), (3, 4), (4, 5)]:
            row = by_pair.get(f"Q{hi}-Q{lo}")
            if row is None:
                continue
            suffix = QUINTILE_NAMES[lo] + QUINTILE_NAMES[hi]
            out.append((header, f"tHac{prefix}{suffix}",
                        _fmt(float(row["nw_t"]), 2)))
            out.append((header, f"pBlockBoot{prefix}{suffix}",
                        _fmt_block_p_relation(float(row["block_boot_p_one_sided"]))))
    return out


def _hac_macros() -> list[tuple[str, str, str]]:
    """Autocorrelation-robust significance of the headline gradient (subsec:main_result).

    The 60-day windows advance 20 trading days, so the per-window paired
    differences are autocorrelated and the binomial sign / Wilcoxon tests
    (which treat the windows as independent) overstate significance. These
    macros route the Newey-West HAC t-statistic, the
    effective sample size n_eff, and the moving-block bootstrap one-sided
    p-value from the pair_significance.csv that verify_claim.py persists."""
    out: list[tuple[str, str, str]] = []
    pairs = _safe_read_csv(DATA / "korea_headline" / "pair_significance.csv")
    if pairs is None or pairs.empty or "nw_t" not in pairs.columns:
        return out
    header = "Autocorrelation-robust headline significance (subsec:main_result, tab:pair_significance)"
    by_pair = {row["pair"]: row for _, row in pairs.iterrows()}
    adj_fracs: list[float] = []
    for lo, hi in [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]:
        row = by_pair.get(f"Q{hi}-Q{lo}")
        if row is None:
            continue
        suffix = QUINTILE_NAMES[lo] + QUINTILE_NAMES[hi]
        out.append((header, f"tHac{suffix}", _fmt(float(row["nw_t"]), 1)))
        # Per-adjacent-pair moving-block-bootstrap one-sided p: the primary
        # inference now reads the strict monotone gradient from the bootstrap
        # (every adjacent step rejects), so each adjacent pair gets a macro. The
        # endpoint Q1-Q5 bootstrap p is emitted separately below.
        if hi == lo + 1:
            out.append((header, f"pBlockBoot{suffix}",
                        _fmt_block_p(float(row["block_boot_p_one_sided"]))))
            adj_fracs.append(float(row["frac_positive"]))
    # Range, across the four adjacent pairs, of the fraction of windows in which
    # the ordering holds (Q_{g+1} >= Q_g, ties counted as holding, full n). The
    # noisy per-window picture subsec:main_result and subsec:limits both cite; frac_positive
    # is already this ties-as-holding full-n fraction (the tab:percolation sign-test convention).
    if adj_fracs:
        out.append((header, "fracWindowsAdjPairLo",
                    _fmt(min(adj_fracs) * 100, 0) + r"\%"))
        out.append((header, "fracWindowsAdjPairHi",
                    _fmt(max(adj_fracs) * 100, 0) + r"\%"))
    ep = by_pair.get("Q5-Q1")
    if ep is not None:
        out.append((header, "pBlockBootQoneQfive",
                    _fmt_block_p(float(ep["block_boot_p_one_sided"]))))
        out.append((header, "nwLagHeadline", f"{int(ep['nw_lag'])}"))
        # Fraction of windows with the strict Q1<Q5 ordering, the Korean
        # comparand for the Taiwan fracWindowsQoneLtQfiveTw in subsec:taiwan.
        # frac_positive counts a tied window as holding (the sign-test
        # convention), and the sentence states the strict ordering.
        q1 = _read_csv(DATA / "korea_headline" / "rolling_percolation_Q1.csv")
        q5 = _read_csv(DATA / "korea_headline" / "rolling_percolation_Q5.csv")
        both = q1[["date", "rho_c"]].merge(
            q5[["date", "rho_c"]], on="date", suffixes=("_q1", "_q5")).dropna()
        out.append((header, "fracWindowsQoneLtQfiveKr",
                    _fmt(float((both["rho_c_q5"] > both["rho_c_q1"]).mean()) * 100, 0)
                    + r"\%"))
    return out


def _mean_corr_macros() -> list[tuple[str, str, str]]:
    """Pooled rank coupling of rho_c with segment mean |rho| (subsec:threshold_vs_mean).

    Co-movement disclosure. The diagnostic
    (scripts/scaling/run_mean_corr_diagnostic.py) pools all (window, quintile) cells and
    reports the Spearman rank correlation between the curvature-inflection rho_c
    and the off-diagonal mean |rho_ij| of the same cell. Per-quintile cross-window
    means of |rho_ij| populate the final row of tab:percolation."""
    out: list[tuple[str, str, str]] = []
    coupling = _safe_read_csv(DATA / "mean_corr_diagnostic" / "coupling.csv")
    if coupling is not None and not coupling.empty:
        out.append((
            "rho_c / mean |rho| coupling (subsec:threshold_vs_mean)",
            "meanCorrSpearman",
            _fmt(float(coupling.iloc[0]["spearman"]), 2),
        ))
        # What conditioning on the mean leaves behind, as a share of the raw
        # endpoint gap (subsec:threshold_vs_mean).
        if "resid_gap_share_of_rho_c_gap" in coupling.columns:
            out.append((
                "rho_c / mean |rho| coupling (subsec:threshold_vs_mean)",
                "rhoCResidGapPctOfRaw",
                _fmt(float(coupling.iloc[0]["resid_gap_share_of_rho_c_gap"]) * 100, 0)
                + r"\%",
            ))

    # Sensitivity of that residual share to the degree of the pooled fit
    # (subsec:threshold_vs_mean). CHOSEN: degrees one to five bracket the cubic
    # of run_mean_corr_diagnostic.py; recomputed here on the same persisted
    # cells (per_window_mean_corr.csv joined to the headline per-window rho_c).
    cells = _read_csv(DATA / "mean_corr_diagnostic" / "per_window_mean_corr.csv")
    rho = pd.concat([
        _read_csv(DATA / "korea_headline" / f"rolling_percolation_Q{g}.csv")[["date", "rho_c"]]
        .assign(quintile=g) for g in range(1, 6)])
    cells = cells.merge(rho, on=["date", "quintile"], validate="one_to_one")
    raw = cells.groupby("quintile")["rho_c"].mean()
    shares = []
    for deg in range(1, 6):
        fit = np.polyval(np.polyfit(cells["mean_abs_corr"], cells["rho_c"], deg),
                         cells["mean_abs_corr"])
        resid = (cells["rho_c"] - fit).groupby(cells["quintile"]).mean()
        shares.append((resid[5] - resid[1]) / (raw[5] - raw[1]))
        if deg == 3:
            # HAC t of the cubic's residual endpoint contrast, on the per-window
            # differences that subsec:inference tests (subsec:threshold_vs_mean).
            per_window = (cells.assign(resid=cells["rho_c"] - fit)
                          .pivot(index="date", columns="quintile", values="resid"))
            endpoint_t = hac_block_stats(
                (per_window[5] - per_window[1]).to_numpy())["nw_t"]
    for name, val in (("rhoCResidGapPctOfRawDegLo", min(shares)),
                      ("rhoCResidGapPctOfRawDegHi", max(shares))):
        out.append(("rho_c / mean |rho| coupling (subsec:threshold_vs_mean)",
                    name, _fmt(val * 100, 0) + r"\%"))
    out.append(("rho_c / mean |rho| coupling (subsec:threshold_vs_mean)",
                "tHacMeanCorrResidQoneQfive", _fmt(endpoint_t, 1)))

    # Per-quintile mean off-diagonal |rho_ij| (tab:percolation final row, co-movement row).
    summary = _safe_read_csv(
        DATA / "mean_corr_diagnostic" / "cross_quintile_summary.csv")
    if summary is not None and not summary.empty:
        summary = summary.sort_values("quintile").reset_index(drop=True)
        for _, row in summary.iterrows():
            g = int(row["quintile"])
            out.append((
                "rho_c / mean |rho| coupling (subsec:threshold_vs_mean)",
                f"meanAbsCorr{QUINTILE_NAMES[g]}",
                _fmt(float(row["mean_abs_corr_mean"]), 3),
            ))
        # HAC t of its Q1-to-Q5 rise, on the per-window differences that
        # subsec:inference tests (subsec:threshold_vs_mean).
        per_window_corr = cells.pivot(index="date", columns="quintile",
                                      values="mean_abs_corr")
        out.append(("rho_c / mean |rho| coupling (subsec:threshold_vs_mean)",
                    "tHacMeanCorrQoneQfive",
                    _fmt(hac_block_stats(
                        (per_window_corr[5] - per_window_corr[1]).dropna().to_numpy())["nw_t"], 1)))
    return out


def _rho_c_vs_N_macros() -> list[tuple[str, str, str]]:
    """Pooled slope of rho_c on sqrt(2 ln(N-1)) over the network-size grid
    (subsec:resampling_method, subsec:threshold_vs_mean), which run_rho_c_vs_N.py
    fits; eq:toy_closed_form makes it sigma_delta."""
    out: list[tuple[str, str, str]] = []
    slopes = _safe_read_csv(
        DATA / "network_size_resampling" / "rho_c_vs_N_slopes.csv")
    if slopes is None or slopes.empty:
        return out
    header = "rho_c over the network size (subsec:threshold_vs_mean)"
    pooled = slopes[slopes["quintile"] == 0]
    if not pooled.empty:
        out.append((header, "rhoCSlopeSqrtLogN",
                    _fmt(float(pooled.iloc[0]["rho_c_slope"]), 3)))
    return out


def _universe_macros() -> list[tuple[str, str, str]]:
    """Korean and Taiwanese distinct-ticker counts plus the Korean panel
    trading-day span (subsec:data_description, subsec:taiwan)."""
    out: list[tuple[str, str, str]] = []

    # Korean universe count: distinct common-stock tickers (panel columns) and
    # the trading-day span (panel rows) in the survivorship-bias-free panel
    # over the sample period (subsec:data_description).
    # A ticker can enter a window only if both panels carry it: the market-cap
    # panel is what assigns its quintile and the returns panel is what supplies
    # its correlations. The two no longer cover the same column set on either
    # market -- FnGuide's export omits 18 trust-like Korean issuers, and the
    # Taiwanese rebuild prices names whose share count FinMind does not serve --
    # so the universe the analysis can reach is the intersection, not either side
    # of it.
    marcap_parquet = DATA / "cache" / "marketcap_panel.parquet"
    returns_parquet = DATA / "cache" / "returns_panel.parquet"
    if marcap_parquet.exists() and returns_parquet.exists():
        marcap_panel = _read_parquet(marcap_parquet)
        returns_panel = _read_parquet(returns_parquet)
        n_tickers_kr = int(len(marcap_panel.columns.intersection(
            returns_panel.columns)))
        n_days_kr = int(marcap_panel.shape[0])
        out.append((
            "Korean universe (subsec:data_description)",
            "nStocksKoreaTotal",
            f"{n_tickers_kr:,}".replace(",", "{,}"),
        ))
        out.append((
            "Korean universe (subsec:data_description)",
            "nDaysKoreaTotal",
            f"{n_days_kr:,}".replace(",", "{,}"),
        ))
        # si:settings prints one span per market side by side. The Taiwanese
        # bounds are the panel's own first and last session, so the Korean ones
        # are read the same way: a year on one side of that row and a date on
        # the other reads as two different spans where the panels now carry one.
        out.append((
            "Korean panel coverage (subsec:data_description)",
            "krPanelStart",
            f"{marcap_panel.index.min().date()}",
        ))
        out.append((
            "Korean panel coverage (subsec:data_description)",
            "krPanelEnd",
            f"{marcap_panel.index.max().date()}",
        ))
    tw_marcap = DATA / "cache" / "tw_marketcap_panel.parquet"
    tw_returns = DATA / "cache" / "tw_returns_panel.parquet"
    if tw_marcap.exists() and tw_returns.exists():
        n_tickers_tw = int(len(_read_parquet(tw_marcap).columns.intersection(
            _read_parquet(tw_returns).columns)))
        out.append((
            "Taiwan universe (subsec:data_description)",
            "nStocksTaiwanTotal",
            f"{n_tickers_tw:,}".replace(",", "{,}"),
        ))
        # The Taiwanese panel's coverage bounds are the sessions the vendor
        # prices inside the paper's own span, so they are read off the built
        # panel rather than restated here.
        tw_index = _read_parquet(tw_returns).index
        # si:settings prints trading days for both markets. Taiwan's row read
        # "---", which the table's note glosses as "does not apply to that
        # market"; a trading-day count applies to Taiwan exactly as it does to
        # Korea.
        out.append((
            "Taiwan universe (subsec:data_description)",
            "nDaysTaiwanTotal",
            f"{len(tw_index):,}".replace(",", "{,}"),
        ))
        out.append((
            "Taiwan panel coverage (subsec:data_description)",
            "twPanelStart",
            f"{tw_index.min().date()}",
        ))
        out.append((
            "Taiwan panel coverage (subsec:data_description)",
            "twPanelEnd",
            f"{tw_index.max().date()}",
        ))

    return out


def _paired_hac(subdir: str, col: str, lo: int, hi: int):
    """Newey-West HAC t on the per-window paired difference ``col(Q_hi) -
    col(Q_lo)``, over the windows that both quintiles share. Needed for
    observables verify_claim.py's pair_significance.csv does not carry a column
    for -- the susceptibility-peak rho_c^chi is one -- so that every contrast the
    manuscript reports is tested with the autocorrelation-robust statistic of
    subsec:inference."""
    a = _safe_read_csv(DATA / subdir / f"rolling_percolation_Q{lo}.csv")
    b = _safe_read_csv(DATA / subdir / f"rolling_percolation_Q{hi}.csv")
    if a is None or b is None or col not in a.columns or col not in b.columns:
        return None
    merged = a[["date", col]].rename(columns={col: "lo"}).merge(
        b[["date", col]].rename(columns={col: "hi"}), on="date"
    )
    diff = (merged["hi"] - merged["lo"]).dropna().to_numpy()
    if diff.size == 0:
        return None
    return hac_block_stats(diff)


def _susceptibility_macros() -> list[tuple[str, str, str]]:
    """Susceptibility-peak rho_c per quintile (tab:percolation chi-peak row, subsec:researcher_choices)."""
    out: list[tuple[str, str, str]] = []

    # Susceptibility-peak rho_c (tab:percolation chi-peak row, subsec:researcher_choices).
    susc = _safe_read_csv(
        DATA / "susceptibility" / "rolling_percolation_summary.csv"
    )
    if susc is None or "rho_c_chi_mean" not in susc.columns:
        susc = _safe_read_csv(DATA / "korea_headline" / "rolling_percolation_summary.csv")
    if susc is not None and "rho_c_chi_mean" in susc.columns:
        susc = susc.sort_values("size_group").reset_index(drop=True)
        for _, row in susc.iterrows():
            g = int(row["size_group"])
            chi = row.get("rho_c_chi_mean")
            if pd.notna(chi):
                out.append((
                    "Susceptibility-peak rho_c (tab:percolation, subsec:main_result, subsec:researcher_choices)",
                    f"barRhoCChi{QUINTILE_NAMES[g]}",
                    _fmt(float(chi), 3),
                ))
        # Q5-Q1 endpoint gap under the chi-peak observable, the chi-peak row of
        # the robustness summary (tab:robustness_summary, sec:robustness).
        # Same endpoint contrast as rhoCGapHeadline, read on rho_c_chi instead.
        chi_means = susc["rho_c_chi_mean"].dropna()
        if len(chi_means) == 5:
            out.append((
                "Susceptibility-peak rho_c (tab:percolation, subsec:main_result, subsec:researcher_choices)",
                "rhoCGapChi",
                _fmt(float(chi_means.iloc[-1] - chi_means.iloc[0]), 3),
            ))

        # Per-quintile gap between the susceptibility-peak threshold and the
        # curvature-inflection threshold (subsec:researcher_choices body): min/max of
        # rho_c_chi_mean - rho_c_mean across Q1..Q5. The gap is largest in the
        # smallest quintile (Q1).
        if "rho_c_mean" in susc.columns:
            offset = (susc["rho_c_chi_mean"] - susc["rho_c_mean"]).dropna()
            if not offset.empty:
                out.append(("Susceptibility-peak rho_c (tab:percolation, subsec:main_result, subsec:researcher_choices)",
                            "rhoCChiOffsetMin", _fmt(float(offset.min()), 3)))
                out.append(("Susceptibility-peak rho_c (tab:percolation, subsec:main_result, subsec:researcher_choices)",
                            "rhoCChiOffsetMax", _fmt(float(offset.max()), 3)))

    # Chi-peak Q3-Q4 and Q4-Q5 steps (subsec:researcher_choices body): under the
    # textbook chi-peak observable both flatten to non-significance, the
    # observable-sensitivity the curvature-monotone gradient does not show.
    for lo, hi in ((3, 4), (4, 5)):
        suffix = QUINTILE_NAMES[lo] + QUINTILE_NAMES[hi]
        hac = _paired_hac("susceptibility", "rho_c_chi", lo, hi)
        if hac is not None:
            out.append(("Susceptibility-peak rho_c (tab:percolation, subsec:main_result, subsec:researcher_choices)",
                        f"tHacChi{suffix}", _fmt(float(hac["nw_t"]), 2)))

    return out


def _robustness_gap_macros() -> list[tuple[str, str, str]]:
    """Q5-Q1 gaps for the KOSPI-only and within-sector controls, plus the
    full RMT-cleaned per-quintile rho_c + gap (subsec:null_robustness, tab:robustness_summary)."""
    out: list[tuple[str, str, str]] = []

    # KOSPI-only headline gap (subsec:size_confounds, tab:robustness_summary).
    kospi = _safe_read_csv(DATA / "kospi_only" / "rolling_percolation_summary.csv")
    if kospi is not None:
        kospi = kospi.sort_values("size_group").reset_index(drop=True)
        gap = float(kospi.iloc[-1]["rho_c_mean"] - kospi.iloc[0]["rho_c_mean"])
        out.append((
            "KOSPI-only robustness (subsec:checks, subsec:main_result, tab:robustness_summary)",
            "rhoCGapKospi",
            _fmt(gap, 3),
        ))

    # Panel-construction shares measured by the builders themselves
    # (tab:settings); data_loader._write_build_stats.
    stats = _safe_read_csv(DATA / "cache" / "panel_build_stats.csv")
    if stats is not None:
        stats = stats.set_index("market")
        hdr = "Panel construction (tab:settings)"
        if "korea" in stats.index:
            kr = stats.loc["korea"]
            out.append((hdr, "noTradeShareKr",
                        f"{100 * float(kr['no_trade_share']):.1f}\\%"))
        if "taiwan" in stats.index:
            tw = stats.loc["taiwan"]
            out.append((hdr, "noTradeShareTw",
                        f"{100 * float(tw['no_trade_share']):.1f}\\%"))

    # Within-sector gap (subsec:size_confounds, tab:robustness_summary).
    ws = _safe_read_csv(DATA / "within_sector" / "rolling_percolation_summary.csv")
    if ws is not None:
        ws = ws.sort_values("size_group").reset_index(drop=True)
        gap = float(ws.iloc[-1]["rho_c_mean"] - ws.iloc[0]["rho_c_mean"])
        out.append((
            "Within-sector robustness (subsec:main_result, subsec:size_confounds, tab:robustness_contrast)",
            "rhoCGapWithinSector",
            _fmt(gap, 3),
        ))
        # Within-sector endpoint gap as a percentage of the headline gap
        # (subsec:size_confounds). Generated so the prose proportion tracks
        # rhoCGapWithinSector / rhoCGapHeadline and cannot drift across reruns.
        headline = _safe_read_csv(DATA / "korea_headline" / "rolling_percolation_summary.csv")
        if headline is not None:
            headline = headline.sort_values("size_group").reset_index(drop=True)
            headline_gap = float(
                headline.iloc[-1]["rho_c_mean"] - headline.iloc[0]["rho_c_mean"]
            )
            out.append((
                "Within-sector robustness (subsec:main_result, subsec:size_confounds, tab:robustness_contrast)",
                "rhoCGapWithinSectorPctOfHeadline",
                _fmt(gap / headline_gap * 100, 0) + r"\%",
            ))

    # Within-sector sector-size-floor sensitivity (subsec:size_confounds): the Q5-Q1 gap band
    # across the swept floors, justifying the researcher-chosen MIN_SECTOR_SIZE.
    wss = _safe_read_csv(DATA / "within_sector" / "sector_size_sensitivity.csv")
    if wss is not None and not wss.empty:
        gaps = (
            wss.sort_values(["min_sector_size", "size_group"])
            .groupby("min_sector_size")["rho_c_mean"]
            .agg(lambda s: float(s.iloc[-1] - s.iloc[0]))
        )
        out.append((
            "Within-sector robustness (subsec:main_result, subsec:size_confounds, tab:robustness_contrast)",
            "rhoCGapWithinSectorFloorMin",
            _fmt(float(gaps.min()), 3),
        ))
        out.append((
            "Within-sector robustness (subsec:main_result, subsec:size_confounds, tab:robustness_contrast)",
            "rhoCGapWithinSectorFloorMax",
            _fmt(float(gaps.max()), 3),
        ))

    # RMT-filtered gap (subsec:null_robustness, tab:robustness_summary).
    rmt = _safe_read_csv(DATA / "rmt_filtered" / "rolling_percolation_summary.csv")
    if rmt is not None:
        rmt = rmt.sort_values("size_group").reset_index(drop=True)
        for _, row in rmt.iterrows():
            g = int(row["size_group"])
            out.append((
                "RMT-cleaned rho_c (subsec:main_result, tab:robustness_summary)",
                f"barRhoCRmt{QUINTILE_NAMES[g]}",
                _fmt(row["rho_c_mean"], 3),
            ))
        gap = float(rmt.iloc[-1]["rho_c_mean"] - rmt.iloc[0]["rho_c_mean"])
        out.append((
            "RMT-cleaned rho_c (subsec:main_result, tab:robustness_summary)",
            "rhoCGapRmt",
            _fmt(gap, 3),
        ))

    return out


def _subperiod_macros() -> list[tuple[str, str, str]]:
    """Subperiod stability of the endpoint gradient (subsec:subperiod, tab:subperiod).

    run_subperiod_split.py splits the aligned rolling windows at their median
    into a first half and a second half and re-runs the headline HAC inference
    on each. The Korean endpoint gap compresses from the first half to the
    second and stays resolved, one five-year calendar period leaves it
    unresolved, and the Taiwan endpoint keeps most of its size across the same
    split."""
    out: list[tuple[str, str, str]] = []
    header = "Subperiod stability (subsec:subperiod, tab:subperiod)"

    ep = _safe_read_csv(DATA / "subperiod" / "subperiod_endpoint.csv")
    if ep is not None and not ep.empty:
        ep = ep.set_index(["market", "segment"])

        def _cell(market: str, segment: str, col: str) -> float | None:
            try:
                return float(ep.loc[(market, segment), col])
            except KeyError:
                return None

        def _signed(v: float) -> str:
            return f"{'+' if v >= 0 else '-'}{abs(v):.3f}"

        for market, segment, name, col in [
            ("korea", "early", "tHacEndpointEarly", "nw_t"),
            ("korea", "late", "tHacEndpointLate", "nw_t"),
            ("taiwan", "early", "tHacEndpointEarlyTw", "nw_t"),
            ("taiwan", "late", "tHacEndpointLateTw", "nw_t"),
        ]:
            v = _cell(market, segment, col)
            if v is not None:
                out.append((header, name, _fmt(v, 1)))
        for market, segment, name in [
            ("korea", "early", "rhoCGapEndpointEarly"),
            ("korea", "late", "rhoCGapEndpointLate"),
            ("taiwan", "early", "rhoCGapEndpointEarlyTw"),
            ("taiwan", "late", "rhoCGapEndpointLateTw"),
        ]:
            v = _cell(market, segment, "mean_diff")
            if v is not None:
                out.append((header, name, _signed(v)))
        late_boot = _cell("korea", "late", "block_boot_p_one_sided")
        if late_boot is not None:
            out.append((header, "pBootEndpointLate", _fmt_block_p(late_boot)))
        late_frac = _cell("korea", "late", "frac_positive")
        if late_frac is not None:
            out.append((header, "fracWindowsLateQoneLtQfive",
                        _fmt(late_frac * 100, 0) + r"\%"))
        # The abstract and subsec:subperiod put each market's second-half gap on
        # one scale, as a share of its own first-half gap.
        for market, name in (("korea", "rhoCGapLateShareOfEarlyPct"),
                             ("taiwan", "rhoCGapLateShareOfEarlyTwPct")):
            early = _cell(market, "early", "mean_diff")
            late = _cell(market, "late", "mean_diff")
            if early is not None and late is not None:
                out.append((header, name, _fmt(late / early * 100, 0) + r"\%"))

    blocks = _safe_read_csv(DATA / "subperiod" / "subperiod_yearblocks.csv")
    if blocks is not None and not blocks.empty:
        kr = blocks[blocks["market"] == "korea"]
        # A block leaves the endpoint unresolved when its HAC t misses the
        # one-sided 5% level of subsec:inference, the one-sided p taken from the
        # normal as overlap_inference.hac_block_stats takes it.
        unresolved = kr.loc[norm.sf(kr["nw_t"]) >= ONE_SIDED_LEVEL, "block"].tolist()
        if len(unresolved) != 1:
            sys.exit("subsec:subperiod names the one Korean calendar period that "
                     "leaves the endpoint unresolved, but "
                     f"data/subperiod/subperiod_yearblocks.csv has {unresolved}. "
                     "Restate that sentence before regenerating.")
        out.append((header, "krUnresolvedBlock", unresolved[0].replace("-", "--")))
        # tab:subperiod prints each block's HAC t for both markets, on one grid
        # of blocks that run_subperiod_split.py shares across the markets.
        tw = blocks[blocks["market"] == "taiwan"].set_index("block")["nw_t"]
        if list(tw.index) != kr["block"].tolist():
            sys.exit("tab:subperiod prints the Korean and Taiwanese five-year "
                     "blocks in one table, but subperiod_yearblocks.csv gives "
                     f"{kr['block'].tolist()} and {list(tw.index)}.")
        for letter, (block, t_kr) in zip("ABCDEFGH", zip(kr["block"], kr["nw_t"])):
            out.append((header, f"yearBlock{letter}", block.replace("-", "--")))
            out.append((header, f"tHacYearBlockKr{letter}", _fmt(t_kr, 1)))
            out.append((header, f"tHacYearBlockTw{letter}", _fmt(tw[block], 1)))

    # Second-half cross-quintile cut: the row tab:robustness_contrast carries and
    # the four adjacent steps subsec:subperiod reads off it -- the surviving
    # small-cap step, the two unresolved interior steps, and the Q4--Q5
    # inversion at the top.
    post = _safe_read_csv(DATA / "subperiod" / "subperiod_second_half.csv")
    if post is not None and not post.empty:
        post_header = "Second-half subsample (subsec:subperiod)"
        post = post.set_index("quintile")
        out.append((post_header, "nWindowsSecondHalf",
                    f"{int(post['n_windows'].iloc[0])}"))
        out.append((post_header, "secondHalfStart",
                    str(post["start_date"].iloc[0])))
        for g, name in ((2, "QoneQtwo"), (3, "QtwoQthree"),
                        (4, "QthreeQfour"), (5, "QfourQfive")):
            if g not in post.index:
                continue
            out.append((post_header, f"tHacSecondHalf{name}",
                        _fmt(float(post.loc[g, "step_nw_t"]), 1)))
            # Only the two outer steps are quoted as magnitudes. The interior
            # pair rounds to the third decimal at 0.000 and 0.003, which the
            # prose reads as a span rather than as two steps.
            if g in (2, 5):
                step = float(post.loc[g, "step_mean_diff"])
                out.append((post_header, f"rhoCStepSecondHalf{name}",
                            f"{'+' if step >= 0 else '-'}{abs(step):.3f}"))
        # Total spread of the three interior means Q2, Q3, Q4. The prose reads
        # the interior as flat rather than stepping, and this is the width of
        # the band the three means sit in.
        interior = post.loc[[2, 3, 4], "mean_rho_c"]
        out.append((post_header, "rhoCSecondHalfInteriorSpan",
                    _fmt(float(interior.max() - interior.min()), 3)))
    return out


def _n_implied_gap_component(panel: str) -> float | None:
    """Share of a panel's Q1-Q5 rho_c gap the unequal quintile universes explain.

    The observation requirement (>MIN_VALID_FRACTION) leaves the quintiles of one
    panel holding different stock counts, and eq:toy_closed_form makes rho_c
    rise with the network size as sqrt(2 ln(N-1)). Pricing that inequality at
    the pooled rho_c slope on sqrt(2 ln(N-1)) measured over the network-size
    grid (subsec:threshold_vs_mean) gives the mechanical component of the observed
    endpoint contrast, which subsec:taiwan reports against the observed gap.
    Returns ``None`` when either input is missing.
    """
    slopes = _safe_read_csv(
        DATA / "network_size_resampling" / "rho_c_vs_N_slopes.csv")
    q1 = _safe_read_csv(DATA / panel / "rolling_percolation_Q1.csv")
    q5 = _safe_read_csv(DATA / panel / "rolling_percolation_Q5.csv")
    if slopes is None or slopes.empty or q1 is None or q5 is None:
        return None
    pooled = slopes[slopes["quintile"] == 0]
    if pooled.empty:
        return None
    # Windows carrying both endpoint quintiles, the pairing the gap is read on.
    merged = q1[["date", "n_stocks"]].merge(
        q5[["date", "n_stocks"]], on="date", suffixes=("_q1", "_q5"))
    if merged.empty:
        return None
    loc = lambda n: float(np.sqrt(2.0 * np.log(n.astype(float) - 1.0)).mean())
    return float(pooled.iloc[0]["rho_c_slope"]) * (
        loc(merged["n_stocks_q5"]) - loc(merged["n_stocks_q1"]))


def _taiwan_macros() -> list[tuple[str, str, str]]:
    """Taiwan re-run per-quintile rho_c, gap, and window count
    (subsec:taiwan, tab:taiwan_rerun)."""
    out: list[tuple[str, str, str]] = []

    # Taiwan re-run (subsec:taiwan, tab:taiwan_rerun).
    tw = _safe_read_csv(DATA / "taiwan" / "rolling_percolation_summary.csv")
    if tw is not None and not tw.empty:
        tw = tw.sort_values("size_group").reset_index(drop=True)
        for _, row in tw.iterrows():
            g = int(row["size_group"])
            # Median stocks per window (the Taiwan tab:taiwan_rerun sample-size row).
            n_stk = row.get("n_stocks_median", row.get("n_stocks"))
            out.append((
                "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
                f"medianStocksTw{QUINTILE_NAMES[g]}",
                _fmt(float(n_stk), 0),
            ))
            out.append((
                "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
                f"barRhoCTw{QUINTILE_NAMES[g]}",
                _fmt(row["rho_c_mean"], 3),
            ))
            # Susceptibility-peak threshold rho_c^chi, all five quintiles, so
            # tab:taiwan_rerun carries the same row tab:percolation does
            # for Korea rather than leaving Q2 and Q3 unreported.
            out.append((
                "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
                f"barRhoCChiTw{QUINTILE_NAMES[g]}",
                _fmt(row["rho_c_chi_mean"], 3),
            ))
        gap = float(tw.iloc[-1]["rho_c_mean"] - tw.iloc[0]["rho_c_mean"])
        out.append((
            "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
            "rhoCGapTw",
            _fmt(gap, 3),
        ))
        # Q5-Q4 gap on Taiwan: small and negative, the Q5 < Q4 top-boundary
        # inversion reported at subsec:taiwan; Korea is monotone at every step.
        gap_q5q4 = float(tw.iloc[-1]["rho_c_mean"] - tw.iloc[-2]["rho_c_mean"])
        out.append((
            "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
            "rhoCGapTwQFiveMinusQFour",
            _fmt(gap_q5q4, 3),
        ))
        out.append((
            "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
            "nWindowsTw",
            f"{int(tw.iloc[0]['n_windows'])}",
        ))
        implied = _n_implied_gap_component("taiwan")
        if implied is not None:
            out.append((
                "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
                "rhoCGapTwNImplied",
                _fmt(implied, 3),
            ))
            out.append((
                "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
                "rhoCGapTwNImpliedPct",
                _fmt(implied / gap * 100, 0) + r"\%",
            ))
        # The Korean twin of the same network-size term (subsec:taiwan). Korea's
        # per-quintile universes are no longer ordered the other way round, so
        # the paragraph now compares two same-signed shares rather than two
        # signs.
        kr_implied = _n_implied_gap_component("korea_headline")
        kr_summary = _safe_read_csv(
            DATA / "korea_headline" / "rolling_percolation_summary.csv")
        if kr_implied is not None and kr_summary is not None:
            kr_means = kr_summary.sort_values("size_group")["rho_c_mean"]
            kr_gap = float(kr_means.iloc[-1] - kr_means.iloc[0])
            out.append((
                "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
                "rhoCGapKrNImplied", _fmt(kr_implied, 3),
            ))
            out.append((
                "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
                "rhoCGapKrNImpliedPct",
                _fmt(kr_implied / kr_gap * 100, 0) + r"\%",
            ))

    # Fraction of windows in which the Q1<Q5 endpoint ordering holds (curvature
    # rho_c), from the per-pair sign-test summary.
    pairs = _safe_read_csv(DATA / "taiwan" / "pair_significance.csv")
    if pairs is not None and not pairs.empty:
        q5q1 = pairs[(pairs["pair"] == "Q5-Q1") & (pairs["observable"] == "rho_c")]
        if not q5q1.empty:
            out.append((
                "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
                "fracWindowsQoneLtQfiveTw",
                _fmt(float(q5q1.iloc[0]["frac_positive"]) * 100, 0) + r"\%",
            ))

        # Per-adjacent-pair and endpoint HAC t for the curvature rho_c
        # observable (tab:taiwan_rerun), one-sided, exactly as persisted by the
        # Taiwan pipeline. Pair labels are stored high-minus-low (e.g. "Q5-Q1");
        # macro names use the low-to-high order the manuscript reads (e.g.
        # QoneQfive).
        for _, prow in pairs[pairs["observable"] == "rho_c"].iterrows():
            hi, lo = (int(tok[1:]) for tok in prow["pair"].split("-"))
            lo_name, hi_name = QUINTILE_NAMES[lo], QUINTILE_NAMES[hi]
            if "nw_t" in prow.index:
                out.append((
                    "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
                    f"tHacTw{lo_name}{hi_name}",
                    _fmt(float(prow["nw_t"]), 1),
                ))
            # subsec:inference reads a step as significant only when the block
            # bootstrap rejects too, so tab:taiwan_rerun prints each step's.
            if ("block_boot_p_one_sided" in prow.index
                    and f"{lo_name}{hi_name}" != "QoneQfive"):
                out.append((
                    "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
                    f"pBlockBootTw{lo_name}{hi_name}",
                    _fmt_block_p(float(prow["block_boot_p_one_sided"])),
                ))

        # Autocorrelation-robust endpoint significance for the curvature rho_c
        # observable (subsec:taiwan): Newey-West HAC t, effective sample size, and
        # moving-block bootstrap one-sided p-value, mirroring the Korean
        # headline. Only emitted once the Taiwan pipeline has been re-run with
        # the HAC columns present.
        q5q1_rho = pairs[(pairs["pair"] == "Q5-Q1")
                         & (pairs["observable"] == "rho_c")]
        if not q5q1_rho.empty and "nw_t" in pairs.columns:
            r = q5q1_rho.iloc[0]
            # tHacTwQoneQfive is emitted by the per-pair loop above (the endpoint
            # is one of its rows); here we add only the endpoint's moving-block
            # bootstrap p.
            out.append((
                "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
                "pBlockBootTwQoneQfive",
                _fmt_block_p(float(r["block_boot_p_one_sided"])),
            ))

        # Endpoint HAC t for the susceptibility-peak observable rho_c^chi
        # (subsec:taiwan third reading), so that paragraph also reads from the
        # autocorrelation-robust statistic rather than an independence-assuming
        # sign-test exponent.
        q5q1_chi = pairs[(pairs["pair"] == "Q5-Q1")
                         & (pairs["observable"] == "rho_c_chi")]
        if not q5q1_chi.empty and "nw_t" in pairs.columns:
            out.append((
                "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
                "tHacTwChiQoneQfive",
                _fmt(float(q5q1_chi.iloc[0]["nw_t"]), 1),
            ))

        # The top step turns over in both observables (subsec:taiwan second and
        # third readings), so the paragraph quotes the HAC t on the turn-over
        # itself rather than reporting only that an increase was not resolved.
        # The three rho_c^chi steps through Q4, which subsec:checks says the
        # rho_c^chi check tests in Taiwan, quoted as the range of their HAC t.
        steps_chi = pairs[pairs["pair"].isin(["Q2-Q1", "Q3-Q2", "Q4-Q3"])
                          & (pairs["observable"] == "rho_c_chi")]
        if len(steps_chi) == 3 and "nw_t" in pairs.columns:
            for name, value in (("tHacTwChiStepMin", steps_chi["nw_t"].min()),
                                ("tHacTwChiStepMax", steps_chi["nw_t"].max())):
                out.append((
                    "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
                    name, _fmt(float(value), 1)))

        q5q4_chi = pairs[(pairs["pair"] == "Q5-Q4")
                         & (pairs["observable"] == "rho_c_chi")]
        if not q5q4_chi.empty and "nw_t" in pairs.columns:
            out.append((
                "Taiwan re-run (subsec:taiwan, tab:taiwan_rerun, subsec:threshold_vs_mean)",
                "tHacTwChiQfourQfive",
                _fmt(float(q5q4_chi.iloc[0]["nw_t"]), 1),
            ))

    return out


def _null_shuffled_macros() -> list[tuple[str, str, str]]:
    """Phase-shuffled-returns null per-quintile rho_c and Q5-Q1 gap
    (subsec:null_robustness, tab:robustness_summary)."""
    out: list[tuple[str, str, str]] = []

    # Shuffled-returns null (subsec:null_robustness, tab:robustness_summary). across_seed_mean is the per-quintile
    # mean rho_c over independent phase-shuffled surrogate ensembles.
    null = _safe_read_csv(DATA / "null_shuffled" / "cross_quintile_summary.csv")
    if null is not None and not null.empty:
        null = null.sort_values("quintile").reset_index(drop=True)
        for _, row in null.iterrows():
            g = int(row["quintile"])
            out.append((
                "Shuffled-returns null (subsec:checks, subsec:main_result, tab:robustness_summary)",
                f"shuffNullRhoC{QUINTILE_NAMES[g]}",
                _fmt(row["across_seed_mean"], 3),
            ))
        gap = float(
            null.iloc[-1]["across_seed_mean"] - null.iloc[0]["across_seed_mean"]
        )
        out.append((
            "Shuffled-returns null (subsec:checks, subsec:main_result, tab:robustness_summary)",
            "shuffNullGapQFiveMinusQone",
            _fmt(gap, 3),
        ))

    # Seed count and across-seed dispersion, both read off the per-seed file
    # rather than off run_null_shuffled.N_SEEDS, so the note in
    # tab:robustness_summary states what the shipped ensemble actually holds.
    # shuffNullSeedSdMax is the largest per-quintile standard deviation of the
    # seed-level mean rho_c; the note quotes it as the ensemble's resolution.
    per_seed = _safe_read_csv(
        DATA / "null_shuffled" / "summary_per_seed_per_quintile.csv")
    if per_seed is not None and not per_seed.empty:
        out.append((
            "Shuffled-returns null (subsec:checks, subsec:main_result, tab:robustness_summary)",
            "nShuffleSeeds",
            f"{int(per_seed['seed'].nunique())}",
        ))
        sd_max = float(
            per_seed.groupby("quintile")["mean_rho_c"].std(ddof=1).max())
        out.append((
            "Shuffled-returns null (subsec:checks, subsec:main_result, tab:robustness_summary)",
            "shuffNullSeedSdMax",
            _fmt(sd_max, 4),
        ))

    # Per-quintile excess kurtosis of raw daily returns (subsec:null_robustness).
    # Shuffling permutes each stock's series in time, so it preserves the
    # marginals exactly and destroys only the cross-sectional structure. The
    # null's rho_c level is therefore set by tail behaviour the shuffle keeps,
    # which is what makes the null's reversed Q1-above-Q5 asymmetry a property
    # of the marginals rather than a residue of the gradient.
    kurt = _read_csv(DATA / "pca_kurtosis_diagnostic" / "excess_kurtosis_summary.csv",
                     header=[0, 1], index_col=0)
    kurt_mean = kurt[("mean_excess_kurtosis", "mean")]
    for g in (1, 5):
        out.append((
            "Per-quintile excess kurtosis of daily returns (subsec:null_robustness)",
            f"excessKurt{QUINTILE_NAMES[g]}",
            _fmt(float(kurt_mean.loc[g]), 1),
        ))

    return out


def _window_sensitivity_macros() -> list[tuple[str, str, str]]:
    """Per-quintile rho_c and Q5-Q1 gap at each window length (subsec:researcher_choices, tab:robustness_summary)."""
    out: list[tuple[str, str, str]] = []

    # Window-length sensitivity (subsec:researcher_choices, tab:robustness_summary).
    wsens = _safe_read_csv(DATA / "window_sensitivity" / "summary.csv")
    if wsens is not None and not wsens.empty:
        # T = 60 is the headline window: its per-quintile means and gap already ship
        # as barRhoCQ*/rhoCGapHeadline off the headline results file, and
        # test_assertions.py asserts the two agree, so re-emitting them here would
        # put one number under two names.
        W_NAMES = {30: "Wthirty", 120: "Whundredtwenty", 252: "Wtwofifty"}
        for _, row in wsens.iterrows():
            w = int(row["window_days"])
            g = int(row["size_group"])
            if w not in W_NAMES:
                continue
            out.append((
                "Window-length sensitivity (subsec:checks, subsec:main_result, tab:robustness_summary)",
                f"barRhoC{W_NAMES[w]}{QUINTILE_NAMES[g]}",
                _fmt(row["rho_c_mean"], 3),
            ))
        for w, wname in W_NAMES.items():
            sub = wsens[wsens["window_days"] == w].sort_values("size_group")
            if len(sub) == 5:
                out.append((
                    "Window-length sensitivity (subsec:checks, subsec:main_result, tab:robustness_summary)",
                    f"windowDays{wname}", f"{w}",
                ))
                gap = float(sub.iloc[-1]["rho_c_mean"] - sub.iloc[0]["rho_c_mean"])
                out.append((
                    "Window-length sensitivity (subsec:checks, subsec:main_result, tab:robustness_summary)",
                    f"rhoCGap{wname}",
                    _fmt(gap, 3),
                ))
                out.append((
                    "Window-length sensitivity (subsec:checks, subsec:main_result, tab:robustness_summary)",
                    f"nWindows{wname}",
                    f"{int(sub.iloc[0]['n_windows']):,}",
                ))

    # Observation-requirement sweep (subsec:researcher_choices): the swept
    # requirements are read from the sweep table that fixes them.
    header = "Observation-requirement sensitivity (subsec:checks, subsec:researcher_choices, tab:robustness_summary)"
    sweep = _module_constant("scripts/sensitivity/run_pipeline_sensitivity.py",
                             "SWEEPS")["min_valid_fraction_sensitivity"]
    out.append((header, "minValidFractionLoPct", _fmt(100.0 * min(sweep.values), 0) + r"\%"))
    out.append((header, "minValidFractionHiPct", _fmt(100.0 * max(sweep.values), 0) + r"\%"))
    vf = _read_csv(DATA / "min_valid_fraction_sensitivity" / "summary.csv")
    gaps = (vf.sort_values(["min_valid_fraction", "size_group"])
            .groupby("min_valid_fraction")["rho_c_mean"]
            .agg(lambda s: float(s.iloc[-1] - s.iloc[0])))
    # The lowest and highest swept requirements are rows of tab:robustness_contrast
    # and tab:robustness_summary; the middle requirement is the headline row.
    for f, fname in ((min(sweep.values), "ObsReqLo"), (max(sweep.values), "ObsReqHi")):
        sub = vf[vf["min_valid_fraction"] == f].sort_values("size_group")
        for _, row in sub.iterrows():
            out.append((header, f"barRhoC{fname}{QUINTILE_NAMES[int(row['size_group'])]}",
                        _fmt(row["rho_c_mean"], 3)))
        out.append((header, f"rhoCGap{fname}", _fmt(float(gaps.loc[f]), 3)))

    return out


def _robustness_per_quintile_macros() -> list[tuple[str, str, str]]:
    """Full per-quintile rho_c tables for the KOSPI-only and within-sector
    confound controls (subsec:size_confounds, subsec:size_confounds)."""
    out: list[tuple[str, str, str]] = []

    # KOSPI-only per-quintile (subsec:size_confounds, tab:robustness_summary).
    kospi_full = _safe_read_csv(DATA / "kospi_only" / "rolling_percolation_summary.csv")
    if kospi_full is not None and not kospi_full.empty:
        kospi_full = kospi_full.sort_values("size_group").reset_index(drop=True)
        for _, row in kospi_full.iterrows():
            g = int(row["size_group"])
            qn = QUINTILE_NAMES[g]
            out.append(("KOSPI-only robustness (subsec:checks, subsec:main_result, tab:robustness_summary)",
                        f"barRhoCKospi{qn}", _fmt(row["rho_c_mean"], 3)))
            # Only Q1 and Q5 reach the tab:robustness_summary note; the middle
            # three would be orphan macros.
            if g in (1, 5):
                out.append(("KOSPI-only robustness (subsec:checks, subsec:main_result, tab:robustness_summary)",
                            f"medianStocksKospi{qn}", _fmt(int(row["n_stocks"]), 0)))
        out.append(("KOSPI-only robustness (subsec:checks, subsec:main_result, tab:robustness_summary)",
                    "nWindowsKospi", f"{int(kospi_full.iloc[0]['n_windows'])}"))
        # KOSPI common-stock universe size (subsec:size_confounds prose + tab:robustness_summary note:
        # "restricted to KOSPI common stocks (N tickers)"), persisted by
        # run_kospi_only.py as len(kospi_codes).
        usize = _safe_read_csv(DATA / "kospi_only" / "universe_size.csv")
        if usize is not None and not usize.empty:
            out.append(("KOSPI-only robustness (subsec:checks, subsec:main_result, tab:robustness_summary)",
                        "nKospiTickers",
                        f"{int(usize.iloc[0]['n_kospi_tickers']):,}".replace(",", "{,}")))

    # Within-sector per-quintile (subsec:size_confounds, tab:robustness_summary).
    ws_full = _safe_read_csv(DATA / "within_sector" / "rolling_percolation_summary.csv")
    if ws_full is not None and not ws_full.empty:
        ws_full = ws_full.sort_values("size_group").reset_index(drop=True)
        for _, row in ws_full.iterrows():
            g = int(row["size_group"])
            qn = QUINTILE_NAMES[g]
            out.append(("Within-sector per-quintile (subsec:size_confounds, tab:robustness_summary)",
                        f"barRhoCWithinSector{qn}", _fmt(row["rho_c_mean"], 3)))
            # Only Q1 and Q5 reach the tab:robustness_summary note; the middle
            # three would be orphan macros.
            if g in (1, 5):
                out.append(("Within-sector per-quintile (subsec:size_confounds, tab:robustness_summary)",
                            f"medianStocksWithinSector{qn}", _fmt(int(row["n_stocks"]), 0)))
        out.append(("Within-sector per-quintile (subsec:size_confounds, tab:robustness_summary)",
                    "nWindowsWithinSector",
                    f"{int(ws_full.iloc[0]['n_windows'])}"))

    return out


def _dgp_macros() -> list[tuple[str, str, str]]:
    """Data-generating-process validation of the toy model: eq:toy_threshold/eq:toy_closed_form
    verification, N-scaling, and simulator constants (si:dgp_validation)."""
    out: list[tuple[str, str, str]] = []

    # DGP validation (si:dgp_validation). Macros are emitted in the order the SI
    # tables and figures reference them.
    DELTA_NAMES = {0.4: "A", 0.5: "B", 0.6: "C", 0.7: "D", 0.85: "E"}

    eq9 = _safe_read_csv(DATA / "dgp_validation" / "eq9_verification.csv")
    if eq9 is not None and not eq9.empty:
        for _, row in eq9.iterrows():
            d = float(row["delta"])
            dn = DELTA_NAMES.get(round(d, 2))
            if dn is None:
                continue
            wt = row["W_type"]
            suffix = "Sector" if wt == "sector" else ""
            out.append(("DGP eq:toy_threshold/eq:toy_closed_form verification (si:eq9_verification)",
                        f"rhoCDgpSim{suffix}Delta{dn}",
                        _fmt(float(row["rho_c_sim_mean"]), 3)))
            out.append(("DGP eq:toy_threshold/eq:toy_closed_form verification (si:eq9_verification)",
                        f"rhoCDgpSim{suffix}DegOneDelta{dn}",
                        _fmt(float(row["rho_c_degree_one_mean"]), 3)))
            out.append(("DGP eq:toy_threshold/eq:toy_closed_form verification (si:eq9_verification)",
                        f"rhoCDgpClosed{suffix}Delta{dn}",
                        _fmt(float(row["rho_c_closed_mean"]), 3)))
        # Max absolute gap between rho_c_sim and rho_c_degree_one across every
        # verification cell; this is the load-bearing summary number for the
        # subsection's lead sentence. si:eq9_verification says the agreement
        # holds "across all N cells", a quantifier over a set no table on the
        # page lets a reader count, so N is emitted beside the gap it qualifies.
        gap = (eq9["rho_c_sim_mean"] - eq9["rho_c_degree_one_mean"]).abs()
        out.append(("DGP eq:toy_threshold/eq:toy_closed_form verification (si:eq9_verification)",
                    "dgpEqEightMaxGap", _fmt_bound(float(gap.max()), 3, "hi")))
        out.append(("DGP eq:toy_threshold/eq:toy_closed_form verification (si:eq9_verification)",
                    "nDgpEqEightCells", str(len(eq9))))
        # The smallest fall of tilde-rho_c between adjacent simulated delta of
        # one peer-coupling matrix. subsec:toy_model sets the largest gap above
        # against it: a gap below half this fall keeps the simulated rho_c
        # falling at every step of the grid.
        falls = [-g.sort_values("delta")["rho_c_degree_one_mean"].diff().dropna()
                 for _, g in eq9.groupby("W_type")]
        out.append(("DGP eq:toy_threshold/eq:toy_closed_form verification (si:eq9_verification)",
                    "dgpDegOneMinFall", _fmt_bound(float(pd.concat(falls).min()), 3, "lo")))
        # eq:toy_closed_form Gaussian Mills-ratio specialization systematically overshoots
        # rho_c_sim. Min/max of the overshoot range pin the cross-(delta,
        # W_type) cell envelope quoted in paper.tex subsec:toy_model and
        # si:eq9_verification.
        overshoot = eq9["rho_c_closed_mean"] - eq9["rho_c_sim_mean"]
        out.append(("DGP eq:toy_threshold/eq:toy_closed_form verification (si:eq9_verification)",
                    "dgpEqNineOvershootLo",
                    _fmt(float(overshoot.min()), 3)))
        out.append(("DGP eq:toy_threshold/eq:toy_closed_form verification (si:eq9_verification)",
                    "dgpEqNineOvershootHi",
                    _fmt(float(overshoot.max()), 3)))

    # DGP simulation constants (si:dgp_validation introduction and table notes).
    # These pin every numeric value the SI prose quotes about the simulator
    # to the actual code constants in src.fn_percolation.dgp and
    # scripts.run_dgp_validation, so a constant change re-emits the macros
    # and the prose follows.
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpSigmaF", _fmt(float(DEFAULT_SIGMA_F), 3)))
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpSigmaU", _fmt(float(DEFAULT_SIGMA_U), 2)))
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpSigmaFUnscaled", _fmt(float(DEFAULT_SIGMA_F_UNSCALED), 3)))
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpSigmaFScale", _fmt(float(DEFAULT_SIGMA_F_SCALE), 2)))
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpBetaMean", _fmt(float(DEFAULT_BETA_MEAN), 1)))
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpBetaStd", _fmt(float(DEFAULT_BETA_STD), 1)))
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpBetaLo", _fmt(float(DEFAULT_BETA_CLIP[0]), 1)))
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpBetaHi", _fmt(float(DEFAULT_BETA_CLIP[1]), 1)))

    # Pull peer-matrix and simulation constants from the experiment
    # driver. A missing or renamed driver fails loudly here rather than
    # silently dropping these macros and breaking the SI at LaTeX compile time.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_dgp_validation",
        REPO_ROOT / "scripts" / "dgp" / "run_dgp_validation.py",
    )
    rdv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rdv)
    # Peer-matrix defaults are kwargs on make_peer_matrix; mirror the
    # SI prose values (k=5, 10 sectors, intra=1.0, inter=0.1).
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpKnnK", "5"))
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpNSectors", "10"))
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpIntraWeight", _fmt(1.0, 1)))
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpInterWeight", _fmt(0.1, 1)))
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpN", f"{int(rdv.N_DEFAULT)}"))
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpT", f"{int(rdv.T_WINDOW)}"))
    # Endpoints of the Exp-1 delta grid, the range over which subsec:toy_model
    # cites the simulated fall of rho_c below delta = 1/2.
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpDeltaGridMin", _fmt(float(min(rdv.DELTA_GRID_EXP1)), 2)))
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpDeltaGridMax", _fmt(float(max(rdv.DELTA_GRID_EXP1)), 2)))
    # Monte-Carlo replicates per cell in the Eq. (9) verification experiment
    # (Exp 1), quoted in the tab:dgp_eq9 note.
    out.append(("DGP simulation constants (si:dgp_validation)",
                "dgpEqNineNMc", f"{int(rdv.DEFAULT_MC['exp1'])}"))

    return out


def _mp_band_macros() -> list[tuple[str, str, str]]:
    """Marchenko-Pastur noise-band edges [lambda_-, lambda_+] for the headline
    T x N block, computed from the *same* marchenko_pastur_bounds(WINDOW,
    PER_QUINTILE) the RMT robustness filter uses (subsec:null_robustness;
    src/fn_percolation/rmt.py). Pins the manuscript band to the code rather than
    a hand-copied number."""
    lam_minus, lam_plus = marchenko_pastur_bounds(WINDOW, PER_QUINTILE)
    header = "Marchenko-Pastur noise band (subsec:checks)"
    return [
        (header, "mpLambdaMinus", _fmt(lam_minus, 2)),
        (header, "mpLambdaPlus", _fmt(lam_plus, 2)),
    ]


_STEP_PAIRS = (("Q2-Q1", "QoneQtwo"), ("Q3-Q2", "QtwoQthree"),
               ("Q4-Q3", "QthreeQfour"), ("Q5-Q4", "QfourQfive"))


def _residual_battery_macros() -> list[tuple[str, str, str]]:
    """Adjacent-quintile steps and the endpoint contrast of rho_c on total
    returns and on every residualization arm data/ holds, with the rise from
    Q1 to Q4 that subsec:main_result reads off them (tab:residual_steps).

    Each step is the cross-window mean of the per-window paired difference,
    read with its Newey-West t from the pair_significance.csv that
    scripts/headline/verify_claim.py writes for every arm. The paragraph
    quantifies over the arms -- their count, and the range the Q1-to-Q4 rise
    spans over them -- so the set is discovered and an arm without per-pair
    statistics is a hard failure rather than one the quantifier silently
    excludes."""
    header = ("Residualization: adjacent steps and endpoint of rho_c "
              "(subsec:residuals, tab:residual_steps)")
    out: list[tuple[str, str, str]] = []

    def signed(value: float) -> str:
        return ("+" if value >= 0 else "") + _fmt(value, 3)

    def steps(subdir: str) -> dict[str, tuple[float, float]]:
        pairs = _safe_read_csv(DATA / subdir / "pair_significance.csv")
        if pairs is None:
            sys.exit(f"data/{subdir}: no pair_significance.csv. Run "
                     f"scripts/headline/verify_claim.py, which writes the per-pair "
                     f"HAC and block-bootstrap statistics that every arm "
                     f"reports.")
        pairs = pairs.set_index("pair")
        return {pair: (float(pairs.loc[pair, "mean_diff"]),
                       float(pairs.loc[pair, "nw_t"]))
                for pair in [*(p for p, _ in _STEP_PAIRS), "Q5-Q1"]}

    def rise(st: dict[str, tuple[float, float]]) -> float:
        """Q1 to Q4: the first three adjacent steps."""
        return sum(st[pair][0] for pair, _ in _STEP_PAIRS[:3])

    # The total-return row reuses the headline's t-statistics (_hac_macros)
    # and endpoint (rhoCGapHeadline); only its steps are new here.
    head = steps("korea_headline")
    for pair, name in _STEP_PAIRS:
        out.append((header, f"rhoCStepHeadline{name}", signed(head[pair][0])))
    out.append((header, "rhoCRiseQoneQfourHeadline", signed(rise(head))))

    arms = _residual_arm_dirs()
    if not arms:
        sys.exit("data/ holds no residualization arm. Run "
                 "scripts/factor_residual/run_external_factor_residual.py, then "
                 "scripts/headline/verify_claim.py.")
    rises = []
    for arm in arms:
        stem = _external_factor_stem(arm.name)
        st = steps(arm.name)
        for pair, name in _STEP_PAIRS:
            out.append((header, f"rhoCStep{stem}{name}", signed(st[pair][0])))
            out.append((header, f"tHac{stem}{name}", _fmt(st[pair][1], 1)))
        out.append((header, f"rhoCGap{stem}", signed(st["Q5-Q1"][0])))
        out.append((header, f"tHac{stem}QoneQfive", _fmt(st["Q5-Q1"][1], 1)))
        rises.append(rise(st))
    # Printed as the range the rise spans over the arms, so each end rounds away
    # from the data (_fmt_bound): a nearest-rounded end can sit inside the range.
    lo, hi = min(rises), max(rises)
    out.append((header, "rhoCRiseQoneQfourResidMin",
                ("+" if lo >= 0 else "") + _fmt_bound(lo, 3, "lo")))
    out.append((header, "rhoCRiseQoneQfourResidMax",
                ("+" if hi >= 0 else "") + _fmt_bound(hi, 3, "hi")))
    return out


# Deliberate crossings: a macro whose value spans two populations on purpose,
# with the reason recorded where the check reads it rather than in prose no
# program dereferences.
REGISTERED_CROSSINGS = {
    "rhoCGapTwNImplied": (
        "The network-size term the Korean-measured rho_c-on-sqrt(2 ln N) slope "
        "implies for the Taiwanese network sizes. The slope is estimated on the "
        "Korean panel and applied to Taiwan's realized N, which is the comparison "
        "subsec:taiwan and si:rho_c_vs_mean make: how much of Taiwan's larger "
        "endpoint gap the network-size difference alone accounts for."),
}


def build_macros() -> list[tuple[str, str, str]]:
    """Return every (group_header, macro_name, formatted_body) tuple for the
    GENERATED region, concatenated in the exact order paper.tex and paper_si.tex
    reference them. Each helper owns one manuscript artifact and reads its own
    CSVs; emission order here is the contract render_block depends on.

    Each helper's data/ reads are recorded into READ_PATHS as it runs, so the
    set of artifacts the manuscript's numbers come from is derived from what the
    generator opened rather than asserted beside it."""
    out: list[tuple[str, str, str]] = []
    for helper in (
        _design_constant_macros,
        _scan_grid_macros,
        _headline_macros,
        _control_arm_hac_macros,
        _hac_macros,
        _draw_seed_macros,
        _bandwidth_macros,
        _weekly_macros,
        _kospi_resolution_macros,
        _setting_step_macros,
        _mean_corr_macros,
        _rho_c_vs_N_macros,
        _universe_macros,
        _susceptibility_macros,
        _robustness_gap_macros,
        _subperiod_macros,
        _taiwan_macros,
        _null_shuffled_macros,
        _window_sensitivity_macros,
        _mp_band_macros,
        _residual_battery_macros,
        _robustness_per_quintile_macros,
        _dgp_macros,
    ):
        out.extend(helper())
    return out


def render_block(macros: list[tuple[str, str, str]]) -> str:
    lines = [BEGIN]
    current_header: str | None = None
    for header, name, body in macros:
        if header != current_header:
            lines.append("")
            lines.append(f"% {header}")
            current_header = header
        lines.append(_macro(name, body))
    lines.append(END)
    return "\n".join(lines)


def write_highlights() -> int:
    """Render paper/highlights.txt from paper.tex's own highlights environment.

    The journal takes the highlights as a separate plain-text file, and a
    hand-maintained one is a second copy of a block that already exists in the
    manuscript, with nothing comparing them, so a hand edit of either can
    state a claim that the other does not. The file is therefore written here
    from the block that renders."""
    paper = (PARAMS.parent / "paper.tex").read_text()

    def one(pattern: str, what: str) -> str:
        m = re.search(pattern, paper, flags=re.DOTALL)
        if not m:
            sys.exit(f"paper/paper.tex: no {what} to render highlights.txt from")
        return m.group(1).strip()

    title = one(r"\\title\[mode=title\]\{(.*?)\}\n", "title")
    authors = ", ".join(re.findall(r"\\author\[\d+\]\{(.*?)\}", paper))
    if not authors:
        sys.exit("paper/paper.tex: no author to render highlights.txt from")
    items = re.findall(r"\\item (.*)",
                       one(r"\\begin\{highlights\}(.*?)\\end\{highlights\}",
                           "highlights environment"))
    if not items:
        sys.exit("paper/paper.tex: the highlights environment is empty")

    values = {name: value for name, value in
              re.findall(r"\\newcommand\{\\(\w+)\}\{(.*?)\\xspace\}",
                         PARAMS.read_text())}

    def resolve(line: str) -> str:
        def macro(m: re.Match) -> str:
            name = m.group(1)
            if name not in values:
                sys.exit(f"highlights item cites \\{name}, which params.tex "
                         f"does not define")
            return values[name]
        return re.sub(r"\\(\w+)", macro, line).replace("\\%", "%").strip()

    body = "\n".join(f"- {resolve(i)}" for i in items)
    HIGHLIGHTS.write_text(f"Highlights\n\n{title}\n\n{authors}\n\n{body}\n")
    return len(items)


def main() -> None:
    if not PARAMS.exists():
        sys.exit(f"params file not found: {PARAMS}")
    content = PARAMS.read_text()
    macros = build_macros()
    block = render_block(macros)

    pattern = re.compile(
        re.escape(BEGIN) + r".*?" + re.escape(END), flags=re.DOTALL
    )
    if not pattern.search(content):
        sys.exit("Could not find GENERATED region markers in params.tex")

    new_content = pattern.sub(lambda _m: block, content)
    PARAMS.write_text(new_content)
    print(f"Wrote {len(macros)} macros to {PARAMS}")
    print(f"Wrote {write_highlights()} highlights to {HIGHLIGHTS}")


if __name__ == "__main__":
    main()
