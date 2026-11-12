"""Rolling percolation pipeline under point-in-time quintile assignment.

Point-in-time (PRIMARY): at each rolling window's end date, sort all common
stocks active on that date into size quintiles by point-in-time market
capitalization, then draw <=300 stocks per quintile under each seed of
DRAW_SEEDS, run a percolation snapshot on each draw's 60-day return window, and
average the snapshots over the draws. This is the rigorous
survivorship-bias-free approach.

The rolling-window x quintile bookkeeping (grid, seeded sampling, column
cleaning, correlation, percolation metrics) lives in :mod:`fn_percolation.windows`;
this module just wires the point-in-time driver and the single-window uniform-N
check to that machinery and writes the per-quintile CSVs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

# Constants live in ``constants.py`` and the rolling-window machinery in
# ``windows.py``; both are imported here for this module's own use. Callers take
# each name from the module that defines it, so there is one import path per
# name and no second one to go stale.
from .constants import (
    DRAW_SEEDS, MIN_VALID_FRACTION, PER_QUINTILE, QUINTILES, STEP, WINDOW)
from .windows import (
    QuintileAssigner,
    WindowCell,
    iter_window_cells,
    n_windows,
    over_draw_seeds,
    percolation_metrics,
    quintile_assigner,
)

# Per-draw columns that write_quintile_results keeps in
# rolling_percolation_per_seed.csv: the thresholds the tests read per draw.
PER_SEED_COLUMNS = ("rho_c", "rho_c_chi")


def progress_hook(n_windows_total: int, *, prefix: str = "  ",
                  every: int = 25) -> Callable[[WindowCell], None]:
    """``on_cell`` hook for :func:`collect_quintile_frames` that prints one
    line per *every* windows: ``{prefix}window i/N (end date)``."""
    last_reported = -1

    def _progress(cell: WindowCell) -> None:
        nonlocal last_reported
        if cell.w != last_reported and (cell.w + 1) % every == 0:
            last_reported = cell.w
            print(f"{prefix}window {cell.w + 1}/{n_windows_total} "
                  f"({cell.end_date.date()})")

    return _progress


# ---------------------------------------------------------------------------
# Per-quintile sweep collection
# ---------------------------------------------------------------------------

class QuintileFrames(dict):
    """``{g: frame}`` of the seed-mean rows of each quintile, with the rows of
    every draw in ``per_seed`` (columns ``seed``, ``quintile``, ``date`` and the
    metrics)."""

    per_seed: pd.DataFrame


def collect_quintile_frames(
    returns: pd.DataFrame,
    *,
    assigner: QuintileAssigner,
    cell_fn: Callable[[WindowCell], dict | None] | None = None,
    on_cell: Callable[[WindowCell], None] | None = None,
    seeds: Sequence[int] = DRAW_SEEDS,
    **iter_kwargs,
) -> QuintileFrames:
    """Run the rolling-window x quintile sweep under each draw seed and average
    each cell over the draws.

    For every cell from :func:`iter_window_cells` (which receives *assigner*,
    the seed_base, and any *iter_kwargs* such as ``window`` / ``step`` /
    ``per_group`` / ``skip_before``), ``cell_fn(cell)`` returns that row's
    metrics dict, or ``None`` to drop the cell. ``cell_fn`` defaults to the
    headline :func:`percolation_metrics` on the zero-diagonal correlation. A
    (window, quintile) row of the result holds the mean of each metric over the
    draws that kept the cell, and ``n_seeds``, the number of those draws
    (subsec:methods). ``on_cell`` is an optional progress hook that the worker
    of the first seed calls once per kept cell, so each driver keeps its own
    stdout cadence without changing the recorded data. Returns the seed-mean
    frame of every g in :data:`QUINTILES`, indexed by date.
    """
    if cell_fn is None:
        cell_fn = lambda c: percolation_metrics(c.corr, c.sub.shape[1])

    def job(seed_base: int) -> pd.DataFrame:
        rows: list[dict] = []
        for cell in iter_window_cells(returns, assigner=assigner,
                                      seed_base=seed_base, **iter_kwargs):
            metrics = cell_fn(cell)
            if metrics is None:
                continue
            rows.append({"quintile": cell.quintile, "date": cell.end_date,
                         **metrics})
            if on_cell is not None and seed_base == seeds[0]:
                on_cell(cell)
        return pd.DataFrame(rows)

    per_seed, mean = over_draw_seeds(job, ["quintile", "date"], seeds)
    out = QuintileFrames()
    for g in QUINTILES:
        out[g] = (mean[mean["quintile"] == g].drop(columns="quintile")
                  .set_index("date").sort_index())
    out.per_seed = per_seed
    return out


def align_quintile_panel(dfs: dict[int, pd.DataFrame], col: str) -> pd.DataFrame:
    """Stack the per-quintile ``col`` series into a wide (date x Q1..Q5) panel.

    Drops any window missing from one or more quintiles (inner join on dates) so
    the paired-difference batteries (subsec:main_result) compare matched windows only.
    """
    series = {g: dfs[g][col] for g in QUINTILES}
    panel = pd.concat(series, axis=1).dropna(how="any")
    panel.columns = [f"Q{g}" for g in QUINTILES]
    return panel


def run_pointintime(
    returns: pd.DataFrame,
    marcap: pd.DataFrame,
    out_dir: Path,
    window: int = WINDOW,
    step: int = STEP,
    per_quintile: int = PER_QUINTILE,
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> QuintileFrames:
    """Point-in-time rolling percolation under every draw seed. Writes one
    seed-mean CSV per quintile, the per-draw thresholds, and the summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_windows_total = n_windows(len(returns.index), window, step)
    print(f"[pointintime] scanning {n_windows_total} windows ...")

    dfs = collect_quintile_frames(
        returns, assigner=quintile_assigner(marcap),
        on_cell=progress_hook(n_windows_total, prefix="[pointintime] "),
        window=window, step=step, per_group=per_quintile,
        min_valid_fraction=min_valid_fraction,
    )
    return write_quintile_results(dfs, out_dir)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _summarize(g: int, df: pd.DataFrame, n_stocks: int,
               with_chi: bool = True) -> dict:
    out = {
        "size_group": g,
        "n_stocks": n_stocks,
        "n_windows": int(len(df)),
        "rho_c_mean": float(df["rho_c"].mean()) if not df.empty else float("nan"),
        "rho_c_std": float(df["rho_c"].std()) if not df.empty else float("nan"),
        "rho_c_min": float(df["rho_c"].min()) if not df.empty else float("nan"),
        "rho_c_max": float(df["rho_c"].max()) if not df.empty else float("nan"),
    }
    if with_chi and not df.empty and "rho_c_chi" in df.columns:
        chi_col = df["rho_c_chi"].dropna()
        out["rho_c_chi_mean"] = float(chi_col.mean()) if len(chi_col) else float("nan")
        out["rho_c_chi_std"] = float(chi_col.std()) if len(chi_col) else float("nan")
        out["rho_c_chi_min"] = float(chi_col.min()) if len(chi_col) else float("nan")
        out["rho_c_chi_max"] = float(chi_col.max()) if len(chi_col) else float("nan")
        out["rho_c_chi_n"] = int(len(chi_col))
    return out


def write_quintile_results(dfs: dict[int, pd.DataFrame],
                           out_dir: Path,
                           *,
                           write_q1_alias: bool = True,
                           with_chi: bool = True) -> dict[int, pd.DataFrame]:
    """Write one ``rolling_percolation_Q{g}.csv`` per quintile plus the
    cross-quintile ``rolling_percolation_summary.csv``. Each ``dfs[g]`` is the
    finalized (date-indexed) per-quintile frame. Frames from
    :func:`collect_quintile_frames` also write the thresholds of every draw to
    ``rolling_percolation_per_seed.csv``.

    ``write_q1_alias`` additionally writes Q1 to the un-suffixed
    ``rolling_percolation.csv`` read by the headline drivers' single-series
    consumers; the confound-control scripts write no such alias and pass
    ``False``. ``with_chi`` includes the susceptibility-peak ``rho_c_chi_*``
    summary columns (subsec:methods) when the frames carry an ``rho_c_chi`` column;
    the confound-control summaries report only ``rho_c`` and pass ``False``.
    """
    summary_rows = []
    for g in QUINTILES:
        df = dfs[g]
        df.to_csv(out_dir / f"rolling_percolation_Q{g}.csv")
        if write_q1_alias and g == 1:
            df.to_csv(out_dir / "rolling_percolation.csv")
        n_stocks = int(df["n_stocks"].median()) if not df.empty else 0
        summary_rows.append(_summarize(g, df, n_stocks, with_chi=with_chi))
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "rolling_percolation_summary.csv", index=False)
    per_seed = getattr(dfs, "per_seed", None)
    if per_seed is not None:
        cols = ["seed", "quintile", "date"] + [
            c for c in PER_SEED_COLUMNS if c in per_seed.columns]
        per_seed[cols].sort_values(["seed", "quintile", "date"]).to_csv(
            out_dir / "rolling_percolation_per_seed.csv", index=False,
            float_format="%.6g")
    return dfs
